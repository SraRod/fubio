"""Mosaic composition — configurable R×C grid on a shared canvas.

Companions can come from a separate pool (companion_pool parameter),
enabling unlabeled data as companion sources without changing the anchor
dataset or sampler. When companion_pool is omitted, companions come from
the anchor dataset (backward-compatible default).

Samples R*C-1 companions, letterbox-resizes each source into its grid cell,
and stacks per-instance arrays along the I axis. The downstream
TransformPipeline applies one shared M_post to the entire canvas,
composing M_final[i] = M_post @ M_tile[i] per instance.

Mosaic modes — how companion tiles relate to the anchor:

  same_task      All companions share the anchor's task. Produces
                 multiple GT instances of the same structure. Requires
                 model n_inst >= mosaic n_instances to match all GTs.
  cross_task     Companions exclude the anchor's task, but two
                 companions may share a non-anchor task. Reduces
                 same-task collisions vs mixed, does not eliminate them
                 among companions.
  distinct_task  Each tile (anchor + companions) has a unique task.
                 Guarantees at most 1 GT instance per task per mosaic.
                 Safe with n_inst=1. Requires n_tasks > n_instances
                 (9 tasks, 4 tiles — always satisfied for 2×2).
  mixed          Companions from the global pool, any task. No
                 task-level control; same-task collisions by chance.
  balanced       Uniform-task-first: pick a task uniformly, then an
                 image from that task. Counters dataset imbalance but
                 allows same-task collisions (including with anchor).
  random         Per-sample selection among concrete modes, weighted
                 by mode_weights (default: uniform over same_task,
                 cross_task, balanced).

Upstream: manifest.py (ManifestDataset provides I=1 SampleDicts).
Downstream: transforms.py (TransformedDataset wraps this for augmentation).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from fubio.data.manifest import SampleDict
from fubio.data.spatial import SpatialParams, apply_spatial, build_affine_matrix
from fubio.data.task_registry import TASKS, int_to_task_id

logger = logging.getLogger(__name__)

_CONCRETE_MODES = ("same_task", "cross_task", "distinct_task", "mixed", "balanced")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MosaicConfig(BaseModel):
    """Grid layout and task-mixing mode for mosaic composition."""

    model_config = ConfigDict(frozen=True)

    rows: int = 2
    cols: int = 2
    prob: float = 1.0
    open_epoch: int | None = None
    close_epoch: int | None = None
    mode: Literal["same_task", "cross_task", "distinct_task", "mixed", "balanced", "random"] = (
        "mixed"
    )
    # Per-sample mode selection weights (only used when mode="random").
    # Keys must be concrete modes (same_task/cross_task/mixed/balanced).
    # None = uniform across same_task/cross_task/balanced.
    mode_weights: dict[str, float] | None = None
    min_labeled: int = 1

    @model_validator(mode="after")
    def _check_range(self) -> MosaicConfig:
        if not (1 <= self.rows <= 3):
            raise ValueError(f"rows must be 1–3, got {self.rows}")
        if not (1 <= self.cols <= 3):
            raise ValueError(f"cols must be 1–3, got {self.cols}")
        n = self.rows * self.cols
        if not (0 <= self.min_labeled <= n):
            raise ValueError(f"min_labeled must be 0–{n}, got {self.min_labeled}")
        if self.mode_weights is not None:
            bad = set(self.mode_weights) - set(_CONCRETE_MODES)
            if bad:
                raise ValueError(f"mode_weights has invalid keys: {bad}")
            if any(v <= 0 for v in self.mode_weights.values()):
                raise ValueError("mode_weights values must be positive")
        return self

    @property
    def n_instances(self) -> int:
        return self.rows * self.cols


# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------


def _cell_sizes(
    canvas_w: int, canvas_h: int, rows: int, cols: int
) -> list[tuple[int, int, int, int]]:
    """Compute (x0, y0, cell_w, cell_h) for each grid cell (row-major).

    Last column/row absorbs rounding remainder so cells tile exactly.
    """
    base_w = canvas_w // cols
    base_h = canvas_h // rows
    cells: list[tuple[int, int, int, int]] = []

    for r in range(rows):
        y0 = r * base_h
        cell_h = canvas_h - y0 if r == rows - 1 else base_h
        for c in range(cols):
            x0 = c * base_w
            cell_w = canvas_w - x0 if c == cols - 1 else base_w
            cells.append((x0, y0, cell_w, cell_h))

    return cells


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MosaicDataset:
    """Wraps ManifestDataset to compose mosaic canvases.

    Anchor samples come from base_dataset (driven by the batch sampler).
    Companion samples come from companion_pool (defaults to base_dataset).
    This separation lets unlabeled data serve as companions without
    affecting anchor selection or epoch length.

    Keys in (from ManifestDataset, I=1):
        image, keypoints, transform_matrix, landmark_*_mask,
        task_ids, is_labeled, image_paths, original_hws

    Keys out (I = rows × cols, or I=1 on passthrough):
        image             — [H, W, 3] uint8, composed canvas
        keypoints         — [I, 60, 2] float32, mapped to canvas coords
        transform_matrix  — [I, 3, 3] float32, M_tile per instance
        landmark_visible_mask — [I, 60] bool, recomputed per cell resize
        (all other per-instance arrays stacked along I)
    """

    def __init__(
        self,
        base_dataset: object,
        config: MosaicConfig,
        seed: int = 42,
        companion_pool: object | None = None,
    ) -> None:
        self._base = base_dataset
        self._companion = companion_pool if companion_pool is not None else base_dataset
        self._cfg = config
        self._seed = seed
        self._epoch = 0
        self._rng = np.random.default_rng(seed)

        # Pre-compute random mode weights as arrays for fast sampling
        if config.mode == "random":
            weights = config.mode_weights
            if weights is None:
                self._rand_modes = np.array(["same_task", "cross_task", "balanced"])
                self._rand_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            else:
                self._rand_modes = np.array(list(weights.keys()))
                p = np.array([weights[m] for m in self._rand_modes])
                self._rand_probs = p / p.sum()
        else:
            self._rand_modes = np.array([])
            self._rand_probs = np.array([])

        # Companion pools — built from companion source, not anchor dataset
        comp = self._companion
        self._task_pools: dict[str, list[int]] = {}
        for task_id in TASKS:
            indices = comp.get_task_indices(task_id)
            if indices:
                self._task_pools[task_id] = list(indices)

        self._global_pool: list[int] = list(range(len(comp)))

        self._cross_pools: dict[str, list[int]] = {}
        for task_id in self._task_pools:
            exclude = set(self._task_pools[task_id])
            self._cross_pools[task_id] = [i for i in self._global_pool if i not in exclude]

        comp_labeled = set(comp.get_labeled_indices())
        self._labeled_global: list[int] = [i for i in self._global_pool if i in comp_labeled]
        self._labeled_task_pools: dict[str, list[int]] = {
            tid: [i for i in indices if i in comp_labeled]
            for tid, indices in self._task_pools.items()
        }
        self._labeled_cross_pools: dict[str, list[int]] = {
            tid: [i for i in indices if i in comp_labeled]
            for tid, indices in self._cross_pools.items()
        }
        self._comp_labeled_set = comp_labeled

        # Anchor labeled check uses base_dataset (anchor might not be in companion_pool)
        self._anchor_labeled_set = set(base_dataset.get_labeled_indices())

        logger.info(
            "MosaicDataset: %s grid (I=%d), prob=%.2f, mode=%s, min_labeled=%d, "
            "%d tasks, companions=%d (%d labeled), anchors=%d",
            f"{config.rows}×{config.cols}",
            config.n_instances,
            config.prob,
            config.mode,
            config.min_labeled,
            len(self._task_pools),
            len(self._global_pool),
            len(comp_labeled),
            len(base_dataset),
        )

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for close_epoch and re-seed RNG."""
        self._epoch = epoch
        self._rng = np.random.default_rng(self._seed + epoch)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> SampleDict:
        anchor = self._base[idx]
        cfg = self._cfg

        # Passthrough conditions
        if cfg.n_instances == 1:
            return anchor
        if cfg.open_epoch is not None and self._epoch < cfg.open_epoch:
            return anchor
        if cfg.close_epoch is not None and self._epoch > cfg.close_epoch:
            return anchor
        if self._rng.random() > cfg.prob:
            return anchor

        n_companions = cfg.n_instances - 1

        # Resolve effective mode (random → concrete per sample)
        effective_mode = self._resolve_mode()

        # Sample companions
        companion_indices = self._sample_companions(anchor, effective_mode, n_companions)
        if companion_indices is None:
            return anchor

        # Enforce min_labeled
        if cfg.min_labeled > 0:
            self._enforce_min_labeled(
                idx,
                companion_indices,
                effective_mode,
                anchor,
            )

        sources = [anchor] + [self._companion[int(ci)] for ci in companion_indices]
        return self._compose(sources, cfg.rows, cfg.cols)

    def _resolve_mode(self) -> str:
        """Pick a concrete mode. Random mode samples per call."""
        if self._cfg.mode != "random":
            return self._cfg.mode
        return str(self._rng.choice(self._rand_modes, p=self._rand_probs))

    def _sample_companions(
        self,
        anchor: SampleDict,
        mode: str,
        n: int,
    ) -> list[int] | None:
        """Sample n companion indices according to the given mode."""
        if mode == "distinct_task":
            return self._sample_distinct_task(anchor, n, labeled_only=False)
        if mode == "balanced":
            indices = self._sample_balanced(n, labeled_only=False)
            return indices if indices else None
        pool = self._pick_pool(anchor, mode)
        if pool is None or len(pool) < 1:
            return None
        return list(self._rng.choice(pool, size=n, replace=True))

    def _enforce_min_labeled(
        self,
        anchor_idx: int,
        companion_indices: list[int],
        mode: str,
        anchor: SampleDict,
    ) -> None:
        """Replace unlabeled companions until min_labeled is satisfied (in-place)."""
        anchor_labeled = 1 if anchor_idx in self._anchor_labeled_set else 0
        n_labeled = anchor_labeled + sum(
            1 for ci in companion_indices if ci in self._comp_labeled_set
        )
        need = self._cfg.min_labeled - n_labeled
        if need <= 0:
            return

        unlabeled_positions = [
            i for i, ci in enumerate(companion_indices) if ci not in self._comp_labeled_set
        ]
        if mode == "balanced":
            replacements = self._sample_balanced(
                min(need, len(unlabeled_positions)),
                labeled_only=True,
            )
        else:
            labeled_pool = self._pick_labeled_pool(anchor, mode)
            lp_len = len(labeled_pool) if labeled_pool else 0
            n_avail = min(need, len(unlabeled_positions), lp_len)
            replacements = (
                [int(x) for x in self._rng.choice(labeled_pool, size=n_avail, replace=True)]
                if labeled_pool and n_avail > 0
                else []
            )

        for pos, repl in zip(
            unlabeled_positions[: len(replacements)],
            replacements,
            strict=True,
        ):
            companion_indices[pos] = repl

        if len(replacements) < need:
            logger.warning(
                "min_labeled=%d but only %d labeled available (mode=%s)",
                self._cfg.min_labeled,
                n_labeled + len(replacements),
                mode,
            )

    def _pick_pool(
        self,
        anchor: SampleDict,
        mode: str,
    ) -> list[int] | None:
        """Select companion sampling pool based on mosaic mode."""
        if mode == "same_task":
            task_id = int_to_task_id(int(anchor["task_ids"][0]))
            return self._task_pools.get(task_id)
        if mode in ("cross_task", "distinct_task"):
            task_id = int_to_task_id(int(anchor["task_ids"][0]))
            return self._cross_pools.get(task_id) or None
        # mixed: any sample from any task
        return self._global_pool

    def _pick_labeled_pool(
        self,
        anchor: SampleDict,
        mode: str,
    ) -> list[int] | None:
        """Select labeled-only companion pool respecting mosaic mode."""
        if mode == "same_task":
            task_id = int_to_task_id(int(anchor["task_ids"][0]))
            return self._labeled_task_pools.get(task_id) or None
        if mode in ("cross_task", "distinct_task"):
            task_id = int_to_task_id(int(anchor["task_ids"][0]))
            return self._labeled_cross_pools.get(task_id) or None
        return self._labeled_global or None

    def _sample_distinct_task(
        self,
        anchor: SampleDict,
        n: int,
        *,
        labeled_only: bool = False,
    ) -> list[int] | None:
        """Sample n companions, each from a different task (and different from anchor).

        Guarantees at most 1 GT instance per task in the final mosaic.
        Requires len(available_tasks) >= n; falls back to passthrough if not.
        """
        anchor_task = int_to_task_id(int(anchor["task_ids"][0]))
        pools = self._labeled_task_pools if labeled_only else self._task_pools
        available = [tid for tid in pools if tid != anchor_task and pools[tid]]
        if len(available) < n:
            return None
        selected_tasks = list(self._rng.choice(available, size=n, replace=False))
        return [int(self._rng.choice(pools[tid])) for tid in selected_tasks]

    def _sample_balanced(
        self,
        n: int,
        *,
        labeled_only: bool = False,
    ) -> list[int]:
        """Sample n indices with uniform-task-first to counter dataset imbalance."""
        pools = self._labeled_task_pools if labeled_only else self._task_pools
        task_ids = [tid for tid, p in pools.items() if p]
        if not task_ids:
            return []
        indices: list[int] = []
        for _ in range(n):
            task = str(self._rng.choice(task_ids))
            idx = int(self._rng.choice(pools[task]))
            indices.append(idx)
        return indices

    def _compose(
        self,
        sources: list[SampleDict],
        rows: int,
        cols: int,
    ) -> SampleDict:
        """Compose R×C sources into a single mosaic canvas.

        Mosaic is a transform scope boundary. Incoming transform_matrix
        (e.g. from per-instance DA) is intentionally not chained — output
        M_tile describes each source image → canvas mapping only.
        """
        anchor_img = sources[0]["image"]
        canvas_h, canvas_w = anchor_img.shape[:2]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cells = _cell_sizes(canvas_w, canvas_h, rows, cols)

        all_kp = []
        all_M = []
        all_valid = []
        all_supervised = []
        all_visible = []
        all_evidence = []
        all_task_ids = []
        all_labeled = []
        all_paths: list[str] = []
        all_hws = []

        for src, (x0, y0, cell_w, cell_h) in zip(sources, cells, strict=True):
            src_img = src["image"]
            src_h, src_w = src_img.shape[:2]
            src_kp = src["keypoints"]  # [1, 60, 2]

            # Letterbox-resize source → cell
            resize_params = SpatialParams(target_size=(cell_w, cell_h))
            M_resize = build_affine_matrix(
                src_size=(src_w, src_h),
                params=resize_params,
                resize_mode="letterbox",
            )

            cell_img, cell_kp, cell_vis = apply_spatial(src_img, src_kp, M_resize, (cell_w, cell_h))

            # Place cell into canvas
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell_img

            # Offset keypoints to canvas coordinates
            kp_canvas = cell_kp.copy()  # [1, 60, 2]
            finite = ~np.isnan(kp_canvas).any(axis=-1)  # [1, 60]
            kp_canvas[finite, 0] += x0
            kp_canvas[finite, 1] += y0

            # Compose tile affine: M_offset @ M_resize @ M_src (original → canvas).
            # M_src is the source's incoming transform (e.g. from cache letterbox).
            M_offset = np.eye(3, dtype=np.float64)
            M_offset[0, 2] = x0
            M_offset[1, 2] = y0
            M_src = src["transform_matrix"][0].astype(np.float64)
            M_tile = (M_offset @ M_resize @ M_src).astype(np.float32)

            # Update visible: cell_vis from apply_spatial is cell-local.
            # After offset, check canvas-level bounds.
            vis_canvas = np.zeros_like(cell_vis)  # [1, 60]
            for inst in range(cell_kp.shape[0]):
                for k in range(cell_kp.shape[1]):
                    if not finite[inst, k]:
                        continue
                    cx = kp_canvas[inst, k, 0]
                    cy = kp_canvas[inst, k, 1]
                    vis_canvas[inst, k] = 0 <= cx <= canvas_w - 1 and 0 <= cy <= canvas_h - 1

            all_kp.append(kp_canvas[0])  # [60, 2]
            all_M.append(M_tile[np.newaxis])  # [1, 3, 3]
            all_valid.append(src["landmark_valid_mask"][0])  # [60]
            all_supervised.append(src["landmark_supervised_mask"][0])  # [60]
            all_visible.append(vis_canvas[0])  # [60]
            all_evidence.append(src["landmark_evidence_mask"][0])  # [60]
            all_task_ids.append(src["task_ids"][0])  # scalar
            all_labeled.append(src["is_labeled"][0])  # scalar
            all_paths.append(src["image_paths"][0])  # str
            all_hws.append(src["original_hws"][0])  # [2]

        return {
            "image": canvas,
            "keypoints": np.stack(all_kp),  # [I, 60, 2]
            "transform_matrix": np.concatenate(all_M, axis=0),  # [I, 3, 3]
            "landmark_valid_mask": np.stack(all_valid),  # [I, 60]
            "landmark_supervised_mask": np.stack(all_supervised),  # [I, 60]
            "landmark_visible_mask": np.stack(all_visible),  # [I, 60]
            "landmark_evidence_mask": np.stack(all_evidence),  # [I, 60]
            "task_ids": np.array(all_task_ids, dtype=np.int64),  # [I]
            "is_labeled": np.array(all_labeled, dtype=bool),  # [I]
            "image_paths": all_paths,
            "original_hws": np.stack(all_hws),  # [I, 2]
        }

    def get_task_indices(self, task_id: str) -> Sequence[int]:
        return self._base.get_task_indices(task_id)

    def get_split_indices(self, split: str) -> Sequence[int]:
        return self._base.get_split_indices(split)

    def get_labeled_indices(self) -> Sequence[int]:
        return self._base.get_labeled_indices()

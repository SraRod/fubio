"""Collate SampleDicts into tensor batches.

Two collate functions:
  - collate_fubio: padded tensors (legacy, for non-model consumers)
  - collate_for_model: BatchDict (list-of-InstanceDict) for FUBioModule

Upstream: transforms.py (TransformedDataset produces augmented SampleDicts).
Downstream: training loop / evaluation (consumes tensor batch dicts).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fubio.data.manifest import SampleDict
from fubio.data.supportive import compute_supportive
from fubio.data.task_registry import TASKS, int_to_task_id
from fubio.data.types import BatchDict, InstanceDict


def collate_fubio(batch: list[SampleDict]) -> dict[str, Any]:
    """Collate a list of SampleDicts into a padded tensor batch.

    Keys in (per sample):
        image             — [H, W, 3] float32, normalized HWC
        keypoints         — [I, 60, 2] float32, may contain NaN
        transform_matrix  — [I, 3, 3] float32
        landmark_valid_mask      — [I, 60] bool
        landmark_supervised_mask — [I, 60] bool
        landmark_visible_mask    — [I, 60] bool
        task_ids          — [I] int64
        is_labeled        — [I] bool
        image_paths       — list[str] length I
        original_hws      — [I, 2] int32

    Keys out (tensor batch):
        image             — [B, 3, H, W] float32
        keypoints         — [B, I_max, 60, 2] float32, NaN→0.0
        transform_matrix  — [B, I_max, 3, 3] float32, padding=zeros
        landmark_valid_mask      — [B, I_max, 60] bool, padding=False
        landmark_supervised_mask — [B, I_max, 60] bool, padding=False
        landmark_visible_mask    — [B, I_max, 60] bool, padding=False
        instance_mask     — [B, I_max] bool, True=real, False=padding
        task_ids          — [B, I_max] int64, padding=-1
        is_labeled        — [B, I_max] bool, padding=False
        original_hws      — [B, I_max, 2] int32, padding=0
        image_paths       — list[list[str]], padding=""
    """
    b = len(batch)
    i_max = max(s["keypoints"].shape[0] for s in batch)

    # --- Images: HWC → CHW, stack ---
    images = np.stack([np.ascontiguousarray(s["image"].transpose(2, 0, 1)) for s in batch])

    # --- Per-instance arrays: pad to I_max ---
    keypoints = np.zeros((b, i_max, 60, 2), dtype=np.float32)
    transform_matrix = np.zeros((b, i_max, 3, 3), dtype=np.float32)
    valid_mask = np.zeros((b, i_max, 60), dtype=bool)
    supervised_mask = np.zeros((b, i_max, 60), dtype=bool)
    visible_mask = np.zeros((b, i_max, 60), dtype=bool)
    instance_mask = np.zeros((b, i_max), dtype=bool)
    task_ids = np.full((b, i_max), -1, dtype=np.int64)
    is_labeled = np.zeros((b, i_max), dtype=bool)
    original_hws = np.zeros((b, i_max, 2), dtype=np.int32)
    image_paths: list[list[str]] = []

    for idx, s in enumerate(batch):
        n_inst = s["keypoints"].shape[0]

        kp = s["keypoints"].copy()
        kp = np.where(np.isnan(kp), 0.0, kp)
        keypoints[idx, :n_inst] = kp

        transform_matrix[idx, :n_inst] = s["transform_matrix"]
        valid_mask[idx, :n_inst] = s["landmark_valid_mask"]
        supervised_mask[idx, :n_inst] = s["landmark_supervised_mask"]
        visible_mask[idx, :n_inst] = s["landmark_visible_mask"]
        instance_mask[idx, :n_inst] = True
        task_ids[idx, :n_inst] = s["task_ids"]
        is_labeled[idx, :n_inst] = s["is_labeled"]
        original_hws[idx, :n_inst] = s["original_hws"]

        paths = list(s["image_paths"])
        paths.extend([""] * (i_max - n_inst))
        image_paths.append(paths)

    return {
        "image": torch.from_numpy(images),
        "keypoints": torch.from_numpy(keypoints),
        "transform_matrix": torch.from_numpy(transform_matrix),
        "landmark_valid_mask": torch.from_numpy(valid_mask),
        "landmark_supervised_mask": torch.from_numpy(supervised_mask),
        "landmark_visible_mask": torch.from_numpy(visible_mask),
        "instance_mask": torch.from_numpy(instance_mask),
        "task_ids": torch.from_numpy(task_ids),
        "is_labeled": torch.from_numpy(is_labeled),
        "original_hws": torch.from_numpy(original_hws),
        "image_paths": image_paths,
    }


# ---------------------------------------------------------------------------
# Model collate: SampleDict list → BatchDict (list-of-InstanceDict)
# ---------------------------------------------------------------------------


def _bbox_from_keypoints(
    kp: np.ndarray,
    mask: np.ndarray,
    target_w: float,
    target_h: float,
    context_scale: float = 1.0,
) -> np.ndarray:
    """Compute context-aware cxcywh bbox from visible keypoints, normalized [0,1].

    Landmarks are internal to the anatomical structure — the model needs
    surrounding context to localize them. The bbox side is expanded by
    context_scale (from task_registry.bbox_context_scale), then padded by
    5% of canvas. context_scale=1.0 keeps tight bbox; >1.0 proportionally
    enlarges the region around the landmark span.

    Args:
        kp: (K, 2) keypoints in pixel space.
        mask: (K,) bool — which keypoints to use.
        target_w, target_h: canvas dimensions for normalization.
        context_scale: multiplicative expansion on landmark span (per-task).

    Returns:
        (4,) float32 — cx, cy, w, h in [0,1].
    """
    if not mask.any():
        return np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)

    pts = kp[mask]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Expand landmark span by context_scale around its center
    cx_px = (x_min + x_max) / 2.0
    cy_px = (y_min + y_max) / 2.0
    half_w = (x_max - x_min) / 2.0 * context_scale
    half_h = (y_max - y_min) / 2.0 * context_scale
    x_min = cx_px - half_w
    y_min = cy_px - half_h
    x_max = cx_px + half_w
    y_max = cy_px + half_h

    # Standard 5% canvas padding on top of scaled span
    pad_x = target_w * 0.05
    pad_y = target_h * 0.05
    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(target_w, x_max + pad_x)
    y_max = min(target_h, y_max + pad_y)

    cx = (x_min + x_max) / 2.0 / target_w
    cy = (y_min + y_max) / 2.0 / target_h
    w = (x_max - x_min) / target_w
    h = (y_max - y_min) / target_h

    return np.array([cx, cy, w, h], dtype=np.float32)


def collate_for_model(batch: list[SampleDict]) -> BatchDict:
    """Collate SampleDicts into BatchDict for FUBioModule.

    Converts canonical K=60 scored keypoints to per-task arrays, computes
    supportive landmarks from scored geometry, and packs into InstanceDicts
    with shape (K_total, ...) where K_total = K_scored + K_supportive.

    Keys in (per sample from TransformPipeline):
        image             — [H, W, 3] float32, normalized HWC
        keypoints         — [I, 60, 2] float32, pixel space, may contain NaN
        transform_matrix  — [I, 3, 3] float32
        landmark_supervised_mask — [I, 60] bool
        landmark_visible_mask    — [I, 60] bool
        landmark_evidence_mask   — [I, 60] int64 (optional, defaults to zeros)
        task_ids          — [I] int64
        is_labeled        — [I] bool
        was_hflipped      — bool (optional, defaults to False)
        was_vflipped      — bool (optional, defaults to False)
        image_paths       — list[str] length I
        original_hws      — [I, 2] int32

    Keys out (BatchDict):
        image   — [B, 3, H, W] float32
        targets — list[list[InstanceDict]], one per image
    """
    images = np.stack([np.ascontiguousarray(s["image"].transpose(2, 0, 1)) for s in batch])

    all_targets: list[list[InstanceDict]] = []

    for s in batch:
        kp_all: np.ndarray = s["keypoints"]  # [I, 60, 2]
        n_inst = kp_all.shape[0]
        target_h, target_w = s["image"].shape[:2]
        was_hflipped: bool = s.get("was_hflipped", False)
        was_vflipped: bool = s.get("was_vflipped", False)

        instances: list[InstanceDict] = []
        for i in range(n_inst):
            task_int = int(s["task_ids"][i])
            task_str = int_to_task_id(task_int)
            t = TASKS[task_str]
            off = t.offset
            k = t.n_keypoints

            # Extract scored task-local from canonical K=60
            kp_local = kp_all[i, off : off + k].copy()  # (K_scored, 2)
            sup = s["landmark_supervised_mask"][i, off : off + k].copy()
            vis = s["landmark_visible_mask"][i, off : off + k].copy()
            evi_scored = s.get(
                "landmark_evidence_mask",
                np.zeros((n_inst, 60), dtype=np.int64),
            )[i, off : off + k].copy()

            # --- Supportive landmarks (derived from scored geometry) ---
            sup_coords, source_indices = compute_supportive(task_str, kp_local)
            n_sup = len(sup_coords)

            if n_sup > 0:
                kp_local = np.vstack([kp_local, sup_coords])

                # Dependency-aware supervision: supervised only if ALL source
                # scored landmarks are supervised AND have finite coordinates
                sup_supervised = np.array(
                    [
                        all(sup[si] and np.isfinite(kp_local[si]).all() for si in deps)
                        for deps in source_indices
                    ],
                    dtype=bool,
                )
                sup_visible = sup_supervised & np.isfinite(sup_coords).all(axis=1)
                sup = np.concatenate([sup, sup_supervised])
                vis = np.concatenate([vis, sup_visible])

                # Supportive evidence: AND of source landmarks' evidence
                evi_sup = np.array(
                    [
                        int(all(evi_scored[si] for si in deps)) if sup_supervised[j] else 0
                        for j, deps in enumerate(source_indices)
                    ],
                    dtype=np.int64,
                )
                evi = np.concatenate([evi_scored, evi_sup])

                # Supportive flip pairs (scored already swapped in transforms.py)
                for flipped, pairs in [
                    (was_hflipped, t.supportive_hflip_pairs),
                    (was_vflipped, t.supportive_vflip_pairs),
                ]:
                    if flipped:
                        for a, b in pairs:
                            sa, sb = k + a, k + b
                            kp_local[sa], kp_local[sb] = kp_local[sb].copy(), kp_local[sa].copy()
                            sup[sa], sup[sb] = sup[sb], sup[sa]
                            vis[sa], vis[sb] = vis[sb], vis[sa]
                            evi[sa], evi[sb] = evi[sb], evi[sa]
            else:
                evi = evi_scored

            # Compute bbox from scored keypoints only (supportive extensions like
            # IVC perpendiculars would expand the box beyond the anatomical ROI)
            scored_finite = ~np.isnan(kp_local[:k]).any(axis=1)
            bbox = _bbox_from_keypoints(
                kp_local[:k],
                vis[:k] & scored_finite,
                target_w,
                target_h,
                context_scale=t.bbox_context_scale,
            )

            # Normalize keypoints to [0,1]
            kp_norm = kp_local.copy()
            kp_norm[:, 0] /= target_w
            kp_norm[:, 1] /= target_h
            kp_norm = np.where(np.isnan(kp_norm), 0.0, kp_norm).astype(np.float32)

            instances.append(
                InstanceDict(
                    task_id=task_int,
                    keypoints=kp_norm,
                    supervised_mask=sup,
                    visible_mask=vis,
                    evidence_mask=evi,
                    bbox=bbox,
                    transform_matrix=s["transform_matrix"][i],
                    original_hw=s["original_hws"][i],
                    is_labeled=bool(s["is_labeled"][i]),
                    image_path=s["image_paths"][i],
                )
            )

        all_targets.append(instances)

    return BatchDict(
        image=torch.from_numpy(images),
        targets=all_targets,
    )

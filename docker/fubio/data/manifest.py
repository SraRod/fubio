"""Manifest → decoded SampleDict. No augmentation, no tensor conversion.

Two implementations:
  - ManifestDataset: reads from disk (cv2.imread per __getitem__).
  - CachedManifestDataset: reads from pre-built numpy memmap cache
    (build_cache.py). Same interface, ~80× faster reads.

Upstream: build_manifest.py (manifest.parquet), build_cache.py (memmap cache).
Downstream: transforms.py (TransformedDataset wraps this for augmentation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from fubio.data.supportive import compute_evidence
from fubio.data.task_registry import TASKS, K, landmark_valid_mask, local_to_canonical

logger = logging.getLogger(__name__)

SampleDict = dict[str, Any]


class ManifestDataset:
    """Parquet-backed dataset producing raw SampleDicts (I=1).

    Keys out:
        image             — [H, W, 3] uint8, BGR→RGB
        keypoints         — [1, 60, 2] float32, NaN for unowned/unannotated
        transform_matrix  — [1, 3, 3] float32, identity (no transform yet)
        landmark_valid_mask      — [1, 60] bool, task-owned slots
        landmark_supervised_mask — [1, 60] bool, valid AND finite coords
        landmark_visible_mask    — [1, 60] bool, init = supervised_mask
        task_ids          — [1] int64
        is_labeled        — [1] bool
        image_paths       — list[str] length 1
        original_hws      — [1, 2] int32, (H, W) of loaded image
    """

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split_filter: str | list[str] | None = None,
        task_filter: str | list[str] | None = None,
    ) -> None:
        df = pl.read_parquet(manifest_path)

        if split_filter is not None:
            if isinstance(split_filter, str):
                split_filter = [split_filter]
            df = df.filter(pl.col("split").is_in(split_filter))

        if task_filter is not None:
            if isinstance(task_filter, str):
                task_filter = [task_filter]
            df = df.filter(pl.col("task_id").is_in(task_filter))

        self._df = df
        self._data_root = data_root
        self._n = len(df)

        # Pre-compute valid masks per task (immutable, reuse across samples)
        self._valid_masks: dict[str, np.ndarray] = {
            task_id: landmark_valid_mask(task_id) for task_id in TASKS
        }

        logger.info(
            "ManifestDataset: %d samples (splits=%s, tasks=%s)",
            self._n,
            df["split"].unique().sort().to_list(),
            df["task_id"].unique().sort().to_list(),
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> SampleDict:
        row = self._df.row(idx, named=True)
        task_id: str = row["task_id"]
        task_def = TASKS[task_id]
        image_path_rel: str = row["image_path"]

        # --- Load image ---
        image_path = self._data_root / image_path_rel
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # --- Keypoints: JSON → local [N, 2] → canonical [60, 2] ---
        kp_json: str | None = row["keypoints"]
        is_labeled = kp_json is not None
        if is_labeled:
            local_kp = np.array(json.loads(kp_json), dtype=np.float32)
            canonical_kp = local_to_canonical(local_kp, task_id)
        else:
            canonical_kp = np.full((K, 2), np.nan, dtype=np.float32)

        # I=1 axis
        keypoints = canonical_kp[np.newaxis]  # [1, 60, 2]

        # --- Masks ---
        valid = self._valid_masks[task_id].copy()  # [60] bool
        finite = ~np.isnan(keypoints[0]).any(axis=1)  # [60] bool
        supervised = valid & finite
        visible = supervised.copy()  # init = supervised, updated by transforms

        # --- Evidence (from raw uint8 image, before normalization) ---
        evidence = np.zeros(K, dtype=np.int64)
        if is_labeled:
            evidence = compute_evidence(keypoints[0], image)

        return {
            "image": image,
            "keypoints": keypoints,
            "transform_matrix": np.eye(3, dtype=np.float32)[np.newaxis],  # [1, 3, 3]
            "landmark_valid_mask": valid[np.newaxis],  # [1, 60]
            "landmark_supervised_mask": supervised[np.newaxis],  # [1, 60]
            "landmark_visible_mask": visible[np.newaxis],  # [1, 60]
            "landmark_evidence_mask": evidence[np.newaxis],  # [1, 60]
            "task_ids": np.array([task_def.task_int], dtype=np.int64),  # [1]
            "is_labeled": np.array([is_labeled], dtype=bool),  # [1]
            "image_paths": [image_path_rel],
            "original_hws": np.array([[h, w]], dtype=np.int32),  # [1, 2]
        }

    def get_task_indices(self, task_id: str) -> list[int]:
        """Return dataset indices for a specific task."""
        return self._df.with_row_index("idx").filter(pl.col("task_id") == task_id)["idx"].to_list()

    def get_split_indices(self, split: str) -> list[int]:
        """Return dataset indices for a specific split value."""
        return self._df.with_row_index("idx").filter(pl.col("split") == split)["idx"].to_list()

    def get_labeled_indices(self) -> list[int]:
        """Return dataset indices where keypoints are present (labeled samples)."""
        return (
            self._df.with_row_index("idx")
            .filter(pl.col("keypoints").is_not_null())["idx"]
            .to_list()
        )


# ---------------------------------------------------------------------------
# Cached variant — reads from pre-built memmap (build_cache.py)
# ---------------------------------------------------------------------------


class CachedManifestDataset:
    """Drop-in replacement for ManifestDataset using pre-built memmap cache.

    Images are pre-decoded and pre-resized to target_size. Reads are ~0.05 ms
    vs ~4.2 ms for ManifestDataset. Same SampleDict interface.

    Keys out: same as ManifestDataset, except:
        image             — [target_H, target_W, 3] uint8 (pre-resized)
        keypoints         — [1, 60, 2] float32, in target_size pixel coords
        transform_matrix  — [1, 3, 3] float32, letterbox affine (not identity)

    Upstream: build_cache.py (memmap files).
    Downstream: transforms.py, mosaic.py (same as ManifestDataset).
    """

    def __init__(
        self,
        cache_dir: Path,
        manifest_path: Path,
        data_root: Path,
        split_filter: str | list[str] | None = None,
        task_filter: str | list[str] | None = None,
    ) -> None:
        # Load manifest for filtering (source of truth for split/task)
        df_full = pl.read_parquet(manifest_path)

        # Validate cache
        info_path = cache_dir / "cache_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"No cache at {cache_dir}")
        with open(info_path) as f:
            info = json.load(f)

        from fubio.data.build_cache import _manifest_hash

        expected_hash = _manifest_hash(df_full)
        if info["manifest_hash"] != expected_hash:
            raise ValueError(
                f"Cache manifest hash mismatch: cache={info['manifest_hash']}, "
                f"current={expected_hash}. Rebuild with: "
                f"uv run python -m fubio.data.build_cache --target-size {info['target_size'][0]}"
            )

        n_total = info["n_samples"]
        target_h, target_w = info["target_size"][1], info["target_size"][0]

        # Apply filters to get filtered row indices into the full manifest
        df = df_full
        if split_filter is not None:
            if isinstance(split_filter, str):
                split_filter = [split_filter]
            df = df.filter(pl.col("split").is_in(split_filter))
        if task_filter is not None:
            if isinstance(task_filter, str):
                task_filter = [task_filter]
            df = df.filter(pl.col("task_id").is_in(task_filter))

        self._df = df
        self._n = len(df)

        # Map filtered indices → cache row indices (position in full manifest)
        df_with_idx = df_full.with_row_index("_cache_idx")
        if split_filter is not None:
            df_with_idx = df_with_idx.filter(pl.col("split").is_in(split_filter))
        if task_filter is not None:
            df_with_idx = df_with_idx.filter(pl.col("task_id").is_in(task_filter))
        self._index_map = np.array(df_with_idx["_cache_idx"].to_list(), dtype=np.int64)

        # Open memmap (read-only, shared across workers via fork COW)
        self._images = np.memmap(
            cache_dir / "images.dat",
            dtype=np.uint8,
            mode="r",
            shape=(n_total, target_h, target_w, 3),
        )

        # Load small arrays into memory (shared via fork COW)
        self._keypoints = np.load(cache_dir / "keypoints.npy")
        self._masks = np.load(cache_dir / "masks.npy")
        self._transforms = np.load(cache_dir / "transforms.npy")
        self._meta = np.load(cache_dir / "meta.npy", allow_pickle=False)

        with open(cache_dir / "image_paths.json") as f:
            self._image_paths: list[str] = json.load(f)

        logger.info(
            "CachedManifestDataset: %d samples from %s (cache %dx%d, %d total cached)",
            self._n,
            cache_dir,
            target_w,
            target_h,
            n_total,
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> SampleDict:
        ci = int(self._index_map[idx])

        image = self._images[ci].copy()
        kp = self._keypoints[ci].copy()
        masks = self._masks[ci]
        M = self._transforms[ci].copy()
        mt = self._meta[ci]
        h, w = image.shape[:2]

        # Evidence from cached uint8 image (letterbox-resized, still uint8)
        evidence = np.zeros(K, dtype=np.int64)
        if bool(mt["is_labeled"]):
            evidence = compute_evidence(kp, image)

        return {
            "image": image,
            "keypoints": kp[np.newaxis],  # [1, 60, 2]
            "transform_matrix": M[np.newaxis],  # [1, 3, 3]
            "landmark_valid_mask": masks[0][np.newaxis].copy(),  # [1, 60]
            "landmark_supervised_mask": masks[1][np.newaxis].copy(),  # [1, 60]
            "landmark_visible_mask": masks[2][np.newaxis].copy(),  # [1, 60]
            "landmark_evidence_mask": evidence[np.newaxis],  # [1, 60]
            "task_ids": np.array([int(mt["task_int"])], dtype=np.int64),
            "is_labeled": np.array([bool(mt["is_labeled"])], dtype=bool),
            "image_paths": [self._image_paths[ci]],
            "original_hws": np.array(
                [[int(mt["orig_h"]), int(mt["orig_w"])]],
                dtype=np.int32,
            ),
        }

    def get_task_indices(self, task_id: str) -> list[int]:
        """Return dataset indices for a specific task."""
        return (
            self._df.with_row_index("idx")
            .filter(
                pl.col("task_id") == task_id,
            )["idx"]
            .to_list()
        )

    def get_split_indices(self, split: str) -> list[int]:
        """Return dataset indices for a specific split value."""
        return (
            self._df.with_row_index("idx")
            .filter(
                pl.col("split") == split,
            )["idx"]
            .to_list()
        )

    def get_labeled_indices(self) -> list[int]:
        """Return dataset indices where keypoints are present (labeled samples)."""
        return (
            self._df.with_row_index("idx")
            .filter(pl.col("keypoints").is_not_null())["idx"]
            .to_list()
        )

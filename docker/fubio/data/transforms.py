"""Spatial + photometric augmentation pipeline, keeping images in HWC.

Wraps ManifestDataset (or MosaicDataset) output through a deterministic
pipeline: spatial affine → photometric intensity → normalize. The shared
affine M is composed with each instance's existing transform_matrix to
preserve per-instance provenance for inverse mapping.

Upstream: manifest.py (ManifestDataset) or mosaic.py (MosaicDataset).
Downstream: collate.py (collate_fubio converts HWC→CHW and pads to batch).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from fubio.data.manifest import SampleDict
from fubio.data.spatial import (
    SpatialParams,
    apply_spatial,
    build_affine_matrix,
)
from fubio.data.task_registry import TASKS, int_to_task_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TransformConfig(BaseModel):
    """Augmentation parameters — all ranges default to no-op.

    For per-instance DA before mosaic, use skip_resize=True (DA at native
    resolution, no fixed target) and skip_normalize=True (output stays uint8
    for downstream mosaic composition).
    """

    model_config = ConfigDict(frozen=True)

    target_size: tuple[int, int] = (518, 518)  # (W, H), DINOv2 patch-14 → 37×37
    resize_mode: str = "letterbox"

    # Spatial
    hflip_prob: float = 0.0
    vflip_prob: float = 0.0
    rotation_range: float = 0.0  # degrees, sampled uniform [-r, +r]
    scale_range: tuple[float, float] = (1.0, 1.0)  # (min, max)
    translate_range: float = 0.0  # pixels, sampled uniform [-t, +t]

    # Per-task scale override: task_id → (lo, hi). Tasks not listed use scale_range.
    task_scale_override: dict[str, tuple[float, float]] = {}
    # Tasks where BOTH flips are disabled regardless of hflip_prob/vflip_prob.
    no_flip_tasks: list[str] = []

    # Photometric (albumentations intensity-only)
    brightness_limit: float = 0.0
    contrast_limit: float = 0.0

    # Normalize
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Per-instance DA mode
    skip_resize: bool = False
    skip_normalize: bool = False


def val_transform_config(
    target_size: tuple[int, int] = (518, 518),
    resize_mode: str = "letterbox",
) -> TransformConfig:
    """Config for val/test: resize + normalize only, zero augmentation."""
    return TransformConfig(target_size=target_size, resize_mode=resize_mode)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _apply_flip_permutation(
    kp: np.ndarray,
    valid: np.ndarray,
    sup: np.ndarray,
    vis: np.ndarray,
    evidence: np.ndarray,
    task_ids: np.ndarray,
    *,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Atomically swap scored landmark indices AND all masks per task's flip pairs.

    axis: "h" for hflip_pairs, "v" for vflip_pairs.
    """
    kp, valid, sup, vis, evidence = (
        kp.copy(), valid.copy(), sup.copy(), vis.copy(), evidence.copy(),
    )
    for i in range(kp.shape[0]):
        t = TASKS[int_to_task_id(int(task_ids[i]))]
        pairs = t.hflip_pairs if axis == "h" else t.vflip_pairs
        for a, b in pairs:
            ga, gb = t.offset + a, t.offset + b
            kp[i, ga], kp[i, gb] = kp[i, gb].copy(), kp[i, ga].copy()
            valid[i, ga], valid[i, gb] = valid[i, gb], valid[i, ga]
            sup[i, ga], sup[i, gb] = sup[i, gb], sup[i, ga]
            vis[i, ga], vis[i, gb] = vis[i, gb], vis[i, ga]
            evidence[i, ga], evidence[i, gb] = evidence[i, gb], evidence[i, ga]
    return kp, valid, sup, vis, evidence


class TransformPipeline:
    """Applies spatial affine + photometric + normalize to a SampleDict.

    Keys in:
        image             — [H, W, 3] uint8
        keypoints         — [I, 60, 2] float32
        transform_matrix  — [I, 3, 3] float32
        landmark_valid_mask      — [I, 60] bool
        landmark_supervised_mask — [I, 60] bool
        landmark_visible_mask    — [I, 60] bool
        landmark_evidence_mask   — [I, 60] int64 (optional, defaults to zeros)
        task_ids          — [I] int64
        (all other keys pass through)

    Keys out (same keys, updated values):
        image             — [tgt_H, tgt_W, 3] float32 normalized (or uint8 if skip_normalize)
        keypoints         — [I, 60, 2] float32, transformed
        transform_matrix  — [I, 3, 3] float32, composed M_post @ M_old
        landmark_visible_mask — [I, 60] bool, recomputed from in-bounds
        landmark_evidence_mask — [I, 60] int64, flip-permuted
        was_hflipped      — bool
        was_vflipped      — bool
    """

    def __init__(self, config: TransformConfig, seed: int | None = None) -> None:
        self._cfg = config
        self._seed = seed
        self._epoch = 0
        self._rng: np.random.Generator | None = None
        self._rng_key: tuple[int, int] | None = None
        self._photo = self._build_photometric(config)
        self._mean = np.array(config.normalize_mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(config.normalize_std, dtype=np.float32).reshape(1, 1, 3)
        self._da_enabled = True
        self._flips_enabled = True

    def set_da_enabled(self, enabled: bool) -> None:
        """Toggle spatial + photometric DA. Resize and normalize always apply."""
        self._da_enabled = enabled

    def set_flips_enabled(self, enabled: bool) -> None:
        """Toggle BOTH flips independently of the rest of the DA schedule.

        Mutable state rather than a config edit because TransformConfig is frozen
        and shared with worker processes; the datamodule flips this per epoch so
        the tail of training sees only unmirrored anatomy, the same reasoning as
        closing mosaic.
        """
        self._flips_enabled = enabled

    @staticmethod
    def _build_photometric(cfg: TransformConfig) -> A.Compose | None:
        import albumentations as A

        transforms: list[A.BasicTransform] = []
        if cfg.brightness_limit > 0 or cfg.contrast_limit > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit,
                    p=1.0,
                )
            )
        if not transforms:
            return None
        return A.Compose(transforms)

    def _sample_spatial_params(
        self,
        rng: np.random.Generator,
        target_size: tuple[int, int],
    ) -> SpatialParams:
        cfg = self._cfg
        task_no_flip = (
            hasattr(self, '_current_task_id')
            and self._current_task_id in cfg.no_flip_tasks
        )
        do_hflip = self._flips_enabled and not task_no_flip and rng.random() < cfg.hflip_prob
        do_vflip = self._flips_enabled and not task_no_flip and rng.random() < cfg.vflip_prob

        rotation = rng.uniform(-cfg.rotation_range, cfg.rotation_range)
        s_lo, s_hi = cfg.scale_range
        if cfg.task_scale_override and hasattr(self, '_current_task_id'):
            s_lo, s_hi = cfg.task_scale_override.get(self._current_task_id, (s_lo, s_hi))
        scale = rng.uniform(s_lo, s_hi)
        tx = rng.uniform(-cfg.translate_range, cfg.translate_range)
        ty = rng.uniform(-cfg.translate_range, cfg.translate_range)

        return SpatialParams(
            target_size=target_size,
            hflip=do_hflip,
            vflip=do_vflip,
            rotation_deg=rotation,
            scale=scale,
            translate_x=tx,
            translate_y=ty,
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-key the augmentation stream so each epoch draws different views."""
        self._epoch = epoch
        self._rng = None

    def _sample_rng(self, sample: SampleDict) -> np.random.Generator:
        """One deterministic stream per (seed, epoch, worker), advanced per call.

        Reproducible because the stream is seeded from (seed, epoch, worker_id)
        and both the sampler order and the worker sharding are deterministic.
        Replaces a bare `np.random.default_rng()`, which drew from OS entropy so
        `seed` controlled nothing and no run — nor any A/B between runs — was
        reproducible.

        Deliberately NOT keyed on sample identity. MixedTaskBatchSampler balances
        tasks by cycling the small ones, so within a single epoch one image is
        drawn many times — up to 327 for IVC (30 labeled images). Keying on
        identity hands every one of those draws the SAME augmentation, wiping out
        augmentation diversity exactly where the data is scarcest. That is what
        an identity-keyed version did, and training collapsed from epoch 1.
        """
        if self._seed is None:
            return np.random.default_rng()

        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else -1
        if self._rng is None or self._rng_key != (self._epoch, worker_id):
            self._rng = np.random.default_rng([self._seed, self._epoch, worker_id + 1])
            self._rng_key = (self._epoch, worker_id)
        return self._rng

    def __call__(self, sample: SampleDict) -> SampleDict:
        rng = self._sample_rng(sample)
        image: np.ndarray = sample["image"]
        keypoints: np.ndarray = sample["keypoints"]
        M_old: np.ndarray = sample["transform_matrix"]
        valid_mask: np.ndarray = sample["landmark_valid_mask"]
        supervised_mask: np.ndarray = sample["landmark_supervised_mask"]
        task_ids: np.ndarray = sample["task_ids"]

        src_h, src_w = image.shape[:2]
        cfg = self._cfg

        # Effective target: source dimensions when skip_resize (identity resize)
        effective_target = (src_w, src_h) if cfg.skip_resize else cfg.target_size

        # 1. Sample spatial parameters (identity when DA disabled)
        # Per-task scale override: set task_id for _sample_spatial_params
        if task_ids.size > 0:
            self._current_task_id = int_to_task_id(int(task_ids[0]))
        if self._da_enabled:
            params = self._sample_spatial_params(rng, effective_target)
        else:
            params = SpatialParams(target_size=effective_target)

        # 2. Build affine matrix (source → augmented)
        M_post = build_affine_matrix(
            src_size=(src_w, src_h),
            params=params,
            resize_mode=cfg.resize_mode,
        )

        # 3. Apply spatial to image and keypoints
        image_out, kp_out, visible_mask = apply_spatial(
            image,
            keypoints,
            M_post,
            effective_target,
        )

        # 4. Landmark permutation — swap anatomically symmetric pairs per axis
        evidence_mask = sample.get(
            "landmark_evidence_mask",
            np.zeros_like(valid_mask, dtype=np.int64),
        )
        for axis, flipped in [("h", params.hflip), ("v", params.vflip)]:
            if flipped:
                kp_out, valid_mask, supervised_mask, visible_mask, evidence_mask = (
                    _apply_flip_permutation(
                        kp_out, valid_mask, supervised_mask, visible_mask,
                        evidence_mask, task_ids, axis=axis,
                    )
                )

        # 5. Compose transform: M_new[i] = M_post @ M_old[i]
        M_post_f32 = M_post.astype(np.float32)
        n_inst = M_old.shape[0]
        M_new = np.stack([M_post_f32 @ M_old[i] for i in range(n_inst)])

        # 6. Photometric augmentation (intensity only, no geometry; skip when DA disabled)
        if self._da_enabled and self._photo is not None:
            augmented = self._photo(image=image_out)
            image_out = augmented["image"]

        # 7. Normalize (skip for per-instance DA — output stays uint8)
        if not cfg.skip_normalize:
            image_out = image_out.astype(np.float32) / 255.0
            image_out = (image_out - self._mean) / self._std

        return {
            **sample,
            "image": image_out,
            "keypoints": kp_out,
            "transform_matrix": M_new,
            "landmark_valid_mask": valid_mask,
            "landmark_supervised_mask": supervised_mask,
            "landmark_visible_mask": visible_mask,
            "landmark_evidence_mask": evidence_mask,
            "was_hflipped": params.hflip,
            "was_vflipped": params.vflip,
        }


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class TransformedDataset:
    """Wraps a base dataset with a TransformPipeline.

    Delegates __len__, get_task_indices, get_split_indices to the base.

    Keys in: whatever the base dataset produces (SampleDict).
    Keys out: TransformPipeline output (augmented + normalized SampleDict).
    """

    def __init__(
        self,
        base_dataset: Any,
        transform: TransformPipeline,
    ) -> None:
        self._base = base_dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> SampleDict:
        sample = self._base[idx]
        return self._transform(sample)

    def get_task_indices(self, task_id: str) -> Sequence[int]:
        """Delegate to base dataset."""
        return self._base.get_task_indices(task_id)

    def get_split_indices(self, split: str) -> Sequence[int]:
        """Delegate to base dataset."""
        return self._base.get_split_indices(split)

    def get_labeled_indices(self) -> Sequence[int]:
        """Delegate to base dataset."""
        return self._base.get_labeled_indices()

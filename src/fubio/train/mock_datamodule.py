"""Synthetic data generator for development and end-to-end pipeline testing.

Generates 518×518 images with landmark dots drawn at GT positions,
providing a learnable signal: if the model can't overfit this, the
gradient path is broken.

Upstream: data/types.py (InstanceDict, BatchDict contracts).
Downstream: train/module.py consumes BatchDict from train/val dataloaders.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fubio.data.task_registry import TASKS
from fubio.data.types import BatchDict, InstanceDict

# Pre-compute task_int → n_keypoints for fast lookup
_TASK_K: dict[int, int] = {t.task_int: t.n_keypoints for t in TASKS.values()}
_N_TASKS = len(TASKS)

# ImageNet normalization (must match transforms.py defaults)

# Distinct colors per task for visual debugging (RGB uint8)
_TASK_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 80, 80),  # A4C — red
    1: (80, 255, 80),  # PLAX — green
    2: (80, 80, 255),  # PSAX — blue
    3: (255, 255, 80),  # IVC — yellow
    4: (255, 80, 255),  # AOP — magenta
    5: (80, 255, 255),  # FUGC — cyan
    6: (255, 160, 80),  # HC — orange
    7: (160, 80, 255),  # FA — purple
    8: (80, 255, 160),  # Femur — teal
}


def _draw_dot(
    image: np.ndarray,
    x_norm: float,
    y_norm: float,
    color: tuple[int, int, int],
    radius: int = 3,
) -> None:
    """Draw a filled circle on HWC uint8 image at normalized coords."""
    h, w = image.shape[:2]
    cx = int(x_norm * w)
    cy = int(y_norm * h)
    y_lo = max(0, cy - radius)
    y_hi = min(h, cy + radius + 1)
    x_lo = max(0, cx - radius)
    x_hi = min(w, cx + radius + 1)
    for y in range(y_lo, y_hi):
        for x in range(x_lo, x_hi):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                image[y, x] = color


def _bbox_from_keypoints(kp: np.ndarray) -> np.ndarray:
    """Compute cx, cy, w, h bbox from (K, 2) normalized keypoints."""
    mins = kp.min(axis=0)
    maxs = kp.max(axis=0)
    margin = 0.02
    return np.array(
        [
            (mins[0] + maxs[0]) / 2,
            (mins[1] + maxs[1]) / 2,
            maxs[0] - mins[0] + margin,
            maxs[1] - mins[1] + margin,
        ],
        dtype=np.float32,
    ).clip(0.0, 1.0)


class MockDataset(Dataset[dict[str, object]]):
    """Generates deterministic synthetic samples with visible landmark dots.

    Each sample: random background + 1-3 instances (random tasks) with
    colored dots at keypoint locations. Per-index seed guarantees
    reproducibility regardless of access order.
    """

    def __init__(
        self,
        n_samples: int = 200,
        seed: int = 42,
        image_size: tuple[int, int] = (518, 518),
        max_instances: int = 3,
    ) -> None:
        self._n = n_samples
        self._seed = seed
        self._w, self._h = image_size
        self._max_instances = max_instances

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, object]:
        rng = np.random.default_rng(self._seed + idx)
        image = rng.integers(30, 120, (self._h, self._w, 3), dtype=np.uint8)

        n_inst = rng.integers(1, self._max_instances + 1)
        instances: list[InstanceDict] = []

        for _ in range(n_inst):
            task_int = int(rng.integers(0, _N_TASKS))
            k = _TASK_K[task_int]

            kp = rng.uniform(0.1, 0.9, (k, 2)).astype(np.float32)

            color = _TASK_COLORS[task_int]
            for j in range(k):
                _draw_dot(image, float(kp[j, 0]), float(kp[j, 1]), color)

            inst: InstanceDict = {
                "task_id": task_int,
                "keypoints": kp,
                "supervised_mask": np.ones(k, dtype=bool),
                "visible_mask": np.ones(k, dtype=bool),
                "bbox": _bbox_from_keypoints(kp),
                "transform_matrix": np.eye(3, dtype=np.float32),
                "original_hw": np.array([self._h, self._w], dtype=np.int32),
                "is_labeled": True,
                "image_path": f"mock/{idx}.png",
            }
            instances.append(inst)

        # Return uint8 — GPU normalize happens in FUBioModule._normalize_image
        return {"image": image, "instances": instances}


def collate_model(batch: list[dict[str, object]]) -> BatchDict:
    """Collate for the model pipeline: stack images, keep targets as lists.

    Keys in (per sample):
        image     — [H, W, 3] uint8 HWC
        instances — list[InstanceDict]

    Keys out (BatchDict):
        image   — [B, 3, H, W] uint8 (CHW), normalized on GPU by FUBioModule
        targets — list[list[InstanceDict]]
    """
    images: list[Tensor] = []
    targets: list[list[InstanceDict]] = []

    for sample in batch:
        img = sample["image"]
        assert isinstance(img, np.ndarray)
        images.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))))
        insts = sample["instances"]
        assert isinstance(insts, list)
        targets.append(insts)

    return {
        "image": torch.stack(images),
        "targets": targets,
    }


class MockDataModule(L.LightningDataModule):
    """Lightning DataModule producing synthetic BatchDicts for development.

    Gate: 5 epochs train + val with loss decrease proves gradient flow
    through the entire pipeline.
    """

    def __init__(
        self,
        n_train: int = 200,
        n_val: int = 50,
        batch_size: int = 4,
        num_workers: int = 0,
        seed: int = 42,
        image_size: tuple[int, int] = (518, 518),
    ) -> None:
        super().__init__()
        self._n_train = n_train
        self._n_val = n_val
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed
        self._image_size = image_size
        self._train_ds: MockDataset | None = None
        self._val_ds: MockDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = MockDataset(
                n_samples=self._n_train,
                seed=self._seed,
                image_size=self._image_size,
            )
        if self._val_ds is None:
            self._val_ds = MockDataset(
                n_samples=self._n_val,
                seed=self._seed + 10_000,
                image_size=self._image_size,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=collate_model,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=collate_model,
            persistent_workers=self._num_workers > 0,
        )

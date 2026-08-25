"""DataLoader factory — wires ManifestDataset → (Mosaic) → Transform → DataLoader.

Provides a single entry point to construct train/val DataLoaders with the
correct pipeline ordering, sampler, and collation.

Upstream: manifest.py, transforms.py, collate.py, sampler.py, mosaic.py (optional).
Downstream: training loop (LightningDataModule or manual loop).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from fubio.data.collate import collate_fubio
from fubio.data.manifest import ManifestDataset
from fubio.data.mosaic import MosaicConfig, MosaicDataset
from fubio.data.sampler import MixedTaskBatchSampler, worker_init_fn
from fubio.data.task_registry import TASKS
from fubio.data.transforms import TransformConfig, TransformedDataset, TransformPipeline

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Top-level data pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    manifest_path: Path = Path("data/manifest.parquet")
    data_root: Path = Path("data")

    # Transform
    train_transform: TransformConfig = TransformConfig()
    val_transform: TransformConfig = TransformConfig()

    # Mosaic (training only)
    mosaic: MosaicConfig | None = None

    # DataLoader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42


def build_train_loader(
    cfg: DataConfig,
    task_filter: str | list[str] | None = None,
) -> DataLoader:
    """Build training DataLoader with balanced task sampling.

    Pipeline: ManifestDataset → [MosaicDataset] → TransformedDataset → DataLoader.
    """
    base = ManifestDataset(
        manifest_path=cfg.manifest_path,
        data_root=cfg.data_root,
        split_filter=["train_local", "train_labeled"],
        task_filter=task_filter,
    )

    # Optional mosaic wrapping (before transforms)
    pre_transform = base
    if cfg.mosaic is not None and cfg.mosaic.n_instances > 1:
        pre_transform = MosaicDataset(base, cfg.mosaic, seed=cfg.seed)

    transform = TransformPipeline(cfg.train_transform)
    dataset = TransformedDataset(pre_transform, transform)

    task_ids = (
        list(TASKS.keys())
        if task_filter is None
        else ([task_filter] if isinstance(task_filter, str) else task_filter)
    )
    task_indices = {tid: base.get_task_indices(tid) for tid in task_ids}
    # Drop tasks with no samples
    task_indices = {k: v for k, v in task_indices.items() if v}

    sampler = MixedTaskBatchSampler(
        task_indices=task_indices,
        batch_size=cfg.batch_size,
        drop_last=True,
        seed=cfg.seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fubio,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
    )


def build_val_loader(
    cfg: DataConfig,
    split: str = "val_local",
    task_filter: str | list[str] | None = None,
) -> DataLoader:
    """Build validation DataLoader with sequential access (no sampling).

    Pipeline: ManifestDataset → TransformedDataset → DataLoader(sequential).
    """
    base = ManifestDataset(
        manifest_path=cfg.manifest_path,
        data_root=cfg.data_root,
        split_filter=split,
        task_filter=task_filter,
    )

    transform = TransformPipeline(cfg.val_transform)
    dataset = TransformedDataset(base, transform)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fubio,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

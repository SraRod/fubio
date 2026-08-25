"""Real-data LightningDataModule — wraps data/ pipeline with model collate.

Uses ManifestDataset → MosaicDataset → TransformPipeline pipeline,
with collate_for_model to produce BatchDict.

Expects manifest to contain train_local / val_local splits
(produced by data/split.py DINOv2 diversity split), plus train_unlabeled
when semi.enabled.

Two training streams, deliberately separate rather than one mixed dataset:

    labeled    train_local (6.4K)      mosaic + albumentations DA   BatchDict
    unlabeled  train_unlabeled (191K)  resize + normalize only      BatchDict

They are kept apart because their augmentation contracts differ. The labeled
stream composes mosaics and applies CPU geometric DA against known keypoints;
the unlabeled stream must stay geometrically untouched here, because the
equivariance loss supplies its own differentiable on-GPU transform and needs a
canonical reference view to transform FROM.

Upstream: data/split.py (split assignment), data/collate.py (collate_for_model).
Downstream: train/train.py, train/module.py (training_step unpacks the dict).
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from fubio.data.collate import collate_for_model
from fubio.data.manifest import CachedManifestDataset, ManifestDataset
from fubio.data.mosaic import MosaicDataset
from fubio.data.sampler import MixedTaskBatchSampler, worker_init_fn
from fubio.data.task_registry import TASKS
from fubio.data.transforms import TransformedDataset, TransformPipeline
from fubio.train.config import DataConfig, SemiConfig

logger = logging.getLogger(__name__)


class FUBioDataModule(L.LightningDataModule):
    """Real-data module producing BatchDict for FUBioModule.

    All data/augmentation/loader parameters come from DataConfig.
    train split: train_local (labeled, from diversity split).
    val split: val_local (labeled, from diversity split).
    """

    def __init__(
        self,
        data_config: DataConfig,
        seed: int = 42,
        manifest_path: Path = Path("data/manifest.parquet"),
        data_root: Path = Path("data"),
        semi_config: SemiConfig | None = None,
        tune_tasks: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._cfg = data_config
        self._semi = semi_config if semi_config is not None else SemiConfig()
        self._tune_tasks = tune_tasks
        self._manifest_path = manifest_path
        self._data_root = data_root
        self._seed = seed

        # GPU normalize: skip CPU-side normalize, FUBioModule._normalize_image handles it
        self._train_transform_cfg = data_config.transform.model_copy(
            update={"skip_normalize": True},
        )
        self._val_transform_cfg = data_config.transform.model_copy(
            update={
                "hflip_prob": 0.0,
                "vflip_prob": 0.0,
                "rotation_range": 0.0,
                "scale_range": (1.0, 1.0),
                "translate_range": 0.0,
                "brightness_limit": 0.0,
                "contrast_limit": 0.0,
                "skip_normalize": True,
            },
        )

        self._train_ds: TransformedDataset | None = None
        self._val_ds: TransformedDataset | None = None
        self._train_mosaic: MosaicDataset | None = None
        self._train_sampler: MixedTaskBatchSampler | None = None
        self._train_transform: TransformPipeline | None = None
        self._unlabeled_ds: TransformedDataset | None = None
        self._unlabeled_base: ManifestDataset | CachedManifestDataset | None = None
        self._unlabeled_sampler: MixedTaskBatchSampler | None = None

    def _resolve_cache_dir(self) -> Path | None:
        """Return cache dir if a valid pre-built cache exists for current target_size."""
        from fubio.data.build_cache import _cache_dir_name

        cfg = self._cfg.transform
        tgt_w, tgt_h = cfg.target_size
        name = _cache_dir_name(tgt_w, tgt_h, cfg.resize_mode)
        cache_dir = self._data_root / "cache" / name
        if (cache_dir / "cache_info.json").exists():
            return cache_dir
        return None

    def _make_base_dataset(
        self,
        split_filter: str | list[str],
        cache_dir: Path | None,
    ) -> ManifestDataset | CachedManifestDataset:
        if cache_dir is not None:
            return CachedManifestDataset(
                cache_dir=cache_dir,
                manifest_path=self._manifest_path,
                data_root=self._data_root,
                split_filter=split_filter,
            )
        return ManifestDataset(
            manifest_path=self._manifest_path,
            data_root=self._data_root,
            split_filter=split_filter,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            cache_dir = self._resolve_cache_dir()
            use_cache = cache_dir is not None
            if use_cache:
                logger.info("Using pre-built cache: %s", cache_dir)

            # When cache provides pre-resized images, skip resize in transforms
            train_tcfg = self._train_transform_cfg
            val_tcfg = self._val_transform_cfg
            if use_cache:
                train_tcfg = train_tcfg.model_copy(update={"skip_resize": True})
                val_tcfg = val_tcfg.model_copy(update={"skip_resize": True})

            if self._cfg.fold_val_local:
                train_base = self._make_base_dataset(
                    ["train_local", "val_local"], cache_dir,
                )
                logger.info("Folded val_local into training: %d total labeled", len(train_base))
            else:
                train_base = self._make_base_dataset("train_local", cache_dir)
            train_mosaic = MosaicDataset(
                train_base,
                config=self._cfg.mosaic,
                seed=self._seed,
            )
            train_transform = TransformPipeline(train_tcfg, seed=self._seed)
            self._train_ds = TransformedDataset(train_mosaic, train_transform)
            self._train_mosaic = train_mosaic
            self._train_transform = train_transform

            if self._cfg.fold_val_local:
                val_base = self._make_base_dataset("train_local", cache_dir)
                logger.info("fold_val_local: using train_local as val proxy (no GT validation)")
            else:
                val_base = self._make_base_dataset("val_local", cache_dir)
            self._val_ds = TransformedDataset(
                val_base,
                TransformPipeline(val_tcfg),
            )

            if self._semi.enabled:
                # Reuses val_tcfg: every geometric and photometric knob already
                # zeroed. Not an oversight that the unlabeled stream gets no DA —
                # the semi step transforms this view itself, on GPU and
                # differentiably, and needs it to be the untransformed reference.
                unlabeled_splits = self._semi.unlabeled_splits
                unlabeled_base = self._make_base_dataset(unlabeled_splits, cache_dir)
                self._unlabeled_base = unlabeled_base
                self._unlabeled_ds = TransformedDataset(
                    unlabeled_base,
                    TransformPipeline(val_tcfg, seed=self._seed),
                )
                logger.info("Semi-supervised: %d unlabeled images", len(unlabeled_base))

    def train_dataloader(self) -> DataLoader | CombinedLoader:
        assert self._train_ds is not None

        active_tasks = self._tune_tasks if self._tune_tasks else list(TASKS)
        task_indices = {tid: self._train_mosaic.get_task_indices(tid) for tid in active_tasks}
        task_indices = {k: v for k, v in task_indices.items() if v}

        sampler = MixedTaskBatchSampler(
            task_indices=task_indices,
            batch_size=self._cfg.batch_size,
            drop_last=True,
            seed=self._seed,
        )
        self._train_sampler = sampler

        # Workers are forked and never see main-process state changes, so any
        # per-epoch state must be re-forked each epoch. The augmentation RNG is
        # now keyed on epoch (see TransformPipeline._sample_rng), which makes
        # set_epoch matter on EVERY run, not just ones with a mosaic/flip
        # schedule — persistent workers would freeze every epoch on epoch 0's
        # augmentation stream. Cost: ~1s worker restart per epoch vs a 5-min epoch.
        persistent = False

        labeled_loader = DataLoader(
            self._train_ds,
            batch_sampler=sampler,
            num_workers=self._cfg.num_workers,
            pin_memory=self._cfg.pin_memory,
            collate_fn=collate_for_model,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent,
        )
        if not self._semi.enabled:
            return labeled_loader

        # "min_size": the labeled loader defines the epoch, so the LR schedule,
        # epoch count and per-step labeled composition stay identical to a
        # supervised run and the comparison stays attributable. The alternative,
        # "max_size_cycle", would stretch one epoch to the unlabeled pool's ~65x
        # length and recycle the labeled set within it without a set_epoch call —
        # freezing its augmentation and mosaic RNG for the whole pass.
        return CombinedLoader(
            {"labeled": labeled_loader, "unlabeled": self._unlabeled_dataloader()},
            mode="min_size",
        )

    def _unlabeled_dataloader(self) -> DataLoader:
        assert self._unlabeled_ds is not None
        assert self._unlabeled_base is not None

        task_indices = {tid: self._unlabeled_base.get_task_indices(tid) for tid in TASKS}
        task_indices = {k: v for k, v in task_indices.items() if v}

        sampler = MixedTaskBatchSampler(
            task_indices=task_indices,
            batch_size=self._semi.batch_size or self._cfg.batch_size,
            drop_last=True,
            seed=self._seed,
            # See MixedTaskBatchSampler: only a prefix of these batches is read
            # each epoch, so the cursor has to carry over for the pool to be
            # covered at all.
            persist_cursor=True,
        )
        self._unlabeled_sampler = sampler

        return DataLoader(
            self._unlabeled_ds,
            batch_sampler=sampler,
            num_workers=self._cfg.num_workers,
            pin_memory=self._cfg.pin_memory,
            collate_fn=collate_for_model,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
        )

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to mosaic, sampler, and DA schedule."""
        if self._train_mosaic is not None:
            self._train_mosaic.set_epoch(epoch)
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        if self._unlabeled_sampler is not None:
            self._unlabeled_sampler.set_epoch(epoch)
        if self._train_transform is not None:
            self._train_transform.set_epoch(epoch)
        if self._train_transform is not None and self._cfg.da_open_epoch is not None:
            self._train_transform.set_da_enabled(epoch >= self._cfg.da_open_epoch)
        # Close both flips for the tail of training, same rationale as mosaic closing:
        # finish on the unmirrored distribution the model is evaluated against.
        if self._train_transform is not None and self._cfg.flip_close_epoch is not None:
            self._train_transform.set_flips_enabled(epoch < self._cfg.flip_close_epoch)

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self._cfg.batch_size,
            shuffle=False,
            num_workers=self._cfg.num_workers,
            pin_memory=self._cfg.pin_memory,
            collate_fn=collate_for_model,
            worker_init_fn=worker_init_fn,
            persistent_workers=self._cfg.num_workers > 0,
            generator=torch.Generator().manual_seed(self._seed),
        )

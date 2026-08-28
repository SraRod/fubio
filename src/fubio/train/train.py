"""CLI entry point for training FU_Biometry model.

Usage:
    uv run python -m fubio.train.train fit --mock
    uv run python -m fubio.train.train fit                    # real data, wandb offline
    uv run python -m fubio.train.train fit --config configs/round1.yaml

Upstream: train/config.py, train/module.py, train/datamodule.py, data/split.py.
Downstream: none (top-level entry point).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path, PosixPath, PurePosixPath

import lightning as L
import polars as pl
import torch

# PyTorch 2.6 defaults weights_only=True; Lightning checkpoints contain PosixPath
torch.serialization.add_safe_globals([Path, PosixPath, PurePosixPath])
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from fubio.train.callbacks import CSVCallback, FinalEpochCheckpoint, InstanceCollapseGuard
from fubio.train.config import ExperimentConfig
from fubio.train.datamodule import FUBioDataModule
from fubio.train.mock_datamodule import MockDataModule
from fubio.train.module import FUBioModule
from fubio.train.viz_callback import VisualizationCallback

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/manifest.parquet")
DATA_ROOT = Path("data")


def _deep_update(base: dict, extra: dict) -> dict:
    """Merge nested override dicts without wiping sibling keys.

    A shallow ``dict.update`` would replace the whole ``optimizer`` block when
    ``--epochs`` overrides only ``optimizer.max_epochs``.
    """
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _ensure_split(
    manifest_path: Path,
    data_root: Path,
    seed: int,
) -> None:
    """Run the stratified split if the manifest lacks train_local/val_local."""
    df = pl.read_parquet(manifest_path)
    splits = set(df["split"].unique().to_list())

    if "train_local" in splits and "val_local" in splits:
        n_train = df.filter(pl.col("split") == "train_local").height
        n_val = df.filter(pl.col("split") == "val_local").height
        logger.info("Split already exists: train_local=%d, val_local=%d", n_train, n_val)
        return

    logger.info("No train_local/val_local split found — running stratified split...")
    from fubio.data.split import _write_manifest, create_stratified_val_split

    result, report = create_stratified_val_split(df, seed=seed)
    _write_manifest(result, manifest_path)

    import json

    report_path = data_root / "split_report_stratified.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Split report saved: %s", report_path)


def fit(
    config_path: Path | None = None,
    mock: bool = False,
    overrides: dict | None = None,
    wandb_offline: bool = False,
    limit_train_batches: int | float | None = None,
    ckpt_path: Path | None = None,
    wandb_run_id: str | None = None,
    init_weights: Path | None = None,
) -> None:
    """Train FUBioModule with given config.

    Args:
        config_path: YAML config file. None uses all defaults.
        mock: Use MockDataModule instead of real data.
        overrides: Dict of config overrides (merged on top of YAML).
        wandb_offline: Save wandb logs locally (upload later with wandb sync).
        init_weights: Load model weights from checkpoint but start fresh
            optimizer/scheduler/epoch. Mutually exclusive with ckpt_path.
    """
    if config_path is not None:
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f) or {}
        if overrides:
            _deep_update(cfg_dict, overrides)
        config = ExperimentConfig(**cfg_dict)
    else:
        config = ExperimentConfig(**(overrides or {}))

    torch.set_float32_matmul_precision("high")
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        # warn_only: DINOv2 bicubic backward has no deterministic CUDA impl
        torch.use_deterministic_algorithms(True, warn_only=True)
    L.seed_everything(config.seed, workers=True)

    # Ensure train/val split exists (runs DINOv2 diversity split on first run)
    if not mock:
        _ensure_split(MANIFEST_PATH, DATA_ROOT, config.seed)
        # After _ensure_split, since that call can rewrite the manifest — the exact
        # sequence that silently invalidated the prior once already.
        from fubio.data.shape_prior import verify_shape_prior_provenance

        verify_shape_prior_provenance(config.loss.shape_prior_path, MANIFEST_PATH)

    module = FUBioModule(config)

    if init_weights is not None:
        ckpt = torch.load(init_weights, map_location="cpu", weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        logger.info("Loaded model weights from %s (fresh optimizer/scheduler)", init_weights)

    if mock:
        datamodule: L.LightningDataModule = MockDataModule(
            n_train=200,
            n_val=50,
            batch_size=4,
            seed=config.seed,
        )
    else:
        datamodule = FUBioDataModule(
            data_config=config.data,
            seed=config.seed,
            semi_config=config.semi,
            tune_tasks=config.head_tune.tune_tasks if config.head_tune.enabled else None,
        )

    # Select on the served metric, not val/loss. val/loss is a weighted sum of
    # bbox/conf/landmark/param/heatmap/shape terms whose composition changes
    # between experiments (R17 added repulsion), so it is not comparable across
    # runs and does not track the scored quantity. mre_pixel/overall is
    # domain-balanced, matching the challenge's ChallengeR aggregation.
    # semi/loss_total when fold_val_local (val_mre is train error);
    # val_mre_overall when val is held out.
    ckpt_monitor = "semi/loss_total" if config.data.fold_val_local else "val_mre_overall"
    ckpt_filename = (
        "fubio-{epoch:03d}-{semi/loss_total:.5f}"
        if config.data.fold_val_local
        else "fubio-{epoch:03d}-{val_mre_overall:.3f}"
    )
    checkpoint = ModelCheckpoint(
        monitor=ckpt_monitor,
        mode="min",
        save_top_k=-1,
        save_last=True,
        filename=ckpt_filename,
    )
    # Second view: worst per-task MRE. Rank aggregation penalises a task ranked
    # poorly regardless of margin, so a checkpoint that improves the mean while
    # tanking one task is a bad trade the mean alone would hide.
    worst_task_checkpoint = ModelCheckpoint(
        monitor="val_mre_worst_task",
        mode="min",
        save_top_k=1,
        filename="fubio-worst-{epoch:03d}-{val_mre_worst_task:.3f}",
    )
    callbacks: list[L.Callback] = [
        checkpoint,
        worst_task_checkpoint,
        CSVCallback(),
        FinalEpochCheckpoint(),
        InstanceCollapseGuard(),
        VisualizationCallback(log_every_n_epochs=5),
    ]

    # Wandb logger — offline by default, sync later
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    if mock:
        run_name = "mock-baseline"
    elif config_path is not None:
        run_name = config_path.stem
    else:
        run_name = "labeled-baseline"
    wandb_kwargs: dict = {
        "project": "fubio",
        "name": run_name,
        "save_dir": "wandb_logs",
        "log_model": False,
    }
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb_logger = WandbLogger(**wandb_kwargs)

    trainer_kwargs: dict = {
        "max_epochs": config.optimizer.max_epochs,
        "precision": config.precision,
        "callbacks": callbacks,
        "deterministic": False if config.cudnn_benchmark else "warn",
        "logger": wandb_logger,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": config.optimizer.accumulate_grad_batches,
    }
    # CLI override > config default
    effective_limit = (
        limit_train_batches if limit_train_batches is not None else config.data.limit_train_batches
    )
    if effective_limit is not None:
        trainer_kwargs["limit_train_batches"] = effective_limit

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)

    logger.info("Training complete. Best checkpoint: %s", checkpoint.best_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FU_Biometry model")
    parser.add_argument("command", choices=["fit"], help="Training command")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--mock", action="store_true", help="Use MockDataModule")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Cap train batches/epoch",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Load model weights only (fresh optimizer/scheduler/epoch)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable cudnn.benchmark (non-deterministic, ~36%% faster)",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="Resume into existing wandb run ID",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Log locally without a wandb account (sync later with `wandb sync`)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "fit":
        overrides: dict = {}
        if args.epochs is not None:
            overrides["optimizer"] = {"max_epochs": args.epochs}
        if args.benchmark:
            overrides["cudnn_benchmark"] = True
        fit(
            config_path=args.config,
            mock=args.mock,
            overrides=overrides or None,
            wandb_offline=args.wandb_offline,
            limit_train_batches=args.limit_train_batches,
            ckpt_path=args.ckpt,
            wandb_run_id=args.wandb_id,
            init_weights=args.init_weights,
        )

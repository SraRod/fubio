"""CLI entry point: _deep_update, _ensure_split early-return, and a fit() smoke.

The smoke test runs the real fit() end to end on MockDataModule with the stub
backbone, forced onto CPU — a GPU training job may be running on this machine,
so the Trainer is wrapped to pin accelerator="cpu" and wandb is disabled.

Upstream: train/train.py.
"""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest
import torch
import yaml

from fubio.train.train import _deep_update, _ensure_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_PRIOR_V4 = REPO_ROOT / "data" / "shape_prior_v4.json"


# ---------------------------------------------------------------------------
# _deep_update
# ---------------------------------------------------------------------------


class TestDeepUpdate:
    def test_nested_override_keeps_sibling_keys(self) -> None:
        """The --epochs regression: overriding optimizer.max_epochs must not
        wipe the rest of the optimizer block."""
        base = {"optimizer": {"max_epochs": 100, "lr_backbone": 1e-5}, "seed": 42}
        out = _deep_update(base, {"optimizer": {"max_epochs": 1}})
        assert out["optimizer"] == {"max_epochs": 1, "lr_backbone": 1e-5}
        assert out["seed"] == 42

    def test_mutates_and_returns_base(self) -> None:
        base = {"a": 1}
        assert _deep_update(base, {"b": 2}) is base
        assert base == {"a": 1, "b": 2}

    def test_non_dict_value_replaces_dict(self) -> None:
        base = {"neck": {"mode": "c2f"}}
        assert _deep_update(base, {"neck": None}) == {"neck": None}

    def test_recurses_multiple_levels(self) -> None:
        base = {"head": {"coord": {"mode": "geo_simcc", "tau": 1.0}}}
        out = _deep_update(base, {"head": {"coord": {"tau": 2.0}}})
        assert out["head"]["coord"] == {"mode": "geo_simcc", "tau": 2.0}


# ---------------------------------------------------------------------------
# _ensure_split
# ---------------------------------------------------------------------------


def test_ensure_split_returns_early_when_split_exists(tmp_path: Path) -> None:
    """A manifest already carrying train_local/val_local must not be rewritten."""
    manifest = tmp_path / "manifest.parquet"
    df = pl.DataFrame({"split": ["train_local", "train_local", "val_local"]})
    df.write_parquet(manifest)
    before = manifest.read_bytes()

    _ensure_split(manifest, tmp_path, seed=0)

    assert manifest.read_bytes() == before
    assert not (tmp_path / "split_report_stratified.json").exists()


def test_ensure_split_runs_split_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No train_local/val_local → run the (stubbed) split and save the report."""
    import fubio.data.split as split_mod

    manifest = tmp_path / "manifest.parquet"
    pl.DataFrame({"split": ["train", "train"]}).write_parquet(manifest)

    calls: dict[str, object] = {}

    def fake_split(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, dict]:
        calls["seed"] = seed
        return df, {"n_val": 1}

    monkeypatch.setattr(split_mod, "create_stratified_val_split", fake_split)
    monkeypatch.setattr(
        split_mod, "_write_manifest", lambda df, path: calls.setdefault("written", path)
    )

    _ensure_split(manifest, tmp_path, seed=7)

    assert calls["seed"] == 7
    assert calls["written"] == manifest
    report = json.loads((tmp_path / "split_report_stratified.json").read_text())
    assert report == {"n_val": 1}


# ---------------------------------------------------------------------------
# fit() smoke — the CLI smoke test.
# Covers config assembly, callback wiring (CSV / FinalEpochCheckpoint /
# InstanceCollapseGuard / VisualizationCallback registration), logger setup,
# and one full train+val epoch on mock data.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
# runpy re-executing an already-imported module is exactly what the CLI leg
# wants (the monkeypatches must keep holding); silence its advisory warning.
@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_fit_mock_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_backbone
) -> None:
    import fubio.train.train as train_mod

    # Keep everything local and off the network / GPU.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    shutil.copy(SHAPE_PRIOR_V4, tmp_path / "data" / "shape_prior_v4.json")

    # Wrap the Trainer so fit() cannot auto-select the GPU, and cap the val
    # loop (MockDataModule's n_val=50 is hardcoded inside fit()).
    real_trainer = train_mod.L.Trainer

    def cpu_trainer(**kwargs: object):
        kwargs["accelerator"] = "cpu"
        kwargs["devices"] = 1
        kwargs["limit_val_batches"] = 2
        kwargs["num_sanity_val_steps"] = 0
        kwargs["enable_progress_bar"] = False
        return real_trainer(**kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", cpu_trainer)

    # Stub-compatible config as YAML so fit() exercises the config-file branch;
    # the overrides dict then exercises _deep_update on top of it.
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "optimizer": {"warmup_steps": 1},
                "backbone": {"freeze_epochs": 0},
                "precision": "32-true",
                "loss": {"shape_prior_path": "data/shape_prior_v4.json"},
                "neck": {"mode": "c2f", "layer_indices": [11], "n_bottleneck": 1},
                "head": {"n_inst": 1, "derive_bbox": True, "coord": {"mode": "geo_simcc"}},
                "d_model": 256,
            }
        )
    )

    train_mod.fit(
        config_path=cfg_path,
        mock=True,
        overrides={"optimizer": {"max_epochs": 1}},
        limit_train_batches=2,
    )

    # FinalEpochCheckpoint wrote next to the ModelCheckpoint outputs.
    finals = list(tmp_path.rglob("final.ckpt"))
    assert len(finals) == 1, f"expected exactly one final.ckpt, found {finals}"
    # ModelCheckpoint itself saved the monitored-epoch checkpoint(s).
    assert any(p.name.startswith("fubio-") for p in finals[0].parent.iterdir())
    # CSVCallback produced one row for the single validation epoch.
    csvs = list(tmp_path.rglob("metrics.csv"))
    assert len(csvs) == 1
    lines = csvs[0].read_text().strip().splitlines()
    assert len(lines) == 2  # header + one epoch row
    assert lines[0].startswith("epoch,")
    # Everything stayed inside tmp_path (chdir guarded the repo's wandb_logs/).
    assert (tmp_path / "wandb_logs").exists()

    # --- CLI leg: run the __main__ argparse block via runpy. The lightning /
    # build_backbone monkeypatches still hold (same module objects), so this
    # stays on CPU with the stub backbone. --init-weights consumes the first
    # run's final.ckpt; --benchmark and --wandb-offline flip their branches.
    argv = [
        "train.py",
        "fit",
        "--config",
        str(cfg_path),
        "--mock",
        "--epochs",
        "1",
        "--limit-train-batches",
        "1",
        "--init-weights",
        str(finals[0]),
        "--wandb-offline",
        "--benchmark",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    import wandb

    # End the first leg's run — WandbLogger reuses any in-progress run, which
    # would silently merge the two legs into one run dir.
    wandb.finish()
    try:
        runpy.run_module("fubio.train.train", run_name="__main__")
    finally:
        # fit() mutates process-global torch/wandb state; put it back.
        wandb.finish()
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        torch.use_deterministic_algorithms(False)

    # The offline run got its own wandb run id, hence a second checkpoint dir.
    assert len(list(tmp_path.rglob("final.ckpt"))) == 2

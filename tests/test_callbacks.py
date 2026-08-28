"""Training callbacks: CSVCallback, FinalEpochCheckpoint, InstanceCollapseGuard.

All three are driven with hand-built stand-in trainers: the callbacks read
only a handful of trainer attributes (callback_metrics, log_dir, callbacks,
save_checkpoint), so a SimpleNamespace exercises the exact logic without a
fit loop. The real-Trainer wiring is covered by the CLI smoke in
tests/test_train_cli.py.

Upstream: train/callbacks.py.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from fubio.train.callbacks import (
    CSVCallback,
    FinalEpochCheckpoint,
    InstanceCollapseGuard,
)

# ---------------------------------------------------------------------------
# InstanceCollapseGuard
# ---------------------------------------------------------------------------


def _guard_trainer(metrics: dict[str, float], sanity: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        sanity_checking=sanity,
        callback_metrics={k: torch.tensor(v) for k, v in metrics.items()},
    )


class TestInstanceCollapseGuard:
    def test_sanity_check_is_skipped(self) -> None:
        guard = InstanceCollapseGuard(patience=1)
        trainer = _guard_trainer(
            {"diagnostic/instance_sep_p5": 0.0, "val/loss_heatmap": 100.0}, sanity=True
        )
        guard.on_validation_end(trainer, None)  # must not raise
        assert guard._strikes == 0

    def test_no_sep_metric_is_a_noop(self) -> None:
        """n_inst=1 runs never log instance_sep_p5 — the guard must stay silent."""
        guard = InstanceCollapseGuard(patience=1)
        guard.on_validation_end(_guard_trainer({"val/loss_heatmap": 100.0}), None)
        assert guard._strikes == 0

    def test_healthy_separation_resets_strikes(self) -> None:
        guard = InstanceCollapseGuard(patience=2)
        guard._strikes = 1
        guard.on_validation_end(
            _guard_trainer({"diagnostic/instance_sep_p5": 0.5}), None
        )
        assert guard._strikes == 0

    def test_collapse_with_failing_heatmap_aborts_after_patience(self) -> None:
        guard = InstanceCollapseGuard(patience=2)
        trainer = _guard_trainer(
            {"diagnostic/instance_sep_p5": 1e-6, "val/loss_heatmap": 20.0}
        )
        guard.on_validation_end(trainer, None)
        assert guard._strikes == 1
        with pytest.raises(RuntimeError, match="Instance-query collapse"):
            guard.on_validation_end(trainer, None)

    def test_collapse_with_healthy_heatmap_only_warns(self) -> None:
        """Second independent signal healthy → watch, don't abort."""
        guard = InstanceCollapseGuard(patience=1)
        trainer = _guard_trainer(
            {"diagnostic/instance_sep_p5": 1e-6, "val/loss_heatmap": 1.0}
        )
        guard.on_validation_end(trainer, None)
        assert guard._strikes == 0

    def test_mre_is_the_fallback_second_signal(self) -> None:
        """Without a heatmap loss the guard escalates on val_mre_overall."""
        guard = InstanceCollapseGuard(patience=1)
        trainer = _guard_trainer(
            {"diagnostic/instance_sep_p5": 1e-6, "val_mre_overall": 200.0}
        )
        with pytest.raises(RuntimeError, match="val_mre_overall"):
            guard.on_validation_end(trainer, None)

    def test_healthy_mre_fallback_does_not_strike(self) -> None:
        guard = InstanceCollapseGuard(patience=1)
        trainer = _guard_trainer(
            {"diagnostic/instance_sep_p5": 1e-6, "val_mre_overall": 10.0}
        )
        guard.on_validation_end(trainer, None)
        assert guard._strikes == 0

    def test_no_second_signal_never_escalates(self) -> None:
        guard = InstanceCollapseGuard(patience=1)
        trainer = _guard_trainer({"diagnostic/instance_sep_p5": 1e-6})
        guard.on_validation_end(trainer, None)
        assert guard._strikes == 0


# ---------------------------------------------------------------------------
# FinalEpochCheckpoint
# ---------------------------------------------------------------------------


class TestFinalEpochCheckpoint:
    def test_saves_final_ckpt_next_to_modelcheckpoint(self, tmp_path: Path) -> None:
        saved: list[Path] = []
        mc = ModelCheckpoint(dirpath=tmp_path / "ckpts")
        trainer = SimpleNamespace(
            callbacks=[mc],
            current_epoch=3,
            save_checkpoint=lambda p: saved.append(Path(p)),
        )
        FinalEpochCheckpoint().on_train_end(trainer, None)
        assert saved == [tmp_path / "ckpts" / "final.ckpt"]
        assert (tmp_path / "ckpts").is_dir()  # created if missing

    def test_no_modelcheckpoint_dirpath_skips(self, tmp_path: Path) -> None:
        saved: list[Path] = []
        trainer = SimpleNamespace(
            callbacks=[],
            current_epoch=0,
            save_checkpoint=lambda p: saved.append(Path(p)),
        )
        FinalEpochCheckpoint().on_train_end(trainer, None)
        assert saved == []


# ---------------------------------------------------------------------------
# CSVCallback
# ---------------------------------------------------------------------------


def _csv_trainer(
    log_dir: Path | None,
    metrics: dict[str, float],
    epoch: int = 0,
    run_id: str | None = None,
    sanity: bool = False,
) -> SimpleNamespace:
    experiment = SimpleNamespace(id=run_id) if run_id else None
    return SimpleNamespace(
        sanity_checking=sanity,
        log_dir=str(log_dir) if log_dir else None,
        default_root_dir=None,
        current_epoch=epoch,
        callback_metrics={k: torch.tensor(v) for k, v in metrics.items()},
        logger=SimpleNamespace(experiment=experiment),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


class TestCSVCallback:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        cb = CSVCallback()
        cb.on_validation_end(_csv_trainer(tmp_path, {"val/loss": 1.5}, epoch=0), None)
        cb.on_validation_end(_csv_trainer(tmp_path, {"val/loss": 1.0}, epoch=1), None)

        rows = _read_csv(tmp_path / "metrics.csv")
        assert [r["epoch"] for r in rows] == ["0", "1"]
        assert float(rows[1]["val/loss"]) == 1.0

    def test_run_id_nests_under_fubio_dir(self, tmp_path: Path) -> None:
        cb = CSVCallback()
        trainer = _csv_trainer(tmp_path, {"val/loss": 2.0}, run_id="abc123")
        cb.on_validation_end(trainer, None)
        assert (tmp_path / "fubio" / "abc123" / "metrics.csv").exists()

    def test_sanity_check_writes_nothing(self, tmp_path: Path) -> None:
        cb = CSVCallback()
        cb.on_validation_end(_csv_trainer(tmp_path, {"val/loss": 1.0}, sanity=True), None)
        assert not (tmp_path / "metrics.csv").exists()

    def test_no_log_dir_writes_nothing(self, tmp_path: Path) -> None:
        cb = CSVCallback()
        cb.on_validation_end(_csv_trainer(None, {"val/loss": 1.0}), None)
        assert list(tmp_path.iterdir()) == []

    def test_midrun_metric_widens_header_and_keeps_history(self, tmp_path: Path) -> None:
        """A key appearing mid-run rewrites the file under the widened header.

        Rows written before the new key appeared must survive, with the new
        column empty — a ramping loss term switching on must not erase the
        epochs logged before it.
        """
        cb = CSVCallback()
        cb.on_validation_end(_csv_trainer(tmp_path, {"val/loss": 1.0}, epoch=0), None)
        cb.on_validation_end(
            _csv_trainer(tmp_path, {"val/loss": 0.8, "val/new_term": 3.0}, epoch=1), None
        )

        rows = _read_csv(tmp_path / "metrics.csv")
        assert rows[0].keys() >= {"epoch", "val/loss", "val/new_term"}
        assert [r["epoch"] for r in rows] == ["0", "1"]
        assert rows[0]["val/new_term"] == ""  # widened column backfills empty
        assert float(rows[1]["val/new_term"]) == pytest.approx(3.0)

    def test_missing_key_fills_empty_not_misaligned(self, tmp_path: Path) -> None:
        """A metric that vanishes later leaves an empty cell, not a shifted row."""
        cb = CSVCallback()
        cb.on_validation_end(
            _csv_trainer(tmp_path, {"val/loss": 1.0, "val/extra": 5.0}, epoch=0), None
        )
        cb.on_validation_end(_csv_trainer(tmp_path, {"val/loss": 0.9}, epoch=1), None)

        rows = _read_csv(tmp_path / "metrics.csv")
        assert rows[1]["val/extra"] == ""
        assert float(rows[1]["val/loss"]) == pytest.approx(0.9)

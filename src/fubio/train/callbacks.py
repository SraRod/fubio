"""Training callbacks: collapse guard, final-epoch checkpoint, CSV logging.

Upstream: none (Lightning Callback protocol).
Downstream: train/train.py (added to Trainer callbacks).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CSV Callback
# ------------------------------------------------------------------


class InstanceCollapseGuard(L.Callback):
    """Abort when instance slots stop depending on their query AND localization fails.

    Slot collapse leaves every logged loss looking plausible while reducing
    n_inst>1 to an effectively single-query model. Five consecutive full-schedule
    runs reached confidences identical to five decimal places across all four
    slots, with centroid separation of 0.003-0.06 px on 800-1000 px images —
    against 10-115 px in healthy runs with no repulsion term at all. So slot
    convergence is permitted by the objective but is NOT the normal outcome.

    Two guards against false positives:

    * Triggers on the 5th percentile, not the minimum. `min` over the whole
      validation set is a double extreme — one anomalous image could trip it.
    * Requires a second, independent failure signal. Collapse is strongly
      correlated with failed optimization but is plausibly a symptom rather than
      the cause, so aborting on the separation statistic alone would destroy runs
      that might still be recoverable.
    """

    def __init__(
        self,
        collapse_below: float = 1e-3,
        heatmap_failed_above: float = 8.0,
        # patience >= freeze_epochs + 2: zero-init coord predictors produce
        # identical outputs for all slots until gradient differentiates the
        # weights, which needs backbone-unfrozen features to be meaningful.
        patience: int = 5,
    ) -> None:
        super().__init__()
        self.collapse_below = collapse_below
        self.heatmap_failed_above = heatmap_failed_above
        self.patience = patience
        self._strikes = 0

    # NOTE on the hook: `on_validation_end`, not `on_validation_epoch_end`.
    # Lightning dispatches callback hooks BEFORE the LightningModule's hook
    # (lightning/pytorch/loops/evaluation_loop.py::_on_evaluation_epoch_end), and
    # FUBioModule._log_epoch_metrics — which publishes val_mre_overall and the
    # other epoch metrics — runs in the module hook. A callback reading
    # trainer.callback_metrics from on_validation_epoch_end therefore sees the
    # PREVIOUS epoch's values. Verified on run j3xi8ip6: metrics.csv row N+1
    # carries the MRE of checkpoint epoch N, for all three saved checkpoints.
    # ModelCheckpoint is unaffected because it already uses on_validation_end.
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        sep = m.get("diagnostic/instance_sep_p5")
        if sep is None:
            return
        sep = float(sep)

        if sep >= self.collapse_below:
            self._strikes = 0
            return

        # Query-invariance confirmed; check a second independent signal before
        # escalating. Heatmap loss for heatmap mode, val_mre_overall otherwise.
        heat = m.get("val/loss_heatmap")
        mre = m.get("val_mre_overall")

        if heat is not None:
            signal_name = "heatmap loss"
            signal_val = float(heat)
            failing = signal_val > self.heatmap_failed_above
        elif mre is not None:
            signal_name = "val_mre_overall (px)"
            signal_val = float(mre)
            failing = signal_val > 50.0
        else:
            # No second signal available — don't escalate
            return

        if not failing:
            logger.warning(
                "Instance separation p5=%.2e is below %.2e — slots are nearly "
                "query-invariant — but %s=%.2f is still in the healthy "
                "range, so training continues. Watch this.",
                sep,
                self.collapse_below,
                signal_name,
                signal_val,
            )
            self._strikes = 0
            return

        self._strikes += 1
        logger.error(
            "Instance separation p5=%.2e with %s=%.2f (strike %d/%d).",
            sep,
            signal_name,
            signal_val,
            self._strikes,
            self.patience,
        )
        if self._strikes >= self.patience:
            raise RuntimeError(
                f"Instance-query collapse: separation p5={sep:.2e} with "
                f"{signal_name}={signal_val:.2f} for {self._strikes} consecutive "
                f"validation epochs. Predictions and confidences are nearly "
                f"query-invariant — aborting."
            )


class FinalEpochCheckpoint(L.Callback):
    """Save the final-epoch weights when training ends.

    ModelCheckpoint's ``save_last`` is tied to its top-k save flow, so
    ``last.ckpt`` freezes at the most recent epoch that entered top-k rather
    than tracking the final one. Verified on R35: it ran epochs 0-149 to
    completion with no early stopping, yet its ``last.ckpt`` holds epoch 138 —
    the last epoch that made top-3. Its final three epochs never entered top-k
    and were therefore never written to disk at all.

    That gap matters here because the schedule deliberately ends on a
    distribution the earlier epochs never saw: mosaic closes at
    ``mosaic.close_epoch`` and the cosine LR anneals to ``eta_min_ratio``, so
    the tail is trained clean and slow. Whether those weights serve better than
    best-val is an empirical question — but it cannot be asked if they are not
    saved.

    Writes ``final.ckpt`` next to the ModelCheckpoint outputs.
    """

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        dirpath = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath:
                dirpath = Path(cb.dirpath)
                break
        if dirpath is None:
            logger.warning("FinalEpochCheckpoint: no ModelCheckpoint dirpath found; skipping")
            return
        dirpath.mkdir(parents=True, exist_ok=True)
        path = dirpath / "final.ckpt"
        trainer.save_checkpoint(path)
        logger.info(
            "FinalEpochCheckpoint: saved epoch %d to %s", trainer.current_epoch, path
        )


class CSVCallback(L.Callback):
    """Append-mode CSV logger — one row per validation epoch.

    Writes ``metrics.csv`` in ``trainer.log_dir`` with columns
    ``epoch, <metric_1>, <metric_2>, ...``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._header_written = False
        self._csv_path: Path | None = None
        self._fieldnames: list[str] | None = None

    # NOTE on the hook: `on_validation_end`, not `on_validation_epoch_end`.
    # Lightning dispatches callback hooks BEFORE the LightningModule's hook
    # (lightning/pytorch/loops/evaluation_loop.py::_on_evaluation_epoch_end), and
    # FUBioModule._log_epoch_metrics — which publishes val_mre_overall and the
    # other epoch metrics — runs in the module hook. A callback reading
    # trainer.callback_metrics from on_validation_epoch_end therefore sees the
    # PREVIOUS epoch's values. Verified on run j3xi8ip6: metrics.csv row N+1
    # carries the MRE of checkpoint epoch N, for all three saved checkpoints.
    # ModelCheckpoint is unaffected because it already uses on_validation_end.
    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # Sanity check has incomplete metrics (val-only) — skip it
        if trainer.sanity_checking:
            return

        log_dir = getattr(trainer, "log_dir", None) or getattr(trainer, "default_root_dir", None)
        if log_dir is None:
            return

        # Per-run file. Writing one shared metrics.csv let every run append under
        # whichever run's header happened to be written first — once the loss
        # composition changed, the accumulated rows no longer matched their
        # column names and the file became unreadable.
        log_dir = Path(log_dir)
        run_id = getattr(getattr(trainer.logger, "experiment", None), "id", None)
        if run_id:
            log_dir = log_dir / "fubio" / str(run_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / "metrics.csv"

        metrics = {
            k: v.item() if isinstance(v, Tensor) else v for k, v in trainer.callback_metrics.items()
        }
        row = {"epoch": trainer.current_epoch, **metrics}

        # Metric keys can appear mid-run (a loss term switching on), so widen the
        # header rather than silently dropping columns or misaligning rows. The
        # accumulated rows are rewritten under the widened header, not discarded.
        if self._fieldnames is None:
            self._fieldnames = list(row)
        elif not set(row) <= set(self._fieldnames):
            self._fieldnames = self._fieldnames + [k for k in row if k not in self._fieldnames]
            previous_rows: list[dict] = []
            if self._csv_path.exists():
                with self._csv_path.open(newline="") as fh:
                    previous_rows = list(csv.DictReader(fh))
            with self._csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames, restval="")
                writer.writeheader()
                writer.writerows(previous_rows)
            self._header_written = True

        write_header = not self._header_written and not self._csv_path.exists()

        with self._csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames, restval="")
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)



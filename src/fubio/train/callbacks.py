"""Training callbacks: CSV logging and SWAD weight averaging.

Upstream: none (Lightning Callback protocol).
Downstream: train/train.py (added to Trainer callbacks).
"""

from __future__ import annotations

import csv
import logging
from collections import deque
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
        # header rather than silently dropping columns or misaligning rows.
        if self._fieldnames is None:
            self._fieldnames = list(row)
        elif not set(row) <= set(self._fieldnames):
            self._fieldnames = self._fieldnames + [k for k in row if k not in self._fieldnames]
            self._header_written = False
            self._csv_path.unlink(missing_ok=True)

        write_header = not self._header_written and not self._csv_path.exists()

        with self._csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames, restval="")
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ------------------------------------------------------------------
# SWAD — Stochastic Weight Averaging Densely (overfit-aware)
# ------------------------------------------------------------------


class SWADCallback(L.Callback):
    """Overfit-Aware Stochastic Weight Averaging Densely.

    Averages the weights seen between the point where the monitored metric stops
    improving (``ts``) and the point where it has clearly started overfitting
    (``te``), per Cha et al. 2021.

    ``start_after_epoch`` delays SWAD to avoid averaging across the
    Phase 0 → Phase 1 freeze/unfreeze transition.

    **The averaged weights are written to ``swad.ckpt`` in the checkpoint
    directory.** This is the only artifact that carries them: Lightning saves its
    own checkpoints *before* ``on_train_end`` runs, so mutating ``pl_module``
    there cannot influence any file Lightning writes. Every run before this fix
    reported "SWAD enabled" and shipped a plain ModelCheckpoint epoch file.

    Memory: snapshots are full CPU state_dicts (~1.5 GB each here), so only
    ``n_converge + 1`` are ever held, and they are released once averaging starts.
    Averaging the whole state_dict is safe in this model because every
    non-parameter tensor (``_frozen_backbone``, ``prior_*``, ``_canon_*``,
    ``_img_*``) is constant across epochs — averaging a constant is a no-op.
    """

    def __init__(
        self,
        n_converge: int = 3,
        n_tolerance: int = 6,
        tolerance_ratio: float = 0.3,
        start_after_epoch: int = 0,
        verbose: bool = True,
        monitor: str = "val_mre_overall",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.start_after_epoch = start_after_epoch
        self.verbose = verbose

        # +1 so the best snapshot is still resident when convergence trips at
        # exactly n_converge non-improving validations.
        self._recent: deque[_Snapshot] = deque(maxlen=n_converge + 1)
        self._tol_losses: deque[float] = deque(maxlen=n_tolerance)

        self._best_loss = float("inf")
        self._since_best = 0
        self._converged = False
        self._stopped = False
        self._threshold: float | None = None
        self._avg_model: _LightAveragedModel | None = None
        self._converge_step: int | None = None
        self._warned_missing = False

    # -- Lightning hooks --------------------------------------------------

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
        # Sanity check runs validation with an untrained model; its metric would
        # seed _best_loss with a meaningless value.
        if trainer.sanity_checking:
            return
        if trainer.current_epoch < self.start_after_epoch:
            return
        if self._stopped:
            return

        # Same rationale as ModelCheckpoint: val/loss changes composition between
        # experiments and does not track the scored quantity. SWAD's dense-window
        # selection must key on the served, domain-balanced metric.
        val_loss = trainer.callback_metrics.get(self.monitor)
        if val_loss is None:
            if self.verbose and not self._warned_missing:
                logger.warning("SWADCallback: %r metric not found.", self.monitor)
                self._warned_missing = True
            return

        loss = float(val_loss)
        snap = _Snapshot(
            state_dict={k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()},
            loss=loss,
            step=trainer.global_step,
        )

        if not self._converged:
            self._track_convergence(snap)
        else:
            self._extend_average(snap)

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if self._avg_model is None:
            logger.warning(
                "SWADCallback: no averaged model — the monitored metric never stopped "
                "improving for %d consecutive validations, so SWAD never started. "
                "No swad.ckpt written.",
                self.n_converge,
            )
            return

        path = self._resolve_output_path(trainer)
        pl_module.load_state_dict(self._avg_model.averaged_state_dict())

        # trainer.save_checkpoint (not torch.save) so the artifact carries
        # hyper_parameters and loads through FUBioModule.load_from_checkpoint
        # exactly like any epoch checkpoint — serving needs no special case.
        trainer.save_checkpoint(path)
        logger.info(
            "SWAD averaged %d snapshots (steps %d–%d) → %s",
            self._avg_model.n_averaged,
            self._avg_model.start_step,
            self._avg_model.end_step,
            path,
        )

    # -- Internal ---------------------------------------------------------

    def _resolve_output_path(self, trainer: L.Trainer) -> Path:
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        dirpath = getattr(ckpt_cb, "dirpath", None)
        if dirpath is None:
            base = getattr(trainer, "log_dir", None) or getattr(trainer, "default_root_dir", ".")
            dirpath = Path(base) / "checkpoints"
        out = Path(dirpath)
        out.mkdir(parents=True, exist_ok=True)
        return out / "swad.ckpt"

    def _track_convergence(self, snap: _Snapshot) -> None:
        """Start averaging once the metric has not improved for n_converge rounds.

        The previous implementation declared convergence when the oldest of three
        snapshots was the window minimum — i.e. after any two-epoch rise. On a
        noisy metric that fires almost immediately (R20: step 5390, epoch ~11 of
        90, at MRE ~66 against a final 27.5), which then froze the overfit
        threshold at 1.3 x 66 = 85.8 — a level a healthy run can never reach, so
        averaging never stopped either. Patience against the running *global*
        minimum is robust to that.
        """
        self._recent.append(snap)

        if snap.loss < self._best_loss:
            self._best_loss = snap.loss
            self._since_best = 0
            return

        self._since_best += 1
        if self._since_best < self.n_converge:
            return

        # The best snapshot is `_since_best` positions back — guaranteed resident
        # because _recent holds n_converge + 1 entries.
        window = list(self._recent)
        best_idx = min(range(len(window)), key=lambda i: window[i].loss)
        best = window[best_idx]

        self._converged = True
        self._converge_step = best.step
        self._threshold = self._best_loss * (1 + self.tolerance_ratio)

        self._avg_model = _LightAveragedModel(best.state_dict, best.step)
        # Dense average from ts: fold in everything observed after the best point.
        for later in window[best_idx + 1 :]:
            self._avg_model.update(later.state_dict, later.step)
            self._tol_losses.append(later.loss)

        # Snapshots are ~1.5 GB each; nothing reads them after seeding.
        self._recent.clear()

        if self.verbose:
            logger.info(
                "SWAD started at step %d (best %s=%.4f, no improvement for %d rounds); "
                "overfit threshold=%.4f.",
                self._converge_step,
                self.monitor,
                self._best_loss,
                self.n_converge,
                self._threshold,
            )

    def _extend_average(self, snap: _Snapshot) -> None:
        """Keep averaging until the metric stays above the threshold for n_tolerance rounds."""
        assert self._avg_model is not None
        assert self._threshold is not None

        self._avg_model.update(snap.state_dict, snap.step)
        self._tol_losses.append(snap.loss)

        if len(self._tol_losses) < self.n_tolerance:
            return
        if min(self._tol_losses) > self._threshold:
            self._stopped = True
            if self.verbose:
                logger.info(
                    "SWAD stopped at step %d — %s stayed above %.4f for %d rounds.",
                    snap.step,
                    self.monitor,
                    self._threshold,
                    self.n_tolerance,
                )


# ------------------------------------------------------------------
# Lightweight helpers (not exported)
# ------------------------------------------------------------------


class _Snapshot:
    """Cheap value object holding a CPU state_dict + metadata."""

    __slots__ = ("state_dict", "loss", "step")

    def __init__(
        self,
        state_dict: dict[str, Tensor],
        loss: float,
        step: int,
    ) -> None:
        self.state_dict = state_dict
        self.loss = loss
        self.step = step


class _LightAveragedModel:
    """Running mean of state_dicts without copying the full LightningModule.

    Avoids the O(n * model_size) memory cost of ``copy.deepcopy`` that the
    upstream SWAD implementation uses.
    """

    def __init__(self, initial_sd: dict[str, Tensor], step: int) -> None:
        self._avg = {k: v.clone().float() for k, v in initial_sd.items()}
        self.n_averaged = 1
        self.start_step = step
        self.end_step = step

    def update(self, sd: dict[str, Tensor], step: int) -> None:
        self.n_averaged += 1
        # Online mean: avg ← avg + (new - avg) / n
        for k, v in sd.items():
            self._avg[k] += (v.float() - self._avg[k]) / self.n_averaged
        self.end_step = step

    def averaged_state_dict(self) -> dict[str, Tensor]:
        return {k: v.clone() for k, v in self._avg.items()}

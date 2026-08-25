"""SWAD must average the right window, and must actually persist the result.

Three defects motivated these tests, all observed on run j3xi8ip6 (R20):

1. The averaged weights were never written to disk. `on_train_end` mutated
   `pl_module`, but Lightning writes its checkpoints *before* that hook, so every
   run that logged "SWAD enabled" shipped a plain ModelCheckpoint epoch file.
2. Convergence was declared whenever the oldest of three snapshots was the window
   minimum — any two-epoch rise. R20 tripped it at epoch ~11 of 90 with MRE ~66
   against a final 27.5.
3. The frozen overfit threshold (1.3 x 66 = 85.8) was then unreachable, so
   averaging never stopped: 80 snapshots spanning MRE 66 -> 27.5 were averaged
   uniformly.

The adversarial case is `test_does_not_start_while_still_improving`: a test that
only checks a converging curve cannot catch the regression it exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from fubio.train.callbacks import SWADCallback


class _TinyModule(nn.Module):
    """One scalar parameter, so the averaged value can be checked by hand."""

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([value]))


class _FakeTrainer(BaseModel):
    """Minimal stand-in for the Trainer surface SWADCallback touches."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    callback_metrics: dict[str, torch.Tensor] = {}
    global_step: int = 0
    current_epoch: int = 0
    sanity_checking: bool = False
    checkpoint_callback: object | None = None
    saved_to: Path | None = None
    save_calls: int = 0

    def save_checkpoint(self, path) -> None:  # noqa: ANN001 — mirrors Lightning's loose signature
        self.saved_to = Path(path)
        self.save_calls += 1
        Path(path).write_bytes(b"stub-checkpoint")


class _CkptCb(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dirpath: str


def _drive(
    losses: list[float],
    tmp_path: Path,
    *,
    n_converge: int = 3,
    n_tolerance: int = 6,
    tolerance_ratio: float = 0.3,
    start_after_epoch: int = 0,
) -> tuple[SWADCallback, _FakeTrainer, _TinyModule]:
    """Feed a loss curve through the callback; parameter value tracks the epoch."""
    cb = SWADCallback(
        n_converge=n_converge,
        n_tolerance=n_tolerance,
        tolerance_ratio=tolerance_ratio,
        start_after_epoch=start_after_epoch,
        verbose=False,
    )
    module = _TinyModule()
    trainer = _FakeTrainer(checkpoint_callback=_CkptCb(dirpath=str(tmp_path)))

    for epoch, loss in enumerate(losses):
        trainer.current_epoch = epoch
        trainer.global_step = epoch * 100
        # Parameter value == epoch, so the average of a window is its mean epoch.
        with torch.no_grad():
            module.w.fill_(float(epoch))
        trainer.callback_metrics = {"val_mre_overall": torch.tensor(loss)}
        cb.on_validation_end(trainer, module)

    return cb, trainer, module


class TestConvergenceDetection:
    def test_does_not_start_while_still_improving(self, tmp_path: Path) -> None:
        """ADVERSARIAL — the R20 pathology.

        A monotonically improving curve must never trigger SWAD, no matter how
        long it runs. The old rule fired on the first two-epoch rise; this curve
        has none, but the old rule also fired on any local non-improvement, which
        real noisy metrics produce constantly.
        """
        losses = [66.0 - 0.4 * i for i in range(90)]  # 66.0 -> 30.4, strictly down
        cb, trainer, _ = _drive(losses, tmp_path)

        assert cb._avg_model is None, "SWAD must not start while the metric is still improving"
        cb.on_train_end(trainer, _TinyModule())
        assert trainer.save_calls == 0, "nothing to write when SWAD never started"

    def test_noisy_but_improving_does_not_start(self, tmp_path: Path) -> None:
        """Two-epoch rises inside an improving trend must not trigger."""
        # Sawtooth: down 3, up 1 — contains many 2-epoch rises, trend is down.
        losses: list[float] = []
        v = 60.0
        for i in range(60):
            v += 1.0 if i % 4 == 3 else -3.0
            losses.append(v)
        assert losses[-1] < losses[0]

        cb, _, _ = _drive(losses, tmp_path, n_converge=3)
        assert cb._avg_model is None

    def test_two_epoch_rise_mid_descent_does_not_start(self, tmp_path: Path) -> None:
        """THE discriminating case — this is exactly what broke R20.

        The old rule declared convergence when the oldest of three snapshots was
        the window minimum, i.e. after two consecutive rises. The window
        (60, 61, 62) below satisfies that, so the old code would have started
        averaging at MRE 60 and frozen its overfit threshold at 1.3 x 61 = 79.3 —
        unreachable for a run that goes on to reach 30, so averaging would also
        never have stopped.

        A strictly monotonic curve does NOT discriminate the two implementations:
        neither fires on it. Only a local two-epoch rise inside a descending trend
        does, which is why this test exists separately from the monotonic one.
        """
        losses = [66.0, 64.0, 62.0, 60.0, 61.0, 62.0]  # <- old rule trips here
        losses += [55.0, 53.0, 48.0, 44.0, 39.0, 35.0, 32.0, 30.0]  # trend continues down

        # Precondition: the window really does contain the old trigger pattern.
        w = losses[3:6]
        assert w[0] == min(w), "test is vacuous unless the old trigger pattern is present"

        cb, _, _ = _drive(losses, tmp_path, n_converge=3)

        assert not cb._converged, (
            "SWAD started on a two-epoch rise while the metric went on to improve "
            "by 50% — this is the R20 defect"
        )
        assert cb._best_loss == pytest.approx(30.0)

    def test_starts_after_n_converge_non_improving_rounds(self, tmp_path: Path) -> None:
        # best at index 3 (value 10.0), then 3 non-improving rounds -> start
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, _, _ = _drive(losses, tmp_path, n_converge=3)

        assert cb._converged
        assert cb._best_loss == pytest.approx(10.0)
        assert cb._converge_step == 3 * 100, "averaging must be seeded at the best epoch"
        # threshold keyed on the best loss, not the window mean
        assert cb._threshold == pytest.approx(10.0 * 1.3)

    def test_threshold_uses_best_not_window_mean(self, tmp_path: Path) -> None:
        """R20 froze the threshold at 1.3 x mean(window) = 85.8 and never stopped."""
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, _, _ = _drive(losses, tmp_path)
        window_mean = (10.0 + 11.0 + 12.0) / 3
        assert cb._threshold != pytest.approx(window_mean * 1.3)


class TestAveragingWindow:
    def test_averages_from_best_epoch_onward(self, tmp_path: Path) -> None:
        """Parameter == epoch, so the averaged value is the mean of the epochs used."""
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, _, _ = _drive(losses, tmp_path, n_converge=3)

        assert cb._avg_model is not None
        avg = cb._avg_model.averaged_state_dict()["w"].item()
        # epochs 3,4,5,6 -> mean 4.5
        assert avg == pytest.approx(4.5)
        assert cb._avg_model.n_averaged == 4

    def test_does_not_fold_in_snapshots_older_than_the_start(self, tmp_path: Path) -> None:
        """The old code sliced a maxlen-6 queue with an index from a maxlen-3 queue.

        At convergence that index was 0, so it re-included snapshots from *before*
        the declared start point. Here epochs 0-2 must never enter the average.
        """
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, _, _ = _drive(losses, tmp_path, n_converge=3)

        assert cb._avg_model is not None
        avg = cb._avg_model.averaged_state_dict()["w"].item()
        assert avg >= 3.0, f"average {avg} includes pre-start epochs 0-2"

    def test_stops_after_n_tolerance_rounds_above_threshold(self, tmp_path: Path) -> None:
        # best 10.0 at idx 3; threshold = 13.0; then a sustained rise well above it
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0] + [50.0] * 10
        cb, _, _ = _drive(losses, tmp_path, n_converge=3, n_tolerance=6)

        assert cb._stopped, "averaging must stop once the metric sits above threshold"
        assert cb._avg_model is not None
        # Stops partway through the 50.0 tail rather than consuming all of it.
        assert cb._avg_model.n_averaged < len(losses)

    def test_keeps_averaging_while_metric_stays_near_best(self, tmp_path: Path) -> None:
        losses = [40.0, 30.0, 20.0, 10.0] + [10.5, 11.0, 10.8, 11.2, 10.9, 11.1] * 3
        cb, _, _ = _drive(losses, tmp_path, n_converge=3, n_tolerance=6)

        assert cb._converged
        assert not cb._stopped, "a flat valley below threshold must not stop averaging"


class TestPersistence:
    def test_writes_swad_checkpoint(self, tmp_path: Path) -> None:
        """A1: the whole point — the averaged weights must reach disk."""
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, trainer, module = _drive(losses, tmp_path, n_converge=3)

        cb.on_train_end(trainer, module)

        assert trainer.save_calls == 1
        assert trainer.saved_to == tmp_path / "swad.ckpt"
        assert (tmp_path / "swad.ckpt").exists()

    def test_averaged_weights_are_loaded_before_saving(self, tmp_path: Path) -> None:
        """The saved artifact must contain the average, not the final epoch."""
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, trainer, module = _drive(losses, tmp_path, n_converge=3)

        assert module.w.item() == pytest.approx(6.0)  # last epoch
        cb.on_train_end(trainer, module)
        assert module.w.item() == pytest.approx(4.5), "module must hold the average at save time"


class TestGuards:
    def test_sanity_check_is_ignored(self, tmp_path: Path) -> None:
        """Sanity-check validation runs an untrained model; it must not seed the best."""
        cb = SWADCallback(n_converge=3, verbose=False)
        module = _TinyModule()
        trainer = _FakeTrainer(checkpoint_callback=_CkptCb(dirpath=str(tmp_path)))
        trainer.sanity_checking = True
        trainer.callback_metrics = {"val_mre_overall": torch.tensor(999.0)}

        cb.on_validation_end(trainer, module)

        assert cb._best_loss == float("inf")

    def test_start_after_epoch_is_respected(self, tmp_path: Path) -> None:
        losses = [40.0, 30.0, 20.0, 10.0, 11.0, 12.0, 13.0]
        cb, _, _ = _drive(losses, tmp_path, n_converge=3, start_after_epoch=5)
        # Only epochs 5,6 seen (12.0, 13.0) — best 12.0, one non-improving round.
        assert not cb._converged
        assert cb._best_loss == pytest.approx(12.0)

    def test_missing_metric_does_not_crash(self, tmp_path: Path) -> None:
        cb = SWADCallback(verbose=True)
        module = _TinyModule()
        trainer = _FakeTrainer(checkpoint_callback=_CkptCb(dirpath=str(tmp_path)))
        trainer.callback_metrics = {"something/else": torch.tensor(1.0)}

        cb.on_validation_end(trainer, module)  # must not raise
        assert cb._avg_model is None


class TestHookChoice:
    """Callbacks must read metrics AFTER the module has published them.

    Lightning dispatches callback hooks before the LightningModule hook
    (evaluation_loop.py::_on_evaluation_epoch_end), and FUBioModule publishes
    val_mre_overall from its own on_validation_epoch_end. A callback reading
    callback_metrics from on_validation_epoch_end therefore sees the PREVIOUS
    epoch — confirmed on run j3xi8ip6, where metrics.csv row N+1 carries the MRE
    of checkpoint epoch N for all three saved checkpoints.
    """

    def test_all_metric_reading_callbacks_use_on_validation_end(self) -> None:
        import lightning as L

        from fubio.train.callbacks import CSVCallback, InstanceCollapseGuard

        for cls in (SWADCallback, CSVCallback, InstanceCollapseGuard):
            own = cls.on_validation_epoch_end
            assert own is L.Callback.on_validation_epoch_end, (
                f"{cls.__name__} overrides on_validation_epoch_end, which fires "
                f"before the module publishes epoch metrics"
            )
            assert cls.on_validation_end is not L.Callback.on_validation_end, (
                f"{cls.__name__} must implement on_validation_end"
            )

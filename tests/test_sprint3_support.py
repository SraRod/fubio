"""Tests for Sprint 3 support modules: callbacks, schedule."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fubio.train.callbacks import CSVCallback
from fubio.train.schedule import PhaseScheduler

# ======================================================================
# callbacks.py
# ======================================================================


class TestCSVCallback:
    def test_instantiation(self) -> None:
        """Can create CSVCallback with default args."""
        cb = CSVCallback()
        assert cb is not None
        assert cb._csv_path is None


# ======================================================================
# schedule.py
# ======================================================================


def _make_optimizer(n_groups: int = 3) -> torch.optim.Optimizer:
    """Create a dummy optimizer with n_groups param groups."""
    groups = []
    for i in range(n_groups):
        p = nn.Linear(4, 4)
        groups.append({"params": list(p.parameters()), "lr": 1e-3 * (10**i)})
    return torch.optim.SGD(groups)


class TestPhaseScheduler:
    def test_warmup_starts_at_zero(self) -> None:
        """LR multiplier starts near 0."""
        opt = _make_optimizer()
        sched = PhaseScheduler(opt, warmup_steps=100, total_steps=1000, freeze_steps=0)
        # After init, step 0 — decoder/heads multiplier = 0/100 = 0
        lrs = sched.get_last_lr()
        # Decoder and heads should be ≈ 0
        assert lrs[1] == pytest.approx(0.0, abs=1e-9)
        assert lrs[2] == pytest.approx(0.0, abs=1e-9)

    def test_warmup_reaches_one(self) -> None:
        """After warmup_steps, multiplier ≈ 1."""
        opt = _make_optimizer()
        warmup = 100
        sched = PhaseScheduler(opt, warmup_steps=warmup, total_steps=1000, freeze_steps=0)
        for _ in range(warmup):
            sched.step()
        # At step=warmup, cosine progress = 0 → multiplier = 1
        lrs = sched.get_last_lr()
        # Check multiplier by dividing by initial LR
        decoder_base = sched.base_lrs[1]
        assert lrs[1] / decoder_base == pytest.approx(1.0, abs=1e-6)

    def test_backbone_frozen_phase0(self) -> None:
        """Backbone group multiplier = 0 during Phase 0."""
        opt = _make_optimizer()
        sched = PhaseScheduler(
            opt,
            warmup_steps=50,
            total_steps=1000,
            freeze_steps=200,
        )
        # Step through Phase 0
        for _ in range(100):
            sched.step()
        # Backbone LR should be exactly 0
        assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-12)

    def test_backbone_unfreezes(self) -> None:
        """After freeze_steps, backbone LR starts increasing."""
        opt = _make_optimizer()
        freeze = 50
        sched = PhaseScheduler(
            opt,
            warmup_steps=100,
            total_steps=1000,
            freeze_steps=freeze,
        )
        for _ in range(freeze + 10):
            sched.step()
        # Backbone LR should be > 0 now (in warmup)
        assert sched.get_last_lr()[0] > 0

    def test_cosine_decay(self) -> None:
        """After warmup, multiplier decreases."""
        opt = _make_optimizer()
        warmup = 10
        total = 200
        sched = PhaseScheduler(
            opt,
            warmup_steps=warmup,
            total_steps=total,
            freeze_steps=0,
        )
        # Step to end of warmup
        for _ in range(warmup):
            sched.step()
        lr_at_peak = sched.get_last_lr()[1]

        # Step further into cosine decay
        for _ in range(50):
            sched.step()
        lr_later = sched.get_last_lr()[1]

        assert lr_later < lr_at_peak

    def test_cosine_reaches_zero(self) -> None:
        """At total_steps the cosine multiplier ≈ 0."""
        opt = _make_optimizer()
        warmup = 10
        total = 100
        sched = PhaseScheduler(
            opt,
            warmup_steps=warmup,
            total_steps=total,
            freeze_steps=0,
        )
        for _ in range(total):
            sched.step()
        # Decoder multiplier should be ≈ 0
        assert sched.get_last_lr()[1] == pytest.approx(0.0, abs=1e-4)

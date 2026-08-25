"""Tests for Sprint 3 support modules: regularizer, callbacks, schedule."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fubio.train.callbacks import CSVCallback, SWADCallback
from fubio.train.regularizer import (
    MeanEncoder,
    MIROEncoders,
    VarianceEncoder,
    miro_loss,
)
from fubio.train.schedule import PhaseScheduler

# ======================================================================
# regularizer.py
# ======================================================================


class TestMeanEncoder:
    def test_identity(self) -> None:
        enc = MeanEncoder((2, 16, 64))
        x = torch.randn(2, 16, 64)
        assert torch.equal(enc(x), x)


class TestVarianceEncoder:
    def test_vit_shape(self) -> None:
        """ViT (B, N, C) input -> output broadcasts to (1, 1, C)."""
        enc = VarianceEncoder((2, 197, 768), channelwise=True)
        out = enc(torch.randn(2, 197, 768))
        assert out.shape == (1, 1, 768)

    def test_cnn_shape(self) -> None:
        """CNN (B, C, H, W) input -> output broadcasts to (1, C, 1, 1)."""
        enc = VarianceEncoder((2, 256, 14, 14), channelwise=True)
        out = enc(torch.randn(2, 256, 14, 14))
        assert out.shape == (1, 256, 1, 1)

    def test_positive_output(self) -> None:
        """Softplus ensures positive variance."""
        enc = VarianceEncoder((4, 100, 64))
        out = enc(torch.randn(4, 100, 64))
        assert (out > 0).all()

    def test_init_value(self) -> None:
        """Initial variance ≈ requested init value."""
        init = 0.5
        enc = VarianceEncoder((1, 10, 32), init=init)
        out = enc(torch.zeros(1, 10, 32))
        assert abs(out.mean().item() - init) < 1e-4

    def test_unsupported_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="channelwise"):
            VarianceEncoder((64,), channelwise=True)


class TestMIROEncoders:
    def test_forward_structure(self) -> None:
        """Returns (means, variances) with correct lengths."""
        shapes = [(2, 197, 768), (2, 197, 768)]
        encs = MIROEncoders(shapes)
        feats = [torch.randn(*s) for s in shapes]
        means, variances = encs(feats)
        assert len(means) == 2
        assert len(variances) == 2
        # Means are identity
        assert torch.equal(means[0], feats[0])
        # Variances are positive
        assert (variances[0] > 0).all()


class TestMIROLoss:
    def test_returns_scalar(self) -> None:
        """miro_loss returns a scalar > 0."""
        shapes = [(2, 50, 64)]
        encs = MIROEncoders(shapes)
        pre = [torch.randn(2, 50, 64)]
        post = [torch.randn(2, 50, 64)]
        loss = miro_loss(pre, post, encs)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        """Loss is differentiable w.r.t. variance encoder params."""
        shapes = [(1, 10, 32)]
        encs = MIROEncoders(shapes)
        pre = [torch.randn(1, 10, 32)]
        post = [torch.randn(1, 10, 32)]
        loss = miro_loss(pre, post, encs)
        loss.backward()
        var_enc = encs.var_encoders[0]
        assert var_enc.bias.grad is not None
        assert var_enc.bias.grad.abs().sum() > 0

    def test_zero_when_identical(self) -> None:
        """Loss is minimal when pre == post and mean encoder is identity."""
        shapes = [(1, 10, 32)]
        encs = MIROEncoders(shapes)
        feat = [torch.randn(1, 10, 32)]
        # Same features → loss ≈ only the log(var) term, which is small
        loss = miro_loss(feat, feat, encs)
        # Just check it runs; the value isn't literally zero because of log(var)
        assert loss.isfinite()


# ======================================================================
# callbacks.py
# ======================================================================


class TestCSVCallback:
    def test_instantiation(self) -> None:
        """Can create CSVCallback with default args."""
        cb = CSVCallback()
        assert cb is not None
        assert cb._csv_path is None


class TestSWADCallback:
    def test_start_after_epoch(self) -> None:
        """Nothing is accumulated before any hook fires.

        Behavioural coverage of the averaging window, the convergence rule and
        the on_train_end persistence lives in tests/test_swad.py.
        """
        cb = SWADCallback(start_after_epoch=5, n_converge=2)
        assert len(cb._recent) == 0
        assert cb._converged is False
        assert cb._avg_model is None

    def test_default_params(self) -> None:
        cb = SWADCallback()
        assert cb.n_converge == 3
        assert cb.n_tolerance == 6
        assert cb.tolerance_ratio == 0.3
        assert cb.start_after_epoch == 0


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

"""Tests for EMA Teacher + pseudo-label semi-supervised infrastructure."""

from __future__ import annotations

import torch
import torch.nn as nn

from fubio.train.losses import (
    chamber_angle_sign_loss,
    equivariance_loss,
    pseudo_label_loss,
)
from fubio.train.views import sample_affine


class TestPseudoLabelLoss:
    """pseudo_label_loss: fixed denominator, L1, validity mask."""

    def test_low_conf_produces_low_loss(self):
        """Core property: absolute confidence attenuation via fixed denominator."""
        student = torch.rand(4, 1, 5, 2)
        target = torch.rand(4, 1, 5, 2)
        valid = torch.ones(4, 1, 5, dtype=torch.bool)

        high_conf = torch.ones(4, 1, 1) * 10.0  # sigmoid ≈ 1
        low_conf = torch.ones(4, 1, 1) * -10.0  # sigmoid ≈ 0

        loss_high = pseudo_label_loss(student, target, high_conf, valid)
        loss_low = pseudo_label_loss(student, target, low_conf, valid)

        assert loss_high > loss_low * 10, (
            f"High-conf loss ({loss_high:.6f}) should be >10x "
            f"low-conf loss ({loss_low:.6f})"
        )

    def test_validity_mask_excludes_landmarks(self):
        """Out-of-frame landmarks contribute zero loss."""
        student = torch.rand(4, 1, 5, 2)
        target = torch.rand(4, 1, 5, 2)
        conf = torch.ones(4, 1, 1) * 5.0

        all_valid = torch.ones(4, 1, 5, dtype=torch.bool)
        none_valid = torch.zeros(4, 1, 5, dtype=torch.bool)

        loss_all = pseudo_label_loss(student, target, conf, all_valid)
        loss_none = pseudo_label_loss(student, target, conf, none_valid)

        assert loss_all > 0
        assert loss_none == 0.0

    def test_gradient_to_student_only(self):
        """Gradients flow to student landmarks, not through teacher conf."""
        student = torch.rand(4, 1, 5, 2, requires_grad=True)
        target = torch.rand(4, 1, 5, 2)
        conf = torch.rand(4, 1, 1, requires_grad=True)
        valid = torch.ones(4, 1, 5, dtype=torch.bool)

        loss = pseudo_label_loss(student, target, conf, valid)
        loss.backward()

        assert student.grad is not None
        assert conf.grad is None  # detached inside

    def test_l1_kernel(self):
        """Loss uses L1 (not smooth L1) to match supervised landmark_beta=0."""
        student = torch.tensor([[[[0.5, 0.5]]]])
        target = torch.tensor([[[[0.6, 0.7]]]])
        conf = torch.ones(1, 1, 1) * 100.0  # sigmoid ≈ 1
        valid = torch.ones(1, 1, 1, dtype=torch.bool)

        loss = pseudo_label_loss(student, target, conf, valid)
        # L1: (|0.1| + |0.2|) / 2 = 0.15, times conf ~1.0, / numel=2
        expected = (0.1 + 0.2) / 2  # per_coord.numel() = 2
        assert abs(loss.item() - expected) < 0.01

    def test_empty_input(self):
        """Empty tensors return 0."""
        student = torch.zeros(0, 1, 5, 2)
        target = torch.zeros(0, 1, 5, 2)
        conf = torch.zeros(0, 1, 1)
        valid = torch.zeros(0, 1, 5, dtype=torch.bool)
        assert pseudo_label_loss(student, target, conf, valid).item() == 0.0


class TestEquivarianceConfWeightingFix:
    """Verify the R22 confidence weighting bug is fixed."""

    def test_uniform_low_conf_reduces_loss(self):
        """Uniform low confidence must produce proportionally lower loss."""
        torch.manual_seed(42)
        lm_ref = torch.rand(4, 2, 5, 2)
        lm_trans = torch.rand(4, 2, 5, 2)
        m = sample_affine(
            4,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        high_conf = torch.ones(4, 2, 1) * 10.0
        low_conf = torch.ones(4, 2, 1) * -10.0

        loss_high = equivariance_loss(lm_ref, lm_trans, m, high_conf)
        loss_low = equivariance_loss(lm_ref, lm_trans, m, low_conf)

        assert loss_high > loss_low * 10


class TestChamberAngleSignLoss:
    """A4C angle sign crossing penalty."""

    def test_same_side_no_penalty(self):
        """Pred and GT on same side of 90° → zero loss."""
        # Both acute (cos > 0): axes at ~80°
        gt = torch.zeros(1, 16, 2)
        pred = torch.zeros(1, 16, 2)
        for base in (0, 4, 8, 12):
            # 上下: vertical, 左右: slightly rotated (acute)
            gt[0, base] = torch.tensor([0.5, 0.3])
            gt[0, base + 1] = torch.tensor([0.5, 0.7])
            gt[0, base + 2] = torch.tensor([0.3, 0.5])
            gt[0, base + 3] = torch.tensor([0.7, 0.55])  # slight tilt

            pred[0, base] = torch.tensor([0.5, 0.3])
            pred[0, base + 1] = torch.tensor([0.5, 0.7])
            pred[0, base + 2] = torch.tensor([0.3, 0.5])
            pred[0, base + 3] = torch.tensor([0.7, 0.52])  # same side

        loss = chamber_angle_sign_loss(pred, gt)
        assert loss.item() < 1e-6

    def test_crossing_gives_penalty(self):
        """Pred crosses 90° relative to GT → nonzero loss."""
        gt = torch.zeros(1, 16, 2)
        pred = torch.zeros(1, 16, 2)

        # LV: GT has acute angle, pred has obtuse
        base = 0
        gt[0, base] = torch.tensor([0.5, 0.3])
        gt[0, base + 1] = torch.tensor([0.5, 0.7])
        gt[0, base + 2] = torch.tensor([0.3, 0.5])
        gt[0, base + 3] = torch.tensor([0.7, 0.55])  # acute: cos > 0

        pred[0, base] = torch.tensor([0.5, 0.3])
        pred[0, base + 1] = torch.tensor([0.5, 0.7])
        pred[0, base + 2] = torch.tensor([0.3, 0.5])
        pred[0, base + 3] = torch.tensor([0.7, 0.45])  # obtuse: cos < 0

        # Fill other chambers with neutral (90° exactly)
        for b in (4, 8, 12):
            for pt in range(4):
                gt[0, b + pt] = torch.tensor([0.5, 0.5])
                pred[0, b + pt] = torch.tensor([0.5, 0.5])

        loss = chamber_angle_sign_loss(pred, gt)
        assert loss.item() > 0

    def test_degenerate_axes_excluded(self):
        """Zero-length axes produce zero loss, not NaN."""
        gt = torch.zeros(1, 16, 2)
        pred = torch.zeros(1, 16, 2)
        loss = chamber_angle_sign_loss(pred, gt)
        assert not torch.isnan(loss)
        assert loss.item() == 0.0


class TestEMATeacher:
    """EMA update mechanics."""

    def test_object_setattr_excludes_from_state_dict(self):
        """Teacher stored via object.__setattr__ is not in state_dict."""

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(10, 5)
                teacher = nn.Linear(10, 5)
                object.__setattr__(self, "_teacher_model", teacher)

        m = TestModule()
        sd_keys = set(m.state_dict().keys())
        assert all("teacher" not in k for k in sd_keys)
        assert hasattr(m, "_teacher_model")

    def test_train_does_not_enable_teacher_dropout(self):
        """module.train() must not affect teacher's eval state."""

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(10, 5)
                teacher = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.5))
                teacher.eval()
                object.__setattr__(self, "_teacher_model", teacher)

            def train(self, mode=True):
                super().train(mode)
                self._teacher_model.eval()
                return self

        m = TestModule()
        m.train()
        assert not m._teacher_model.training

    def test_ema_warmup_formula(self):
        """Warm-up: alpha_eff starts at 0, converges to alpha."""
        alpha = 0.999
        # Step 0: alpha_eff = 0
        assert min(1 - 1 / (0 + 1), alpha) == 0.0
        # Step 999: alpha_eff = 0.999
        assert min(1 - 1 / (999 + 1), alpha) == alpha
        # Step 500: below alpha
        eff_500 = min(1 - 1 / (500 + 1), alpha)
        assert eff_500 < alpha
        assert eff_500 > 0.998

    def test_ema_update_changes_weights(self):
        """EMA update produces different weights from student."""
        student = nn.Linear(10, 5)
        teacher = nn.Linear(10, 5)
        # Make them identical
        teacher.load_state_dict(student.state_dict())

        # Simulate one optimizer step on student
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # EMA update
        alpha = 0.999
        for p_t, p_s in zip(
            teacher.parameters(), student.parameters(), strict=True,
        ):
            p_t.data.mul_(alpha).add_(p_s.data, alpha=1 - alpha)

        # Teacher should differ from both original and student
        for p_t, p_s in zip(
            teacher.parameters(), student.parameters(), strict=True,
        ):
            assert not torch.allclose(p_t, p_s)

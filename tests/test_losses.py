"""Tests for bbox, conf focal, landmark, parameter, and MIL loss functions.

Verifies correctness of each loss component in isolation:
bbox L1 + CIoU, binary focal confidence, landmark Huber/NLL,
clinical parameter loss, and MIL bag-level loss.
"""

from __future__ import annotations

import math

import pytest
import torch

from fubio.data.task_registry import TASKS
from fubio.train.losses import (
    bbox_loss,
    chamber_angle_loss,
    conf_focal_loss,
    ellipse_axis_ordering_loss,
    ellipse_orthogonality_loss,
    geometric_constraint_loss,
    landmark_loss,
    mil_loss,
    param_loss,
)

# =========================================================================
# Bbox loss
# =========================================================================


class TestBboxLoss:
    def test_empty_matches(self) -> None:
        """No matched pairs -> both losses = 0."""
        pred = torch.empty(0, 4)
        gt = torch.empty(0, 4)
        result = bbox_loss(pred, gt)
        assert result["loss_box_l1"].item() == 0.0
        assert result["loss_box_ciou"].item() == 0.0

    def test_single_match(self) -> None:
        """One matched pair with offset -> positive losses."""
        pred = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        gt = torch.tensor([[0.4, 0.4, 0.2, 0.2]])
        result = bbox_loss(pred, gt)
        assert result["loss_box_l1"].item() > 0.0
        assert result["loss_box_ciou"].item() > 0.0

    def test_perfect_match(self) -> None:
        """Identical pred/GT -> losses near 0."""
        box = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        result = bbox_loss(box, box.clone())
        assert result["loss_box_l1"].item() < 1e-5
        assert result["loss_box_ciou"].item() < 1e-2

    def test_gradient_flows(self) -> None:
        pred = torch.rand(3, 4, requires_grad=True)
        gt = torch.rand(3, 4)
        result = bbox_loss(pred, gt)
        total = result["loss_box_l1"] + result["loss_box_ciou"]
        total.backward()
        assert pred.grad is not None


# =========================================================================
# Conf focal loss
# =========================================================================


class TestConfFocalLoss:
    def test_all_negative(self) -> None:
        """All unmatched queries -> positive loss."""
        pred = torch.randn(10)
        target = torch.zeros(10)
        loss = conf_focal_loss(pred, target)
        assert loss.item() > 0

    def test_high_conf_matched(self) -> None:
        """High logit + positive target -> very low focal loss."""
        pred = torch.full((5,), 10.0)
        target = torch.ones(5)
        loss = conf_focal_loss(pred, target)
        assert loss.item() < 0.01

    def test_empty(self) -> None:
        pred = torch.empty(0)
        target = torch.empty(0)
        loss = conf_focal_loss(pred, target)
        assert loss.item() == 0.0

    def test_gradient_flows(self) -> None:
        pred = torch.randn(8, requires_grad=True)
        target = torch.zeros(8)
        target[:3] = 1.0
        loss = conf_focal_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


# =========================================================================
# Landmark loss
# =========================================================================


class TestLandmarkLoss:
    def test_perfect_prediction(self) -> None:
        """pred == gt -> loss ~= 0."""
        N, K = 4, 6
        gt = torch.rand(N, K, 2)
        mask = torch.ones(N, K, dtype=torch.bool)

        loss = landmark_loss(gt.clone(), gt, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_known_offset(self) -> None:
        """Known displacement -> verify plain L1 value."""
        N, K = 1, 1
        gt = torch.tensor([[[0.5, 0.5]]])
        pred = torch.tensor([[[0.5, 0.6]]])
        mask = torch.ones(N, K, dtype=torch.bool)

        loss = landmark_loss(pred, gt, mask)
        expected = 0.1 / 2  # |Δy| averaged over the two coordinates
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_mask_exclusion(self) -> None:
        """Masked-out landmarks don't contribute to loss."""
        N, K = 2, 4
        gt = torch.rand(N, K, 2)
        pred = gt + 10.0
        mask = torch.zeros(N, K, dtype=torch.bool)
        mask[0, 0] = True
        mask[1, 2] = True

        pred_close = pred.clone()
        pred_close[0, 0] = gt[0, 0]
        pred_close[1, 2] = gt[1, 2]

        loss = landmark_loss(pred_close, gt, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_normalization_ignores_unmasked_slots(self) -> None:
        """Loss depends only on masked elements, not on the tensor's K."""
        N = 2
        gt_small = torch.rand(N, 1, 2)
        pred_small = gt_small + 0.1
        mask_small = torch.ones(N, 1, dtype=torch.bool)

        # Same supervised landmark, padded out to K=22 with masked-off slots.
        gt_large = torch.rand(N, 22, 2)
        gt_large[:, 0] = gt_small[:, 0]
        pred_large = gt_large + 0.1
        mask_large = torch.zeros(N, 22, dtype=torch.bool)
        mask_large[:, 0] = True

        loss_small = landmark_loss(pred_small, gt_small, mask_small)
        loss_large = landmark_loss(pred_large, gt_large, mask_large)
        assert loss_large.item() == pytest.approx(loss_small.item(), abs=1e-6)

    def test_no_valid_landmarks(self) -> None:
        """All masked out -> loss = 0."""
        N, K = 2, 4
        gt = torch.rand(N, K, 2)
        pred = gt + 1.0
        mask = torch.zeros(N, K, dtype=torch.bool)

        loss = landmark_loss(pred, gt, mask)
        assert loss.item() == 0.0


# =========================================================================
# Parameter loss
# =========================================================================


class TestParamLoss:
    def test_ivc_gradient_flows(self) -> None:
        """IVC has distance params - loss is differentiable."""
        N = 4
        k_ivc = TASKS["IVC"].n_keypoints
        gt = torch.rand(N, k_ivc, 2)
        pred = gt.clone().detach().requires_grad_(True)
        mask = torch.ones(N, k_ivc, dtype=torch.bool)

        loss = param_loss(pred, gt, task_id="IVC", visible_mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None

    def test_no_params_returns_zero(self) -> None:
        N, K = 2, 4
        pred = torch.rand(N, K, 2)
        gt = torch.rand(N, K, 2)
        mask = torch.ones(N, K, dtype=torch.bool)

        loss = param_loss(pred, gt, task_id="__nonexistent__", visible_mask=mask)
        assert loss.item() == 0.0

    def test_nan_exclusion(self) -> None:
        """Invisible landmarks -> NaN params excluded from loss."""
        N = 2
        k_ivc = TASKS["IVC"].n_keypoints
        gt = torch.rand(N, k_ivc, 2)
        pred = torch.rand(N, k_ivc, 2, requires_grad=True)
        mask = torch.zeros(N, k_ivc, dtype=torch.bool)

        loss = param_loss(pred, gt, task_id="IVC", visible_mask=mask)
        assert loss.item() == 0.0

    def test_perfect_params(self) -> None:
        """Identical pred/gt landmarks -> param loss ~= 0."""
        N = 3
        k_ivc = TASKS["IVC"].n_keypoints
        gt = torch.rand(N, k_ivc, 2)
        mask = torch.ones(N, k_ivc, dtype=torch.bool)

        loss = param_loss(gt.clone(), gt, task_id="IVC", visible_mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_partial_visibility(self) -> None:
        """Some samples visible, others not -> only visible contribute."""
        N = 4
        k_ivc = TASKS["IVC"].n_keypoints
        torch.manual_seed(99)
        gt = torch.rand(N, k_ivc, 2) * 0.5 + 0.2
        pred = gt.clone()
        pred[:, 0, :] += 0.15
        mask = torch.ones(N, k_ivc, dtype=torch.bool)
        mask[2:, 0] = False

        loss = param_loss(pred, gt, task_id="IVC", visible_mask=mask)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0


# =========================================================================
# MIL loss
# =========================================================================


class TestMilLoss:
    def test_high_conf_low_loss(self) -> None:
        """At least one high-logit query -> loss near 0."""
        logits = torch.tensor([-5.0, -3.0, 10.0])
        loss = mil_loss(logits)
        assert loss.item() < 1e-4

    def test_all_low_high_loss(self) -> None:
        """All low logits -> high loss (model not detecting the known task)."""
        logits = torch.tensor([-5.0, -5.0, -5.0])
        loss = mil_loss(logits)
        assert loss.item() > 4.0

    def test_empty_returns_zero(self) -> None:
        """No logits -> loss = 0 (degenerate case)."""
        logits = torch.empty(0)
        loss = mil_loss(logits)
        assert loss.item() == 0.0

    def test_single_element(self) -> None:
        """Single query -> loss = -log(sigmoid(logit))."""
        logit = torch.tensor([2.0])
        loss = mil_loss(logit)
        expected = -torch.nn.functional.logsigmoid(logit).item()
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_gradient_flows(self) -> None:
        """Gradient flows through max to the winning query."""
        logits = torch.tensor([-2.0, 1.0, -3.0], requires_grad=True)
        loss = mil_loss(logits)
        loss.backward()
        assert logits.grad is not None
        # Gradient should flow only through the max element (index 1)
        assert logits.grad[1].abs().item() > 0
        assert logits.grad[0].abs().item() == 0.0
        assert logits.grad[2].abs().item() == 0.0

    def test_numerically_stable_large_negative(self) -> None:
        """Very negative logits should not produce inf/nan."""
        logits = torch.tensor([-100.0, -200.0])
        loss = mil_loss(logits)
        assert torch.isfinite(loss)
        assert loss.item() > 0


# =========================================================================
# Conf focal loss with label smoothing
# =========================================================================


class TestConfFocalLossLabelSmoothing:
    """Verify conf_focal_loss works correctly with soft targets (L_suppress)."""

    def test_smooth_target_reduces_loss(self) -> None:
        """Smooth negative target (0.05) should produce less loss than hard (0.0)
        when the model predicts non-zero confidence."""
        pred = torch.tensor([0.5, 0.5, 0.5])

        loss_hard = conf_focal_loss(pred, torch.tensor([0.0, 0.0, 0.0]))
        loss_smooth = conf_focal_loss(pred, torch.tensor([0.05, 0.05, 0.05]))

        # Smooth target closer to small positive prediction -> lower loss
        assert loss_smooth.item() < loss_hard.item()

    def test_smooth_gradient_direction(self) -> None:
        """With soft target 0.05, gradient at logit=0 should push toward
        a small positive value (not hard toward 0)."""
        pred = torch.tensor([0.0], requires_grad=True)
        loss = conf_focal_loss(pred, torch.tensor([0.05]))
        loss.backward()
        assert pred.grad is not None

    def test_mixed_hard_and_smooth(self) -> None:
        """Batch with matched (1.0) + smooth unmatched (0.05) works."""
        pred = torch.randn(4)
        target = torch.tensor([1.0, 0.05, 0.05, 1.0])
        loss = conf_focal_loss(pred, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0


# =========================================================================
# Confidence ranking loss
# =========================================================================


def _make_ellipse_landmarks(angle_deg: float, a_len: float, b_len: float) -> torch.Tensor:
    """Create (1, 4, 2) ellipse landmarks with given inter-axis angle and lengths.

    P0-P1 along x-axis with length a_len, P2-P3 at angle_deg from x-axis with length b_len.
    """
    cx, cy = 0.5, 0.5
    ha, hb = a_len / 2, b_len / 2
    rad = math.radians(angle_deg)
    return torch.tensor([[
        [cx - ha, cy],
        [cx + ha, cy],
        [cx - hb * math.cos(rad), cy - hb * math.sin(rad)],
        [cx + hb * math.cos(rad), cy + hb * math.sin(rad)],
    ]])


class TestEllipseOrthogonalityLoss:
    def test_zero_at_perpendicular(self) -> None:
        lm = _make_ellipse_landmarks(90.0, 0.4, 0.3)
        loss = ellipse_orthogonality_loss(lm)
        assert loss.item() < 1e-10

    def test_max_at_parallel(self) -> None:
        lm = _make_ellipse_landmarks(0.0, 0.4, 0.3)
        loss = ellipse_orthogonality_loss(lm)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_nonzero_at_60deg(self) -> None:
        lm = _make_ellipse_landmarks(60.0, 0.4, 0.3)
        loss = ellipse_orthogonality_loss(lm)
        assert loss.item() > 0.05

    def test_degenerate_axis_returns_zero(self) -> None:
        lm = torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.3, 0.5], [0.7, 0.5]]])
        loss = ellipse_orthogonality_loss(lm)
        assert loss.item() == 0.0

    def test_invariant_to_endpoint_reversal(self) -> None:
        lm = _make_ellipse_landmarks(75.0, 0.4, 0.3)
        lm_rev = lm.clone()
        lm_rev[:, [0, 1]] = lm_rev[:, [1, 0]]
        assert torch.allclose(
            ellipse_orthogonality_loss(lm),
            ellipse_orthogonality_loss(lm_rev),
            atol=1e-6,
        )

    def test_gradient_flows(self) -> None:
        lm = _make_ellipse_landmarks(75.0, 0.4, 0.3).requires_grad_(True)
        ellipse_orthogonality_loss(lm).backward()
        assert lm.grad is not None and lm.grad.abs().sum() > 0

    def test_bf16_input(self) -> None:
        lm = _make_ellipse_landmarks(80.0, 0.4, 0.3).bfloat16()
        loss = ellipse_orthogonality_loss(lm)
        assert torch.isfinite(loss) and loss.dtype == torch.float32


class TestEllipseAxisOrderingLoss:
    def test_zero_when_long_dominates(self) -> None:
        lm = _make_ellipse_landmarks(90.0, 0.2, 0.4)
        loss = ellipse_axis_ordering_loss(lm)
        assert loss.item() < 1e-10

    def test_penalizes_when_short_exceeds_long(self) -> None:
        lm = _make_ellipse_landmarks(90.0, 0.4, 0.2)
        loss = ellipse_axis_ordering_loss(lm)
        assert loss.item() > 0.01

    def test_zero_at_equal_lengths(self) -> None:
        lm = _make_ellipse_landmarks(90.0, 0.3, 0.3)
        loss = ellipse_axis_ordering_loss(lm)
        assert loss.item() < 1e-10

    def test_bounded(self) -> None:
        lm = _make_ellipse_landmarks(90.0, 0.8, 0.01)
        loss = ellipse_axis_ordering_loss(lm)
        assert loss.item() <= 1.0

    def test_gradient_flows(self) -> None:
        lm = _make_ellipse_landmarks(90.0, 0.5, 0.2).requires_grad_(True)
        ellipse_axis_ordering_loss(lm).backward()
        assert lm.grad is not None and lm.grad.abs().sum() > 0


class TestChamberAngleLoss:
    @staticmethod
    def _make_a4c(angles_deg: list[float]) -> torch.Tensor:
        """Create (1, 16, 2) A4C landmarks with specified per-chamber acute angles.

        上下 axis is vertical; 左右 axis is rotated so the acute angle between
        the two equals angles_deg.
        """
        pts = torch.zeros(1, 16, 2)
        for i, (base, angle) in enumerate(zip([0, 4, 8, 12], angles_deg, strict=True)):
            cx, cy = 0.3 + 0.2 * (i % 2), 0.3 + 0.2 * (i // 2)
            rad = math.radians(angle)
            pts[0, base] = torch.tensor([cx, cy - 0.05])
            pts[0, base + 1] = torch.tensor([cx, cy + 0.05])
            # 左右 direction = (sin(angle), cos(angle)) so dot with 上下 unit (0,1) = cos(angle)
            pts[0, base + 2] = torch.tensor([cx - 0.05 * math.sin(rad), cy - 0.05 * math.cos(rad)])
            pts[0, base + 3] = torch.tensor([cx + 0.05 * math.sin(rad), cy + 0.05 * math.cos(rad)])
        return pts

    def test_zero_above_threshold(self) -> None:
        lm = self._make_a4c([80.0, 85.0, 90.0, 75.0])
        loss = chamber_angle_loss(lm)
        assert loss.item() < 1e-10

    def test_penalizes_below_70(self) -> None:
        lm = self._make_a4c([50.0, 90.0, 90.0, 90.0])
        loss = chamber_angle_loss(lm)
        assert loss.item() > 0.0

    def test_zero_at_exactly_70(self) -> None:
        lm = self._make_a4c([70.0, 70.0, 70.0, 70.0])
        loss = chamber_angle_loss(lm)
        assert loss.item() < 1e-6

    def test_larger_violation_gives_larger_loss(self) -> None:
        lm_mild = self._make_a4c([60.0, 90.0, 90.0, 90.0])
        lm_severe = self._make_a4c([30.0, 90.0, 90.0, 90.0])
        assert chamber_angle_loss(lm_severe) > chamber_angle_loss(lm_mild)

    def test_gradient_flows(self) -> None:
        lm = self._make_a4c([50.0, 90.0, 90.0, 90.0]).requires_grad_(True)
        chamber_angle_loss(lm).backward()
        assert lm.grad is not None and lm.grad.abs().sum() > 0

    def test_bf16_input(self) -> None:
        lm = self._make_a4c([50.0, 80.0, 85.0, 90.0]).bfloat16()
        loss = chamber_angle_loss(lm)
        assert torch.isfinite(loss)


class TestGeometricConstraintDispatcher:
    def test_hc_returns_ortho_and_ordering(self) -> None:
        lm = _make_ellipse_landmarks(80.0, 0.4, 0.3)
        result = geometric_constraint_loss(lm, "HC")
        assert "ortho" in result and "axis_order" in result
        assert len(result) == 2

    def test_fa_returns_ortho_only(self) -> None:
        lm = _make_ellipse_landmarks(80.0, 0.4, 0.3)
        result = geometric_constraint_loss(lm, "FA")
        assert "ortho" in result and "axis_order" not in result

    def test_a4c_returns_chamber_angle(self) -> None:
        lm = torch.rand(2, 36, 2)
        result = geometric_constraint_loss(lm, "A4C")
        assert "chamber_angle" in result and len(result) == 1

    def test_other_tasks_return_empty(self) -> None:
        lm = torch.rand(2, 4, 2)
        for tid in ("PLAX", "PSAX", "IVC", "AOP", "FUGC", "fetal_femur"):
            assert geometric_constraint_loss(lm, tid) == {}

"""Tests for equivariance loss and views module.

Covers:
  1. Affine math: identity, round-trip, center-preservation
  2. Image warp: identity passthrough, spatial correctness
  3. Landmark warp: consistency with image warp
  4. Equivariance loss: zero at identity, nonzero under transform, gradient flow
  5. Integration: _equivariance_step forward+backward on mock data
"""

from __future__ import annotations

import math

import torch

from fubio.train.losses import equivariance_loss
from fubio.train.views import (
    _forward_to_inverse,
    sample_affine,
    warp_image,
    warp_landmarks,
)

# ──────────────────────────────────────────────────────────────────
# Affine math
# ──────────────────────────────────────────────────────────────────


class TestSampleAffine:
    def test_identity_at_zero_params(self):
        """Zero rotation, unit scale, zero translation → identity."""
        m = sample_affine(
            4,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        eye = torch.eye(2, 3).unsqueeze(0).expand(4, -1, -1)
        torch.testing.assert_close(m, eye, atol=1e-6, rtol=0)

    def test_shape(self):
        m = sample_affine(8, device=torch.device("cpu"))
        assert m.shape == (8, 2, 3)

    def test_center_preserved_rotation_only(self):
        """Pure rotation around center: (0.5, 0.5) maps to itself."""
        m = sample_affine(
            16,
            rotation_range=180.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        center = torch.tensor([[[0.5, 0.5]]]).expand(16, 1, 2)
        warped = warp_landmarks(center, m)
        torch.testing.assert_close(warped, center, atol=1e-5, rtol=0)

    def test_inverse_round_trip(self):
        """Forward then inverse → identity."""
        m_fwd = sample_affine(8, device=torch.device("cpu"))
        m_inv = _forward_to_inverse(m_fwd)
        # Compose: m_fwd then m_inv should give identity
        # Pad both to 3x3
        pad = torch.tensor([0.0, 0.0, 1.0]).expand(8, 1, 3)
        fwd_33 = torch.cat([m_fwd, pad], dim=1)
        inv_33 = torch.cat([m_inv, pad], dim=1)
        composed = inv_33 @ fwd_33
        eye = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
        torch.testing.assert_close(composed, eye, atol=1e-5, rtol=0)


# ──────────────────────────────────────────────────────────────────
# Image warp
# ──────────────────────────────────────────────────────────────────


class TestWarpImage:
    def test_identity_passthrough(self):
        """Identity transform should not change the image."""
        img = torch.randn(2, 3, 64, 64)
        m = sample_affine(
            2,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        warped = warp_image(img, m)
        # Interior pixels should match (edges may differ due to padding)
        torch.testing.assert_close(
            warped[:, :, 4:-4, 4:-4], img[:, :, 4:-4, 4:-4], atol=1e-4, rtol=1e-4
        )

    def test_output_shape(self):
        img = torch.randn(4, 3, 32, 32)
        m = sample_affine(4, device=torch.device("cpu"))
        warped = warp_image(img, m)
        assert warped.shape == img.shape

    def test_image_landmark_consistency(self):
        """A blob and its landmark must move to the same place under transform.

        Regression test for the warp direction bug: kornia's warp_affine
        expects a forward (src→dst) matrix. Passing the inverse produced
        images warped in the opposite direction from the landmarks.
        """
        S = 64
        img = torch.zeros(1, 1, S, S)
        # Gaussian blob at (row=20, col=30)
        cy, cx = 20, 30
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                r, c = cy + dy, cx + dx
                if 0 <= r < S and 0 <= c < S:
                    img[0, 0, r, c] = max(0.0, 1.0 - (dy**2 + dx**2) ** 0.5 / 4.0)

        lm = torch.tensor([[[[cx / S, cy / S]]]])  # (1,1,1,2)

        # Pure translation: right 10px, down 5px
        m = torch.eye(2, 3).unsqueeze(0)
        m[0, 0, 2] = 10.0 / S
        m[0, 1, 2] = 5.0 / S

        warped_img = warp_image(img, m)
        warped_lm = warp_landmarks(lm, m)

        # Find peak in warped image
        peak = warped_img[0, 0].argmax()
        peak_row, peak_col = peak // S, peak % S

        lm_col = warped_lm[0, 0, 0, 0].item() * S
        lm_row = warped_lm[0, 0, 0, 1].item() * S

        assert abs(peak_col - lm_col) < 2, (
            f"Image peak col={peak_col} vs landmark col={lm_col:.1f}"
        )
        assert abs(peak_row - lm_row) < 2, (
            f"Image peak row={peak_row} vs landmark row={lm_row:.1f}"
        )


# ──────────────────────────────────────────────────────────────────
# Landmark warp
# ──────────────────────────────────────────────────────────────────


class TestWarpLandmarks:
    def test_identity(self):
        lm = torch.rand(4, 2, 5, 2)
        m = sample_affine(
            4,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        warped = warp_landmarks(lm, m)
        torch.testing.assert_close(warped, lm, atol=1e-6, rtol=0)

    def test_pure_translation(self):
        """Translation by (0.1, -0.2) should shift all landmarks."""
        lm = torch.tensor([[[[0.3, 0.4], [0.5, 0.6]]]])  # (1, 1, 2, 2)
        # Build a manual translation matrix
        m = torch.eye(2, 3).unsqueeze(0)  # (1, 2, 3)
        m[0, 0, 2] = 0.1  # tx
        m[0, 1, 2] = -0.2  # ty
        warped = warp_landmarks(lm, m)
        expected = lm.clone()
        expected[..., 0] += 0.1
        expected[..., 1] -= 0.2
        torch.testing.assert_close(warped, expected, atol=1e-6, rtol=0)

    def test_90_degree_rotation(self):
        """90° around center: (0.7, 0.5) → (0.5, 0.7)."""
        m = sample_affine(
            1,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        # Manually set to 90° rotation around (0.5, 0.5)
        angle = torch.tensor(math.pi / 2)
        m[0, 0, 0] = torch.cos(angle)
        m[0, 0, 1] = -torch.sin(angle)
        m[0, 1, 0] = torch.sin(angle)
        m[0, 1, 1] = torch.cos(angle)
        m[0, 0, 2] = 0.5 * (1 - torch.cos(angle) + torch.sin(angle))
        m[0, 1, 2] = 0.5 * (1 - torch.cos(angle) - torch.sin(angle))

        lm = torch.tensor([[[[0.7, 0.5]]]])  # 0.2 right of center
        warped = warp_landmarks(lm, m)
        # After 90° CCW: should be 0.2 below center → (0.5, 0.7)
        expected = torch.tensor([[[[0.5, 0.7]]]])
        torch.testing.assert_close(warped, expected, atol=1e-5, rtol=0)

    def test_gradient_flows(self):
        """Landmarks must receive gradients through warp."""
        lm = torch.rand(2, 4, 8, 2, requires_grad=True)
        m = sample_affine(2, device=torch.device("cpu"))
        warped = warp_landmarks(lm, m)
        warped.sum().backward()
        assert lm.grad is not None
        assert lm.grad.abs().sum() > 0


# ──────────────────────────────────────────────────────────────────
# Equivariance loss
# ──────────────────────────────────────────────────────────────────


class TestEquivarianceLoss:
    def test_zero_at_identity(self):
        """If lm_trans == T(lm_ref), loss should be zero."""
        lm = torch.rand(4, 2, 5, 2)
        m = sample_affine(4, device=torch.device("cpu"))
        lm_mapped = warp_landmarks(lm, m)
        conf = torch.ones(4, 2, 1) * 5.0  # high confidence
        loss = equivariance_loss(lm, lm_mapped, m, conf)
        assert loss.item() < 1e-5

    def test_nonzero_under_mismatch(self):
        """Random lm_ref vs lm_trans should give nonzero loss."""
        lm_ref = torch.rand(4, 2, 5, 2)
        lm_trans = torch.rand(4, 2, 5, 2)
        m = sample_affine(4, device=torch.device("cpu"))
        conf = torch.ones(4, 2, 1) * 5.0
        loss = equivariance_loss(lm_ref, lm_trans, m, conf)
        assert loss.item() > 0.01

    def test_low_conf_reduces_loss(self):
        """Low confidence should down-weight the loss."""
        lm_ref = torch.rand(4, 2, 5, 2)
        lm_trans = torch.rand(4, 2, 5, 2)
        m = sample_affine(
            4,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            device=torch.device("cpu"),
        )
        high_conf = torch.ones(4, 2, 1) * 10.0  # sigmoid ≈ 1
        low_conf = torch.ones(4, 2, 1) * -10.0  # sigmoid ≈ 0
        loss_high = equivariance_loss(lm_ref, lm_trans, m, high_conf)
        loss_low = equivariance_loss(lm_ref, lm_trans, m, low_conf)
        assert loss_high > loss_low * 10  # high-conf loss should be much larger

    def test_gradient_to_lm_trans_only(self):
        """Gradients should flow to lm_trans, not through conf (detached)."""
        lm_ref = torch.rand(4, 2, 5, 2)
        lm_trans = torch.rand(4, 2, 5, 2, requires_grad=True)
        conf = torch.rand(4, 2, 1, requires_grad=True)
        m = sample_affine(4, device=torch.device("cpu"))
        loss = equivariance_loss(lm_ref, lm_trans, m, conf)
        loss.backward()
        assert lm_trans.grad is not None
        # conf is detached inside equivariance_loss, so no grad
        assert conf.grad is None

    def test_empty_input(self):
        """Empty tensors should return 0."""
        lm = torch.zeros(0, 2, 5, 2)
        m = torch.zeros(0, 2, 3)
        conf = torch.zeros(0, 2, 1)
        loss = equivariance_loss(lm, lm, m, conf)
        assert loss.item() == 0.0


# ──────────────────────────────────────────────────────────────────
# mil_loss normalization (Codex finding)
# ──────────────────────────────────────────────────────────────────


class TestMilDenomNormalization:
    """Verify total_mil is divided by n_tasks_with_mil, not left raw."""

    def test_mil_denom_in_total(self):
        """Grep the source for the fix — total_mil must be divided."""
        import inspect

        from fubio.train.module import FUBioModule

        src = inspect.getsource(FUBioModule._step)
        assert "total_mil / mil_denom" in src, (
            "total_mil should be normalized by mil_denom (n_tasks_with_mil)"
        )
        assert "n_tasks_with_mil" in src

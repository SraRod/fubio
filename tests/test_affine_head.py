"""R12 AffineHead verification: identity init, oriented prior, ortho regularization.

Tests:
- AffineHead identity init: T ≈ I at construction → coords match static prior baseline.
- Oriented prior: rotated T shifts the prior mean → coords follow the rotation.
- ortho_regularization: 0 for identity/rotation, >0 for shear.
- Forward contract: TaskModule with use_affine=True produces affine_T in output.
"""

from __future__ import annotations

import math

import torch

from fubio.models.coord_predictors import HeatmapPredictor
from fubio.models.heads import TaskModule
from fubio.train.losses import ortho_regularization, shape_consistency_loss


def _flatten_affinity(pred: HeatmapPredictor) -> None:
    """Zero q/k projections so q·memory is constant → uniform likelihood."""
    for lin in (pred.q_proj, pred.k_proj):
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)


def test_identity_affine_matches_static_prior() -> None:
    """Identity T produces the same coords as static prior (no affine_T)."""
    d, k, hp, wp = 16, 3, 8, 8
    means = [[0.2, 0.3], [0.7, 0.8], [0.5, 0.5]]
    stds = [[0.05, 0.05]] * k

    pred = HeatmapPredictor(d, k, prior_mean=means, prior_std=stds)
    _flatten_affinity(pred)

    B, N = 1, 1
    memory = torch.randn(B, hp * wp, d)
    lm_feat = torch.zeros(B, N, k, d)
    inst_feat = torch.zeros(B, N, d)

    # Static prior (no affine_T)
    c_static, _ = pred(inst_feat, lm_feat, memory, (hp, wp), affine_T=None)

    # Identity affine_T
    T_identity = torch.eye(2, 3).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 3)
    c_affine, _ = pred(inst_feat, lm_feat, memory, (hp, wp), affine_T=T_identity)

    assert torch.allclose(c_static, c_affine, atol=1e-5)


def test_rotated_affine_shifts_prior() -> None:
    """A 90° rotation T moves the prior means → coords shift accordingly."""
    d, k, hp, wp = 16, 3, 16, 16
    means = [[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]]
    stds = [[0.05, 0.05]] * k

    pred = HeatmapPredictor(d, k, prior_mean=means, prior_std=stds)
    _flatten_affinity(pred)

    B, N = 1, 1
    memory = torch.randn(B, hp * wp, d)
    lm_feat = torch.zeros(B, N, k, d)
    inst_feat = torch.zeros(B, N, d)

    # Static prior
    c_static, _ = pred(inst_feat, lm_feat, memory, (hp, wp), affine_T=None)

    # 90° CCW rotation: [x,y] → [-y, x], plus translation to keep in [0,1]
    # R = [[0, -1], [1, 0]], t = [1, 0] (so rotated coords land in [0,1])
    theta = math.pi / 2
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    T_rot = (
        torch.tensor(
            [
                [cos_t, -sin_t, 0.5 * (1 - cos_t + sin_t)],
                [sin_t, cos_t, 0.5 * (1 - sin_t - cos_t)],
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, 2, 3)

    c_rotated, _ = pred(inst_feat, lm_feat, memory, (hp, wp), affine_T=T_rot)

    # Coords should differ from static (prior means are rotated)
    diff = (c_static - c_rotated).abs().max()
    assert diff.item() > 0.05, f"Rotation should shift coords, but max diff = {diff.item()}"


def test_ortho_reg_zero_for_identity() -> None:
    """Identity T → ortho loss = 0."""
    T = torch.eye(2, 3).unsqueeze(0).unsqueeze(0).expand(2, 4, 2, 3)
    loss = ortho_regularization(T)
    assert loss.item() < 1e-10


def test_ortho_reg_zero_for_rotation() -> None:
    """Pure rotation → RᵀR = det(R)·I = I → ortho loss = 0."""
    theta = math.pi / 6  # 30°
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]])
    T = R.unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 3)
    loss = ortho_regularization(T)
    assert loss.item() < 1e-10


def test_ortho_reg_positive_for_shear() -> None:
    """Shear makes RᵀR ≠ det(R)·I → positive penalty."""
    T = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
    T = T.unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 3)
    loss = ortho_regularization(T)
    assert loss.item() > 0.01


def test_taskmodule_affine_output() -> None:
    """TaskModule with use_affine=True includes affine_T in output."""
    torch.manual_seed(0)
    d, k, n_inst = 32, 3, 2
    hp, wp = 6, 6

    means = [[0.3, 0.3], [0.7, 0.7], [0.5, 0.5]]
    stds = [[0.1, 0.1]] * k

    coord_pred = HeatmapPredictor(d, k, prior_mean=means, prior_std=stds)
    tm = TaskModule(
        n_keypoints=k,
        d_model=d,
        n_inst=n_inst,
        use_affine=True,
        coord_predictor=coord_pred,
    )

    attended = torch.randn(2, n_inst * (1 + k), d)
    memory = torch.randn(2, hp * wp, d)
    out = tm.head(attended, memory, (hp, wp))

    assert out.affine_T is not None
    assert out.affine_T.shape == (2, n_inst, 2, 3)
    assert out.landmarks.shape == (2, n_inst, k, 2)


def test_taskmodule_no_affine_output() -> None:
    """TaskModule with use_affine=False (default) produces affine_T=None."""
    torch.manual_seed(0)
    d, k, n_inst = 32, 3, 2
    hp, wp = 6, 6

    coord_pred = HeatmapPredictor(d, k)
    tm = TaskModule(
        n_keypoints=k,
        d_model=d,
        n_inst=n_inst,
        use_affine=False,
        coord_predictor=coord_pred,
    )

    attended = torch.randn(2, n_inst * (1 + k), d)
    memory = torch.randn(2, hp * wp, d)
    out = tm.head(attended, memory, (hp, wp))

    assert out.affine_T is None


def test_shape_consistency_center_scale_invariant() -> None:
    """Translated/scaled predictions should NOT be penalized (center+scale normalized).

    Rotation IS expected to change the loss — the PCA basis was fit without
    rotation alignment, so rotational variance is captured by the basis
    components. This is intentional: the loss and basis must use the same
    normalization convention (center + scale only).
    """
    K = 4
    # Canonical shape: a square-ish pattern, centered + unit Frobenius
    raw_mean = torch.tensor([[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]])
    centered = raw_mean - raw_mean.mean(dim=0, keepdim=True)
    canonical_mean = centered / centered.flatten().norm().clamp(min=1e-8)  # (K, 2)

    # Trivial basis (1 component) — just a direction in 2K space
    canonical_basis = torch.randn(1, 2 * K)
    canonical_basis = canonical_basis / canonical_basis.norm()

    mask = torch.ones(1, K, dtype=torch.bool)

    # Baseline: prediction = canonical_mean (perfect match)
    loss_baseline = shape_consistency_loss(
        canonical_mean.unsqueeze(0),
        mask,
        canonical_mean,
        canonical_basis,
    )

    # Translated prediction (shift by +0.5 in both axes)
    translated = (canonical_mean + 0.5).unsqueeze(0)
    loss_translated = shape_consistency_loss(
        translated,
        mask,
        canonical_mean,
        canonical_basis,
    )

    # Scaled prediction (2× larger)
    scaled = (canonical_mean * 2.0).unsqueeze(0)
    loss_scaled = shape_consistency_loss(
        scaled,
        mask,
        canonical_mean,
        canonical_basis,
    )

    # All should be near-zero (center+scale normalization removes these)
    assert loss_baseline.item() < 1e-6, f"Baseline loss too high: {loss_baseline.item()}"
    assert loss_translated.item() < 1e-6, f"Translated loss too high: {loss_translated.item()}"
    assert loss_scaled.item() < 1e-6, f"Scaled loss too high: {loss_scaled.item()}"


def test_shape_consistency_penalizes_bad_structure() -> None:
    """Structurally implausible shapes SHOULD be penalized."""
    K = 4
    raw_mean = torch.tensor([[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]])
    centered = raw_mean - raw_mean.mean(dim=0, keepdim=True)
    canonical_mean = centered / centered.flatten().norm().clamp(min=1e-8)

    canonical_basis = torch.randn(1, 2 * K)
    canonical_basis = canonical_basis / canonical_basis.norm()

    mask = torch.ones(1, K, dtype=torch.bool)

    # Scramble landmark positions → structurally wrong
    scrambled = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.1, 0.1], [0.9, 0.9]])
    loss_scrambled = shape_consistency_loss(
        scrambled.unsqueeze(0),
        mask,
        canonical_mean,
        canonical_basis,
    )

    # Should be significantly larger than zero
    assert loss_scrambled.item() > 1e-3, f"Scrambled loss too low: {loss_scrambled.item()}"


def test_shape_consistency_no_nan_gradients() -> None:
    """SVD backward must not produce NaN — R is detached from gradient graph."""
    K = 4
    raw_mean = torch.tensor([[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]])
    centered = raw_mean - raw_mean.mean(dim=0, keepdim=True)
    canonical_mean = centered / centered.flatten().norm().clamp(min=1e-8)

    canonical_basis = torch.randn(1, 2 * K)
    canonical_basis = canonical_basis / canonical_basis.norm()

    mask = torch.ones(8, K, dtype=torch.bool)

    # Near-isotropic predictions → cross_cov has near-degenerate singular values
    # (this is what triggered NaN before the detach fix)
    pred = torch.randn(8, K, 2, requires_grad=True)
    loss = shape_consistency_loss(pred, mask, canonical_mean, canonical_basis)
    loss.backward()

    assert not torch.isnan(pred.grad).any(), "NaN in shape_consistency_loss gradient"
    assert not torch.isinf(pred.grad).any(), "Inf in shape_consistency_loss gradient"


def test_affine_identity_init() -> None:
    """AffineHead's linear layer initializes to produce identity transforms."""
    torch.manual_seed(0)
    d, k, n_inst = 32, 3, 2

    tm = TaskModule(n_keypoints=k, d_model=d, n_inst=n_inst, use_affine=True)

    # With identity init (W=0), any input should produce T ≈ I
    inst_feat = torch.randn(1, n_inst, d)
    raw = tm.affine_fc(inst_feat).view(1, n_inst, 2, 3)

    I_23 = torch.eye(2, 3)
    for n in range(n_inst):
        assert torch.allclose(raw[0, n], I_23, atol=1e-6), (
            f"Slot {n}: expected identity, got {raw[0, n]}"
        )

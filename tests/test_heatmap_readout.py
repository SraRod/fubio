"""Phase B verification: heatmap readout math + landmark-aware matching.

Covers the two Tier-1 overfit fixes:
- HeatmapPredictor: coords read from grid geometry, in [0,1] without sigmoid,
  gradient healthy at edges (contrast: sigmoid saturates).
- PerTaskMatcher: assignment follows landmark quality, not the bbox head.
"""

from __future__ import annotations

import numpy as np
import torch

from fubio.data.task_registry import TASKS
from fubio.data.types import TaskOutput
from fubio.models.coord_predictors import HeatmapPredictor, normalized_grid
from fubio.train.losses import shape_consistency_loss
from fubio.train.matcher import PerTaskMatcher


def test_normalized_grid_pixel_centers() -> None:
    xs, ys = normalized_grid(4, 4, torch.device("cpu"))
    # first token = top-left cell center = (0.5/4, 0.5/4)
    assert xs[0].item() == 0.125
    assert ys[0].item() == 0.125
    # row-major: last token = bottom-right center
    assert xs[-1].item() == 0.875
    assert ys[-1].item() == 0.875


def test_soft_argmax_recovers_planted_peak() -> None:
    """A distribution peaked at cell (r,c) decodes to that cell's center."""
    hp, wp = 8, 8
    xs, ys = normalized_grid(hp, wp, torch.device("cpu"))
    r, c = 5, 2
    tgt_x, tgt_y = (c + 0.5) / wp, (r + 0.5) / hp

    d2 = (xs - tgt_x) ** 2 + (ys - tgt_y) ** 2
    heat = torch.softmax(-d2 / (0.5 / wp) ** 2, dim=-1)  # sharp peak at (r,c)

    coord_x = (heat * xs).sum()
    coord_y = (heat * ys).sum()
    assert abs(coord_x.item() - tgt_x) < 1e-3
    assert abs(coord_y.item() - tgt_y) < 1e-3


def test_heatmap_predictor_forward_contract() -> None:
    torch.manual_seed(0)
    d, k, hp, wp = 32, 3, 6, 6
    pred = HeatmapPredictor(d_model=d, n_keypoints=k)
    inst_feat = torch.randn(2, 2, d)
    lm_feat = torch.randn(2, 2, k, d)
    memory = torch.randn(2, hp * wp, d)

    coords, heat = pred(inst_feat, lm_feat, memory, (hp, wp))

    assert coords.shape == (2, 2, k, 2)
    assert heat.shape == (2, 2, k, hp * wp)
    assert (coords >= 0).all() and (coords <= 1).all()  # [0,1] without sigmoid
    assert torch.allclose(heat.sum(-1), torch.ones_like(heat.sum(-1)), atol=1e-5)


def test_micro_residual_bounded_and_in_range() -> None:
    """Micro residual refines within its cell bound and coords stay in [0,1]."""
    torch.manual_seed(2)
    d, k, hp, wp = 32, 3, 6, 6
    memory = torch.randn(2, hp * wp, d)
    lm_feat = torch.randn(2, 2, k, d)
    inst_feat = torch.randn(2, 2, d)

    base = HeatmapPredictor(d_model=d, n_keypoints=k, micro_residual=False)
    micro = HeatmapPredictor(d_model=d, n_keypoints=k, micro_residual=True, micro_cells=1.0)
    # share the affinity projections so only the residual differs
    micro.q_proj.load_state_dict(base.q_proj.state_dict())
    micro.k_proj.load_state_dict(base.k_proj.state_dict())

    c_base, _ = base(inst_feat, lm_feat, memory, (hp, wp))
    c_micro, _ = micro(inst_feat, lm_feat, memory, (hp, wp))

    assert (c_micro >= 0).all() and (c_micro <= 1).all()
    # correction is bounded by micro_cells/grid = 1/6 per axis
    assert (c_micro - c_base).abs().max().item() <= 1.0 / 6 + 1e-5


def _flatten_affinity(pred: HeatmapPredictor) -> None:
    """Zero q/k projections so q·memory is constant → uniform likelihood."""
    for lin in (pred.q_proj, pred.k_proj):
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)


def test_data_prior_breaks_collapse() -> None:
    """With flat affinity, the data prior anchors each landmark at its μ_k."""
    d, hp, wp = 16, 8, 8
    means = [[0.2, 0.2], [0.8, 0.8], [0.5, 0.3]]  # 3 distinct landmarks
    stds = [[0.05, 0.05]] * 3
    k = len(means)

    with_prior = HeatmapPredictor(d, k, prior_mean=means, prior_std=stds)
    no_prior = HeatmapPredictor(d, k)
    _flatten_affinity(with_prior)
    _flatten_affinity(no_prior)

    memory = torch.randn(1, hp * wp, d)
    lm_feat = torch.zeros(1, 1, k, d)
    inst_feat = torch.zeros(1, 1, d)

    c_prior, _ = with_prior(inst_feat, lm_feat, memory, (hp, wp))
    c_flat, _ = no_prior(inst_feat, lm_feat, memory, (hp, wp))

    c_prior = c_prior[0, 0]  # (k, 2)
    c_flat = c_flat[0, 0]

    # No prior + flat affinity → all landmarks collapse to the grid centroid
    assert torch.allclose(c_flat, c_flat[0].expand_as(c_flat), atol=1e-4)
    # With prior → each landmark sits near its own data mean (distinct, no collapse)
    assert torch.allclose(c_prior, torch.tensor(means), atol=0.03)
    pairwise = (c_prior[:, None] - c_prior[None, :]).norm(dim=-1)
    assert pairwise[pairwise > 0].min() > 0.2  # clearly separated


def test_memory_pos_is_opt_in() -> None:
    """memory_pos must be inert unless use_memory_pos is explicitly set.

    It arrived unconditionally with the ConvNeXtV2 work and silently altered the
    DINOv2 readout, which R15 — the best online run — never had. The default
    must reproduce the R15 path; enabling it is an ablation, not a side effect.
    """
    torch.manual_seed(7)
    d, k, hp, wp = 32, 3, 6, 6
    inst_feat = torch.randn(2, 2, d)
    lm_feat = torch.randn(2, 2, k, d)
    memory = torch.randn(2, hp * wp, d)

    from fubio.models.neck import sinusoidal_2d_pos_enc

    pos = sinusoidal_2d_pos_enc(hp, wp, d)

    torch.manual_seed(7)
    off = HeatmapPredictor(d_model=d, n_keypoints=k)
    base, _ = off(inst_feat, lm_feat, memory, (hp, wp))
    passed_but_off, _ = off(inst_feat, lm_feat, memory, (hp, wp), memory_pos=pos)
    assert torch.allclose(base, passed_but_off, atol=1e-6), (
        "default must ignore memory_pos entirely"
    )

    torch.manual_seed(7)
    on = HeatmapPredictor(d_model=d, n_keypoints=k, use_memory_pos=True)
    enabled, _ = on(inst_feat, lm_feat, memory, (hp, wp), memory_pos=pos)
    assert not torch.allclose(base, enabled, atol=1e-4), (
        "with use_memory_pos=True it must change the key representation"
    )


def test_prior_dropout_is_train_only_and_zeros_the_prior() -> None:
    """Binary per-instance drop, active only in training.

    KNOWN HARMFUL at p>0: 3 of 4 runs at p=0.1 collapsed the instance slots,
    against 5 of 5 healthy at p=0. The test pins the behaviour so the hazard is
    visible in code, not so the setting is recommended — reducing prior
    dependence should use an auxiliary prior-free visual heatmap loss instead.
    """
    torch.manual_seed(0)
    d, hp, wp = 16, 8, 8
    means = [[0.2, 0.2], [0.8, 0.8]]
    stds = [[0.05, 0.05]] * 2
    k = len(means)
    memory = torch.randn(1, hp * wp, d)
    lm_feat = torch.zeros(1, 1, k, d)
    inst_feat = torch.zeros(1, 1, d)

    pred = HeatmapPredictor(d, k, prior_mean=means, prior_std=stds, prior_dropout=1.0)
    _flatten_affinity(pred)

    # train, p=1.0: prior fully dropped -> flat affinity -> collapse to centroid.
    # This IS the failure regime; the point is that it is a cliff, not a gradient.
    pred.train()
    c_train = pred(inst_feat, lm_feat, memory, (hp, wp))[0][0, 0]
    assert torch.allclose(c_train[0], c_train[1], atol=1e-3)

    # eval: prior always applied, landmarks stay at their separate means
    pred.eval()
    c_eval = pred(inst_feat, lm_feat, memory, (hp, wp))[0][0, 0]
    assert not torch.allclose(c_eval[0], c_eval[1], atol=1e-2)


def test_prior_dropout_zero_is_identity() -> None:
    """p=0 must reproduce the un-attenuated prior exactly, in train and eval."""
    torch.manual_seed(0)
    d, hp, wp = 16, 8, 8
    means = [[0.2, 0.2], [0.8, 0.8]]
    stds = [[0.05, 0.05]] * 2
    pred = HeatmapPredictor(d, 2, prior_mean=means, prior_std=stds, prior_dropout=0.0)
    _flatten_affinity(pred)
    memory = torch.randn(1, hp * wp, d)
    lm_feat = torch.zeros(1, 1, 2, d)
    inst_feat = torch.zeros(1, 1, d)

    pred.train()
    a = pred(inst_feat, lm_feat, memory, (hp, wp))[0]
    pred.eval()
    b = pred(inst_feat, lm_feat, memory, (hp, wp))[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_heatmap_gradient_healthy_at_edge() -> None:
    """Coords can approach the border with non-vanishing gradient.

    Sigmoid-based readout needs logit→-inf to reach coord=0, so its gradient
    dies at the edge; heatmap soft-argmax does not.
    """
    torch.manual_seed(1)
    d, k, hp, wp = 16, 1, 5, 5
    pred = HeatmapPredictor(d_model=d, n_keypoints=k)
    memory = torch.randn(1, hp * wp, d, requires_grad=True)
    lm_feat = torch.randn(1, 1, k, d)

    coords, _ = pred(torch.randn(1, 1, d), lm_feat, memory, (hp, wp))
    # push x toward the left border (0.0)
    loss = coords[..., 0].sum()
    loss.backward()

    assert memory.grad is not None
    assert memory.grad.abs().sum().item() > 1e-6


def _canonical_triangle() -> tuple[torch.Tensor, torch.Tensor]:
    raw = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    raw = raw - raw.mean(0)
    raw = raw / np.linalg.norm(raw)  # unit Frobenius, matches _procrustes_align
    mean = torch.tensor(raw, dtype=torch.float32)  # (3, 2)
    basis = torch.zeros(1, 6)
    basis[0, 0] = 1.0  # a single (arbitrary) shape direction
    return mean, basis


def test_shape_consistency_zero_on_mean_and_pose_invariant() -> None:
    mean, basis = _canonical_triangle()
    k = mean.shape[0]
    full_mask = torch.ones(1, k, dtype=torch.bool)

    on_manifold = mean.unsqueeze(0)  # (1, K, 2) == canonical mean
    loss0 = shape_consistency_loss(on_manifold, full_mask, mean, basis)
    assert loss0.item() < 1e-6

    # translate + scale — pose must not change the loss
    posed = (mean * 3.0 + 10.0).unsqueeze(0)
    loss_pose = shape_consistency_loss(posed, full_mask, mean, basis)
    assert loss_pose.item() < 1e-6


def test_shape_consistency_penalizes_off_manifold() -> None:
    mean, basis = _canonical_triangle()
    k = mean.shape[0]
    full_mask = torch.ones(1, k, dtype=torch.bool)

    bad = mean.clone()
    bad[2] = torch.tensor([0.8, -0.6])  # move one point off the shape manifold
    loss = shape_consistency_loss(bad.unsqueeze(0), full_mask, mean, basis)
    assert loss.item() > 1e-3


def test_shape_consistency_skips_partially_visible() -> None:
    mean, basis = _canonical_triangle()
    k = mean.shape[0]
    partial = torch.zeros(1, k, dtype=torch.bool)
    partial[0, 0] = True  # not all visible → no valid Procrustes alignment
    loss = shape_consistency_loss(mean.unsqueeze(0), partial, mean, basis)
    assert loss.item() == 0.0


def _inst(task_int: int, kp: np.ndarray, bbox: np.ndarray) -> dict:
    k = kp.shape[0]
    return {
        "task_id": task_int,
        "keypoints": kp.astype(np.float32),
        "supervised_mask": np.ones(k, dtype=bool),
        "visible_mask": np.ones(k, dtype=bool),
        "bbox": bbox.astype(np.float32),
        "transform_matrix": np.eye(3, dtype=np.float32),
        "original_hw": np.array([100, 100], dtype=np.int32),
        "is_labeled": True,
        "image_path": "x",
    }


def test_matcher_follows_landmarks_not_bbox() -> None:
    """Slot with correct landmarks wins under cost_land, even with worse bbox."""
    tid = "IVC"
    task_int = TASKS[tid].task_int
    k = TASKS[tid].n_keypoints

    gt_kp = np.array([[0.3, 0.3], [0.6, 0.6]])[:k]
    gt_bbox = np.array([0.45, 0.45, 0.35, 0.35])

    # Both slots share the same (uninformative) bbox — only landmarks differ.
    # This is the situation the fix targets: when the bbox head can't
    # discriminate slots, landmark distance must decide the assignment.
    slot0 = [[0.9, 0.9] for _ in range(k)]  # wrong
    slot1 = gt_kp.tolist()  # exact
    landmarks = torch.tensor([[slot0, slot1]]).float()  # (1, 2, k, 2)
    bbox = torch.tensor([[gt_bbox.tolist(), gt_bbox.tolist()]]).float()  # (1,2,4)
    conf = torch.zeros(1, 2, 1)

    out = {tid: TaskOutput(bbox=bbox, conf=conf, landmarks=landmarks)}
    targets = [[_inst(task_int, gt_kp, gt_bbox)]]

    bbox_only = PerTaskMatcher(cost_land=0.0)(out, targets)[tid][0]
    land_aware = PerTaskMatcher(cost_land=10.0)(out, targets)[tid][0]

    assert bbox_only[0] == [0]  # tie on bbox → arbitrary (slot 0)
    assert land_aware[0] == [1]  # landmark-aware breaks the tie → correct slot 1

"""Readout geometry support: normalized grid, shape consistency, matcher.

The GeoSimCC readout reads coordinates off the patch grid via soft-argmax;
these tests pin the grid convention, the shape-consistency regularizer and
the landmark-aware matcher that the readout depends on.
"""

from __future__ import annotations

import numpy as np
import torch

from fubio.data.task_registry import TASKS
from fubio.data.types import TaskOutput
from fubio.models.coord_predictors import normalized_grid
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

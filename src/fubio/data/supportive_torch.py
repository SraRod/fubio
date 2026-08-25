"""Differentiable supportive landmark computation in PyTorch.

Torch mirror of supportive.py — same geometric constructions, but vectorized
for batched inputs and fully differentiable so gradients flow from supportive
error back to scored landmark predictions.

All operations (midpoint, projection, trig, norm) have well-defined PyTorch
gradients. Numerical guards (eps in sqrt, safe division for parallel lines,
float32 casting) prevent NaN in degenerate configurations under bf16-mixed.

Upstream: supportive.py (reference numpy implementation, used by collate).
Downstream: train/module.py (geometric consistency loss).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Helpers — all operate on (..., 2) tensors, fully differentiable
# ---------------------------------------------------------------------------


def _midpoint(a: Tensor, b: Tensor) -> Tensor:
    return (a + b) * 0.5


def _line_intersection(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """Intersection of line(A→B) and line(C→D). Falls back to centroid if parallel.

    All inputs: (..., 2). Output: (..., 2).
    Near-parallel lines (|cross| < threshold) use centroid to avoid gradient
    explosion from the 1/cross² Jacobian.
    """
    d1 = b - a
    d2 = d - c
    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    parallel = cross.abs() < 1e-3
    safe_cross = torch.where(parallel, torch.ones_like(cross), cross)
    t_num = (c[..., 0] - a[..., 0]) * d2[..., 1] - (c[..., 1] - a[..., 1]) * d2[..., 0]
    t = t_num / safe_cross
    result = a + t.unsqueeze(-1) * d1
    centroid = (a + b + c + d) * 0.25
    return torch.where(parallel.unsqueeze(-1), centroid, result)


# ---------------------------------------------------------------------------
# Per-task differentiable geometry
# ---------------------------------------------------------------------------


def _a4c_torch(gt: Tensor) -> Tensor:
    """A4C: 4 chambers × (cross + 4 edge midpoints) = 20 supportive.

    gt: (N, 16, 2) → (N, 20, 2)
    """
    parts: list[Tensor] = []
    chambers = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
    for i0, i1, i2, i3 in chambers:
        parts.append(_line_intersection(gt[:, i0], gt[:, i1], gt[:, i2], gt[:, i3]))
        for a, b in [(i0, i2), (i2, i1), (i1, i3), (i3, i0)]:
            parts.append(_midpoint(gt[:, a], gt[:, b]))
    return torch.stack(parts, dim=1)


def _plax_torch(gt: Tensor) -> Tensor:
    """PLAX: midpoint of each of 11 measurement lines.

    gt: (N, 22, 2) → (N, 11, 2)
    """
    parts: list[Tensor] = []
    for i in range(11):
        parts.append(_midpoint(gt[:, 2 * i], gt[:, 2 * i + 1]))
    return torch.stack(parts, dim=1)


def _psax_torch(gt: Tensor) -> Tensor:
    """PSAX: CTR (centroid) + mid_R2P2 (bifurcation) + mid_extX (convergence).

    Extension direction is index-based (always past p2) rather than
    X-conditional, so the construction is flip-equivariant when PSAX
    uses empty swap pairs.

    gt: (N, 4, 2) → (N, 3, 2)
    """
    ctr = (gt[:, 0] + gt[:, 1] + gt[:, 2] + gt[:, 3]) * 0.25
    mid_r2p2 = _midpoint(gt[:, 1], gt[:, 3])
    ext_rvot = gt[:, 1] + 0.5 * (gt[:, 1] - gt[:, 0])
    ext_pa = gt[:, 3] + 0.5 * (gt[:, 3] - gt[:, 2])
    mid_ext_x = _midpoint(ext_rvot, ext_pa)
    return torch.stack([ctr, mid_r2p2, mid_ext_x], dim=1)


def _ivc_torch(gt: Tensor) -> Tensor:
    """IVC: extension 1.0× + bilateral perpendicular = 6 points.

    Avoids norm/division: length*d_hat == d, length*n_hat == rotate90(d).
    Gives finite, simple gradients even when endpoints coincide.

    gt: (N, 2, 2) → (N, 6, 2)
    """
    p0 = gt[:, 0]  # (N, 2)
    p1 = gt[:, 1]
    d = p1 - p0
    n = torch.stack([-d[..., 1], d[..., 0]], dim=-1)  # perpendicular, same length

    return torch.stack([
        p0 - d,
        p1 + d,
        p0 + n,
        p1 + n,
        p0 - n,
        p1 - n,
    ], dim=1)


def _rect_ext_torch(gt: Tensor, ext_factor: float = 0.25) -> Tensor:
    """FUGC/fetal_femur: rectangle projection + extension past GT.

    gt: (N, 2, 2) → (N, 4, 2)
    """
    p0 = gt[:, 0]  # (N, 2)
    p1 = gt[:, 1]
    rect_1 = torch.stack([p0[..., 0], p1[..., 1]], dim=-1)
    rect_2 = torch.stack([p1[..., 0], p0[..., 1]], dim=-1)
    dir_1 = p0 - rect_1
    dir_2 = p1 - rect_2
    ext_1 = p0 + ext_factor * dir_1
    ext_2 = p1 + ext_factor * dir_2
    return torch.stack([rect_1, rect_2, ext_1, ext_2], dim=1)


def _ellipse_torch(gt: Tensor) -> Tensor:
    """HC/FA: 2 foci + 8 ellipse sample points = 10 supportive.

    Computed in float32 regardless of input dtype for numerical stability
    (cross products and near-circle differences are precision-sensitive).

    gt: (N, 4, 2) → (N, 10, 2)
    """
    orig_dtype = gt.dtype
    gt = gt.float()
    p0, p1, p2, p3 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    cx = (p0[:, 0] + p1[:, 0]) * 0.5
    cy = (p0[:, 1] + p1[:, 1]) * 0.5
    a = (p0 - p1).norm(dim=-1) * 0.5  # semi-major
    b = (p2 - p3).norm(dim=-1) * 0.5  # semi-minor
    # Guard atan2 for coincident endpoints: add eps to denominator direction
    dy = p1[:, 1] - p0[:, 1]
    dx = p1[:, 0] - p0[:, 0]
    theta = torch.atan2(dy, dx + 1e-8 * (dx.abs() < 1e-8).float())

    c = torch.sqrt((a * a - b * b).abs() + 1e-4)
    cos_t = theta.cos()
    sin_t = theta.sin()
    fx = torch.where(a >= b, cos_t, -sin_t)
    fy = torch.where(a >= b, sin_t, cos_t)

    parts: list[Tensor] = [
        torch.stack([cx + c * fx, cy + c * fy], dim=-1),
        torch.stack([cx - c * fx, cy - c * fy], dim=-1),
    ]

    for deg in (30, 60, 120, 150, 210, 240, 300, 330):
        phi = math.radians(deg)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        ex = cx + a * cos_phi * cos_t - b * sin_phi * sin_t
        ey = cy + a * cos_phi * sin_t + b * sin_phi * cos_t
        parts.append(torch.stack([ex, ey], dim=-1))

    return torch.stack(parts, dim=1).to(orig_dtype)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Callable[..., Tensor] | None] = {
    "A4C": _a4c_torch,
    "PLAX": _plax_torch,
    "PSAX": _psax_torch,
    "IVC": _ivc_torch,
    "AOP": None,
    "FUGC": _rect_ext_torch,
    "HC": _ellipse_torch,
    "FA": _ellipse_torch,
    "fetal_femur": _rect_ext_torch,
}


def compute_supportive_torch(task_id: str, scored_coords: Tensor) -> Tensor | None:
    """Differentiable supportive landmark computation from predicted scored coords.

    Args:
        task_id: task identifier (e.g. "A4C", "HC").
        scored_coords: (N, K_scored, 2) predicted scored landmarks in [0, 1].

    Returns:
        (N, K_sup, 2) derived supportive landmarks, or None if task has no supportive.
        Gradient flows back to scored_coords through the geometry.
    """
    fn = _DISPATCH.get(task_id)
    if fn is None:
        return None
    return fn(scored_coords)

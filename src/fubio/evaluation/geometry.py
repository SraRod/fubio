"""Differentiable clinical parameter geometry — pure torch, 0 learnable params.

All 28 clinical parameters computed from landmark coordinates via three
primitives: Euclidean distance, Ramanujan ellipse circumference, and
angle via arccos. ALL computation forced to fp32 regardless of model
precision — bf16 is numerically unsafe for arccos/sqrt/division.

Upstream: data/task_registry.py (ParamDef definitions).
Downstream: train/losses.py imports for L_param (gradient flows through);
            evaluation/metrics.py imports for eval (under no_grad).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from fubio.data.task_registry import PARAMS

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def compute_distance(p1: Tensor, p2: Tensor) -> Tensor:
    """Euclidean distance ||p1 - p2||_2, batched. Smoothed — training path.

    The +_EPS under the sqrt keeps the gradient finite at coincident points, which
    L_param needs. It also means coincident points return ~3.2e-4 rather than 0,
    so this must not be used for evaluation: see _exact_distance.

    Args:
        p1: (*, 2) landmark coordinates.
        p2: (*, 2) landmark coordinates.

    Returns:
        (*,) distance values in fp32.
    """
    p1 = p1.float()
    p2 = p2.float()
    return torch.sqrt(((p1 - p2) ** 2).sum(dim=-1) + _EPS)


def compute_ellipse_circumference(
    p1: Tensor,
    p2: Tensor,
    p3: Tensor,
    p4: Tensor,
) -> Tensor:
    """Ramanujan approximation for ellipse circumference. Smoothed — training path.

    Axes: a = ||p1 - p2||/2, b = ||p3 - p4||/2 (semi-axes).
    C ≈ π(3(a+b) - √((3a+b)(a+3b)))

    Args:
        p1, p2: (*, 2) endpoints of axis a.
        p3, p4: (*, 2) endpoints of axis b.

    Returns:
        (*,) circumference values in fp32.
    """
    a = compute_distance(p1, p2) / 2.0
    b = compute_distance(p3, p4) / 2.0
    inner = (3.0 * a + b) * (a + 3.0 * b)
    return math.pi * (3.0 * (a + b) - torch.sqrt(inner.clamp(min=_EPS)))


def compute_angle(vertex: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Angle at vertex between rays vertex→p1 and vertex→p2. Smoothed — training path.

    WARNING — do not use for evaluation. The +_EPS inside each norm means a
    zero-length ray (p1 == vertex) yields cos_theta = 0 and therefore exactly
    π/2: an undefined angle is reported as a plausible 90°, which no downstream
    consumer can distinguish from a real measurement. Use _valid_angle instead.

    Args:
        vertex: (*, 2) angle vertex.
        p1: (*, 2) ray endpoint 1.
        p2: (*, 2) ray endpoint 2.

    Returns:
        (*,) angle in radians, fp32.
    """
    vertex = vertex.float()
    p1 = p1.float()
    p2 = p2.float()
    u = p1 - vertex
    v = p2 - vertex
    dot = (u * v).sum(dim=-1)
    norm_u = torch.sqrt((u**2).sum(dim=-1) + _EPS)
    norm_v = torch.sqrt((v**2).sum(dim=-1) + _EPS)
    cos_theta = dot / (norm_u * norm_v)
    cos_theta = cos_theta.clamp(-1.0 + _EPS, 1.0 - _EPS)
    return torch.acos(cos_theta)


# --- Evaluation primitives -------------------------------------------------
#
# Separate functions rather than a `strict=` flag on the ones above: a flag whose
# default selects training behaviour means every evaluation call site has to
# remember to pass it, and a single omission silently restores the fabricated 90°.
# The difference is also not merely numerical — degeneracy handling is a clinical
# validity policy, which is what these names say.

# Below this length (in the coordinate space the landmarks are expressed in) a ray
# or semi-axis carries no orientation/size information and the derived parameter is
# reported invalid rather than approximated.
_DEGENERATE_TOL = 1e-6


def _exact_distance(p1: Tensor, p2: Tensor) -> Tensor:
    """Euclidean distance, exact at zero. Evaluation path."""
    return torch.linalg.vector_norm((p1.float() - p2.float()), dim=-1)


def _valid_ellipse_circumference(
    p1: Tensor,
    p2: Tensor,
    p3: Tensor,
    p4: Tensor,
) -> Tensor:
    """Ramanujan circumference, NaN when either axis is degenerate.

    A zero semi-axis still has a finite Ramanujan limit, so this is a clinical
    validity decision, not a numerical one: an ellipse with a collapsed axis is
    not a measurement we are willing to report.
    """
    a = _exact_distance(p1, p2) / 2.0
    b = _exact_distance(p3, p4) / 2.0
    inner = (3.0 * a + b) * (a + 3.0 * b)
    circ = math.pi * (3.0 * (a + b) - torch.sqrt(inner.clamp(min=0.0)))
    degenerate = (a <= _DEGENERATE_TOL) | (b <= _DEGENERATE_TOL)
    return torch.where(degenerate, torch.full_like(circ, float("nan")), circ)


def _valid_angle(vertex: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Angle at vertex, NaN when either ray has no length. Evaluation path."""
    vertex = vertex.float()
    u = p1.float() - vertex
    v = p2.float() - vertex
    norm_u = torch.linalg.vector_norm(u, dim=-1)
    norm_v = torch.linalg.vector_norm(v, dim=-1)

    degenerate = (norm_u <= _DEGENERATE_TOL) | (norm_v <= _DEGENERATE_TOL)
    denom = (norm_u * norm_v).clamp(min=_DEGENERATE_TOL)
    cos_theta = ((u * v).sum(dim=-1) / denom).clamp(-1.0, 1.0)
    angle = torch.acos(cos_theta)
    return torch.where(degenerate, torch.full_like(angle, float("nan")), angle)


# ---------------------------------------------------------------------------
# Per-task dispatcher
# ---------------------------------------------------------------------------

# Pre-group ParamDefs by task_id for fast lookup
_PARAMS_BY_TASK: dict[str, list[tuple[str, str, list[int]]]] = {}
for _p in PARAMS.values():
    _PARAMS_BY_TASK.setdefault(_p.task_id, []).append(
        (_p.name, _p.formula_type, _p.landmark_indices)
    )


def _compute_params(
    landmarks: Tensor,
    task_id: str,
    visible_mask: Tensor | None,
    *,
    distance_fn,
    ellipse_fn,
    angle_fn,
) -> dict[str, Tensor]:
    """Shared registry dispatch. Callers pick the primitive set; see the two
    public wrappers below."""
    landmarks = landmarks.float()
    n = landmarks.shape[0]
    param_defs = _PARAMS_BY_TASK.get(task_id, [])
    result: dict[str, Tensor] = {}

    for name, formula_type, indices in param_defs:
        # Per-parameter validity: all required landmarks must be visible
        if visible_mask is not None:
            param_valid = torch.ones(n, dtype=torch.bool, device=landmarks.device)
            for idx in indices:
                param_valid = param_valid & visible_mask[:, idx]
        else:
            param_valid = None

        if formula_type == "distance":
            val = distance_fn(
                landmarks[:, indices[0]],
                landmarks[:, indices[1]],
            )
        elif formula_type == "ellipse_circumference":
            val = ellipse_fn(
                landmarks[:, indices[0]],
                landmarks[:, indices[1]],
                landmarks[:, indices[2]],
                landmarks[:, indices[3]],
            )
        elif formula_type == "angle":
            val = angle_fn(
                landmarks[:, indices[0]],
                landmarks[:, indices[1]],
                landmarks[:, indices[2]],
            )
        else:
            raise ValueError(f"Unknown formula_type: {formula_type}")

        if param_valid is not None:
            val = torch.where(param_valid, val, torch.tensor(float("nan"), device=val.device))

        result[name] = val

    return result


def compute_params_for_training(
    landmarks: Tensor,
    task_id: str,
    visible_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Clinical parameters for L_param — epsilon-smoothed, differentiable everywhere.

    Gradients flow through this. Degenerate geometry is approximated rather than
    rejected, because a NaN here would poison the loss.

    Args:
        landmarks: (N, K_task, 2) coordinates in any consistent space.
        task_id: task string id (e.g. "HC", "A4C").
        visible_mask: (N, K_task) bool — if a param's required landmarks are
            not all visible, output NaN for that param.

    Returns:
        {param_name: (N,)} values. NaN only where landmarks are not visible.
    """
    return _compute_params(
        landmarks,
        task_id,
        visible_mask,
        distance_fn=compute_distance,
        ellipse_fn=compute_ellipse_circumference,
        angle_fn=compute_angle,
    )


def compute_params_for_evaluation(
    landmarks: Tensor,
    task_id: str,
    visible_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Clinical parameters for metrics — exact, with degenerate geometry marked NaN.

    Differs from the training variant in two ways, both deliberate: distances are
    exact at zero, and a zero-length ray or collapsed ellipse axis yields NaN
    instead of a plausible-looking value (the smoothed path returns exactly 90°
    for an undefined angle).

    Args:
        landmarks: (N, K_task, 2) coordinates in any consistent space. For the
            competition-comparable metric this must be **original pixel space**.
        task_id: task string id (e.g. "HC", "A4C").
        visible_mask: (N, K_task) bool — if a param's required landmarks are
            not all visible, output NaN for that param.

    Returns:
        {param_name: (N,)} values. NaN where landmarks are not visible OR the
        geometry is degenerate.
    """
    return _compute_params(
        landmarks,
        task_id,
        visible_mask,
        distance_fn=_exact_distance,
        ellipse_fn=_valid_ellipse_circumference,
        angle_fn=_valid_angle,
    )

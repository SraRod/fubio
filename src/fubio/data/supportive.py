"""Supportive landmark computation and evidence detection.

Pure numpy functions that derive training-only supportive landmarks from
scored GT geometry, plus per-landmark evidence (ultrasound signal presence).

Upstream: task_registry.py (task definitions, n_supportive counts).
Downstream: collate.py (compute supportive at collate time),
            manifest.py (compute evidence from raw image),
            shape_prior.py (expand PCA to include supportive).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


# (coords, source_indices) — coords: (K_sup, 2), source_indices: per-point
# list of scored landmark indices this supportive point depends on.
SupportiveResult = tuple[npt.NDArray[np.float32], tuple[tuple[int, ...], ...]]

_EMPTY_RESULT: SupportiveResult = (
    np.empty((0, 2), dtype=np.float32),
    (),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _midpoint(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    return (a + b) / 2.0


def _line_intersection(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray,
) -> npt.NDArray:
    """Intersection of line(A→B) and line(C→D). Falls back to centroid if parallel."""
    d1 = b - a
    d2 = d - c
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return (a + b + c + d) / 4.0
    t = ((c[0] - a[0]) * d2[1] - (c[1] - a[1]) * d2[0]) / cross
    return a + t * d1


# ---------------------------------------------------------------------------
# Per-task supportive computation
# ---------------------------------------------------------------------------


def _compute_a4c(gt: npt.NDArray) -> SupportiveResult:
    """A4C: 4 chambers × (cross + 4 edge midpoints) = 20 points."""
    coords: list[npt.NDArray] = []
    sources: list[tuple[int, ...]] = []
    chambers = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
    for i0, i1, i2, i3 in chambers:
        coords.append(_line_intersection(gt[i0], gt[i1], gt[i2], gt[i3]))
        sources.append((i0, i1, i2, i3))
        for a, b in [(i0, i2), (i2, i1), (i1, i3), (i3, i0)]:
            coords.append(_midpoint(gt[a], gt[b]))
            sources.append((a, b))
    return np.array(coords, dtype=np.float32), tuple(sources)


def _compute_plax(gt: npt.NDArray) -> SupportiveResult:
    """PLAX: midpoint of each of 11 measurement lines."""
    coords: list[npt.NDArray] = []
    sources: list[tuple[int, ...]] = []
    for i in range(11):
        a, b = 2 * i, 2 * i + 1
        coords.append(_midpoint(gt[a], gt[b]))
        sources.append((a, b))
    return np.array(coords, dtype=np.float32), tuple(sources)


def _compute_psax(gt: npt.NDArray) -> SupportiveResult:
    """PSAX: CTR (centroid) + mid_R2P2 (bifurcation) + mid_extX (convergence).

    Extension direction is index-based (always past p2) rather than
    X-conditional, so the construction is flip-equivariant when PSAX
    uses empty swap pairs.
    """
    ctr = (gt[0] + gt[1] + gt[2] + gt[3]) / 4.0
    mid_r2p2 = _midpoint(gt[1], gt[3])
    # Extend each line 0.5× past its p2 endpoint
    ext_rvot = gt[1] + 0.5 * (gt[1] - gt[0])
    ext_pa = gt[3] + 0.5 * (gt[3] - gt[2])
    mid_ext_x = _midpoint(ext_rvot, ext_pa)
    coords = np.array([ctr, mid_r2p2, mid_ext_x], dtype=np.float32)
    sources = ((0, 1, 2, 3), (1, 3), (0, 1, 2, 3))
    return coords, sources


def _compute_ivc(gt: npt.NDArray) -> SupportiveResult:
    """IVC: extension 1.0× + bilateral perpendicular = 6 points."""
    p0, p1 = gt[0], gt[1]
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return np.full((6, 2), np.nan, dtype=np.float32), (
            (0,), (1,), (0,), (1,), (0,), (1,),
        )
    d_hat = d / length
    n_hat = np.array([-d_hat[1], d_hat[0]])
    factor = 1.0
    coords = np.array([
        p0 - factor * length * d_hat,   # ext_beyond_p1
        p1 + factor * length * d_hat,   # ext_beyond_p2
        p0 + factor * length * n_hat,   # perp_at_p1_L
        p1 + factor * length * n_hat,   # perp_at_p2_L
        p0 - factor * length * n_hat,   # perp_at_p1_R
        p1 - factor * length * n_hat,   # perp_at_p2_R
    ], dtype=np.float32)
    sources = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
    return coords, sources


def _compute_rect_with_ext(gt: npt.NDArray, ext_factor: float = 0.25) -> SupportiveResult:
    """FUGC/fetal_femur: rectangle projection + extension past GT along rect sides."""
    p0, p1 = gt[0], gt[1]
    rect_1 = np.array([p0[0], p1[1]], dtype=np.float32)
    rect_2 = np.array([p1[0], p0[1]], dtype=np.float32)
    dir_1 = p0 - rect_1
    dir_2 = p1 - rect_2
    if abs(float(p0[1] - p1[1])) < 1e-6:
        ext_1 = np.full(2, np.nan, dtype=np.float32)
        ext_2 = np.full(2, np.nan, dtype=np.float32)
    else:
        ext_1 = p0 + ext_factor * dir_1
        ext_2 = p1 + ext_factor * dir_2
    coords = np.array([rect_1, rect_2, ext_1, ext_2], dtype=np.float32)
    sources = ((0, 1), (0, 1), (0, 1), (0, 1))
    return coords, sources


def _compute_ellipse(gt: npt.NDArray) -> SupportiveResult:
    """HC/FA: 2 foci + 8 ellipse sample points = 10."""
    p0, p1, p2, p3 = gt[0], gt[1], gt[2], gt[3]
    cx, cy = float((p0[0] + p1[0]) / 2), float((p0[1] + p1[1]) / 2)
    a = float(np.linalg.norm(p0 - p1)) / 2
    b = float(np.linalg.norm(p2 - p3)) / 2
    theta = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))

    c = math.sqrt(abs(a**2 - b**2))
    if a >= b:
        fx, fy = math.cos(theta), math.sin(theta)
    else:
        fx, fy = -math.sin(theta), math.cos(theta)

    coords: list[npt.NDArray] = [
        np.array([cx + c * fx, cy + c * fy], dtype=np.float32),
        np.array([cx - c * fx, cy - c * fy], dtype=np.float32),
    ]

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for deg in (30, 60, 120, 150, 210, 240, 300, 330):
        phi = math.radians(deg)
        ex = cx + a * math.cos(phi) * cos_t - b * math.sin(phi) * sin_t
        ey = cy + a * math.cos(phi) * sin_t + b * math.sin(phi) * cos_t
        coords.append(np.array([ex, ey], dtype=np.float32))

    # All 10 supportive points depend on all 4 scored landmarks
    sources = tuple((0, 1, 2, 3) for _ in range(10))
    return np.array(coords, dtype=np.float32), sources


# Dispatch table
_ComputeFn = Callable[[npt.NDArray], SupportiveResult]
_COMPUTE_FNS: dict[str, _ComputeFn | None] = {
    "A4C": _compute_a4c,
    "PLAX": _compute_plax,
    "PSAX": _compute_psax,
    "IVC": _compute_ivc,
    "AOP": None,
    "FUGC": _compute_rect_with_ext,
    "HC": _compute_ellipse,
    "FA": _compute_ellipse,
    "fetal_femur": _compute_rect_with_ext,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_supportive(task_id: str, gt: npt.NDArray) -> SupportiveResult:
    """Compute supportive landmarks from scored GT for a given task.

    Args:
        task_id: task identifier (e.g. "A4C", "HC")
        gt: scored GT landmarks, shape (K_scored, 2) in pixel coordinates.
            May contain NaN for missing landmarks.

    Returns:
        (coords, source_indices) where:
        - coords: (K_sup, 2) float32, may contain NaN for degenerate geometry
        - source_indices: per supportive point, tuple of scored indices it
          depends on (used for dependency-aware supervision in collate)
    """
    fn = _COMPUTE_FNS.get(task_id)
    if fn is None:
        return _EMPTY_RESULT
    return fn(gt)


def compute_evidence(
    pts: npt.NDArray,
    image: npt.NDArray,
    threshold: int = 5,
) -> npt.NDArray:
    """Per-landmark evidence: does the image have ultrasound signal at this location?

    Args:
        pts: landmark coordinates (K, 2) in pixel space.
        image: uint8 RGB image, shape (H, W, 3).
        threshold: 3×3 patch mean across all channels ≤ threshold → evidence=0.

    Returns:
        int array (K,) with values 0 or 1.
    """
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")
    h, w = image.shape[:2]
    ev = np.ones(len(pts), dtype=np.int64)
    for i, pt in enumerate(pts):
        if not np.isfinite(pt).all():
            ev[i] = 0
            continue
        px, py = int(round(float(pt[0]))), int(round(float(pt[1])))
        if px < 0 or px >= w or py < 0 or py >= h:
            ev[i] = 0
            continue
        # 3×3 patch, clipped at borders
        y0, y1 = max(0, py - 1), min(h, py + 2)
        x0, x1 = max(0, px - 1), min(w, px + 2)
        patch = image[y0:y1, x0:x1]
        if patch.mean() <= threshold:
            ev[i] = 0
    return ev

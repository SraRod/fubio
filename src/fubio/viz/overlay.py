"""Shared plotly overlay drawing for landmark geometry.

Consumes OverlayDef from task_registry. All notebooks import from here
instead of reimplementing distance / ellipse / angle dispatch.

Upstream: data/task_registry.py (OVERLAYS_BY_TASK).
Downstream: notebooks (instance_viewer, model_compare, submission_viewer, data_inspector).
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from fubio.data.task_registry import OVERLAYS_BY_TASK, OverlayDef


def add_roi_rect(
    fig: go.Figure,
    bbox: tuple[float, float, float, float] | list[float],
    *,
    color: str = "#00FF88",
    width: float = 2.0,
    dash: str = "dash",
) -> None:
    """Draw an ROI bounding box on *fig*. bbox is (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    fig.add_shape(
        type="rect",
        x0=x1,
        y0=y1,
        x1=x2,
        y1=y2,
        line=dict(color=color, width=width, dash=dash),
    )


def add_landmark_bbox(
    fig: go.Figure,
    points: np.ndarray,
    *,
    color: str = "#00FF88",
    width: float = 1.5,
    dash: str = "dash",
    padding: float = 0.03,
    img_hw: tuple[int, int] | None = None,
) -> None:
    """Draw a bounding box computed from visible (finite) landmarks.

    Mirrors viz_callback._draw_bbox: pad by *padding* fraction of image
    size, clamp to image bounds.
    """
    valid = np.isfinite(points).all(axis=1)
    if not valid.any():
        return
    pts = points[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    if img_hw is not None:
        h, w = img_hw
        pad_x, pad_y = w * padding, h * padding
        x0 = max(0, x_min - pad_x)
        y0 = max(0, y_min - pad_y)
        x1 = min(w, x_max + pad_x)
        y1 = min(h, y_max + pad_y)
    else:
        rx = (x_max - x_min) * padding
        ry = (y_max - y_min) * padding
        x0, y0 = x_min - rx, y_min - ry
        x1, y1 = x_max + rx, y_max + ry
    fig.add_shape(
        type="rect",
        x0=float(x0),
        y0=float(y0),
        x1=float(x1),
        y1=float(y1),
        line=dict(color=color, width=width, dash=dash),
    )


def add_overlays(
    fig: go.Figure,
    points: np.ndarray,
    task_id: str,
    *,
    color: str,
    width: float = 2.0,
    dash: str | None = None,
    legendgroup: str = "",
    showlegend: bool = False,
    show_annotations: bool = False,
    name_prefix: str = "",
    role_filter: str | None = None,
) -> None:
    """Draw all overlay geometry for *task_id* onto *fig*.

    Args:
        fig: Plotly figure (image-coordinate axes, y-axis reversed).
        points: (N, 2) array of task-local landmark coordinates in pixel space.
        task_id: Which task's overlays to draw.
        color: CSS color string for lines.
        width: Line width.
        dash: Plotly dash style ("dash", "dot", etc.). None = solid.
        legendgroup: Plotly legend group.
        showlegend: Whether to show legend entries.
        show_annotations: Whether to add value annotations (distance/angle text).
        name_prefix: Prefix for trace names (e.g. "I0 " for instance 0).
        role_filter: If set, only draw overlays with this role ("measurement" / "supportive").
    """
    overlays = OVERLAYS_BY_TASK.get(task_id, [])
    for ov in overlays:
        if role_filter is not None and ov.role != role_filter:
            continue
        _draw_one(
            fig,
            ov,
            points,
            color=color,
            width=width,
            dash=dash,
            legendgroup=legendgroup,
            showlegend=showlegend,
            show_annotations=show_annotations,
            name_prefix=name_prefix,
        )


def _draw_one(
    fig: go.Figure,
    ov: OverlayDef,
    points: np.ndarray,
    *,
    color: str,
    width: float,
    dash: str | None,
    legendgroup: str,
    showlegend: bool,
    show_annotations: bool,
    name_prefix: str,
) -> None:
    idx = ov.landmark_indices
    if not all(i < len(points) and np.isfinite(points[i]).all() for i in idx):
        return

    if ov.formula_type == "distance":
        _draw_distance(
            fig,
            ov,
            points,
            color=color,
            width=width,
            dash=dash,
            legendgroup=legendgroup,
            showlegend=showlegend,
            show_annotations=show_annotations,
            name_prefix=name_prefix,
        )
    elif ov.formula_type == "ellipse_circumference":
        _draw_ellipse(
            fig,
            ov,
            points,
            color=color,
            width=width,
            dash=dash,
            legendgroup=legendgroup,
            showlegend=showlegend,
            show_annotations=show_annotations,
            name_prefix=name_prefix,
        )
    elif ov.formula_type == "angle":
        _draw_angle(
            fig,
            ov,
            points,
            color=color,
            width=width,
            dash=dash,
            legendgroup=legendgroup,
            showlegend=showlegend,
            show_annotations=show_annotations,
            name_prefix=name_prefix,
        )


# ---------------------------------------------------------------------------
# Formula-type renderers
# ---------------------------------------------------------------------------


def _draw_distance(
    fig: go.Figure,
    ov: OverlayDef,
    pts: np.ndarray,
    *,
    color: str,
    width: float,
    dash: str | None,
    legendgroup: str,
    showlegend: bool,
    show_annotations: bool,
    name_prefix: str,
) -> None:
    a, b = ov.landmark_indices
    dist = float(np.sqrt(((pts[a] - pts[b]) ** 2).sum()))
    label = f"{name_prefix}{ov.name}"

    fig.add_trace(
        go.Scatter(
            x=[float(pts[a, 0]), float(pts[b, 0])],
            y=[float(pts[a, 1]), float(pts[b, 1])],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=f"{label} ({dist:.0f}px)" if show_annotations else label,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="text",
            text=[f"{ov.name}: {dist:.1f}px"] * 2,
        )
    )

    if show_annotations:
        mx = float((pts[a, 0] + pts[b, 0]) / 2)
        my = float((pts[a, 1] + pts[b, 1]) / 2)
        fig.add_annotation(
            x=mx,
            y=my,
            text=f"{dist:.0f}",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(0,0,0,0.5)",
        )


def _draw_ellipse(
    fig: go.Figure,
    ov: OverlayDef,
    pts: np.ndarray,
    *,
    color: str,
    width: float,
    dash: str | None,
    legendgroup: str,
    showlegend: bool,
    show_annotations: bool,
    name_prefix: str,
) -> None:
    i0, i1, i2, i3 = ov.landmark_indices
    a_len = float(np.sqrt(((pts[i0] - pts[i1]) ** 2).sum())) / 2
    b_len = float(np.sqrt(((pts[i2] - pts[i3]) ** 2).sum())) / 2

    # Axis lines
    for ea, eb in [(i0, i1), (i2, i3)]:
        fig.add_trace(
            go.Scatter(
                x=[float(pts[ea, 0]), float(pts[eb, 0])],
                y=[float(pts[ea, 1]), float(pts[eb, 1])],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                showlegend=False,
                legendgroup=legendgroup,
                hoverinfo="skip",
            )
        )

    # Ellipse outline
    if a_len > 0 and b_len > 0:
        cx = float((pts[i0, 0] + pts[i1, 0]) / 2)
        cy = float((pts[i0, 1] + pts[i1, 1]) / 2)
        dx = float(pts[i1, 0] - pts[i0, 0])
        dy = float(pts[i1, 1] - pts[i0, 1])
        angle = math.atan2(dy, dx)
        t = np.linspace(0, 2 * math.pi, 80)
        ca, sa = math.cos(angle), math.sin(angle)
        ex = cx + a_len * np.cos(t) * ca - b_len * np.sin(t) * sa
        ey = cy + a_len * np.cos(t) * sa + b_len * np.sin(t) * ca

        label = f"{name_prefix}{ov.name}"
        inner = (3 * a_len + b_len) * (a_len + 3 * b_len)
        circ = math.pi * (3 * (a_len + b_len) - math.sqrt(max(inner, 0)))

        fig.add_trace(
            go.Scatter(
                x=ex.tolist(),
                y=ey.tolist(),
                mode="lines",
                line=dict(color=color, width=max(width - 0.5, 1.0)),
                name=f"{label} (C={circ:.0f}px)" if show_annotations else label,
                legendgroup=legendgroup,
                showlegend=showlegend,
                hoverinfo="text",
                text=[f"{ov.name}: C={circ:.1f}px"] * len(t),
            )
        )


def _draw_angle(
    fig: go.Figure,
    ov: OverlayDef,
    pts: np.ndarray,
    *,
    color: str,
    width: float,
    dash: str | None,
    legendgroup: str,
    showlegend: bool,
    show_annotations: bool,
    name_prefix: str,
) -> None:
    iv, i1, i2 = ov.landmark_indices
    v = pts[iv]
    p1 = pts[i1]
    p2 = pts[i2]

    label = f"{name_prefix}{ov.name}"

    # Compute angle
    u1 = p1 - v
    u2 = p2 - v
    n1 = float(np.linalg.norm(u1))
    n2 = float(np.linalg.norm(u2))
    if n1 == 0 or n2 == 0:
        return
    cos_val = np.clip(np.dot(u1, u2) / (n1 * n2), -1, 1)
    ang_deg = float(np.degrees(np.arccos(cos_val)))

    # Ray lines
    fig.add_trace(
        go.Scatter(
            x=[float(v[0]), float(p1[0])],
            y=[float(v[1]), float(p1[1])],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=f"{label} ({ang_deg:.1f}°)" if show_annotations else label,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="text",
            text=[f"{ov.name}: {ang_deg:.1f}°"] * 2,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(v[0]), float(p2[0])],
            y=[float(v[1]), float(p2[1])],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )

    # Arc
    a1 = math.atan2(float(u1[1]), float(u1[0]))
    a2 = math.atan2(float(u2[1]), float(u2[0]))
    if (a2 - a1) % (2 * math.pi) > math.pi:
        a1, a2 = a2, a1
    arc_r = min(n1, n2) * 0.3
    ta = np.linspace(a1, a1 + (a2 - a1) % (2 * math.pi), 30)
    ax = float(v[0]) + arc_r * np.cos(ta)
    ay = float(v[1]) + arc_r * np.sin(ta)
    fig.add_trace(
        go.Scatter(
            x=ax.tolist(),
            y=ay.tolist(),
            mode="lines",
            line=dict(color=color, width=max(width - 0.5, 1.0)),
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )

    if show_annotations:
        mid_a = (a1 + a2) / 2 if a2 > a1 else a1 + ((a2 - a1) % (2 * math.pi)) / 2
        fig.add_annotation(
            x=float(v[0] + arc_r * 1.3 * math.cos(mid_a)),
            y=float(v[1] + arc_r * 1.3 * math.sin(mid_a)),
            text=f"{ang_deg:.1f}°",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(0,0,0,0.5)",
        )

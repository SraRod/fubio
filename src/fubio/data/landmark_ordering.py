"""Landmark ordering engine — bidirectional competition ↔ canonical conversion.

All operations use original-image pixel coordinates (x right-increasing,
y down-increasing). Do NOT apply in augmented or transformed coordinate space.

Upstream: ordering_schema.py (data models), task_registry.py (TASKS).
Downstream: build_manifest.py (intake), predict.py (submission output).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fubio.data.ordering_schema import (
    OrderingDiagnostic,
    OrderingResult,
    OrderingStatus,
    OrderingValidation,
    OrderingViolation,
    PairAxisRule,
)
from fubio.data.task_registry import TASKS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permutation utilities
# ---------------------------------------------------------------------------


def apply_permutation(
    values: NDArray,
    permutation: NDArray[np.intp],
    *,
    axis: int = 0,
) -> NDArray:
    """Reorder `values` along `axis` using gather permutation.

    Satisfies: result[i] == values[permutation[i]] (along `axis`).
    """
    return np.take(values, permutation, axis=axis)


def invert_permutation(permutation: NDArray[np.intp]) -> NDArray[np.intp]:
    """Compute inverse: if P maps A→B, inverse maps B→A."""
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(permutation))
    return inv


# ---------------------------------------------------------------------------
# Shared comparator — used by both apply and validate
# ---------------------------------------------------------------------------

PairDecision = Literal["swap", "ok", "tie", "nan"]


def _compare_pair(
    points: NDArray[np.floating],
    idx_a: int,
    idx_b: int,
    rule: PairAxisRule,
) -> tuple[PairDecision, float | None, int | None]:
    """Lexicographic comparison of a landmark pair.

    Returns (decision, delta, key_index):
    - decision: "swap" | "ok" | "tie" | "nan"
    - delta: the deciding axis delta (after rounding), None if NaN
    - key_index: which key decided (None if NaN or all-tied)
    """
    for ki, key in enumerate(rule.keys):
        axis_col = 0 if key.axis == "x" else 1
        va = float(points[idx_a, axis_col])
        vb = float(points[idx_b, axis_col])

        if np.isnan(va) or np.isnan(vb):
            return "nan", None, None

        if key.round_digits is not None:
            va = round(va, key.round_digits)
            vb = round(vb, key.round_digits)

        delta = vb - va

        if abs(delta) <= key.tolerance:
            continue

        need_swap = (key.order == "asc" and delta < 0) or (
            key.order == "desc" and delta > 0
        )
        return ("swap" if need_swap else "ok"), delta, ki

    return "tie", None, None


def _resolve_pair_indices(
    rule: PairAxisRule,
    landmark_names: tuple[str, ...],
) -> tuple[int, int]:
    """Map rule's named pair to integer indices."""
    return (
        landmark_names.index(rule.pair[0]),
        landmark_names.index(rule.pair[1]),
    )


# ---------------------------------------------------------------------------
# Core engine — apply rules to points
# ---------------------------------------------------------------------------


def _apply_rules(
    points: NDArray[np.floating],
    rules: tuple[PairAxisRule, ...],
    landmark_names: tuple[str, ...],
) -> OrderingResult:
    """Apply ordering rules to a single sample's points.

    Points shape: (K, 2) where col 0=x, col 1=y.
    Returns OrderingResult with gather-convention permutation.
    """
    k = len(landmark_names)

    if not rules:
        perm = np.arange(k, dtype=np.intp)
        return OrderingResult(
            points=np.array(points, copy=True),
            permutation=perm,
            changed=False,
            status="identity",
        )

    out = np.array(points, copy=True)
    perm = np.arange(k, dtype=np.intp)
    diagnostics: list[OrderingDiagnostic] = []
    any_swap = False
    any_ambiguous = False

    for ri, rule in enumerate(rules):
        idx_a, idx_b = _resolve_pair_indices(rule, landmark_names)
        decision, delta, key_index = _compare_pair(out, idx_a, idx_b, rule)

        if decision == "nan":
            diagnostics.append(
                OrderingDiagnostic(
                    rule_index=ri,
                    code="nan",
                    landmark_names=rule.pair,
                    delta=None,
                    key_index=None,
                )
            )
            any_ambiguous = True

        elif decision == "tie":
            diagnostics.append(
                OrderingDiagnostic(
                    rule_index=ri,
                    code="tie",
                    landmark_names=rule.pair,
                    delta=delta,
                    key_index=None,
                )
            )
            any_ambiguous = True

        elif decision == "swap":
            out[[idx_a, idx_b]] = out[[idx_b, idx_a]]
            perm[[idx_a, idx_b]] = perm[[idx_b, idx_a]]
            any_swap = True
            diagnostics.append(
                OrderingDiagnostic(
                    rule_index=ri,
                    code="swap",
                    landmark_names=rule.pair,
                    delta=delta,
                    key_index=key_index,
                )
            )

        else:
            diagnostics.append(
                OrderingDiagnostic(
                    rule_index=ri,
                    code="ok",
                    landmark_names=rule.pair,
                    delta=delta,
                    key_index=key_index,
                )
            )

    status: OrderingStatus = "ambiguous" if any_ambiguous else "resolved"

    return OrderingResult(
        points=out,
        permutation=perm,
        changed=any_swap,
        status=status,
        diagnostics=tuple(diagnostics),
    )


def _validate_against_rules(
    points: NDArray[np.floating],
    rules: tuple[PairAxisRule, ...],
    landmark_names: tuple[str, ...],
) -> OrderingValidation:
    """Check whether points satisfy ordering rules without modifying them."""
    if not rules:
        return OrderingValidation(valid=True)

    violations: list[OrderingViolation] = []
    any_ambiguous = False

    for rule in rules:
        idx_a, idx_b = _resolve_pair_indices(rule, landmark_names)
        decision, delta, key_index = _compare_pair(points, idx_a, idx_b, rule)

        if decision == "nan":
            violations.append(
                OrderingViolation(
                    pair=rule.pair,
                    axis=rule.keys[0].axis,
                    expected_order=rule.keys[0].order,
                    actual_delta=float("nan"),
                    code="nan",
                    key_index=None,
                )
            )
            any_ambiguous = True

        elif decision == "tie":
            violations.append(
                OrderingViolation(
                    pair=rule.pair,
                    axis=rule.keys[0].axis,
                    expected_order=rule.keys[0].order,
                    actual_delta=0.0,
                    code="tie",
                    key_index=None,
                )
            )
            any_ambiguous = True

        elif decision == "swap":
            deciding_key = rule.keys[key_index] if key_index is not None else rule.keys[0]
            violations.append(
                OrderingViolation(
                    pair=rule.pair,
                    axis=deciding_key.axis,
                    expected_order=deciding_key.order,
                    actual_delta=delta if delta is not None else 0.0,
                    code="reversed",
                    key_index=key_index,
                )
            )

    return OrderingValidation(
        valid=len(violations) == 0,
        ambiguous=any_ambiguous,
        violations=tuple(violations),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def canonicalize(
    points: NDArray[np.floating],
    task_id: str,
) -> OrderingResult:
    """Competition-order original-pixel points → canonical-order points.

    Gather convention: result.points == input_points[result.permutation]
    """
    task = TASKS[task_id]
    if points.shape != (task.n_keypoints, 2):
        raise ValueError(f"Expected shape ({task.n_keypoints}, 2), got {points.shape}")
    return _apply_rules(
        points,
        task.ordering.canonical_rules,
        task.landmark_names,
    )


def competitionize(
    points: NDArray[np.floating],
    task_id: str,
) -> OrderingResult:
    """Canonical-order original-pixel points → competition-order points.

    Gather convention: result.points == input_points[result.permutation]
    """
    task = TASKS[task_id]
    if points.shape != (task.n_keypoints, 2):
        raise ValueError(f"Expected shape ({task.n_keypoints}, 2), got {points.shape}")
    return _apply_rules(
        points,
        task.ordering.competition_rules,
        task.effective_competition_names,
    )


def validate_canonical(
    points: NDArray[np.floating],
    task_id: str,
) -> OrderingValidation:
    """Check whether points satisfy canonical ordering constraints."""
    task = TASKS[task_id]
    return _validate_against_rules(
        points,
        task.ordering.canonical_rules,
        task.landmark_names,
    )


def validate_competition(
    points: NDArray[np.floating],
    task_id: str,
) -> OrderingValidation:
    """Check whether points satisfy competition ordering constraints."""
    task = TASKS[task_id]
    return _validate_against_rules(
        points,
        task.ordering.competition_rules,
        task.effective_competition_names,
    )

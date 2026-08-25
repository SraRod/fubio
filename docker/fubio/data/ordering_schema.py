"""Landmark ordering data models — pure schema, no registry dependency.

Defines the rule and result types for bidirectional landmark reordering
between competition (organizer CSV) and canonical (training) orderings.

Upstream: none (leaf dependency).
Downstream: task_registry.py (imports spec types), landmark_ordering.py (imports all).
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Ordering rules — lexicographic key-based comparison
# ---------------------------------------------------------------------------

OrderingStatus = Literal["identity", "resolved", "ambiguous"]


class AxisOrderingKey(BaseModel):
    """One comparison key in a lexicographic pair ordering.

    Keys are evaluated left-to-right: the first key that produces a
    non-tie result decides the swap. Only if ALL keys tie is the
    outcome ambiguous.
    """

    model_config = ConfigDict(frozen=True)

    axis: Literal["x", "y"]
    order: Literal["asc", "desc"]
    round_digits: int | None = Field(default=None, ge=0)
    tolerance: float = Field(default=0.0, ge=0.0)


class PairAxisRule(BaseModel):
    """Swap a pair of landmarks based on lexicographic axis comparison.

    Operates in original-image pixel coordinates (x right-increasing,
    y down-increasing). The rule compares the two named landmarks using
    `keys` in order and swaps them if they violate the specified ordering.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["pair_axis"] = "pair_axis"
    pair: tuple[str, str]
    keys: tuple[AxisOrderingKey, ...]

    @model_validator(mode="after")
    def _check_keys(self) -> PairAxisRule:
        if not self.keys:
            raise ValueError("keys must have at least one entry")
        axes = [k.axis for k in self.keys]
        if len(axes) != len(set(axes)):
            raise ValueError("ordering keys must use distinct axes")
        return self


# Discriminated union — extend with GroupAxisRule etc. in the future.
OrderingRule = Annotated[
    PairAxisRule,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Per-task ordering specification
# ---------------------------------------------------------------------------


class TaskOrderingSpec(BaseModel):
    """Declarative ordering spec for one task.

    canonical_rules: applied during intake to normalize GT ordering.
    competition_rules: applied during submission to restore organizer ordering.
    Empty tuples = identity (no reordering).

    Round-trip properties (O(C(x))==x, C(O(x))==x) are directional and
    task-specific — validated in tests, not enforced here.
    """

    model_config = ConfigDict(frozen=True)

    canonical_rules: tuple[OrderingRule, ...] = ()
    competition_rules: tuple[OrderingRule, ...] = ()

    @model_validator(mode="after")
    def _validate_disjoint_pairs(self) -> TaskOrderingSpec:
        """Each landmark may appear in at most one rule per side."""
        for side_name, rules in [
            ("canonical_rules", self.canonical_rules),
            ("competition_rules", self.competition_rules),
        ]:
            seen: set[str] = set()
            for rule in rules:
                a, b = rule.pair
                if a == b:
                    raise ValueError(f"{side_name}: pair ({a!r}, {b!r}) is identity")
                if a in seen or b in seen:
                    raise ValueError(
                        f"{side_name}: landmark {a!r} or {b!r} appears in "
                        f"multiple rules (overlapping pairs not supported)"
                    )
                seen.update((a, b))
        return self


# ---------------------------------------------------------------------------
# Ordering results and diagnostics
# ---------------------------------------------------------------------------


class OrderingDiagnostic(BaseModel):
    """Per-rule diagnostic from an ordering operation."""

    model_config = ConfigDict(frozen=True)

    rule_index: int
    code: Literal["swap", "ok", "tie", "nan"]
    landmark_names: tuple[str, str]
    delta: float | None = None
    key_index: int | None = None


class OrderingResult(BaseModel):
    """Result of canonicalize() or competitionize().

    Permutation uses gather convention:
        result.points == input_points[result.permutation]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    points: NDArray[np.floating]
    permutation: NDArray[np.intp]
    changed: bool
    status: OrderingStatus
    diagnostics: tuple[OrderingDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _freeze_arrays(self) -> OrderingResult:
        self.points.setflags(write=False)
        self.permutation.setflags(write=False)
        return self


class OrderingViolation(BaseModel):
    """One violated ordering constraint."""

    model_config = ConfigDict(frozen=True)

    pair: tuple[str, str]
    axis: Literal["x", "y"]
    expected_order: Literal["asc", "desc"]
    actual_delta: float
    code: Literal["reversed", "tie", "nan"]
    key_index: int | None = None


class OrderingValidation(BaseModel):
    """Result of validate_canonical() or validate_competition()."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    ambiguous: bool = False
    violations: tuple[OrderingViolation, ...] = ()

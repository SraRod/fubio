"""Canonical K=60 landmark schema — single source of truth.

Defines all 9 tasks, 60 landmark slots, 28 clinical parameters,
overlay geometry (superset of params + supportive), and per-axis flip permutations.
All downstream code (manifest, dataset, model, eval, viz) imports from here.

Upstream: ordering_schema.py (ordering data models).
Downstream: build_manifest.py, manifest.py, transforms.py, collate.py, eval, viz.
"""

from __future__ import annotations

import typing

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from fubio.data.ordering_schema import AxisOrderingKey, PairAxisRule, TaskOrderingSpec

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

K = 60

# Valid shape types for overlay geometry.  New types (e.g. "polyline",
# "circle") should be added here and handled in the viz drawing layer.
FORMULA_ARITY: dict[str, int] = {
    "distance": 2,
    "ellipse_circumference": 4,
    "angle": 3,
}

OverlayRole = typing.Literal["measurement", "supportive"]


class TaskDef(BaseModel):
    """Immutable task definition with explicit offset and task_int."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    task_int: int
    domain: str
    n_keypoints: int
    landmark_names: tuple[str, ...]
    offset: int

    # Landmark index swap pairs per flip axis.  All tasks allow both hflip
    # and vflip; these lists control which scored-landmark slots swap.
    hflip_pairs: tuple[tuple[int, int], ...] = ()
    vflip_pairs: tuple[tuple[int, int], ...] = ()

    # Multiplicative scale on landmark bbox side length to define the context
    # region for matching/bbox loss. Landmarks are internal to the anatomy —
    # the model needs surrounding context to identify them. 1.0 = tight bbox
    # + standard 5% canvas padding. >1.0 = proportionally larger. Only set
    # for tasks where the landmark ROI is far smaller than the visual context
    # needed for identification.
    bbox_context_scale: float = 1.0

    # --- Supportive landmark metadata (training-only, not submitted) ---
    n_supportive: int = 0
    supportive_names: tuple[str, ...] = ()
    # Swap pairs for supportive landmarks per flip axis (indices local to
    # supportive array, i.e. 0-indexed within the n_supportive range)
    supportive_hflip_pairs: tuple[tuple[int, int], ...] = ()
    supportive_vflip_pairs: tuple[tuple[int, int], ...] = ()

    # --- Landmark ordering (competition ↔ canonical) ---
    # Organizer CSV slot names. None = same as landmark_names.
    competition_landmark_names: tuple[str, ...] | None = None
    ordering: TaskOrderingSpec = TaskOrderingSpec()

    @property
    def n_total(self) -> int:
        """Total landmarks: scored + supportive."""
        return self.n_keypoints + self.n_supportive

    @property
    def scored_mask(self) -> tuple[bool, ...]:
        """Per-landmark flag: True for scored (submitted), False for supportive."""
        return tuple([True] * self.n_keypoints + [False] * self.n_supportive)

    @property
    def effective_competition_names(self) -> tuple[str, ...]:
        """Competition landmark names — falls back to landmark_names."""
        return self.competition_landmark_names or self.landmark_names

    @property
    def csv_landmark_columns(self) -> tuple[str, ...]:
        """CSV column names for landmark coordinates, derived from competition names."""
        return tuple(f"{name}_xy" for name in self.effective_competition_names)

    @model_validator(mode="after")
    def _check_landmark_count(self) -> TaskDef:
        if len(self.landmark_names) != self.n_keypoints:
            raise ValueError(
                f"{self.task_id}: landmark_names has {len(self.landmark_names)} entries, "
                f"expected {self.n_keypoints}"
            )
        return self

    @model_validator(mode="after")
    def _check_supportive(self) -> TaskDef:
        if len(self.supportive_names) != self.n_supportive:
            raise ValueError(
                f"{self.task_id}: supportive_names has {len(self.supportive_names)} entries, "
                f"expected {self.n_supportive}"
            )
        for axis_name, pairs in [
            ("supportive_hflip_pairs", self.supportive_hflip_pairs),
            ("supportive_vflip_pairs", self.supportive_vflip_pairs),
        ]:
            seen: set[int] = set()
            for a, b in pairs:
                if a >= self.n_supportive or b >= self.n_supportive:
                    raise ValueError(
                        f"{self.task_id}: {axis_name} ({a},{b}) "
                        f"out of range [0,{self.n_supportive})"
                    )
                if a == b:
                    raise ValueError(f"{self.task_id}: {axis_name} ({a},{b}) is identity")
                if a in seen or b in seen:
                    raise ValueError(
                        f"{self.task_id}: {axis_name} ({a},{b}) overlaps with another pair"
                    )
                seen.update((a, b))
        return self

    @model_validator(mode="after")
    def _check_ordering(self) -> TaskDef:
        comp = self.effective_competition_names
        if len(comp) != self.n_keypoints:
            raise ValueError(
                f"{self.task_id}: competition_landmark_names has {len(comp)} entries, "
                f"expected {self.n_keypoints}"
            )
        if set(comp) != set(self.landmark_names):
            raise ValueError(
                f"{self.task_id}: competition_landmark_names and landmark_names "
                f"must be the same set"
            )
        if len(set(comp)) != len(comp):
            raise ValueError(f"{self.task_id}: competition_landmark_names has duplicates")
        # Verify ordering rule names exist in respective landmark name sets
        for rule in self.ordering.canonical_rules:
            for name in rule.pair:
                if name not in self.landmark_names:
                    raise ValueError(
                        f"{self.task_id}: canonical rule references unknown landmark {name!r}"
                    )
        for rule in self.ordering.competition_rules:
            for name in rule.pair:
                if name not in comp:
                    raise ValueError(
                        f"{self.task_id}: competition rule references unknown landmark {name!r}"
                    )
        return self


class ParamDef(BaseModel):
    """Clinical parameter derived from landmarks. Formula arity validated."""

    model_config = ConfigDict(frozen=True)

    name: str
    task_id: str
    formula_type: str
    landmark_indices: list[int]
    has_gt: bool

    @model_validator(mode="after")
    def _check_arity(self) -> ParamDef:
        if self.formula_type not in FORMULA_ARITY:
            raise ValueError(f"{self.name}: unknown formula_type '{self.formula_type}'")
        n_expected = FORMULA_ARITY[self.formula_type]
        if len(self.landmark_indices) != n_expected:
            raise ValueError(
                f"{self.name}: {self.formula_type} expects {n_expected} indices, "
                f"got {len(self.landmark_indices)}"
            )
        return self


class OverlayDef(BaseModel):
    """Drawable geometry between landmarks — superset of clinical params.

    role="measurement": derived from a ParamDef, scored in evaluation.
    role="supportive": visual aid only, not scored (e.g. reference lines).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    task_id: str
    formula_type: str
    landmark_indices: list[int]
    role: OverlayRole

    @model_validator(mode="after")
    def _check_arity(self) -> OverlayDef:
        if self.formula_type not in FORMULA_ARITY:
            raise ValueError(f"{self.name}: unknown formula_type '{self.formula_type}'")
        n_expected = FORMULA_ARITY[self.formula_type]
        if len(self.landmark_indices) != n_expected:
            raise ValueError(
                f"{self.name}: {self.formula_type} expects {n_expected} indices, "
                f"got {len(self.landmark_indices)}"
            )
        return self


# ---------------------------------------------------------------------------
# Reusable ordering keys and rule builders
# ---------------------------------------------------------------------------

_X_ASC = AxisOrderingKey(axis="x", order="asc")
_X_DESC = AxisOrderingKey(axis="x", order="desc")
_Y_ASC = AxisOrderingKey(axis="y", order="asc")
_Y_ASC_ROUND0 = AxisOrderingKey(axis="y", order="asc", round_digits=0)


def _pair_x_asc(a: str, b: str) -> PairAxisRule:
    return PairAxisRule(pair=(a, b), keys=(_X_ASC,))


def _pair_y_asc(a: str, b: str) -> PairAxisRule:
    return PairAxisRule(pair=(a, b), keys=(_Y_ASC,))


def _pair_x_desc(a: str, b: str) -> PairAxisRule:
    return PairAxisRule(pair=(a, b), keys=(_X_DESC,))


def _pair_y_round_x(a: str, b: str) -> PairAxisRule:
    """Y-ascending (integer-rounded) with X-ascending tie-break."""
    return PairAxisRule(pair=(a, b), keys=(_Y_ASC_ROUND0, _X_ASC))


# ---------------------------------------------------------------------------
# Task definitions — offsets and task_int are EXPLICIT LITERALS
# ---------------------------------------------------------------------------

_TASK_DEFS: tuple[TaskDef, ...] = (
    TaskDef(
        task_id="A4C",
        task_int=0,
        domain="cardiac",
        n_keypoints=16,
        offset=0,
        landmark_names=(
            "LV_上下_p1",
            "LV_上下_p2",
            "LV_左右_p1",
            "LV_左右_p2",
            "RV_上下_p1",
            "RV_上下_p2",
            "RV_左右_p1",
            "RV_左右_p2",
            "LA_上下_p1",
            "LA_上下_p2",
            "LA_左右_p1",
            "LA_左右_p2",
            "RA_上下_p1",
            "RA_上下_p2",
            "RA_左右_p1",
            "RA_左右_p2",
        ),
        hflip_pairs=(),
        vflip_pairs=(),
        # 4 chambers × 5 sup (cross + 4 midpoints) = 20
        n_supportive=20,
        supportive_names=(
            "LV_cross",
            "LV_mid_上左",
            "LV_mid_左下",
            "LV_mid_下右",
            "LV_mid_右上",
            "RV_cross",
            "RV_mid_上左",
            "RV_mid_左下",
            "RV_mid_下右",
            "RV_mid_右上",
            "LA_cross",
            "LA_mid_上左",
            "LA_mid_左下",
            "LA_mid_下右",
            "LA_mid_右上",
            "RA_cross",
            "RA_mid_上左",
            "RA_mid_左下",
            "RA_mid_下右",
            "RA_mid_右上",
        ),
        supportive_hflip_pairs=(),
        supportive_vflip_pairs=(),
        ordering=TaskOrderingSpec(
            canonical_rules=(
                _pair_x_asc("LV_左右_p1", "LV_左右_p2"),
                _pair_x_asc("RV_左右_p1", "RV_左右_p2"),
                _pair_x_asc("LA_左右_p1", "LA_左右_p2"),
                _pair_x_asc("RA_左右_p1", "RA_左右_p2"),
            ),
            competition_rules=(
                _pair_y_round_x("LV_左右_p1", "LV_左右_p2"),
                _pair_y_round_x("RV_左右_p1", "RV_左右_p2"),
                _pair_y_round_x("LA_左右_p1", "LA_左右_p2"),
                _pair_y_round_x("RA_左右_p1", "RA_左右_p2"),
            ),
        ),
    ),
    TaskDef(
        task_id="PLAX",
        task_int=1,
        domain="cardiac",
        n_keypoints=22,
        offset=16,
        landmark_names=(
            "LV_p1",
            "LV_p2",
            "RV_p1",
            "RV_p2",
            "IVS_p1",
            "IVS_p2",
            "LVPW_p1",
            "LVPW_p2",
            "VAO_p1",
            "VAO_p2",
            "STJ_p1",
            "STJ_p2",
            "AAO_p1",
            "AAO_p2",
            "AV_p1",
            "AV_p2",
            "LVOT_p1",
            "LVOT_p2",
            "LA_p1",
            "LA_p2",
            "RVOT_p1",
            "RVOT_p2",
        ),
        # 11 measurement-line midpoints
        n_supportive=11,
        supportive_names=(
            "LV_mid",
            "RV_mid",
            "IVS_mid",
            "LVPW_mid",
            "VAO_mid",
            "STJ_mid",
            "AAO_mid",
            "AV_mid",
            "LVOT_mid",
            "LA_mid",
            "RVOT_mid",
        ),
    ),
    TaskDef(
        task_id="PSAX",
        task_int=2,
        domain="cardiac",
        n_keypoints=4,
        offset=38,
        landmark_names=("RVOT_p1", "RVOT_p2", "PA_p1", "PA_p2"),
        bbox_context_scale=3.0,
        # CTR (centroid) + mid_R2P2 (bifurcation) + mid_extX (convergence)
        n_supportive=3,
        supportive_names=(
            "CTR",
            "mid_R2P2",
            "mid_extX",
        ),
        ordering=TaskOrderingSpec(
            canonical_rules=(_pair_x_asc("PA_p1", "PA_p2"),),
            competition_rules=(_pair_y_round_x("PA_p1", "PA_p2"),),
        ),
    ),
    TaskDef(
        task_id="IVC",
        task_int=3,
        domain="cardiac",
        n_keypoints=2,
        offset=42,
        landmark_names=("IVC_p1", "IVC_p2"),
        # no vflip swap — force model to learn proximal vs distal IVC by anatomy
        bbox_context_scale=8.0,
        # 2 extensions + 4 perpendiculars (bilateral)
        n_supportive=6,
        supportive_names=(
            "ext_beyond_p1",
            "ext_beyond_p2",
            "perp_at_p1_L",
            "perp_at_p2_L",
            "perp_at_p1_R",
            "perp_at_p2_R",
        ),
        # No ordering rules: CAN=identity, COMP=identity.
        # IVC was never canonicalized — model learns CSV order directly,
        # which is already competition order. Applying y_round_x would
        # risk swapping on pred/GT y-disagreement (same failure mode as PSAX).
    ),
    TaskDef(
        task_id="AOP",
        task_int=4,
        domain="intrapartum",
        n_keypoints=4,
        offset=44,
        landmark_names=("point_1", "point_2", "point_3", "point_4"),
    ),
    TaskDef(
        task_id="FUGC",
        task_int=5,
        domain="intrapartum",
        n_keypoints=2,
        offset=48,
        landmark_names=("point_1", "point_2"),
        # no hflip swap — force model to learn internal vs external os by anatomy
        # 2 rect_proj + 2 rect_ext (0.25× extension)
        n_supportive=4,
        supportive_names=(
            "rect_proj_1",
            "rect_proj_2",
            "rect_ext_1",
            "rect_ext_2",
        ),
        ordering=TaskOrderingSpec(
            canonical_rules=(_pair_x_asc("point_1", "point_2"),),
            competition_rules=(_pair_x_asc("point_1", "point_2"),),
        ),
    ),
    TaskDef(
        task_id="HC",
        task_int=6,
        domain="prenatal",
        n_keypoints=4,
        offset=50,
        # P0-P1 = short axis, P2-P3 = long axis (GT: 100% consistent)
        landmark_names=("point_1", "point_2", "point_3", "point_4"),
        # 2 foci + 8 ellipse samples (30°–330° at 30° steps, excluding 0°/90°/180°/270°)
        n_supportive=10,
        supportive_names=(
            "focus_1",
            "focus_2",
            "ellipse_30",
            "ellipse_60",
            "ellipse_120",
            "ellipse_150",
            "ellipse_210",
            "ellipse_240",
            "ellipse_300",
            "ellipse_330",
        ),
    ),
    TaskDef(
        task_id="FA",
        task_int=7,
        domain="prenatal",
        n_keypoints=4,
        offset=54,
        # Near-circular (ratio ~0.96); P0-P1 vs P2-P3 長短不穩定 (33% P0-P1 > P2-P3)
        landmark_names=("point_1", "point_2", "point_3", "point_4"),
        # Same structure as HC: 2 foci + 8 ellipse samples
        n_supportive=10,
        supportive_names=(
            "focus_1",
            "focus_2",
            "ellipse_30",
            "ellipse_60",
            "ellipse_120",
            "ellipse_150",
            "ellipse_210",
            "ellipse_240",
            "ellipse_300",
            "ellipse_330",
        ),
        ordering=TaskOrderingSpec(
            # p1-p2 vertical axis: CSV 100% y-asc (p1=top)
            # p3-p4 horizontal axis: CSV 100% x-desc (p3=right)
            competition_rules=(
                _pair_y_asc("point_1", "point_2"),
                _pair_x_desc("point_3", "point_4"),
            ),
        ),
    ),
    TaskDef(
        task_id="fetal_femur",
        task_int=8,
        domain="prenatal",
        n_keypoints=2,
        offset=58,
        landmark_names=("point_1", "point_2"),
        # no hflip swap — force model to learn proximal vs distal femur by anatomy
        bbox_context_scale=4.0,
        # Same structure as FUGC: rect_proj + rect_ext (0.25×)
        n_supportive=4,
        supportive_names=(
            "rect_proj_1",
            "rect_proj_2",
            "rect_ext_1",
            "rect_ext_2",
        ),
        ordering=TaskOrderingSpec(
            canonical_rules=(_pair_x_asc("point_1", "point_2"),),
            competition_rules=(_pair_x_asc("point_1", "point_2"),),
        ),
    ),
)

TASKS: dict[str, TaskDef] = {t.task_id: t for t in _TASK_DEFS}


# ---------------------------------------------------------------------------
# Parameter definitions — all 28 clinical parameters
# ---------------------------------------------------------------------------

_PARAM_DEFS: tuple[ParamDef, ...] = (
    # A4C — 8 distances (4 chambers × long/trans)
    ParamDef(
        name="A4C_LV_long",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[0, 1],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_LV_trans",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[2, 3],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_RV_long",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[4, 5],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_RV_trans",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[6, 7],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_LA_long",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[8, 9],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_LA_trans",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[10, 11],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_RA_long",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[12, 13],
        has_gt=False,
    ),
    ParamDef(
        name="A4C_RA_trans",
        task_id="A4C",
        formula_type="distance",
        landmark_indices=[14, 15],
        has_gt=False,
    ),
    # PLAX — 11 distances
    ParamDef(
        name="PLAX_LV",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[0, 1],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_RV",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[2, 3],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_IVS",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[4, 5],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_LVPW",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[6, 7],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_VAO",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[8, 9],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_STJ",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[10, 11],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_AAO",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[12, 13],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_AV",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[14, 15],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_LVOT",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[16, 17],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_LA",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[18, 19],
        has_gt=False,
    ),
    ParamDef(
        name="PLAX_RVOT",
        task_id="PLAX",
        formula_type="distance",
        landmark_indices=[20, 21],
        has_gt=False,
    ),
    # PSAX — 2 distances
    ParamDef(
        name="PSAX_RVOT",
        task_id="PSAX",
        formula_type="distance",
        landmark_indices=[0, 1],
        has_gt=True,
    ),
    ParamDef(
        name="PSAX_PA",
        task_id="PSAX",
        formula_type="distance",
        landmark_indices=[2, 3],
        has_gt=True,
    ),
    # IVC — 1 distance
    ParamDef(
        name="IVC_diameter",
        task_id="IVC",
        formula_type="distance",
        landmark_indices=[0, 1],
        has_gt=True,
    ),
    # AOP — 1 angle + 1 distance
    ParamDef(
        name="AOP_angle",
        task_id="AOP",
        formula_type="angle",
        landmark_indices=[0, 1, 3],
        has_gt=True,
    ),
    ParamDef(
        name="AOP_HSD", task_id="AOP", formula_type="distance", landmark_indices=[0, 2], has_gt=True
    ),
    # FUGC — 1 distance
    ParamDef(
        name="FUGC_CL",
        task_id="FUGC",
        formula_type="distance",
        landmark_indices=[0, 1],
        has_gt=False,
    ),
    # HC — 1 ellipse circumference
    ParamDef(
        name="HC_circumference",
        task_id="HC",
        formula_type="ellipse_circumference",
        landmark_indices=[0, 1, 2, 3],
        has_gt=False,
    ),
    # FA — 1 ellipse circumference
    ParamDef(
        name="FA_circumference",
        task_id="FA",
        formula_type="ellipse_circumference",
        landmark_indices=[0, 1, 2, 3],
        has_gt=True,
    ),
    # fetal_femur — 1 distance
    ParamDef(
        name="FL_length",
        task_id="fetal_femur",
        formula_type="distance",
        landmark_indices=[0, 1],
        has_gt=False,
    ),
)

PARAMS: dict[str, ParamDef] = {p.name: p for p in _PARAM_DEFS}


# ---------------------------------------------------------------------------
# Overlay definitions — all drawable geometry (params + supportive)
#
# Measurement overlays are auto-generated from PARAMS.
# Supportive overlays (reference lines, construction aids) go in
# _SUPPORTIVE_OVERLAY_DEFS — adding one is purely additive.
# ---------------------------------------------------------------------------

_SUPPORTIVE_OVERLAY_DEFS: tuple[OverlayDef, ...] = (
    # Add supportive geometry here, e.g.:
    # OverlayDef(
    #     name="A4C_LV_diagonal",
    #     task_id="A4C",
    #     formula_type="distance",
    #     landmark_indices=[0, 3],
    #     role="supportive",
    # ),
)

_MEASUREMENT_OVERLAYS: tuple[OverlayDef, ...] = tuple(
    OverlayDef(
        name=p.name,
        task_id=p.task_id,
        formula_type=p.formula_type,
        landmark_indices=p.landmark_indices,
        role="measurement",
    )
    for p in _PARAM_DEFS
)

_ALL_OVERLAYS = _MEASUREMENT_OVERLAYS + _SUPPORTIVE_OVERLAY_DEFS

OVERLAYS_BY_TASK: dict[str, list[OverlayDef]] = {}
for _ov in _ALL_OVERLAYS:
    OVERLAYS_BY_TASK.setdefault(_ov.task_id, []).append(_ov)


# ---------------------------------------------------------------------------
# Registry validation — runs at import time
# ---------------------------------------------------------------------------


def _validate_registry() -> None:
    """Assert offset coverage, task_int uniqueness, and param consistency."""
    # Offsets cover [0, K) with no gaps or overlaps
    covered: set[int] = set()
    for t in _TASK_DEFS:
        rng = set(range(t.offset, t.offset + t.n_keypoints))
        overlap = covered & rng
        if overlap:
            raise ValueError(f"Offset overlap at {overlap}")
        covered |= rng
    if covered != set(range(K)):
        raise ValueError(f"Offset gaps: {set(range(K)) - covered}")

    # task_int unique
    ints = [t.task_int for t in _TASK_DEFS]
    if len(ints) != len(set(ints)):
        raise ValueError(f"Duplicate task_int values: {ints}")

    # Flip pair indices in range and self-inverse
    for t in _TASK_DEFS:
        for axis_name, pairs in [("hflip_pairs", t.hflip_pairs), ("vflip_pairs", t.vflip_pairs)]:
            for a, b in pairs:
                if a >= t.n_keypoints or b >= t.n_keypoints:
                    raise ValueError(
                        f"{t.task_id}: {axis_name} ({a},{b}) out of range [0,{t.n_keypoints})"
                    )
                if a == b:
                    raise ValueError(f"{t.task_id}: {axis_name} ({a},{b}) is identity")

    # Supportive totals
    total_sup = sum(t.n_supportive for t in _TASK_DEFS)
    total_all = K + total_sup
    if total_all != 128:
        raise ValueError(f"Expected K + K_sup = 128, got {total_all}")

    # ParamDef consistency
    for p in _PARAM_DEFS:
        if p.task_id not in TASKS:
            raise ValueError(f"Param {p.name}: task_id '{p.task_id}' not in TASKS")
        task = TASKS[p.task_id]
        for idx in p.landmark_indices:
            if idx >= task.n_keypoints:
                raise ValueError(
                    f"Param {p.name}: landmark index {idx} >= {task.n_keypoints} (task {p.task_id})"
                )

    # Param names unique
    names = [p.name for p in _PARAM_DEFS]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate param names")

    # Overlay consistency — every overlay references valid task/indices
    overlay_names = [o.name for o in _ALL_OVERLAYS]
    if len(overlay_names) != len(set(overlay_names)):
        raise ValueError("Duplicate overlay names")
    for o in _SUPPORTIVE_OVERLAY_DEFS:
        if o.task_id not in TASKS:
            raise ValueError(f"Overlay {o.name}: task_id '{o.task_id}' not in TASKS")
        task = TASKS[o.task_id]
        for idx in o.landmark_indices:
            if idx >= task.n_keypoints:
                raise ValueError(
                    f"Overlay {o.name}: index {idx} >= {task.n_keypoints} (task {o.task_id})"
                )


_validate_registry()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def landmark_valid_mask(task_id: str) -> np.ndarray:
    """bool[60] — which canonical slots belong to this task."""
    mask = np.zeros(K, dtype=bool)
    t = TASKS[task_id]
    mask[t.offset : t.offset + t.n_keypoints] = True
    return mask


def local_to_canonical(keypoints: np.ndarray, task_id: str) -> np.ndarray:
    """Map task-local [N, 2] into canonical [60, 2]. NaN for unowned.

    Rejects half-NaN (one coord finite, the other NaN).
    """
    canonical = np.full((K, 2), np.nan, dtype=np.float32)
    t = TASKS[task_id]
    if keypoints.shape != (t.n_keypoints, 2):
        raise ValueError(
            f"Expected shape ({t.n_keypoints}, 2) for {task_id}, got {keypoints.shape}"
        )
    for i in range(t.n_keypoints):
        n_nan = int(np.isnan(keypoints[i]).sum())
        if n_nan == 1:
            raise ValueError(f"Half-NaN at local index {i} in {task_id}: {keypoints[i]}")
    canonical[t.offset : t.offset + t.n_keypoints] = keypoints
    return canonical


def _build_flip_permutation(
    axis: str,
) -> np.ndarray:
    """int[60] — canonical index permutation for the given flip axis.

    Self-inverse: applying twice restores original order.
    """
    perm = np.arange(K)
    for task in TASKS.values():
        pairs = task.hflip_pairs if axis == "h" else task.vflip_pairs
        for a, b in pairs:
            ga, gb = task.offset + a, task.offset + b
            perm[ga], perm[gb] = gb, ga
    return perm


def global_hflip_permutation() -> np.ndarray:
    """int[60] — canonical index permutation for horizontal flip."""
    return _build_flip_permutation("h")


def global_vflip_permutation() -> np.ndarray:
    """int[60] — canonical index permutation for vertical flip."""
    return _build_flip_permutation("v")


def task_id_to_int(task_id: str) -> int:
    """Map task_id string to stable integer."""
    return TASKS[task_id].task_int


def int_to_task_id(task_int: int) -> str:
    """Map stable integer back to task_id string."""
    for t in TASKS.values():
        if t.task_int == task_int:
            return t.task_id
    raise ValueError(f"Unknown task_int: {task_int}")

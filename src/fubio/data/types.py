"""Shared type contracts for the FU_Biometry pipeline.

Boundary types between data pipeline, model, and training loop.
InstanceDict / BatchDict flow from data/ to train/.
TaskOutput flows from model to training loop.

Upstream: none (pure type definitions).
Downstream: data/*, models/*, train/* — everyone imports from here.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import numpy as np
from torch import Tensor  # noqa: TC002 — used in type annotations throughout

# ---------------------------------------------------------------------------
# Data pipeline contracts
# ---------------------------------------------------------------------------


class InstanceDict(TypedDict):
    """Per-instance annotation, shaped by task (K_total, not global K=60).

    K_total = K_scored + K_supportive (per-task, varies).
    Produced by data pipeline (manifest → mosaic → transforms → collate).
    Consumed by training loop for Hungarian matching and loss computation.
    """

    task_id: int  # 0-8, stable integer from task_registry
    keypoints: np.ndarray  # (K_total, 2) float32, normalized [0,1] or NaN
    supervised_mask: np.ndarray  # (K_total,) bool — has finite GT (dependency-aware for supportive)
    visible_mask: np.ndarray  # (K_total,) bool — in bounds after augmentation
    evidence_mask: np.ndarray  # (K_total,) int — 0/1, ultrasound signal presence
    bbox: np.ndarray  # (4,) float32 — cx, cy, w, h normalized [0,1]
    transform_matrix: np.ndarray  # (3, 3) float32 — original → augmented
    original_hw: np.ndarray  # (2,) int32
    is_labeled: bool
    image_path: str


class BatchDict(TypedDict):
    """Collated batch: stacked images + per-instance targets.

    Collation only stacks images to (B, 3, H, W).
    Targets remain as nested list — no K-dim or I-dim padding.
    """

    image: Tensor  # (B, 3, H, W) uint8 from collate; float32 after FUBioModule._normalize_image
    targets: list[list[InstanceDict]]  # B images × variable instances


# ---------------------------------------------------------------------------
# Model output contract
# ---------------------------------------------------------------------------


class TaskOutput(NamedTuple):
    """Per-task output from FUBioModel.forward().

    Each task produces N_inst instance slots. Unmatched slots are
    suppressed by low confidence during training (via Hungarian matching)
    and at inference (via thresholding).
    """

    bbox: Tensor  # (B, N_inst, 4) — cx, cy, w, h normalized [0,1]
    conf: Tensor  # (B, N_inst, 1) — raw logit (sigmoid → confidence)
    landmarks: Tensor  # (B, N_inst, K, 2) — normalized [0,1]
    heatmap: Tensor | None = None  # (B, N_inst, K, S) — GeoSimCC stage-1 affinity dist

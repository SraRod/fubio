"""Postprocessing: model output → original pixel coordinates → submission JSON.

Coordinate flow:
    model output [0,1] × (W_input, H_input) → augmented pixel space
    → inverse transform_matrix → original pixel space → clip → JSON

Upstream: data/types.py (TaskOutput, InstanceDict).
Downstream: serving/predict.py (the only submission writer).
"""

from __future__ import annotations

import numpy as np
from torch import Tensor


def select_serving_query(conf: Tensor) -> Tensor:
    """Pick the instance slot that inference emits: the most confident one.

    THE single definition of the serving selection policy. Both the validation
    metric path (train/module.py) and the inference path (serving/predict.py)
    must route through here — when they drifted apart, validation scored the
    Hungarian/GT-matched query while inference scored the confidence-argmax one,
    making every local number optimistic and inverting the platform ranking of
    R15 vs R17 (measured 2026-07-25).

    Args:
        conf: (..., N_inst) or (..., N_inst, 1) raw confidence logits. Sigmoid is
            monotonic, so logits and probabilities give the same argmax; both are
            accepted so callers need not agree on which they hold.

    Returns:
        (...,) int64 indices into the instance dimension.
    """
    if conf.shape[-1] == 1 and conf.dim() > 1:
        conf = conf.squeeze(-1)
    return conf.argmax(dim=-1)


def inverse_transform_landmarks(
    landmarks_pixel: Tensor,
    transform_matrix: np.ndarray,
    original_hw: np.ndarray,
) -> np.ndarray:
    """Map landmarks from augmented pixel space to original pixel space.

    Args:
        landmarks_pixel: (K, 2) augmented pixel coordinates.
        transform_matrix: (3, 3) original → augmented affine.
        original_hw: (2,) original image (H, W).

    Returns:
        (K, 2) float32 coordinates in original pixel space, clipped.
    """
    lm_np = landmarks_pixel.detach().cpu().numpy().astype(np.float64)
    k = lm_np.shape[0]

    M_inv = np.linalg.inv(transform_matrix.astype(np.float64))

    ones = np.ones((k, 1), dtype=np.float64)
    lm_homo = np.hstack([lm_np, ones])

    lm_orig = (lm_homo @ M_inv.T)[:, :2]

    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])
    lm_orig[:, 0] = np.clip(lm_orig[:, 0], 0, orig_w - 1)
    lm_orig[:, 1] = np.clip(lm_orig[:, 1], 0, orig_h - 1)

    return lm_orig.astype(np.float32)

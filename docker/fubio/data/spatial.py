"""Spatial affine transforms — compose once, apply once.

All elementary transforms (resize, center, scale, rotate, translate, flip)
are composed into a single 3×3 matrix M before application. Image sees one
cv2.warpAffine; keypoints see one matrix multiply. No per-transform sequential
application.

Upstream: task_registry.py (global_hflip_permutation / global_vflip_permutation).
Downstream: transforms.py, mosaic.py, serving/predict.py (map_to_original).
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class SpatialParams(BaseModel):
    """Parameters for a single spatial augmentation draw."""

    model_config = ConfigDict(frozen=True)

    target_size: tuple[int, int]  # (W, H) of output canvas
    hflip: bool = False
    vflip: bool = False
    rotation_deg: float = 0.0
    scale: float = 1.0
    translate_x: float = 0.0  # pixels in target space
    translate_y: float = 0.0


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def build_affine_matrix(
    src_size: tuple[int, int],
    params: SpatialParams,
    resize_mode: str = "letterbox",
) -> np.ndarray:
    """Build composed 3×3 affine: source pixel → augmented pixel.

    M = M_vflip @ M_hflip @ M_translate_back @ M_rotate @ M_scale @ M_center @ M_resize

    Right-to-left application order:
      M_resize         — source → target (letterbox or stretch)
      M_center         — translate target center to origin
      M_scale          — uniform scale around origin
      M_rotate         — CCW rotation around origin
      M_translate_back — origin → center + random offset
      M_hflip          — x' = (W-1) - x  (if enabled)
      M_vflip          — y' = (H-1) - y  (if enabled)
    """
    src_w, src_h = src_size
    tgt_w, tgt_h = params.target_size

    # --- M_resize: maps source pixel coords to target pixel coords ---
    if resize_mode == "letterbox":
        # Uniform scale to fit within target, preserving aspect ratio
        s = min(tgt_w / src_w, tgt_h / src_h)
        pad_x = (tgt_w - src_w * s) / 2.0
        pad_y = (tgt_h - src_h * s) / 2.0
        M_resize = np.array([[s, 0, pad_x], [0, s, pad_y], [0, 0, 1]], dtype=np.float64)
    elif resize_mode == "stretch":
        sx = tgt_w / src_w
        sy = tgt_h / src_h
        M_resize = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode!r}")

    cx, cy = tgt_w / 2.0, tgt_h / 2.0

    M_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)

    sc = float(params.scale)
    M_scale = np.array([[sc, 0, 0], [0, sc, 0], [0, 0, 1]], dtype=np.float64)

    cos_r = np.cos(np.radians(params.rotation_deg))
    sin_r = np.sin(np.radians(params.rotation_deg))
    M_rotate = np.array(
        [[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]],
        dtype=np.float64,
    )

    M_translate_back = np.array(
        [
            [1, 0, cx + params.translate_x],
            [0, 1, cy + params.translate_y],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    M_hflip = np.eye(3, dtype=np.float64)
    if params.hflip:
        # pixel-center flip: x' = (W-1) - x (not W - x)
        M_hflip = np.array(
            [[-1, 0, tgt_w - 1], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )

    M_vflip = np.eye(3, dtype=np.float64)
    if params.vflip:
        # pixel-center flip: y' = (H-1) - y
        M_vflip = np.array(
            [[1, 0, 0], [0, -1, tgt_h - 1], [0, 0, 1]],
            dtype=np.float64,
        )

    # float64 composition for numerical precision
    return M_vflip @ M_hflip @ M_translate_back @ M_rotate @ M_scale @ M_center @ M_resize


# ---------------------------------------------------------------------------
# Apply to image and keypoints
# ---------------------------------------------------------------------------


def apply_spatial(
    image: np.ndarray,
    keypoints: np.ndarray,
    M: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply composed affine to image and keypoints.

    Args:
        image: [H, W, 3] uint8 source image.
        keypoints: [I, K, 2] float32, NaN for unowned/unannotated.
        M: [3, 3] float64 forward affine (source → augmented).
        target_size: (W, H) of output canvas.

    Returns:
        image_out: [tgt_H, tgt_W, 3] uint8.
        kp_out: [I, K, 2] float32, NaN preserved for non-finite inputs.
        visible_mask: [I, K] bool — True only if input was finite AND
            transformed coordinate is within [0, W-1] × [0, H-1].
    """
    tgt_w, tgt_h = target_size

    # Forward map — cv2.warpAffine applies M directly, no WARP_INVERSE_MAP
    image_out = cv2.warpAffine(
        image,
        M[:2].astype(np.float64),
        (tgt_w, tgt_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    n_inst, k_dim, _ = keypoints.shape
    kp_out = np.full_like(keypoints, np.nan)
    visible_mask = np.zeros((n_inst, k_dim), dtype=bool)

    for inst in range(n_inst):
        kp_inst = keypoints[inst]
        finite = ~np.isnan(kp_inst).any(axis=1)
        if not finite.any():
            continue

        pts = kp_inst[finite].astype(np.float64)
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        pts_t = (M @ pts_h.T).T[:, :2]

        kp_out[inst, finite] = pts_t.astype(np.float32)

        # In-bounds check: closed interval [0, W-1] × [0, H-1]
        visible_mask[inst, finite] = (
            (pts_t[:, 0] >= 0)
            & (pts_t[:, 0] <= tgt_w - 1)
            & (pts_t[:, 1] >= 0)
            & (pts_t[:, 1] <= tgt_h - 1)
        )

    return image_out, kp_out, visible_mask


def map_to_original(
    pred_keypoints: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Map predictions back to original pixel space.

    Args:
        pred_keypoints: [..., K, 2] predicted coords in augmented space.
        M: [..., 3, 3] forward affine matrices (same leading dims).

    Returns:
        [..., K, 2] float32 coords in original space.
        Padded instances (zero M) produce all-NaN output.

    Note: this reverses geometry only. If flip was applied, the caller
    must also apply inverse permutation via the corresponding
    global_hflip_permutation() / global_vflip_permutation().
    """
    lead_shape = pred_keypoints.shape[:-2]
    K = pred_keypoints.shape[-2]
    flat_kp = pred_keypoints.reshape(-1, K, 2)
    # float64 for numerical precision in matrix inversion
    flat_M = M.reshape(-1, 3, 3).astype(np.float64)
    result = np.full_like(flat_kp, np.nan)

    for idx in range(len(flat_kp)):
        m = flat_M[idx]
        # Zero M = padded instance, skip
        if np.allclose(m, 0):
            continue
        M_inv = np.linalg.inv(m)
        finite = ~np.isnan(flat_kp[idx]).any(axis=1)
        if not finite.any():
            continue

        pts = flat_kp[idx, finite].astype(np.float64)
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        pts_orig = (M_inv @ pts_h.T).T[:, :2]
        result[idx, finite] = pts_orig.astype(np.float32)

    return result.reshape(*lead_shape, K, 2)

"""Differentiable geometric views for semi-supervised consistency.

Generates bounded random affine transforms (rotation + scale + translate +
optional flip) and applies them on GPU via kornia. The same matrix that warps
the image maps the landmarks, so the coordinate contract is explicit and
testable.

Upstream: train/module.py (_semi_step).
Downstream: none (leaf module).
"""

from __future__ import annotations

import kornia.geometry.transform as KGT
import torch
from torch import Tensor


def sample_affine(
    batch_size: int,
    rotation_range: float = 30.0,
    scale_range: tuple[float, float] = (0.8, 1.2),
    translate_range: float = 0.1,
    flip_prob: float = 0.0,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample random affine matrices (B, 2, 3) in normalized [0,1] space.

    The matrix maps source coordinates to destination coordinates (forward).
    Rotation and scale are applied around the image center (0.5, 0.5);
    translation is a fraction of image size. Horizontal flip is composed
    after rotation/scale/translate as x → (1 - x).

    Returns the FORWARD matrix (src→dst). warp_image passes it directly
    to kornia's warp_affine, which expects a forward pixel-space matrix.
    """
    # Rotation in radians
    angle = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
        -rotation_range, rotation_range
    ) * (torch.pi / 180.0)

    # Scale
    scale = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
        scale_range[0], scale_range[1]
    )

    # Translation (fraction of image, applied after rotation+scale)
    tx = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
        -translate_range, translate_range
    )
    ty = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
        -translate_range, translate_range
    )

    cos_a = torch.cos(angle) * scale
    sin_a = torch.sin(angle) * scale

    # Rotate+scale around center (0.5, 0.5), then translate
    # M = T(0.5+tx, 0.5+ty) @ R(θ,s) @ T(-0.5, -0.5)
    matrix = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = cos_a
    matrix[:, 0, 1] = -sin_a
    matrix[:, 0, 2] = 0.5 * (1 - cos_a + sin_a) + tx
    matrix[:, 1, 0] = sin_a
    matrix[:, 1, 1] = cos_a
    matrix[:, 1, 2] = 0.5 * (1 - cos_a - sin_a) + ty

    # Horizontal flip: compose Fh @ M where Fh = [-1,0,1; 0,1,0] in [0,1] space.
    # Result: x' = 1 - x_affine, y' = y_affine.
    # Per-sample: flip_mask[b] = True with probability flip_prob.
    if flip_prob > 0:
        flip_mask = torch.rand(batch_size, device=device) < flip_prob
        if flip_mask.any():
            # Fh @ M: negate row 0, then add 1 to translation
            matrix[flip_mask, 0, 0] *= -1
            matrix[flip_mask, 0, 1] *= -1
            matrix[flip_mask, 0, 2] = 1.0 - matrix[flip_mask, 0, 2]

    return matrix


def make_rotation_affine(
    batch_size: int,
    angle_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Deterministic rotation affine (B, 2, 3) around image center (0.5, 0.5)."""
    angle = torch.full((batch_size,), angle_deg, device=device, dtype=dtype) * (torch.pi / 180.0)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    matrix = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = cos_a
    matrix[:, 0, 1] = -sin_a
    matrix[:, 0, 2] = 0.5 * (1 - cos_a + sin_a)
    matrix[:, 1, 0] = sin_a
    matrix[:, 1, 1] = cos_a
    matrix[:, 1, 2] = 0.5 * (1 - cos_a - sin_a)
    return matrix


def _forward_to_inverse(matrix_fwd: Tensor) -> Tensor:
    """Convert (B, 2, 3) forward affine to inverse for kornia warp_affine."""
    B = matrix_fwd.shape[0]
    # Pad to 3×3
    pad = torch.tensor([0.0, 0.0, 1.0], device=matrix_fwd.device, dtype=matrix_fwd.dtype)
    pad = pad.expand(B, 1, 3)
    full = torch.cat([matrix_fwd, pad], dim=1)  # (B, 3, 3)
    inv = torch.linalg.inv(full)
    return inv[:, :2, :]  # (B, 2, 3)


def warp_image(
    images: Tensor,
    matrix_fwd: Tensor,
    *,
    padding_mode: str = "reflection",
) -> Tensor:
    """Apply affine transform to images.

    images: (B, C, H, W) — already normalized floats.
    matrix_fwd: (B, 2, 3) forward affine in [0,1] space.

    kornia's warp_affine expects a FORWARD pixel-space matrix (src→dst);
    it normalizes and inverts internally for grid sampling. The [0,1]→pixel
    conversion only scales the translation column — correct for square
    images where the rotation/scale submatrix is dimensionless.
    """
    _, _, H, W = images.shape
    if H != W:
        raise ValueError(
            f"warp_image assumes square images (H==W) for the [0,1]→pixel "
            f"conversion; got {H}×{W}. Non-square requires S @ M @ S⁻¹."
        )
    # Convert from [0,1] to pixel space: only translation needs scaling
    pixel_fwd = matrix_fwd.clone()
    pixel_fwd[:, 0, 2] = pixel_fwd[:, 0, 2] * W
    pixel_fwd[:, 1, 2] = pixel_fwd[:, 1, 2] * H

    return KGT.warp_affine(
        images,
        pixel_fwd,
        dsize=(H, W),
        mode="bilinear",
        padding_mode=padding_mode,
    )


def warp_landmarks(
    landmarks: Tensor,
    matrix_fwd: Tensor,
) -> Tensor:
    """Map landmarks through the forward affine.

    landmarks: (B, ..., 2) in [0,1] normalized coordinates.
    matrix_fwd: (B, 2, 3) forward affine in [0,1] space.

    Returns landmarks in [0,1] space after transformation.
    """
    shape = landmarks.shape  # (B, ..., 2)
    B = shape[0]
    flat = landmarks.reshape(B, -1, 2)  # (B, N, 2)

    # Homogeneous: (B, N, 3)
    ones = torch.ones(B, flat.shape[1], 1, device=flat.device, dtype=flat.dtype)
    hom = torch.cat([flat, ones], dim=-1)

    # Apply: result[b, n, :] = matrix_fwd[b] @ hom[b, n, :]
    result = torch.einsum("brc,bnc->bnr", matrix_fwd, hom)  # (B, N, 2)

    return result.reshape(shape)

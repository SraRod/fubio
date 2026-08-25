"""Query position encoding utilities.

Provides Fourier position encoding for learnable anchor positions.
Shares the same coordinate convention as neck.sinusoidal_2d_pos_enc
(first half encodes y, second half encodes x, temperature=10000,
positions in [0, 2π]) — this alignment is critical for cross-attention
Q/K positions to be in the same space.

Upstream: used by TaskModule (heads.py) for anchor-derived query positions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def fourier_position_encoding(xy: Tensor, d_model: int) -> Tensor:
    """Encode (*, 2) xy coords in [0,1] to (*, d_model) Fourier features.

    Same formula as neck.sinusoidal_2d_pos_enc but for arbitrary
    positions (not just a regular grid). Coords are scaled to [0, 2π]
    (DETR convention). y-encoding in the first half, x-encoding in the
    second — matches the memory positional encoding so query and key
    positions are in the same space.
    """
    half_d = d_model // 2
    temperature = 10000.0

    dim_t = torch.arange(half_d // 2, device=xy.device, dtype=torch.float32)
    dim_t = temperature ** (2 * dim_t / half_d)

    flat_xy = xy.reshape(-1, 2) * (2 * math.pi)

    pos_y = flat_xy[:, 1:2] / dim_t  # (N, half_d//2)
    pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).reshape(-1, half_d)

    pos_x = flat_xy[:, 0:1] / dim_t
    pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).reshape(-1, half_d)

    pe = torch.cat([pos_y, pos_x], dim=-1)  # (N, d_model)
    return pe.reshape(*xy.shape[:-1], d_model)

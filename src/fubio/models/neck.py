"""Neck — Layer 2 of the five-layer architecture.

Fuses backbone features into spatial memory for the downstream decoder.
Dense → dense: backbone patch tokens become the spatial token sequence
that decoder queries cross-attend to.

Shared, not per-task. All tasks read the same spatial memory.

Contract:
  - memory tokens are row-major (left-to-right, top-to-bottom) and must
    match spatial_shape. coord_predictors.normalized_grid() is the single
    source of truth for this ordering. Changing resolution requires
    spatial_shape to update in sync; swapping x/y silently produces
    transposed coordinates.
  - memory_pos uses sinusoidal_2d_pos_enc (first half encodes y, second
    half encodes x), in the same coordinate space as
    queries.fourier_position_encoding — otherwise cross-attention Q/K
    positions diverge.
  - PE cache is keyed on (h, w), not a bool flag.

Upstream: backbone.py (BackboneOutput with features list).
Downstream: decoder.py (memory and memory_pos for cross-attention).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fubio.models.backbone import BackboneOutput


def sinusoidal_2d_pos_enc(h: int, w: int, d_model: int) -> Tensor:
    """Fixed sinusoidal 2D positional encoding.

    Returns (1, h*w, d_model). d_model//2 dims encode y, d_model//2 encode x.
    Standard DETR pattern with temperature=10000 and [0, 2π] position range.
    """
    half_d = d_model // 2
    temperature = 10000.0

    y_pos = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
    x_pos = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)

    # Normalize to [0, 2π] (DETR convention — full sinusoidal period)
    y_pos = y_pos / (h - 1 + 1e-6) * (2 * math.pi)
    x_pos = x_pos / (w - 1 + 1e-6) * (2 * math.pi)

    dim_t = torch.arange(half_d // 2, dtype=torch.float32)
    dim_t = temperature ** (2 * dim_t / half_d)

    # y encoding: sin/cos interleaved
    pos_y = y_pos.reshape(-1, 1) / dim_t.unsqueeze(0)  # (h*w, half_d//2)
    pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).reshape(-1, half_d)

    # x encoding: sin/cos interleaved
    pos_x = x_pos.reshape(-1, 1) / dim_t.unsqueeze(0)
    pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).reshape(-1, half_d)

    pe = torch.cat([pos_y, pos_x], dim=-1)  # (h*w, d_model)
    return pe.unsqueeze(0)  # (1, h*w, d_model)


class NeckOutput:
    """Return type for all neck modules."""

    __slots__ = ("memory", "memory_pos", "spatial_shape")

    def __init__(
        self,
        memory: Tensor,
        memory_pos: Tensor,
        spatial_shape: tuple[int, int],
    ) -> None:
        self.memory = memory
        self.memory_pos = memory_pos
        self.spatial_shape = spatial_shape


class LinearNeck(nn.Module):
    """Single linear projection for flat-token backbones (DINOv2).

    Layer 2 (neck): dense → dense, shared.

    Upstream: backbone.py (BackboneOutput, flat tokens).
    Downstream: decoder.py (memory / memory_pos for cross-attention).
    """

    def __init__(self, c_backbone: int = 768, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(c_backbone, d_model)
        self.dropout = nn.Dropout(dropout)
        self._d_model = d_model
        self._pe_hw: tuple[int, int] | None = None

    def _ensure_pos_enc(self, h: int, w: int, device: torch.device) -> None:
        if self._pe_hw == (h, w):
            return
        pe = sinusoidal_2d_pos_enc(h, w, self._d_model).to(device)
        self.register_buffer("pos_enc", pe, persistent=False)
        self._pe_hw = (h, w)

    def forward(self, backbone_out: BackboneOutput) -> NeckOutput:
        h, w = backbone_out.spatial_shape
        tokens = backbone_out.features[-1]  # (B, N, C_backbone)

        memory = self.dropout(self.proj(tokens))  # (B, N, D_model)

        self._ensure_pos_enc(h, w, memory.device)
        memory_pos: Tensor = self.pos_enc  # type: ignore[assignment]

        return NeckOutput(memory=memory, memory_pos=memory_pos, spatial_shape=(h, w))


class FPNNeck(nn.Module):
    """Top-down feature fusion for hierarchical backbones (ConvNeXtV2).

    Layer 2 (neck): dense → dense, shared.

    Fuses selected stages via lateral 1x1 conv + top-down upsample + add,
    then 3x3 smoothing conv. Output at the shallowest selected stage's stride.

    Upstream: backbone.py (BackboneOutput with spatial feature maps).
    Downstream: decoder.py (memory / memory_pos for cross-attention).
    """

    def __init__(
        self,
        stage_channels: list[int],
        stages: list[int],
        d_model: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("stages must be non-empty")
        self._stages = sorted(stages)
        self._d_model = d_model

        self.laterals = nn.ModuleDict(
            {str(s): nn.Conv2d(stage_channels[s], d_model, 1) for s in self._stages}
        )
        self.smooths = nn.ModuleDict(
            {str(s): nn.Conv2d(d_model, d_model, 3, padding=1) for s in self._stages[:-1]}
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._pe_hw: tuple[int, int] | None = None

    def _ensure_pos_enc(self, h: int, w: int, device: torch.device) -> None:
        if self._pe_hw == (h, w):
            return
        pe = sinusoidal_2d_pos_enc(h, w, self._d_model).to(device)
        self.register_buffer("pos_enc", pe, persistent=False)
        self._pe_hw = (h, w)

    def forward(self, backbone_out: BackboneOutput) -> NeckOutput:
        all_feats = backbone_out.features

        stages_desc = sorted(self._stages, reverse=True)
        x = self.laterals[str(stages_desc[0])](all_feats[stages_desc[0]])

        for s in stages_desc[1:]:
            lateral = self.laterals[str(s)](all_feats[s])
            x = F.interpolate(x, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            x = x + lateral
            x = self.smooths[str(s)](x)

        h_out, w_out = x.shape[2], x.shape[3]
        memory = x.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        memory = self.dropout(self.output_norm(memory))

        self._ensure_pos_enc(h_out, w_out, memory.device)
        memory_pos: Tensor = self.pos_enc  # type: ignore[assignment]

        return NeckOutput(
            memory=memory,
            memory_pos=memory_pos,
            spatial_shape=(h_out, w_out),
        )


class MultiLayerNeck(nn.Module):
    """Fuse multiple ViT block outputs with learned weights.

    Layer 2 (neck): dense → dense, shared.

    Each intermediate block's tokens are linearly projected to d_model, then
    combined via softmax-normalized learned weights.

    Upstream: backbone.py (BackboneOutput with multi-layer flat tokens).
    Downstream: decoder.py (memory / memory_pos for cross-attention).
    """

    def __init__(
        self,
        n_layers: int,
        c_backbone: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._n_layers = n_layers
        self._d_model = d_model

        self.projections = nn.ModuleList([nn.Linear(c_backbone, d_model) for _ in range(n_layers)])
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._pe_hw: tuple[int, int] | None = None

    def _ensure_pos_enc(self, h: int, w: int, device: torch.device) -> None:
        if self._pe_hw == (h, w):
            return
        pe = sinusoidal_2d_pos_enc(h, w, self._d_model).to(device)
        self.register_buffer("pos_enc", pe, persistent=False)
        self._pe_hw = (h, w)

    def forward(self, backbone_out: BackboneOutput) -> NeckOutput:
        h, w = backbone_out.spatial_shape
        features = backbone_out.features

        weights = self.layer_weights.softmax(dim=0)
        fused: Tensor | None = None
        for i, (feat, proj) in enumerate(zip(features, self.projections, strict=True)):
            projected = proj(feat)
            fused = weights[i] * projected if fused is None else fused + weights[i] * projected

        assert fused is not None
        memory = self.dropout(self.norm(fused))

        self._ensure_pos_enc(h, w, memory.device)
        memory_pos: Tensor = self.pos_enc  # type: ignore[assignment]

        return NeckOutput(memory=memory, memory_pos=memory_pos, spatial_shape=(h, w))


# ---------------------------------------------------------------------------
# C2f Neck — RF-DETR-style spatial fusion (Commit B)
# ---------------------------------------------------------------------------


class _ConvNormAct(nn.Module):
    """Conv2d → LayerNorm (channel-wise) → SiLU. No bias when followed by norm."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, bias=False)
        self.norm = nn.LayerNorm(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C_in, H, W) → conv → (B, C_out, H, W) → permute for LN → back
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.act(x)


class _Bottleneck(nn.Module):
    """Two 3x3 convs + residual. Spatial mixing at fixed channel width."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.cv1 = _ConvNormAct(c, c, 3, padding=1)
        self.cv2 = _ConvNormAct(c, c, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x))


class _C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (C2f from YOLOv8 / RF-DETR).

    Split → chain of bottleneck blocks → concat all → compress.
    Provides cross-layer + cross-position mixing with residual gradient flow.

    c_in: input channels (typically 4 * c_backbone for concat of 4 ViT layers).
    c_out: output channels (= d_model).
    n_bottleneck: number of chained bottleneck blocks (default 2).
    """

    def __init__(self, c_in: int, c_out: int, n_bottleneck: int = 2) -> None:
        super().__init__()
        c_half = c_out // 2
        self.cv1 = _ConvNormAct(c_in, c_out, 1)
        self.bottlenecks = nn.ModuleList([_Bottleneck(c_half) for _ in range(n_bottleneck)])
        # concat: chunk_0 + chunk_1 + n_bottleneck outputs = (2 + n_bottleneck) * c_half
        self.cv2 = _ConvNormAct((2 + n_bottleneck) * c_half, c_out, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)  # (B, c_out, H, W)
        chunks = x.chunk(2, dim=1)  # two halves of c_out
        parts = [chunks[0], chunks[1]]
        y = chunks[1]
        for bn in self.bottlenecks:
            y = bn(y)
            parts.append(y)
        return self.cv2(torch.cat(parts, dim=1))


class C2fNeck(nn.Module):
    """C2f spatial fusion of multiple ViT block outputs (RF-DETR pattern).

    Layer 2 (neck): dense → dense, shared.

    Each intermediate block's tokens are normalized, reshaped to 2D spatial
    maps, concatenated channel-wise, then processed by a C2f block that
    provides nonlinear cross-layer + cross-position fusion via 3x3 spatial
    convolutions with residual bottlenecks.

    Upstream: backbone.py (BackboneOutput with multi-layer flat tokens).
    Downstream: decoder.py (memory / memory_pos for cross-attention).
    """

    def __init__(
        self,
        n_layers: int,
        c_backbone: int,
        d_model: int = 256,
        n_bottleneck: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._n_layers = n_layers
        self._d_model = d_model

        # Per-layer normalization before concat (different ViT depths have different magnitudes)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(c_backbone) for _ in range(n_layers)])
        self.c2f = _C2f(n_layers * c_backbone, d_model, n_bottleneck)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._pe_hw: tuple[int, int] | None = None

    def _ensure_pos_enc(self, h: int, w: int, device: torch.device) -> None:
        if self._pe_hw == (h, w):
            return
        pe = sinusoidal_2d_pos_enc(h, w, self._d_model).to(device)
        self.register_buffer("pos_enc", pe, persistent=False)
        self._pe_hw = (h, w)

    def forward(self, backbone_out: BackboneOutput) -> NeckOutput:
        h, w = backbone_out.spatial_shape
        features = backbone_out.features

        # Per-layer LN → reshape to 2D → concat along channel dim
        maps: list[Tensor] = []
        for feat, ln in zip(features, self.layer_norms, strict=True):
            normed = ln(feat)  # (B, N, C)
            spatial = normed.transpose(1, 2).reshape(-1, normed.shape[-1], h, w)  # (B, C, H, W)
            maps.append(spatial)

        x = torch.cat(maps, dim=1)  # (B, n_layers * C, H, W)
        x = self.c2f(x)  # (B, d_model, H, W)

        # Flatten back to token sequence
        memory = x.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        memory = self.dropout(self.output_norm(memory))

        self._ensure_pos_enc(h, w, memory.device)
        memory_pos: Tensor = self.pos_enc  # type: ignore[assignment]

        return NeckOutput(memory=memory, memory_pos=memory_pos, spatial_shape=(h, w))


def build_neck(
    config: object,
    backbone: nn.Module,
    d_model: int,
    dropout: float = 0.1,
) -> LinearNeck | FPNNeck | MultiLayerNeck | C2fNeck:
    """Factory: pick neck by config type.

    Deferred import of config types to avoid circular dependency.
    """
    from fubio.train.config import (
        C2fNeckConfig,
        FPNNeckConfig,
        LinearNeckConfig,
        MultiLayerNeckConfig,
    )

    if isinstance(config, LinearNeckConfig):
        return LinearNeck(c_backbone=backbone.embed_dim, d_model=d_model, dropout=dropout)
    if isinstance(config, FPNNeckConfig):
        if not hasattr(backbone, "stage_channels"):
            raise ValueError(
                "FPNNeck requires a hierarchical backbone with stage_channels "
                f"(got {type(backbone).__name__})"
            )
        return FPNNeck(
            stage_channels=backbone.stage_channels,
            stages=config.stages,
            d_model=d_model,
            dropout=dropout,
        )
    if isinstance(config, MultiLayerNeckConfig):
        return MultiLayerNeck(
            n_layers=len(config.layer_indices),
            c_backbone=backbone.embed_dim,
            d_model=d_model,
            dropout=dropout,
        )
    if isinstance(config, C2fNeckConfig):
        return C2fNeck(
            n_layers=len(config.layer_indices),
            c_backbone=backbone.embed_dim,
            d_model=d_model,
            n_bottleneck=config.n_bottleneck,
            dropout=dropout,
        )
    raise ValueError(f"Unknown neck config: {type(config).__name__}")

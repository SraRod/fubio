"""MIRO — Mutual Information Regularization with Oracle (Round 2+).

Ports and adapts kakaobrain/miro for ViT-based backbone.
Toggle via MIROConfig.enabled (default OFF).

Upstream: models/backbone.py (BackboneOutput.features).
Downstream: train/module.py (MIRO loss term).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fubio.train.config import MIROConfig

logger = logging.getLogger(__name__)


class MeanEncoder(nn.Module):
    """Identity — deterministic feature mean for MIRO."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x


class VarianceEncoder(nn.Module):
    """Bias-only diagonal variance via softplus.

    Learns a per-channel variance parameter that is broadcast across
    spatial / token positions.  Works for both ViT (B, N, C) and
    CNN (B, C, H, W) feature tensors.

    Args:
        shape: Full shape of one feature tensor *including batch*.
        init: Desired initial variance (post-softplus).
        channelwise: Per-channel (True) or per-element (False).
        eps: Stability floor added after softplus.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        init: float = 0.1,
        channelwise: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps

        # Invert softplus so that softplus(init_val) + eps ≈ init
        init_val = torch.as_tensor(init - eps).clamp(min=1e-8).expm1().log()

        if channelwise:
            ndim = len(shape)
            if ndim == 4:
                # CNN: (B, C, H, W) → param (1, C, 1, 1)
                param_shape = (1, shape[1], 1, 1)
            elif ndim == 3:
                # ViT: (B, N, C) → param (1, 1, C)
                param_shape = (1, 1, shape[2])
            else:
                raise ValueError(
                    f"channelwise variance expects 3-D (ViT) or 4-D (CNN) shape, "
                    f"got {ndim}-D: {shape}"
                )
        else:
            param_shape = tuple(shape)

        self.bias = nn.Parameter(torch.full(param_shape, init_val.item()))

    def forward(self, x: Tensor) -> Tensor:  # noqa: ARG002 — x unused by design
        return F.softplus(self.bias) + self.eps


class MIROEncoders(nn.Module):
    """Paired mean + variance encoders for each intermediate feature layer.

    Forward returns (means, variances) lists aligned with the input
    feature list from the backbone.
    """

    def __init__(
        self,
        shapes: list[tuple[int, ...]],
        init: float = 0.1,
        channelwise: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mean_encoders = nn.ModuleList(
            [MeanEncoder(shape) for shape in shapes],
        )
        self.var_encoders = nn.ModuleList(
            [
                VarianceEncoder(shape, init=init, channelwise=channelwise, eps=eps)
                for shape in shapes
            ],
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MIROEncoders: %d layers, channelwise=%s, init=%.4f, params=%d",
            len(shapes),
            channelwise,
            init,
            total_params,
        )

    def forward(
        self,
        features: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        means = [enc(f) for enc, f in zip(self.mean_encoders, features, strict=True)]
        variances = [enc(f) for enc, f in zip(self.var_encoders, features, strict=True)]
        return means, variances


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def build_miro_encoders(
    backbone: nn.Module,
    input_shape: tuple[int, int],
    config: MIROConfig,
) -> MIROEncoders:
    """Probe backbone with a dummy forward to discover feature shapes, then build encoders.

    Args:
        backbone: Must return BackboneOutput with non-empty features list.
        input_shape: Spatial (H, W) *without* batch or channel dims.
        config: MIRO hyperparameters (init_variance, etc.).
    """
    device = next(backbone.parameters()).device
    dummy = torch.zeros(1, 3, *input_shape, device=device)

    backbone.eval()
    with torch.no_grad():
        out = backbone(dummy)

    feats = out.features
    if not feats:
        raise RuntimeError(
            "Backbone returned empty features list. "
            "Set return_intermediate=True when MIRO is enabled."
        )

    shapes = [tuple(f.shape) for f in feats]
    logger.info("MIRO feature shapes (from backbone probe): %s", shapes)

    return MIROEncoders(shapes, init=config.init_variance)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------


def miro_loss(
    pre_feats: list[Tensor],
    post_feats: list[Tensor],
    encoders: MIROEncoders,
) -> Tensor:
    """MI regularization loss: -log N(pre | mean(post), var(post)).

    Encourages the training backbone's features to stay close to the
    frozen pretrained backbone's features, measured per-layer with a
    diagonal Gaussian parameterised by the MIRO encoders.

    Args:
        pre_feats: Intermediate features from frozen pretrained backbone.
        post_feats: Intermediate features from training backbone.
        encoders: Paired mean / variance encoders.

    Returns:
        Scalar loss (summed over layers, averaged over batch).
    """
    means, variances = encoders(post_feats)

    total = torch.tensor(0.0, device=post_feats[0].device)
    for pre, mu, var in zip(pre_feats, means, variances, strict=True):
        # -log N(pre | mu, var) ∝ 0.5 * [(pre-mu)^2 / var + log(var)]
        total = total + 0.5 * (((pre - mu) ** 2) / var + var.log()).mean()

    return total

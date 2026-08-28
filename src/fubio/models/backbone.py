"""DINOv2 backbone wrapper producing structured multi-level features.

Produces BackboneOutput with a list of token tensors from evenly spaced
blocks; the neck (neck.py) fuses them into the shared d_model memory.

Upstream: none (pretrained weights from torch.hub, or the vendored copy via
FUBIO_DINOV2_LOCAL in the offline container).
Downstream: neck.py (adapts backbone features to d_model).
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class BackboneOutput(NamedTuple):
    """Structured return from any backbone.

    features: list of (B, N, C) flat token sequences — the tapped
        intermediate blocks, or just the last block.
    spatial_shape: (H, W) spatial dims of features[-1].
    """

    features: list[Tensor]
    spatial_shape: tuple[int, int]


class DINOv2Backbone(nn.Module):
    """DINOv2 backbone via torch.hub.

    Input:  (B, 3, H, W) where H, W are divisible by patch_size (14).
    Output: BackboneOutput with flat patch tokens (B, N, C_backbone).

    return_intermediate=False → features = [last_block_output].
    return_intermediate=True  → features = the tapped intermediate blocks.

    Upstream: none (pretrained).
    Downstream: neck.py (C2fNeck).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        return_intermediate: bool = False,
        n_intermediate_layers: int = 4,
        intermediate_layer_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._return_intermediate = return_intermediate
        self._n_intermediate_layers = n_intermediate_layers
        # [P0: Multi-layer fusion] list → specific block indices; int → last N blocks
        self._intermediate_layer_indices = intermediate_layer_indices

        _dinov2_local = os.environ.get("FUBIO_DINOV2_LOCAL")
        if _dinov2_local:
            hub_model = torch.hub.load(
                _dinov2_local, model_name,
                source="local", pretrained=False,
            )
        else:
            hub_model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
        assert isinstance(hub_model, nn.Module)
        self._model: nn.Module = hub_model
        self._model.eval()

        self._embed_dim: int = hub_model.embed_dim  # type: ignore[attr-defined]
        self._patch_size: int = hub_model.patch_size  # type: ignore[attr-defined]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def freeze(self) -> None:
        self._is_frozen = True
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        self._is_frozen = False
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def train(self, mode: bool = True) -> DINOv2Backbone:
        if getattr(self, "_is_frozen", False):
            return super().train(False)
        return super().train(mode)

    def param_groups(self, lr: float, layer_decay: float = 1.0) -> list[dict]:
        """Param groups with optional layer-wise LR decay.

        layer_decay=1.0: single group at base lr.
        layer_decay<1.0: per-block groups, lr(level) = lr × decay^(top - level).
        Levels: embeddings=0, blocks.i=i+1, final norm=top.
        """
        if layer_decay >= 1.0:
            return [{"params": list(self.parameters()), "lr": lr, "name": "backbone"}]

        try:
            n_blocks = len(self._model.blocks)  # type: ignore[attr-defined]
        except AttributeError:
            n_blocks = 12
        top = n_blocks + 1

        embed_keys = ("patch_embed", "pos_embed", "cls_token", "mask_token", "register_tokens")

        def level(name: str) -> int:
            if any(k in name for k in embed_keys):
                return 0
            if ".blocks." in name:
                return int(name.split(".blocks.")[1].split(".")[0]) + 1
            return top

        groups: dict[int, dict] = {}
        for n, p in self.named_parameters():
            lid = level(n)
            g = groups.setdefault(
                lid,
                {
                    "params": [],
                    "lr": lr * layer_decay ** (top - lid),
                    "name": f"backbone_L{lid}",
                },
            )
            g["params"].append(p)
        return [groups[k] for k in sorted(groups)]

    def forward(self, x: Tensor) -> BackboneOutput:
        _, _, h_in, w_in = x.shape
        h_patches = h_in // self._patch_size
        w_patches = w_in // self._patch_size

        if self._return_intermediate:
            # [P0: Multi-layer fusion] list gives specific block outputs;
            # int gives last N blocks (backward compat)
            n = self._intermediate_layer_indices or self._n_intermediate_layers
            feats = self._model.get_intermediate_layers(  # type: ignore[operator]
                x,
                n=n,
                reshape=False,
            )
            features = list(feats)
        else:
            out = self._model.forward_features(x)  # type: ignore[operator]
            features = [out["x_norm_patchtokens"]]

        return BackboneOutput(
            features=features,
            spatial_shape=(h_patches, w_patches),
        )


def build_backbone(
    name: str,
    return_intermediate: bool = False,
    intermediate_layer_indices: list[int] | None = None,
) -> DINOv2Backbone:
    """Factory: pick backbone by model name."""
    if "dinov2" in name:
        return DINOv2Backbone(
            model_name=name,
            return_intermediate=return_intermediate,
            intermediate_layer_indices=intermediate_layer_indices,
        )
    raise ValueError(f"Unknown backbone: {name!r}. Expected 'dinov2*'.")

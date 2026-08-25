"""Backbone wrappers producing structured multi-level features.

DINOv2 (ViT) via torch.hub, DINOv3 (ViT) via timm, ConvNeXtV2 (CNN) via timm.
All produce BackboneOutput with a list of feature tensors at different depths.
Projection layer (neck.py) adapts these into the shared d_model
space for the cross-attention neck.

Upstream: none (pretrained weights from torch.hub / timm).
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

    features: multi-level feature tensors in backbone-native format.
        DINOv2: list of (B, N, C) flat token sequences (1 or N_intermediate).
        ConvNeXtV2: list of (B, C_i, H_i, W_i) spatial feature maps (4 stages).
        Projection reads features[-1] (or multiple levels for FPN);
        MIRO reads all levels for regularization.
    spatial_shape: (H, W) spatial dims of features[-1].
    """

    features: list[Tensor]
    spatial_shape: tuple[int, int]


class DINOv2Backbone(nn.Module):
    """DINOv2 backbone via torch.hub.

    Input:  (B, 3, H, W) where H, W are divisible by patch_size (14).
    Output: BackboneOutput with flat patch tokens (B, N, C_backbone).

    return_intermediate=False → features = [last_block_output].
    return_intermediate=True  → features = [block_{-N}, ..., block_{-1}] (for MIRO).

    Upstream: none (pretrained).
    Downstream: neck.py (LinearNeck).
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
            # int gives last N blocks (MIRO backward compat)
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


class DINOv3Backbone(nn.Module):
    """DINOv3 backbone via timm (weights are gated on torch.hub CDN).

    Same ViT architecture as DINOv2 (patch_size=16 vs 14, distilled from 7B
    teacher). Loaded via timm which mirrors the weights without gating.
    timm's features_only mode returns (B, C, H, W); we reshape to (B, N, C)
    to match DINOv2Backbone's output contract.

    Input:  (B, 3, H, W) where H, W are divisible by 16.
    Output: BackboneOutput with flat patch tokens (B, N, C_backbone).

    Upstream: none (pretrained via timm).
    Downstream: neck.py (MultiLayerNeck / LinearNeck).
    """

    # timm model name for each config shorthand
    _TIMM_NAMES: dict[str, str] = {
        "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
        "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
        "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    }

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        return_intermediate: bool = False,
        intermediate_layer_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        import timm

        self._model_name = model_name
        timm_name = self._TIMM_NAMES.get(model_name)
        if timm_name is None:
            raise ValueError(
                f"Unknown DINOv3 model: {model_name!r}. Supported: {sorted(self._TIMM_NAMES)}"
            )

        self._return_intermediate = return_intermediate
        self._intermediate_layer_indices = intermediate_layer_indices

        # Always create plain model first for metadata; features_only wraps
        # the model in FeatureGetterNet which hides embed_dim/patch_size.
        plain = timm.create_model(timm_name, pretrained=True)
        self._embed_dim: int = plain.embed_dim
        self._patch_size: int = plain.patch_embed.patch_size[0]
        self._n_blocks = len(plain.blocks)
        n_blocks = self._n_blocks

        if return_intermediate and intermediate_layer_indices:
            del plain
            self._model = timm.create_model(
                timm_name,
                pretrained=True,
                features_only=True,
                out_indices=intermediate_layer_indices,
            )
            self._features_only = True
        else:
            self._model = plain
            self._features_only = False

        logger.info(
            "DINOv3: %s (timm: %s), embed_dim=%d, patch_size=%d, blocks=%d",
            model_name,
            timm_name,
            self._embed_dim,
            self._patch_size,
            n_blocks,
        )

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

    def train(self, mode: bool = True) -> DINOv3Backbone:
        if getattr(self, "_is_frozen", False):
            return super().train(False)
        return super().train(mode)

    def param_groups(self, lr: float, layer_decay: float = 1.0) -> list[dict]:
        """Param groups with optional layer-wise LR decay.

        Same structure as DINOv2: embeddings=level 0, blocks.i=level i+1,
        norm=top. timm uses the same `model.blocks.N` naming convention.
        """
        if layer_decay >= 1.0:
            return [{"params": list(self.parameters()), "lr": lr, "name": "backbone"}]

        n_blocks = self._n_blocks
        top = n_blocks + 1

        embed_keys = ("patch_embed", "pos_embed", "cls_token", "mask_token", "reg_token")

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

        if self._features_only:
            # features_only mode → list of (B, C, H, W)
            spatial_feats: list[Tensor] = self._model(x)
            # Reshape to (B, N, C) to match DINOv2's flat token format
            features = [f.flatten(2).transpose(1, 2) for f in spatial_feats]
        else:
            out = self._model.forward_features(x)
            # timm returns (B, 1+num_reg+N, C) with CLS/register prefix
            n_tokens = h_patches * w_patches
            features = [out[:, -n_tokens:, :]]

        return BackboneOutput(
            features=features,
            spatial_shape=(h_patches, w_patches),
        )


class ConvNeXtV2Backbone(nn.Module):
    """ConvNeXtV2 backbone via timm — hierarchical multi-stage features.

    Outputs all 4 stages as spatial feature maps (B, C_i, H_i, W_i).
    FPNNeck selects and fuses the desired stages downstream.

    Stage channels (Base): [128, 256, 512, 1024] at strides [4, 8, 16, 32].
    Stage depths  (Base): [3, 3, 27, 3] blocks.

    Upstream: none (pretrained via timm).
    Downstream: neck.py (FPNNeck fuses selected stages).
    """

    def __init__(
        self,
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k_384",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self._model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        self._stage_channels: list[int] = self._model.feature_info.channels()
        self._stage_strides: list[int] = self._model.feature_info.reduction()
        self._embed_dim = self._stage_channels[-1]

        # Discover block counts from parameter names.
        # timm features_only uses "stages_0.blocks.0..." (underscore for stage index).
        stage_max_block: dict[int, int] = {}
        import re

        _stage_block_re = re.compile(r"stages_(\d+)\.blocks\.(\d+)\.")
        for n, _ in self._model.named_parameters():
            m = _stage_block_re.search(n)
            if m:
                s, b = int(m.group(1)), int(m.group(2))
                stage_max_block[s] = max(stage_max_block.get(s, 0), b)
        self._blocks_per_stage = [
            stage_max_block.get(s, 0) + 1 for s in range(len(self._stage_channels))
        ]

        logger.info(
            "ConvNeXtV2: %s, channels=%s, strides=%s, blocks=%s",
            model_name,
            self._stage_channels,
            self._stage_strides,
            self._blocks_per_stage,
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def stage_channels(self) -> list[int]:
        return list(self._stage_channels)

    @property
    def stage_strides(self) -> list[int]:
        return list(self._stage_strides)

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

    def train(self, mode: bool = True) -> ConvNeXtV2Backbone:
        if getattr(self, "_is_frozen", False):
            return super().train(False)
        return super().train(mode)

    # Group-wise decay: every GROUP_SIZE consecutive blocks share one LR
    # multiplier, aligning the number of decay groups with ViT (~12) so the
    # same layer_decay value has comparable effect across architectures.
    _GROUP_SIZE: int = 3

    def param_groups(self, lr: float, layer_decay: float = 1.0) -> list[dict]:
        """Param groups with group-wise LR decay (ConvNeXtV2 official recipe).

        layer_decay=1.0: single group.
        layer_decay<1.0: every _GROUP_SIZE consecutive blocks share one decay
        level.  ConvNeXtV2-Base [3,3,27,3]=36 blocks ÷ 3 = 12 groups + stem
        + top = 14 levels — comparable to ViT-B's 12 block levels.
        Downsample layers share the group of their stage's first block.
        """
        if layer_decay >= 1.0:
            return [{"params": list(self.parameters()), "lr": lr, "name": "backbone"}]

        bps = self._blocks_per_stage
        n_groups = sum(bps) // self._GROUP_SIZE
        top = n_groups + 1

        import re

        _sb_re = re.compile(r"stages_(\d+)\.blocks\.(\d+)")
        _sd_re = re.compile(r"stages_(\d+)\.downsample")

        def group_id(name: str) -> int:
            if "stem" in name:
                return 0
            m = _sb_re.search(name)
            if m:
                s, b = int(m.group(1)), int(m.group(2))
                flat = sum(bps[:s]) + b
                return flat // self._GROUP_SIZE + 1
            m = _sd_re.search(name)
            if m:
                s = int(m.group(1))
                flat = sum(bps[:s])
                return flat // self._GROUP_SIZE + 1
            return top

        groups: dict[int, dict] = {}
        for n, p in self.named_parameters():
            gid = group_id(n)
            g = groups.setdefault(
                gid,
                {
                    "params": [],
                    "lr": lr * layer_decay ** (top - gid),
                    "name": f"backbone_G{gid}",
                },
            )
            g["params"].append(p)
        return [groups[k] for k in sorted(groups)]

    def forward(self, x: Tensor) -> BackboneOutput:
        features: list[Tensor] = self._model(x)
        last = features[-1]
        spatial_shape = (last.shape[2], last.shape[3])
        return BackboneOutput(features=features, spatial_shape=spatial_shape)


def build_backbone(
    name: str,
    pretrained: bool = True,
    return_intermediate: bool = False,
    intermediate_layer_indices: list[int] | None = None,
) -> DINOv2Backbone | DINOv3Backbone | ConvNeXtV2Backbone:
    """Factory: pick backbone by model name.

    return_intermediate applies to DINOv2/v3 only (for MIRO / multi-layer fusion);
    ConvNeXtV2 always returns all stage features.
    """
    if "dinov3" in name:
        return DINOv3Backbone(
            model_name=name,
            return_intermediate=return_intermediate,
            intermediate_layer_indices=intermediate_layer_indices,
        )
    if "dinov2" in name:
        return DINOv2Backbone(
            model_name=name,
            return_intermediate=return_intermediate,
            intermediate_layer_indices=intermediate_layer_indices,
        )
    if "convnext" in name:
        return ConvNeXtV2Backbone(
            model_name=name,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown backbone: {name!r}. Expected 'dinov2*', 'dinov3*', or 'convnext*'.")

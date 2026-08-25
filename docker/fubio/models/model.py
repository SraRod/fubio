"""FUBioModel — per-task architecture for multi-task biometry.

Layer 1: Backbone  → patch embeddings (dense, shared, pretrained).
Layer 2: Neck      → spatial memory (dense, shared blackboard — read by
                     Layer 3 AND Layer 4 for spatial grounding).
Layer 3: Decoder   → task queries × memory → per-task features
                     (cross-attn, shared weights, task-agnostic;
                      queries are task-specific, owned by TaskModule).
Layer 4: Task Head → refinement + prediction (per-task self-attn,
                     coord/conf/bbox predictors; reads memory for
                     spatial grounding via coord_predictor).

Forward: all tasks' queries are batched through the shared decoder
(no mask — cross-attn is per-query independent), then split to
per-task TaskModules for refinement and prediction.

Upstream: backbone.py, neck.py, decoder.py, heads.py.
Downstream: train/module.py (training loop).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fubio.data.shape_prior import ShapePrior
from fubio.data.task_registry import TASKS
from fubio.data.types import TaskOutput
from fubio.models.backbone import BackboneOutput, build_backbone
from fubio.models.coord_predictors import build_coord_predictor
from fubio.models.decoder import CrossAttentionDecoder
from fubio.models.heads import TaskModule
from fubio.models.neck import NeckOutput, build_neck
from fubio.train.config import (
    C2fNeckConfig,
    CoordConfig,
    DirectCoordConfig,
    GeoSimCCCoordConfig,
    MultiLayerNeckConfig,
    NeckModeConfig,
)


class ModelOutput(NamedTuple):
    """Structured return from FUBioModel.forward."""

    task_outputs: dict[str, TaskOutput]
    backbone_out: BackboneOutput
    spatial_shape: tuple[int, int]


class FUBioModel(nn.Module):
    """Per-task multi-task biometry model.

    forward(images) → ModelOutput(task_outputs, backbone_out, spatial_shape)

    All 9 tasks run in a single forward pass:
      1. Backbone + Neck produce shared spatial memory (blackboard).
      2. Task queries (owned by TaskModules) are batched through a shared
         cross-attention decoder — task-agnostic, no mask needed.
      3. Per-task heads refine features and predict coords/conf/bbox,
         reading memory again for spatial grounding.

    Upstream: all models/* submodules.
    Downstream: train/module.py.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        head_ffn_dim: int = 1024,
        n_decoder_layers: int = 3,
        n_head_layers: int = 2,
        n_inst: int = 2,
        use_uncertainty: bool = False,
        derive_bbox: bool = False,
        use_affine: bool = False,
        return_intermediate: bool = False,
        coord_config: CoordConfig | None = None,
        shape_prior: ShapePrior | None = None,
        neck_config: NeckModeConfig | None = None,
        neck_dropout: float = 0.1,
        dropout: float = 0.1,
        conf_mlp_layers: int = 1,
        input_size: int = 518,
        use_supportive: bool = False,
    ) -> None:
        super().__init__()
        self.n_inst = n_inst
        _coord_config = coord_config or DirectCoordConfig()

        # Layer 1: Backbone → patch embeddings
        _need_intermediate = return_intermediate
        _layer_indices: list[int] | None = None
        if isinstance(neck_config, (MultiLayerNeckConfig, C2fNeckConfig)):
            _need_intermediate = True
            _layer_indices = neck_config.layer_indices
        self.backbone = build_backbone(
            name=backbone_name,
            return_intermediate=_need_intermediate,
            intermediate_layer_indices=_layer_indices,
        )

        # Layer 2: Neck → shared spatial memory (blackboard for Layer 3 + 4)
        if neck_config is None:
            from fubio.train.config import LinearNeckConfig

            neck_config = LinearNeckConfig()
        self.neck = build_neck(
            config=neck_config,
            backbone=self.backbone,
            d_model=d_model,
            dropout=neck_dropout,
        )

        # Layer 3: Decoder (shared cross-attention weights, task-agnostic)
        self.decoder = CrossAttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_decoder_layers,
            dropout=dropout,
        )

        # Shared localisation key projection. GeoSimCC's stage 1 correlates each
        # landmark query against memory; the key side of that is task-agnostic
        # work, so it lives with the decoder rather than being repeated inside
        # all nine TaskModules (which projected the same (B,S,D) nine times).
        # Task specificity stays in the per-task query, as everywhere else.
        self._geo = isinstance(_coord_config, GeoSimCCCoordConfig)
        if self._geo:
            self.loc_k_proj = nn.Linear(d_model, d_model)
            # High-resolution map for stage-3 profile sampling. Deliberately NOT
            # part of `memory`: attention over it would cost O(S) at this
            # resolution, but sampling costs only O(K * n_profile), so the fine
            # detail is affordable exactly where it is needed. The stem reads the
            # image directly — the backbone's patch embedding has already
            # discarded sub-patch structure by the time memory exists.
            _st = _coord_config.fine_stride
            _df = _coord_config.d_fine
            self.fine_stem = nn.Sequential(
                nn.Conv2d(3, _df // 2, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(_df // 2, _df, 3, stride=_st // 2, padding=1),
            )
            self.fine_proj = nn.Conv2d(d_model, _df, 1)
            self._fine_stride = _st

        # Layer 4: Per-task modules (queries + refiner + predictors)
        # use_supportive=True → n_total (scored + supportive);
        # False → n_keypoints (scored only, for loading pre-supportive checkpoints).
        _use_anchors = shape_prior is not None
        self.tasks = nn.ModuleDict(
            {
                tid: TaskModule(
                    n_keypoints=tdef.n_total if use_supportive else tdef.n_keypoints,
                    d_model=d_model,
                    n_inst=n_inst,
                    n_head_layers=n_head_layers,
                    n_heads=n_heads,
                    ffn_dim=head_ffn_dim,
                    dropout=dropout,
                    task_shape_prior=shape_prior.tasks.get(tid) if shape_prior else None,
                    use_anchors=_use_anchors,
                    derive_bbox=derive_bbox,
                    use_affine=use_affine,
                    bbox_context_scale=tdef.bbox_context_scale,
                    conf_mlp_layers=conf_mlp_layers,
                    coord_predictor=build_coord_predictor(
                        config=_coord_config,
                        d_model=d_model,
                        n_keypoints=tdef.n_total if use_supportive else tdef.n_keypoints,
                        task_shape_prior=shape_prior.tasks.get(tid) if shape_prior else None,
                        input_size=input_size,
                        dropout=dropout,
                        task_id=tid,
                        d_fine=(
                            _coord_config.d_fine
                            if isinstance(_coord_config, GeoSimCCCoordConfig)
                            else None
                        ),
                        own_keys=False,
                    ),
                    scored_mask=tdef.scored_mask if use_supportive else None,
                )
                for tid, tdef in TASKS.items()
            }
        )

    def _batched_decode(self, neck_out: NeckOutput) -> dict[str, Tensor]:
        """Gather all task queries, batch through shared decoder, split back.

        Cross-attention softmax normalizes over memory positions (keys), so
        concatenating queries from different tasks is mathematically identical
        to running them separately — but ~2-4x faster on GPU (one kernel
        launch, one memory read).
        """
        B = neck_out.memory.shape[0]
        task_ids = list(self.tasks.keys())

        query_parts: list[Tensor] = []
        pos_parts: list[Tensor] = []
        sizes: list[int] = []
        for tid in task_ids:
            q, qp = self.tasks[tid].get_queries(B)
            query_parts.append(q)
            pos_parts.append(qp)
            sizes.append(self.tasks[tid].n_queries)

        all_q = torch.cat(query_parts, dim=1)
        all_qp = torch.cat(pos_parts, dim=1)

        all_feats = self.decoder(
            all_q,
            neck_out.memory,
            neck_out.memory_pos,
            query_pos=all_qp,
        )

        return dict(zip(task_ids, all_feats.split(sizes, dim=1), strict=True))

    def forward(self, images: Tensor) -> ModelOutput:
        """Single-pass forward: all tasks at once.

        Args:
            images: (B, 3, H, W) normalized float32.

        Returns:
            ModelOutput with task_outputs dict, raw backbone_out, and spatial_shape.
        """
        # Layer 1+2: image → shared spatial memory
        backbone_out = self.backbone(images)
        neck_out = self.neck(backbone_out)

        # Layer 3: task queries × memory → per-task features (batched)
        task_feats = self._batched_decode(neck_out)

        # Shared, built once: localisation keys and the high-resolution map that
        # stage 3 samples. Both are task-agnostic reads of the image/memory, so
        # computing them per task would be pure duplication.
        _mem_k = _fine = None
        if self._geo:
            _mem_k = self.loc_k_proj(neck_out.memory + neck_out.memory_pos)
            _hp, _wp = neck_out.spatial_shape
            _m2d = neck_out.memory.transpose(1, 2).reshape(images.shape[0], -1, _hp, _wp)
            _fine_stem_out = self.fine_stem(images)
            _fh, _fw = _fine_stem_out.shape[-2], _fine_stem_out.shape[-1]
            _fine = _fine_stem_out + F.interpolate(
                self.fine_proj(_m2d), size=(_fh, _fw), mode="bilinear", align_corners=False
            )

        # Layer 4: per-task refinement + prediction (reads memory for spatial grounding)
        results: dict[str, TaskOutput] = {}
        for tid, feat in task_feats.items():
            results[tid] = self.tasks[tid].head(
                feat,
                neck_out.memory,
                neck_out.spatial_shape,
                neck_out.memory_pos,
                mem_k=_mem_k,
                fine=_fine,
            )

        return ModelOutput(
            task_outputs=results,
            backbone_out=backbone_out,
            spatial_shape=neck_out.spatial_shape,
        )

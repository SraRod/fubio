"""Per-task modules — query ownership + refinement + prediction.

TaskModule is the single per-task unit: it owns the task's queries
(nn.Embedding + learnable anchor positions), per-task self-attention
refinement (TaskRefinerLayer), and output predictors (coords/conf/bbox).

Adding a task = registering a TaskDef → model auto-creates a TaskModule.
The shared decoder is task-agnostic and has no knowledge of tasks.

Upstream: decoder.py (CrossAttentionDecoder output, TaskRefinerLayer).
Downstream: model.py (assembles per-task TaskOutput).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from fubio.data.shape_prior import TaskShapePrior
from fubio.data.types import TaskOutput
from fubio.models.coord_predictors import GeoSimCCPredictor
from fubio.models.decoder import TaskRefinerLayer
from fubio.models.queries import fourier_position_encoding


def _derive_bbox(landmarks: Tensor, pad_frac: float = 0.05, context_scale: float = 1.0) -> Tensor:
    """Derive cxcywh bbox as tight bound of predicted landmarks.

    Differentiable — gradients flow back to landmark coords via min/max.
    context_scale expands the landmark span proportionally (matching the GT
    bbox convention from collate) before applying canvas padding.

    Args:
        landmarks: (B, N_inst, K, 2) in [0, 1].
        pad_frac: fractional padding around tight bound.
        context_scale: multiplicative expansion on landmark span (per-task).

    Returns:
        (B, N_inst, 4) cxcywh in [0, 1].
    """
    x_min = landmarks[..., 0].min(dim=-1).values
    x_max = landmarks[..., 0].max(dim=-1).values
    y_min = landmarks[..., 1].min(dim=-1).values
    y_max = landmarks[..., 1].max(dim=-1).values

    if context_scale != 1.0:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        half_w = (x_max - x_min) / 2 * context_scale
        half_h = (y_max - y_min) / 2 * context_scale
        x_min = cx - half_w
        x_max = cx + half_w
        y_min = cy - half_h
        y_max = cy + half_h

    x_min = (x_min - pad_frac).clamp(min=0.0)
    x_max = (x_max + pad_frac).clamp(max=1.0)
    y_min = (y_min - pad_frac).clamp(min=0.0)
    y_max = (y_max + pad_frac).clamp(max=1.0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1)


def derive_bbox_masked(
    landmarks: Tensor,
    vis_mask: Tensor,
    pad_frac: float = 0.05,
    context_scale: float = 1.0,
) -> Tensor:
    """Derive cxcywh bbox from visible landmarks only. Differentiable.

    Used during loss computation to align pred bbox with GT bbox that
    was computed from visible keypoints only.

    Args:
        landmarks: (N_matched, K, 2) in [0, 1].
        vis_mask: (N_matched, K) bool — True = visible.
        pad_frac: fractional padding.
        context_scale: multiplicative expansion on landmark span (per-task).

    Returns:
        (N_matched, 4) cxcywh in [0, 1].
    """
    N, K, _ = landmarks.shape
    device = landmarks.device

    big = torch.tensor(float("inf"), device=device)

    x = landmarks[..., 0]  # (N, K)
    y = landmarks[..., 1]  # (N, K)

    x_for_min = x.where(vis_mask, big)
    x_for_max = x.where(vis_mask, -big)
    y_for_min = y.where(vis_mask, big)
    y_for_max = y.where(vis_mask, -big)

    x_min = x_for_min.min(dim=-1).values
    x_max = x_for_max.max(dim=-1).values
    y_min = y_for_min.min(dim=-1).values
    y_max = y_for_max.max(dim=-1).values

    no_visible = ~vis_mask.any(dim=-1)  # (N,)
    x_min = x_min.where(~no_visible, torch.zeros_like(x_min))
    x_max = x_max.where(~no_visible, torch.ones_like(x_max))
    y_min = y_min.where(~no_visible, torch.zeros_like(y_min))
    y_max = y_max.where(~no_visible, torch.ones_like(y_max))

    if context_scale != 1.0:
        cx_t = (x_min + x_max) / 2
        cy_t = (y_min + y_max) / 2
        half_w = (x_max - x_min) / 2 * context_scale
        half_h = (y_max - y_min) / 2 * context_scale
        x_min = cx_t - half_w
        x_max = cx_t + half_w
        y_min = cy_t - half_h
        y_max = cy_t + half_h

    x_min = (x_min - pad_frac).clamp(min=0.0)
    x_max = (x_max + pad_frac).clamp(max=1.0)
    y_min = (y_min - pad_frac).clamp(min=0.0)
    y_max = (y_max + pad_frac).clamp(max=1.0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1)


class TaskModule(nn.Module):
    """Complete per-task definition: queries + refiner + predictors.

    Owns everything task-specific. The shared CrossAttentionDecoder is
    task-agnostic and has no knowledge of this class.

    Query parameters:
      - query_embedding: learned content embeddings, one per query slot.
      - anchor_logits (optional): sigmoid-bounded learnable positions,
        initialized from shape prior mean_xy. Encoded via Fourier features
        matching the neck's memory positional encoding convention.

    Head pipeline:
      1. TaskRefinerLayer × N: per-task self-attention (landmark coordination)
      2. Split into instance / landmark features
      3. Predict bbox, conf, landmark coords

    Input: (B, n_queries, D) — decoder cross-attention output for this task.
    Output: TaskOutput(bbox, conf, landmarks, residual, heatmap, ...).
    """

    # Attributes for parameter grouping: query_params() returns parameters
    # that should be trained at lr_decoder; all others at lr_heads.
    QUERY_PARAM_NAMES = frozenset({"query_embedding.weight", "anchor_logits"})

    def __init__(
        self,
        n_keypoints: int,
        d_model: int = 256,
        n_inst: int = 2,
        n_head_layers: int = 1,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        task_shape_prior: TaskShapePrior | None = None,
        use_anchors: bool = False,
        derive_bbox: bool = False,
        bbox_context_scale: float = 1.0,
        coord_predictor: GeoSimCCPredictor | None = None,
    ) -> None:
        super().__init__()
        self.n_inst = n_inst
        self.K = n_keypoints
        self._d_model = d_model
        self._use_anchors = use_anchors
        self.derive_bbox = derive_bbox
        self._bbox_context_scale = bbox_context_scale

        # --- Query parameters (from TaskQueryBank) ---
        self.n_queries = n_inst * (1 + n_keypoints)
        self.query_embedding = nn.Embedding(self.n_queries, d_model)

        if use_anchors:
            K = n_keypoints
            if task_shape_prior and task_shape_prior.mean_xy:
                mean_xy = torch.tensor(task_shape_prior.mean_xy, dtype=torch.float32)
                lm_logits = torch.logit(mean_xy.clamp(1e-4, 1 - 1e-4))
                centroid_xy = mean_xy.mean(dim=0, keepdim=True)
                inst_logit = torch.logit(centroid_xy.clamp(1e-4, 1 - 1e-4))
            else:
                lm_logits = torch.zeros(K, 2)
                inst_logit = torch.zeros(1, 2)

            per_slot = torch.cat([inst_logit, lm_logits], dim=0)  # (1+K, 2)
            anchor = per_slot.unsqueeze(0).expand(n_inst, -1, -1).clone()
            anchor = anchor.reshape(self.n_queries, 2)
            self.anchor_logits = nn.Parameter(anchor)

        # --- Head: refiner + predictors (from TaskHead) ---
        self.self_attn_layers = nn.ModuleList(
            [TaskRefinerLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_head_layers)]
        )

        self.feat_dropout = nn.Dropout(dropout)

        if not derive_bbox:
            self.bbox_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 4),
            )
        self.conf_head = nn.Linear(d_model, 1)

        if coord_predictor is None:
            raise ValueError("TaskModule requires a coord_predictor (GeoSimCCPredictor)")
        self.coord_predictor = coord_predictor

    def query_params(self) -> list[nn.Parameter]:
        """Parameters that should be trained at lr_decoder (not lr_heads)."""
        params = [self.query_embedding.weight]
        if self._use_anchors:
            params.append(self.anchor_logits)
        return params

    def reinit_learned_params(self, task_shape_prior: TaskShapePrior | None = None) -> None:
        """Reset all learned parameters to their initial state, preserving buffers.

        Buffers (canon_mean, canon_basis, partner, scored_mask) are shape-prior
        or config data and must survive. Learned weights/biases are reinitialized
        to match the constructor's init logic.
        """
        # Query embedding: default nn.Embedding init (normal)
        self.query_embedding.reset_parameters()

        # Anchor logits: reinit from shape prior mean_xy
        if self._use_anchors:
            K = self.K
            if task_shape_prior and task_shape_prior.mean_xy:
                mean_xy = torch.tensor(task_shape_prior.mean_xy, dtype=torch.float32)
                lm_logits = torch.logit(mean_xy.clamp(1e-4, 1 - 1e-4))
                centroid_xy = mean_xy.mean(dim=0, keepdim=True)
                inst_logit = torch.logit(centroid_xy.clamp(1e-4, 1 - 1e-4))
            else:
                lm_logits = torch.zeros(K, 2)
                inst_logit = torch.zeros(1, 2)
            per_slot = torch.cat([inst_logit, lm_logits], dim=0)
            anchor = per_slot.unsqueeze(0).expand(self.n_inst, -1, -1).clone()
            self.anchor_logits.data.copy_(anchor.reshape(self.n_queries, 2))

        # Self-attention refiner layers
        for layer in self.self_attn_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # Confidence head
        nn.init.xavier_uniform_(self.conf_head.weight)
        nn.init.zeros_(self.conf_head.bias)

        # Bbox MLP
        if hasattr(self, "bbox_mlp"):
            for m in self.bbox_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Coord predictor: reinit learned params, keep buffers
        cp = self.coord_predictor
        for m in cp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # GeoSimCC alpha_head: zero bias is intentional (start at mean shape)
        if isinstance(cp, GeoSimCCPredictor):
            nn.init.zeros_(cp.alpha_head.bias)

    def get_queries(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Build query content + positional embeddings for this task.

        Returns:
            content: (B, n_queries, D) — learned content embeddings.
            query_pos: (B, n_queries, D) — Fourier-encoded anchor positions,
                or zeros when anchors are disabled (no-op under addition).
        """
        q = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        if self._use_anchors:
            anchor_xy = self.anchor_logits.sigmoid()
            pos = fourier_position_encoding(anchor_xy, self._d_model)
            query_pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            query_pos = torch.zeros_like(q)

        return q, query_pos

    def head(
        self,
        attended: Tensor,
        memory: Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
        memory_pos: Tensor | None = None,
        mem_k: Tensor | None = None,
        fine: Tensor | None = None,
    ) -> TaskOutput:
        """Per-task refinement + prediction.

        attended: (B, n_queries, D) — cross-attention output for this task.
        memory: (B, S, D) spatial tokens — required by HeatmapPredictor.
        spatial_shape: (Hp, Wp) patch grid — required by HeatmapPredictor.
        memory_pos: (1, S, D) positional encoding — required by HeatmapPredictor.
        """
        B = attended.shape[0]

        # Per-task self-attention (landmark coordination)
        x = attended
        for layer in self.self_attn_layers:
            x = layer(x)

        # Split into instance / landmark features
        x = x.view(B, self.n_inst, 1 + self.K, -1)
        inst_feat = self.feat_dropout(x[:, :, 0, :])  # (B, N_inst, D)
        lm_feat = self.feat_dropout(x[:, :, 1:, :])  # (B, N_inst, K, D)

        conf = self.conf_head(inst_feat)
        coords, extra = self.coord_predictor(
            inst_feat,
            lm_feat,
            memory,
            spatial_shape,
            memory_pos=memory_pos,
            mem_k=mem_k,
            fine=fine,
        )

        # GeoSimCC returns its stage-1 heat here on purpose: routing it to
        # TaskOutput.heatmap is what makes lambda_heatmap supervise the
        # geometric locator directly, with no prior mixed into the sum.
        bbox = (
            _derive_bbox(coords, context_scale=self._bbox_context_scale)
            if self.derive_bbox
            else self.bbox_mlp(inst_feat).sigmoid()
        )

        return TaskOutput(
            bbox=bbox,
            conf=conf,
            landmarks=coords,
            heatmap=extra,
        )

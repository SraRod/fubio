"""Cross-attention decoder and per-task refiner — Layers 3 and 4.

CrossAttentionDecoder (Layer 3): task-agnostic feature extraction.
Cross-attn (to spatial memory) + FFN. All tasks' queries share the same
weights. No self-attention, no task mask — cross-attention is per-query
independent, so concatenated queries from different tasks never interact.

TaskRefinerLayer (Layer 4): per-task within-task self-attention + FFN.
Sparse → sparse, per-task. No cross-attention — this layer does not see
the image. It refines inter-query relationships (instance <-> landmark
interaction) using per-task weights.

Upstream: neck.py (memory, memory_pos), heads.py (TaskModule queries).
Downstream: heads.py (TaskModule consumes decoder output).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class CrossAttentionLayer(nn.Module):
    """Cross-attn → FFN. Task-agnostic — no self-attention, no task mask.

    Each query independently attends to the full spatial memory.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """
        queries:    (B, N_q, D) — task query content (any subset or all tasks).
        memory:     (B, S, D) — spatial tokens from neck.
        memory_pos: (1, S, D) — sinusoidal positional encoding.
        query_pos:  (B, N_q, D) | None — anchor-derived position encoding.
        Returns:    (B, N_q, D)
        """
        q_for_attn = queries + query_pos if query_pos is not None else queries

        ca_out, _ = self.cross_attn(
            query=q_for_attn,
            key=memory + memory_pos,
            value=memory,
        )
        x = self.cross_attn_norm(queries + self.dropout(ca_out))

        x = self.ffn_norm(x + self.ffn(x))
        return x


class CrossAttentionDecoder(nn.Module):
    """Layer 3: task-agnostic cross-attention feature extractor.

    N layers of CrossAttentionLayer. Queries from all tasks can be
    concatenated and processed in a single batched call — cross-attention
    is per-query independent (softmax normalizes over memory positions,
    not across queries), so no mask is needed.

    forward(queries, memory, memory_pos) → (B, N_q, D)

    Upstream: neck.py (memory, memory_pos), heads.py (TaskModule queries).
    Downstream: model.py splits output per-task → TaskModule heads.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        out = queries
        for layer in self.layers:
            out = layer(out, memory, memory_pos, query_pos=query_pos)
        return out


class TaskRefinerLayer(nn.Module):
    """Layer 4: per-task self-attention refinement.

    Sparse → sparse, per-task. No cross-attention — does not access the
    image. Operates on attended features from the shared decoder, refining
    inter-query relationships (instance <-> landmark interaction) using
    per-task weights.

    Upstream: CrossAttentionDecoder output (split per-task).
    Downstream: coord_predictors.py (landmark coordinate prediction).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N_q_task, D) → (B, N_q_task, D)"""
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(sa_out))
        x = self.norm2(x + self.ffn(x))
        return x

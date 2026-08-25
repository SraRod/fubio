"""Phase-aware LR scheduling: warmup + cosine with backbone freeze.

Phase 0 (epoch < freeze_epochs): backbone frozen, only decoder/heads train.
Phase 1 (epoch >= freeze_epochs): backbone unfreezes, differential LR.

Upstream: train/config.py (OptimizerConfig, BackboneConfig).
Downstream: train/module.py (configure_optimizers).
"""

from __future__ import annotations

import math

import torch.optim


class PhaseScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Per-group LR multiplier with backbone freeze awareness.

    The optimizer must have at least three parameter groups in order:
    [backbone, neck, heads, ...extra]. Extra groups (e.g. MIRO encoders)
    follow the same schedule as neck/heads.

    - **Backbone** (group 0): multiplier = 0 during Phase 0 (``step < freeze_steps``).
      After Phase 0: linear warmup 0 → 1 over ``warmup_steps``, then cosine → 0.
    - **All other groups**: linear warmup 0 → 1 from step 0,
      then cosine → 0 over the remaining budget.

    All multipliers are applied to the *base LR* already set in each param group.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        freeze_steps: int,
        eta_min_ratio: float = 0.0,
        n_backbone_groups: int = 1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.freeze_steps = freeze_steps
        self.eta_min_ratio = eta_min_ratio

        # First n_backbone_groups follow the freeze-aware backbone schedule
        # (LLRD produces one group per ViT layer, all placed before neck/heads).
        n_groups = len(optimizer.param_groups)
        lr_lambdas = [self._backbone_lambda for _ in range(n_backbone_groups)] + [
            self._decoder_heads_lambda for _ in range(n_groups - n_backbone_groups)
        ]
        super().__init__(optimizer, lr_lambdas, last_epoch=last_epoch)

    # -- lambda factories -------------------------------------------------

    def _backbone_lambda(self, step: int) -> float:
        """Backbone: zero during Phase 0, then warmup + cosine."""
        if step < self.freeze_steps:
            return 0.0
        # Steps elapsed since unfreeze
        elapsed = step - self.freeze_steps
        return self._warmup_cosine(elapsed)

    def _decoder_heads_lambda(self, step: int) -> float:
        """Decoder / heads: warmup + cosine from step 0."""
        return self._warmup_cosine(step)

    def _warmup_cosine(self, elapsed: int) -> float:
        """Linear warmup eta_min→1, then cosine decay 1→eta_min."""
        r = self.eta_min_ratio
        if self.warmup_steps > 0 and elapsed < self.warmup_steps:
            return r + (1.0 - r) * elapsed / self.warmup_steps
        remaining = self.total_steps - self.warmup_steps
        if remaining <= 0:
            return 1.0
        progress = (elapsed - self.warmup_steps) / remaining
        progress = min(max(progress, 0.0), 1.0)
        return r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * progress))

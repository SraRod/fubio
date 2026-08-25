"""Phase D verification: PhaseScheduler covers all backbone groups (LLRD)."""

from __future__ import annotations

import torch

from fubio.train.schedule import PhaseScheduler


def _dummy_optimizer(n_groups: int) -> torch.optim.Optimizer:
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3} for _ in range(n_groups)]
    return torch.optim.SGD(groups)


def test_freeze_applies_to_all_backbone_groups() -> None:
    """With n_backbone_groups=3, all three backbone groups stay frozen in Phase 0."""
    opt = _dummy_optimizer(5)  # 3 backbone (LLRD) + neck + heads
    sched = PhaseScheduler(
        optimizer=opt,
        warmup_steps=10,
        total_steps=100,
        freeze_steps=20,
        n_backbone_groups=3,
    )
    sched.step()  # step 0

    lrs = [g["lr"] for g in opt.param_groups]
    assert all(lr == 0.0 for lr in lrs[:3])  # backbone frozen during Phase 0
    assert all(lr > 0.0 for lr in lrs[3:])  # neck + heads warm up from step 0


def test_backbone_unfreezes_after_freeze_steps() -> None:
    opt = _dummy_optimizer(5)
    sched = PhaseScheduler(
        optimizer=opt,
        warmup_steps=1,
        total_steps=100,
        freeze_steps=10,
        n_backbone_groups=3,
    )
    for _ in range(30):
        sched.step()
    lrs = [g["lr"] for g in opt.param_groups]
    assert all(lr > 0.0 for lr in lrs[:3])  # backbone active after Phase 0

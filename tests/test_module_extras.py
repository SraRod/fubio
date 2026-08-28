"""FUBioModule extras: EMA teacher lifecycle, head-tune, semi training_step.

Everything runs the real FUBioModule via the conftest stub backbone, on CPU,
with hand-built batches shaped like the semi dataloader's CombinedLoader
output ({"labeled": ..., "unlabeled": ...}).

Upstream: train/module.py (setup, on_save/load_checkpoint, on_before_zero_grad,
_apply_head_tune, _head_tune_param_groups, training_step, _semi_step).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import make_module
from fubio.data.task_registry import TASKS
from fubio.train.module import FUBioModule


def _instance(task_str: str, *, labeled: bool) -> dict:
    """Minimal InstanceDict for one task in one image (test_semi_routing pattern)."""
    k = TASKS[task_str].n_keypoints
    return {
        "task_id": TASKS[task_str].task_int,
        "keypoints": np.full((k, 2), 0.5, dtype=np.float32),
        "supervised_mask": np.full(k, labeled, dtype=bool),
        "visible_mask": np.full(k, labeled, dtype=bool),
        "bbox": np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32),
        "transform_matrix": np.eye(3, dtype=np.float32),
        "original_hw": np.array([800, 600], dtype=np.int32),
        "is_labeled": labeled,
        "image_path": f"{task_str}.png",
    }


def _batch(module: FUBioModule, specs: list[tuple[str, bool]]) -> dict:
    size_w, size_h = module.config.backbone.input_size
    torch.manual_seed(0)
    return {
        "image": torch.randint(0, 255, (len(specs), 3, size_h, size_w), dtype=torch.uint8),
        "targets": [[_instance(t, labeled=lab)] for t, lab in specs],
    }


def _capture_log(module: FUBioModule, monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    logged: dict[str, float] = {}

    def _capture(name: str, value, **_kw) -> None:
        logged[name] = float(value.detach() if isinstance(value, torch.Tensor) else value)

    monkeypatch.setattr(module, "log", _capture)
    return logged


# ---------------------------------------------------------------------------
# EMA teacher
# ---------------------------------------------------------------------------


class TestEMATeacher:
    def test_setup_creates_teacher_outside_state_dict(self, stub_backbone) -> None:
        module = make_module(semi={"enabled": True, "lambda_pseudo": 1.0, "batch_size": 2})
        module.setup("fit")

        teacher = module._teacher_model
        assert not teacher.training
        assert all(not p.requires_grad for p in teacher.parameters())
        assert all("teacher" not in k for k in module.state_dict())

        # Recursive train() must not flip the teacher back to train mode.
        module.train()
        assert not module._teacher_model.training
        assert module.model.training

    def test_setup_skips_teacher_when_semi_off(self, stub_backbone) -> None:
        module = make_module()
        module.setup("fit")
        assert not hasattr(module, "_teacher_model")
        # Non-fit stages are a no-op, and the EMA/restore hooks tolerate
        # a module that never created a teacher.
        module.setup("validate")
        module.on_before_zero_grad(None)
        ckpt: dict = {}
        module.on_save_checkpoint(ckpt)
        assert ckpt == {}
        module.on_load_checkpoint({"teacher_state_dict": {}, "ema_step": 3})

    def test_checkpoint_roundtrip_restores_teacher(self, stub_backbone) -> None:
        module = make_module(semi={"enabled": True, "lambda_pseudo": 1.0, "batch_size": 2})
        module.setup("fit")

        ckpt: dict = {}
        module.on_save_checkpoint(ckpt)
        assert ckpt["ema_step"] == 0
        # Clone: on_save_checkpoint stores live references, and a real resume
        # goes through disk, so the restore must be tested against a copy.
        saved = {k: v.clone() for k, v in ckpt["teacher_state_dict"].items()}

        with torch.no_grad():
            for p in module._teacher_model.parameters():
                p.add_(1.0)

        module.on_load_checkpoint({"teacher_state_dict": saved, "ema_step": 7})
        assert module._ema_step == 7
        restored = module._teacher_model.state_dict()
        for k, v in saved.items():
            assert torch.equal(restored[k], v)

    def test_ema_step_moves_teacher_toward_student(self, stub_backbone) -> None:
        module = make_module(semi={"enabled": True, "lambda_pseudo": 1.0, "batch_size": 2})
        module.setup("fit")

        with torch.no_grad():
            for p in module.model.parameters():
                p.add_(1.0)

        # Step 0: warm-up alpha_eff = 0 → teacher snaps to the student.
        module.on_before_zero_grad(None)
        assert module._ema_step == 1
        for p_t, p_s in zip(
            module._teacher_model.parameters(), module.model.parameters(), strict=True
        ):
            assert torch.allclose(p_t, p_s)

        # Step 1: alpha_eff = 0.5 → teacher lands halfway to the new student.
        with torch.no_grad():
            for p in module.model.parameters():
                p.add_(1.0)
        module.on_before_zero_grad(None)
        p_t = next(module._teacher_model.parameters())
        p_s = next(module.model.parameters())
        assert torch.allclose(p_t, p_s - 0.5)


# ---------------------------------------------------------------------------
# Head-tune
# ---------------------------------------------------------------------------

HEAD_TUNE = {"tune_tasks": ["IVC"], "reinit_tasks": ["IVC"], "lr_scale": {"IVC": 0.5}}


class TestHeadTune:
    def test_setup_freezes_everything_but_tuned_task(self, stub_backbone) -> None:
        module = make_module(head_tune=HEAD_TUNE)
        module.setup("fit")

        assert all(p.requires_grad for p in module.model.tasks["IVC"].parameters())
        for name in ("HC", "A4C"):
            assert all(not p.requires_grad for p in module.model.tasks[name].parameters())
        assert all(not p.requires_grad for p in module.model.backbone.parameters())
        assert all(not p.requires_grad for p in module.model.decoder.parameters())

    def test_configure_optimizers_builds_per_task_groups(self, stub_backbone) -> None:
        module = make_module(head_tune=HEAD_TUNE)
        module.setup("fit")

        result = module.configure_optimizers()
        assert set(result) == {"optimizer"}  # no trainer → no scheduler
        groups = {g["name"]: g for g in result["optimizer"].param_groups}
        assert set(groups) == {"query_IVC", "heads_IVC"}

        opt_cfg = module.config.optimizer
        assert groups["query_IVC"]["lr"] == pytest.approx(opt_cfg.lr_decoder * 0.5)
        assert groups["heads_IVC"]["lr"] == pytest.approx(opt_cfg.lr_heads * 0.5)
        for g in groups.values():
            assert all(p.requires_grad for p in g["params"])


# ---------------------------------------------------------------------------
# Semi-supervised training_step
# ---------------------------------------------------------------------------

# All ramps forced to 1.0 (ramp_end <= current_epoch = 0) so every semi branch
# is active in one step: pseudo, pseudo_param, pseudo_geo, pseudo_gc, eq, MIL,
# TTA teacher, per-task lambda, and the no-flip undo for IVC.
SEMI_FULL = {
    "enabled": True,
    "batch_size": 2,
    "lambda_pseudo": 1.0,
    "lambda_pseudo_param": 0.5,
    "lambda_pseudo_geo": 0.5,
    "lambda_eq": 0.5,
    "lambda_mil": 0.2,
    "ramp_start": 0,
    "ramp_end": 0,
    "eq_ramp_start": 0,
    "eq_ramp_end": 0,
    "flip_prob": 1.0,
    "no_flip_tasks": ["IVC"],
    "tta_teacher_angles": [5.0],
    "task_pseudo_lambda": {"HC": 2.0},
}


class TestSemiTrainingStep:
    def test_full_semi_step_covers_all_loss_branches(
        self, stub_backbone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = make_module(semi=SEMI_FULL, loss={"lambda_geo_constraint": 0.1})
        module.setup("fit")
        logged = _capture_log(module, monkeypatch)

        batch = {
            "labeled": _batch(module, [("HC", True), ("A4C", True)]),
            # HC + IVC both have supportive geometry (pseudo_geo branch);
            # HC carries ortho/axis constraints (pseudo_gc branch); IVC is
            # in no_flip_tasks with flip_prob=1 (flip-undo branch).
            "unlabeled": _batch(module, [("HC", False), ("IVC", False)]),
        }
        loss = module.training_step(batch, 0)

        assert torch.isfinite(loss)
        assert loss.requires_grad
        assert logged["semi/ramp_pseudo"] == 1.0
        assert logged["semi/ramp_eq"] == 1.0
        for key in (
            "train/loss",
            "train_unlabeled/loss",
            "train_unlabeled/loss_mil",
            "semi/loss_pseudo",
            "semi/loss_eq",
            "semi/loss_pseudo_param",
            "semi/loss_pseudo_geo",
            "semi/loss_pseudo_gc",
            "semi/loss_total",
            "semi/teacher_conf_HC",
            "semi/teacher_conf_IVC",
        ):
            assert key in logged, f"missing logged key {key}"

    def test_burn_in_logs_zero_semi_total_and_skips_semi_step(
        self, stub_backbone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default ramp_start=2 → at epoch 0 only MIL/absence terms run."""
        module = make_module(
            semi={"enabled": True, "lambda_pseudo": 1.0, "lambda_mil": 0.2, "batch_size": 2}
        )
        module.setup("fit")
        logged = _capture_log(module, monkeypatch)

        semi_step_calls: list[dict] = []
        monkeypatch.setattr(
            module, "_semi_step", lambda *a, **kw: semi_step_calls.append({})
        )

        batch = {
            "labeled": _batch(module, [("HC", True)]),
            "unlabeled": _batch(module, [("HC", False)]),
        }
        loss = module.training_step(batch, 0)

        assert torch.isfinite(loss)
        assert semi_step_calls == []
        assert logged["semi/loss_total"] == 0.0

    def test_ramp_schedules(self, stub_backbone, monkeypatch: pytest.MonkeyPatch) -> None:
        module = make_module(
            semi={
                "enabled": True,
                "lambda_pseudo": 1.0,
                "ramp_start": 2,
                "ramp_end": 7,
                "eq_ramp_start": 7,
                "eq_ramp_end": 12,
            }
        )
        # current_epoch reads the (absent) trainer; shadow the property.
        monkeypatch.setattr(
            FUBioModule, "current_epoch", property(lambda self: 4), raising=False
        )
        assert module._semi_ramp() == pytest.approx((4 - 2) / (7 - 2))
        assert module._eq_ramp() == 0.0

        monkeypatch.setattr(
            FUBioModule, "current_epoch", property(lambda self: 20), raising=False
        )
        assert module._semi_ramp() == 1.0
        assert module._eq_ramp() == 1.0

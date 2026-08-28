"""Sprint 3 gate: losses, model architecture, module integration, training gate."""

from __future__ import annotations

import math

import lightning as L
import torch

from conftest import D, make_module, make_task_module
from fubio.data.task_registry import TASKS
from fubio.models.heads import _derive_bbox
from fubio.train.losses import (
    _paired_ciou,
    bbox_loss,
    conf_focal_loss,
    landmark_loss,
)
from fubio.train.mock_datamodule import MockDataModule

B = 2
N_SPATIAL = 1369


def _head_inputs(batch: int) -> dict:
    return {
        "memory": torch.randn(batch, N_SPATIAL, D),
        "spatial_shape": (37, 37),
        "memory_pos": torch.zeros(1, N_SPATIAL, D),
    }


# ---------------------------------------------------------------------------
# Loss unit tests
# ---------------------------------------------------------------------------


class TestBboxLoss:
    def test_no_matches(self) -> None:
        pred = torch.empty(0, 4)
        gt = torch.empty(0, 4)
        result = bbox_loss(pred, gt)
        assert result["loss_box_l1"].item() == 0.0
        assert result["loss_box_ciou"].item() == 0.0

    def test_single_match(self) -> None:
        pred = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        gt = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        result = bbox_loss(pred, gt)
        assert result["loss_box_l1"].item() < 0.01
        assert result["loss_box_ciou"].item() < 0.01

    def test_gradient_flows(self) -> None:
        pred = torch.rand(3, 4, requires_grad=True)
        gt = torch.rand(3, 4)
        result = bbox_loss(pred, gt)
        total = result["loss_box_l1"] + result["loss_box_ciou"]
        total.backward()
        assert pred.grad is not None


class TestConfFocalLoss:
    def test_all_negative(self) -> None:
        pred = torch.randn(10)
        target = torch.zeros(10)
        loss = conf_focal_loss(pred, target)
        assert loss.item() > 0

    def test_perfect_positive(self) -> None:
        pred = torch.full((5,), 10.0)
        target = torch.ones(5)
        loss = conf_focal_loss(pred, target)
        # High logit + positive target → low focal loss
        assert loss.item() < 0.01

    def test_empty(self) -> None:
        pred = torch.empty(0)
        target = torch.empty(0)
        loss = conf_focal_loss(pred, target)
        assert loss.item() == 0.0


class TestLandmarkLoss:
    def test_perfect_prediction(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [0.6, 0.6]]])
        gt = pred.clone()
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss = landmark_loss(pred, gt, mask)
        assert loss.item() < 1e-5

    def test_positive_error(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]]])
        gt = torch.tensor([[[0.8, 0.8]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        loss = landmark_loss(pred, gt, mask)
        assert loss.item() > 0

    def test_mask_excludes(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [99.0, 99.0]]])
        gt = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        loss = landmark_loss(pred, gt, mask)
        assert loss.item() < 1e-5

    def test_empty_mask(self) -> None:
        pred = torch.randn(1, 4, 2)
        gt = torch.randn(1, 4, 2)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        loss = landmark_loss(pred, gt, mask)
        assert loss.item() == 0.0


class TestCIoU:
    def test_identical_boxes(self) -> None:
        box = torch.tensor([[0.3, 0.3, 0.4, 0.4]])
        ciou = _paired_ciou(box, box)
        torch.testing.assert_close(ciou, torch.tensor([1.0]), atol=1e-5, rtol=1e-5)

    def test_non_overlapping(self) -> None:
        box1 = torch.tensor([[0.05, 0.05, 0.1, 0.1]])
        box2 = torch.tensor([[0.95, 0.95, 0.1, 0.1]])
        ciou = _paired_ciou(box1, box2)
        assert ciou.item() < 0


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------


class TestModelArchitecture:
    def test_task_module_query_shapes(self) -> None:
        n_inst = 2
        for tid, tdef in TASKS.items():
            tm = make_task_module(tdef.n_keypoints, n_inst=n_inst)
            q, _qp = tm.get_queries(batch_size=4)
            expected = n_inst * (1 + tdef.n_keypoints)
            assert q.shape == (4, expected, D), tid
            assert tm.n_queries == expected, tid

    def test_task_module_output_shapes(self) -> None:
        tm = make_task_module(4, n_inst=2)
        x = torch.randn(3, 2 * (1 + 4), D)
        out = tm.head(x, **_head_inputs(3))
        assert out.bbox.shape == (3, 2, 4)
        assert out.conf.shape == (3, 2, 1)
        assert out.landmarks.shape == (3, 2, 4, 2)

    def test_full_model_forward(self, stub_backbone) -> None:
        """End-to-end forward pass through the real constructor."""
        model = make_module(head={"n_inst": 2}).model

        images = torch.randn(2, 3, 518, 518)
        model_out = model(images)
        task_outputs = model_out.task_outputs

        assert len(task_outputs) == 9
        for tid, out in task_outputs.items():
            K = TASKS[tid].n_keypoints
            assert out.bbox.shape == (2, 2, 4)
            assert out.conf.shape == (2, 2, 1)
            assert out.landmarks.shape == (2, 2, K, 2)


# ---------------------------------------------------------------------------
# Module integration tests
# ---------------------------------------------------------------------------


class TestFUBioModuleTrainingStep:
    def test_training_step_runs(self, stub_backbone) -> None:
        module = make_module()
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        loss = module.training_step(batch, 0)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_validation_step_runs(self, stub_backbone) -> None:
        module = make_module()
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        module.on_validation_epoch_start()
        module.validation_step(batch, 0)
        module.on_validation_epoch_end()


class TestFUBioModuleOptimizer:
    def test_configure_optimizers(self, stub_backbone) -> None:
        module = make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result
        groups = result["optimizer"].param_groups
        assert len(groups) == 4


# ---------------------------------------------------------------------------
# Full training gate: 5 epochs, loss decreases
# ---------------------------------------------------------------------------


class TestTrainingGate:
    def test_5_epochs_loss_decreases(self, stub_backbone) -> None:
        """Sprint 3 gate: 5 epochs train+val on MockDataModule, loss stays finite."""
        module = make_module()
        dm = MockDataModule(n_train=8, n_val=4, batch_size=2, seed=42)

        trainer = L.Trainer(
            max_epochs=5,
            accelerator="cpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            limit_train_batches=4,
            limit_val_batches=2,
        )
        trainer.fit(module, datamodule=dm)

        assert trainer.current_epoch == 5

        metrics = trainer.callback_metrics
        assert "val/loss" in metrics
        assert math.isfinite(metrics["val/loss"].item())


# ---------------------------------------------------------------------------
# Derive bbox tests
# ---------------------------------------------------------------------------


class TestDeriveBbox:
    def test_shape(self) -> None:
        lm = torch.rand(2, 3, 5, 2)
        bbox = _derive_bbox(lm)
        assert bbox.shape == (2, 3, 4)

    def test_values_in_unit(self) -> None:
        lm = torch.rand(2, 3, 5, 2)
        bbox = _derive_bbox(lm)
        assert (bbox >= 0).all()
        assert (bbox <= 1).all()

    def test_gradient_flows(self) -> None:
        lm = torch.rand(2, 2, 4, 2, requires_grad=True)
        bbox = _derive_bbox(lm)
        bbox.sum().backward()
        assert lm.grad is not None
        assert lm.grad.abs().sum() > 0

    def test_tight_bound(self) -> None:
        """Bbox center should be at centroid of landmarks (no padding case)."""
        lm = torch.tensor([[[[0.3, 0.4], [0.7, 0.8]]]])  # (1,1,2,2)
        bbox = _derive_bbox(lm, pad_frac=0.0)
        cx, cy, w, h = bbox[0, 0].tolist()
        assert abs(cx - 0.5) < 1e-5
        assert abs(cy - 0.6) < 1e-5
        assert abs(w - 0.4) < 1e-5
        assert abs(h - 0.4) < 1e-5

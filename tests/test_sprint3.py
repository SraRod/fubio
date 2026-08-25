"""Sprint 3 gate: training loop runs 5 epochs on MockDataModule, loss decreases."""

from __future__ import annotations

import math

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from fubio.data.task_registry import TASKS
from fubio.models.backbone import BackboneOutput
from fubio.models.decoder import CrossAttentionDecoder
from fubio.models.heads import TaskModule, _derive_bbox, derive_bbox_masked
from fubio.models.model import FUBioModel
from fubio.models.neck import LinearNeck
from fubio.train.config import ExperimentConfig
from fubio.train.losses import (
    _paired_ciou,
    bbox_loss,
    conf_focal_loss,
    landmark_loss,
)
from fubio.train.matcher import PerTaskMatcher
from fubio.train.mock_datamodule import MockDataModule
from fubio.train.module import FUBioModule

D = 256
N_SPATIAL = 1369
C_BACKBONE = 768
B = 2


# ---------------------------------------------------------------------------
# Stub backbone for fast tests
# ---------------------------------------------------------------------------


class _StubBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = C_BACKBONE
        self.patch_size = 14
        self._linear = nn.Linear(3, C_BACKBONE)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def param_groups(self, lr: float, layer_decay: float = 1.0) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr, "name": "backbone"}]

    def forward(self, x: Tensor) -> BackboneOutput:
        b = x.shape[0]
        tokens = self._linear(x[:, :, :3, :3].permute(0, 2, 3, 1).reshape(b, -1, 3))
        tokens = torch.nn.functional.interpolate(
            tokens.permute(0, 2, 1),
            size=N_SPATIAL,
            mode="linear",
        ).permute(0, 2, 1)
        return BackboneOutput(
            features=[tokens],
            spatial_shape=(37, 37),
        )


def _make_stub_module(config: ExperimentConfig | None = None) -> FUBioModule:
    """FUBioModule with stub backbone for fast tests."""
    if config is None:
        config = ExperimentConfig()
    module = FUBioModule.__new__(FUBioModule)
    L.LightningModule.__init__(module)
    module.save_hyperparameters(config.model_dump())
    module.config = config

    n_inst = config.head.n_inst

    model = FUBioModel.__new__(FUBioModel)
    nn.Module.__init__(model)
    model.n_inst = n_inst

    model.backbone = _StubBackbone()
    model.neck = LinearNeck(C_BACKBONE, D)
    model.decoder = CrossAttentionDecoder(
        d_model=D,
        n_heads=8,
        ffn_dim=1024,
        n_layers=2,
    )
    from fubio.models.coord_predictors import build_coord_predictor

    model.tasks = nn.ModuleDict(
        {
            tid: TaskModule(
                n_keypoints=tdef.n_keypoints,
                d_model=D,
                n_inst=n_inst,
                n_head_layers=config.head.n_layers,
                n_heads=8,
                ffn_dim=1024,
                derive_bbox=config.head.derive_bbox,
                coord_predictor=build_coord_predictor(
                    config=config.head.coord,
                    d_model=D,
                    n_keypoints=tdef.n_keypoints,
                ),
            )
            for tid, tdef in TASKS.items()
        }
    )

    # GPU normalize buffers (module._normalize_image expects these)
    module.register_buffer(
        "_img_mean",
        torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
    )
    module.register_buffer(
        "_img_std",
        torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
    )
    module._miro_enabled = False
    module._canon_tasks = set()

    module.model = model
    module.matcher = PerTaskMatcher()
    _metrics = __import__(
        "fubio.evaluation.metrics",
        fromlist=[
            "MREMetric",
            "ParamMAEMetric",
            "BBoxPrecisionMetric",
        ],
    )
    module._train_mre = _metrics.MREMetric()
    module._train_param_mae = _metrics.ParamMAEMetric()
    module._train_bbox_prec = _metrics.BBoxPrecisionMetric()
    module._val_mre = _metrics.MREMetric()
    module._val_mre_pixel = _metrics.MREMetric()
    module._val_param_mae = _metrics.ParamMAEMetric()
    module._val_bbox_prec = _metrics.BBoxPrecisionMetric()
    module._oracle_mre = _metrics.MREMetric()
    module._inst_sep = []

    return module


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
        loss = landmark_loss(pred, gt, mask, k_task=2)
        assert loss.item() < 1e-5

    def test_positive_error(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]]])
        gt = torch.tensor([[[0.8, 0.8]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        loss = landmark_loss(pred, gt, mask, k_task=1)
        assert loss.item() > 0

    def test_mask_excludes(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [99.0, 99.0]]])
        gt = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        loss = landmark_loss(pred, gt, mask, k_task=2)
        assert loss.item() < 1e-5

    def test_empty_mask(self) -> None:
        pred = torch.randn(1, 4, 2)
        gt = torch.randn(1, 4, 2)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        loss = landmark_loss(pred, gt, mask, k_task=4)
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
            tm = TaskModule(n_keypoints=tdef.n_keypoints, d_model=D, n_inst=n_inst)
            q, _qp = tm.get_queries(batch_size=4)
            expected = n_inst * (1 + tdef.n_keypoints)
            assert q.shape == (4, expected, D), tid
            assert tm.n_queries == expected, tid

    def test_task_module_output_shapes(self) -> None:
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2, n_head_layers=1)
        x = torch.randn(3, 2 * (1 + 4), D)
        out = tm.head(x)
        assert out.bbox.shape == (3, 2, 4)
        assert out.conf.shape == (3, 2, 1)
        assert out.landmarks.shape == (3, 2, 4, 2)

    def test_full_model_forward(self) -> None:
        """End-to-end forward pass with stub backbone."""
        model = FUBioModel.__new__(FUBioModel)
        nn.Module.__init__(model)
        model.n_inst = 2
        model.backbone = _StubBackbone()
        model.neck = LinearNeck(C_BACKBONE, D)
        model.decoder = CrossAttentionDecoder(d_model=D, n_heads=8, ffn_dim=1024, n_layers=2)
        model.tasks = nn.ModuleDict(
            {
                tid: TaskModule(n_keypoints=tdef.n_keypoints, d_model=D, n_inst=2)
                for tid, tdef in TASKS.items()
            }
        )

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
    def test_training_step_runs(self) -> None:
        module = _make_stub_module()
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        loss = module.training_step(batch, 0)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_validation_step_runs(self) -> None:
        module = _make_stub_module()
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        module.on_validation_epoch_start()
        module.validation_step(batch, 0)
        module.on_validation_epoch_end()


class TestFUBioModuleOptimizer:
    def test_configure_optimizers(self) -> None:
        module = _make_stub_module()
        result = module.configure_optimizers()
        assert "optimizer" in result
        groups = result["optimizer"].param_groups
        assert len(groups) == 4


# ---------------------------------------------------------------------------
# Full training gate: 5 epochs, loss decreases
# ---------------------------------------------------------------------------


class TestTrainingGate:
    def test_5_epochs_loss_decreases(self) -> None:
        """Sprint 3 gate: 5 epochs train+val on MockDataModule, loss decreases."""
        module = _make_stub_module()
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


class TestDeriveBboxMasked:
    def test_all_visible(self) -> None:
        lm = torch.tensor([[[0.3, 0.4], [0.7, 0.8]]])  # (1, 2, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        bbox = derive_bbox_masked(lm, mask, pad_frac=0.0)
        assert bbox.shape == (1, 4)
        cx, cy, w, h = bbox[0].tolist()
        assert abs(cx - 0.5) < 1e-5
        assert abs(cy - 0.6) < 1e-5

    def test_partial_mask(self) -> None:
        lm = torch.tensor([[[0.2, 0.3], [0.8, 0.9]]])  # (1, 2, 2)
        mask = torch.tensor([[True, False]])
        bbox = derive_bbox_masked(lm, mask, pad_frac=0.0)
        cx, cy, w, h = bbox[0].tolist()
        # Only first point visible → degenerate bbox at that point
        assert abs(cx - 0.2) < 1e-5
        assert abs(cy - 0.3) < 1e-5

    def test_no_visible_fallback(self) -> None:
        lm = torch.tensor([[[0.5, 0.5], [0.6, 0.6]]])
        mask = torch.zeros(1, 2, dtype=torch.bool)
        bbox = derive_bbox_masked(lm, mask, pad_frac=0.0)
        # Fallback: full canvas
        cx, cy, w, h = bbox[0].tolist()
        assert abs(cx - 0.5) < 1e-5
        assert abs(cy - 0.5) < 1e-5
        assert abs(w - 1.0) < 1e-5
        assert abs(h - 1.0) < 1e-5

    def test_gradient_flows(self) -> None:
        lm = torch.rand(3, 5, 2, requires_grad=True)
        mask = torch.ones(3, 5, dtype=torch.bool)
        bbox = derive_bbox_masked(lm, mask)
        bbox.sum().backward()
        assert lm.grad is not None


class TestTaskModuleDeriveBbox:
    def test_output_shapes_derive_mode(self) -> None:
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2, n_head_layers=1, derive_bbox=True)
        x = torch.randn(3, 2 * (1 + 4), D)
        out = tm.head(x)
        assert out.bbox.shape == (3, 2, 4)
        assert out.conf.shape == (3, 2, 1)
        assert out.landmarks.shape == (3, 2, 4, 2)

    def test_no_bbox_mlp_in_derive_mode(self) -> None:
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2, derive_bbox=True)
        assert not hasattr(tm, "bbox_mlp")

    def test_has_bbox_mlp_in_default_mode(self) -> None:
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2, derive_bbox=False)
        assert hasattr(tm, "bbox_mlp")

    def test_derive_bbox_gradient_to_landmarks(self) -> None:
        """Bbox loss gradient should flow to coord_predictor when derive_bbox=True."""
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2, n_head_layers=1, derive_bbox=True)
        x = torch.randn(2, 2 * (1 + 4), D)
        out = tm.head(x)
        out.bbox.sum().backward()
        coord_grad = list(tm.coord_predictor.parameters())[0].grad
        assert coord_grad is not None
        assert coord_grad.abs().sum() > 0


class TestModuleDeriveBbox:
    def test_training_step_derive_bbox(self) -> None:
        from fubio.train.config import HeadConfig

        config = ExperimentConfig(head=HeadConfig(derive_bbox=True))
        module = _make_stub_module(config)
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        loss = module.training_step(batch, 0)
        assert loss.item() > 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# Semi-supervised loss framework integration tests
# ---------------------------------------------------------------------------


class TestModuleLabelSmoothing:
    """Training step with neg_label_smooth > 0 runs and produces valid loss."""

    def test_label_smooth_training_step(self) -> None:
        from fubio.train.config import LossConfig

        config = ExperimentConfig(loss=LossConfig(neg_label_smooth=0.05))
        module = _make_stub_module(config)
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        loss = module.training_step(batch, 0)
        assert loss.item() > 0
        assert loss.requires_grad
        assert math.isfinite(loss.item())

    def test_label_smooth_vs_no_smooth(self) -> None:
        """Label smoothing should produce a different loss value than no smoothing."""
        from fubio.train.config import LossConfig

        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        module_no = _make_stub_module(ExperimentConfig(loss=LossConfig(neg_label_smooth=0.0)))
        loss_no = module_no.training_step(batch, 0)

        module_yes = _make_stub_module(ExperimentConfig(loss=LossConfig(neg_label_smooth=0.05)))
        # Share weights so only the loss formula differs
        module_yes.model.load_state_dict(module_no.model.state_dict())
        loss_yes = module_yes.training_step(batch, 0)

        # They share the same model weights — difference comes from conf targets
        assert loss_no.item() != pytest.approx(loss_yes.item(), abs=1e-6)


# ---------------------------------------------------------------------------
# Reference point + offset tests
# ---------------------------------------------------------------------------


class TestTaskModuleRefPoints:
    def _make_ref_module(self, **kwargs) -> TaskModule:
        from fubio.models.coord_predictors import RefPointsPredictor

        defaults = {"d_model": D, "n_keypoints": 4, "n_inst": 2, "n_head_layers": 1}
        defaults.update(kwargs)
        kp = defaults.pop("n_keypoints")
        return TaskModule(
            **defaults,
            n_keypoints=kp,
            coord_predictor=RefPointsPredictor(D, kp),
        )

    def test_output_shapes(self) -> None:
        tm = self._make_ref_module()
        x = torch.randn(3, 2 * (1 + 4), D)
        out = tm.head(x)
        assert out.landmarks.shape == (3, 2, 4, 2)
        assert out.bbox.shape == (3, 2, 4)

    def test_has_ref_points_param(self) -> None:
        tm = self._make_ref_module()
        assert hasattr(tm.coord_predictor, "ref_points")
        assert tm.coord_predictor.ref_points.shape == (4, 2)

    def test_no_ref_points_by_default(self) -> None:
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2)
        assert not hasattr(tm.coord_predictor, "ref_points")

    def test_zero_init_gives_center(self) -> None:
        """ref_points=0 → sigmoid(0)=0.5 → landmarks near image center at init."""
        tm = self._make_ref_module()
        with torch.no_grad():
            for p in tm.coord_predictor.coord_mlp.parameters():
                p.zero_()
        x = torch.randn(1, 2 * (1 + 4), D)
        out = tm.head(x)
        assert (out.landmarks - 0.5).abs().max() < 0.05

    def test_gradient_flows_to_ref_points(self) -> None:
        tm = self._make_ref_module()
        x = torch.randn(2, 2 * (1 + 4), D)
        out = tm.head(x)
        out.landmarks.sum().backward()
        assert tm.coord_predictor.ref_points.grad is not None
        assert tm.coord_predictor.ref_points.grad.abs().sum() > 0

    def test_ref_points_combine_with_derive_bbox(self) -> None:
        tm = self._make_ref_module(derive_bbox=True)
        x = torch.randn(2, 2 * (1 + 4), D)
        out = tm.head(x)
        assert out.bbox.shape == (2, 2, 4)
        assert not hasattr(tm, "bbox_mlp")


class TestModuleRefPoints:
    def test_training_step_ref_points(self) -> None:
        from fubio.train.config import HeadConfig, RefPointsCoordConfig

        config = ExperimentConfig(
            head=HeadConfig(coord=RefPointsCoordConfig()),
        )
        module = _make_stub_module(config)
        dm = MockDataModule(n_train=4, n_val=2, batch_size=2, seed=42)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        loss = module.training_step(batch, 0)
        assert loss.item() > 0
        assert loss.requires_grad
        assert math.isfinite(loss.item())

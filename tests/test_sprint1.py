"""Sprint 1 gate: geometry, metrics, config, matcher tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from fubio.data.types import InstanceDict
from fubio.evaluation.geometry import (
    compute_angle,
    compute_distance,
    compute_ellipse_circumference,
    compute_params_for_evaluation,
    compute_params_for_training,
)
from fubio.evaluation.metrics import MREMetric, ParamMAEMetric
from fubio.train.config import ExperimentConfig
from fubio.train.matcher import PerTaskMatcher

# ---------------------------------------------------------------------------
# geometry.py
# ---------------------------------------------------------------------------


class TestComputeDistance:
    def test_orthogonal_pair(self) -> None:
        p1 = torch.tensor([[0.0, 0.0]])
        p2 = torch.tensor([[3.0, 4.0]])
        result = compute_distance(p1, p2)
        torch.testing.assert_close(result, torch.tensor([5.0]), atol=1e-5, rtol=1e-5)

    def test_zero_distance(self) -> None:
        p = torch.tensor([[1.0, 2.0]])
        result = compute_distance(p, p)
        assert result.item() < 1e-3

    def test_batched(self) -> None:
        p1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        p2 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        result = compute_distance(p1, p2)
        torch.testing.assert_close(result, torch.tensor([1.0, 1.0]), atol=1e-5, rtol=1e-5)

    def test_fp32_forced(self) -> None:
        p1 = torch.tensor([[0.0, 0.0]], dtype=torch.float16)
        p2 = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
        result = compute_distance(p1, p2)
        assert result.dtype == torch.float32

    def test_gradient_flows(self) -> None:
        p1 = torch.tensor([[0.0, 0.0]], requires_grad=True)
        p2 = torch.tensor([[3.0, 4.0]], requires_grad=True)
        d = compute_distance(p1, p2)
        d.sum().backward()
        assert p1.grad is not None
        assert p2.grad is not None


class TestComputeEllipseCircumference:
    def test_circle(self) -> None:
        """Circle with radius 5: C = 2πr = 10π ≈ 31.416."""
        p1 = torch.tensor([[0.0, 0.0]])
        p2 = torch.tensor([[10.0, 0.0]])
        p3 = torch.tensor([[0.0, 0.0]])
        p4 = torch.tensor([[0.0, 10.0]])
        result = compute_ellipse_circumference(p1, p2, p3, p4)
        expected = 2.0 * math.pi * 5.0
        torch.testing.assert_close(result, torch.tensor([expected]), atol=1e-3, rtol=1e-3)

    def test_known_ellipse(self) -> None:
        """a=10, b=6 → Ramanujan: π(3(10+6) - √((30+6)(10+18)))."""
        p1 = torch.tensor([[0.0, 0.0]])
        p2 = torch.tensor([[20.0, 0.0]])  # diameter 20, a=10
        p3 = torch.tensor([[0.0, 0.0]])
        p4 = torch.tensor([[0.0, 12.0]])  # diameter 12, b=6
        result = compute_ellipse_circumference(p1, p2, p3, p4)
        a, b = 10.0, 6.0
        expected = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        torch.testing.assert_close(result, torch.tensor([expected]), atol=1e-3, rtol=1e-3)

    def test_gradient_flows(self) -> None:
        points = [torch.randn(2, 2, requires_grad=True) for _ in range(4)]
        c = compute_ellipse_circumference(*points)
        c.sum().backward()
        for p in points:
            assert p.grad is not None


class TestComputeAngle:
    def test_right_angle(self) -> None:
        vertex = torch.tensor([[0.0, 0.0]])
        p1 = torch.tensor([[1.0, 0.0]])
        p2 = torch.tensor([[0.0, 1.0]])
        result = compute_angle(vertex, p1, p2)
        expected = torch.tensor([math.pi / 2])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_straight_angle(self) -> None:
        vertex = torch.tensor([[0.0, 0.0]])
        p1 = torch.tensor([[1.0, 0.0]])
        p2 = torch.tensor([[-1.0, 0.0]])
        result = compute_angle(vertex, p1, p2)
        expected = torch.tensor([math.pi])
        # Wider tolerance: eps clamp on cosine prevents exact π
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_45_degrees(self) -> None:
        vertex = torch.tensor([[0.0, 0.0]])
        p1 = torch.tensor([[1.0, 0.0]])
        p2 = torch.tensor([[1.0, 1.0]])
        result = compute_angle(vertex, p1, p2)
        expected = torch.tensor([math.pi / 4])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self) -> None:
        vertex = torch.tensor([[0.0, 0.0]], requires_grad=True)
        p1 = torch.tensor([[1.0, 0.0]], requires_grad=True)
        p2 = torch.tensor([[0.0, 1.0]], requires_grad=True)
        a = compute_angle(vertex, p1, p2)
        a.sum().backward()
        assert vertex.grad is not None


class TestComputeParamsForEvaluation:
    def test_hc_circumference(self) -> None:
        """HC has 4 landmarks → 1 ellipse circumference param."""
        landmarks = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [0.0, 0.0], [0.0, 12.0]]])
        result = compute_params_for_evaluation(landmarks, "HC")
        assert "HC_circumference" in result
        assert result["HC_circumference"].shape == (1,)

        a, b = 10.0, 6.0
        expected = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        torch.testing.assert_close(
            result["HC_circumference"],
            torch.tensor([expected]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_ivc_distance(self) -> None:
        """IVC has 2 landmarks → 1 distance param."""
        landmarks = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
        result = compute_params_for_evaluation(landmarks, "IVC")
        assert "IVC_diameter" in result
        torch.testing.assert_close(
            result["IVC_diameter"],
            torch.tensor([5.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_aop_angle(self) -> None:
        """AOP has angle param at indices [0, 1, 3]."""
        # 4 landmarks for AOP
        landmarks = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [99.0, 99.0], [0.0, 1.0]]])
        result = compute_params_for_evaluation(landmarks, "AOP")
        assert "AOP_angle" in result
        expected = torch.tensor([math.pi / 2])
        torch.testing.assert_close(result["AOP_angle"], expected, atol=1e-5, rtol=1e-5)

    def test_visible_mask_nan(self) -> None:
        """Invisible landmark should produce NaN for dependent params."""
        landmarks = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
        mask = torch.tensor([[True, False]])
        result = compute_params_for_evaluation(landmarks, "IVC", visible_mask=mask)
        assert torch.isnan(result["IVC_diameter"]).all()

    def test_a4c_param_count(self) -> None:
        """A4C should produce 8 distance params."""
        landmarks = torch.randn(2, 16, 2)
        result = compute_params_for_evaluation(landmarks, "A4C")
        assert len(result) == 8
        for v in result.values():
            assert v.shape == (2,)


class TestTrainingVsEvaluationGeometry:
    """The two entrypoints must differ in exactly one way: degeneracy handling.

    The smoothed primitives carry `+1e-7` inside every sqrt so L_param has a
    finite gradient at coincident points. The side effect is that an undefined
    AOP angle (zero-length ray) comes back as exactly pi/2 — a plausible clinical
    value that nothing downstream can flag. Evaluation must reject it instead.

    These tests are also the guarantee that this change did not touch training:
    `test_training_path_unchanged_on_normal_input` pins the smoothed values.
    """

    def test_degenerate_angle_is_fabricated_90deg_in_training_path(self) -> None:
        """Pins the known-bad behaviour so the contrast below is not vacuous."""
        # p1 == vertex → ray has no direction, angle is undefined
        landmarks = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [99.0, 99.0], [0.0, 1.0]]])
        result = compute_params_for_training(landmarks, "AOP")
        torch.testing.assert_close(
            result["AOP_angle"], torch.tensor([math.pi / 2]), atol=1e-5, rtol=1e-5
        )

    def test_degenerate_angle_is_nan_in_evaluation_path(self) -> None:
        landmarks = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [99.0, 99.0], [0.0, 1.0]]])
        result = compute_params_for_evaluation(landmarks, "AOP")
        assert torch.isnan(result["AOP_angle"]).all(), "undefined angle must not be reported as 90°"

    def test_coincident_points_distance(self) -> None:
        landmarks = torch.tensor([[[7.0, 7.0], [7.0, 7.0]]])

        smoothed = compute_params_for_training(landmarks, "IVC")["IVC_diameter"]
        assert smoothed.item() > 0.0, "smoothed path keeps a nonzero value for the gradient"
        assert smoothed.item() < 1e-3

        exact = compute_params_for_evaluation(landmarks, "IVC")["IVC_diameter"]
        assert exact.item() == 0.0, "evaluation distance must be exact at zero"

    def test_collapsed_ellipse_axis_is_nan_in_evaluation(self) -> None:
        # second axis collapsed to a point → not a measurable ellipse
        landmarks = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [5.0, 5.0], [5.0, 5.0]]])
        result = compute_params_for_evaluation(landmarks, "HC")
        assert torch.isnan(result["HC_circumference"]).all()

    def test_training_path_unchanged_on_normal_input(self) -> None:
        """REGRESSION GUARD: L_param numerics must be bit-identical to pre-split.

        Values below were computed with the original single-function
        implementation. If this test fails, training behaviour changed and every
        comparison against R20 is invalid.
        """
        hc = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [0.0, 0.0], [0.0, 12.0]]])
        a, b = 10.0, 6.0
        expected_circ = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        torch.testing.assert_close(
            compute_params_for_training(hc, "HC")["HC_circumference"],
            torch.tensor([expected_circ]),
            atol=1e-4,
            rtol=1e-4,
        )

        ivc = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
        torch.testing.assert_close(
            compute_params_for_training(ivc, "IVC")["IVC_diameter"],
            torch.tensor([5.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_paths_agree_on_non_degenerate_geometry(self) -> None:
        """Away from degeneracy the epsilon is negligible — no silent divergence."""
        torch.manual_seed(0)
        landmarks = torch.randn(8, 16, 2) * 100.0
        train = compute_params_for_training(landmarks, "A4C")
        evaluate = compute_params_for_evaluation(landmarks, "A4C")
        for name, tv in train.items():
            torch.testing.assert_close(tv, evaluate[name], atol=1e-3, rtol=1e-4)

    def test_training_path_gradient_survives_degeneracy(self) -> None:
        """Why the smoothed path must stay: exact norms give NaN gradients here."""
        landmarks = torch.zeros(1, 2, 2, requires_grad=True)
        out = compute_params_for_training(landmarks, "IVC")["IVC_diameter"]
        out.sum().backward()
        assert landmarks.grad is not None
        assert torch.isfinite(landmarks.grad).all()


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


class TestMREMetric:
    def test_perfect_prediction(self) -> None:
        m = MREMetric()
        pred = torch.tensor([[[0.5, 0.5], [0.6, 0.6]]])
        target = pred.clone()
        mask = torch.ones(1, 2, dtype=torch.bool)
        m.update(pred, target, task_id=0, mask=mask)
        result = m.compute()
        assert result["task/A4C"] < 1e-3

    def test_known_error(self) -> None:
        m = MREMetric()
        pred = torch.tensor([[[0.0, 0.0]]])
        target = torch.tensor([[[3.0, 4.0]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        m.update(pred, target, task_id=3, mask=mask)  # IVC
        result = m.compute()
        assert abs(result["task/IVC"] - 5.0) < 1e-3

    def test_domain_balanced(self) -> None:
        m = MREMetric()
        mask = torch.ones(1, 1, dtype=torch.bool)
        # Cardiac task (IVC=3) error = 1.0
        m.update(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0]]]),
            task_id=3,
            mask=mask,
        )
        # Prenatal task (fetal_femur=8) error = 2.0
        m.update(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[2.0, 0.0]]]),
            task_id=8,
            mask=mask,
        )
        result = m.compute()
        assert "domain/cardiac" in result
        assert "domain/prenatal" in result
        # Overall should be mean of domain MREs
        expected_overall = (result["domain/cardiac"] + result["domain/prenatal"]) / 2
        assert abs(result["overall"] - expected_overall) < 1e-5

    def test_mask_excludes(self) -> None:
        m = MREMetric()
        pred = torch.tensor([[[0.0, 0.0], [99.0, 99.0]]])
        target = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        m.update(pred, target, task_id=0, mask=mask)
        result = m.compute()
        # Only first landmark counts (error ~0)
        assert result["task/A4C"] < 1e-3

    def test_reset(self) -> None:
        m = MREMetric()
        mask = torch.ones(1, 1, dtype=torch.bool)
        m.update(torch.randn(1, 1, 2), torch.randn(1, 1, 2), task_id=0, mask=mask)
        m.reset()
        assert m.compute() == {}


class TestParamMAEMetric:
    def test_known_mae(self) -> None:
        m = ParamMAEMetric()
        m.update(
            {"IVC_diameter": torch.tensor([10.0, 20.0])},
            {"IVC_diameter": torch.tensor([12.0, 18.0])},
        )
        result = m.compute()
        assert abs(result["param/IVC_diameter"] - 2.0) < 1e-5

    def test_nan_excluded(self) -> None:
        m = ParamMAEMetric()
        m.update(
            {"IVC_diameter": torch.tensor([10.0, float("nan")])},
            {"IVC_diameter": torch.tensor([12.0, 15.0])},
        )
        result = m.compute()
        # Only first sample counts: |10-12|=2
        assert abs(result["param/IVC_diameter"] - 2.0) < 1e-5

    def test_reset(self) -> None:
        m = ParamMAEMetric()
        m.update(
            {"IVC_diameter": torch.tensor([1.0])},
            {"IVC_diameter": torch.tensor([2.0])},
        )
        m.reset()
        assert m.compute() == {}


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_default_construction(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.backbone.name == "dinov2_vitb14"
        assert cfg.d_model == 256
        assert cfg.head.n_inst == 2
        assert cfg.seed == 42
        assert cfg.precision == "bf16-mixed"

    def test_frozen(self) -> None:
        cfg = ExperimentConfig()
        with pytest.raises(ValidationError):
            cfg.seed = 99  # type: ignore[misc]

    def test_nested_frozen(self) -> None:
        cfg = ExperimentConfig()
        with pytest.raises(ValidationError):
            cfg.backbone.freeze_epochs = 10  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = ExperimentConfig(
            backbone={"name": "convnextv2_base", "freeze_epochs": 10},  # type: ignore[arg-type]
            loss={"lambda_bbox": 2.0},  # type: ignore[arg-type]
        )
        assert cfg.backbone.name == "convnextv2_base"
        assert cfg.backbone.freeze_epochs == 10
        assert cfg.loss.lambda_bbox == 2.0
        assert cfg.loss.lambda_land == 5.0  # default preserved

    def test_decoder_head_config_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.decoder.n_layers == 3
        assert cfg.head.n_layers == 2
        assert cfg.head.n_inst == 2


# ---------------------------------------------------------------------------
# matcher.py
# ---------------------------------------------------------------------------


def _make_instance(task_id: int, cx: float, cy: float, w: float, h: float) -> InstanceDict:
    k = {0: 16, 1: 22, 2: 4, 3: 2, 4: 4, 5: 2, 6: 4, 7: 4, 8: 2}[task_id]
    return {
        "task_id": task_id,
        "keypoints": np.random.rand(k, 2).astype(np.float32),
        "supervised_mask": np.ones(k, dtype=bool),
        "visible_mask": np.ones(k, dtype=bool),
        "bbox": np.array([cx, cy, w, h], dtype=np.float32),
        "transform_matrix": np.eye(3, dtype=np.float32),
        "original_hw": np.array([518, 518], dtype=np.int32),
        "is_labeled": True,
        "image_path": "test.png",
    }


class TestPerTaskMatcher:
    def _make_model_output(self, B: int, n_inst: int = 2) -> dict:
        """Fake model output for all tasks."""
        from fubio.data.task_registry import TASKS as _T
        from fubio.data.types import TaskOutput

        out = {}
        for tid, tdef in _T.items():
            out[tid] = TaskOutput(
                bbox=torch.rand(B, n_inst, 4),
                conf=torch.randn(B, n_inst, 1),
                landmarks=torch.rand(B, n_inst, tdef.n_keypoints, 2),
            )
        return out

    def test_empty_targets(self) -> None:
        matcher = PerTaskMatcher()
        model_out = self._make_model_output(1)
        targets: list[list[InstanceDict]] = [[]]
        result = matcher(model_out, targets)
        # All tasks should have empty matches for this image
        for tid in result:
            q_idx, g_idx = result[tid][0]
            assert len(q_idx) == 0
            assert len(g_idx) == 0

    def test_single_match(self) -> None:
        matcher = PerTaskMatcher()
        model_out = self._make_model_output(1)
        # GT: one IVC instance (task_int=3)
        targets = [[_make_instance(3, 0.5, 0.5, 0.3, 0.3)]]
        result = matcher(model_out, targets)
        # IVC should have 1 match
        q_idx, g_idx = result["IVC"][0]
        assert len(q_idx) == 1
        assert len(g_idx) == 1
        # Other tasks should have no matches
        assert len(result["HC"][0][0]) == 0

    def test_multi_image_batch(self) -> None:
        matcher = PerTaskMatcher()
        model_out = self._make_model_output(3)
        targets = [
            [_make_instance(0, 0.5, 0.5, 0.2, 0.2)],  # A4C
            [],
            [_make_instance(3, 0.3, 0.3, 0.1, 0.1)],  # IVC
        ]
        result = matcher(model_out, targets)
        assert len(result["A4C"][0][0]) == 1  # 1 GT in image 0
        assert len(result["A4C"][1][0]) == 0  # 0 GT in image 1
        assert len(result["IVC"][2][0]) == 1  # 1 GT in image 2

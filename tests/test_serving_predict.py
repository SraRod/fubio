"""serving/predict.py: geometry helpers, discovery, and the stubbed pipeline.

The inference pipeline is exercised end-to-end with a stub module whose model
returns preset TaskOutputs, so every expected coordinate is known in advance:
targets are chosen in ORIGINAL pixel space, forward-mapped through the same
letterbox/rotation matrices predict.py builds, baked into the stub as [0,1]
canvas coordinates, and the pipeline must map them back to the original
targets. That makes each assertion a genuine round-trip through
prepare_image → model → inverse_transform_landmarks (→ TTA inverse warp /
ellipse consensus / refinement crop mapping) rather than a smoke test.

`load_module` is covered with a real FUBioModule (stub backbone, conftest)
saved via torch.save with the Lightning checkpoint dict layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from conftest import make_module
from fubio.data.spatial import SpatialParams, build_affine_matrix
from fubio.data.task_registry import TASKS
from fubio.data.types import TaskOutput
from fubio.evaluation.postprocessing import inverse_transform_landmarks
from fubio.models.model import ModelOutput
from fubio.serving.predict import (
    _compute_crop_region,
    _compute_roi_pct,
    _ellipse_consensus_avg,
    _invert_affine_2x3,
    _make_tta_matrix,
    _mre_table,
    _summary,
    discover_competition_images,
    discover_from_manifest,
    load_module,
    parse_tta_color_specs,
    prepare_image,
    run_inference,
    run_inference_tta,
    run_refinement,
    save_outputs,
)
from fubio.serving.validate import SubmissionError
from fubio.train.views import warp_landmarks

CPU = torch.device("cpu")
INPUT_SIZE = (64, 64)  # (W, H); square because warp_image requires H == W


# ---------------------------------------------------------------------------
# Geometry helpers shared by the tests
# ---------------------------------------------------------------------------


def _letterbox_matrix(w: int, h: int, input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """The same original → canvas affine prepare_image builds."""
    params = SpatialParams(target_size=input_size)
    return build_affine_matrix((w, h), params, resize_mode="letterbox")


def _forward_map(pts_px: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3×3 affine to (K, 2) points."""
    hom = np.hstack([pts_px, np.ones((len(pts_px), 1))])
    return (hom @ matrix.astype(np.float64).T)[:, :2]


def _bake_norm(
    pts_orig_px: np.ndarray,
    w: int,
    h: int,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Original-pixel targets → the [0,1] canvas coords a model would emit."""
    aug = _forward_map(pts_orig_px, _letterbox_matrix(w, h, input_size))
    return (aug / np.array(input_size, dtype=np.float64)).astype(np.float32)


def _write_png(path: Path, w: int, h: int, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _sorted_rows(flat: list[float] | np.ndarray) -> np.ndarray:
    pts = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    return pts[np.lexsort((pts[:, 1], pts[:, 0]))]


def _assert_same_points(actual_flat: list[float], expected_px: np.ndarray, tol: float = 0.05):
    """Order-insensitive comparison (competition ordering may permute rows)."""
    np.testing.assert_allclose(_sorted_rows(actual_flat), _sorted_rows(expected_px), atol=tol)


# ---------------------------------------------------------------------------
# Stub module: exactly the interface run_inference* calls
# ---------------------------------------------------------------------------


def _stub_output(
    per_sample: list[tuple[str, np.ndarray]],
    conf_logits: dict[tuple[int, str], float] | None = None,
) -> ModelOutput:
    """ModelOutput for a batch aligned with `per_sample` (task_id, lm_norm (K,2)).

    Two instance slots: slot 0 is a decoy (lm=0.9, logit −3) so tests fail if
    selection stops following confidence; slot 1 carries the payload. Owned
    tasks get logit 2.0 unless overridden via conf_logits[(b, task_id)].
    """
    B = len(per_sample)
    conf_logits = conf_logits or {}
    outs: dict[str, TaskOutput] = {}
    for tid, tdef in TASKS.items():
        K = tdef.n_keypoints
        lm = torch.full((B, 2, K, 2), 0.5)
        lm[:, 0] = 0.9
        conf = torch.full((B, 2, 1), -4.0)
        conf[:, 0, 0] = -3.0
        for b, (own_tid, pts_norm) in enumerate(per_sample):
            if own_tid == tid:
                lm[b, 1] = torch.from_numpy(np.asarray(pts_norm, dtype=np.float32))
                conf[b, 1, 0] = 2.0
            if (b, tid) in conf_logits:
                conf[b, 1, 0] = conf_logits[(b, tid)]
        outs[tid] = TaskOutput(bbox=torch.zeros(B, 2, 4), conf=conf, landmarks=lm)
    return ModelOutput(task_outputs=outs, backbone_out=None, spatial_shape=(37, 37))


class _StubNet:
    """Returns one preset ModelOutput per forward call, in order."""

    def __init__(self, outputs: list[ModelOutput]) -> None:
        self.outputs = outputs
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> ModelOutput:
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        assert x.shape[0] == out.task_outputs["IVC"].conf.shape[0], "batch/stub misalignment"
        return out


class StubModule:
    """Duck-types the FUBioModule surface predict.py touches."""

    def __init__(self, outputs: list[ModelOutput]) -> None:
        self.model = _StubNet(outputs)
        self._img_mean = torch.tensor(0.5)
        self._img_std = torch.tensor(0.25)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return image.float() / 255.0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPrepareImage:
    def test_letterbox_shapes_and_roundtrip(self):
        w, h = 80, 64
        img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
        chw, M, orig_hw = prepare_image(img, INPUT_SIZE)

        assert chw.shape == (3, INPUT_SIZE[1], INPUT_SIZE[0])
        assert chw.dtype == np.uint8
        assert orig_hw.tolist() == [h, w]

        pts = np.array([[10.0, 12.0], [70.0, 50.0], [0.0, 0.0]])
        aug = _forward_map(pts, M)
        # Letterbox keeps everything on the canvas
        assert (aug >= -1e-6).all()
        assert (aug[:, 0] <= INPUT_SIZE[0]).all() and (aug[:, 1] <= INPUT_SIZE[1]).all()

        back = inverse_transform_landmarks(torch.from_numpy(aug), M, orig_hw)
        np.testing.assert_allclose(back, pts, atol=1e-4)

    def test_inverse_transform_clips_to_bounds(self):
        M = _letterbox_matrix(80, 64)
        orig_hw = np.array([64, 80], dtype=np.int32)
        # Far outside the canvas → must clip into [0, W-1] × [0, H-1]
        out = inverse_transform_landmarks(torch.tensor([[-500.0, 500.0]]), M, orig_hw)
        assert out[0, 0] == 0.0
        assert out[0, 1] == 63.0


class TestTtaMatrices:
    def test_rotation_fixes_canvas_center(self):
        mat = _make_tta_matrix(2, 27.0, CPU)
        center = torch.full((2, 1, 2), 0.5)
        moved = warp_landmarks(center, mat)
        torch.testing.assert_close(moved, center, atol=1e-6, rtol=0)

    def test_zero_angle_is_identity(self):
        mat = _make_tta_matrix(1, 0.0, CPU)
        expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        torch.testing.assert_close(mat, expected, atol=1e-7, rtol=0)

    def test_invert_affine_roundtrip(self):
        mat = _make_tta_matrix(2, 27.0, CPU)
        inv = _invert_affine_2x3(mat)
        pts = torch.tensor([[[0.2, 0.3], [0.7, 0.9]], [[0.5, 0.5], [0.1, 0.8]]])
        roundtrip = warp_landmarks(warp_landmarks(pts, mat), inv)
        torch.testing.assert_close(roundtrip, pts, atol=1e-4, rtol=0)

    def test_invert_affine_composes_to_identity(self):
        mat = _make_tta_matrix(1, -13.5, CPU)
        inv = _invert_affine_2x3(mat)
        pad = torch.tensor([[[0.0, 0.0, 1.0]]])
        full = torch.cat([mat, pad], dim=1)
        full_inv = torch.cat([inv, pad], dim=1)
        torch.testing.assert_close(
            full @ full_inv, torch.eye(3).unsqueeze(0), atol=1e-5, rtol=0
        )


class TestParseTtaColorSpecs:
    def test_brightness_and_contrast_tokens(self):
        specs = parse_tta_color_specs("bright+20,bright-20,contr+25,contr-25")
        assert specs == [(0.2, 1.0), (-0.2, 1.0), (0.0, 1.25), (0.0, 0.75)]

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown color TTA token"):
            parse_tta_color_specs("gamma+10")


def _ellipse_endpoints(
    cx: float, cy: float, a: float, b: float, theta: float
) -> np.ndarray:
    """4 endpoints (P0,P1 on axis a; P2,P3 on axis b) of an axis-orthogonal ellipse."""
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])
    c = np.array([cx, cy])
    return np.stack([c - a * u, c + a * u, c - b * v, c + b * v]).astype(np.float32)


class TestEllipseConsensus:
    def test_identical_views_are_a_fixed_point(self):
        ep = _ellipse_endpoints(50.0, 40.0, 20.0, 10.0, 0.4)
        out = _ellipse_consensus_avg(torch.from_numpy(np.stack([ep, ep])))
        np.testing.assert_allclose(out, ep, atol=1e-4)

    def test_swapped_endpoint_pair_is_realigned(self):
        """P0/P1 swapped in view 2 flips the axis direction; consensus must not cancel it."""
        ep = _ellipse_endpoints(50.0, 40.0, 20.0, 10.0, 0.4)
        out = _ellipse_consensus_avg(torch.from_numpy(np.stack([ep, ep[[1, 0, 2, 3]]])))
        np.testing.assert_allclose(out, ep, atol=1e-4)

    def test_swapped_b_axis_pair_is_realigned(self):
        """Same sign-ambiguity resolution for the P2/P3 (b) axis."""
        ep = _ellipse_endpoints(50.0, 40.0, 20.0, 10.0, 0.4)
        out = _ellipse_consensus_avg(torch.from_numpy(np.stack([ep, ep[[0, 1, 3, 2]]])))
        np.testing.assert_allclose(out, ep, atol=1e-4)

    def test_reversed_b_direction_is_preserved(self):
        """Views whose b axis points opposite the (-y, x) perpendicular keep their sign."""
        ep = _ellipse_endpoints(50.0, 40.0, 20.0, 10.0, 0.4)[[0, 1, 3, 2]]
        out = _ellipse_consensus_avg(torch.from_numpy(np.stack([ep, ep])))
        np.testing.assert_allclose(out, ep, atol=1e-4)

    def test_averages_parameters_and_reconstructs_valid_ellipse(self):
        ep1 = _ellipse_endpoints(50.0, 40.0, 20.0, 10.0, 0.4)
        ep2 = _ellipse_endpoints(54.0, 44.0, 24.0, 12.0, 0.4)
        out = _ellipse_consensus_avg(torch.from_numpy(np.stack([ep1, ep2])))

        va, vb = out[1] - out[0], out[3] - out[2]
        # Averaged in parameter space: center, semi-axes
        np.testing.assert_allclose((out[0] + out[1]) / 2, [52.0, 42.0], atol=1e-3)
        np.testing.assert_allclose((out[2] + out[3]) / 2, [52.0, 42.0], atol=1e-3)
        assert np.linalg.norm(va) / 2 == pytest.approx(22.0, abs=1e-3)
        assert np.linalg.norm(vb) / 2 == pytest.approx(11.0, abs=1e-3)
        # Reconstruction contract: orthogonal axes
        assert abs(np.dot(va, vb)) < 1e-3


class TestRefinementGeometry:
    def test_roi_pct(self):
        pts = np.array([[10.0, 10.0], [30.0, 50.0]])
        assert _compute_roi_pct(pts, (100, 200)) == pytest.approx(4.0)
        assert _compute_roi_pct(pts, (0, 0)) == 0.0

    def test_fixed_strategy_centers_on_roi(self):
        pts = np.array([[30.0, 20.0], [50.0, 40.0]])
        x1, y1, side = _compute_crop_region(pts, (64, 96), "fixed", "IVC", 2.0, 0.5)
        assert (x1, y1, side) == (24, 14, 32)

    def test_half_strategy(self):
        pts = np.array([[30.0, 20.0], [50.0, 40.0]])
        _, _, side = _compute_crop_region(pts, (64, 96), "half", "IVC", 2.0)
        assert side == 32  # half the short side, ROI fits

    def test_adaptive_strategy_uses_task_context_scale(self):
        pts = np.array([[40.0, 30.0], [44.0, 33.0]])  # roi_side = 4
        _, _, side = _compute_crop_region(pts, (64, 96), "adaptive", "IVC", 2.0)
        expected = round(4 * TASKS["IVC"].bbox_context_scale * 2.0)
        assert side == min(expected, 64)

    def test_per_task_crop_ratio_dict(self):
        pts = np.array([[40.0, 30.0], [44.0, 33.0]])
        _, _, side = _compute_crop_region(
            pts, (64, 96), "fixed", "IVC", 2.0, {"IVC": 0.25}
        )
        assert side == 16
        _, _, side = _compute_crop_region(
            pts, (64, 96), "fixed", "PSAX", 2.0, {"IVC": 0.25}
        )
        assert side == 32  # not in dict → default 0.5

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown refine strategy"):
            _compute_crop_region(np.zeros((2, 2)), (64, 96), "bogus", "IVC", 2.0)

    def test_crop_clamped_inside_image(self):
        pts = np.array([[1.0, 1.0], [3.0, 3.0]])  # ROI at the corner
        x1, y1, side = _compute_crop_region(pts, (64, 96), "fixed", "IVC", 2.0, 0.5)
        assert x1 == 0 and y1 == 0 and side == 32


class TestSummaries:
    def test_summary_lines(self):
        rows = [
            {"task_id": "IVC", "confidence": 0.8},
            {"task_id": "IVC", "confidence": 0.6},
            {"task_id": "HC", "confidence": 0.9},
        ]
        text = _summary(rows)
        assert "IVC" in text and "n=   2" in text and "conf=0.700" in text

    def test_mre_table_with_and_without_gt(self):
        assert "no GT" in _mre_table([{"task_id": "IVC"}])
        rows = [
            {
                "task_id": "IVC",
                "predicted_points_pixels": [3.0, 4.0, 0.0, 0.0],
                "gt_points_pixels": [0.0, 0.0, 0.0, 0.0],
            }
        ]
        text = _mre_table(rows)
        assert "MRE=2.50" in text  # mean of distances 5 and 0
        assert "Average" in text


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def _make_val_layout(
    data_root: Path,
    images: dict[str, list[tuple[str, int, int]]],
    subdir: str = "val",
) -> None:
    """Create {data_root}/{subdir}/{task}/ for ALL tasks; write the given PNGs."""
    for tid in TASKS:
        (data_root / subdir / tid).mkdir(parents=True, exist_ok=True)
    for tid, files in images.items():
        for name, w, h in files:
            _write_png(data_root / subdir / tid / name, w, h)


class TestDiscovery:
    def test_competition_layout(self, tmp_path: Path):
        _make_val_layout(
            tmp_path, {"IVC": [("b.png", 80, 64), ("a.png", 80, 64)], "PSAX": [("c.png", 96, 64)]}
        )
        (tmp_path / "val" / "IVC" / "notes.txt").write_text("not an image")

        samples = discover_competition_images(tmp_path)
        assert [s["submission_path"] for s in samples] == ["IVC/a.png", "IVC/b.png", "PSAX/c.png"]
        assert samples[0]["image_path"] == "val/IVC/a.png"
        assert samples[0]["task_id"] == "IVC"
        assert samples[0]["gt_pixels_flat"] is None

    def test_missing_task_dir_raises(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        (tmp_path / "val" / "HC").rmdir()
        with pytest.raises(SubmissionError, match="HC"):
            discover_competition_images(tmp_path)

    def test_missing_task_dir_allowed_when_partial_intended(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        (tmp_path / "val" / "HC").rmdir()
        samples = discover_competition_images(tmp_path, allow_missing_tasks=True)
        assert len(samples) == 1

    def test_manifest_split_and_gt(self, tmp_path: Path):
        import polars as pl

        manifest = tmp_path / "manifest.parquet"
        pl.DataFrame(
            {
                "split": ["val_local", "val_local", "train"],
                "task_id": ["IVC", "PSAX", "IVC"],
                "image_path": ["images/IVC/a.png", "images/PSAX/b.png", "images/IVC/c.png"],
                "keypoints": [json.dumps([[10.0, 20.0], [30.0, 40.0]]), None, None],
            }
        ).write_parquet(manifest)

        samples = discover_from_manifest(tmp_path, manifest, "val_local")
        assert len(samples) == 2
        assert samples[0]["submission_path"] == "IVC/a.png"  # "images/" stripped
        assert samples[0]["image_path"] == "images/IVC/a.png"
        assert samples[0]["gt_pixels_flat"] == [10.0, 20.0, 30.0, 40.0]
        assert samples[1]["gt_pixels_flat"] is None

        with pytest.raises(ValueError, match="val_local"):
            discover_from_manifest(tmp_path, manifest, "no_such_split")


# ---------------------------------------------------------------------------
# run_inference with a stub model
# ---------------------------------------------------------------------------


class TestRunInference:
    def test_landmarks_round_trip_to_original_pixels(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)], "PSAX": [("b.png", 96, 72)]})
        samples = discover_competition_images(tmp_path)

        target_ivc = np.array([[12.0, 20.0], [70.0, 50.0]])
        target_psax = np.array([[10.0, 10.0], [88.0, 12.0], [40.0, 60.0], [5.0, 66.0]])
        module = StubModule(
            [
                _stub_output(
                    [
                        ("IVC", _bake_norm(target_ivc, 80, 64)),
                        ("PSAX", _bake_norm(target_psax, 96, 72)),
                    ]
                )
            ]
        )

        results = run_inference(module, samples, tmp_path, INPUT_SIZE, CPU, batch_size=16)

        assert [r["task_id"] for r in results] == ["IVC", "PSAX"]
        np.testing.assert_allclose(
            np.array(results[0]["predicted_points_pixels"]).reshape(-1, 2), target_ivc, atol=0.05
        )
        np.testing.assert_allclose(
            np.array(results[1]["predicted_points_pixels"]).reshape(-1, 2), target_psax, atol=0.05
        )
        # Slot selection followed confidence (slot 1, logit 2.0), not the decoy
        assert results[0]["confidence"] == pytest.approx(torch.tensor(2.0).sigmoid(), abs=1e-3)
        assert results[0]["original_hw"] == [64, 80]
        assert results[1]["original_hw"] == [72, 96]
        for r in results:
            norm = np.array(r["predicted_points_normalized"])
            assert ((norm >= 0.0) & (norm <= 1.0)).all()
            full = r["_canonical_points_full"]
            assert full.shape == (TASKS[r["task_id"]].n_keypoints, 2)

    def test_student_teacher_ensemble_averages_landmarks(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)
        target_s = np.array([[12.0, 20.0], [70.0, 50.0]])
        target_t = np.array([[16.0, 24.0], [66.0, 46.0]])
        module = StubModule([_stub_output([("IVC", _bake_norm(target_s, 80, 64))])])
        teacher_out = _stub_output([("IVC", _bake_norm(target_t, 80, 64))])
        module._ensemble_teacher = lambda normed: teacher_out

        results = run_inference(module, samples, tmp_path, INPUT_SIZE, CPU)

        # Canvas-space average maps to the pixel-space average (affine letterbox)
        expected = (target_s + target_t) / 2
        np.testing.assert_allclose(
            np.array(results[0]["predicted_points_pixels"]).reshape(-1, 2), expected, atol=0.05
        )

    def test_high_conf_reroutes_below_threshold(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)
        target = np.array([[12.0, 20.0], [70.0, 50.0]])
        # Declared IVC scores sigmoid(0)=0.5 < 0.9; PSAX scores sigmoid(4)≈0.982
        module = StubModule(
            [
                _stub_output(
                    [("IVC", _bake_norm(target, 80, 64))],
                    conf_logits={(0, "IVC"): 0.0, (0, "PSAX"): 4.0},
                )
            ]
        )
        results = run_inference(
            module, samples, tmp_path, INPUT_SIZE, CPU, task_routing="high_conf"
        )
        assert results[0]["task_id"] == "PSAX"
        assert len(results[0]["predicted_points_pixels"]) == 2 * TASKS["PSAX"].n_keypoints

    def test_high_conf_keeps_declared_task_above_threshold(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)
        target = np.array([[12.0, 20.0], [70.0, 50.0]])
        # Declared IVC sigmoid(3)≈0.953 ≥ 0.9 — kept even though PSAX scores higher
        module = StubModule(
            [
                _stub_output(
                    [("IVC", _bake_norm(target, 80, 64))],
                    conf_logits={(0, "IVC"): 3.0, (0, "PSAX"): 6.0},
                )
            ]
        )
        results = run_inference(
            module, samples, tmp_path, INPUT_SIZE, CPU, task_routing="high_conf"
        )
        assert results[0]["task_id"] == "IVC"
        np.testing.assert_allclose(
            np.array(results[0]["predicted_points_pixels"]).reshape(-1, 2), target, atol=0.05
        )

    def test_unreadable_image_fails_loudly(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)
        samples[0]["image_path"] = "val/IVC/missing.png"
        module = StubModule([_stub_output([("IVC", np.full((2, 2), 0.5))])])
        with pytest.raises(SubmissionError, match="Cannot read"):
            run_inference(module, samples, tmp_path, INPUT_SIZE, CPU)


# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------


def _rotated_view_norm(desired_norm: np.ndarray, angle: float) -> np.ndarray:
    """What a rotation-equivariant model would emit on the rotated view.

    predict.py inverse-warps the model output through the TTA matrix inverse,
    so baking mat @ desired makes the view resolve back to `desired`.
    """
    mat = _make_tta_matrix(1, angle, CPU)
    pts = torch.from_numpy(desired_norm.astype(np.float32)).reshape(1, -1, 2)
    return warp_landmarks(pts, mat).reshape(-1, 2).numpy()


class TestRunInferenceTta:
    def test_single_view_falls_back_to_plain_inference(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)
        target = np.array([[12.0, 20.0], [70.0, 50.0]])
        module = StubModule([_stub_output([("IVC", _bake_norm(target, 80, 64))])])
        results = run_inference_tta(
            module, samples, tmp_path, INPUT_SIZE, CPU, tta_angles=[0.0]
        )
        assert module.model.calls == 1
        np.testing.assert_allclose(
            np.array(results[0]["predicted_points_pixels"]).reshape(-1, 2), target, atol=0.05
        )

    def test_views_average_in_original_pixel_space(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)

        target_a = np.array([[12.0, 20.0], [70.0, 50.0]])  # base view resolves here
        target_b = np.array([[16.0, 24.0], [66.0, 46.0]])  # rotated view resolves here
        angle = 20.0
        module = StubModule(
            [
                _stub_output([("IVC", _bake_norm(target_a, 80, 64))]),
                _stub_output(
                    [("IVC", _rotated_view_norm(_bake_norm(target_b, 80, 64), angle))]
                ),
            ]
        )

        results = run_inference_tta(
            module, samples, tmp_path, INPUT_SIZE, CPU, tta_angles=[0.0, angle]
        )
        assert module.model.calls == 2
        expected = (target_a + target_b) / 2
        np.testing.assert_allclose(
            np.array(results[0]["predicted_points_pixels"]).reshape(-1, 2), expected, atol=0.05
        )
        norm = np.array(results[0]["predicted_points_normalized"]).reshape(-1, 2)
        np.testing.assert_allclose(norm, expected / np.array([80.0, 64.0]), atol=1e-3)

    def test_color_views_have_no_geometric_inverse(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"IVC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)
        target_a = np.array([[12.0, 20.0], [70.0, 50.0]])
        target_b = np.array([[20.0, 28.0], [62.0, 42.0]])
        module = StubModule(
            [
                _stub_output([("IVC", _bake_norm(target_a, 80, 64))]),
                _stub_output([("IVC", _bake_norm(target_b, 80, 64))]),
            ]
        )
        results = run_inference_tta(
            module,
            samples,
            tmp_path,
            INPUT_SIZE,
            CPU,
            tta_angles=[0.0],
            tta_color=[(0.2, 1.0)],
        )
        assert module.model.calls == 2
        expected = (target_a + target_b) / 2
        np.testing.assert_allclose(
            np.array(results[0]["predicted_points_pixels"]).reshape(-1, 2), expected, atol=0.05
        )

    def test_ellipse_consensus_for_hc(self, tmp_path: Path):
        _make_val_layout(tmp_path, {"HC": [("a.png", 80, 64)]})
        samples = discover_competition_images(tmp_path)

        ep = _ellipse_endpoints(40.0, 32.0, 20.0, 10.0, 0.3)
        angle = 15.0
        module = StubModule(
            [
                _stub_output([("HC", _bake_norm(ep, 80, 64))]),
                _stub_output([("HC", _rotated_view_norm(_bake_norm(ep, 80, 64), angle))]),
            ]
        )
        results = run_inference_tta(
            module,
            samples,
            tmp_path,
            INPUT_SIZE,
            CPU,
            tta_angles=[0.0, angle],
            ellipse_consensus=True,
        )
        out = np.array(results[0]["predicted_points_pixels"]).reshape(4, 2)
        # Both views resolve to the same ellipse → consensus reproduces it
        np.testing.assert_allclose(out, ep, atol=0.1)
        va, vb = out[1] - out[0], out[3] - out[2]
        cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
        assert abs(cos) < 1e-3  # reconstructed axes stay orthogonal


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


class TestRunRefinement:
    def _first_pass_entry(self, pts: np.ndarray, w: int, h: int) -> dict:
        normed = pts / np.array([w, h], dtype=np.float64)
        return {
            "image_path": "val/IVC/a.png",
            "submission_path": "IVC/a.png",
            "task_id": "IVC",
            "predicted_points_pixels": [round(float(v), 2) for v in pts.flatten()],
            "predicted_points_normalized": [round(float(v), 6) for v in normed.flatten()],
            "confidence": 0.9,
            "original_hw": [h, w],
            "gt_points_pixels": [1.0, 2.0, 3.0, 4.0],
        }

    def test_crop_coordinates_map_back_to_original(self, tmp_path: Path):
        w, h = 96, 64
        _write_png(tmp_path / "val" / "IVC" / "a.png", w, h)
        pass1_pts = np.array([[30.0, 20.0], [50.0, 40.0]])
        first_pass = [self._first_pass_entry(pass1_pts, w, h)]

        x1, y1, side = _compute_crop_region(pass1_pts, (h, w), "fixed", "IVC", 2.0, 0.5)
        target2 = np.array([[34.5, 22.25], [47.0, 33.5]])  # inside the crop window
        crop_M = _letterbox_matrix(side, side)
        crop_norm = (
            _forward_map(target2 - np.array([x1, y1]), crop_M)
            / np.array(INPUT_SIZE, dtype=np.float64)
        ).astype(np.float32)

        module = StubModule([_stub_output([("IVC", crop_norm)])])
        results = run_refinement(
            module,
            first_pass,
            tmp_path,
            INPUT_SIZE,
            CPU,
            batch_size=4,
            strategy="fixed",
            crop_ratio=0.5,
        )

        r = results[0]
        np.testing.assert_allclose(
            np.array(r["predicted_points_pixels"]).reshape(-1, 2), target2, atol=0.05
        )
        assert r["refine_crop"] == {"x1": x1, "y1": y1, "side": side, "strategy": "fixed"}
        assert r["gt_points_pixels"] == [1.0, 2.0, 3.0, 4.0]  # carried through
        norm = np.array(r["predicted_points_normalized"])
        assert ((norm >= 0.0) & (norm <= 1.0)).all()

    def test_task_filter_skips_untouched_entries(self, tmp_path: Path):
        w, h = 96, 64
        _write_png(tmp_path / "val" / "IVC" / "a.png", w, h)
        entry = self._first_pass_entry(np.array([[30.0, 20.0], [50.0, 40.0]]), w, h)
        module = StubModule([_stub_output([("IVC", np.full((2, 2), 0.5))])])
        results = run_refinement(
            module,
            [entry],
            tmp_path,
            INPUT_SIZE,
            CPU,
            batch_size=4,
            strategy="fixed",
            refine_tasks={"PSAX"},
        )
        assert results[0] is entry  # not refined
        assert module.model.calls == 0


# ---------------------------------------------------------------------------
# Output writing + ordering re-encoding
# ---------------------------------------------------------------------------


def _result_entry(tid: str, name: str, pts: np.ndarray, w: int, h: int) -> dict:
    normed = pts / np.array([w, h], dtype=np.float64)
    return {
        "image_path": f"val/{tid}/{name}",
        "submission_path": f"{tid}/{name}",
        "task_id": tid,
        "predicted_points_normalized": [round(float(v), 6) for v in normed.flatten()],
        "predicted_points_pixels": [round(float(v), 2) for v in pts.flatten()],
        "_canonical_points_full": pts.astype(np.float32),
        "confidence": 0.9,
        "original_hw": [h, w],
    }


class TestSaveOutputs:
    # fetal_femur competition rule orders the pair by ascending x —
    # canonical points with descending x MUST come out swapped.
    CANONICAL = np.array([[50.0, 10.0], [20.0, 12.0]])

    def test_competition_ordering_applied(self, tmp_path: Path):
        results = [_result_entry("fetal_femur", "a.png", self.CANONICAL, 80, 64)]
        save_outputs(results, tmp_path, {"run": "test"})

        sub = json.loads((tmp_path / "regression_predictions.json").read_text())
        dup = json.loads((tmp_path / "landmark_predictions.json").read_text())
        assert sub == dup  # both organizer-specified names, identical content
        assert sub[0]["image_path"] == "fetal_femur/a.png"
        assert sub[0]["predicted_points_pixels"] == [20.0, 12.0, 50.0, 10.0]
        assert sub[0]["predicted_points_normalized"] == pytest.approx(
            [20 / 80, 12 / 64, 50 / 80, 10 / 64], abs=1e-5
        )

        detail = json.loads((tmp_path / "predictions_detail.json").read_text())
        assert detail["metadata"]["run"] == "test"
        assert detail["metadata"]["competition_ordering"] is True
        assert detail["metadata"]["detail_landmark_order"] == "canonical"
        assert detail["metadata"]["submission_landmark_order"] == "competition"
        # Detail keeps canonical order and drops internal fields
        assert detail["predictions"][0]["predicted_points_pixels"] == [50.0, 10.0, 20.0, 12.0]
        assert "_canonical_points_full" not in detail["predictions"][0]

    def test_ordering_disabled_keeps_canonical(self, tmp_path: Path):
        results = [_result_entry("fetal_femur", "a.png", self.CANONICAL, 80, 64)]
        save_outputs(results, tmp_path, {}, competition_ordering=False)
        sub = json.loads((tmp_path / "regression_predictions.json").read_text())
        assert sub[0]["predicted_points_pixels"] == [50.0, 10.0, 20.0, 12.0]

    def test_missing_canonical_full_falls_back_to_rounded(self, tmp_path: Path):
        entry = _result_entry("fetal_femur", "a.png", self.CANONICAL, 80, 64)
        del entry["_canonical_points_full"]
        save_outputs([entry], tmp_path, {})
        sub = json.loads((tmp_path / "regression_predictions.json").read_text())
        assert sub[0]["predicted_points_pixels"] == [50.0, 10.0, 20.0, 12.0]


# ---------------------------------------------------------------------------
# load_module (real FUBioModule, stub backbone, hand-saved checkpoint)
# ---------------------------------------------------------------------------


class TestLoadModule:
    def test_roundtrip_from_saved_checkpoint(self, tmp_path: Path, stub_backbone):
        module = make_module()
        ckpt_path = tmp_path / "test.ckpt"
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": "2.0.0",
                "epoch": 0,
                "global_step": 0,
            },
            ckpt_path,
        )

        loaded, config = load_module(str(ckpt_path), device="cpu")

        assert config.backbone.input_size == module.config.backbone.input_size
        assert not loaded.training
        assert all(not p.requires_grad for p in loaded.parameters())
        sd_orig, sd_loaded = module.state_dict(), loaded.state_dict()
        assert sd_orig.keys() == sd_loaded.keys()
        assert all(torch.equal(sd_orig[k], sd_loaded[k]) for k in sd_orig)


# ---------------------------------------------------------------------------
# main() — CLI end to end with a stubbed load_module
# ---------------------------------------------------------------------------


def _install_stub_loader(monkeypatch: pytest.MonkeyPatch, module: StubModule) -> None:
    import fubio.serving.predict as predict_mod

    class _Cfg:
        class backbone:  # noqa: N801 — duck-types ExperimentConfig.backbone
            input_size = INPUT_SIZE

    monkeypatch.setattr(predict_mod, "load_module", lambda ckpt, device: (module, _Cfg))


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    from fubio.serving.predict import main

    monkeypatch.setattr(sys, "argv", ["predict", *argv])
    main()


class TestMain:
    def _targets_for_all_tasks(self, w: int, h: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(7)
        return {
            tid: rng.uniform([4.0, 4.0], [w - 5.0, h - 5.0], size=(t.n_keypoints, 2)).round(1)
            for tid, t in TASKS.items()
        }

    def test_competition_mode_writes_validated_submission(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        w, h = 80, 64
        data_root = tmp_path / "data"
        out_dir = tmp_path / "out"
        _make_val_layout(data_root, {tid: [("a.png", w, h)] for tid in TASKS})
        # Batch order follows sorted(TASKS) — the discovery order
        targets = self._targets_for_all_tasks(w, h)
        per_sample = [(tid, _bake_norm(targets[tid], w, h)) for tid in sorted(TASKS)]
        module = StubModule([_stub_output(per_sample)])
        _install_stub_loader(monkeypatch, module)

        _run_main(
            monkeypatch,
            [
                "--ckpt", "unused.ckpt",
                "--data-root", str(data_root),
                "--output-dir", str(out_dir),
                "--device", "cpu",
            ],
        )

        sub = json.loads((out_dir / "regression_predictions.json").read_text())
        assert len(sub) == len(TASKS)
        by_path = {e["image_path"]: e for e in sub}
        for tid in TASKS:
            entry = by_path[f"{tid}/a.png"]
            assert entry["task_id"] == tid
            # Competition ordering may permute rows — compare as point sets,
            # in-bounds original pixel space
            _assert_same_points(entry["predicted_points_pixels"], targets[tid])
            pts = np.array(entry["predicted_points_pixels"]).reshape(-1, 2)
            assert (pts[:, 0] < w).all() and (pts[:, 1] < h).all() and (pts >= 0).all()
            norm = np.array(entry["predicted_points_normalized"])
            assert ((norm >= 0.0) & (norm <= 1.0)).all()

        detail = json.loads((out_dir / "predictions_detail.json").read_text())
        assert detail["metadata"]["n_predictions"] == len(TASKS)
        assert detail["metadata"]["mode"] == "competition"

        # The ordering re-encoding demonstrably ran: fetal_femur's submission
        # pair is x-ascending regardless of the canonical (detail) order.
        ff = by_path["fetal_femur/a.png"]["predicted_points_pixels"]
        assert ff[0] <= ff[2]

    def test_val_local_mode_with_refinement_and_gt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import polars as pl

        w, h = 80, 64
        data_root = tmp_path / "data"
        out_dir = tmp_path / "out"
        for tid, name in [("IVC", "a.png"), ("PSAX", "b.png")]:
            _write_png(data_root / "images" / tid / name, w, h)

        target_ivc = np.array([[30.0, 24.0], [50.0, 40.0]])
        target_psax = np.array([[20.0, 20.0], [60.0, 20.0], [60.0, 44.0], [20.0, 44.0]])
        manifest = tmp_path / "manifest.parquet"
        pl.DataFrame(
            {
                "split": ["val_local", "val_local"],
                "task_id": ["IVC", "PSAX"],
                "image_path": ["images/IVC/a.png", "images/PSAX/b.png"],
                "keypoints": [
                    json.dumps(target_ivc.tolist()),
                    json.dumps(target_psax.tolist()),
                ],
            }
        ).write_parquet(manifest)

        pass1 = _stub_output(
            [
                ("IVC", _bake_norm(target_ivc, w, h)),
                ("PSAX", _bake_norm(target_psax, w, h)),
            ]
        )
        # Refinement pass: keep predictions at the crop-canvas center; the
        # exact crop mapping is covered in TestRunRefinement.
        pass2 = _stub_output([("IVC", np.full((2, 2), 0.5)), ("PSAX", np.full((4, 2), 0.5))])
        module = StubModule([pass1, pass2])
        _install_stub_loader(monkeypatch, module)

        _run_main(
            monkeypatch,
            [
                "--ckpt", "unused.ckpt",
                "--data-root", str(data_root),
                "--output-dir", str(out_dir),
                "--device", "cpu",
                "--mode", "val_local",
                "--manifest", str(manifest),
                "--allow-missing-tasks",
                "--refine", "fixed",
                "--refine-tasks", "IVC,PSAX",
            ],
        )

        assert module.model.calls == 2  # first pass + refinement pass
        detail = json.loads((out_dir / "predictions_detail.json").read_text())
        preds = detail["predictions"]
        assert len(preds) == 2
        for p in preds:
            assert p["refine_crop"]["strategy"] == "fixed"
            assert p["gt_points_pixels"]  # GT carried into detail output
            pts = np.array(p["predicted_points_pixels"]).reshape(-1, 2)
            assert (pts[:, 0] < w).all() and (pts[:, 1] < h).all() and (pts >= 0).all()

    def test_empty_discovery_fails_before_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        data_root = tmp_path / "data"
        out_dir = tmp_path / "out"
        _make_val_layout(data_root, {})  # all task dirs exist but hold no images
        _install_stub_loader(monkeypatch, StubModule([]))

        with pytest.raises(SubmissionError, match="No samples discovered"):
            _run_main(
                monkeypatch,
                [
                    "--ckpt", "unused.ckpt",
                    "--data-root", str(data_root),
                    "--output-dir", str(out_dir),
                    "--device", "cpu",
                ],
            )
        assert not out_dir.exists()

    def test_high_conf_routing_with_tta(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        w, h = 80, 64
        data_root = tmp_path / "data"
        out_dir = tmp_path / "out"
        _make_val_layout(data_root, {"IVC": [("a.png", w, h)]})
        for tid in TASKS:
            if tid != "IVC":
                (data_root / "val" / tid).rmdir()

        target = np.array([[12.0, 20.0], [70.0, 50.0]])
        angle = 10.0
        baked = _bake_norm(target, w, h)
        conf = {(0, "IVC"): 3.0}  # above threshold → stays IVC
        module = StubModule(
            [
                _stub_output([("IVC", baked)], conf_logits=conf),
                _stub_output([("IVC", _rotated_view_norm(baked, angle))], conf_logits=conf),
                _stub_output([("IVC", baked)], conf_logits=conf),
            ]
        )
        _install_stub_loader(monkeypatch, module)

        _run_main(
            monkeypatch,
            [
                "--ckpt", "unused.ckpt",
                "--data-root", str(data_root),
                "--output-dir", str(out_dir),
                "--device", "cpu",
                "--allow-missing-tasks",
                "--task-routing", "high_conf",
                "--tta-angles", f"0,{angle}",
                "--tta-color", "bright+20",
            ],
        )

        assert module.model.calls == 3  # base + one angle + one color view
        sub = json.loads((out_dir / "regression_predictions.json").read_text())
        assert len(sub) == 1
        assert sub[0]["task_id"] == "IVC"
        # All three views resolve to the same target → the average equals it
        np.testing.assert_allclose(
            np.array(sub[0]["predicted_points_pixels"]).reshape(-1, 2), target, atol=0.05
        )

"""Behavioural lock on horizontal-flip augmentation.

Written before removing the unconsumed `flip_applied` field so the removal can be
shown not to alter augmentation. Everything asserted here is a property of the
augmentation itself — image content, keypoint coordinates, the affine matrix and
the per-task gate — none of which ever read `flip_applied`.

Flip is driven entirely by `SpatialParams.hflip` -> `build_affine_matrix`
(`data/spatial.py`), which uses the pixel-centre convention x' = (W-1) - x.

Note: landmarks are deliberately NOT permuted under flip — they are anatomically
defined, not positional (`transforms.py`). That decision is out of scope here;
these tests pin the geometry, not the labelling semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from fubio.data.spatial import SpatialParams, apply_spatial, build_affine_matrix
from fubio.data.transforms import TransformConfig, TransformPipeline


def _sample(n_inst: int = 1, size: int = 64) -> dict:
    """Minimal sample dict with an asymmetric image so mirroring is detectable."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    image[:, : size // 4] = 0  # dark left edge — asymmetry marker

    kp = np.full((n_inst, 60, 2), np.nan, dtype=np.float32)
    kp[0, 0] = [10.0, 20.0]
    kp[0, 1] = [50.0, 40.0]

    valid = np.zeros((n_inst, 60), dtype=bool)
    valid[0, :2] = True

    return {
        "image": image,
        "keypoints": kp,
        "transform_matrix": np.stack([np.eye(3, dtype=np.float32)] * n_inst),
        "landmark_valid_mask": valid,
        "landmark_supervised_mask": valid.copy(),
        "task_ids": np.zeros(n_inst, dtype=np.int64),
        "image_path": "x.png",
        "original_hw": np.array([size, size], dtype=np.int32),
        "is_labeled": np.ones(n_inst, dtype=bool),
    }


class TestFlipGeometry:
    def test_hflip_matrix_uses_pixel_centre_convention(self) -> None:
        """x' = (W-1) - x, not W - x. An off-by-one here shifts every landmark."""
        w = h = 64
        m = build_affine_matrix(
            src_size=(w, h),
            params=SpatialParams(target_size=(w, h), hflip=True),
            resize_mode="letterbox",
        )
        pt = np.array([[10.0, 20.0, 1.0]])
        out = (pt @ m.T)[0, :2]
        assert out[0] == pytest.approx(w - 1 - 10.0)
        assert out[1] == pytest.approx(20.0)

    def test_hflip_mirrors_image_and_keypoints_consistently(self) -> None:
        """The transformed keypoint must land on the transformed image content."""
        size = 64
        s = _sample(size=size)
        m = build_affine_matrix(
            src_size=(size, size),
            params=SpatialParams(target_size=(size, size), hflip=True),
            resize_mode="letterbox",
        )
        img_out, kp_out, _ = apply_spatial(s["image"], s["keypoints"], m, (size, size))

        # The dark quarter moves from the left edge to the right edge.
        assert img_out[:, -1].sum() == 0
        assert img_out[:, 0].sum() > 0

        assert kp_out[0, 0, 0] == pytest.approx(size - 1 - 10.0)
        assert kp_out[0, 0, 1] == pytest.approx(20.0)

    def test_double_flip_is_identity(self) -> None:
        size = 64
        s = _sample(size=size)
        m = build_affine_matrix(
            src_size=(size, size),
            params=SpatialParams(target_size=(size, size), hflip=True),
            resize_mode="letterbox",
        )
        _, kp1, _ = apply_spatial(s["image"], s["keypoints"], m, (size, size))
        _, kp2, _ = apply_spatial(s["image"], kp1, m, (size, size))
        np.testing.assert_allclose(kp2[0, :2], s["keypoints"][0, :2], atol=1e-4)


class TestFlipPipelineIntegration:
    def test_pipeline_applies_flip_when_probability_is_one(self) -> None:
        cfg = TransformConfig(target_size=(64, 64), hflip_prob=1.0, skip_normalize=True)
        pipe = TransformPipeline(cfg, seed=0)
        rng = np.random.default_rng(0)
        assert pipe._sample_spatial_params(rng, (64, 64)).hflip is True

    def test_pipeline_never_flips_when_probability_is_zero(self) -> None:
        cfg = TransformConfig(target_size=(64, 64), hflip_prob=0.0, skip_normalize=True)
        pipe = TransformPipeline(cfg, seed=0)
        rng = np.random.default_rng(0)
        assert pipe._sample_spatial_params(rng, (64, 64)).hflip is False

    def test_end_to_end_flip_changes_keypoints(self) -> None:
        """Full __call__ path: with flip on, x coordinates must mirror."""
        s = _sample(size=64)

        off = TransformPipeline(
            TransformConfig(target_size=(64, 64), hflip_prob=0.0, skip_normalize=True), seed=0
        )(dict(s))
        on = TransformPipeline(
            TransformConfig(target_size=(64, 64), hflip_prob=1.0, skip_normalize=True), seed=0
        )(dict(s))

        x_off = off["keypoints"][0, 0, 0]
        x_on = on["keypoints"][0, 0, 0]
        assert x_on == pytest.approx(63.0 - x_off, abs=1e-3)

    def test_vflip_mirrors_y_coordinates(self) -> None:
        """With vflip on, y coordinates must mirror."""
        s = _sample(size=64)

        off = TransformPipeline(
            TransformConfig(target_size=(64, 64), vflip_prob=0.0, skip_normalize=True), seed=0
        )(dict(s))
        on = TransformPipeline(
            TransformConfig(target_size=(64, 64), vflip_prob=1.0, skip_normalize=True), seed=0
        )(dict(s))

        y_off = off["keypoints"][0, 0, 1]
        y_on = on["keypoints"][0, 0, 1]
        assert y_on == pytest.approx(63.0 - y_off, abs=1e-3)

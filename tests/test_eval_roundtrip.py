"""Guard tests for the eval coordinate-transform invariant.

Locks two facts the validation MRE (train/module.py) silently depends on:

1. build_affine_matrix ∘ map_to_original is identity — the forward affine
   (original → augmented pixel) inverts cleanly, so predictions mapped back
   to original space are unbiased.
2. For letterbox resize to a SQUARE canvas with no geometric augmentation
   (the validation setup: rotation=0, scale=1, translate=0), the original-pixel
   radial error equals normalized_error × max(orig_h, orig_w). This is why
   module._accumulate_metrics can use a single scalar `pixel_scale` instead of
   the full inverse affine — it is exact, not an approximation, UNDER THESE
   CONDITIONS. If validation ever enables geometric aug or a non-square canvas,
   this identity breaks and the metric must switch to map_to_original.
"""

from __future__ import annotations

import numpy as np
import pytest

from fubio.data.spatial import SpatialParams, build_affine_matrix, map_to_original


def _pad_to_60(pts: np.ndarray) -> np.ndarray:
    """map_to_original hardcodes 60 landmarks; pad a small point set to fit."""
    out = np.full((60, 2), np.nan, dtype=np.float32)
    out[: len(pts)] = pts
    return out


@pytest.mark.parametrize("orig_wh", [(800, 600), (600, 800), (512, 512), (1024, 300)])
@pytest.mark.parametrize("target", [224, 518])
def test_letterbox_roundtrip_is_identity(orig_wh: tuple[int, int], target: int) -> None:
    """original → augmented → original recovers the input within sub-pixel error."""
    src_w, src_h = orig_wh
    gt = np.array(
        [[10.0, 10.0], [src_w - 20.0, 30.0], [src_w / 2, src_h / 2], [40.0, src_h - 15.0]],
        dtype=np.float64,
    )

    m = build_affine_matrix(
        src_size=(src_w, src_h),
        params=SpatialParams(target_size=(target, target)),
        resize_mode="letterbox",
    )

    ones = np.ones((len(gt), 1))
    aug = (np.hstack([gt, ones]) @ m.T)[:, :2]  # original → augmented pixel

    recovered = map_to_original(_pad_to_60(aug.astype(np.float32)), m.astype(np.float32))
    recovered = recovered[: len(gt)]

    assert np.max(np.abs(recovered - gt)) < 1e-3


@pytest.mark.parametrize("orig_wh", [(800, 600), (600, 800), (1024, 300)])
def test_pixel_scale_identity_holds_for_square_letterbox(orig_wh: tuple[int, int]) -> None:
    """norm_error × max(orig_h, orig_w) == true original-pixel radial error.

    Justifies the scalar pixel_scale in module._accumulate_metrics.
    """
    src_w, src_h = orig_wh
    target = 224
    p1 = np.array([120.0, 200.0])
    p2 = np.array([180.0, 260.0])
    e_orig = float(np.linalg.norm(p1 - p2))

    m = build_affine_matrix(
        src_size=(src_w, src_h),
        params=SpatialParams(target_size=(target, target)),
        resize_mode="letterbox",
    )
    ones = np.ones((2, 1))
    aug = (np.hstack([np.stack([p1, p2]), ones]) @ m.T)[:, :2]
    norm = aug / target  # keypoints are stored as aug_pixel / target_size
    e_norm = float(np.linalg.norm(norm[0] - norm[1]))

    pixel_scale = max(src_w, src_h)
    assert e_norm * pixel_scale == pytest.approx(e_orig, rel=1e-4)


class TestServedInverseIsTheTestedInverse:
    """The repo has two implementations of the same inverse; only one was tested.

    `spatial.map_to_original` is what the tests above exercise, but
    `serving/predict.py` calls `postprocessing.inverse_transform_landmarks`. A
    defect in the served one would have passed every existing test.

    They must agree everywhere in bounds, and differ ONLY by the clip to
    [0, W-1] x [0, H-1] that the serving variant applies.
    """

    @pytest.mark.parametrize("orig_wh", [(800, 600), (600, 800), (512, 512), (1024, 300)])
    @pytest.mark.parametrize("target", [224, 518])
    def test_two_inverses_agree_in_bounds(self, orig_wh: tuple[int, int], target: int) -> None:
        import torch

        from fubio.evaluation.postprocessing import inverse_transform_landmarks

        src_w, src_h = orig_wh
        gt = np.array(
            [[10.0, 10.0], [src_w - 20.0, 30.0], [src_w / 2, src_h / 2], [40.0, src_h - 15.0]],
            dtype=np.float64,
        )
        m = build_affine_matrix(
            src_size=(src_w, src_h),
            params=SpatialParams(target_size=(target, target)),
            resize_mode="letterbox",
        )
        aug = (np.hstack([gt, np.ones((len(gt), 1))]) @ m.T)[:, :2]

        via_spatial = map_to_original(_pad_to_60(aug.astype(np.float32)), m.astype(np.float32))[
            : len(gt)
        ]
        via_serving = inverse_transform_landmarks(
            torch.from_numpy(aug.astype(np.float32)),
            m.astype(np.float32),
            np.array([src_h, src_w], dtype=np.int32),
        )

        assert np.max(np.abs(via_serving - via_spatial)) < 1e-2
        assert np.max(np.abs(via_serving - gt)) < 1e-2

    def test_clip_is_the_only_difference(self) -> None:
        """Out of bounds: serving clips into the image, spatial does not."""
        import torch

        from fubio.evaluation.postprocessing import inverse_transform_landmarks

        src_w, src_h, target = 800, 600, 224
        m = build_affine_matrix(
            src_size=(src_w, src_h),
            params=SpatialParams(target_size=(target, target)),
            resize_mode="letterbox",
        )
        # A point in the letterbox padding maps outside the original image.
        aug = np.array([[-50.0, -50.0], [target + 50.0, target + 50.0]], dtype=np.float32)

        via_spatial = map_to_original(_pad_to_60(aug), m.astype(np.float32))[:2]
        via_serving = inverse_transform_landmarks(
            torch.from_numpy(aug),
            m.astype(np.float32),
            np.array([src_h, src_w], dtype=np.int32),
        )

        assert (via_spatial[0] < 0).any(), "test is vacuous unless the point is out of bounds"
        expected = np.clip(via_spatial, [0, 0], [src_w - 1, src_h - 1])
        assert np.max(np.abs(via_serving - expected)) < 1e-3

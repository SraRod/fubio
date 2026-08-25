"""Integration tests for fubio.data pipeline (Steps 1–9).

All tests use synthetic data — no real images or manifest required.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from fubio.data.collate import collate_fubio
from fubio.data.manifest import SampleDict
from fubio.data.mosaic import MosaicConfig, MosaicDataset
from fubio.data.sampler import MixedTaskBatchSampler
from fubio.data.spatial import (
    SpatialParams,
    apply_spatial,
    build_affine_matrix,
    map_to_original,
)
from fubio.data.split import create_stratified_val_split
from fubio.data.task_registry import (
    PARAMS,
    TASKS,
    K,
    global_hflip_permutation,
    global_vflip_permutation,
    int_to_task_id,
    landmark_valid_mask,
    local_to_canonical,
    task_id_to_int,
)
from fubio.data.transforms import TransformConfig, TransformPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sample(
    task_id: str,
    labeled: bool = True,
    h: int = 480,
    w: int = 640,
    seed: int = 0,
) -> SampleDict:
    """Create a synthetic SampleDict matching ManifestDataset output."""
    rng = np.random.default_rng(seed)
    task = TASKS[task_id]
    image = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    kp = np.full((1, K, 2), np.nan, dtype=np.float32)
    if labeled:
        local_kp = rng.random((task.n_keypoints, 2)).astype(np.float32)
        local_kp[:, 0] = local_kp[:, 0] * w * 0.8 + w * 0.1
        local_kp[:, 1] = local_kp[:, 1] * h * 0.8 + h * 0.1
        kp[0, task.offset : task.offset + task.n_keypoints] = local_kp
    valid = landmark_valid_mask(task_id)
    finite = ~np.isnan(kp[0]).any(axis=1)
    supervised = valid & finite
    return {
        "image": image,
        "keypoints": kp,
        "transform_matrix": np.eye(3, dtype=np.float32)[np.newaxis],
        "landmark_valid_mask": valid[np.newaxis],
        "landmark_supervised_mask": supervised[np.newaxis],
        "landmark_visible_mask": supervised.copy()[np.newaxis],
        "task_ids": np.array([task.task_int], dtype=np.int64),
        "is_labeled": np.array([labeled], dtype=bool),
        "image_paths": [f"fake/{task_id}/test.png"],
        "original_hws": np.array([[h, w]], dtype=np.int32),
    }


def make_multi_instance_sample(
    task_id: str, n_instances: int = 4, h: int = 518, w: int = 518
) -> SampleDict:
    """Create a SampleDict with I > 1 (as MosaicDataset would produce)."""
    task = TASKS[task_id]
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (h, w, 3), dtype=np.uint8).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32)

    kp = np.full((n_instances, K, 2), np.nan, dtype=np.float32)
    valid = np.zeros((n_instances, K), dtype=bool)
    supervised = np.zeros((n_instances, K), dtype=bool)
    visible = np.zeros((n_instances, K), dtype=bool)
    for i in range(n_instances):
        v = landmark_valid_mask(task_id)
        valid[i] = v
        local_kp = rng.random((task.n_keypoints, 2)).astype(np.float32)
        local_kp[:, 0] *= w * 0.8
        local_kp[:, 1] *= h * 0.8
        kp[i, task.offset : task.offset + task.n_keypoints] = local_kp
        fin = ~np.isnan(kp[i]).any(axis=1)
        supervised[i] = v & fin
        visible[i] = supervised[i].copy()

    return {
        "image": image,
        "keypoints": kp,
        "transform_matrix": np.stack([np.eye(3, dtype=np.float32)] * n_instances),
        "landmark_valid_mask": valid,
        "landmark_supervised_mask": supervised,
        "landmark_visible_mask": visible,
        "task_ids": np.full(n_instances, task.task_int, dtype=np.int64),
        "is_labeled": np.ones(n_instances, dtype=bool),
        "image_paths": [f"fake/{task_id}/img_{i}.png" for i in range(n_instances)],
        "original_hws": np.full((n_instances, 2), [h, w], dtype=np.int32),
    }


# ===================================================================
# 1. task_registry
# ===================================================================


class TestTaskRegistry:
    def test_valid_mask_a4c(self):
        """A4C occupies slots [0, 16) — exactly 16 True bits."""
        mask = landmark_valid_mask("A4C")
        assert mask.shape == (60,)
        assert mask.sum() == 16
        assert mask[:16].all()
        assert not mask[16:].any()

    def test_valid_mask_fetal_femur(self):
        """fetal_femur occupies slots [58, 60) — exactly 2 True bits."""
        mask = landmark_valid_mask("fetal_femur")
        assert mask.sum() == 2
        assert mask[58:60].all()
        assert not mask[:58].any()

    def test_offsets_cover_full_range(self):
        """All 60 slots are covered by exactly one task — no gaps or overlaps."""
        covered = np.zeros(K, dtype=bool)
        for task in TASKS.values():
            rng = slice(task.offset, task.offset + task.n_keypoints)
            assert not covered[rng].any(), f"{task.task_id} overlaps"
            covered[rng] = True
        assert covered.all(), f"Gaps: {np.where(~covered)[0]}"

    def test_local_to_canonical_shape(self):
        """Correct mapping from local [N, 2] → canonical [60, 2]."""
        local = np.random.rand(16, 2).astype(np.float32)
        canon = local_to_canonical(local, "A4C")
        assert canon.shape == (60, 2)
        np.testing.assert_array_equal(canon[:16], local)
        assert np.isnan(canon[16:]).all()

    def test_local_to_canonical_shape_mismatch(self):
        """Rejects wrong number of keypoints."""
        wrong = np.random.rand(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            local_to_canonical(wrong, "A4C")

    def test_local_to_canonical_half_nan(self):
        """Rejects half-NaN (one coord finite, the other NaN)."""
        local = np.random.rand(16, 2).astype(np.float32)
        local[3, 0] = np.nan
        with pytest.raises(ValueError, match="Half-NaN"):
            local_to_canonical(local, "A4C")

    @pytest.mark.parametrize("perm_fn", [global_hflip_permutation, global_vflip_permutation])
    def test_flip_permutation_self_inverse(self, perm_fn):
        """perm[perm] == arange(60) — applying twice restores order."""
        perm = perm_fn()
        np.testing.assert_array_equal(perm[perm], np.arange(K))

    @pytest.mark.parametrize("task_id", list(TASKS.keys()))
    def test_task_id_int_roundtrip(self, task_id: str):
        """task_id → int → task_id round-trip."""
        t_int = task_id_to_int(task_id)
        assert int_to_task_id(t_int) == task_id

    def test_params_count(self):
        """28 clinical parameters defined."""
        assert len(PARAMS) == 28


# ===================================================================
# 2. spatial
# ===================================================================


class TestSpatial:
    def test_identity_stretch(self):
        """Same-size stretch → M ≈ identity."""
        params = SpatialParams(target_size=(640, 480))
        M = build_affine_matrix((640, 480), params, resize_mode="stretch")
        np.testing.assert_allclose(M, np.eye(3), atol=1e-10)

    def test_letterbox_golden(self):
        """640×480 → 518×518 letterbox: known scale and padding."""
        params = SpatialParams(target_size=(518, 518))
        M = build_affine_matrix((640, 480), params, resize_mode="letterbox")

        s = 518 / 640  # 0.809375
        pad_x = (518 - 640 * s) / 2.0  # 0.5
        pad_y = (518 - 480 * s) / 2.0  # 65.0

        # Source (0, 0) → (pad_x, pad_y) in target
        src_pt = np.array([0.0, 0.0, 1.0])
        dst_pt = M @ src_pt
        np.testing.assert_allclose(dst_pt[:2], [pad_x, pad_y], atol=1e-6)

        # Source (639, 479) → (pad_x + 639*s, pad_y + 479*s)
        src_pt2 = np.array([639.0, 479.0, 1.0])
        dst_pt2 = M @ src_pt2
        np.testing.assert_allclose(dst_pt2[:2], [pad_x + 639 * s, pad_y + 479 * s], atol=1e-6)

    def test_round_trip(self):
        """Forward → inverse recovers original coordinates within tolerance."""
        params = SpatialParams(
            target_size=(518, 518),
            rotation_deg=15.0,
            scale=1.1,
            translate_x=5.0,
            translate_y=-3.0,
        )
        M = build_affine_matrix((640, 480), params, resize_mode="letterbox")

        kp_orig = np.array([[[320.0, 240.0], [100.0, 50.0]]], dtype=np.float32)
        kp_full = np.full((1, K, 2), np.nan, dtype=np.float32)
        kp_full[0, :2] = kp_orig[0]

        image_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _, kp_aug, _ = apply_spatial(image_dummy, kp_full, M, (518, 518))
        kp_back = map_to_original(kp_aug, M[np.newaxis])

        np.testing.assert_allclose(kp_back[0, 0], kp_orig[0, 0], atol=0.05)
        np.testing.assert_allclose(kp_back[0, 1], kp_orig[0, 1], atol=0.05)

    def test_zero_M_produces_nan(self):
        """map_to_original with zero M → all-NaN output."""
        kp = np.random.rand(1, K, 2).astype(np.float32)
        M_zero = np.zeros((1, 3, 3), dtype=np.float64)
        result = map_to_original(kp, M_zero)
        assert np.isnan(result).all()

    def test_oob_points_not_visible(self):
        """Points transformed outside canvas bounds are marked not visible."""
        params = SpatialParams(target_size=(100, 100))
        M = build_affine_matrix((100, 100), params, resize_mode="stretch")

        kp = np.full((1, K, 2), np.nan, dtype=np.float32)
        kp[0, 0] = [50.0, 50.0]  # in bounds
        kp[0, 1] = [200.0, 50.0]  # out of bounds x
        kp[0, 2] = [50.0, -10.0]  # out of bounds y

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, _, vis = apply_spatial(image, kp, M, (100, 100))

        assert vis[0, 0], "In-bounds point should be visible"
        assert not vis[0, 1], "OOB x should not be visible"
        assert not vis[0, 2], "OOB y should not be visible"


# ===================================================================
# 3. transforms
# ===================================================================


class TestTransforms:
    def test_val_transform_shapes(self):
        """Val transform (no augmentation) produces correct shapes."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        sample = make_sample("A4C")
        out = pipe(sample)

        assert out["image"].shape == (518, 518, 3)
        assert out["image"].dtype == np.float32
        assert out["keypoints"].shape == (1, 60, 2)
        assert out["transform_matrix"].shape == (1, 3, 3)

    def test_mask_invariant_after_transform(self):
        """visible ⊆ supervised ⊆ valid after transform."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)

        for task_id in ["A4C", "HC", "IVC", "fetal_femur"]:
            out = pipe(make_sample(task_id))
            vis = out["landmark_visible_mask"][0]
            sup = out["landmark_supervised_mask"][0]
            val = out["landmark_valid_mask"][0]
            assert np.all(vis <= sup), f"{task_id}: visible ⊄ supervised"
            assert np.all(sup <= val), f"{task_id}: supervised ⊄ valid"

    def test_unlabeled_masks_all_false(self):
        """Unlabeled samples have supervised=False and visible=False."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        out = pipe(make_sample("HC", labeled=False))
        assert not out["landmark_supervised_mask"].any()
        assert not out["landmark_visible_mask"].any()
        assert not out["is_labeled"].any()

    def test_normalization_range(self):
        """Normalized image values should be in reasonable range."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        out = pipe(make_sample("A4C"))
        img = out["image"]
        assert img.dtype == np.float32
        assert img.min() > -3.0
        assert img.max() < 4.0

    def test_hwc_preserved(self):
        """Output image is still HWC (CHW conversion happens in collate)."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        out = pipe(make_sample("A4C"))
        assert out["image"].ndim == 3
        assert out["image"].shape[2] == 3


# ===================================================================
# 4. collate
# ===================================================================


class TestCollate:
    def test_basic_shapes(self):
        """B=3, I=1 produces correct tensor shapes."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        batch = [pipe(make_sample(t, seed=i)) for i, t in enumerate(["A4C", "HC", "IVC"])]
        out = collate_fubio(batch)

        assert out["image"].shape == (3, 3, 518, 518)
        assert out["keypoints"].shape == (3, 1, 60, 2)
        assert out["transform_matrix"].shape == (3, 1, 3, 3)
        assert out["instance_mask"].shape == (3, 1)
        assert out["task_ids"].shape == (3, 1)
        assert out["landmark_valid_mask"].shape == (3, 1, 60)

    def test_no_nan_after_collate(self):
        """NaN keypoints are replaced with 0.0."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        batch = [pipe(make_sample("A4C")), pipe(make_sample("IVC", labeled=False))]
        out = collate_fubio(batch)
        assert not torch.isnan(out["keypoints"]).any()

    def test_mixed_instance_padding(self):
        """I=1 + I=4 → padded to I=4 with instance_mask."""
        s1 = make_sample("A4C")
        s4 = make_multi_instance_sample("HC", n_instances=4)

        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        t1 = pipe(s1)

        out = collate_fubio([t1, s4])

        assert out["keypoints"].shape == (2, 4, 60, 2)
        assert out["instance_mask"].shape == (2, 4)
        # First sample: 1 real, 3 padding
        assert out["instance_mask"][0, 0]
        assert not out["instance_mask"][0, 1:].any()
        # Second sample: all 4 real
        assert out["instance_mask"][1].all()

    def test_padding_values(self):
        """Padded slots have correct sentinel values."""
        s1 = make_sample("A4C")
        s4 = make_multi_instance_sample("HC", n_instances=4)

        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        t1 = pipe(s1)
        out = collate_fubio([t1, s4])

        # Padded instances of first sample (indices 1, 2, 3)
        assert (out["keypoints"][0, 1:] == 0.0).all()
        assert (out["transform_matrix"][0, 1:] == 0.0).all()
        assert not out["landmark_valid_mask"][0, 1:].any()
        assert (out["task_ids"][0, 1:] == -1).all()

    def test_hwc_to_chw(self):
        """Image converted from HWC to CHW."""
        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        out = collate_fubio([pipe(make_sample("A4C"))])
        assert out["image"].shape == (1, 3, 518, 518)

    def test_image_paths_structure(self):
        """image_paths is list[list[str]], padded with empty strings."""
        s1 = make_sample("A4C")
        s4 = make_multi_instance_sample("HC", n_instances=4)

        cfg = TransformConfig()
        pipe = TransformPipeline(cfg)
        out = collate_fubio([pipe(s1), s4])

        assert isinstance(out["image_paths"], list)
        assert len(out["image_paths"]) == 2
        assert len(out["image_paths"][0]) == 4  # padded to I_max
        assert out["image_paths"][0][0] != ""
        assert out["image_paths"][0][1] == ""  # padding


# ===================================================================
# 5. sampler
# ===================================================================


class TestSampler:
    @pytest.fixture()
    def task_indices(self) -> dict[str, list[int]]:
        return {
            "AOP": list(range(100)),
            "HC": list(range(100, 150)),
            "IVC": list(range(150, 160)),
        }

    def test_batch_size(self, task_indices: dict[str, list[int]]):
        """All batches have the configured size."""
        sampler = MixedTaskBatchSampler(task_indices, batch_size=9, seed=42)
        for batch in sampler:
            assert len(batch) == 9

    def test_reproducibility(self, task_indices: dict[str, list[int]]):
        """Same seed → same batches."""
        s1 = list(MixedTaskBatchSampler(task_indices, batch_size=9, seed=42))
        s2 = list(MixedTaskBatchSampler(task_indices, batch_size=9, seed=42))
        assert s1 == s2

    def test_set_epoch_changes_output(self, task_indices: dict[str, list[int]]):
        """Different epoch → different batch order."""
        sampler = MixedTaskBatchSampler(task_indices, batch_size=9, seed=42)
        b0 = list(sampler)
        sampler.set_epoch(1)
        b1 = list(sampler)
        assert b0 != b1

    def test_largest_task_fully_covered(self, task_indices: dict[str, list[int]]):
        """With drop_last=False, all indices of the largest task appear."""
        sampler = MixedTaskBatchSampler(task_indices, batch_size=9, seed=42, drop_last=False)
        all_indices: set[int] = set()
        for batch in sampler:
            all_indices.update(batch)
        aop_indices = set(task_indices["AOP"])
        assert aop_indices.issubset(all_indices), "Largest task not fully covered"


# ===================================================================
# 6. split
# ===================================================================


def _split_df(n: int, task: str, group_of, extra_splits: dict[str, int] | None = None):
    """Manifest-shaped frame for split tests. Sizes vary so strata are non-degenerate."""
    splits = ["train_labeled"] * n
    for value, count in (extra_splits or {}).items():
        splits += [value] * count
    total = len(splits)
    return pl.DataFrame(
        {
            "split": splits,
            "task_id": [task] * total,
            "group_id": [group_of(i) for i in range(total)],
            "image_path": [f"img_{i}.png" for i in range(total)],
            "height": [400 + (i % 4) * 50 for i in range(total)],
            "width": [600 + (i % 3) * 80 for i in range(total)],
            "roi_bbox": [
                f"[10, 10, {100 + (i % 5) * 30}, {90 + (i % 5) * 20}]" for i in range(total)
            ],
        }
    )


class TestSplit:
    def test_group_level_isolation(self):
        """No group_id appears in both train_local and val_local."""
        df = _split_df(20, "A4C", lambda i: f"study_{i // 2}")
        result, _ = create_stratified_val_split(df, val_targets={"A4C": 4}, seed=42)

        train_groups = set(result.filter(pl.col("split") == "train_local")["group_id"].to_list())
        val_groups = set(result.filter(pl.col("split") == "val_local")["group_id"].to_list())
        assert not train_groups & val_groups, "Group leakage between splits"

    def test_non_labeled_pass_through(self):
        """Rows outside the labeled pool keep their original split."""
        df = _split_df(10, "A4C", lambda i: f"g_{i}", extra_splits={"val": 3, "train_unlabeled": 5})
        result, _ = create_stratified_val_split(df, val_targets={"A4C": 2}, seed=42)

        assert result.filter(pl.col("split") == "val").height == 3
        assert result.filter(pl.col("split") == "train_unlabeled").height == 5
        assert result.filter(pl.col("split") == "train_labeled").height == 0

    def test_all_labeled_rows_assigned(self):
        """Every labeled row becomes either train_local or val_local."""
        df = _split_df(30, "HC", lambda i: f"g_{i}")
        result, _ = create_stratified_val_split(df, val_targets={"HC": 6}, seed=42)

        n_train = result.filter(pl.col("split") == "train_local").height
        n_val = result.filter(pl.col("split") == "val_local").height
        assert n_train + n_val == 30

    def test_rerunnable_from_existing_split(self):
        """Re-splitting reclaims train_local/val_local instead of compounding.

        The previous CoreSet builder consumed `train_labeled` and never restored
        it, so a second run silently did nothing.
        """
        df = _split_df(30, "HC", lambda i: f"g_{i}")
        once, _ = create_stratified_val_split(df, val_targets={"HC": 6}, seed=42)
        twice, _ = create_stratified_val_split(once, val_targets={"HC": 12}, seed=7)

        assert twice.filter(pl.col("split") == "val_local").height >= 12
        assert twice.height == once.height

    def test_every_stratum_represented(self):
        """Each stratum contributes at least one group before any contributes two."""
        df = _split_df(40, "FA", lambda i: f"g_{i}")
        result, report = create_stratified_val_split(df, val_targets={"FA": 8}, seed=42, n_bins=2)

        task_report = report["tasks"]["FA"]
        assert task_report["n_strata_covered"] == task_report["n_strata"]
        assert not task_report["uncovered_strata"]
        assert result.filter(pl.col("split") == "val_local").height >= 8

    def test_preserves_row_order(self):
        """Row order must survive the re-split — silent corruption otherwise.

        CachedManifestDataset indexes the image memmap by manifest row POSITION
        (manifest.py `_cache_idx`). A reordered manifest therefore pairs cached
        images with another row's keypoints, and nothing raises.
        """
        df = _split_df(40, "FA", lambda i: f"g_{i // 2}")
        result, _ = create_stratified_val_split(df, val_targets={"FA": 8}, seed=42)

        assert result["image_path"].to_list() == df["image_path"].to_list()
        assert result.height == df.height

    def test_only_split_column_changes(self):
        """Everything the image cache is keyed on must come through untouched."""
        df = _split_df(40, "FA", lambda i: f"g_{i // 2}")
        result, _ = create_stratified_val_split(df, val_targets={"FA": 8}, seed=42)

        for col in ("image_path", "task_id", "group_id", "height", "width", "roi_bbox"):
            assert result[col].to_list() == df[col].to_list(), f"{col} was modified"

    def test_deterministic_under_seed(self):
        """Same seed reproduces the split exactly; a different seed does not."""
        df = _split_df(40, "FA", lambda i: f"g_{i // 2}")
        a, _ = create_stratified_val_split(df, val_targets={"FA": 8}, seed=42)
        b, _ = create_stratified_val_split(df, val_targets={"FA": 8}, seed=42)
        c, _ = create_stratified_val_split(df, val_targets={"FA": 8}, seed=99)

        val_of = lambda d: set(d.filter(pl.col("split") == "val_local")["image_path"].to_list())  # noqa: E731
        assert val_of(a) == val_of(b)
        assert val_of(a) != val_of(c)


# ===================================================================
# 7. mosaic min_labeled
# ===================================================================


class FakeDataset:
    """Minimal dataset stub for MosaicDataset tests."""

    def __init__(self, samples: list[SampleDict]):
        self._samples = samples
        self._task_map: dict[str, list[int]] = {}
        self._labeled: list[int] = []
        for i, s in enumerate(samples):
            tid = int_to_task_id(int(s["task_ids"][0]))
            self._task_map.setdefault(tid, []).append(i)
            if s["is_labeled"][0]:
                self._labeled.append(i)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SampleDict:
        return self._samples[idx]

    def get_task_indices(self, task_id: str) -> list[int]:
        return self._task_map.get(task_id, [])

    def get_split_indices(self, split: str) -> list[int]:
        return list(range(len(self._samples)))

    def get_labeled_indices(self) -> list[int]:
        return list(self._labeled)


class TestMosaicMinLabeled:
    def _make_mixed_dataset(self, n_labeled: int = 5, n_unlabeled: int = 5):
        """Create a FakeDataset with labeled + unlabeled HC samples."""
        samples = []
        for i in range(n_labeled):
            samples.append(make_sample("HC", labeled=True, seed=i))
        for i in range(n_unlabeled):
            samples.append(make_sample("HC", labeled=False, seed=100 + i))
        return FakeDataset(samples)

    def test_min_labeled_1_guarantee(self):
        """With min_labeled=1, every mosaic has at least 1 labeled instance."""
        ds = self._make_mixed_dataset(n_labeled=3, n_unlabeled=10)
        cfg = MosaicConfig(rows=2, cols=2, min_labeled=1, mode="mixed")
        mosaic = MosaicDataset(ds, cfg, seed=42)

        for idx in range(len(ds)):
            out = mosaic[idx]
            n_labeled = int(out["is_labeled"].sum())
            assert n_labeled >= 1, f"idx={idx}: got {n_labeled} labeled"

    def test_min_labeled_0_allows_all_unlabeled(self):
        """With min_labeled=0, all-unlabeled mosaic is permitted."""
        ds = self._make_mixed_dataset(n_labeled=0, n_unlabeled=10)
        cfg = MosaicConfig(rows=2, cols=2, min_labeled=0, mode="mixed")
        mosaic = MosaicDataset(ds, cfg, seed=42)
        out = mosaic[0]
        assert out["is_labeled"].shape[0] == 4

    def test_invalid_min_labeled_rejected(self):
        """min_labeled > n_instances raises ValueError."""
        with pytest.raises(ValueError, match="min_labeled"):
            MosaicConfig(rows=2, cols=2, min_labeled=5)

    def test_same_task_labeled_guarantee(self):
        """same_task mode: replaced companions are same task and labeled."""
        samples = []
        for i in range(3):
            samples.append(make_sample("A4C", labeled=True, seed=i))
        for i in range(7):
            samples.append(make_sample("A4C", labeled=False, seed=50 + i))
        ds = FakeDataset(samples)
        cfg = MosaicConfig(rows=2, cols=2, min_labeled=2, mode="same_task")
        mosaic = MosaicDataset(ds, cfg, seed=42)

        for idx in range(3):
            out = mosaic[idx]
            n_labeled = int(out["is_labeled"].sum())
            assert n_labeled >= 2, f"idx={idx}: got {n_labeled} labeled"
            for i in range(out["task_ids"].shape[0]):
                assert int_to_task_id(int(out["task_ids"][i])) == "A4C"

    def test_100_runs_min_labeled(self):
        """Statistical: 100 runs, all satisfy min_labeled=1."""
        ds = self._make_mixed_dataset(n_labeled=2, n_unlabeled=20)
        cfg = MosaicConfig(rows=2, cols=2, min_labeled=1, mode="mixed")
        for seed in range(100):
            mosaic = MosaicDataset(ds, cfg, seed=seed)
            out = mosaic[0]
            assert int(out["is_labeled"].sum()) >= 1


# ===================================================================
# 8. transforms skip_resize / skip_normalize / flip XOR
# ===================================================================


class TestTransformFlags:
    def test_skip_resize_identity(self):
        """skip_resize=True → output same size as input."""
        sample = make_sample("A4C", h=480, w=640)
        cfg = TransformConfig(skip_resize=True, skip_normalize=True)
        pipe = TransformPipeline(cfg)
        out = pipe(sample)
        assert out["image"].shape == (480, 640, 3)
        assert out["image"].dtype == np.uint8

    def test_skip_normalize_uint8(self):
        """skip_normalize=True → output is uint8."""
        sample = make_sample("HC")
        cfg = TransformConfig(skip_normalize=True)
        pipe = TransformPipeline(cfg)
        out = pipe(sample)
        assert out["image"].dtype == np.uint8

    def test_skip_resize_with_da(self):
        """skip_resize + rotation → output same size, keypoints rotated."""
        sample = make_sample("HC", h=300, w=400, seed=7)
        cfg = TransformConfig(
            skip_resize=True,
            skip_normalize=True,
            rotation_range=15.0,
            scale_range=(0.9, 1.1),
        )
        pipe = TransformPipeline(cfg)

        np.random.default_rng_orig = np.random.default_rng
        np.random.default_rng = lambda s=None: np.random.default_rng_orig(42)
        try:
            out = pipe(sample)
        finally:
            np.random.default_rng = np.random.default_rng_orig

        assert out["image"].shape == (300, 400, 3)
        assert not np.array_equal(out["keypoints"], sample["keypoints"]), (
            "Keypoints should change with rotation"
        )


class TestHflipSchedule:
    """hflip must be closable per-epoch without mutating the frozen config."""

    @staticmethod
    def _pipeline_with_hflip():
        cfg = TransformConfig(target_size=(64, 64), hflip_prob=1.0)
        return TransformPipeline(cfg)

    def test_enabled_by_default(self):
        pipe = self._pipeline_with_hflip()
        rng = np.random.default_rng(0)
        params = pipe._sample_spatial_params(rng, (64, 64))
        assert params.hflip is True

    def test_disabled_after_close(self):
        pipe = self._pipeline_with_hflip()
        pipe.set_flips_enabled(False)
        rng = np.random.default_rng(0)
        # hflip_prob=1.0, so any flip observed here comes from the toggle failing.
        assert pipe._sample_spatial_params(rng, (64, 64)).hflip is False

    def test_toggle_does_not_mutate_config(self):
        """TransformConfig is frozen and shared with workers — must stay untouched."""
        pipe = self._pipeline_with_hflip()
        pipe.set_flips_enabled(False)
        assert pipe._cfg.hflip_prob == 1.0

    def test_reversible(self):
        pipe = self._pipeline_with_hflip()
        pipe.set_flips_enabled(False)
        pipe.set_flips_enabled(True)
        rng = np.random.default_rng(0)
        assert pipe._sample_spatial_params(rng, (64, 64)).hflip is True


class TestAugmentationDeterminism:
    """seed must actually control augmentation.

    TransformPipeline previously built `np.random.default_rng()` per call with no
    seed, so `seed: 42` controlled nothing and no run — nor any A/B between two
    runs — was reproducible.
    """

    @staticmethod
    def _pipe(seed):
        cfg = TransformConfig(
            target_size=(64, 64), rotation_range=30.0, translate_range=10.0, hflip_prob=0.5
        )
        return TransformPipeline(cfg, seed=seed)

    def test_same_seed_same_epoch_reproduces(self):
        a = self._pipe(42)(make_sample("A4C"))
        b = self._pipe(42)(make_sample("A4C"))
        np.testing.assert_allclose(a["keypoints"], b["keypoints"], rtol=0, atol=0)

    def test_different_seed_differs(self):
        a = self._pipe(42)(make_sample("A4C"))
        b = self._pipe(7)(make_sample("A4C"))
        assert not np.allclose(a["keypoints"], b["keypoints"], equal_nan=True)

    def test_epoch_advances_the_stream(self):
        """Otherwise every epoch would replay identical augmentations."""
        p = self._pipe(42)
        a = p(make_sample("A4C"))["keypoints"].copy()
        p.set_epoch(1)
        b = p(make_sample("A4C"))["keypoints"]
        assert not np.allclose(a, b, equal_nan=True)

    def test_repeated_draws_get_different_augmentations(self):
        """The sampler draws one image up to 327x per epoch (IVC has 30 images).

        An earlier version keyed the stream on sample identity, so every one of
        those draws got an identical augmentation — destroying diversity exactly
        where the data is scarcest. Training collapsed from epoch 1.
        """
        p = self._pipe(42)
        draws = [p(make_sample("A4C"))["keypoints"].copy() for _ in range(5)]
        assert any(not np.allclose(draws[0], d, equal_nan=True) for d in draws[1:]), (
            "repeated draws of the same sample must not share one augmentation"
        )

    def test_unseeded_still_works(self):
        """seed=None keeps the old nondeterministic behaviour for ad-hoc use."""
        cfg = TransformConfig(target_size=(64, 64), rotation_range=30.0)
        out = TransformPipeline(cfg)(make_sample("A4C"))
        assert out["keypoints"].shape == (1, K, 2)

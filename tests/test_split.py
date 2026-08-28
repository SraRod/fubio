"""Tests for the group-stratified val split and its pure helpers.

Covers create_stratified_val_split on a synthetic manifest: determinism
(including PYTHONHASHSEED independence via the pinned seed-42 assignment),
row-order preservation, group-leakage isolation, val target sizing, and the
missing-target KeyError. The DINOv2 diversity path is deliberately untested
(GPU + network); only its pure helper kcenter_greedy is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from fubio.data.split import (
    VAL_TARGETS_DEFAULT,
    _audit_no_group_leakage,
    _preprocess_image,
    _quantile_bins,
    app,
    create_stratified_val_split,
    kcenter_greedy,
)

# =========================================================================
# Synthetic manifest
# =========================================================================


def make_manifest() -> pl.DataFrame:
    """Two real tasks × 12 groups with varying group sizes, image geometry,
    ROI area (some rows without ROI), plus unlabeled pass-through rows.

    Initial split values cycle through all LABELED_SPLITS so the builder's
    re-run safety (reclaiming already-split rows) is exercised.
    """
    rows: list[dict] = []
    for task_id in ("IVC", "PSAX"):
        for g in range(12):
            gid = f"{task_id}_g{g:02d}"
            n_imgs = 1 + (g % 3)
            split = ("train_labeled", "train_local", "val_local")[g % 3]
            for i in range(n_imgs):
                roi = None if g % 4 == 0 else json.dumps([10, 10, 30 + 5 * g, 40 + 4 * g])
                rows.append(
                    {
                        "split": split,
                        "task_id": task_id,
                        "group_id": gid,
                        "image_path": f"{task_id}/{gid}_{i}.png",
                        "height": 480 + 10 * g,
                        "width": 600 + 20 * (g % 5),
                        "roi_bbox": roi,
                    }
                )
    for j in range(3):
        rows.append(
            {
                "split": "train_unlabeled",
                "task_id": "IVC",
                "group_id": f"u{j}",
                "image_path": f"unlabeled/{j}.png",
                "height": 480,
                "width": 640,
                "roi_bbox": None,
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def manifest_df() -> pl.DataFrame:
    return make_manifest()


# Captured from seed=42 on make_manifest(); stable across processes and
# PYTHONHASHSEED because the per-task stream is seeded with a blake2b digest
# of the task_id, not builtin hash().
PINNED_VAL_GROUPS_SEED42: dict[str, list[str]] = {
    "IVC": ["IVC_g01", "IVC_g03", "IVC_g06", "IVC_g08", "IVC_g09"],
    "PSAX": ["PSAX_g01", "PSAX_g04", "PSAX_g08", "PSAX_g09", "PSAX_g10"],
}


# =========================================================================
# create_stratified_val_split
# =========================================================================


class TestStratifiedValSplit:
    def test_deterministic_same_seed(self, manifest_df: pl.DataFrame) -> None:
        result1, report1 = create_stratified_val_split(manifest_df, seed=42)
        result2, report2 = create_stratified_val_split(manifest_df, seed=42)
        assert result1.equals(result2)
        assert report1 == report2

    def test_pinned_seed42_assignment(self, manifest_df: pl.DataFrame) -> None:
        """Seed 42 must always produce this exact assignment — a change means
        the split is no longer reproducible across processes/versions."""
        result, _ = create_stratified_val_split(manifest_df, seed=42)
        val = result.filter(pl.col("split") == "val_local")
        for task_id, expected in PINNED_VAL_GROUPS_SEED42.items():
            groups = sorted(set(val.filter(pl.col("task_id") == task_id)["group_id"].to_list()))
            assert groups == expected

    def test_row_order_preserved(self, manifest_df: pl.DataFrame) -> None:
        """CachedManifestDataset indexes the cache by manifest row position —
        the output must keep the exact input row order."""
        result, _ = create_stratified_val_split(manifest_df, seed=42)
        assert result.height == manifest_df.height
        assert result["image_path"].to_list() == manifest_df["image_path"].to_list()

    def test_no_group_leakage(self, manifest_df: pl.DataFrame) -> None:
        result, _ = create_stratified_val_split(manifest_df, seed=42)
        assert _audit_no_group_leakage(result) == []
        # Same invariant checked directly, per task.
        for task_id in ("IVC", "PSAX"):
            task = result.filter(pl.col("task_id") == task_id)
            train_groups = set(task.filter(pl.col("split") == "train_local")["group_id"])
            val_groups = set(task.filter(pl.col("split") == "val_local")["group_id"])
            assert not train_groups & val_groups

    def test_val_targets_approximately_honored(self, manifest_df: pl.DataFrame) -> None:
        """Round-robin stops once the target is met, so overshoot is bounded
        by the largest group size (3 images in this fixture)."""
        result, report = create_stratified_val_split(manifest_df, seed=42)
        max_group_size = 3
        for task_id in ("IVC", "PSAX"):
            target = VAL_TARGETS_DEFAULT[task_id]
            n_val = result.filter(
                (pl.col("task_id") == task_id) & (pl.col("split") == "val_local")
            ).height
            assert target <= n_val <= target + max_group_size - 1
            assert report["tasks"][task_id]["actual_val_images"] == n_val

    def test_labeled_rows_fully_reassigned(self, manifest_df: pl.DataFrame) -> None:
        """Every labeled row becomes train_local or val_local; unlabeled rows
        pass through untouched."""
        result, report = create_stratified_val_split(manifest_df, seed=42)
        labeled = result.filter(pl.col("task_id").is_in(("IVC", "PSAX")))
        unlabeled = result.filter(pl.col("split") == "train_unlabeled")
        # Fixture: unlabeled rows are the only non-labeled ones.
        assert unlabeled.height == 3
        assert unlabeled["image_path"].to_list() == [f"unlabeled/{j}.png" for j in range(3)]
        assert set(labeled.filter(pl.col("group_id").str.starts_with("u").not_())["split"]) == {
            "train_local",
            "val_local",
        }
        assert report["summary"]["n_other"] == 3

    def test_provenance_columns_added(self, manifest_df: pl.DataFrame) -> None:
        result, _ = create_stratified_val_split(manifest_df, seed=7, split_version="v-test")
        assert result["split_version"].unique().to_list() == ["v-test"]
        assert result["split_seed"].unique().to_list() == [7]
        assert result["selection_method"].unique().to_list() == ["group_stratified_random"]

    def test_missing_val_target_raises_keyerror(self, manifest_df: pl.DataFrame) -> None:
        with pytest.raises(KeyError, match="No val target for task 'PSAX'"):
            create_stratified_val_split(manifest_df, val_targets={"IVC": 8})

    def test_no_labeled_rows_returns_error_report(self) -> None:
        df = make_manifest().filter(pl.col("split") == "train_unlabeled")
        result, report = create_stratified_val_split(df, seed=42)
        assert result.equals(df)
        assert report == {"error": "no labeled rows to split"}

    def test_target_beyond_pool_takes_everything(self, manifest_df: pl.DataFrame) -> None:
        """Unreachable target: round-robin exhausts every stratum, then stops —
        all labeled images end up in val_local."""
        result, report = create_stratified_val_split(
            manifest_df, val_targets={"IVC": 10_000, "PSAX": 10_000}, seed=42
        )
        for task_id in ("IVC", "PSAX"):
            task = result.filter(pl.col("task_id") == task_id)
            labeled = task.filter(pl.col("split") != "train_unlabeled")
            assert set(labeled["split"]) == {"val_local"}
            assert report["tasks"][task_id]["actual_train_images"] == 0


# =========================================================================
# CLI (stratified command only — diversity needs GPU + network)
# =========================================================================


class TestStratifiedCLI:
    def _paths(self, tmp_path: Path) -> tuple[Path, Path]:
        manifest = tmp_path / "manifest.parquet"
        make_manifest().write_parquet(manifest)
        return manifest, tmp_path / "report.json"

    def test_dry_run_leaves_manifest_untouched(self, tmp_path: Path) -> None:
        manifest, report_path = self._paths(tmp_path)
        before = manifest.read_bytes()

        result = CliRunner().invoke(
            app,
            [
                "stratified",
                "--manifest", str(manifest),
                "--report-path", str(report_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert manifest.read_bytes() == before
        report = json.loads(report_path.read_text())
        assert report["group_leakage"] == []
        assert set(report["tasks"]) == {"IVC", "PSAX"}

    def test_write_updates_manifest_and_backs_up(self, tmp_path: Path) -> None:
        manifest, report_path = self._paths(tmp_path)
        before = manifest.read_bytes()

        result = CliRunner().invoke(
            app,
            ["stratified", "--manifest", str(manifest), "--report-path", str(report_path)],
        )
        assert result.exit_code == 0, result.output
        # Manifest rewritten with the split applied; previous bytes backed up.
        updated = pl.read_parquet(manifest)
        assert "split_version" in updated.columns
        backups = list(tmp_path.glob("manifest.parquet.bak-*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == before


# =========================================================================
# _preprocess_image (CPU-only piece of the DINOv2 path)
# =========================================================================


class TestPreprocessImage:
    def test_shape_dtype_and_normalization(self, tmp_path: Path) -> None:
        path = tmp_path / "img.png"
        rng = np.random.default_rng(0)
        cv2.imwrite(str(path), rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8))

        arr = _preprocess_image(path, target_size=70)  # 70 = 5 * 14 (patch size)
        assert arr.shape == (3, 70, 70)
        assert arr.dtype == np.float32
        # ImageNet normalization keeps values well within (-3, 3).
        assert np.abs(arr).max() < 3.0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Cannot load"):
            _preprocess_image(tmp_path / "nope.png")


# =========================================================================
# kcenter_greedy
# =========================================================================


class TestKCenterGreedy:
    def test_select_all_when_target_exceeds_pool(self) -> None:
        features = np.eye(3, dtype=np.float32)
        selections = kcenter_greedy(features, n_select=5)
        assert [s["index"] for s in selections] == [0, 1, 2]
        assert all(s["distance"] == 0.0 for s in selections)

    def test_greedy_selection_order(self) -> None:
        """First = farthest from centroid, then argmax of min-distance."""
        features = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [10.0, 10.0]], dtype=np.float32
        )
        selections = kcenter_greedy(features, n_select=2)
        # Farthest from centroid (2.75, 3.25) is [10, 10]; farthest from it is [0, 0].
        assert [s["index"] for s in selections] == [3, 0]
        assert selections[0]["order"] == 0
        assert selections[1]["order"] == 1
        assert selections[1]["distance"] == pytest.approx(np.sqrt(200.0))

    def test_indices_unique_and_count(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(30, 4)).astype(np.float32)
        selections = kcenter_greedy(features, n_select=10)
        indices = [s["index"] for s in selections]
        assert len(indices) == 10
        assert len(set(indices)) == 10
        assert [s["order"] for s in selections] == list(range(10))


# =========================================================================
# _quantile_bins
# =========================================================================


class TestQuantileBins:
    def test_single_bin_is_all_zeros(self) -> None:
        values = np.array([1.0, 5.0, 9.0])
        np.testing.assert_array_equal(_quantile_bins(values, n_bins=1), [0, 0, 0])

    def test_empty_input(self) -> None:
        out = _quantile_bins(np.array([]), n_bins=3)
        assert out.shape == (0,)

    def test_median_split(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        bins = _quantile_bins(values, n_bins=2)
        np.testing.assert_array_equal(bins, [0, 0, 1, 1])

    def test_bins_in_range_and_monotonic(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.uniform(size=100)
        n_bins = 4
        bins = _quantile_bins(values, n_bins)
        assert bins.min() >= 0
        assert bins.max() <= n_bins - 1
        # Bin assignment respects value ordering.
        order = np.argsort(values)
        assert np.all(np.diff(bins[order]) >= 0)

    def test_constant_values_collapse_to_one_bin(self) -> None:
        """Ties collapse downward — identical values land in a single bin."""
        values = np.full(10, 3.14)
        bins = _quantile_bins(values, n_bins=3)
        assert len(set(bins.tolist())) == 1

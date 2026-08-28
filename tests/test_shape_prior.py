"""Tests for the PCA shape prior: schema validation, Procrustes alignment,
provenance verification, and the end-to-end build from a synthetic manifest.

The synthetic manifest mirrors the real schema consumed by _load_task_coords
(split / task_id / keypoints JSON / width / height) plus the split identity
columns compute_shape_prior records.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from fubio.data.shape_prior import (
    ShapePrior,
    TaskShapePrior,
    _procrustes_align,
    compute_shape_prior,
    verify_shape_prior_provenance,
)
from fubio.data.task_registry import TASKS

# =========================================================================
# Helpers
# =========================================================================


def _tiny_task_prior(K: int = 2, M: int = 1, n_samples: int = 5) -> TaskShapePrior:
    """Minimal internally-consistent TaskShapePrior."""
    return TaskShapePrior(
        K=K,
        M=M,
        n_samples=n_samples,
        variance_explained=0.9,
        mean_logit=[[0.0, 0.0]] * K,
        basis=[[1.0] * (2 * K)] * M,
        eigenvalues=[1.0] * M,
    )


def _write_manifest(
    path: Path,
    n_ivc: int = 5,
    seed: int = 0,
    split_seed: int = 42,
    extra_rows: list[dict] | None = None,
) -> None:
    """Synthetic manifest parquet with n_ivc valid IVC train_local rows.

    Keypoints are pixel coordinates well inside the frame so normalization
    stays in [0, 1] and the derived supportive landmarks contain no NaN.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_ivc):
        kps = rng.uniform(30, 70, size=(2, 2)).round(1).tolist()
        rows.append(
            {
                "split": "train_local",
                "split_version": "v2-stratified",
                "split_seed": split_seed,
                "task_id": "IVC",
                "keypoints": json.dumps(kps),
                "width": 100,
                "height": 100,
                "image_path": f"IVC/{i:04d}.png",
            }
        )
    rows.extend(extra_rows or [])
    pl.DataFrame(rows).write_parquet(path)


# =========================================================================
# Schema validation
# =========================================================================


class TestTaskShapePriorValidation:
    def test_valid_prior_accepted(self) -> None:
        prior = _tiny_task_prior(K=3, M=2)
        assert prior.K == 3
        assert prior.canonical_mean is None

    def test_mean_logit_wrong_row_count(self) -> None:
        with pytest.raises(ValidationError, match="mean_logit has 1 rows"):
            TaskShapePrior(
                K=2,
                M=1,
                n_samples=5,
                variance_explained=0.9,
                mean_logit=[[0.0, 0.0]],
                basis=[[1.0] * 4],
                eigenvalues=[1.0],
            )

    def test_mean_logit_row_not_2d(self) -> None:
        with pytest.raises(ValidationError, match=r"mean_logit\[0\] has 3 elements"):
            TaskShapePrior(
                K=2,
                M=1,
                n_samples=5,
                variance_explained=0.9,
                mean_logit=[[0.0, 0.0, 0.0], [0.0, 0.0]],
                basis=[[1.0] * 4],
                eigenvalues=[1.0],
            )

    def test_basis_wrong_row_count(self) -> None:
        with pytest.raises(ValidationError, match="basis has 2 rows, expected M=1"):
            TaskShapePrior(
                K=2,
                M=1,
                n_samples=5,
                variance_explained=0.9,
                mean_logit=[[0.0, 0.0]] * 2,
                basis=[[1.0] * 4, [1.0] * 4],
                eigenvalues=[1.0],
            )

    def test_basis_row_wrong_width(self) -> None:
        with pytest.raises(ValidationError, match=r"basis\[0\] has 3 elements, expected 2K=4"):
            TaskShapePrior(
                K=2,
                M=1,
                n_samples=5,
                variance_explained=0.9,
                mean_logit=[[0.0, 0.0]] * 2,
                basis=[[1.0] * 3],
                eigenvalues=[1.0],
            )

    def test_eigenvalues_wrong_length(self) -> None:
        with pytest.raises(ValidationError, match="eigenvalues has 2 elements, expected M=1"):
            TaskShapePrior(
                K=2,
                M=1,
                n_samples=5,
                variance_explained=0.9,
                mean_logit=[[0.0, 0.0]] * 2,
                basis=[[1.0] * 4],
                eigenvalues=[1.0, 2.0],
            )


class TestShapePriorValidation:
    def test_v3_without_sha_is_malformed(self) -> None:
        with pytest.raises(ValidationError, match="malformed"):
            ShapePrior(
                schema_version=3,
                variance_threshold=0.85,
                m_cap=10,
                manifest_sha256=None,
                tasks={},
            )

    def test_legacy_v1_without_sha_is_accepted(self) -> None:
        prior = ShapePrior(
            schema_version=1,
            variance_threshold=0.85,
            m_cap=10,
            tasks={"IVC": _tiny_task_prior()},
        )
        assert prior.manifest_sha256 is None


# =========================================================================
# Procrustes alignment
# =========================================================================


class TestProcrustesAlign:
    def _base_shape(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        return rng.uniform(0.2, 0.8, size=(1, 5, 2))

    def test_output_is_pose_normalized(self) -> None:
        aligned = _procrustes_align(self._base_shape())
        np.testing.assert_allclose(aligned.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sqrt((aligned**2).sum()), 1.0, atol=1e-12)

    def test_translation_and_scale_removed(self) -> None:
        shape = self._base_shape()
        transformed = shape * 3.7 + np.array([0.5, -1.2])
        np.testing.assert_allclose(
            _procrustes_align(transformed), _procrustes_align(shape), atol=1e-12
        )

    def test_rotation_not_removed(self) -> None:
        """v1 alignment keeps rotational variance — a rotated shape must NOT
        map to the same canonical form."""
        shape = self._base_shape()
        theta = np.pi / 4
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated = shape @ rot.T
        assert not np.allclose(
            _procrustes_align(rotated), _procrustes_align(shape), atol=1e-3
        )


# =========================================================================
# Provenance verification
# =========================================================================


class TestVerifyProvenance:
    def test_missing_prior_returns_silently(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.parquet"
        _write_manifest(manifest)
        verify_shape_prior_provenance(tmp_path / "missing.json", manifest)

    def test_missing_manifest_returns_silently(self, tmp_path: Path) -> None:
        prior_path = tmp_path / "prior.json"
        prior_path.write_text("{not even json")
        verify_shape_prior_provenance(prior_path, tmp_path / "missing.parquet")

    def test_legacy_prior_only_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        manifest = tmp_path / "manifest.parquet"
        _write_manifest(manifest)
        prior_path = tmp_path / "prior.json"
        prior = ShapePrior(
            schema_version=1,
            variance_threshold=0.85,
            m_cap=10,
            tasks={"IVC": _tiny_task_prior()},
        )
        prior_path.write_text(prior.model_dump_json())

        with caplog.at_level(logging.WARNING, logger="fubio.data.shape_prior"):
            verify_shape_prior_provenance(prior_path, manifest)
        assert "legacy prior" in caplog.text

    def test_matching_hash_passes(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.parquet"
        _write_manifest(manifest)
        prior_path = tmp_path / "prior.json"
        prior = ShapePrior(
            schema_version=3,
            variance_threshold=0.85,
            m_cap=10,
            split_version="v2-stratified",
            split_seed=42,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            tasks={"IVC": _tiny_task_prior()},
        )
        prior_path.write_text(prior.model_dump_json())
        verify_shape_prior_provenance(prior_path, manifest)

    def test_mismatched_hash_raises_with_drift_report(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.parquet"
        _write_manifest(manifest, n_ivc=1)
        prior_path = tmp_path / "prior.json"
        prior = ShapePrior(
            schema_version=3,
            variance_threshold=0.85,
            m_cap=10,
            split_version="v2-stratified",
            split_seed=42,
            manifest_sha256="0" * 64,
            tasks={"IVC": _tiny_task_prior(n_samples=5)},
        )
        prior_path.write_text(prior.model_dump_json())

        with pytest.raises(ValueError, match="Stale shape prior") as exc_info:
            verify_shape_prior_provenance(prior_path, manifest)
        # Drift report: prior claims 5 samples, manifest has 1 valid IVC row.
        msg = str(exc_info.value)
        assert "IVC" in msg
        assert "CHANGED" in msg

    def test_mismatched_hash_unknown_task_skipped(self, tmp_path: Path) -> None:
        """Tasks the registry no longer knows are skipped in the drift report."""
        manifest = tmp_path / "manifest.parquet"
        _write_manifest(manifest, n_ivc=1)
        prior_path = tmp_path / "prior.json"
        prior = ShapePrior(
            schema_version=3,
            variance_threshold=0.85,
            m_cap=10,
            manifest_sha256="0" * 64,
            tasks={"RETIRED_TASK": _tiny_task_prior()},
        )
        prior_path.write_text(prior.model_dump_json())

        with pytest.raises(ValueError, match="Stale shape prior") as exc_info:
            verify_shape_prior_provenance(prior_path, manifest)
        assert "RETIRED_TASK" not in str(exc_info.value)


# =========================================================================
# End-to-end build from a synthetic manifest
# =========================================================================


class TestComputeShapePrior:
    def test_build_and_verify_roundtrip(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.parquet"
        # One extra PSAX row: a single valid sample must be skipped (<2 samples).
        psax_row = {
            "split": "train_local",
            "split_version": "v2-stratified",
            "split_seed": 42,
            "task_id": "PSAX",
            "keypoints": json.dumps([[20, 20], [40, 20], [30, 60], [50, 60]]),
            "width": 100,
            "height": 100,
            "image_path": "PSAX/0000.png",
        }
        _write_manifest(manifest, n_ivc=5, extra_rows=[psax_row])

        prior = compute_shape_prior(manifest, variance_threshold=0.85, m_cap=10)

        assert set(prior.tasks) == {"IVC"}
        assert prior.split_version == "v2-stratified"
        assert prior.split_seed == 42
        assert prior.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()

        tp = prior.tasks["IVC"]
        # V3 prior: K includes the 6 IVC supportive landmarks.
        assert tp.K == TASKS["IVC"].n_total == 8
        assert tp.n_samples == 5
        assert 1 <= tp.M <= 4  # capped at N-1
        assert tp.canonical_mean is not None
        assert tp.mean_xy is not None and tp.std_xy is not None
        assert np.all(np.array(tp.std_xy) >= 0.02)  # floored std

        # Round-trip: the freshly built prior must pass its own provenance check.
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(prior.model_dump_json())
        verify_shape_prior_provenance(prior_path, manifest)

    def test_mixed_split_identity_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.parquet"
        foreign_row = {
            "split": "train_local",
            "split_version": "v2-stratified",
            "split_seed": 7,  # differs from the other rows' 42
            "task_id": "IVC",
            "keypoints": json.dumps([[30, 30], [60, 60]]),
            "width": 100,
            "height": 100,
            "image_path": "IVC/9999.png",
        }
        _write_manifest(manifest, n_ivc=3, extra_rows=[foreign_row])

        with pytest.raises(ValueError, match="split identities"):
            compute_shape_prior(manifest)

"""PCA shape prior for per-task landmark structure.

Computes a low-dimensional shape basis from labeled training landmarks
in logit space. The model predicts shape coefficients (global structure)
plus a per-landmark residual (fine correction), enabling data-driven
mean shape initialization and inter-landmark structural coupling.

Upstream: build_shape_prior.py (CLI entry), task_registry.py.
Downstream: models/coord_predictors.py (ShapePriorPredictor loads tensors),
            train/module.py (loads JSON at construction).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, model_validator

from fubio.data.supportive import compute_supportive
from fubio.data.task_registry import TASKS

logger = logging.getLogger(__name__)

EPS = 1e-4

# 1 = pre-provenance (no record of which manifest split produced the statistics).
# 2 = carries split_version / split_seed / manifest_sha256.
# 3 = K includes supportive landmarks (K_total = K_scored + K_supportive).
SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TaskShapePrior(BaseModel, frozen=True):
    """PCA shape prior for one task.

    Two representations, independently consumed:
    - mean_logit / basis (logit space): drives ShapePriorPredictor forward path.
    - canonical_mean / canonical_basis (Procrustes-aligned normalized space):
      drives shape_consistency_loss. Pose (translation + scale) is removed before
      PCA, so these capture SHAPE only — unlike the logit basis whose leading
      components are dominated by pose. Optional (older priors lack them).
    """

    K: int
    M: int
    n_samples: int
    variance_explained: float
    mean_logit: list[list[float]]
    basis: list[list[float]]
    eigenvalues: list[float]
    canonical_mean: list[list[float]] | None = None  # (K, 2) unit-Frobenius aligned
    canonical_basis: list[list[float]] | None = None  # (Mc, 2K) shape subspace
    # Absolute (un-aligned) per-landmark position stats in normalized [0,1] space.
    # Drive HeatmapPredictor's data positional prior (log-Gaussian per landmark):
    # mean = where landmark k usually is, std = how tightly (→ prior anchor size).
    mean_xy: list[list[float]] | None = None  # (K, 2)
    std_xy: list[list[float]] | None = None  # (K, 2)

    @model_validator(mode="after")
    def _check_shapes(self) -> TaskShapePrior:
        if len(self.mean_logit) != self.K:
            raise ValueError(f"mean_logit has {len(self.mean_logit)} rows, expected K={self.K}")
        for i, row in enumerate(self.mean_logit):
            if len(row) != 2:
                raise ValueError(f"mean_logit[{i}] has {len(row)} elements, expected 2")
        if len(self.basis) != self.M:
            raise ValueError(f"basis has {len(self.basis)} rows, expected M={self.M}")
        for i, row in enumerate(self.basis):
            if len(row) != 2 * self.K:
                raise ValueError(f"basis[{i}] has {len(row)} elements, expected 2K={2 * self.K}")
        if len(self.eigenvalues) != self.M:
            raise ValueError(
                f"eigenvalues has {len(self.eigenvalues)} elements, expected M={self.M}"
            )
        return self


class ShapePrior(BaseModel, frozen=True):
    """Collection of per-task shape priors with build metadata."""

    schema_version: int = 1
    coordinate_space: Literal["logit"] = "logit"
    variance_threshold: float
    m_cap: int
    # Which manifest split these statistics came from. None only in schema_version 1
    # files, which predate the field — those are unverifiable, not valid.
    split_version: str | None = None
    split_seed: int | None = None
    manifest_sha256: str | None = None
    tasks: dict[str, TaskShapePrior]

    @model_validator(mode="after")
    def _check_provenance(self) -> ShapePrior:
        if self.schema_version >= SCHEMA_VERSION and self.manifest_sha256 is None:
            raise ValueError(
                f"schema_version={self.schema_version} promises provenance but "
                f"manifest_sha256 is missing — the file is malformed, not merely old"
            )
        return self


# ---------------------------------------------------------------------------
# PCA computation
# ---------------------------------------------------------------------------


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def _procrustes_align(coords_01: np.ndarray) -> np.ndarray:
    """Remove translation + scale from each sample. (N, K, 2) → (N, K, 2).

    Centroid-subtract then divide by Frobenius norm, so each aligned shape has
    zero mean and unit size. Rotation is NOT aligned (v1) — rotational variance
    stays in the data and is absorbed by the PCA basis. shape_consistency_loss
    normalizes predictions the same way; the two must stay in lockstep.
    """
    centered = coords_01 - coords_01.mean(axis=1, keepdims=True)
    scale = np.sqrt((centered**2).sum(axis=(1, 2), keepdims=True))
    return centered / np.maximum(scale, 1e-8)


def _compute_canonical(
    coords_01: np.ndarray, variance_threshold: float, m_cap: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """PCA in the Procrustes-aligned normalized frame. Returns (mean (K,2), basis (Mc,2K))."""
    aligned = _procrustes_align(coords_01).reshape(coords_01.shape[0], 2 * K)
    mean = aligned.mean(axis=0)
    centered = aligned - mean

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    total = eigvals.sum()
    if total < 1e-12:
        mc = 1
    else:
        mc = int(np.searchsorted(np.cumsum(eigvals) / total, variance_threshold) + 1)
    mc = min(mc, m_cap, 2 * K, max(coords_01.shape[0] - 1, 1))

    return mean.reshape(K, 2), eigvecs[:, :mc].T


def _compute_task_pca(
    coords_01: np.ndarray,
    variance_threshold: float,
    m_cap: int,
    K: int,
) -> TaskShapePrior:
    """Compute PCA for a single task's landmarks in logit space.

    Args:
        coords_01: (N, K, 2) normalized landmark coordinates.
        variance_threshold: cumulative variance threshold for selecting M.
        m_cap: maximum number of components.
        K: number of landmarks.

    Returns:
        TaskShapePrior with PCA results.
    """
    N = coords_01.shape[0]
    coords_flat = coords_01.reshape(N, 2 * K)
    coords_logit = _logit(coords_flat)

    mean_logit = coords_logit.mean(axis=0)
    centered = coords_logit - mean_logit

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending — reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        M = 1
    else:
        cumvar = np.cumsum(eigenvalues) / total_var
        M = int(np.searchsorted(cumvar, variance_threshold) + 1)

    # Cap at min(m_cap, K, N-1) — N-1 for rank-deficient cases
    M = min(M, m_cap, K, max(N - 1, 1))

    basis = eigenvectors[:, :M].T  # (M, 2K)
    explained = float(eigenvalues[:M].sum() / total_var) if total_var > 1e-12 else 0.0

    canonical_mean, canonical_basis = _compute_canonical(coords_01, variance_threshold, m_cap, K)

    # Absolute position stats for the heatmap data prior. Floor std at 0.02
    # (~1/3 grid cell @16x16) so a near-constant landmark doesn't produce an
    # infinitely sharp prior that pins the prediction and ignores the image.
    mean_xy = coords_01.mean(axis=0)  # (K, 2)
    std_xy = np.maximum(coords_01.std(axis=0), 0.02)  # (K, 2)

    return TaskShapePrior(
        K=K,
        M=M,
        n_samples=N,
        variance_explained=explained,
        mean_logit=mean_logit.reshape(K, 2).tolist(),
        basis=basis.tolist(),
        eigenvalues=eigenvalues[:M].tolist(),
        canonical_mean=canonical_mean.tolist(),
        canonical_basis=canonical_basis.tolist(),
        mean_xy=mean_xy.tolist(),
        std_xy=std_xy.tolist(),
    )


def _load_task_coords(
    task_df: pl.DataFrame, task_id: str | None = None,
) -> list[np.ndarray]:
    """Manifest rows → per-sample (K_total, 2) coords normalized to [0, 1].

    When task_id is provided AND the task has supportive landmarks,
    supportive coordinates are computed from scored GT and appended.
    Samples with NaN in scored OR supportive are dropped.

    Deterministic given the rows, so verify_shape_prior_provenance can
    replay it to report count drift.
    """
    tdef = TASKS[task_id] if task_id else None
    has_sup = tdef is not None and tdef.n_supportive > 0

    coords: list[np.ndarray] = []
    for row in task_df.iter_rows(named=True):
        kps = json.loads(row["keypoints"])
        w, h = row["width"], row["height"]
        normed = np.array([[x / w, y / h] for x, y in kps], dtype=np.float64)

        if np.any(np.isnan(normed)):
            continue
        if np.any(normed < 0) or np.any(normed > 1):
            continue

        if has_sup:
            sup_coords, _ = compute_supportive(task_id, normed.astype(np.float32))
            if np.any(np.isnan(sup_coords)):
                continue
            normed = np.vstack([normed, sup_coords.astype(np.float64)])

        coords.append(normed)
    return coords


def _manifest_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _split_identity(labeled: pl.DataFrame) -> tuple[str, int]:
    """(split_version, split_seed) of the train_local rows.

    A manifest carrying more than one split identity means two split runs were
    merged; the resulting statistics would be a blend of both, so refuse it.
    """
    rows = labeled.select("split_version", "split_seed").unique().rows()
    if len(rows) != 1:
        raise ValueError(f"manifest train_local has {len(rows)} split identities: {rows}")
    version, seed = rows[0]
    return str(version), int(seed)


def verify_shape_prior_provenance(prior_path: Path, manifest_path: Path) -> None:
    """Fail fast when a prior on disk was built from a different manifest.

    The prior is a pure function of the manifest's train_local rows, so ANY
    manifest change makes it stale — including changes that leave sample counts
    untouched, e.g. a corrected CSV column order. The gate is therefore the
    manifest's content hash; per-task counts only enrich the error message.

    Deliberately called from train.py and NOT from FUBioModule.__init__: serving
    rebuilds the same module via load_from_checkpoint, whose state_dict already
    carries prior_mean/prior_std/_canon_* as buffers, in environments (Docker)
    where no manifest exists.

    Missing files are silently ignored — module.py raises the actionable error
    when a prior is genuinely required. This function only reports staleness.

    Raises:
        ValueError: the prior records provenance and it disagrees with the manifest.
    """
    if not prior_path.exists() or not manifest_path.exists():
        return

    prior = ShapePrior.model_validate_json(prior_path.read_text())

    if prior.manifest_sha256 is None:
        logger.warning(
            "%s is a legacy prior (schema_version=%d) with no provenance — it CANNOT be "
            "checked against %s. Rebuild with `uv run python -m fubio.data.build_shape_prior` "
            "to enable the check.",
            prior_path,
            prior.schema_version,
            manifest_path,
        )
        return

    if _manifest_sha256(manifest_path) == prior.manifest_sha256:
        return

    labeled = pl.read_parquet(manifest_path).filter(pl.col("split") == "train_local")
    drift: list[str] = []
    # V3 priors include supportive; pass task_id so _load_task_coords matches
    _pass_tid = prior.schema_version >= 3
    for tid, tp in sorted(prior.tasks.items()):
        if tid not in TASKS:
            continue
        n_now = len(
            _load_task_coords(
                labeled.filter(pl.col("task_id") == tid),
                task_id=tid if _pass_tid else None,
            )
        )
        mark = "  <-- CHANGED" if n_now != tp.n_samples else ""
        drift.append(f"    {tid:14s} prior={tp.n_samples:5d}  manifest={n_now:5d}{mark}")
    raise ValueError(
        f"Stale shape prior: {prior_path} was built from a different {manifest_path}.\n"
        f"  prior split: {prior.split_version} / seed {prior.split_seed}\n"
        f"  per-task train_local sample counts:\n" + "\n".join(drift) + "\n"
        f"  Rebuild: uv run python -m fubio.data.build_shape_prior --output {prior_path}\n"
        f"  NOTE: rebuilding changes Mc for some tasks, which resizes the _canon_basis_*\n"
        f"  buffers and makes existing checkpoints unloadable. Write a NEW versioned file\n"
        f"  and point loss.shape_prior_path at it instead of overwriting."
    )


def compute_shape_prior(
    manifest_path: Path,
    variance_threshold: float = 0.85,
    m_cap: int = 10,
    include_supportive: bool = True,
) -> ShapePrior:
    """Compute PCA shape prior for all tasks from labeled training data.

    Reads manifest.parquet, filters to train_local split, computes
    per-task PCA in logit space.

    Args:
        manifest_path: path to manifest.parquet (must have split column).
        variance_threshold: cumulative variance threshold for M selection.
        m_cap: hard cap on number of components per task.
        include_supportive: if True, append supportive landmarks (K_total);
            if False, use scored landmarks only (K_scored).

    Returns:
        ShapePrior ready for JSON serialization.
    """
    df = pl.read_parquet(manifest_path)
    labeled = df.filter(pl.col("split") == "train_local")
    split_version, split_seed = _split_identity(labeled)

    task_priors: dict[str, TaskShapePrior] = {}

    for tid, tdef in TASKS.items():
        task_df = labeled.filter(pl.col("task_id") == tid)
        if len(task_df) == 0:
            logger.warning("Task %s: no labeled training samples, skipping", tid)
            continue

        all_coords = _load_task_coords(
            task_df, task_id=tid if include_supportive else None,
        )
        if len(all_coords) < 2:
            logger.warning("Task %s: only %d valid samples, skipping", tid, len(all_coords))
            continue

        coords_01 = np.stack(all_coords)
        k = tdef.n_total if include_supportive else tdef.n_keypoints
        prior = _compute_task_pca(coords_01, variance_threshold, m_cap, k)
        task_priors[tid] = prior

        logger.info(
            "  %-14s K=%2d  M=%2d  Mc=%2d  var=%.1f%%  samples=%d",
            tid,
            k,
            prior.M,
            len(prior.canonical_basis or []),
            prior.variance_explained * 100,
            prior.n_samples,
        )

    return ShapePrior(
        schema_version=SCHEMA_VERSION,
        variance_threshold=variance_threshold,
        m_cap=m_cap,
        split_version=split_version,
        split_seed=split_seed,
        manifest_sha256=_manifest_sha256(manifest_path),
        tasks=task_priors,
    )

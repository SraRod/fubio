"""Train/val split builders for local development.

Two strategies live here, and they answer different questions:

`create_stratified_val_split` (DEFAULT, use this for model selection)
    Group-aware stratified random sampling on metadata derived from the
    manifest alone — no GPU, no image reads. Produces a val set whose
    per-task distribution mirrors the labeled pool, so its mean is an
    approximately unbiased estimate of population risk.

`create_diversity_val_split` (DIAGNOSTIC ONLY, do not select on it)
    k-Center-Greedy CoreSet on DINOv2 CLS features. Selects groups
    farthest from the already-selected val set, i.e. the feature-space
    boundary. This is a deliberately non-representative stress set: its
    mean is a BIASED estimate of population risk, and selecting models on
    it favours whatever happens to help on outliers. Measured 2026-07-25:
    val_local built this way inverted the platform ranking of R15 vs R17.

Val sizing follows coverage-first, not precision-first: guarantee at least
one group per stratum, then return everything else to training. This is
only sound because models are compared *pairwise on the same fixed val
set*, where per-image difficulty cancels — absolute values from a small
val set remain unreliable, so report paired bootstrap CIs on differences,
never CIs on absolute values.

Upstream: build_manifest.py (manifest.parquet with group_id column).
Downstream: ManifestDataset (filters on split column).

Usage:
    uv run python -m fubio.data.split stratified [--manifest data/manifest.parquet]
    uv run python -m fubio.data.split diversity   # stress set, needs GPU
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import polars as pl
import torch
import typer
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

MANIFEST_DEFAULT = Path("data/manifest.parquet")
DATA_ROOT_DEFAULT = Path("data")
REPORT_DEFAULT = Path("data/split_report.json")


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------


def _load_dinov2(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-B/14 for feature extraction."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device).eval()
    return model


def _preprocess_image(image_path: Path, target_size: int = 518) -> np.ndarray:
    """Load and preprocess a single image for DINOv2.

    Returns (3, H, W) float32 normalized with ImageNet stats.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    arr = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)  # HWC → CHW


@torch.no_grad()
def extract_features(
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract DINOv2 CLS features for a list of images.

    Returns (N, D) float32 L2-normalized feature matrix.
    """
    model = _load_dinov2(device)
    d_model = 768  # ViT-B/14

    all_features = np.empty((len(image_paths), d_model), dtype=np.float32)

    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch_imgs = []
        for p in image_paths[start:end]:
            batch_imgs.append(_preprocess_image(p))

        batch_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
        features = model(batch_tensor)  # (B, D)
        features = torch.nn.functional.normalize(features, dim=-1)
        all_features[start:end] = features.cpu().numpy()

        if (start // batch_size) % 10 == 0:
            logger.info("  features: %d/%d images", end, len(image_paths))

    del model
    torch.cuda.empty_cache()
    return all_features


# ---------------------------------------------------------------------------
# k-Center-Greedy (CoreSet selection)
# ---------------------------------------------------------------------------


def kcenter_greedy(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> list[dict]:
    """Select n_select points maximizing minimum pairwise distance.

    Returns list of selection records with rationale:
        [{"index": int, "order": int, "distance": float}, ...]
    distance = min distance to any previously selected point at time of selection.
    First point: distance = distance to feature centroid.
    """
    n = len(features)
    if n_select >= n:
        return [{"index": i, "order": i, "distance": 0.0} for i in range(n)]

    # Start with point farthest from centroid — most "outlier-like"
    centroid = features.mean(axis=0)
    dists_to_centroid = np.linalg.norm(features - centroid, axis=1)
    first = int(dists_to_centroid.argmax())

    selections = [{"index": first, "order": 0, "distance": float(dists_to_centroid[first])}]
    selected_set = {first}

    # min distance from each point to nearest selected point
    min_dists = np.linalg.norm(features - features[first], axis=1)

    for step in range(1, n_select):
        # Mask already selected
        min_dists_masked = min_dists.copy()
        for s in selected_set:
            min_dists_masked[s] = -1.0

        next_idx = int(min_dists_masked.argmax())
        dist_val = float(min_dists[next_idx])

        selections.append({"index": next_idx, "order": step, "distance": dist_val})
        selected_set.add(next_idx)

        # Update min distances
        dists_to_new = np.linalg.norm(features - features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)

    return selections


# ---------------------------------------------------------------------------
# Group-stratified split (default)
# ---------------------------------------------------------------------------

# Per-task val image targets. Coverage-first: small tasks get enough to catch a
# catastrophic per-task regression, large tasks are capped so the surplus goes
# back to training. The previous CoreSet split held 600 AOP images — 15% of a
# 4000-image pool — for a task carrying 1/12 of the domain-balanced score.
VAL_TARGETS_DEFAULT: dict[str, int] = {
    "IVC": 8,
    "PSAX": 10,
    "PLAX": 14,
    "A4C": 16,
    "FUGC": 30,
    "FA": 60,
    "fetal_femur": 70,
    "HC": 90,
    "AOP": 80,
}

# Splits that hold labeled rows eligible for re-assignment. Includes the
# already-split values so the builder is re-runnable — create_diversity_val_split
# consumed `train_labeled` and never restored it, which made it single-shot.
LABELED_SPLITS = ("train_labeled", "train_local", "val_local")


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a within-task quantile bin. Ties collapse downward."""
    if n_bins <= 1 or len(values) == 0:
        return np.zeros(len(values), dtype=int)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, values, side="right")


def _group_strata(task_df: pl.DataFrame, n_bins: int) -> dict[str, tuple[int, int]]:
    """Map each group_id to a stratum key.

    Strata combine ROI area fraction (how much of the frame the target
    occupies — the dominant driver of pixel-error scale) with image aspect
    ratio (a proxy for acquisition/framing convention). Both come from the
    manifest, so this needs no image reads.

    Returns: {group_id: (roi_bin, aspect_bin)}
    """
    rows = task_df.select("group_id", "height", "width", "roi_bbox").to_dicts()

    per_group: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        h, w = float(r["height"]), float(r["width"])
        aspect = w / max(h, 1.0)
        roi_frac = 0.0
        if r["roi_bbox"]:
            x1, y1, x2, y2 = json.loads(r["roi_bbox"])
            roi_frac = abs((x2 - x1) * (y2 - y1)) / max(h * w, 1.0)
        per_group.setdefault(r["group_id"], []).append((roi_frac, aspect))

    groups = sorted(per_group)
    roi = np.array([np.mean([v[0] for v in per_group[g]]) for g in groups])
    asp = np.array([np.mean([v[1] for v in per_group[g]]) for g in groups])

    roi_bin = _quantile_bins(roi, n_bins)
    asp_bin = _quantile_bins(asp, n_bins)
    return {g: (int(roi_bin[i]), int(asp_bin[i])) for i, g in enumerate(groups)}


def _select_val_groups(
    strata: dict[str, tuple[int, int]],
    group_sizes: dict[str, int],
    target_images: int,
    rng: np.random.Generator,
) -> list[str]:
    """Round-robin across strata, one group at a time, until target is met.

    Round-robin (rather than proportional allocation) is what guarantees the
    user-requested invariant: every stratum contributes before any stratum
    contributes twice.
    """
    by_stratum: dict[tuple[int, int], list[str]] = {}
    for gid, key in strata.items():
        by_stratum.setdefault(key, []).append(gid)
    for gids in by_stratum.values():
        rng.shuffle(gids)

    order = sorted(by_stratum)
    selected: list[str] = []
    n_images = 0
    round_idx = 0
    while n_images < target_images:
        progressed = False
        for key in order:
            pool = by_stratum[key]
            if round_idx >= len(pool):
                continue
            gid = pool[round_idx]
            selected.append(gid)
            n_images += group_sizes[gid]
            progressed = True
            if n_images >= target_images:
                break
        if not progressed:  # every stratum exhausted
            break
        round_idx += 1
    return selected


def create_stratified_val_split(
    df: pl.DataFrame,
    val_targets: dict[str, int] | None = None,
    seed: int = 42,
    n_bins: int = 2,
    split_version: str = "v2-stratified",
) -> tuple[pl.DataFrame, dict]:
    """Re-split labeled rows into train_local / val_local, group-aware.

    Pure dataframe transformation — no image access, no GPU. Safe to re-run:
    it reclaims rows from any of LABELED_SPLITS, so an existing split is
    replaced rather than compounded.

    Args:
        df: Manifest with [split, task_id, group_id, image_path, height, width, roi_bbox].
        val_targets: Per-task val image targets; defaults to VAL_TARGETS_DEFAULT.
        seed: Controls within-stratum group shuffling.
        n_bins: Quantile bins per stratification variable (strata = n_bins²).
        split_version: Recorded in the manifest so a run's split is identifiable.

    Returns:
        (updated_df, audit_report). The df gains split_version / split_seed /
        selection_method columns. Rows outside LABELED_SPLITS pass through
        untouched.
    """
    targets = val_targets or VAL_TARGETS_DEFAULT
    labeled = df.filter(pl.col("split").is_in(LABELED_SPLITS))

    if labeled.height == 0:
        return df, {"error": "no labeled rows to split"}

    # Row ORDER must survive untouched. CachedManifestDataset indexes the image
    # memmap by position in the full manifest (manifest.py `_cache_idx`), so a
    # reordered manifest silently pairs images with the wrong keypoints. Only
    # the `split` column values change; assignment is collected as a path set
    # and written back in place rather than concatenating per-task frames.
    val_paths: set[str] = set()

    report: dict = {
        "selection_method": "group_stratified_random",
        "split_version": split_version,
        "seed": seed,
        "n_bins": n_bins,
        "n_labeled_total": labeled.height,
        "tasks": {},
    }

    for task_id in sorted(labeled["task_id"].unique().to_list()):
        task_df = labeled.filter(pl.col("task_id") == task_id)
        target = targets.get(task_id)
        if target is None:
            raise KeyError(
                f"No val target for task {task_id!r}; add it to VAL_TARGETS_DEFAULT "
                f"or pass val_targets explicitly (tasks: {sorted(targets)})"
            )

        group_sizes = dict(task_df.group_by("group_id").len().rename({"len": "n"}).iter_rows())
        strata = _group_strata(task_df, n_bins)

        # Stable per-task stream: builtin hash() is randomized per process
        # (PYTHONHASHSEED), which made --seed alone non-reproducible.
        task_digest = int.from_bytes(
            hashlib.blake2b(task_id.encode(), digest_size=4).digest(), "big"
        )
        rng = np.random.default_rng([seed, task_digest])
        val_groups = set(_select_val_groups(strata, group_sizes, target, rng))
        val_paths.update(
            task_df.filter(pl.col("group_id").is_in(val_groups))["image_path"].to_list()
        )

        n_val_img = sum(group_sizes[g] for g in val_groups)
        covered = {strata[g] for g in val_groups}
        all_strata = set(strata.values())
        report["tasks"][task_id] = {
            "target_val_images": target,
            "actual_val_images": n_val_img,
            "actual_train_images": task_df.height - n_val_img,
            "n_groups_total": len(group_sizes),
            "n_val_groups": len(val_groups),
            "n_strata": len(all_strata),
            "n_strata_covered": len(covered),
            "uncovered_strata": sorted(str(s) for s in all_strata - covered),
        }
        logger.info(
            "  %-12s val %d/%d img (%d groups), strata %d/%d covered",
            task_id,
            n_val_img,
            target,
            len(val_groups),
            len(covered),
            len(all_strata),
        )

    # In-place rewrite of the split column — same rows, same order.
    is_labeled = pl.col("split").is_in(LABELED_SPLITS)
    result = df.with_columns(
        pl.when(is_labeled & pl.col("image_path").is_in(val_paths))
        .then(pl.lit("val_local"))
        .when(is_labeled)
        .then(pl.lit("train_local"))
        .otherwise(pl.col("split"))
        .alias("split"),
        pl.lit(split_version).alias("split_version"),
        pl.lit(seed).alias("split_seed"),
        pl.lit("group_stratified_random").alias("selection_method"),
    )

    report["summary"] = {
        "n_train_local": result.filter(pl.col("split") == "train_local").height,
        "n_val_local": result.filter(pl.col("split") == "val_local").height,
        "n_other": result.filter(~pl.col("split").is_in(("train_local", "val_local"))).height,
    }
    return result, report


# ---------------------------------------------------------------------------
# DINOv2 diversity split (stress set — diagnostic only)
# ---------------------------------------------------------------------------


def create_diversity_val_split(
    df: pl.DataFrame,
    data_root: Path = DATA_ROOT_DEFAULT,
    val_fraction: float = 0.15,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Split train_labeled into train_local / val_local using DINOv2 diversity.

    Args:
        df: Manifest with columns [split, task_id, group_id, image_path, ...].
        data_root: Root directory for resolving image paths.
        val_fraction: Fraction of groups assigned to val per task.
        seed: Random seed for CoreSet tie-breaking.
        device: Torch device for DINOv2 inference.

    Returns:
        (updated_df, report_dict) — df with split column updated,
        report with full selection rationale.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled = df.filter(pl.col("split") == "train_labeled")
    rest = df.filter(pl.col("split") != "train_labeled")

    if labeled.height == 0:
        logger.warning("No train_labeled rows to split")
        return df, {"error": "no train_labeled rows"}

    # Extract features for ALL labeled images at once (one DINOv2 pass)
    logger.info("Extracting DINOv2 features for %d labeled images...", labeled.height)
    t0 = time.time()

    image_paths = [data_root / p for p in labeled["image_path"].to_list()]
    features = extract_features(image_paths, device)

    feat_time = time.time() - t0
    logger.info("Feature extraction: %.1fs (%.0f img/s)", feat_time, labeled.height / feat_time)

    # Add row index for tracking
    labeled_idx = labeled.with_row_index("_row_idx")

    report: dict = {
        "val_fraction": val_fraction,
        "seed": seed,
        "backbone": "dinov2_vitb14",
        "n_labeled_total": labeled.height,
        "feature_dim": int(features.shape[1]),
        "feature_extraction_seconds": round(feat_time, 1),
        "tasks": {},
    }

    assignments: list[pl.DataFrame] = []

    for task_id in sorted(labeled_idx["task_id"].unique().to_list()):
        task_rows = labeled_idx.filter(pl.col("task_id") == task_id)
        task_indices = task_rows["_row_idx"].to_list()
        task_features = features[task_indices]  # (N_task, D)
        task_group_ids = task_rows["group_id"].to_list()

        # Aggregate features at group level
        unique_groups = sorted(set(task_group_ids))
        group_to_idx: dict[str, list[int]] = {}
        for i, gid in enumerate(task_group_ids):
            group_to_idx.setdefault(gid, []).append(i)

        group_features = np.stack(
            [task_features[group_to_idx[g]].mean(axis=0) for g in unique_groups]
        )
        # Re-normalize after averaging
        norms = np.linalg.norm(group_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        group_features = group_features / norms

        n_val = max(1, round(len(unique_groups) * val_fraction))

        # CoreSet selection
        selections = kcenter_greedy(group_features, n_val, seed=seed)
        val_group_indices = {s["index"] for s in selections}
        val_group_ids = {unique_groups[i] for i in val_group_indices}

        # Build selection rationale
        selection_details = []
        for s in selections:
            gid = unique_groups[s["index"]]
            n_images_in_group = len(group_to_idx[gid])
            selection_details.append(
                {
                    "group_id": gid,
                    "selection_order": s["order"],
                    "distance_at_selection": round(s["distance"], 4),
                    "n_images": n_images_in_group,
                }
            )

        # Coverage stats: mean pairwise distance in val vs overall
        # Uses scipy pdist to avoid O(N²×D) intermediate from broadcasting
        if len(val_group_indices) > 1:
            val_feats = group_features[list(val_group_indices)]
            val_mean_dist = float(pdist(val_feats, metric="euclidean").mean())
        else:
            val_mean_dist = 0.0

        if len(unique_groups) > 1:
            all_mean_dist = float(pdist(group_features, metric="euclidean").mean())
        else:
            all_mean_dist = 0.0

        # Assign splits
        task_assigned = task_rows.with_columns(
            pl.when(pl.col("group_id").is_in(val_group_ids))
            .then(pl.lit("val_local"))
            .otherwise(pl.lit("train_local"))
            .alias("split")
        ).drop("_row_idx")

        n_val_images = task_assigned.filter(pl.col("split") == "val_local").height
        n_train_images = task_assigned.filter(pl.col("split") == "train_local").height
        assignments.append(task_assigned)

        task_report = {
            "n_images": len(task_indices),
            "n_groups": len(unique_groups),
            "n_val_groups": n_val,
            "n_val_images": n_val_images,
            "n_train_groups": len(unique_groups) - n_val,
            "n_train_images": n_train_images,
            "coverage": {
                "val_mean_pairwise_distance": round(val_mean_dist, 4),
                "all_mean_pairwise_distance": round(all_mean_dist, 4),
                "diversity_ratio": round(val_mean_dist / max(all_mean_dist, 1e-8), 3),
            },
            "val_selections": selection_details,
        }
        report["tasks"][task_id] = task_report

        logger.info(
            "  %s: %d groups → train=%d (%d img), val=%d (%d img) | "
            "val diversity=%.3f (all=%.3f, ratio=%.2f)",
            task_id,
            len(unique_groups),
            len(unique_groups) - n_val,
            n_train_images,
            n_val,
            n_val_images,
            val_mean_dist,
            all_mean_dist,
            val_mean_dist / max(all_mean_dist, 1e-8),
        )

    split_labeled = pl.concat(assignments)
    result = pl.concat([split_labeled, rest])

    n_train = result.filter(pl.col("split") == "train_local").height
    n_val = result.filter(pl.col("split") == "val_local").height
    report["summary"] = {
        "n_train_local": n_train,
        "n_val_local": n_val,
        "n_other": rest.height,
    }

    logger.info(
        "Split complete: train_local=%d, val_local=%d, other=%d",
        n_train,
        n_val,
        rest.height,
    )

    return result, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)

STRATIFIED_REPORT_DEFAULT = Path("data/split_report_stratified.json")


def _audit_no_group_leakage(df: pl.DataFrame) -> list[str]:
    """Return (task_id, group_id) pairs appearing in both train_local and val_local.

    Keyed by task because group_id is NOT globally unique: extract_group_id
    falls back to the filename stem for AOP/FA/FUGC/HC, and those datasets use
    plain numeric names, so "0005" exists in several tasks at once.
    """

    def keys(split: str) -> set[tuple[str, str]]:
        rows = df.filter(pl.col("split") == split).select("task_id", "group_id")
        return set(rows.iter_rows())

    return sorted(f"{t}/{g}" for t, g in keys("train_local") & keys("val_local"))


def _write_manifest(df: pl.DataFrame, manifest: Path) -> None:
    """Overwrite the manifest, keeping the previous version recoverable.

    The split is not reconstructible from the manifest once overwritten, and
    an unrecoverable split makes every past run un-reproducible.
    """
    if manifest.exists():
        backup = manifest.with_suffix(f".parquet.bak-{int(time.time())}")
        backup.write_bytes(manifest.read_bytes())
        logger.info("Previous manifest backed up: %s", backup)
    df.write_parquet(manifest)
    logger.info("Manifest updated: %s", manifest)


@app.command()
def stratified(
    manifest: Annotated[Path, typer.Option(help="Manifest parquet path")] = MANIFEST_DEFAULT,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    n_bins: Annotated[int, typer.Option(help="Quantile bins per stratum variable")] = 2,
    report_path: Annotated[Path, typer.Option(help="Audit output")] = STRATIFIED_REPORT_DEFAULT,
    dry_run: Annotated[bool, typer.Option(help="Report only, do not write manifest")] = False,
) -> None:
    """Group-stratified random val split (default; no GPU, no image reads)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pl.read_parquet(manifest)
    result, report = create_stratified_val_split(df, seed=seed, n_bins=n_bins)

    leaked = _audit_no_group_leakage(result)
    report["group_leakage"] = leaked
    if leaked:
        raise RuntimeError(f"Group leakage between train_local/val_local: {leaked[:10]}")

    logger.info("Summary: %s", report["summary"])

    if dry_run:
        logger.info("Dry run — manifest not written")
    else:
        _write_manifest(result, manifest)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Split audit: %s", report_path)


@app.command()
def diversity(
    manifest: Annotated[Path, typer.Option(help="Manifest parquet path")] = MANIFEST_DEFAULT,
    data_root: Annotated[Path, typer.Option(help="Root data directory")] = DATA_ROOT_DEFAULT,
    val_fraction: Annotated[float, typer.Option(help="Fraction of groups for val")] = 0.15,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    report_path: Annotated[Path, typer.Option(help="Split report output")] = REPORT_DEFAULT,
) -> None:
    """DINOv2 CoreSet stress split. Diagnostic only — never select models on it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pl.read_parquet(manifest)
    result, report = create_diversity_val_split(
        df,
        data_root=data_root,
        val_fraction=val_fraction,
        seed=seed,
    )

    _write_manifest(result, manifest)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Split report: %s", report_path)


if __name__ == "__main__":
    app()

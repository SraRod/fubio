"""Build unified manifest.parquet from 9 heterogeneous CSV files.

Each manifest record represents ONE original image with exactly one instance
of landmark annotations. Mosaic composition (multi-instance) happens at
Dataset runtime, not here.

Upstream: task_registry.py (n_keypoints, task_id validation).
Downstream: split.py (consumes group_id), manifest.py (Dataset loading).

Usage:
    uv run python -m fubio.data.build_manifest [--data-root data/] [--output data/manifest.parquet]
"""

from __future__ import annotations

import ast
import contextlib
import csv
import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from fubio.data.landmark_ordering import canonicalize
from fubio.data.task_registry import TASKS

logger = logging.getLogger(__name__)

DATA_ROOT_DEFAULT = Path("data")
OUTPUT_DEFAULT = Path("data/manifest.parquet")

# CSV-specific parsing config. n_keypoints comes from task_registry (single
# source of truth); everything else here is organizer CSV format details
# that don't belong in the registry.
_CSV_CONFIGS: dict[str, dict] = {
    "A4C": {
        "csv": "Data/A4C/labeled/A4C_train.csv",
        "encoding": "gbk",
        "measurement_cols": [],
        "image_path_fix": None,
    },
    "AOP": {
        "csv": "Data/AOP/labeled/key_points_xy.csv",
        "encoding": "utf-8",
        "measurement_cols": ["AOP", "HSD"],
        "image_path_fix": None,
    },
    "FA": {
        "csv": "Data/FA/labeled/FA_train_new.csv",
        "encoding": "utf-8",
        "measurement_cols": ["FA"],
        "image_path_fix": None,
    },
    "FUGC": {
        "csv": "Data/FUGC/labeled/FUGC.csv",
        "encoding": "utf-8",
        "measurement_cols": [],
        "image_path_fix": None,
    },
    "HC": {
        "csv": "Data/HC/labeled/HC.csv",
        "encoding": "utf-8",
        "measurement_cols": [],
        "image_path_fix": "strip_images_prefix",
    },
    "IVC": {
        "csv": "Data/IVC/labeled/train.csv",
        "encoding": "utf-8",
        "measurement_cols": ["IVC"],
        "image_path_fix": None,
    },
    "PLAX": {
        "csv": "Data/PLAX/labeled/train.csv",
        "encoding": "utf-8",
        "measurement_cols": [],
        "image_path_fix": None,
    },
    "PSAX": {
        "csv": "Data/PSAX/labeled/train.csv",
        "encoding": "utf-8",
        "measurement_cols": ["RVOT", "PA"],
        "image_path_fix": None,
    },
    "fetal_femur": {
        "csv": "Data/fetal_femur/labeled/Reg-Two_3.fetal_femur.csv",
        "encoding": "utf-8",
        "measurement_cols": [],
        "image_path_fix": None,
    },
}


def extract_group_id(image_path: str, task_id: str) -> str:
    """Extract patient/study grouping from image path for split isolation.

    Same group_id → same split, preventing data leakage from multiple
    frames of the same study appearing in both train and val.

    Cardiac tasks (A4C, PLAX, PSAX, IVC) use study-level grouping via
    _frame / - suffixes. fetal_femur uses _Plane patient prefix.
    Other tasks have no patient signal in filenames — each file is its
    own group (documented limitation; perceptual hash audit optional).
    """
    fname = Path(image_path).stem

    if task_id in ("A4C", "PLAX", "PSAX", "IVC"):
        if "_frame" in fname:
            return fname.rsplit("_frame", 1)[0]
        if "-" in fname:
            return fname.rsplit("-", 1)[0]
        # Some cardiac images have no frame/dash suffix (standalone shots)
        return fname

    if task_id == "fetal_femur":
        if "_Plane" in fname:
            return fname.split("_Plane")[0]
        # Val/test images use numeric filenames without _Plane pattern
        return fname

    # AOP, FA, FUGC, HC: no patient ID signal in filename
    return fname


def _parse_coord(raw: str) -> list[float]:
    """Parse '[x, y]' string → [x, y] floats."""
    try:
        val = ast.literal_eval(raw)
        return [float(val[0]), float(val[1])]
    except (ValueError, SyntaxError, IndexError):
        return [float("nan"), float("nan")]


def _compute_roi_bbox(keypoints: list[list[float]]) -> list[float] | None:
    """Compute tight bounding box from finite keypoint coordinates."""
    xs = [p[0] for p in keypoints if p[0] == p[0]]  # noqa: PLR0124 — NaN != NaN
    ys = [p[1] for p in keypoints if p[1] == p[1]]  # noqa: PLR0124
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _fix_image_path(raw_path: str, task_id: str, fix_type: str | None) -> str:
    """Normalize image_path to images/{task_id}/{filename}."""
    filename = raw_path.split("/")[-1]
    return f"images/{task_id}/{filename}"


def _parse_labeled_csv(
    data_root: Path,
    task_id: str,
    config: dict,
) -> list[dict]:
    """Parse one task's labeled CSV into manifest records."""
    csv_path = data_root / config["csv"]
    n_keypoints = TASKS[task_id].n_keypoints

    with open(csv_path, encoding=config["encoding"]) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        logger.warning("Empty CSV: %s", csv_path)
        return []

    ordering_stats = {"changed": 0, "ambiguous": 0}

    task_def = TASKS[task_id]
    expected_landmark_cols = task_def.csv_landmark_columns
    measurement_col_names = config["measurement_cols"]

    col_index: dict[str, int] = {}
    for i, h in enumerate(header):
        col_index[h.strip()] = i

    # --- Name-based landmark column resolution ---
    point_positions: list[int] = []
    for col_name in expected_landmark_cols:
        if col_name not in col_index:
            raise ValueError(
                f"{task_id}: expected landmark column {col_name!r} not found "
                f"in {csv_path}. Available: {list(col_index.keys())}"
            )
        point_positions.append(col_index[col_name])

    # Warn on unexpected _xy columns not in schema
    known_xy = set(expected_landmark_cols)
    for h in col_index:
        if h.endswith("_xy") and h not in known_xy:
            logger.warning(
                "%s: unexpected landmark column %r in %s (not in schema)",
                task_id,
                h,
                csv_path,
            )

    measurement_indices = {mc: col_index[mc] for mc in measurement_col_names if mc in col_index}
    height_idx = col_index.get("height")
    width_idx = col_index.get("width")

    records = []
    for cols in rows:
        if len(cols) < len(header):
            continue

        raw_path = cols[0]
        image_path = _fix_image_path(raw_path, task_id, config["image_path_fix"])

        actual_file = data_root / image_path
        if not actual_file.exists():
            logger.warning("Image not found: %s (from CSV: %s)", actual_file, raw_path)
            continue

        keypoints_raw = [_parse_coord(cols[i]) for i in point_positions]

        # Canonicalize landmark ordering (e.g. enforce left-first for A4C 左右 pairs)
        pts_array = np.array(keypoints_raw, dtype=np.float64)
        ordering_result = canonicalize(pts_array, task_id)
        keypoints = ordering_result.points.tolist()
        if ordering_result.changed:
            ordering_stats["changed"] += 1
        if ordering_result.status == "ambiguous":
            ordering_stats["ambiguous"] += 1

        roi_bbox = _compute_roi_bbox(keypoints)

        measurements: dict[str, float] | None = None
        if measurement_indices:
            measurements = {}
            for mc, idx in measurement_indices.items():
                with contextlib.suppress(ValueError, IndexError):
                    measurements[mc] = float(cols[idx])
            if not measurements:
                measurements = None

        try:
            height = int(cols[height_idx]) if height_idx is not None else 0
            width = int(cols[width_idx]) if width_idx is not None else 0
        except (ValueError, IndexError):
            height, width = 0, 0

        group_id = extract_group_id(image_path, task_id)

        records.append(
            {
                "image_path": image_path,
                "height": height,
                "width": width,
                "split": "train_labeled",
                "task_id": task_id,
                "n_keypoints": n_keypoints,
                "keypoints": json.dumps(keypoints),
                "measurements": json.dumps(measurements) if measurements else None,
                "roi_bbox": json.dumps(roi_bbox) if roi_bbox else None,
                "group_id": group_id,
            }
        )

    if ordering_stats["changed"] or ordering_stats["ambiguous"]:
        logger.info(
            "  %s ordering: %d/%d changed, %d ambiguous",
            task_id,
            ordering_stats["changed"],
            len(records),
            ordering_stats["ambiguous"],
        )

    return records


def _collect_unlabeled(data_root: Path) -> list[dict]:
    """Collect unlabeled images from data/images/{domain}/ not in labeled CSVs."""
    labeled_paths: set[str] = set()

    for task_id, config in _CSV_CONFIGS.items():
        csv_path = data_root / config["csv"]
        try:
            with open(csv_path, encoding=config["encoding"]) as f:
                lines = f.readlines()
            for line in lines[1:]:
                raw_path = line.strip().split(",")[0]
                filename = raw_path.split("/")[-1]
                labeled_paths.add(f"images/{task_id}/{filename}")
        except FileNotFoundError:
            pass

    records = []
    images_root = data_root / "images"
    for domain_dir in sorted(images_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        task_id = domain_dir.name
        if task_id not in TASKS:
            logger.warning("Unknown domain directory: %s", domain_dir)
            continue

        for img_file in sorted(domain_dir.iterdir()):
            rel_path = f"images/{task_id}/{img_file.name}"
            if rel_path in labeled_paths:
                continue

            group_id = extract_group_id(rel_path, task_id)

            records.append(
                {
                    "image_path": rel_path,
                    "height": 0,
                    "width": 0,
                    "split": "train_unlabeled",
                    "task_id": task_id,
                    "n_keypoints": TASKS[task_id].n_keypoints,
                    "keypoints": None,
                    "measurements": None,
                    "roi_bbox": None,
                    "group_id": group_id,
                }
            )

    return records


def _collect_val(data_root: Path) -> list[dict]:
    """Collect validation images from data/val/{domain}/."""
    records = []
    val_root = data_root / "val"
    if not val_root.exists():
        return records

    for domain_dir in sorted(val_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        task_id = domain_dir.name
        if task_id not in TASKS:
            logger.warning("Unknown val domain: %s", domain_dir)
            continue

        for img_file in sorted(domain_dir.iterdir()):
            rel_path = f"val/{task_id}/{img_file.name}"
            group_id = extract_group_id(rel_path, task_id)

            records.append(
                {
                    "image_path": rel_path,
                    "height": 0,
                    "width": 0,
                    "split": "val",
                    "task_id": task_id,
                    "n_keypoints": TASKS[task_id].n_keypoints,
                    "keypoints": None,
                    "measurements": None,
                    "roi_bbox": None,
                    "group_id": group_id,
                }
            )

    return records


def _validate_no_duplicates(records: list[dict]) -> None:
    """Assert no duplicate image_path within the same task_id.

    If violated, the I=1 assumption breaks and the pipeline would
    silently drop duplicate annotations.
    """
    seen: dict[tuple[str, str], int] = {}
    duplicates: list[str] = []
    for r in records:
        key = (r["task_id"], r["image_path"])
        if key in seen:
            duplicates.append(f"{r['task_id']}:{r['image_path']}")
        seen[key] = seen.get(key, 0) + 1

    if duplicates:
        raise ValueError(
            f"Duplicate image_path within task: {duplicates[:10]} ({len(duplicates)} total)"
        )


def _load_exclude_paths(flags_path: Path) -> set[str]:
    """Load image_path keys from quality_flags.json to exclude."""
    if not flags_path.exists():
        return set()
    with open(flags_path) as f:
        flags = json.load(f)
    return set(flags.keys())


def build_manifest(
    data_root: Path = DATA_ROOT_DEFAULT,
    output: Path = OUTPUT_DEFAULT,
    exclude_flags: Path | None = None,
) -> pl.DataFrame:
    """Build the unified manifest and write to parquet."""
    all_records: list[dict] = []

    logger.info("Parsing labeled CSVs...")
    for task_id, config in _CSV_CONFIGS.items():
        records = _parse_labeled_csv(data_root, task_id, config)
        logger.info("  %s: %d labeled records", task_id, len(records))
        all_records.extend(records)

    logger.info("Collecting unlabeled images...")
    unlabeled = _collect_unlabeled(data_root)
    logger.info("  %d unlabeled records", len(unlabeled))
    all_records.extend(unlabeled)

    logger.info("Collecting validation images...")
    val = _collect_val(data_root)
    logger.info("  %d val records", len(val))
    all_records.extend(val)

    _validate_no_duplicates(all_records)

    # Exclude flagged samples (quality_flags.json)
    _flags_file = exclude_flags or (data_root / "quality_flags.json")
    _exclude = _load_exclude_paths(_flags_file)
    if _exclude:
        _before = len(all_records)
        all_records = [r for r in all_records if r["image_path"] not in _exclude]
        _dropped = _before - len(all_records)
        logger.info("Excluded %d flagged samples (%s)", _dropped, _flags_file)

    df = pl.DataFrame(
        all_records,
        schema={
            "image_path": pl.Utf8,
            "height": pl.Int32,
            "width": pl.Int32,
            "split": pl.Utf8,
            "task_id": pl.Utf8,
            "n_keypoints": pl.Int32,
            "keypoints": pl.Utf8,
            "measurements": pl.Utf8,
            "roi_bbox": pl.Utf8,
            "group_id": pl.Utf8,
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    logger.info("Manifest written to %s (%d records)", output, len(df))

    summary = df.group_by("split", "task_id").agg(pl.len().alias("count")).sort("split", "task_id")
    logger.info("Summary:\n%s", summary)

    return df


app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_root: Annotated[Path, typer.Option(help="Root data directory")] = DATA_ROOT_DEFAULT,
    output: Annotated[Path, typer.Option(help="Output parquet path")] = OUTPUT_DEFAULT,
    exclude_flags: Annotated[
        Path | None,
        typer.Option(help="quality_flags.json path (default: data_root/)"),
    ] = None,
) -> None:
    """Build unified manifest.parquet from raw CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_manifest(data_root, output, exclude_flags=exclude_flags)


if __name__ == "__main__":
    app()

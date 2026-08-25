"""Pre-decode + pre-resize all images into numpy memmap cache.

Eliminates cv2.imread decode cost (~4 ms/image) from the training loop.
Images are letterbox-resized to target_size and stored as uint8 along with
pre-computed canonical keypoints, masks, and transform matrices.

Usage:
    uv run python -m fubio.data.build_cache --target-size 224
    uv run python -m fubio.data.build_cache --target-size 518 --workers 16

Upstream: manifest.py (ManifestDataset), spatial.py (build_affine_matrix, apply_spatial).
Downstream: manifest.py (CachedManifestDataset reads these files).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from fubio.data.spatial import SpatialParams, apply_spatial, build_affine_matrix
from fubio.data.task_registry import TASKS, K, landmark_valid_mask, local_to_canonical

logger = logging.getLogger(__name__)


def _manifest_hash(df: pl.DataFrame) -> str:
    """Content hash of what the cache actually stores, for invalidation.

    Covers image_path (IN ORDER — CachedManifestDataset indexes the memmap by
    manifest row position, so a reordered manifest would pair images with the
    wrong keypoints), task_id, and keypoints.

    Deliberately EXCLUDES `split`: the cache holds images/keypoints/transforms/
    masks, none of which depend on which split a row lands in — split only
    selects which rows a dataset reads. Including it forced a full 28GB rebuild
    on every re-split, which is pure waste and creates pressure to skip the
    integrity check entirely.
    """
    raw = df.select("image_path", "task_id", "keypoints").to_pandas().to_csv(index=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_dir_name(target_w: int, target_h: int, resize_mode: str) -> str:
    return f"{target_w}x{target_h}_{resize_mode}"


def _process_row(
    args: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Process a single manifest row: load → resize → keypoints → masks.

    Returns (image, keypoints, masks_stacked, transform_matrix, meta), or None if image missing.
    """
    row, data_root, target_w, target_h, resize_mode = args
    task_id: str = row["task_id"]
    task_def = TASKS[task_id]
    image_path = Path(data_root) / row["image_path"]

    # Load + decode
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    src_h, src_w = image.shape[:2]

    # Letterbox resize to target
    params = SpatialParams(target_size=(target_w, target_h))
    M = build_affine_matrix(
        src_size=(src_w, src_h),
        params=params,
        resize_mode=resize_mode,
    )

    # Keypoints
    kp_json: str | None = row["keypoints"]
    is_labeled = kp_json is not None
    if is_labeled:
        local_kp = np.array(json.loads(kp_json), dtype=np.float32)
        canonical_kp = local_to_canonical(local_kp, task_id)
    else:
        canonical_kp = np.full((K, 2), np.nan, dtype=np.float32)

    kp_batch = canonical_kp[np.newaxis]  # [1, 60, 2]

    # Apply spatial to image + keypoints
    image_out, kp_out, _visible = apply_spatial(
        image,
        kp_batch,
        M,
        (target_w, target_h),
    )

    kp_out_2d = kp_out[0]  # [60, 2]

    # Masks
    valid = landmark_valid_mask(task_id)  # [60] bool
    finite = ~np.isnan(kp_out_2d).any(axis=1)  # [60] bool
    supervised = valid & finite
    visible = supervised.copy()

    masks = np.stack([valid, supervised, visible])  # [3, 60] bool

    M_f32 = M.astype(np.float32)  # [3, 3]

    meta = np.array(
        (task_def.task_int, is_labeled, src_h, src_w),
        dtype=[("task_int", "i8"), ("is_labeled", "?"), ("orig_h", "i4"), ("orig_w", "i4")],
    )

    return image_out, kp_out_2d, masks, M_f32, meta


def build_cache(
    manifest_path: Path = Path("data/manifest.parquet"),
    data_root: Path = Path("data"),
    target_size: int | tuple[int, int] = 224,
    resize_mode: str = "letterbox",
    num_workers: int = 10,
) -> Path:
    """Build numpy memmap cache for all manifest samples.

    Returns the cache directory path.
    """
    if isinstance(target_size, int):
        target_w = target_h = target_size
    else:
        target_w, target_h = target_size
    df = pl.read_parquet(manifest_path)
    n = len(df)

    cache_dir = data_root / "cache" / _cache_dir_name(target_w, target_h, resize_mode)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building cache: %d samples → %s (%dx%d %s)",
        n,
        cache_dir,
        target_w,
        target_h,
        resize_mode,
    )

    # Prepare worker args
    rows = [df.row(i, named=True) for i in range(n)]
    worker_args = [(row, str(data_root), target_w, target_h, resize_mode) for row in rows]

    # Allocate memmap for images
    images_path = cache_dir / "images.dat"
    images_mm = np.memmap(
        images_path,
        dtype=np.uint8,
        mode="w+",
        shape=(n, target_h, target_w, 3),
    )

    # In-memory arrays for small data
    keypoints = np.zeros((n, K, 2), dtype=np.float32)
    masks = np.zeros((n, 3, K), dtype=bool)
    transforms = np.zeros((n, 3, 3), dtype=np.float32)
    meta_dtype = [("task_int", "i8"), ("is_labeled", "?"), ("orig_h", "i4"), ("orig_w", "i4")]
    meta = np.zeros(n, dtype=meta_dtype)

    t0 = time.perf_counter()
    missing: list[str] = []

    results_iter: iter
    if num_workers <= 1:
        results_iter = (_process_row(a) for a in worker_args)
    else:
        pool = Pool(num_workers)
        results_iter = pool.imap(_process_row, worker_args, chunksize=256)

    try:
        for i, result in enumerate(results_iter):
            if result is None:
                missing.append(rows[i]["image_path"])
            else:
                img, kp, msk, M, mt = result
                images_mm[i] = img
                keypoints[i] = kp
                masks[i] = msk
                transforms[i] = M
                meta[i] = mt
            if (i + 1) % 5000 == 0:
                elapsed = time.perf_counter() - t0
                logger.info("  %d/%d (%.0f samples/s)", i + 1, n, (i + 1) / elapsed)
    finally:
        if num_workers > 1:
            pool.close()
            pool.join()

    if missing:
        logger.warning("%d images missing (black placeholder in cache):", len(missing))
        for p in missing[:20]:
            logger.warning("  %s", p)
        if len(missing) > 20:
            logger.warning("  ... and %d more", len(missing) - 20)

    # Flush memmap
    images_mm.flush()
    del images_mm

    # Save small arrays
    np.save(cache_dir / "keypoints.npy", keypoints)
    np.save(cache_dir / "masks.npy", masks)
    np.save(cache_dir / "transforms.npy", transforms)
    np.save(cache_dir / "meta.npy", meta)

    # Image paths
    image_paths = df["image_path"].to_list()
    with open(cache_dir / "image_paths.json", "w") as f:
        json.dump(image_paths, f)

    # Cache info for validation
    info = {
        "manifest_hash": _manifest_hash(df),
        "target_size": [target_w, target_h],
        "resize_mode": resize_mode,
        "n_samples": n,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(cache_dir / "cache_info.json", "w") as f:
        json.dump(info, f, indent=2)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Cache built: %d samples in %.1fs (%.0f samples/s) → %s",
        n,
        elapsed,
        n / elapsed,
        cache_dir,
    )

    return cache_dir


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def _parse_size(s: str) -> int | tuple[int, int]:
        if "x" in s:
            w, h = s.split("x", 1)
            return (int(w), int(h))
        return int(s)

    parser = argparse.ArgumentParser(description="Build image cache for FU_Biometry")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.parquet"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--target-size", type=_parse_size, default=224,
        help="Square int (e.g. 224) or WxH (e.g. 728x546)",
    )
    parser.add_argument("--resize-mode", default="letterbox")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    build_cache(
        manifest_path=args.manifest,
        data_root=args.data_root,
        target_size=args.target_size,
        resize_mode=args.resize_mode,
        num_workers=args.workers,
    )

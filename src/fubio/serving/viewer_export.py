"""Generate all-instance viewer cache from a trained checkpoint.

Discovers val_local (with GT), val (competition, no GT), and quality-flagged
images, runs inference retaining all instance slots, and writes a JSON cache
that the marimo instance viewer loads directly.

Upstream: predict.py (model loading, image prep, discovery), evaluation/postprocessing.py.
Downstream: notebooks/instance_viewer.py (pure loader).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fubio.data.task_registry import TASKS
from fubio.evaluation.postprocessing import inverse_transform_landmarks
from fubio.serving.predict import (
    discover_competition_images,
    discover_from_manifest,
    load_module,
    prepare_image,
)
from fubio.train.config import ExperimentConfig
from fubio.train.module import FUBioModule

logger = logging.getLogger(__name__)


def _discover_flagged(data_root: Path) -> list[dict]:
    """Load quality-flagged images from data/quality_flags.json."""
    flags_path = data_root / "quality_flags.json"
    if not flags_path.exists():
        return []

    with open(flags_path) as f:
        flags_data = json.load(f)

    samples: list[dict] = []
    for fp, finfo in flags_data.items():
        if not (data_root / fp).exists():
            continue
        parts = fp.split("/")
        if len(parts) >= 2 and parts[1] in TASKS:
            samples.append(
                {
                    "image_path": fp,
                    "submission_path": f"{parts[1]}/{(data_root / fp).name}",
                    "task_id": parts[1],
                    "gt_pixels_flat": None,
                    "split": "flagged",
                    "flagged": True,
                    "flag_type": finfo.get("type", ""),
                }
            )
    return samples


def generate_viewer_cache(
    ckpt_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    batch_size: int = 16,
    device: str | None = None,
    *,
    module: FUBioModule | None = None,
    config: ExperimentConfig | None = None,
) -> dict:
    """Run all-instance inference on val_local + val + flagged and write viewer cache.

    Args:
        module/config: Pre-loaded model to reuse. If None, loads from ckpt_path.

    Returns:
        Metadata dict with checkpoint, input_size, n_predictions, elapsed_seconds.
    """
    ckpt_path = Path(ckpt_path)
    data_root = Path(data_root)
    output_path = Path(output_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    if module is None or config is None:
        logger.info("Loading checkpoint: %s", ckpt_path)
        module, config = load_module(str(ckpt_path), device)

    input_size = config.backbone.input_size
    dev = next(module.parameters()).device

    # --- Discover all splits ---
    manifest_path = data_root / "manifest.parquet"

    val_local = discover_from_manifest(data_root, manifest_path, "val_local")
    for s in val_local:
        s["split"] = "val_local"
        s["flagged"] = False

    val_comp = discover_competition_images(data_root, "val", allow_missing_tasks=True)
    for s in val_comp:
        s["split"] = "val"
        s["flagged"] = False

    flagged = _discover_flagged(data_root)

    all_samples = val_local + val_comp + flagged
    logger.info(
        "Discovered %d samples (val_local=%d, val=%d, flagged=%d)",
        len(all_samples),
        len(val_local),
        len(val_comp),
        len(flagged),
    )

    # --- Batched inference — all instances ---
    results: list[dict] = []
    n_inst = 4
    for start in tqdm(range(0, len(all_samples), batch_size), desc="Viewer cache"):
        chunk = all_samples[start : start + batch_size]
        imgs: list[np.ndarray] = []
        metas: list[dict] = []
        for s in chunk:
            bgr = cv2.imread(str(data_root / s["image_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("Cannot read %s, skipping", s["image_path"])
                continue
            chw, M, ohw = prepare_image(bgr, input_size)
            imgs.append(chw)
            metas.append({**s, "M": M, "orig_hw": ohw})
        if not imgs:
            continue

        with torch.no_grad():
            tensor = torch.from_numpy(np.stack(imgs)).to(dev)
            out = module.model(module._normalize_image(tensor))

        for b, meta in enumerate(metas):
            tid = meta["task_id"]
            to = out.task_outputs[tid]
            n_inst = to.conf.shape[1]
            instances = []
            for ii in range(n_inst):
                c = float(to.conf[b, ii].sigmoid().item())
                _canvas_wh = torch.tensor(input_size, dtype=torch.float32)
                lm = to.landmarks[b, ii].detach().cpu().float() * _canvas_wh
                lm_orig = inverse_transform_landmarks(lm, meta["M"], meta["orig_hw"])
                instances.append(
                    {
                        "conf": round(c, 4),
                        "landmarks": [
                            [round(float(lm_orig[j, 0]), 2), round(float(lm_orig[j, 1]), 2)]
                            for j in range(lm_orig.shape[0])
                        ],
                    }
                )
            oh, ow = int(meta["orig_hw"][0]), int(meta["orig_hw"][1])
            entry: dict = {
                "image_path": meta["image_path"],
                "task_id": tid,
                "split": meta.get("split", "unknown"),
                "flagged": meta.get("flagged", False),
                "original_hw": [oh, ow],
                "instances": instances,
            }
            if meta.get("gt_pixels_flat") is not None:
                gf = meta["gt_pixels_flat"]
                entry["gt_landmarks"] = [[gf[i], gf[i + 1]] for i in range(0, len(gf), 2)]
            else:
                entry["gt_landmarks"] = None
            if meta.get("flag_type"):
                entry["flag_type"] = meta["flag_type"]
            results.append(entry)

    elapsed = time.perf_counter() - t0
    metadata = {
        "checkpoint": str(ckpt_path),
        "input_size": input_size,
        "n_inst": n_inst,
        "n_predictions": len(results),
        "elapsed_seconds": round(elapsed, 1),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"metadata": metadata, "predictions": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Viewer cache: %s (%d predictions, %.1fs)", output_path, len(results), elapsed)
    return metadata


def main() -> None:
    """CLI entry point for standalone viewer cache generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate viewer cache from checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_viewer_cache(args.ckpt, args.data_root, args.output, args.batch_size, args.device)


if __name__ == "__main__":
    main()

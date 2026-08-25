"""Standalone inference: checkpoint + images → submission JSON.

Three outputs written to output_dir:
  regression_predictions.json — submission-ready (what the evaluator reads)
  landmark_predictions.json   — identical copy; the organizer's docs and their
                                own code disagree on the name, so we write both
  predictions_detail.json     — extended with confidence + GT (for viewer)

Usage:
  # Competition validation (no GT, scans data/val/)
  uv run python -m fubio.serving.predict \\
    --ckpt wandb_logs/fubio/.../last.ckpt \\
    --data-root data --output-dir predictions/r10

  # Training validation split (with GT for analysis)
  uv run python -m fubio.serving.predict \\
    --ckpt wandb_logs/fubio/.../last.ckpt \\
    --data-root data --output-dir predictions/r10 \\
    --mode val_local

Upstream: train/module.py, data/spatial.py, evaluation/postprocessing.py.
Downstream: submission platform, notebooks/submission_viewer.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fubio.data.landmark_ordering import competitionize
from fubio.data.spatial import SpatialParams, build_affine_matrix
from fubio.data.task_registry import TASKS
from fubio.evaluation.postprocessing import inverse_transform_landmarks, select_serving_query
from fubio.serving.validate import (
    SubmissionError,
    _check_normalized_range,
    _check_vector_lengths,
    validate_inference_results,
    validate_submission_document,
)
from fubio.train.config import ExperimentConfig
from fubio.train.views import warp_image, warp_landmarks

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_module(
    ckpt_path: str,
    device: str = "cuda",
) -> tuple[FUBioModule, ExperimentConfig]:
    """Load FUBioModule from checkpoint, reconstructing config from hparams."""
    from fubio.train.module import FUBioModule

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(raw["hyper_parameters"])
    # Patch use_affine + ShapeSimCC incompatibility (pre-validator checkpoints).
    head = hp.get("head", {})
    coord_mode = head.get("coord", {}).get("mode", "")
    if head.get("use_affine") and coord_mode in ("simcc", "shape_simcc"):
        hp["head"] = {**head, "use_affine": False}
        loss = hp.get("loss", {})
        if loss.get("lambda_ortho", 0) > 0:
            hp["loss"] = {**loss, "lambda_ortho": 0.0}
        logger.warning(
            "Patched use_affine=True→False for %s checkpoint "
            "(inference-only, no effect on predictions)",
            coord_mode,
        )
    config = ExperimentConfig(**hp)
    # weights_only=False: checkpoint contains PosixPath in hparams
    module = FUBioModule.load_from_checkpoint(
        ckpt_path,
        config=config,
        map_location=device,
        weights_only=False,
    )
    module.eval()
    module.freeze()
    return module, config


# ---------------------------------------------------------------------------
# Image preprocessing (identical to training val: letterbox only, no aug)
# ---------------------------------------------------------------------------


def prepare_image(
    image_bgr: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Letterbox resize a single image for inference.

    Returns:
        image_chw: (3, H, W) uint8 ready for model._normalize_image
        transform_matrix: (3, 3) float32 original → augmented affine
        original_hw: (2,) int32 [H, W]
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    target_w, target_h = input_size

    params = SpatialParams(target_size=(target_w, target_h))
    M = build_affine_matrix((w, h), params, resize_mode="letterbox")
    resized = cv2.warpAffine(
        image_rgb,
        M[:2].astype(np.float32),
        (target_w, target_h),  # cv2 expects (W, H)
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
    return chw, M.astype(np.float32), np.array([h, w], dtype=np.int32)


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def discover_competition_images(
    data_root: Path,
    subdir: str = "val",
    *,
    allow_missing_tasks: bool = False,
) -> list[dict]:
    """Scan {data_root}/{subdir}/{task_id}/*.png → sample list.

    A missing task directory is an error by default: the platform scores missing
    results as the worst valid submission, so a mistyped data root or an
    incomplete mount must not degrade quietly into a smaller submission.
    """
    samples: list[dict] = []
    base = data_root / subdir
    missing_tasks: list[str] = []
    for task_id in sorted(TASKS):
        task_dir = base / task_id
        if not task_dir.exists():
            missing_tasks.append(task_id)
            continue
        for img_path in sorted(task_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            # Submission image_path must match evaluator CSV convention:
            # "{task_id}/{filename}" (relative to images/, not data_root)
            samples.append(
                {
                    "image_path": str(img_path.relative_to(data_root)),
                    "submission_path": f"{task_id}/{img_path.name}",
                    "task_id": task_id,
                    "gt_pixels_flat": None,
                }
            )
    if missing_tasks and not allow_missing_tasks:
        raise SubmissionError(
            f"No directory for task(s) {missing_tasks} under {base}. "
            f"Pass --allow-missing-tasks only if a partial submission is intended."
        )
    if missing_tasks:
        logger.warning(
            "--allow-missing-tasks: producing a PARTIAL submission, no data for %s",
            missing_tasks,
        )

    logger.info(
        "%d competition images from %s (%d tasks)",
        len(samples),
        base,
        len({s["task_id"] for s in samples}),
    )
    return samples


def discover_from_manifest(
    data_root: Path,
    manifest_path: Path,
    split: str = "val_local",
) -> list[dict]:
    """Read manifest for a given split, include GT pixels if available."""
    import polars as pl

    df = pl.read_parquet(manifest_path)
    df = df.filter(pl.col("split") == split)
    if df.is_empty():
        raise ValueError(
            f"No samples in manifest for split='{split}'. "
            f"Available: {pl.read_parquet(manifest_path)['split'].unique().sort().to_list()}"
        )

    samples: list[dict] = []
    for row in df.iter_rows(named=True):
        gt_flat = None
        kp_json: str | None = row["keypoints"]
        if kp_json is not None:
            local_kp = np.array(json.loads(kp_json), dtype=np.float32)
            gt_flat = local_kp.flatten().tolist()
        _img_path: str = row["image_path"]
        # Manifest paths are like "images/A4C/file.png"; evaluator CSV
        # expects "{task_id}/{filename}" — strip leading "images/" if present
        _sub_path = _img_path.removeprefix("images/")
        samples.append(
            {
                "image_path": _img_path,
                "submission_path": _sub_path,
                "task_id": row["task_id"],
                "gt_pixels_flat": gt_flat,
            }
        )

    n_gt = sum(1 for s in samples if s["gt_pixels_flat"] is not None)
    logger.info(
        "%d samples from manifest split='%s' (%d with GT)",
        len(samples),
        split,
        n_gt,
    )
    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    module: FUBioModule,
    samples: list[dict],
    data_root: Path,
    input_size: tuple[int, int],
    device: torch.device,
    batch_size: int = 16,
    task_routing: str = "directory",
    task_routing_threshold: float = 0.9,
) -> list[dict]:
    """Batched inference → list of prediction dicts.

    Each dict has submission-spec fields (image_path, task_id,
    predicted_points_normalized, predicted_points_pixels) plus
    analysis fields (confidence, original_hw, gt_points_pixels).

    Args:
        task_routing: "directory" uses the filesystem task assignment;
            "high_conf" picks the task head with the highest confidence
            when the directory-assigned head's conf is below threshold.
        task_routing_threshold: only reroute when the assigned task's
            confidence is below this value (default 0.9).
    """
    canvas_wh = torch.tensor(input_size, dtype=torch.float32)  # (2,) [W, H]
    results: list[dict] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        chunk = samples[start : start + batch_size]

        images: list[np.ndarray] = []
        metas: list[dict] = []
        for s in chunk:
            bgr = cv2.imread(str(data_root / s["image_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise SubmissionError(
                    f"Cannot read image: {data_root / s['image_path']}. "
                    f"Refusing to emit a submission that omits it."
                )
            chw, M, orig_hw = prepare_image(bgr, input_size)
            images.append(chw)
            metas.append({**s, "M": M, "orig_hw": orig_hw})

        tensor = torch.from_numpy(np.stack(images)).to(device)
        normed = module._normalize_image(tensor)
        out = module.model(normed)

        # Student+Teacher ensemble: run teacher, will average landmarks below
        ensemble_teacher = getattr(module, "_ensemble_teacher", None)
        out_teacher = ensemble_teacher(normed) if ensemble_teacher is not None else None

        for b, meta in enumerate(metas):
            if task_routing == "high_conf":
                assigned_conf = (
                    out.task_outputs[meta["task_id"]]
                    .conf[b].sigmoid().squeeze(-1).max().item()
                )
                if assigned_conf < task_routing_threshold:
                    best_tid, best_conf_val = "", -1.0
                    for cand_tid, cand_out in out.task_outputs.items():
                        c = cand_out.conf[b].sigmoid().squeeze(-1).max().item()
                        if c > best_conf_val:
                            best_tid, best_conf_val = cand_tid, c
                    logger.info(
                        "Rerouted %s: %s → %s (conf %.4f → %.4f, threshold %.2f)",
                        meta.get("submission_path", meta["image_path"]),
                        meta["task_id"],
                        best_tid,
                        assigned_conf,
                        best_conf_val,
                        task_routing_threshold,
                    )
                    tid = best_tid
                else:
                    tid = meta["task_id"]
            else:
                tid = meta["task_id"]

            task_out = out.task_outputs[tid]

            conf = task_out.conf[b].sigmoid().squeeze(-1)  # (N_inst,)
            best = int(select_serving_query(conf).item())
            best_conf = float(conf[best].item())

            tdef = TASKS[tid]
            lm_norm = task_out.landmarks[b, best, :tdef.n_keypoints].detach().cpu()

            # Ensemble: average student + teacher landmarks
            if out_teacher is not None:
                t_task_out = out_teacher.task_outputs[tid]
                t_conf = t_task_out.conf[b].sigmoid().squeeze(-1)
                t_best = int(select_serving_query(t_conf).item())
                t_lm = t_task_out.landmarks[b, t_best, :tdef.n_keypoints].detach().cpu()
                lm_norm = (lm_norm + t_lm) * 0.5

            lm_aug_px = lm_norm.float() * canvas_wh  # (K, 2) × (2,)

            lm_orig = inverse_transform_landmarks(
                lm_aug_px,
                meta["M"],
                meta["orig_hw"],
            )  # (K_task, 2) original pixel coords

            oh, ow = int(meta["orig_hw"][0]), int(meta["orig_hw"][1])
            lm_orig_normed = lm_orig.copy()
            lm_orig_normed[:, 0] /= max(ow, 1)
            lm_orig_normed[:, 1] /= max(oh, 1)

            entry: dict = {
                "image_path": meta["image_path"],
                "submission_path": meta.get("submission_path", meta["image_path"]),
                "task_id": tid,
                "predicted_points_normalized": [
                    round(float(v), 6) for v in lm_orig_normed.flatten()
                ],
                "predicted_points_pixels": [round(float(v), 2) for v in lm_orig.flatten()],
                "_canonical_points_full": lm_orig,
                "confidence": round(best_conf, 4),
                "original_hw": [oh, ow],
            }
            if meta["gt_pixels_flat"] is not None:
                entry["gt_points_pixels"] = meta["gt_pixels_flat"]

            results.append(entry)

    return results


def parse_tta_color_specs(spec_str: str) -> list[tuple[float, float]]:
    """Parse color TTA spec string → list of (brightness_delta, contrast_factor).

    Format: comma-separated tokens like 'bright+20,bright-20,contr+25,contr-25'.
    bright±N → delta=±N/100, contrast=1.0
    contr±N  → delta=0.0, contrast=1±N/100
    """
    specs: list[tuple[float, float]] = []
    for token in spec_str.split(","):
        token = token.strip()
        if token.startswith("bright"):
            val = float(token[6:]) / 100.0
            specs.append((val, 1.0))
        elif token.startswith("contr"):
            val = float(token[5:]) / 100.0
            specs.append((0.0, 1.0 + val))
        else:
            raise ValueError(f"Unknown color TTA token: {token!r}")
    return specs


def _make_tta_matrix(
    batch_size: int,
    rotation_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a deterministic (B, 2, 3) rotation-only affine in [0,1] space."""
    angle = torch.full((batch_size,), rotation_deg, device=device) * (torch.pi / 180.0)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    matrix = torch.zeros(batch_size, 2, 3, device=device)
    matrix[:, 0, 0] = cos_a
    matrix[:, 0, 1] = -sin_a
    matrix[:, 0, 2] = 0.5 * (1 - cos_a + sin_a)
    matrix[:, 1, 0] = sin_a
    matrix[:, 1, 1] = cos_a
    matrix[:, 1, 2] = 0.5 * (1 - cos_a - sin_a)
    return matrix


def _invert_affine_2x3(matrix: torch.Tensor) -> torch.Tensor:
    """Invert (B, 2, 3) affine matrices → (B, 2, 3)."""
    B = matrix.shape[0]
    pad = torch.tensor([0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype)
    full = torch.cat([matrix, pad.expand(B, 1, 3)], dim=1)
    return torch.linalg.inv(full)[:, :2, :]


_ELLIPSE_TASKS = frozenset({"HC", "FA"})


def _ellipse_consensus_avg(views: torch.Tensor) -> np.ndarray:
    """Average HC/FA TTA views in ellipse parameter space.

    Instead of averaging landmark coordinates (which can break ellipse
    geometry), averages center, semi-axis lengths, and axis direction,
    then reconstructs 4 endpoints guaranteed to form a valid ellipse
    with orthogonal axes and shared center.

    Args:
        views: (N_views, 4, 2) landmarks in pixel space.
            P0-P1 = axis a (short for HC), P2-P3 = axis b (long for HC).

    Returns:
        (4, 2) averaged landmarks on a geometrically consistent ellipse.
    """
    centers = []
    a_lengths = []
    b_lengths = []
    dir_a_vecs = []
    dir_b_vecs = []

    for v in range(views.shape[0]):
        p0, p1, p2, p3 = views[v, 0], views[v, 1], views[v, 2], views[v, 3]

        center = ((p0 + p1) / 2 + (p2 + p3) / 2) / 2
        centers.append(center)

        va = p1 - p0
        a_lengths.append(torch.linalg.norm(va) / 2)
        dir_a = va / (torch.linalg.norm(va) + 1e-8)
        dir_a_vecs.append(dir_a)

        vb = p3 - p2
        b_lengths.append(torch.linalg.norm(vb) / 2)
        dir_b = vb / (torch.linalg.norm(vb) + 1e-8)
        dir_b_vecs.append(dir_b)

    # Align direction vectors to first view (resolve sign ambiguity)
    ref_a = dir_a_vecs[0]
    ref_b = dir_b_vecs[0]
    for i in range(1, len(dir_a_vecs)):
        if torch.dot(dir_a_vecs[i], ref_a) < 0:
            dir_a_vecs[i] = -dir_a_vecs[i]
        if torch.dot(dir_b_vecs[i], ref_b) < 0:
            dir_b_vecs[i] = -dir_b_vecs[i]

    avg_center = torch.stack(centers).mean(dim=0)
    avg_a = torch.stack(a_lengths).mean()
    avg_b = torch.stack(b_lengths).mean()

    # Average axis-a direction, then force axis-b perpendicular
    avg_dir_a = torch.stack(dir_a_vecs).mean(dim=0)
    avg_dir_a = avg_dir_a / (torch.linalg.norm(avg_dir_a) + 1e-8)

    # Pick the perpendicular direction closest to the actual b-axis average
    perp = torch.tensor([-avg_dir_a[1].item(), avg_dir_a[0].item()])
    avg_dir_b_raw = torch.stack(dir_b_vecs).mean(dim=0)
    if torch.dot(avg_dir_b_raw, perp) < 0:
        perp = -perp
    avg_dir_b = perp

    result = torch.stack([
        avg_center - avg_a * avg_dir_a,
        avg_center + avg_a * avg_dir_a,
        avg_center - avg_b * avg_dir_b,
        avg_center + avg_b * avg_dir_b,
    ])
    return result.numpy().astype(np.float32)


@torch.no_grad()
def run_inference_tta(
    module: FUBioModule,
    samples: list[dict],
    data_root: Path,
    input_size: tuple[int, int],
    device: torch.device,
    batch_size: int = 16,
    task_routing: str = "directory",
    task_routing_threshold: float = 0.9,
    tta_angles: list[float] | None = None,
    tta_color: list[tuple[float, float]] | None = None,
    ellipse_consensus: bool = False,
) -> list[dict]:
    """TTA via GPU-space rotation warps + color jitter, averaged in pixel space.

    Pass-0 (first angle) determines task routing and confidence;
    all angles share the same task assignment. Landmarks are averaged
    in original pixel space across all views.

    tta_color: list of (brightness_delta, contrast_factor) pairs.
    Color jitter doesn't change geometry — no inverse warp needed.
    """
    n_views = (len(tta_angles) if tta_angles else 1) + (len(tta_color) if tta_color else 0)
    if n_views <= 1:
        return run_inference(
            module, samples, data_root, input_size, device,
            batch_size, task_routing, task_routing_threshold,
        )

    # Pass 0: normal inference for task routing + base metadata
    logger.info("TTA pass 1/%d: 0° (base)", len(tta_angles))
    base_results = run_inference(
        module, samples, data_root, input_size, device,
        batch_size, task_routing, task_routing_threshold,
    )
    routed_tids = [r["task_id"] for r in base_results]

    # Collect base landmarks in [0,1] canvas-normalized space
    canvas_wh = torch.tensor(input_size, dtype=torch.float32)
    all_lm_norm: list[list[torch.Tensor]] = [[] for _ in range(len(samples))]
    for i, r in enumerate(base_results):
        px = np.array(r["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
        # pixel → canvas-normalized [0,1]: reverse of (lm_norm * canvas_wh → px via M⁻¹)
        # We need the pre-inverse-transform coords. Reconstruct from the model output path:
        # lm_norm → *canvas_wh → inverse_transform → px.
        # Since we only have px, and inverse_transform involves M⁻¹ which is non-trivial,
        # we use the normalized submission coords (px / image_size) as a proxy.
        # BUT: that's NOT the same as model [0,1] because letterboxing shifts coords.
        # Instead, just store px and average in pixel space for base pass.
        all_lm_norm[i].append(torch.from_numpy(px))

    # Additional angle passes: GPU warp on normalized tensor
    non_zero_angles = [a for a in tta_angles if a != 0.0]
    for angle_num, angle in enumerate(non_zero_angles, start=2):
        logger.info("TTA pass %d/%d: %.1f°", angle_num, len(tta_angles), angle)

        angle_px: list[np.ndarray] = []
        for start in tqdm(range(0, len(samples), batch_size), desc=f"TTA {angle:+.0f}°"):
            chunk = samples[start : start + batch_size]
            batch_tids = routed_tids[start : start + batch_size]

            imgs: list[np.ndarray] = []
            metas: list[dict] = []
            for s in chunk:
                bgr = cv2.imread(str(data_root / s["image_path"]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise SubmissionError(f"Cannot read: {data_root / s['image_path']}")
                chw, M, orig_hw = prepare_image(bgr, input_size)
                imgs.append(chw)
                metas.append({"M": M, "orig_hw": orig_hw})

            tensor = torch.from_numpy(np.stack(imgs)).to(device)
            normed = module._normalize_image(tensor)
            B = normed.shape[0]

            tta_mat = _make_tta_matrix(B, angle, device=normed.device)
            img_01 = normed * module._img_std + module._img_mean
            warped = warp_image(img_01, tta_mat)
            model_input = (warped - module._img_mean) / module._img_std
            inv_mat = _invert_affine_2x3(tta_mat)

            out = module.model(model_input)

            for b in range(B):
                tid = batch_tids[b]
                task_out = out.task_outputs[tid]
                conf = task_out.conf[b].sigmoid().squeeze(-1)
                best = int(select_serving_query(conf).item())

                tdef = TASKS[tid]
                lm = task_out.landmarks[b, best, :tdef.n_keypoints].detach()

                # Inverse-warp landmarks from rotated [0,1] → unrotated [0,1]
                lm_back = warp_landmarks(
                    lm.unsqueeze(0).unsqueeze(0), inv_mat[b : b + 1]
                ).squeeze(0).squeeze(0)

                lm_aug_px = lm_back.cpu().float() * canvas_wh
                lm_orig = inverse_transform_landmarks(
                    lm_aug_px, metas[b]["M"], metas[b]["orig_hw"]
                )
                angle_px.append(lm_orig)

        for i, lm in enumerate(angle_px):
            all_lm_norm[i].append(torch.from_numpy(lm))

    # Color jitter passes: no geometric inverse needed
    if tta_color:
        n_geo = len(tta_angles)
        for ci, (bright_delta, contr_factor) in enumerate(tta_color):
            label = f"bright{bright_delta:+.0%}" if bright_delta else f"contr{contr_factor:.2f}"
            logger.info(
                "TTA pass %d/%d: color %s",
                n_geo + ci + 1, n_views, label,
            )

            color_px: list[np.ndarray] = []
            for start in tqdm(
                range(0, len(samples), batch_size), desc=f"TTA {label}"
            ):
                chunk = samples[start : start + batch_size]
                batch_tids = routed_tids[start : start + batch_size]

                imgs: list[np.ndarray] = []
                metas: list[dict] = []
                for s in chunk:
                    bgr = cv2.imread(str(data_root / s["image_path"]), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise SubmissionError(
                            f"Cannot read: {data_root / s['image_path']}"
                        )
                    chw, M, orig_hw = prepare_image(bgr, input_size)
                    imgs.append(chw)
                    metas.append({"M": M, "orig_hw": orig_hw})

                tensor = torch.from_numpy(np.stack(imgs)).to(device)
                normed = module._normalize_image(tensor)

                # Apply color jitter in [0,1] space
                img_01 = normed * module._img_std + module._img_mean
                jittered = (img_01 * contr_factor + bright_delta).clamp(0, 1)
                model_input = (jittered - module._img_mean) / module._img_std

                out = module.model(model_input)

                for b in range(len(chunk)):
                    tid = batch_tids[b]
                    task_out = out.task_outputs[tid]
                    conf = task_out.conf[b].sigmoid().squeeze(-1)
                    best = int(select_serving_query(conf).item())

                    tdef = TASKS[tid]
                    lm = task_out.landmarks[b, best, :tdef.n_keypoints].detach()

                    lm_px = lm.cpu().float() * canvas_wh
                    lm_orig = inverse_transform_landmarks(
                        lm_px, metas[b]["M"], metas[b]["orig_hw"]
                    )
                    color_px.append(lm_orig)

            for i, lm in enumerate(color_px):
                all_lm_norm[i].append(torch.from_numpy(lm))

    # Average in original pixel space across all views
    n_ellipse_consensus = 0
    averaged: list[dict] = []
    for i, r in enumerate(base_results):
        entry = {k: v for k, v in r.items() if k != "_canonical_points_full"}
        stacked = torch.stack(all_lm_norm[i]).float()  # (n_views, K, 2)

        if ellipse_consensus and routed_tids[i] in _ELLIPSE_TASKS and stacked.shape[1] == 4:
            avg_px = _ellipse_consensus_avg(stacked)
            n_ellipse_consensus += 1
        else:
            avg_px = stacked.mean(dim=0).numpy().astype(np.float32)

        oh, ow = entry["original_hw"]
        # Clamp to image bounds (ellipse consensus can push endpoints slightly outside)
        avg_px[:, 0] = np.clip(avg_px[:, 0], 0, ow - 1)
        avg_px[:, 1] = np.clip(avg_px[:, 1], 0, oh - 1)

        avg_norm = avg_px.copy()
        avg_norm[:, 0] /= max(ow, 1)
        avg_norm[:, 1] /= max(oh, 1)

        entry["predicted_points_pixels"] = [round(float(v), 2) for v in avg_px.flatten()]
        entry["predicted_points_normalized"] = [round(float(v), 6) for v in avg_norm.flatten()]
        entry["_canonical_points_full"] = avg_px
        averaged.append(entry)

    logger.info(
        "TTA: averaged %d samples over %d views (angles=%s, color=%d%s)",
        len(averaged), n_views, tta_angles,
        len(tta_color) if tta_color else 0,
        f", ellipse_consensus={n_ellipse_consensus}" if ellipse_consensus else "",
    )
    return averaged


# ---------------------------------------------------------------------------
# Iterative refinement (coarse-to-fine second pass)
# ---------------------------------------------------------------------------


def _compute_roi_pct(landmarks_px: np.ndarray, orig_hw: tuple[int, int]) -> float:
    """ROI area as percentage of image area, from predicted landmarks."""
    oh, ow = orig_hw
    roi_w = float(landmarks_px[:, 0].max() - landmarks_px[:, 0].min())
    roi_h = float(landmarks_px[:, 1].max() - landmarks_px[:, 1].min())
    img_area = oh * ow
    return (roi_w * roi_h) / img_area * 100 if img_area > 0 else 0.0


def _compute_crop_region(
    landmarks_px: np.ndarray,
    orig_hw: tuple[int, int],
    strategy: str,
    task_id: str,
    safety_factor: float,
    crop_ratio: float | dict[str, float] = 0.5,
) -> tuple[int, int, int]:
    """Compute square crop centred on first-pass ROI for refinement.

    crop_ratio: global float or per-task dict (e.g. {"IVC": 0.25, "PSAX": 0.5}).
    Tasks not in the dict use the default 0.5.

    Returns:
        (x1, y1, side) in original pixel coords.
    """
    oh, ow = orig_hw
    short_side = min(oh, ow)

    x_min = float(landmarks_px[:, 0].min())
    x_max = float(landmarks_px[:, 0].max())
    y_min = float(landmarks_px[:, 1].min())
    y_max = float(landmarks_px[:, 1].max())
    roi_cx = (x_min + x_max) / 2
    roi_cy = (y_min + y_max) / 2
    roi_side = max(x_max - x_min, y_max - y_min, 1.0)

    ratio = crop_ratio.get(task_id, 0.5) if isinstance(crop_ratio, dict) else crop_ratio

    if strategy == "half":
        crop_side = max(short_side / 2, roi_side)
    elif strategy == "adaptive":
        scale = TASKS[task_id].bbox_context_scale
        crop_side = roi_side * scale * safety_factor
        crop_side = max(crop_side, roi_side)
    elif strategy == "fixed":
        crop_side = max(short_side * ratio, roi_side)
    else:
        raise ValueError(f"Unknown refine strategy: {strategy!r}")

    crop_side = min(crop_side, short_side)
    crop_side = max(int(round(crop_side)), 1)

    x1 = int(round(roi_cx - crop_side / 2))
    y1 = int(round(roi_cy - crop_side / 2))
    x1 = max(0, min(x1, ow - crop_side))
    y1 = max(0, min(y1, oh - crop_side))

    return x1, y1, crop_side


@torch.no_grad()
def run_refinement(
    module: FUBioModule,
    first_pass: list[dict],
    data_root: Path,
    input_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
    strategy: str,
    safety_factor: float = 2.0,
    crop_ratio: float | dict[str, float] = 0.5,
    roi_threshold: float = 100.0,
    refine_tasks: set[str] | None = None,
) -> list[dict]:
    """Second-pass inference on zoomed crops from original images.

    Crops from the original image (never from the first-pass resized input),
    runs the identical prepare_image → model → inverse_transform pipeline,
    then maps crop-local coords back to original pixel space.

    crop_ratio: global float or per-task dict (e.g. {"IVC": 0.25, "PSAX": 0.5}).
    refine_tasks: if provided, only refine these task IDs (overrides roi_threshold).
    roi_threshold: only refine samples whose ROI% < this value.
    """
    # Split into refine vs skip
    to_refine: list[tuple[int, dict]] = []
    results_map: dict[int, dict] = {}

    for i, r in enumerate(first_pass):
        should_refine = False
        if refine_tasks is not None:
            should_refine = r["task_id"] in refine_tasks
        else:
            oh, ow = r["original_hw"]
            pts = np.array(r["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
            roi_pct = _compute_roi_pct(pts, (oh, ow))
            should_refine = roi_pct < roi_threshold

        if should_refine:
            to_refine.append((i, r))
        else:
            results_map[i] = r

    n_skip = len(first_pass) - len(to_refine)
    if n_skip > 0:
        logger.info(
            "Refinement: %d samples (skipping %d)",
            len(to_refine),
            n_skip,
        )

    refine_items = [r for _, r in to_refine]
    refine_indices = [i for i, _ in to_refine]

    for start in tqdm(range(0, len(refine_items), batch_size), desc="Refinement"):
        chunk = refine_items[start : start + batch_size]
        chunk_indices = refine_indices[start : start + batch_size]

        images: list[np.ndarray] = []
        metas: list[dict] = []

        for r in chunk:
            bgr = cv2.imread(str(data_root / r["image_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise SubmissionError(f"Cannot read image: {data_root / r['image_path']}")

            oh, ow = r["original_hw"]
            pts = np.array(r["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)

            x1, y1, side = _compute_crop_region(
                pts,
                (oh, ow),
                strategy,
                r["task_id"],
                safety_factor,
                crop_ratio,
            )
            crop_bgr = bgr[y1 : y1 + side, x1 : x1 + side]

            chw, M, crop_hw = prepare_image(crop_bgr, input_size)
            images.append(chw)
            metas.append(
                {
                    **r,
                    "M": M,
                    "crop_hw": crop_hw,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "crop_side": side,
                }
            )

        tensor = torch.from_numpy(np.stack(images)).to(device)
        normed = module._normalize_image(tensor)
        out = module.model(normed)

        for b, meta in enumerate(metas):
            idx = chunk_indices[b]
            tid = meta["task_id"]
            task_out = out.task_outputs[tid]

            conf = task_out.conf[b].sigmoid().squeeze(-1)
            best = int(select_serving_query(conf).item())
            best_conf = float(conf[best].item())

            tdef_v = TASKS[tid]
            lm_norm = task_out.landmarks[b, best, :tdef_v.n_keypoints].detach().cpu()
            canvas_wh = torch.tensor(input_size, dtype=torch.float32)
            lm_aug_px = lm_norm.float() * canvas_wh

            lm_crop = inverse_transform_landmarks(
                lm_aug_px,
                meta["M"],
                meta["crop_hw"],
            )

            lm_orig = lm_crop.copy()
            lm_orig[:, 0] += meta["crop_x1"]
            lm_orig[:, 1] += meta["crop_y1"]

            oh, ow = meta["original_hw"]
            lm_orig[:, 0] = np.clip(lm_orig[:, 0], 0, ow - 1)
            lm_orig[:, 1] = np.clip(lm_orig[:, 1], 0, oh - 1)

            lm_orig_normed = lm_orig.copy()
            lm_orig_normed[:, 0] /= max(ow, 1)
            lm_orig_normed[:, 1] /= max(oh, 1)

            entry: dict = {
                "image_path": meta["image_path"],
                "submission_path": meta.get("submission_path", meta["image_path"]),
                "task_id": tid,
                "predicted_points_normalized": [
                    round(float(v), 6) for v in lm_orig_normed.flatten()
                ],
                "predicted_points_pixels": [round(float(v), 2) for v in lm_orig.flatten()],
                "_canonical_points_full": lm_orig,
                "confidence": round(best_conf, 4),
                "original_hw": [oh, ow],
                "refine_crop": {
                    "x1": meta["crop_x1"],
                    "y1": meta["crop_y1"],
                    "side": meta["crop_side"],
                    "strategy": strategy,
                },
            }
            if "gt_points_pixels" in meta:
                entry["gt_points_pixels"] = meta["gt_points_pixels"]

            results_map[idx] = entry

    return [results_map[i] for i in range(len(first_pass))]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(
    results: list[dict],
    output_dir: Path,
    metadata: dict,
    *,
    competition_ordering: bool = True,
) -> None:
    """Write submission JSON + detail JSON.

    When competition_ordering=True (default), submission JSONs use
    competition-ordered landmarks (via competitionize on full-precision
    coordinates). Detail JSON always stores canonical ordering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not competition_ordering:
        logger.warning(
            "Competition ordering DISABLED — submission uses canonical order. "
            "This is for debug/ablation only."
        )

    # 1. Submission format — evaluator expects regression_predictions.json
    # with image_path matching the CSV convention ("{task_id}/{filename}")
    submission = []
    n_reordered = 0
    for r in results:
        tid = r["task_id"]
        canonical_full = r.get("_canonical_points_full")

        if competition_ordering and canonical_full is not None:
            oh, ow = r["original_hw"]
            result = competitionize(canonical_full, tid)

            comp_normed = result.points.copy()
            comp_normed[:, 0] /= max(ow, 1)
            comp_normed[:, 1] /= max(oh, 1)

            sub_normed = [round(float(v), 6) for v in comp_normed.flatten()]
            sub_pixels = [round(float(v), 2) for v in result.points.flatten()]
            if result.changed:
                n_reordered += 1
        else:
            sub_normed = r["predicted_points_normalized"]
            sub_pixels = r["predicted_points_pixels"]

        submission.append(
            {
                "image_path": r["submission_path"],
                "task_id": tid,
                "predicted_points_normalized": sub_normed,
                "predicted_points_pixels": sub_pixels,
            }
        )

    if n_reordered > 0:
        logger.info("Competition ordering: %d/%d predictions reordered", n_reordered, len(results))
    # Written under BOTH names the organizer's own materials specify, because
    # they contradict each other and we get one final submission:
    #   baseline/README.md and docs/submission.md say landmark_predictions.json
    #   baseline/baseline/model.py writes and evaluate.py READS
    #     regression_predictions.json  <- what our four scored submissions used
    # Emitting both costs nothing and removes the ambiguity entirely.
    sub_path = output_dir / "regression_predictions.json"
    for name in ("regression_predictions.json", "landmark_predictions.json"):
        with open(output_dir / name, "w") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False, allow_nan=False)
    logger.info(
        "Submission JSON: %s (+ landmark_predictions.json, %d entries)", sub_path, len(submission)
    )

    # 2. Detail format (for notebooks/submission_viewer.py)
    # Strip internal fields not serializable / not for external use
    detail_preds = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ]
    metadata["competition_ordering"] = competition_ordering
    metadata["detail_landmark_order"] = "canonical"
    metadata["submission_landmark_order"] = "competition" if competition_ordering else "canonical"
    detail = {"metadata": metadata, "predictions": detail_preds}
    det_path = output_dir / "predictions_detail.json"
    with open(det_path, "w") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False, allow_nan=False)
    logger.info("Detail JSON: %s", det_path)


def _summary(results: list[dict]) -> str:
    """One-line per-task summary for console output."""
    from collections import defaultdict

    by_task: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r["confidence"])
    lines = []
    for tid in sorted(by_task):
        confs = by_task[tid]
        lines.append(
            f"  {tid:12s}  n={len(confs):4d}  conf={np.mean(confs):.3f} (min {np.min(confs):.3f})"
        )
    return "\n".join(lines)


def _mre_table(results: list[dict]) -> str:
    """Per-task MRE from results that carry GT."""
    from collections import defaultdict

    errors: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if "gt_points_pixels" not in r:
            continue
        pred = np.array(r["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
        gt = np.array(r["gt_points_pixels"], dtype=np.float32).reshape(-1, 2)
        dists = np.sqrt(((pred - gt) ** 2).sum(axis=1))
        errors[r["task_id"]].append(float(dists.mean()))

    if not errors:
        return "  (no GT available)"

    lines = []
    task_means = []
    for tid in sorted(errors):
        m = float(np.mean(errors[tid]))
        task_means.append(m)
        lines.append(f"  {tid:12s}  n={len(errors[tid]):4d}  MRE={m:.2f}")
    avg = float(np.mean(task_means))
    lines.append(f"  {'Average':12s}            MRE={avg:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission predictions from a trained checkpoint.",
    )
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--output-dir", default="predictions", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["competition", "val_local"],
        default="competition",
        help="competition: scan data/val/; val_local: manifest with GT",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-missing-tasks",
        action="store_true",
        help="Permit a PARTIAL submission when a task directory is absent",
    )
    parser.add_argument(
        "--val-subdir",
        default="val",
        help="Subdirectory for competition images (default: val)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Manifest split name (default: val_local for val_local mode)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path (default: {data_root}/manifest.parquet)",
    )
    parser.add_argument(
        "--refine",
        choices=["none", "half", "adaptive", "fixed"],
        default="none",
        help="Iterative refinement: none (single pass), "
        "half (crop = half short side), "
        "adaptive (crop = ROI × bbox_context_scale × safety_factor), "
        "fixed (crop = crop_ratio × short side, ROI-centred)",
    )
    parser.add_argument(
        "--refine-safety-factor",
        type=float,
        default=2.0,
        help="Safety multiplier for adaptive crop sizing (default: 2.0)",
    )
    parser.add_argument(
        "--refine-crop-ratio",
        type=float,
        default=0.7,
        help="Crop ratio for fixed strategy: fraction of image short side (default: 0.7)",
    )
    parser.add_argument(
        "--refine-roi-threshold",
        type=float,
        default=100.0,
        help="Only refine samples with ROI%% below this threshold (default: 100 = all)",
    )
    parser.add_argument(
        "--refine-tasks",
        default=None,
        help="Comma-separated task IDs to refine (overrides roi_threshold). Example: A4C,AOP,IVC",
    )
    parser.add_argument(
        "--competition-ordering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply competition landmark ordering to submission output (default: enabled)",
    )
    parser.add_argument(
        "--task-routing",
        choices=["directory", "high_conf"],
        default="directory",
        help="directory: use filesystem task assignment (default); "
        "high_conf: pick the task head with highest confidence per image",
    )
    parser.add_argument(
        "--task-routing-threshold",
        type=float,
        default=0.9,
        help="Only reroute when directory-assigned task conf < threshold (default: 0.9)",
    )
    parser.add_argument(
        "--tta-angles",
        default=None,
        help="Comma-separated rotation angles for TTA (e.g. '-5,0,5'). "
        "Runs inference at each angle and averages landmarks in pixel space.",
    )
    parser.add_argument(
        "--tta-color",
        default=None,
        help="Comma-separated color jitter specs for TTA "
        "(e.g. 'bright+20,bright-20,contr+25,contr-25'). "
        "No geometric inverse needed; averaged with rotation views.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_root = Path(args.data_root)
    device = torch.device(args.device)

    logger.info("Loading checkpoint: %s", args.ckpt)
    module, config = load_module(args.ckpt, args.device)
    input_size = config.backbone.input_size
    logger.info("Model input size: %s", input_size)

    if args.mode == "competition":
        samples = discover_competition_images(
            data_root, args.val_subdir, allow_missing_tasks=args.allow_missing_tasks
        )
    else:
        manifest = Path(args.manifest) if args.manifest else data_root / "manifest.parquet"
        split = args.split or "val_local"
        samples = discover_from_manifest(data_root, manifest, split)

    if not samples:
        raise SubmissionError(
            f"No samples discovered under {data_root} (mode={args.mode}). "
            f"Check --data-root and --mode."
        )

    tta_angles = None
    if args.tta_angles:
        tta_angles = [float(a.strip()) for a in args.tta_angles.split(",")]

    tta_color = None
    if args.tta_color:
        tta_color = parse_tta_color_specs(args.tta_color)

    results = run_inference_tta(
        module,
        samples,
        data_root,
        input_size,
        device,
        args.batch_size,
        task_routing=args.task_routing,
        task_routing_threshold=args.task_routing_threshold,
        tta_angles=tta_angles,
        tta_color=tta_color,
    )

    has_gt = any(r.get("gt_points_pixels") for r in results)
    if has_gt:
        logger.info("Val MRE (pass 1):\n%s", _mre_table(results))

    if args.refine != "none":
        task_filter = None
        if args.refine_tasks:
            task_filter = set(args.refine_tasks.split(","))
        results = run_refinement(
            module,
            results,
            data_root,
            input_size,
            device,
            args.batch_size,
            args.refine,
            args.refine_safety_factor,
            args.refine_crop_ratio,
            args.refine_roi_threshold,
            task_filter,
        )
        if has_gt:
            logger.info(
                "Val MRE (pass 2, %s):\n%s",
                args.refine,
                _mre_table(results),
            )

    # Fail before writing: a partial artifact on disk is one a human can upload.
    if args.task_routing == "directory":
        validate_inference_results(
            samples,
            results,
            required_tasks=None if args.allow_missing_tasks else set(TASKS),
        )
    else:
        # high_conf: task_ids may differ from directory — skip identity check,
        # but still validate count, vector lengths, and coordinate ranges.
        if len(results) != len(samples):
            raise SubmissionError(
                f"Prediction count {len(results)} != sample count {len(samples)}"
            )
        for r in results:
            label = f"{r['task_id']}/{r.get('submission_path', r.get('image_path'))}"
            _check_vector_lengths(r, label)
            _check_normalized_range(r, label)

    logger.info("Results per task:\n%s", _summary(results))

    metadata = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "input_size": input_size,
        "mode": args.mode,
        "data_root": str(data_root),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "n_predictions": len(results),
    }
    save_outputs(
        results,
        Path(args.output_dir),
        metadata,
        competition_ordering=args.competition_ordering,
    )

    # Defence in depth: validate the bytes that will actually be zipped and
    # uploaded, which is a different object from the in-memory results above.
    written = json.loads((Path(args.output_dir) / "regression_predictions.json").read_text())
    if args.task_routing == "high_conf":
        expected_idx = {(r["task_id"], r["submission_path"]) for r in results}
    else:
        expected_idx = {(str(s["task_id"]), str(s["submission_path"])) for s in samples}
    validate_submission_document(written, expected_idx)
    logger.info("Submission validated: %d entries, all inputs covered.", len(written))


if __name__ == "__main__":
    main()

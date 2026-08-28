"""Produce submission.zip from a checkpoint (single model load).

Usage:
    uv run python scripts/make_submission.py \\
        --ckpt wandb_logs/fubio/<RUN_ID>/checkpoints/<CKPT_FILE> \\
        --tag r21

    # With val_local ordering comparison table:
    uv run python scripts/make_submission.py \\
        --ckpt ... --tag r21 --eval-local

Output:
    predictions/{tag}/submission.zip              — upload to platform
    predictions/{tag}/regression_predictions.json  — evaluator reads this name
    predictions/{tag}/landmark_predictions.json    — docs spec this name
    predictions/{tag}/predictions_detail.json      — extended with confidence + GT
    predictions/SUBMISSION_LOG.md                   — appended template entry
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from fubio.data.landmark_ordering import competitionize
from fubio.data.task_registry import TASKS
from fubio.serving.predict import (
    discover_competition_images,
    discover_from_manifest,
    load_module,
    parse_tta_color_specs,
    run_inference,
    run_inference_tta,
    run_refinement,
    save_outputs,
)
from fubio.serving.validate import (
    SubmissionError,
    _check_normalized_range,
    _check_vector_lengths,
    validate_inference_results,
    validate_submission_document,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ELLIPSE_TASKS = {"HC", "FA"}


def _apply_ellipse_rebuild(results: list[dict]) -> list[dict]:
    """Post-process HC/FA: fit ellipse from 4 axis endpoints, rebuild with consistent center.

    The 4 scored landmarks define two axes. Their midpoints should coincide
    (= ellipse center), but prediction noise makes them differ by ~3-6px.
    Rebuild forces a shared center = average of the two midpoints, keeping
    axis lengths and directions unchanged.
    """
    n_rebuilt = 0
    for r in results:
        if r["task_id"] not in _ELLIPSE_TASKS:
            continue

        pts_px = np.array(r["predicted_points_pixels"]).reshape(-1, 2)
        if len(pts_px) < 4:
            continue

        P0, P1, P2, P3 = pts_px[0], pts_px[1], pts_px[2], pts_px[3]

        # Axis directions and lengths (preserved)
        va = P1 - P0
        vb = P3 - P2
        a = np.linalg.norm(va) / 2
        b = np.linalg.norm(vb) / 2
        dir_a = va / (np.linalg.norm(va) + 1e-8)
        dir_b = vb / (np.linalg.norm(vb) + 1e-8)

        # Unified center = average of two axis midpoints
        center = ((P0 + P1) / 2 + (P2 + P3) / 2) / 2

        # Rebuild 4 endpoints on the fitted ellipse
        pts_px[0] = center - a * dir_a
        pts_px[1] = center + a * dir_a
        pts_px[2] = center - b * dir_b
        pts_px[3] = center + b * dir_b

        # Clamp to image bounds (rebuild can push points outside)
        oh, ow = r["original_hw"]
        pts_px[:, 0] = np.clip(pts_px[:, 0], 0, ow - 1)
        pts_px[:, 1] = np.clip(pts_px[:, 1], 0, oh - 1)

        # Update both pixel and normalized representations
        r["predicted_points_pixels"] = [round(float(v), 2) for v in pts_px.flatten()]
        normed = pts_px.copy()
        normed[:, 0] /= max(ow, 1)
        normed[:, 1] /= max(oh, 1)
        r["predicted_points_normalized"] = [round(float(v), 6) for v in normed.flatten()]

        # Update internal canonical points if present
        if "_canonical_points_full" in r:
            r["_canonical_points_full"] = pts_px

        n_rebuilt += 1

    logger.info("Ellipse rebuild: %d/%d samples processed", n_rebuilt, len(results))
    return results
_TAG_RE = re.compile(r"[A-Za-z0-9._-]+")


def eval_local_ordering(
    results: list[dict],
    output_dir: Path | None = None,
) -> str:
    """Compare CAN vs COMP ordering on val_local predictions with GT.

    Three comparisons:
      CAN→CAN:  canonical pred vs canonical GT (training-time metric)
      CAN→COMP: canonical pred vs competition GT (ordering mismatch cost)
      COMP→COMP: competition pred vs competition GT (what evaluator scores)

    Returns formatted table string. Optionally writes to output_dir/eval_local.txt.
    """
    task_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results:
        gt_flat = r.get("gt_points_pixels")
        if gt_flat is None:
            continue
        tid = r["task_id"]
        tdef = TASKS[tid]
        k = tdef.n_keypoints

        pred_can = np.array(r["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)[:k]
        gt_can = np.array(gt_flat, dtype=np.float64).reshape(-1, 2)[:k]

        pred_comp = competitionize(pred_can.copy(), tid).points
        gt_comp = competitionize(gt_can.copy(), tid).points

        def _mre_mae(p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
            dists = np.sqrt(((p - g) ** 2).sum(axis=1))
            return float(dists.mean()), float(np.abs(p - g).mean())

        cc_mre, cc_mae = _mre_mae(pred_can, gt_can)
        cx_mre, cx_mae = _mre_mae(pred_can, gt_comp)
        xx_mre, xx_mae = _mre_mae(pred_comp, gt_comp)

        task_metrics[tid]["cc_mre"].append(cc_mre)
        task_metrics[tid]["cc_mae"].append(cc_mae)
        task_metrics[tid]["cx_mre"].append(cx_mre)
        task_metrics[tid]["cx_mae"].append(cx_mae)
        task_metrics[tid]["xx_mre"].append(xx_mre)
        task_metrics[tid]["xx_mae"].append(xx_mae)

    # Build table
    lines: list[str] = []
    lines.append("")
    lines.append("Val-local ordering comparison (pixels)")
    lines.append("=" * 100)
    hdr = (
        f"{'Task':>12} | {'n':>3} | "
        f"{'CAN→CAN':>8} {'':>8} | "
        f"{'CAN→COMP':>8} {'':>8} | "
        f"{'COMP→COMP':>9} {'':>8} | "
        f"{'Δ(C→C)':>7}"
    )
    sub = (
        f"{'':>12} | {'':>3} | "
        f"{'MRE':>8} {'MAE':>8} | "
        f"{'MRE':>8} {'MAE':>8} | "
        f"{'MRE':>9} {'MAE':>8} | "
        f"{'MRE':>7}"
    )
    lines.append(hdr)
    lines.append(sub)
    lines.append("-" * 100)

    all_cc_mre, all_xx_mre = [], []
    task_order = sorted(task_metrics.keys())
    for tid in task_order:
        m = task_metrics[tid]
        n = len(m["cc_mre"])
        cc_m = np.mean(m["cc_mre"])
        cc_a = np.mean(m["cc_mae"])
        cx_m = np.mean(m["cx_mre"])
        cx_a = np.mean(m["cx_mae"])
        xx_m = np.mean(m["xx_mre"])
        xx_a = np.mean(m["xx_mae"])
        delta = xx_m - cc_m

        all_cc_mre.append(cc_m)
        all_xx_mre.append(xx_m)

        lines.append(
            f"{tid:>12} | {n:>3} | "
            f"{cc_m:>8.2f} {cc_a:>8.2f} | "
            f"{cx_m:>8.2f} {cx_a:>8.2f} | "
            f"{xx_m:>9.2f} {xx_a:>8.2f} | "
            f"{delta:>+7.2f}"
        )

    lines.append("-" * 100)
    avg_cc = np.mean(all_cc_mre)
    avg_xx = np.mean(all_xx_mre)
    lines.append(
        f"{'Avg':>12} | {'':>3} | "
        f"{avg_cc:>8.2f} {'':>8} | "
        f"{'':>8} {'':>8} | "
        f"{avg_xx:>9.2f} {'':>8} | "
        f"{avg_xx - avg_cc:>+7.2f}"
    )
    lines.append("")

    table = "\n".join(lines)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_local.txt").write_text(table)

    return table


def _assert_all_tasks_present(document: list[dict]) -> None:
    """Every registered task must appear."""
    present = {str(e["task_id"]) for e in document}
    absent = set(TASKS) - present
    if absent:
        raise SubmissionError(f"Submission has no predictions for task(s): {sorted(absent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make submission ZIP + viewer cache")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--tag", required=True, help="Output tag (e.g. r21)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--competition-ordering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply competition landmark ordering to submission output (default: enabled)",
    )
    parser.add_argument(
        "--eval-local",
        action="store_true",
        help="Run val_local CAN/COMP ordering comparison and print table",
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
        "--refine",
        type=str,
        default=None,
        help=(
            "Per-task crop ratio for coarse-to-fine refinement. "
            "Format: TASK:ratio,TASK:ratio (e.g. 'IVC:0.25,PSAX:0.5,HC:0.75'). "
            "Tasks not listed are NOT refined."
        ),
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
    parser.add_argument(
        "--model-source",
        choices=["student", "teacher", "ensemble"],
        default="student",
        help="student: normal model weights (default). "
        "teacher: EMA teacher weights (temporal ensemble, may generalize better). "
        "ensemble: average student + teacher predictions per image.",
    )
    parser.add_argument(
        "--ellipse-rebuild",
        action="store_true",
        help="Post-process HC/FA predictions: fit ellipse from 4 axis endpoints, "
        "project back to force consistent center. Fixes center mismatch (~3-6px).",
    )
    parser.add_argument(
        "--ellipse-consensus",
        action="store_true",
        help="HC/FA TTA averaging in ellipse parameter space instead of landmark "
        "space. Forces orthogonal axes, shared center, consistent geometry.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")

    if not _TAG_RE.fullmatch(args.tag):
        sys.exit(f"Invalid --tag {args.tag!r}: allowed characters are [A-Za-z0-9._-]")

    output_dir = PROJECT_ROOT / "predictions" / args.tag
    if output_dir.exists():
        sys.exit(
            f"{output_dir} already exists. Submission tags are immutable — "
            f"use a new --tag rather than overwriting evidence."
        )

    data_root = Path(args.data_root)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load model once ===
    print(f"\n{'=' * 60}")
    print(f"Loading model: {ckpt}")
    print(f"{'=' * 60}")
    module, config = load_module(str(ckpt), device_str)
    input_size = config.backbone.input_size
    device = next(module.parameters()).device

    # Swap to teacher or prepare ensemble if requested
    if args.model_source in ("teacher", "ensemble"):
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        teacher_sd = raw.get("teacher_state_dict")
        if teacher_sd is None:
            sys.exit("Checkpoint has no teacher_state_dict — was semi.lambda_pseudo > 0?")
        if args.model_source == "teacher":
            module.model.load_state_dict(teacher_sd)
            logger.info("Swapped to EMA teacher weights for inference")
        else:
            # Ensemble: keep student in module.model, build teacher copy
            from copy import deepcopy

            teacher_model = deepcopy(module.model)
            teacher_model.load_state_dict(teacher_sd)
            teacher_model.eval()
            object.__setattr__(module, "_ensemble_teacher", teacher_model)
            logger.info("Student + Teacher ensemble mode")

    # === Step 1: Competition inference → staging dir ===
    print(f"\n{'=' * 60}")
    print(f"Step 1: Competition inference → {output_dir}")
    print(f"{'=' * 60}")

    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{args.tag}.staging-"))
    try:
        samples = discover_competition_images(data_root, "val")

        tta_angles = None
        if args.tta_angles:
            tta_angles = [float(a.strip()) for a in args.tta_angles.split(",")]

        tta_color = None
        if args.tta_color:
            tta_color = parse_tta_color_specs(args.tta_color)

        results = run_inference_tta(
            module, samples, data_root, input_size, device, args.batch_size,
            task_routing=args.task_routing,
            task_routing_threshold=args.task_routing_threshold,
            tta_angles=tta_angles,
            tta_color=tta_color,
            ellipse_consensus=args.ellipse_consensus,
        )

        # Coarse-to-fine refinement: re-run selected tasks on zoomed crops
        if args.refine:
            refine_map: dict[str, float] = {}
            for pair in args.refine.split(","):
                task, ratio = pair.strip().split(":")
                refine_map[task.strip()] = float(ratio.strip())
            logger.info("Refinement: %s", refine_map)
            results = run_refinement(
                module,
                results,
                data_root,
                input_size,
                device,
                args.batch_size,
                strategy="fixed",
                crop_ratio=refine_map,
                refine_tasks=set(refine_map.keys()),
            )

        # Ellipse rebuild: force HC/FA predictions onto a consistent ellipse
        if args.ellipse_rebuild:
            results = _apply_ellipse_rebuild(results)

        if args.task_routing == "directory":
            validate_inference_results(samples, results, required_tasks=set(TASKS))
        else:
            if len(results) != len(samples):
                raise SubmissionError(
                    f"Prediction count {len(results)} != sample count {len(samples)}"
                )
            for r in results:
                label = f"{r['task_id']}/{r.get('submission_path', r.get('image_path'))}"
                _check_vector_lengths(r, label)
                _check_normalized_range(r, label)

        metadata = {
            "checkpoint": str(ckpt.resolve()),
            "input_size": input_size,
            "mode": "competition",
            "data_root": str(data_root),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "n_predictions": len(results),
        }
        save_outputs(results, staging, metadata, competition_ordering=args.competition_ordering)

        # === Step 2: Validate serialized bytes + zip ===
        reg_json = staging / "regression_predictions.json"
        lm_json = staging / "landmark_predictions.json"
        document = json.loads(reg_json.read_text())
        validate_submission_document(
            document, {(str(e["task_id"]), str(e["image_path"])) for e in document}
        )
        _assert_all_tasks_present(document)

        # Evaluator reads regression_predictions.json; docs spec landmark_predictions.json.
        zip_path = staging / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(reg_json, "regression_predictions.json")
            zf.write(lm_json, "landmark_predictions.json")

        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    zip_path = output_dir / "submission.zip"
    print(f"\n{'=' * 60}")
    print(f"Step 2: submission.zip ({zip_path.stat().st_size / 1024:.1f} KB)")
    print(f"  → {zip_path}")
    print(f"{'=' * 60}")

    # === Step 3: Val-local ordering comparison ===
    if args.eval_local:
        print(f"\n{'=' * 60}")
        print("Step 4: Val-local CAN/COMP ordering comparison")
        print(f"{'=' * 60}")

        manifest = data_root / "manifest.parquet"
        local_samples = discover_from_manifest(data_root, manifest, "val_local")
        local_results = run_inference(
            module, local_samples, data_root, input_size, device, args.batch_size,
        )
        table = eval_local_ordering(local_results, output_dir)
        print(table)

    # === Step 4: Submission log ===
    _append_log(ckpt, args.tag, output_dir)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"  Upload:  {zip_path}")
    print("  Log:     predictions/SUBMISSION_LOG.md (fill in scores after eval)")
    print(f"{'=' * 60}\n")


def _append_log(ckpt: Path, tag: str, output_dir: Path) -> None:
    """Append a template entry to predictions/SUBMISSION_LOG.md."""
    log_path = PROJECT_ROOT / "predictions" / "SUBMISSION_LOG.md"

    ckpt_abs = ckpt.resolve()
    try:
        ckpt_rel = ckpt_abs.relative_to(PROJECT_ROOT)
    except ValueError:
        ckpt_rel = ckpt
    parts = ckpt_rel.parts
    try:
        ci = list(parts).index("checkpoints")
        run_id = parts[ci - 1]
        rest = parts[ci + 1 :]
        epoch_dir = rest[0] if len(rest) >= 2 else ckpt.name
        loss_file = rest[1] if len(rest) >= 2 else ckpt.name
    except (ValueError, IndexError):
        run_id = "?"
        epoch_dir = "?"
        loss_file = ckpt.name

    epoch_match = re.search(r"epoch=(\d+)", epoch_dir)
    epoch_str = epoch_match.group(1) if epoch_match else epoch_dir

    loss_match = re.search(r"(?:loss|overall)=([\d.]+)", loss_file)
    loss_str = loss_match.group(1) if loss_match else "?"

    detail_path = output_dir / "predictions_detail.json"
    n_preds = "?"
    if detail_path.exists():
        with open(detail_path) as f:
            n_preds = json.load(f).get("metadata", {}).get("n_predictions", "?")

    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    entry = f"""
---

## {tag.upper()} — {today}

- **Checkpoint**: `{run_id}` / `epoch={epoch_str}` / val_loss={loss_str}
- **Resolution**: (check config)
- **N predictions**: {n_preds}

| Task | MRE | MAE | Local MRE | Local MAE | Local TTA MRE | Local TTA MAE |
|------|----:|----:|----------:|----------:|--------------:|--------------:|
| A4C | | | | | | |
| AOP | | | | | | |
| FA | | | | | | |
| fetal_femur | | | | | | |
| FUGC | | | | | | |
| HC | | | | | | |
| IVC | | | | | | |
| PLAX | | | | | | |
| PSAX | | | | | | |
| **Average** | **** | **** | | | | |
"""

    if not log_path.exists():
        log_path.write_text("# Submission Log\n")

    content = log_path.read_text()
    first_hr = content.find("\n---\n")
    if first_hr >= 0:
        content = content[:first_hr] + entry + content[first_hr:]
    else:
        content += entry

    table_row = f"| ? | {tag.upper()} | {today} | {run_id} | {epoch_str} | ? | — | — |"
    header_pattern = r"(\| # \| Tag \|.*?\|\n)"
    match = re.search(header_pattern, content)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + table_row + "\n" + content[insert_pos:]

    log_path.write_text(content)

    print(f"\n{'=' * 60}")
    print("Step 4: SUBMISSION_LOG.md updated — fill in scores after platform eval")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

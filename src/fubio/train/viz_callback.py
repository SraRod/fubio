"""Validation visualization callback — Ultralytics-style prediction grids.

Every N epochs, randomly samples up to 9 val images per task, runs inference,
and logs a 3×3 annotated grid per task to wandb. Each cell shows GT vs Pred
overlays (bbox, landmarks, parameter lines). Grid title shows task-level
MRE and ParamMAE; each cell shows sample filename and per-sample metrics.

Upstream: train/module.py (FUBioModule), data/task_registry.py, evaluation/geometry.py.
Downstream: train/train.py (added to Trainer callbacks).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from fubio.data.task_registry import OVERLAYS_BY_TASK, PARAMS

logger = logging.getLogger(__name__)

# --- Colors (BGR for cv2) ---
GT_COLOR = (107, 255, 75)  # #4BFF6B green
PRED_COLOR = (75, 75, 255)  # #FF4B4B red
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)
CELL_BG = (40, 40, 40)

# ImageNet stats for denormalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _denormalize_image(img_tensor: Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor → (H, W, 3) uint8 BGR."""
    img = img_tensor.cpu().float().numpy().transpose(1, 2, 0)  # CHW→HWC
    img = np.clip(img * _STD + _MEAN, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _draw_bbox(
    canvas: np.ndarray,
    kp: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    h: int,
    w: int,
) -> None:
    """Draw bounding box from visible keypoints."""
    valid = mask & np.isfinite(kp).all(axis=1)
    if not valid.any():
        return
    pts = kp[valid]
    # Denormalize from [0,1] to pixel
    px = pts[:, 0] * w
    py = pts[:, 1] * h
    pad_x, pad_y = w * 0.03, h * 0.03
    x0 = int(max(0, px.min() - pad_x))
    y0 = int(max(0, py.min() - pad_y))
    x1 = int(min(w, px.max() + pad_x))
    y1 = int(min(h, py.max() + pad_y))
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


def _draw_keypoints(
    canvas: np.ndarray,
    kp: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    h: int,
    w: int,
    marker: str = "circle",
) -> None:
    """Draw keypoints on canvas. marker: 'circle' or 'cross'."""
    for k in range(len(kp)):
        if not mask[k] or not np.isfinite(kp[k]).all():
            continue
        x = int(kp[k, 0] * w)
        y = int(kp[k, 1] * h)
        if marker == "circle":
            cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            s = 4
            cv2.line(canvas, (x - s, y - s), (x + s, y + s), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (x - s, y + s), (x + s, y - s), color, 2, cv2.LINE_AA)


def _draw_param_lines(
    canvas: np.ndarray,
    kp: np.ndarray,
    mask: np.ndarray,
    task_id: str,
    color: tuple[int, int, int],
    h: int,
    w: int,
) -> None:
    """Draw overlay geometry (measurement + supportive) for a task."""
    for pdef in OVERLAYS_BY_TASK.get(task_id, []):
        li = pdef.landmark_indices
        pts_valid = all(
            li_k < len(kp) and mask[li_k] and np.isfinite(kp[li_k]).all() for li_k in li
        )
        if not pts_valid:
            continue

        if pdef.formula_type == "distance":
            p1 = (int(kp[li[0], 0] * w), int(kp[li[0], 1] * h))
            p2 = (int(kp[li[1], 0] * w), int(kp[li[1], 1] * h))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

        elif pdef.formula_type == "ellipse_circumference":
            # Draw ellipse from 4 axis endpoints
            p1 = np.array([kp[li[0], 0] * w, kp[li[0], 1] * h])
            p2 = np.array([kp[li[1], 0] * w, kp[li[1], 1] * h])
            p3 = np.array([kp[li[2], 0] * w, kp[li[2], 1] * h])
            p4 = np.array([kp[li[3], 0] * w, kp[li[3], 1] * h])
            center = ((p1 + p2) / 2).astype(int)
            a = float(np.linalg.norm(p1 - p2)) / 2
            b = float(np.linalg.norm(p3 - p4)) / 2
            # Angle of major axis
            d = p2 - p1
            angle = float(np.degrees(np.arctan2(d[1], d[0])))
            if a > 0 and b > 0:
                cv2.ellipse(
                    canvas, tuple(center), (int(a), int(b)), angle, 0, 360, color, 1, cv2.LINE_AA
                )

        elif pdef.formula_type == "angle":
            v = (int(kp[li[0], 0] * w), int(kp[li[0], 1] * h))
            a1 = (int(kp[li[1], 0] * w), int(kp[li[1], 1] * h))
            a2 = (int(kp[li[2], 0] * w), int(kp[li[2], 1] * h))
            cv2.line(canvas, v, a1, color, 1, cv2.LINE_AA)
            cv2.line(canvas, v, a2, color, 1, cv2.LINE_AA)
            # Small arc at vertex
            d1 = np.array(a1, dtype=float) - np.array(v, dtype=float)
            d2 = np.array(a2, dtype=float) - np.array(v, dtype=float)
            ang1 = float(np.degrees(np.arctan2(d1[1], d1[0])))
            ang2 = float(np.degrees(np.arctan2(d2[1], d2[0])))
            cv2.ellipse(canvas, v, (20, 20), 0, ang1, ang2, color, 1, cv2.LINE_AA)


def _put_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.35,
    color: tuple[int, int, int] = TEXT_FG,
) -> int:
    """Put text with dark background. Returns new y position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(canvas, (x, y - th - 2), (x + tw + 4, y + baseline + 2), TEXT_BG, -1)
    cv2.putText(canvas, text, (x + 2, y), font, scale, color, 1, cv2.LINE_AA)
    return y + th + baseline + 4


# ---------------------------------------------------------------------------
# Single cell renderer
# ---------------------------------------------------------------------------


def _render_cell(
    image: np.ndarray,
    gt_kp: np.ndarray,
    pred_kp: np.ndarray,
    gt_mask: np.ndarray,
    task_id: str,
    filename: str,
    sample_mre: float,
    sample_mae: float,
    cell_size: int = 518,
) -> np.ndarray:
    """Render a single GT vs Pred overlay cell.

    Args:
        image: (H, W, 3) BGR uint8.
        gt_kp: (K_task, 2) normalized [0,1].
        pred_kp: (K_task, 2) normalized [0,1].
        gt_mask: (K_task,) bool — visible landmarks.
        task_id: task string.
        filename: source image filename for annotation.
        sample_mre: per-sample MRE in pixels.
        sample_mae: per-sample parameter MAE.
        cell_size: output cell dimension.
    """
    h, w = image.shape[:2]
    canvas = cv2.resize(image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
    ch, cw = cell_size, cell_size

    # Scale factors for drawing (kp are in [0,1] relative to original H,W)
    # But since we resize to cell_size, we draw in cell_size coords

    # Draw GT: bbox + param lines + keypoints
    _draw_bbox(canvas, gt_kp, gt_mask, GT_COLOR, ch, cw)
    _draw_param_lines(canvas, gt_kp, gt_mask, task_id, GT_COLOR, ch, cw)
    _draw_keypoints(canvas, gt_kp, gt_mask, GT_COLOR, ch, cw, marker="circle")

    # Draw Pred: bbox + param lines + keypoints
    _draw_bbox(canvas, pred_kp, gt_mask, PRED_COLOR, ch, cw)
    _draw_param_lines(canvas, pred_kp, gt_mask, task_id, PRED_COLOR, ch, cw)
    _draw_keypoints(canvas, pred_kp, gt_mask, PRED_COLOR, ch, cw, marker="cross")

    # Text annotations
    short_name = Path(filename).stem[:25]
    y = 12
    y = _put_text(canvas, short_name, 2, y, scale=0.30)
    mre_str = f"MRE:{sample_mre:.1f}px"
    mae_str = f"MAE:{sample_mae:.2f}" if not math.isnan(sample_mae) else "MAE:N/A"
    _put_text(canvas, f"{mre_str}  {mae_str}", 2, y, scale=0.30)

    return canvas


# ---------------------------------------------------------------------------
# Grid composer
# ---------------------------------------------------------------------------


def _compose_grid(
    cells: list[np.ndarray],
    title: str,
    cell_size: int = 518,
    cols: int = 3,
) -> np.ndarray:
    """Compose cells into a 3×3 grid with title bar."""
    rows = 3
    title_h = 36
    grid_h = title_h + rows * cell_size
    grid_w = cols * cell_size

    grid = np.full((grid_h, grid_w, 3), CELL_BG, dtype=np.uint8)

    # Title bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 0.6, 1)
    tx = (grid_w - tw) // 2
    cv2.putText(grid, title, (tx, title_h - 10), font, 0.6, TEXT_FG, 1, cv2.LINE_AA)

    # Place cells
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        y0 = title_h + r * cell_size
        x0 = c * cell_size
        grid[y0 : y0 + cell_size, x0 : x0 + cell_size] = cell

    return grid


# ---------------------------------------------------------------------------
# Lightning Callback
# ---------------------------------------------------------------------------

try:
    import lightning as L
    import wandb as _wandb

    class VisualizationCallback(L.Callback):
        """Log GT vs Pred landmark visualization grids to wandb.

        Every ``log_every_n_epochs`` validation epochs, samples up to 9 val
        images per task, runs inference, and logs an annotated 3×3 grid.
        """

        def __init__(
            self,
            log_every_n_epochs: int = 5,
            n_samples_per_task: int = 9,
            cell_size: int = 518,
        ) -> None:
            super().__init__()
            self._log_every = log_every_n_epochs
            self._n_samples = n_samples_per_task
            self._cell_size = cell_size

        def on_validation_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
        ) -> None:
            if trainer.sanity_checking:
                return
            if (trainer.current_epoch + 1) % self._log_every != 0:
                return

            val_dl = trainer.val_dataloaders
            if val_dl is None:
                return

            logger.info("VisualizationCallback: rendering epoch %d", trainer.current_epoch)

            # Collect per-task samples from val dataloader
            task_samples = self._collect_samples(val_dl, pl_module)

            # Get task-level metrics from module — use pixel-space MRE
            # so the displayed value matches val_mre_pixel in the logs.
            mre_metrics = pl_module._val_mre_pixel.compute()  # type: ignore[attr-defined]
            param_metrics = pl_module._val_param_mae.compute()  # type: ignore[attr-defined]

            for task_id in sorted(task_samples.keys()):
                samples = task_samples[task_id]
                if not samples:
                    continue

                task_mre = mre_metrics.get(f"task/{task_id}", float("nan"))
                if isinstance(task_mre, Tensor):
                    task_mre = task_mre.item()

                # Per-task MAE: average of constituent param MAEs
                task_param_vals = [
                    v.item() if isinstance(v, Tensor) else v
                    for pname, pdef in PARAMS.items()
                    if pdef.task_id == task_id
                    and (v := param_metrics.get(f"param/{pname}")) is not None
                ]
                task_mae = (
                    sum(task_param_vals) / len(task_param_vals) if task_param_vals else float("nan")
                )

                cells = []
                for s in samples:
                    cell = _render_cell(
                        image=s["image"],
                        gt_kp=s["gt_kp"],
                        pred_kp=s["pred_kp"],
                        gt_mask=s["gt_mask"],
                        task_id=task_id,
                        filename=s["filename"],
                        sample_mre=s["mre"],
                        sample_mae=s["mae"],
                        cell_size=self._cell_size,
                    )
                    cells.append(cell)

                mre_str = f"{task_mre:.1f}" if not math.isnan(task_mre) else "N/A"
                mae_str = f"{task_mae:.2f}" if not math.isnan(task_mae) else "N/A"
                title = f"{task_id}  |  MRE: {mre_str}px  |  MAE: {mae_str}"

                grid = _compose_grid(cells, title, self._cell_size)
                grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

                trainer.logger.experiment.log(  # type: ignore[union-attr]
                    {f"val/viz/{task_id}": _wandb.Image(grid_rgb)},
                    step=trainer.global_step,
                )

            logger.info("VisualizationCallback: logged %d task grids", len(task_samples))

        @torch.no_grad()
        def _collect_samples(
            self,
            val_dl: object,
            pl_module: L.LightningModule,
        ) -> dict[str, list[dict]]:
            """Collect random val samples with predictions, grouped by task."""
            from fubio.data.task_registry import TASKS as _TASKS
            from fubio.evaluation.geometry import compute_params_for_evaluation

            device = pl_module.device
            task_samples: dict[str, list[dict]] = {tid: [] for tid in _TASKS}
            task_budget = {tid: self._n_samples for tid in _TASKS}

            all_batches = list(val_dl)  # type: ignore[arg-type]
            rng = np.random.default_rng(seed=None)
            rng.shuffle(all_batches)

            for batch in all_batches:
                if all(b <= 0 for b in task_budget.values()):
                    break

                images = pl_module._normalize_image(batch["image"].to(device))
                targets = batch["targets"]

                # Single-pass forward — unpack ModelOutput
                model_out = pl_module.model(images)
                results = model_out.task_outputs
                matched = pl_module.matcher(results, targets)

                for tid, task_out in results.items():
                    if task_budget[tid] <= 0:
                        continue
                    task_matched = matched[tid]

                    for b, (q_idx, g_idx) in enumerate(task_matched):
                        if len(q_idx) == 0:
                            continue
                        if task_budget[tid] <= 0:
                            break

                        for qi, gi in zip(q_idx, g_idx, strict=True):
                            if task_budget[tid] <= 0:
                                break

                            inst = targets[b][gi]
                            K_scored = _TASKS[tid].n_keypoints
                            # Slice to scored landmarks — supportive are training-only
                            gt_kp = np.array(inst["keypoints"], dtype=np.float32)[:K_scored]
                            pred_kp = task_out.landmarks[b, qi, :K_scored].detach().cpu().numpy()
                            vis_mask = np.array(inst["visible_mask"], dtype=bool)[:K_scored]
                            sup_mask = np.array(inst["supervised_mask"], dtype=bool)[:K_scored]
                            mask = vis_mask & sup_mask

                            valid = (
                                mask
                                & np.isfinite(gt_kp).all(axis=1)
                                & np.isfinite(pred_kp).all(axis=1)
                            )
                            # Pixel scale = max(orig_h, orig_w), matching
                            # val_mre_pixel in module.py (letterbox → square canvas).
                            orig_hw = inst.get("original_hw", (518, 518))
                            pixel_scale = float(max(orig_hw[0], orig_hw[1]))
                            if valid.any():
                                diff = (pred_kp[valid] - gt_kp[valid]) * pixel_scale
                                mre_val = float(np.sqrt((diff**2).sum(axis=1)).mean())
                            else:
                                mre_val = float("nan")

                            gt_t = torch.from_numpy(gt_kp).unsqueeze(0) * pixel_scale
                            pred_t = torch.from_numpy(pred_kp).unsqueeze(0) * pixel_scale
                            vis_t = torch.from_numpy(vis_mask).unsqueeze(0)
                            gt_params = compute_params_for_evaluation(gt_t, tid, vis_t)
                            pred_params = compute_params_for_evaluation(pred_t, tid, vis_t)
                            param_errors = []
                            for k in gt_params:
                                gv = gt_params[k].item()
                                pv = pred_params[k].item()
                                if not math.isnan(gv) and not math.isnan(pv):
                                    param_errors.append(abs(gv - pv))
                            mae_val = float(np.mean(param_errors)) if param_errors else float("nan")

                            img_bgr = _denormalize_image(images[b])

                            filename = inst.get("image_path", "unknown")

                            task_samples[tid].append(
                                {
                                    "image": img_bgr,
                                    "gt_kp": gt_kp,
                                    "pred_kp": pred_kp,
                                    "gt_mask": mask,
                                    "filename": filename,
                                    "mre": mre_val,
                                    "mae": mae_val,
                                }
                            )
                            task_budget[tid] -= 1

            return {k: v for k, v in task_samples.items() if v}

except ImportError:
    pass

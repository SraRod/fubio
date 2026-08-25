"""Per-task Hungarian matching between instance queries and GT instances.

For each task present in the batch, matches N_inst query slots to that
task's GT instances using conf + bbox L1 + CIoU cost. Tasks with no GT
in an image produce empty matches.

Cost computation runs entirely on CPU (numpy) — the cost matrices are
tiny (n_inst × M_gt, typically 2×1 or 2×2) so GPU kernel launch overhead
dominates. One bulk GPU→CPU transfer per task, then numpy-only loops.

Upstream: data/types.py (InstanceDict for GT bbox/task_id).
Downstream: train/module.py (training_step calls matcher before loss).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from fubio.data.task_registry import TASKS
from fubio.data.types import InstanceDict

_INT_TO_TID: dict[int, str] = {tdef.task_int: tid for tid, tdef in TASKS.items()}


# ---------------------------------------------------------------------------
# Numpy cost utilities (replacing GPU torch ops for tiny matrices)
# ---------------------------------------------------------------------------


def _ciou_cost_np(pred_cxcywh: np.ndarray, gt_cxcywh: np.ndarray) -> np.ndarray:
    """CIoU cost (1 - CIoU) in numpy. (N, 4) × (M, 4) → (N, M)."""

    # cxcywh → xyxy
    def to_xyxy(b: np.ndarray) -> np.ndarray:
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    p = to_xyxy(pred_cxcywh)
    g = to_xyxy(gt_cxcywh)

    x1 = np.maximum(p[:, None, 0], g[None, :, 0])
    y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2])
    y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area1 = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / np.maximum(union, 1e-7)

    ex1 = np.minimum(p[:, None, 0], g[None, :, 0])
    ey1 = np.minimum(p[:, None, 1], g[None, :, 1])
    ex2 = np.maximum(p[:, None, 2], g[None, :, 2])
    ey2 = np.maximum(p[:, None, 3], g[None, :, 3])
    c_diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

    d_sq = (pred_cxcywh[:, None, 0] - gt_cxcywh[None, :, 0]) ** 2 + (
        pred_cxcywh[:, None, 1] - gt_cxcywh[None, :, 1]
    ) ** 2

    v = (4.0 / (math.pi**2)) * (
        np.arctan(gt_cxcywh[None, :, 2] / np.maximum(gt_cxcywh[None, :, 3], 1e-7))
        - np.arctan(pred_cxcywh[:, None, 2] / np.maximum(pred_cxcywh[:, None, 3], 1e-7))
    ) ** 2
    alpha = v / ((1.0 - iou) + v + 1e-7)

    return 1.0 - (iou - d_sq / c_diag_sq - alpha * v)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.clip(-50, 50)))


def _landmark_cost_np(pred_lm: np.ndarray, gt_kp: np.ndarray, gt_vis: np.ndarray) -> np.ndarray:
    """Mean L1 landmark distance over visible GT points. (n_inst, M).

    pred_lm: (n_inst, K, 2), gt_kp: (M, K, 2), gt_vis: (M, K) bool.
    """
    diff = np.abs(pred_lm[:, None, :, :] - gt_kp[None, :, :, :]).sum(-1)  # (n_inst, M, K)
    vis = gt_vis[None, :, :].astype(np.float32)  # (1, M, K)
    denom = vis.sum(-1)  # (1, M)
    return (diff * vis).sum(-1) / np.maximum(denom, 1e-6)


_EMPTY: tuple[list[int], list[int]] = ([], [])


class PerTaskMatcher(nn.Module):
    """Per-task bipartite matching on CPU (numpy).

    One bulk GPU→CPU transfer per task, then all cost computation and
    Hungarian matching in numpy. Eliminates ~576 GPU kernel launches
    per step.
    """

    def __init__(
        self,
        cost_conf: float = 1.0,
        cost_box: float = 5.0,
        cost_ciou: float = 2.0,
        cost_land: float = 0.0,
    ) -> None:
        super().__init__()
        self.cost_conf = cost_conf
        self.cost_box = cost_box
        self.cost_ciou = cost_ciou
        # cost_land dominates when > 0 so assignment tracks landmark quality
        # instead of the landmark-independent bbox head (see MatcherConfig).
        self.cost_land = cost_land

    @torch.no_grad()
    def forward(
        self,
        model_output: dict[str, tuple[Tensor, Tensor, Tensor]],
        targets: list[list[InstanceDict]],
    ) -> dict[str, list[tuple[list[int], list[int]]]]:
        B = len(targets)

        gt_by_task: dict[str, list[list[int]]] = {tid: [[] for _ in range(B)] for tid in TASKS}
        for b, instances in enumerate(targets):
            for gi, inst in enumerate(instances):
                tid = _INT_TO_TID.get(inst["task_id"])
                if tid is not None:
                    gt_by_task[tid][b].append(gi)

        results: dict[str, list[tuple[list[int], list[int]]]] = {}

        for tid, task_out in model_output.items():
            pred_bbox, pred_conf = task_out.bbox, task_out.conf
            task_gt_lists = gt_by_task[tid]

            # Collect GT boxes (numpy, no GPU)
            all_gt_boxes_np: list[np.ndarray] = []
            image_ranges: list[tuple[int, int]] = []
            for b in range(B):
                start = len(all_gt_boxes_np)
                for gi in task_gt_lists[b]:
                    all_gt_boxes_np.append(targets[b][gi]["bbox"])
                image_ranges.append((start, len(all_gt_boxes_np)))

            if not all_gt_boxes_np:
                results[tid] = [_EMPTY] * B
                continue

            gt_boxes_all = np.stack(all_gt_boxes_np).astype(np.float32)

            # One bulk GPU→CPU transfer per task (float() for bf16 compat)
            pred_bbox_np = pred_bbox.float().cpu().numpy()
            pred_conf_np = pred_conf.float().cpu().numpy()
            pred_lm_np = task_out.landmarks.float().cpu().numpy() if self.cost_land > 0 else None

            per_image: list[tuple[list[int], list[int]]] = []

            for b in range(B):
                start, end = image_ranges[b]
                if start == end:
                    per_image.append(_EMPTY)
                    continue

                p_box = pred_bbox_np[b]  # (n_inst, 4)
                p_conf = pred_conf_np[b]  # (n_inst, 1)
                gt_boxes = gt_boxes_all[start:end]  # (M, 4)
                n_gt = end - start

                cost_conf = -np.broadcast_to(_sigmoid(p_conf), (p_conf.shape[0], n_gt))
                cost_box = cdist(p_box, gt_boxes, metric="cityblock")
                cost_ciou = _ciou_cost_np(p_box, gt_boxes)

                cost = (
                    self.cost_conf * cost_conf
                    + self.cost_box * cost_box
                    + self.cost_ciou * cost_ciou
                )

                if pred_lm_np is not None:
                    gis = task_gt_lists[b]
                    gt_kp = np.stack([targets[b][gi]["keypoints"] for gi in gis])  # (M,K_gt,2)
                    gt_vis = np.stack([targets[b][gi]["visible_mask"] for gi in gis])  # (M,K_gt)
                    # GT may have supportive landmarks beyond model's prediction range
                    K_pred = pred_lm_np.shape[2]
                    if gt_kp.shape[1] > K_pred:
                        gt_kp = gt_kp[:, :K_pred]
                        gt_vis = gt_vis[:, :K_pred]
                    cost = cost + self.cost_land * _landmark_cost_np(pred_lm_np[b], gt_kp, gt_vis)

                if not np.isfinite(cost).all():
                    # NaN/Inf in cost → replace with large finite value so
                    # matching can proceed (training step will produce high
                    # loss but won't crash). Logged for diagnosis.
                    import logging as _log

                    _log.getLogger(__name__).warning(
                        "NaN/Inf in matcher cost for task %s image %d — replacing with 1e6",
                        tid,
                        b,
                    )
                    cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
                row_ind, col_ind = linear_sum_assignment(cost)

                orig_gt_idx = [task_gt_lists[b][c] for c in col_ind]
                per_image.append((row_ind.tolist(), orig_gt_idx))

            results[tid] = per_image

        return results

"""Evaluation metrics for FU_Biometry: MRE, Parameter MAE, BBox Precision.

MRE (Mean Radial Error) is the primary landmark quality metric.
Aggregation: per-landmark → per-image → per-task → per-domain → overall.
Overall is domain-balanced (mean of domain-level MREs) to avoid
large-K tasks dominating.

ParamMAE tracks per-parameter mean absolute error in native units
(pixels for lengths, degrees for angles). It emits no cross-parameter
aggregate on purpose — see the class docstring.

BBoxPrecision tracks localization quality of matched bounding boxes
at standard IoU thresholds (0.5, 0.75).

Upstream: evaluation/geometry.py (for param computation).
Downstream: train/module.py (validation_step accumulates, on_validation_epoch_end computes).
"""

from __future__ import annotations

import torch
from torch import Tensor

from fubio.data.task_registry import PARAMS, TASKS, int_to_task_id

# Domain grouping — derived from task_registry at import time
_TASK_TO_DOMAIN: dict[int, str] = {t.task_int: t.domain for t in TASKS.values()}
_DOMAIN_TASKS: dict[str, list[int]] = {}
for _t in TASKS.values():
    _DOMAIN_TASKS.setdefault(_t.domain, []).append(_t.task_int)


class MREMetric:
    """Mean Radial Error metric with domain-balanced aggregation.

    Aggregation matches the organizer baseline (baseline/evaluate.py):
    per-landmark → per-image mean → per-task mean → domain-balanced overall.
    """

    def __init__(self) -> None:
        self._image_mres: dict[int, list[Tensor]] = {}

    def reset(self) -> None:
        self._image_mres.clear()

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        task_id: int,
        mask: Tensor,
    ) -> None:
        """Accumulate per-image MRE for one batch of predictions.

        Args:
            pred: (N, K_task, 2) predicted landmarks.
            target: (N, K_task, 2) ground truth landmarks.
            task_id: integer task identifier (0-8).
            mask: (N, K_task) bool — only compute error where True.
        """
        pred = pred.float().detach()
        target = target.float().detach()

        radial = torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12)  # (N, K)
        valid = mask.bool()

        # Per-image mean: each image contributes equally regardless of valid count
        for i in range(pred.shape[0]):
            if valid[i].any():
                img_mre = radial[i][valid[i]].mean()
                self._image_mres.setdefault(task_id, []).append(img_mre.cpu())

    def compute(self) -> dict[str, float]:
        """Compute per-task, per-domain, and overall MRE.

        Returns:
            {
                "task/{name}": float,    # per-task MRE
                "domain/{name}": float,  # per-domain MRE (mean of task MREs)
                "overall": float,        # domain-balanced overall
            }
        """
        result: dict[str, float] = {}
        task_mres: dict[int, float] = {}

        for task_int, mre_list in self._image_mres.items():
            if not mre_list:
                continue
            # Per-task MRE = mean of per-image MREs (organizer convention)
            mre = torch.stack(mre_list).mean().item()
            task_name = int_to_task_id(task_int)
            result[f"task/{task_name}"] = mre
            task_mres[task_int] = mre

        # Domain-level: mean of constituent task MREs
        domain_mres: list[float] = []
        for domain, task_ints in _DOMAIN_TASKS.items():
            domain_task_vals = [task_mres[t] for t in task_ints if t in task_mres]
            if domain_task_vals:
                domain_mre = sum(domain_task_vals) / len(domain_task_vals)
                result[f"domain/{domain}"] = domain_mre
                domain_mres.append(domain_mre)

        if domain_mres:
            result["overall"] = sum(domain_mres) / len(domain_mres)

        # Unweighted mean over tasks — the formula the platform's ingestion
        # script prints. Kept alongside `overall` because the two disagree by a
        # stable ~1.9 (domain balancing gives each cardiac task 1/12 and each
        # intrapartum task 1/6), so neither substitutes for the other.
        if task_mres:
            result["simple"] = sum(task_mres.values()) / len(task_mres)
            result["worst_task"] = max(task_mres.values())

        return result


class ParamMAEMetric:
    """Per-parameter Mean Absolute Error in native units.

    Units follow the caller's landmark space: with original-pixel landmarks
    (what `module._accumulate_metrics` now passes) distances and circumferences
    are pixels. Angles are converted to **degrees** here, matching how the
    challenge reports them.

    Non-finite pairs (invalid landmarks, degenerate geometry) are excluded from
    the mean AND counted, so a model cannot appear to improve by shrinking its
    own denominator.

    Deliberately emits **no cross-parameter aggregate**. The previous
    `param_overall` averaged pixel lengths, ellipse circumferences and radians
    together, which is a sum of incommensurable quantities. The challenge
    normalizes each parameter by clinical tolerance or IQR
    (`docs/evaluation.md`), and those ranges are unpublished — so no faithful
    scalar is available here, and inventing one invites exactly the kind of
    false confidence this metric already caused once.

    Keys out:
        param/{name}          — MAE in px (lengths) or degrees (angles)
        param_invalid/{name}  — fraction of pairs excluded as non-finite
    """

    # Registry says which parameters are angles; only AOP_angle today.
    _ANGLE_PARAMS = frozenset(p.name for p in PARAMS.values() if p.formula_type == "angle")

    def __init__(self) -> None:
        self._errors: dict[str, list[Tensor]] = {}
        self._n_seen: dict[str, int] = {}
        self._n_valid: dict[str, int] = {}

    def reset(self) -> None:
        self._errors.clear()
        self._n_seen.clear()
        self._n_valid.clear()

    def update(
        self,
        pred_params: dict[str, Tensor],
        gt_params: dict[str, Tensor],
    ) -> None:
        """Accumulate absolute errors for each parameter.

        Args:
            pred_params: {param_name: (N,)} predicted values (radians for angles).
            gt_params: {param_name: (N,)} ground truth values (radians for angles).
        """
        for name in pred_params:
            if name not in gt_params:
                continue
            pred = pred_params[name].float().detach()
            gt = gt_params[name].float().detach()

            if name in self._ANGLE_PARAMS:
                pred = torch.rad2deg(pred)
                gt = torch.rad2deg(gt)

            # isfinite, not just isnan: +/-inf would otherwise poison the mean.
            valid = torch.isfinite(pred) & torch.isfinite(gt)
            self._n_seen[name] = self._n_seen.get(name, 0) + int(valid.numel())
            self._n_valid[name] = self._n_valid.get(name, 0) + int(valid.sum().item())

            if valid.any():
                ae = (pred[valid] - gt[valid]).abs()
                self._errors.setdefault(name, []).append(ae.cpu())

    def compute(self) -> dict[str, float]:
        """Compute per-parameter MAE and invalid rate.

        Returns:
            {"param/{name}": float, "param_invalid/{name}": float}
        """
        result: dict[str, float] = {}

        for name, error_list in self._errors.items():
            if not error_list:
                continue
            result[f"param/{name}"] = torch.cat(error_list).mean().item()

        for name, seen in self._n_seen.items():
            if seen == 0:
                continue
            result[f"param_invalid/{name}"] = 1.0 - self._n_valid.get(name, 0) / seen

        return result


# ---------------------------------------------------------------------------
# IoU utility
# ---------------------------------------------------------------------------


def _paired_iou(pred_cxcywh: Tensor, gt_cxcywh: Tensor) -> Tensor:
    """Paired IoU between N predicted and N GT boxes in cxcywh format.

    Returns:
        (N,) IoU values, clamped to [0, 1].
    """

    # cxcywh → xyxy
    def _to_xyxy(b: Tensor) -> Tensor:
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    p = _to_xyxy(pred_cxcywh)
    g = _to_xyxy(gt_cxcywh)

    x1 = torch.max(p[:, 0], g[:, 0])
    y1 = torch.max(p[:, 1], g[:, 1])
    x2 = torch.min(p[:, 2], g[:, 2])
    y2 = torch.min(p[:, 3], g[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = area_p + area_g - inter

    return inter / union.clamp(min=1e-7)


class BBoxPrecisionMetric:
    """Bbox localization precision at multiple IoU thresholds.

    Tracks IoU of matched (pred, GT) bbox pairs per task.
    compute() returns mean IoU + Precision@threshold, by task and overall.
    """

    def __init__(self, thresholds: tuple[float, ...] = (0.5, 0.75)) -> None:
        self.thresholds = thresholds
        self._ious: dict[int, list[Tensor]] = {}

    def reset(self) -> None:
        self._ious.clear()

    def update(
        self,
        pred_cxcywh: Tensor,
        gt_cxcywh: Tensor,
        task_id: int,
    ) -> None:
        """Accumulate IoU for matched bbox pairs.

        Args:
            pred_cxcywh: (N, 4) matched predicted boxes.
            gt_cxcywh: (N, 4) matched GT boxes.
            task_id: integer task identifier (0-8).
        """
        if pred_cxcywh.shape[0] == 0:
            return
        iou = _paired_iou(
            pred_cxcywh.float().detach(),
            gt_cxcywh.float().detach(),
        )
        self._ious.setdefault(task_id, []).append(iou.cpu())

    def compute(self) -> dict[str, float]:
        """Compute per-task and overall bbox metrics.

        Returns:
            {
                "bbox/task/{name}/mean_iou": float,
                "bbox/task/{name}/P@0.5": float,
                "bbox/task/{name}/P@0.75": float,
                "bbox/overall/mean_iou": float,
                "bbox/overall/P@0.5": float,
                "bbox/overall/P@0.75": float,
            }
        """
        result: dict[str, float] = {}
        task_mean_ious: dict[int, float] = {}
        task_precs: dict[float, list[float]] = {t: [] for t in self.thresholds}

        for task_int, iou_list in self._ious.items():
            all_ious = torch.cat(iou_list)
            task_name = int_to_task_id(task_int)

            mean_iou = all_ious.mean().item()
            result[f"bbox/task/{task_name}/mean_iou"] = mean_iou
            task_mean_ious[task_int] = mean_iou

            for t in self.thresholds:
                prec = (all_ious >= t).float().mean().item()
                result[f"bbox/task/{task_name}/P@{t}"] = prec
                task_precs[t].append(prec)

        # Overall: task-balanced average
        if task_mean_ious:
            result["bbox/overall/mean_iou"] = sum(task_mean_ious.values()) / len(task_mean_ious)
            for t in self.thresholds:
                if task_precs[t]:
                    result[f"bbox/overall/P@{t}"] = sum(task_precs[t]) / len(task_precs[t])

        return result

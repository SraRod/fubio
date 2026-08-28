"""Loss functions for multi-task biometry training.

bbox_loss: L1 + CIoU on matched query-GT pairs.
conf_focal_loss: binary focal loss — matched queries are positive, rest negative.
landmark_loss: L1 on per-task landmarks.
heatmap_loss: soft cross-entropy vs Gaussian target on the affinity distribution.
param_loss: scale-normalized L1 on derived clinical parameters.
mil_loss: MIL bag-level loss for task-known unlabeled images.
equivariance_loss: landmark consistency under known geometric transforms.
pseudo_label_loss: confidence-weighted L1 distillation from the EMA teacher.
geometric_constraint_loss + the ellipse/chamber terms: anatomy validity penalties.
shape_consistency_loss: residual outside the PCA shape subspace.

Upstream: train/matcher.py (per-task matched indices), evaluation/geometry.py,
          models/coord_predictors.py (normalized_grid — shared grid convention).
Downstream: train/module.py (training_step composes all losses).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from fubio.evaluation.geometry import compute_distance, compute_params_for_training
from fubio.models.coord_predictors import normalized_grid

# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """(*, 4) cxcywh → xyxy."""
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _paired_ciou(boxes1_cxcywh: Tensor, boxes2_cxcywh: Tensor) -> Tensor:
    """Complete IoU between paired boxes (same N, element-wise).

    Args:
        boxes1_cxcywh: (N, 4) predicted, cxcywh format.
        boxes2_cxcywh: (N, 4) ground truth, cxcywh format.

    Returns:
        (N,) CIoU values in [-1, 1].
    """
    b1 = _cxcywh_to_xyxy(boxes1_cxcywh)
    b2 = _cxcywh_to_xyxy(boxes2_cxcywh)

    x1 = torch.max(b1[:, 0], b2[:, 0])
    y1 = torch.max(b1[:, 1], b2[:, 1])
    x2 = torch.min(b1[:, 2], b2[:, 2])
    y2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)

    ex1 = torch.min(b1[:, 0], b2[:, 0])
    ey1 = torch.min(b1[:, 1], b2[:, 1])
    ex2 = torch.max(b1[:, 2], b2[:, 2])
    ey2 = torch.max(b1[:, 3], b2[:, 3])
    c_diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

    d_sq = (boxes1_cxcywh[:, 0] - boxes2_cxcywh[:, 0]) ** 2 + (
        boxes1_cxcywh[:, 1] - boxes2_cxcywh[:, 1]
    ) ** 2

    pred_w, pred_h = boxes1_cxcywh[:, 2], boxes1_cxcywh[:, 3]
    gt_w, gt_h = boxes2_cxcywh[:, 2], boxes2_cxcywh[:, 3]
    v = (4.0 / (math.pi**2)) * (
        torch.atan(gt_w / gt_h.clamp(min=1e-7)) - torch.atan(pred_w / pred_h.clamp(min=1e-7))
    ) ** 2
    alpha = v / ((1.0 - iou) + v + 1e-7)

    return iou - d_sq / c_diag_sq - alpha * v


# ---------------------------------------------------------------------------
# Bbox loss
# ---------------------------------------------------------------------------


def bbox_loss(
    pred_boxes: Tensor,
    gt_boxes: Tensor,
) -> dict[str, Tensor]:
    """L1 + CIoU loss on matched bbox pairs.

    Args:
        pred_boxes: (N, 4) matched predicted boxes, cxcywh normalized.
        gt_boxes: (N, 4) matched GT boxes, cxcywh normalized.

    Returns:
        {"loss_box_l1": Tensor, "loss_box_ciou": Tensor}
    """
    if pred_boxes.shape[0] == 0:
        zero = torch.tensor(0.0, device=pred_boxes.device)
        return {"loss_box_l1": zero, "loss_box_ciou": zero}

    loss_l1 = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")
    ciou = _paired_ciou(pred_boxes, gt_boxes)
    loss_ciou = (1.0 - ciou).mean()

    return {"loss_box_l1": loss_l1, "loss_box_ciou": loss_ciou}


# ---------------------------------------------------------------------------
# Confidence focal loss
# ---------------------------------------------------------------------------


def conf_focal_loss(
    pred_conf: Tensor,
    target_conf: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Binary focal loss for instance confidence.

    Args:
        pred_conf: (N,) raw logits — positive for matched, negative for unmatched.
        target_conf: (N,) float — 1.0 for matched queries, 0.0 for unmatched.
        alpha: weighting factor for positives.
        gamma: focusing parameter.

    Returns:
        Scalar focal loss.
    """
    if pred_conf.shape[0] == 0:
        return torch.tensor(0.0, device=pred_conf.device)

    p = pred_conf.sigmoid()
    bce = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="none")

    # Focal modulation
    p_t = p * target_conf + (1 - p) * (1 - target_conf)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * target_conf + (1 - alpha) * (1 - target_conf)

    return (alpha_t * focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Landmark loss
# ---------------------------------------------------------------------------


def landmark_loss(
    pred_xy: Tensor,
    gt_xy: Tensor,
    mask: Tensor,
) -> Tensor:
    """Landmark L1 loss — per-element mean over visible landmarks.

    Plain L1 (constant gradient) on purpose: coords live in [0,1], where
    SmoothL1's quadratic regime (error << beta always) shrinks the gradient
    as predictions approach the target.

    Args:
        pred_xy: (N, K_task, 2) predicted normalized coords.
        gt_xy: (N, K_task, 2) ground truth normalized coords.
        mask: (N, K_task) bool — only supervised & visible landmarks.

    Returns:
        Scalar loss.
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred_xy.device, dtype=pred_xy.dtype)

    mask_2d = mask.unsqueeze(-1).expand_as(pred_xy)
    return F.l1_loss(pred_xy[mask_2d], gt_xy[mask_2d], reduction="mean")


def heatmap_loss(
    pred_heat: Tensor,
    gt_xy: Tensor,
    mask: Tensor,
    spatial_shape: tuple[int, int],
    sigma_cells: float = 1.5,
) -> Tensor:
    """Soft cross-entropy between the affinity distribution and a Gaussian target.

    Dense spatial supervision on HeatmapPredictor's distribution — this is what
    forces it single-peaked so soft-argmax reduces to the true landmark rather
    than a midpoint between spurious peaks. Supervises only masked (visible &
    supervised) landmarks, so missing labels contribute no signal.

    Args:
        pred_heat: (N, K, S) softmax distribution over the S=Hp*Wp patch grid.
        gt_xy: (N, K, 2) GT normalized coords in [0,1].
        mask: (N, K) bool — supervised & visible landmarks.
        spatial_shape: (Hp, Wp) patch grid; grid convention shared with predictor.
        sigma_cells: Gaussian width in grid cells (resolution-relative).

    Returns:
        Scalar loss.
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred_heat.device, dtype=pred_heat.dtype)

    hp, wp = spatial_shape
    xs, ys = normalized_grid(hp, wp, pred_heat.device)  # each (S,)

    gt_x = gt_xy[..., 0:1]  # (N, K, 1)
    gt_y = gt_xy[..., 1:2]

    # Anisotropic sigma in normalized units so σ is a fixed number of cells
    # regardless of grid aspect ratio.
    sx = sigma_cells / wp
    sy = sigma_cells / hp
    d2 = ((xs - gt_x) / sx) ** 2 + ((ys - gt_y) / sy) ** 2  # (N, K, S)
    target = torch.exp(-0.5 * d2)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    log_pred = (pred_heat.clamp(min=1e-12)).log()
    ce = -(target * log_pred).sum(dim=-1)  # (N, K)
    return ce[mask].mean()


def shape_consistency_loss(
    pred_lm: Tensor,
    mask: Tensor,
    canonical_mean: Tensor,
    canonical_basis: Tensor,
) -> Tensor:
    """Penalize predicted shapes that leave the canonical PCA shape subspace.

    Center-and-scale regularizer: normalizes each predicted instance to match
    the convention used when fitting the PCA basis (centroid removal + unit
    Frobenius norm), then penalizes the component orthogonal to the shape
    subspace. No rotation alignment — the PCA basis was fit without it, so
    rotational variance is captured by the leading components. Only
    fully-visible instances participate (needs all K points).

    Args:
        pred_lm: (N, K, 2) predicted normalized coords.
        mask: (N, K) bool — supervised & visible.
        canonical_mean: (K, 2) centered, scale-normalized mean shape.
        canonical_basis: (Mc, 2K) shape subspace rows.

    Returns:
        Scalar loss (0 if no fully-visible instance).
    """
    full = mask.all(dim=1)
    if not full.any():
        return pred_lm.new_tensor(0.0)

    p = pred_lm[full]  # (Nf, K, 2)

    # Center + scale normalize (matches build_shape_prior's _procrustes_align)
    centered = p - p.mean(dim=1, keepdim=True)
    scale = centered.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = centered / scale.unsqueeze(-1)  # (Nf, K, 2)

    # PCA projection — penalize off-subspace residual
    v = normed.flatten(1) - canonical_mean.flatten()  # (Nf, 2K)
    proj = v @ canonical_basis.T  # (Nf, Mc)
    recon = proj @ canonical_basis  # (Nf, 2K)
    resid = v - recon
    return (resid**2).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Parameter loss
# ---------------------------------------------------------------------------


def param_loss(
    pred_landmarks: Tensor,
    gt_landmarks: Tensor,
    task_id: str,
    visible_mask: Tensor,
) -> Tensor:
    """Clinical parameter loss: scale-normalized L1 between derived params.

    Each parameter's L1 is divided by the GT magnitude, making the loss
    approximately relative error. This ensures commensurability across formula
    types: AOP_angle (~1.5 rad), HC_circumference (~1.2), and distances (~0.1)
    all contribute proportionally to their relative prediction quality.

    Args:
        pred_landmarks: (N, K_task, 2) predicted landmarks.
        gt_landmarks: (N, K_task, 2) GT landmarks.
        task_id: task string id (e.g. "HC", "A4C").
        visible_mask: (N, K_task) bool — landmark visibility.

    Returns:
        Scalar loss (0 if task has no params or all params are NaN).
    """
    pred_params = compute_params_for_training(pred_landmarks, task_id, visible_mask)
    gt_params = compute_params_for_training(gt_landmarks, task_id, visible_mask)

    if not pred_params:
        return torch.tensor(0.0, device=pred_landmarks.device)

    losses: list[Tensor] = []
    for name in pred_params:
        pred_val = pred_params[name]
        gt_val = gt_params[name]
        valid = ~torch.isnan(pred_val) & ~torch.isnan(gt_val)
        if not valid.any():
            continue
        scale = gt_val[valid].abs().mean().clamp(min=0.01)
        losses.append(F.l1_loss(pred_val[valid], gt_val[valid], reduction="mean") / scale)

    if not losses:
        return torch.tensor(0.0, device=pred_landmarks.device)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Pseudo-label loss (semi-supervised, EMA Teacher)
# ---------------------------------------------------------------------------


def pseudo_label_loss(
    student_lm: Tensor,
    pseudo_target: Tensor,
    teacher_conf: Tensor,
    pseudo_valid: Tensor,
) -> Tensor:
    """Confidence-weighted L1 between student landmarks and EMA teacher pseudo-targets.

    Fixed denominator (per_coord.numel()) ensures absolute confidence
    attenuation: a batch of all-low-confidence slots produces proportionally
    small loss, not a normalized-away identical loss.

    L1 kernel matches supervised landmark_loss at beta=0.0.

    Args:
        student_lm: (N, n_inst, K, 2) student predicted landmarks.
        pseudo_target: (N, n_inst, K, 2) teacher landmarks projected through
            the known affine transform.
        teacher_conf: (N, n_inst) or (N, n_inst, 1) raw confidence logits
            from the teacher. Detached inside — teacher conf must not receive
            gradient from this loss.
        pseudo_valid: (N, n_inst, K) bool — True for landmarks that remain
            inside [0,1] after affine projection. Out-of-frame landmarks
            contribute zero loss.

    Returns:
        Scalar loss (0 if no valid landmarks).
    """
    if student_lm.numel() == 0:
        return torch.tensor(0.0, device=student_lm.device)

    w = teacher_conf.detach().squeeze(-1).sigmoid()  # (N, n_inst)
    per_coord = (student_lm - pseudo_target).abs()  # (N, n_inst, K, 2)
    mask = pseudo_valid.float() * w.unsqueeze(-1)  # (N, n_inst, K)
    weighted = mask.unsqueeze(-1) * per_coord  # (N, n_inst, K, 2)
    return weighted.sum() / per_coord.numel()


# ---------------------------------------------------------------------------
# MIL loss (semi-supervised)
# ---------------------------------------------------------------------------


def mil_loss(task_conf_logits: Tensor) -> Tensor:
    """Multiple Instance Learning bag-level loss for task-known unlabeled images.

    Task-known unlabeled data: we know the task is present in the image (from
    the filename/folder structure) but have no landmark annotations. L_mil
    requires at least one query slot to be confident — preventing the model
    from learning to suppress all detections for tasks it has seen but not
    been annotated for. Without this, the conf_focal_loss (L_absent) would
    push all queries toward zero for these images, teaching the model that
    the task is absent when it actually isn't.

    L_mil = -log(σ(max(logits)))   [numerically stable via F.logsigmoid]

    Args:
        task_conf_logits: (N, N_inst) raw confidence logits for one task over
            the N images eligible for it — all query slots, no reduction over
            slots beforehand (the max IS the bag operator).

    Returns:
        Scalar mean over images (0 if empty input).
    """
    if task_conf_logits.numel() == 0:
        return torch.tensor(0.0, device=task_conf_logits.device)
    return -F.logsigmoid(task_conf_logits.max(dim=-1).values).mean()


def equivariance_loss(
    lm_ref: Tensor,
    lm_trans: Tensor,
    matrix_fwd: Tensor,
    conf_ref: Tensor,
) -> Tensor:
    """Equivariance consistency: predicted landmarks should transform with the image.

    Given reference landmarks L(x) and transformed-view landmarks L(T(x)),
    the loss is ||L(T(x)) - T(L(x))||₂ weighted by reference confidence.
    Confidence is detached — EC should not suppress confidence to reduce
    its own loss.

    Slot alignment: DETR-style slots are permutation-equivariant, and
    augmentation can swap which slot claims a detection. Detached Hungarian
    matching aligns slots between views before computing consistency — without
    this, a slot swap becomes a false penalty that can drive collapse.

    Args:
        lm_ref: (B, n_inst, K, 2) reference view landmarks in [0,1].
        lm_trans: (B, n_inst, K, 2) transformed view landmarks in [0,1].
        matrix_fwd: (B, 2, 3) forward affine in [0,1] space.
        conf_ref: (B, n_inst, 1) or (B, n_inst) raw logits — detached inside.

    Returns:
        Scalar confidence-weighted L2 (weighted sum / weight sum).
    """
    from fubio.train.views import warp_landmarks

    if lm_ref.numel() == 0:
        return torch.tensor(0.0, device=lm_ref.device)

    B, N, K, _ = lm_ref.shape

    # Map reference landmarks through the known transform.
    # Detach: ref is the pseudo-target, only the transformed view learns.
    lm_ref_mapped = warp_landmarks(lm_ref.detach(), matrix_fwd)

    # Slot alignment via detached Hungarian matching on landmark distance.
    # Cost: per-slot mean L1 between mapped-ref and trans. Detached so
    # matching does not contribute gradients.
    with torch.no_grad():
        # (B, N_ref, N_trans) cost matrix
        cost = (lm_ref_mapped.unsqueeze(2) - lm_trans.unsqueeze(1)).abs().sum(-1).mean(-1)
        # Greedy argmin per ref slot (N is small, typically 2-4)
        perm = cost.argmin(dim=-1)  # (B, N) — which trans slot matches each ref slot

    # Gather aligned trans landmarks
    perm_exp = perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 2)
    lm_trans_aligned = lm_trans.gather(1, perm_exp)

    # Per-keypoint L2 distance
    l2 = (lm_trans_aligned - lm_ref_mapped).pow(2).sum(dim=-1).sqrt()  # (B, N, K)

    # Confidence weighting (detached — EC must not learn to suppress conf).
    # Fixed denominator: l2.numel() ensures absolute confidence attenuation.
    # A batch of all-low-confidence slots produces a proportionally small loss,
    # not a normalized-away identical loss (the R22 bug).
    w = conf_ref.detach().squeeze(-1).sigmoid()  # (B, N)
    weighted = l2 * w.unsqueeze(-1)  # (B, N, K)
    return weighted.sum() / l2.numel()


# ---------------------------------------------------------------------------
# Geometric constraint losses (non-official — structural priors, no GT target)
#
# These encode anatomical invariants that the official metric doesn't score
# directly but whose violation produces clinically implausible geometry.
# Managed separately from ParamDef because they have no GT reference —
# targets are fixed geometric constraints (orthogonality, ordering, angle
# bounds).
# ---------------------------------------------------------------------------

_GEO_EPS = 1e-7

# Axis shorter than this (in normalized coords) is degenerate — angular
# constraints are meaningless and ordering is noise.
_DEGEN_LEN = 1e-4

_COS_70 = math.cos(math.radians(70.0))


def ellipse_orthogonality_loss(pred_landmarks: Tensor) -> Tensor:
    """Penalize deviation from 90° between the two axes of an ellipse (HC/FA).

    cos²θ is zero at perfect orthogonality, max 1.0 at parallel.
    Degenerate axes (either shorter than _DEGEN_LEN) are excluded.

    Landmarks: [0, 1] = axis a; [2, 3] = axis b.

    Args:
        pred_landmarks: (N, K_total, 2) predicted landmarks for one task.

    Returns:
        Scalar loss (0 if all instances are degenerate).
    """
    pts = pred_landmarks.float()
    u = pts[:, 1] - pts[:, 0]
    v = pts[:, 3] - pts[:, 2]
    norm_u = torch.sqrt((u**2).sum(dim=-1) + _GEO_EPS)
    norm_v = torch.sqrt((v**2).sum(dim=-1) + _GEO_EPS)

    valid = (norm_u > _DEGEN_LEN) & (norm_v > _DEGEN_LEN)
    if not valid.any():
        return pred_landmarks.new_tensor(0.0)

    cos_theta = (u * v).sum(dim=-1) / (norm_u * norm_v)
    return (cos_theta[valid] ** 2).mean()


def ellipse_axis_ordering_loss(pred_landmarks: Tensor) -> Tensor:
    """Penalize when the short axis (P2-P3) exceeds the long axis (P2-P3 > P0-P1).

    HC GT: P2-P3 is always the longer axis, P0-P1 the shorter.
    This loss enforces ‖P2-P3‖ ≥ ‖P0-P1‖ via bounded relative excess.

    L = mean(ReLU((a - b) / (a + b))²) where a = ‖P0-P1‖, b = ‖P2-P3‖.
    Only active when a > b (short axis incorrectly exceeds long axis).

    Args:
        pred_landmarks: (N, K_total, 2) predicted landmarks for one task.

    Returns:
        Scalar loss (0 if ordering is correct or axes are degenerate).
    """
    a = compute_distance(pred_landmarks[:, 0], pred_landmarks[:, 1])  # short axis
    b = compute_distance(pred_landmarks[:, 2], pred_landmarks[:, 3])  # long axis
    denom = (a + b).clamp(min=_DEGEN_LEN)
    # Penalize when a > b (short axis incorrectly longer)
    return F.relu((a - b) / denom).square().mean()


def chamber_angle_loss(pred_landmarks: Tensor) -> Tensor:
    """Penalize A4C chamber angles below 70° (acute angle between 上下 and 左右 axes).

    Works in cosine space to avoid arccos and its unstable gradient.
    |cos θ| > cos 70° ↔ acute angle < 70°, so we penalize the excess.
    Degenerate axes are excluded.

    Landmarks per chamber: [base, base+1] = 上下 axis, [base+2, base+3] = 左右 axis.
    Bases: LV=0, RV=4, LA=8, RA=12.

    Args:
        pred_landmarks: (N, K_total, 2) predicted landmarks for one task.

    Returns:
        Scalar loss (0 if all chambers ≥ 70° or degenerate).
    """
    pts = pred_landmarks.float()
    losses: list[Tensor] = []

    for base in (0, 4, 8, 12):
        u = pts[:, base + 1] - pts[:, base]
        v = pts[:, base + 3] - pts[:, base + 2]
        norm_u = torch.sqrt((u**2).sum(dim=-1) + _GEO_EPS)
        norm_v = torch.sqrt((v**2).sum(dim=-1) + _GEO_EPS)

        valid = (norm_u > _DEGEN_LEN) & (norm_v > _DEGEN_LEN)
        if not valid.any():
            losses.append(pts.new_tensor(0.0))
            continue

        cos_theta = (u * v).sum(dim=-1) / (norm_u * norm_v)
        # |cos θ| > cos 70° means acute angle < 70°
        violation = F.relu(cos_theta[valid].abs() - _COS_70)
        losses.append(violation.square().mean())

    return torch.stack(losses).mean()


def chamber_angle_sign_loss(pred_landmarks: Tensor, gt_landmarks: Tensor) -> Tensor:
    """Penalize A4C chambers whose predicted angle crosses 90° relative to GT.

    For each chamber, if GT's 上下-左右 angle is acute (cos > 0), a predicted
    obtuse angle (cos < 0) is penalized, and vice versa. Small deviations on
    the same side of 90° produce zero loss from this term.

    Works in cosine space: penalty = relu(-cos_gt * cos_pred)².
    When signs agree (same side of 90°), the product is positive → relu = 0.
    When signs disagree (crossing 90°), the product is negative → quadratic penalty.

    Args:
        pred_landmarks: (N, K_total, 2) predicted landmarks.
        gt_landmarks: (N, K_total, 2) ground-truth landmarks.

    Returns:
        Scalar loss (0 if no chambers cross or all degenerate).
    """
    pts_pred = pred_landmarks.float()
    pts_gt = gt_landmarks.float()
    losses: list[Tensor] = []

    for base in (0, 4, 8, 12):
        u_gt = pts_gt[:, base + 1] - pts_gt[:, base]
        v_gt = pts_gt[:, base + 3] - pts_gt[:, base + 2]
        u_pred = pts_pred[:, base + 1] - pts_pred[:, base]
        v_pred = pts_pred[:, base + 3] - pts_pred[:, base + 2]

        norm_u_gt = torch.sqrt((u_gt**2).sum(dim=-1) + _GEO_EPS)
        norm_v_gt = torch.sqrt((v_gt**2).sum(dim=-1) + _GEO_EPS)
        norm_u_pred = torch.sqrt((u_pred**2).sum(dim=-1) + _GEO_EPS)
        norm_v_pred = torch.sqrt((v_pred**2).sum(dim=-1) + _GEO_EPS)

        valid = (
            (norm_u_gt > _DEGEN_LEN)
            & (norm_v_gt > _DEGEN_LEN)
            & (norm_u_pred > _DEGEN_LEN)
            & (norm_v_pred > _DEGEN_LEN)
        )
        if not valid.any():
            losses.append(pts_pred.new_tensor(0.0))
            continue

        cos_gt = (u_gt * v_gt).sum(dim=-1) / (norm_u_gt * norm_v_gt)
        cos_pred = (u_pred * v_pred).sum(dim=-1) / (norm_u_pred * norm_v_pred)
        crossing = F.relu(-cos_gt[valid] * cos_pred[valid])
        losses.append(crossing.square().mean())

    return torch.stack(losses).mean()


def geometric_constraint_loss(
    pred_landmarks: Tensor,
    task_id: str,
    gt_landmarks: Tensor | None = None,
) -> dict[str, Tensor]:
    """Task-specific geometric constraint losses — dispatcher.

    Returns per-component dict for separate logging. The caller sums for the
    total. HC gets ortho + ordering; FA gets ortho only (near-circular, axis
    ordering is ambiguous in GT); A4C gets chamber angle + angle sign.

    Args:
        pred_landmarks: (N, K_total, 2) predicted landmarks for one task.
        task_id: task string id.
        gt_landmarks: (N, K_total, 2) ground-truth landmarks. Required for
            angle_sign (A4C only); ignored for other tasks.

    Returns:
        {component_name: scalar loss}. Empty dict for tasks with no constraints.
    """
    if task_id == "HC":
        return {
            "ortho": ellipse_orthogonality_loss(pred_landmarks),
            "axis_order": ellipse_axis_ordering_loss(pred_landmarks),
        }
    if task_id == "FA":
        # FA is near-circular (ratio ~0.96); 33% of GT has P0-P1 > P2-P3.
        # Ordering constraint would fight 1/3 of GT, so ortho only.
        return {
            "ortho": ellipse_orthogonality_loss(pred_landmarks),
        }
    if task_id == "A4C":
        result: dict[str, Tensor] = {
            "chamber_angle": chamber_angle_loss(pred_landmarks),
        }
        if gt_landmarks is not None:
            result["angle_sign"] = chamber_angle_sign_loss(pred_landmarks, gt_landmarks)
        return result
    return {}

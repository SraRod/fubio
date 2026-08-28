"""FUBioModule — LightningModule for single-pass multi-task training.

training_step: forward → per-task matching → bbox/conf/landmark/param losses.
validation_step: same forward + losses + metric accumulation.

Semi-supervised loss framework — two data tiers with distinct supervision:

  | Tier       | L_pos | L_suppress | L_absent | L_mil | L_eq  |
  |------------|-------|------------|----------|-------|-------|
  | Labeled    |   ✓   | ✓ (0.05)   | ✓ (0.0)  |   —   |   —   |
  | Task-known |   —   |     —      | ✓ (0.0)  |   ✓   |   ✓   |

Every unlabeled image in this dataset carries a task id (it comes from the
folder structure), so a fully task-unknown tier does not arise and is not
modelled.

Why separate tiers:
- ~0.1% of labeled images are split-screen (one side annotated, other side
  absent from GT) → correct detections on the unlabeled side get penalized
  as false positives. L_suppress softens this via label smoothing.
- 191K task-known unlabeled images need semi-supervised signal. L_mil
  provides positive bag-level supervision; L_absent on other tasks gives
  clean negative supervision (cross-task absence is certain). On its OWN
  task an unlabeled image gets neither — it has no match, so the row is
  dropped from the confidence loss entirely rather than defaulting to a
  zero target that L_mil would then have to fight.
- L_eq (equivariance): landmark consistency under random affine transforms.
  Computed in _equivariance_step via a second forward on warped unlabeled
  images. Weighted by reference confidence (detached). Needs kornia.

The two tiers arrive as separate batches (train/datamodule.py CombinedLoader)
and take separate passes through _step, distinguished by `stage`. Keeping them
apart means the unlabeled pass contributes confidence terms only: every other
loss is either structurally inert without GT or explicitly gated off, so a
semi-supervised run differs from its supervised anchor by the semi-supervised
terms and nothing else.

Upstream: models/model.py (FUBioModel), train/matcher.py, train/losses.py,
          evaluation/metrics.py, train/config.py.
Downstream: train/train.py (CLI entry point).
"""

from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from fubio.data.supportive_torch import compute_supportive_torch
from fubio.data.task_registry import TASKS
from fubio.evaluation.geometry import compute_params_for_evaluation
from fubio.evaluation.metrics import BBoxPrecisionMetric, MREMetric, ParamMAEMetric
from fubio.evaluation.postprocessing import select_serving_query
from fubio.models.heads import derive_bbox_masked
from fubio.models.model import FUBioModel, ModelOutput
from fubio.train.config import (
    ExperimentConfig,
    GeoSimCCCoordConfig,
    HeadTuneConfig,
    HeatmapCoordConfig,
    OptimizerConfig,
    ShapePriorCoordConfig,
    ShapeSimCCCoordConfig,
    SimCCCoordConfig,
)
from fubio.train.losses import (
    bbox_loss,
    conf_focal_loss,
    conf_ranking_loss,
    equivariance_loss,
    geometric_constraint_loss,
    heatmap_loss,
    instance_repulsion_loss,
    landmark_loss,
    mil_loss,
    ortho_regularization,
    param_loss,
    pseudo_label_loss,
    shape_consistency_loss,
    shape_residual_loss,
    simcc_loss,
)
from fubio.train.matcher import PerTaskMatcher
from fubio.train.schedule import PhaseScheduler

logger = logging.getLogger(__name__)


class FUBioModule(L.LightningModule):
    """Single-pass multi-task training: forward → match → losses.

    Config drives all hyperparameters. Model is constructed from config
    in __init__ (not passed in) to keep serialization clean.

    Upstream: models/model.py, train/losses.py, train/matcher.py.
    Downstream: train/train.py.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())

        self.config = config

        # GPU normalize: move ImageNet mean/std from CPU (transforms.py) to GPU
        self.register_buffer(
            "_img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self._miro_enabled = config.miro.lambda_miro > 0

        # Load shape prior when any component needs it:
        # - ShapePriorPredictor: mean/basis for PCA decomposition
        # - HeatmapPredictor with data_prior: mean_xy/std_xy for spatial prior
        # - [P1: Anchor query position] TaskModule: mean_xy for anchor init
        # - [P2: SimCC readout] SimCCPredictor doesn't need it, but anchors do
        #
        # Staleness is NOT checked here on purpose. load_from_checkpoint re-runs this
        # __init__ during serving, where no manifest exists (Docker) and where the
        # checkpoint's own buffers override whatever this reads anyway. The manifest
        # comparison lives in train.py (verify_shape_prior_provenance).
        shape_prior = None
        coord = config.head.coord
        needs_prior = isinstance(
            coord, (ShapePriorCoordConfig, ShapeSimCCCoordConfig, GeoSimCCCoordConfig)
        ) or (
            isinstance(coord, HeatmapCoordConfig) and coord.data_prior
        )
        if needs_prior:
            from pathlib import Path

            from fubio.data.shape_prior import ShapePrior

            prior_path = (
                Path(coord.prior_path)
                if isinstance(coord, ShapePriorCoordConfig)
                else config.loss.shape_prior_path
            )
            if not prior_path.exists():
                raise FileNotFoundError(
                    f"Shape prior not found: {prior_path}\n"
                    f"Build with: uv run python -m fubio.data.build_shape_prior"
                )
            shape_prior = ShapePrior.model_validate_json(prior_path.read_text())

        # [P1: Anchor query position] load shape prior for anchor init even if
        # the coord predictor doesn't need it (e.g. SimCC mode). Soft: if the
        # file doesn't exist, anchors fall back to zeros — no hard failure.
        if shape_prior is None:
            from pathlib import Path

            from fubio.data.shape_prior import ShapePrior

            anchor_path = config.loss.shape_prior_path
            if anchor_path.exists():
                shape_prior = ShapePrior.model_validate_json(anchor_path.read_text())

        _use_sup = config.loss.lambda_supportive > 0 or config.loss.lambda_evidence > 0
        self.model = FUBioModel(
            backbone_name=config.backbone.name,
            d_model=config.d_model,
            n_heads=config.decoder.n_heads,
            ffn_dim=config.decoder.ffn_dim,
            head_ffn_dim=config.head.ffn_dim,
            n_decoder_layers=config.decoder.n_layers,
            n_head_layers=config.head.n_layers,
            n_inst=config.head.n_inst,
            use_uncertainty=config.loss.use_uncertainty,
            derive_bbox=config.head.derive_bbox,
            use_affine=config.head.use_affine,
            return_intermediate=self._miro_enabled,
            coord_config=config.head.coord,
            shape_prior=shape_prior,
            neck_config=config.neck,
            neck_dropout=config.neck_dropout,
            dropout=config.decoder.dropout,
            conf_mlp_layers=config.head.conf_mlp_layers,
            input_size=max(config.backbone.input_size),
            use_supportive=_use_sup,
        )
        self.matcher = PerTaskMatcher(
            cost_conf=config.matcher.cost_conf,
            cost_box=config.matcher.cost_box,
            cost_ciou=config.matcher.cost_ciou,
            cost_land=config.matcher.cost_land,
        )

        # Canonical shape templates for shape_consistency_loss (heatmap mode
        # regularizer). Registered as buffers so they follow the model to device.
        self._canon_tasks: set[str] = set()
        if config.loss.lambda_shape_consistency > 0:
            from fubio.data.shape_prior import ShapePrior

            sp = shape_prior  # reuse if already loaded (data_prior / shape_prior mode)
            if sp is None:
                sp_path = config.loss.shape_prior_path
                if not sp_path.exists():
                    raise FileNotFoundError(
                        f"Shape prior not found: {sp_path}\n"
                        f"Build with: uv run python -m fubio.data.build_shape_prior"
                    )
                sp = ShapePrior.model_validate_json(sp_path.read_text())
            for tid, tp in sp.tasks.items():
                if tp.canonical_mean is None or tp.canonical_basis is None:
                    continue
                self.register_buffer(
                    f"_canon_mean_{tid}", torch.tensor(tp.canonical_mean, dtype=torch.float32)
                )
                self.register_buffer(
                    f"_canon_basis_{tid}", torch.tensor(tp.canonical_basis, dtype=torch.float32)
                )
                self._canon_tasks.add(tid)

        # MIRO: frozen backbone copy + variational encoders
        if self._miro_enabled:
            from copy import deepcopy

            from fubio.train.regularizer import MIROEncoders, build_miro_encoders

            self._frozen_backbone = deepcopy(self.model.backbone)
            self._frozen_backbone.eval()
            for p in self._frozen_backbone.parameters():
                p.requires_grad = False
            _input_w, _input_h = config.backbone.input_size
            self._miro_encoders: MIROEncoders = build_miro_encoders(
                self.model.backbone,
                (_input_h, _input_w),  # build_miro_encoders expects (H, W)
                config.miro,
            )

        self._train_mre = MREMetric()
        self._train_param_mae = ParamMAEMetric()
        self._train_bbox_prec = BBoxPrecisionMetric()
        self._val_mre = MREMetric()
        self._val_mre_pixel = MREMetric()
        self._val_param_mae = ParamMAEMetric()
        self._val_bbox_prec = BBoxPrecisionMetric()
        # Hungarian-selected MRE in pixel space. Diagnostic only — the gap to
        # val/mre_pixel measures confidence-head ranking error. Never select on it.
        self._oracle_mre = MREMetric()
        # Per-image minimum pairwise distance between instance slots' landmarks.
        self._inst_sep: list[Tensor] = []

    # ------------------------------------------------------------------
    # EMA Teacher (semi-supervised pseudo-labeling)
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Post-checkpoint setup: head-tune freeze/reinit + EMA teacher.

        Called by Lightning AFTER init_weights / ckpt_path has been applied
        to self.model, so the deepcopy inherits the loaded weights — not the
        random-init heads that __init__ constructs.
        """
        if stage != "fit":
            return

        # --- Head-tune: freeze everything, unfreeze selected tasks, reinit ---
        ht = self.config.head_tune
        if ht.enabled:
            self._apply_head_tune(ht)

        semi = self.config.semi
        if not semi.enabled or semi.lambda_pseudo <= 0:
            return

        from copy import deepcopy

        teacher = deepcopy(self.model)
        teacher.eval()
        teacher.requires_grad_(False)
        # Store via object.__setattr__ to AVOID nn.Module registration.
        # This keeps the teacher out of state_dict(), SWAD snapshots, and
        # recursive train()/eval() calls. Device placement is handled
        # manually in _ensure_teacher_device().
        object.__setattr__(self, "_teacher_model", teacher)
        object.__setattr__(self, "_ema_step", 0)
        # Cache param lists for foreach EMA (avoids rebuilding every step)
        object.__setattr__(self, "_teacher_params", list(teacher.parameters()))
        object.__setattr__(self, "_student_params", list(self.model.parameters()))

        # Photometric aug for student path (applied in [0,1] range)
        import kornia.augmentation as KA

        photo_aug = KA.ColorJiggle(
            brightness=semi.photo_brightness,
            contrast=semi.photo_contrast,
            p=1.0,
        )
        object.__setattr__(self, "_photo_aug", photo_aug)

        logger.info(
            "EMA Teacher created: alpha=%.4f, lambda_pseudo=%.2f, "
            "ramp=[%d, %d] epochs, photo=(b=%.2f, c=%.2f)",
            semi.alpha_ema, semi.lambda_pseudo,
            semi.ramp_start, semi.ramp_end,
            semi.photo_brightness, semi.photo_contrast,
        )

    def _apply_head_tune(self, ht: HeadTuneConfig) -> None:
        """Freeze shared components + non-tuned tasks; reinit selected tasks.

        Must run AFTER init_weights loads checkpoint — reinit would be
        overwritten otherwise, and freeze must see the loaded params.
        """
        # 1. Freeze entire model
        self.model.requires_grad_(False)

        # 2. Unfreeze selected TaskModules
        tune_set = set(ht.tune_tasks)
        for tid, task_mod in self.model.tasks.items():
            if tid in tune_set:
                task_mod.requires_grad_(True)

        # 3. Reinit learned params for reinit_tasks (after checkpoint load)
        if ht.reinit_tasks:
            from fubio.data.shape_prior import ShapePrior

            sp_path = self.config.loss.shape_prior_path
            sp = ShapePrior.model_validate_json(sp_path.read_text()) if sp_path.exists() else None

            for tid in ht.reinit_tasks:
                task_mod = self.model.tasks[tid]
                task_sp = sp.tasks.get(tid) if sp else None
                task_mod.reinit_learned_params(task_sp)
                logger.info("Reinitialized learned params for task '%s'", tid)

        n_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in self.model.parameters())
        n_trainable = n_total - n_frozen
        logger.info(
            "Head-tune: %d/%d params frozen, %d trainable (tasks: %s, reinit: %s)",
            n_frozen, n_total, n_trainable,
            ht.tune_tasks, ht.reinit_tasks,
        )

    def _ensure_teacher_device(self) -> None:
        """Move teacher to the same device as the student if needed."""
        teacher: nn.Module = self._teacher_model  # type: ignore[assignment]
        student_device = next(self.model.parameters()).device
        teacher_device = next(teacher.parameters()).device
        if teacher_device != student_device:
            teacher.to(student_device)

    def train(self, mode: bool = True) -> FUBioModule:
        """Protect teacher from recursive train() — teacher is always eval."""
        super().train(mode)
        if hasattr(self, "_teacher_model"):
            self._teacher_model.eval()
        return self

    def on_before_zero_grad(self, optimizer: object) -> None:
        """EMA teacher update — runs AFTER optimizer.step(), BEFORE zero_grad.

        Uses warm-up formula: alpha_eff = min(1 - 1/(step+1), alpha).
        At step 0, teacher = student (full copy). Converges to alpha by
        step ~1/(1-alpha). For alpha=0.999, that is ~step 999 (~2 epochs).
        """
        if not hasattr(self, "_teacher_model"):
            return

        self._ensure_teacher_device()
        alpha = min(
            1 - 1 / (self._ema_step + 1),
            self.config.semi.alpha_ema,
        )
        # Fused EMA: two kernels instead of hundreds of per-parameter ops
        torch._foreach_mul_(self._teacher_params, alpha)
        torch._foreach_add_(self._teacher_params, self._student_params, alpha=1 - alpha)

        self._ema_step += 1
        if self._ema_step % 100 == 0:
            self.log("semi/alpha_eff", alpha)
            self.log("semi/ema_step", float(self._ema_step))

    def _semi_ramp(self) -> float:
        """Pseudo-label loss ramp: 0 during burn-in, linear, 1.0 after."""
        ep = self.current_epoch
        s, e = self.config.semi.ramp_start, self.config.semi.ramp_end
        if ep < s:
            return 0.0
        if ep >= e:
            return 1.0
        return (ep - s) / (e - s)

    def _eq_ramp(self) -> float:
        """L_eq ramp: starts after pseudo ramp, separate schedule."""
        ep = self.current_epoch
        s, e = self.config.semi.eq_ramp_start, self.config.semi.eq_ramp_end
        if ep < s:
            return 0.0
        if ep >= e:
            return 1.0
        return (ep - s) / (e - s)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist EMA teacher state for resume."""
        if hasattr(self, "_teacher_model"):
            checkpoint["teacher_state_dict"] = self._teacher_model.state_dict()
            checkpoint["ema_step"] = self._ema_step

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore EMA teacher on resume (setup() must run first)."""
        if "teacher_state_dict" in checkpoint and hasattr(self, "_teacher_model"):
            self._teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
            self._ema_step = checkpoint.get("ema_step", 0)
            logger.info("Restored EMA teacher (step=%d)", self._ema_step)

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def _normalize_image(self, image: Tensor) -> Tensor:
        """uint8 (B,3,H,W) → float32 ImageNet-normalized on GPU."""
        return image.float().div_(255.0).sub_(self._img_mean).div_(self._img_std)

    def _step(
        self,
        batch: dict,
        stage: str,
        return_eq_refs: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Shared logic for training_step and validation_step.

        stage is "train", "val", or "train_unlabeled" — the last being the
        task-known unlabeled pass, which produces confidence terms only and
        logs under its own prefix so its losses stay separable from the
        labeled pass's.

        return_eq_refs: if True, also return per-task detached landmarks
        from the forward pass for use as L_eq reference targets. Used by
        _semi_step to avoid a redundant clean-image forward.
        """
        unlabeled_pass = stage == "train_unlabeled"
        images: Tensor = self._normalize_image(batch["image"])
        targets: list[list[dict]] = batch["targets"]

        # Single forward pass — all tasks at once
        model_out: ModelOutput = self.model(images)
        results = model_out.task_outputs
        spatial_shape = model_out.spatial_shape

        B = images.shape[0]
        n_inst = self.config.head.n_inst
        device = images.device
        zero = torch.tensor(0.0, device=device)
        neg_smooth = self.config.loss.neg_label_smooth
        lambda_mil = self.config.semi.lambda_mil
        lambda_conf_rank = self.config.loss.lambda_conf_rank
        lambda_heatmap = self.config.loss.lambda_heatmap
        lambda_simcc = self.config.loss.lambda_simcc
        lambda_shape_cons = self.config.loss.lambda_shape_consistency
        lambda_supportive = self.config.loss.lambda_supportive
        lambda_evidence = self.config.loss.lambda_evidence
        lambda_geo_consistency = self.config.loss.lambda_geo_consistency
        lambda_geo_constraint = self.config.loss.lambda_geo_constraint
        # Both readouts route their spatial distribution to TaskOutput.heatmap,
        # but they are supervised at different widths on purpose: HeatmapPredictor
        # owns the final coordinate, GeoSimCC's stage 1 only has to land in the
        # right cell (stage 3 refines), so its target is deliberately narrower.
        if isinstance(self.config.head.coord, HeatmapCoordConfig):
            heatmap_sigma = self.config.head.coord.gaussian_sigma
        elif isinstance(self.config.head.coord, GeoSimCCCoordConfig):
            heatmap_sigma = self.config.head.coord.coarse_sigma_cells
        else:
            heatmap_sigma = 1.5
        # [P2: SimCC readout] precompute bins/sigma for SimCC loss
        simcc_n_bins = 0
        simcc_sigma = 5.0
        if isinstance(self.config.head.coord, (SimCCCoordConfig, ShapeSimCCCoordConfig)):
            simcc_n_bins = int(
                max(self.config.backbone.input_size) * self.config.head.coord.split_ratio
            )
            simcc_sigma = self.config.head.coord.sigma_bins

        # -- Supervision status per (image, task) --
        # Determines which loss terms apply to each query:
        #   labeled_tasks[b]: L_pos + L_suppress + L_absent
        #   unlabeled_tasks[b]: L_mil (bag-level) + L_absent (cross-task)
        # See module docstring for the full tier × loss table.
        labeled_tasks: list[set[int]] = [set() for _ in range(B)]
        unlabeled_tasks: list[set[int]] = [set() for _ in range(B)]
        for b, instances in enumerate(targets):
            for inst in instances:
                bucket = labeled_tasks if inst["is_labeled"] else unlabeled_tasks
                bucket[b].add(inst["task_id"])

        # Matcher only sees labeled instances — unlabeled data has no GT to match
        labeled_targets = [[inst for inst in imgs if inst["is_labeled"]] for imgs in targets]
        matched = self.matcher(results, labeled_targets)

        total_bbox = zero
        total_conf = zero
        total_conf_rank = zero
        total_lm = zero
        total_param = zero
        total_mil = zero
        total_heatmap = zero
        total_simcc = zero
        total_shape_cons = zero
        total_evidence = zero
        total_geo = zero
        total_geo_constraint = zero
        # Per-component accumulators for separate logging
        total_gc_ortho = zero
        total_gc_axis_order = zero
        total_gc_chamber_angle = zero
        total_gc_angle_sign = zero
        n_tasks_with_gt = 0
        n_tasks_with_geo = 0
        n_tasks_with_geo_constraint = 0
        n_tasks_with_mil = 0
        n_conf_terms = 0

        for tid, task_out in results.items():
            task_matched = matched[tid]  # list of (q_idx, g_idx) per image
            tdef = TASKS[tid]

            # -- Confidence targets --
            # Matched queries → 1.0 (L_pos).
            # Unmatched in images WITH GT for this task → neg_smooth (L_suppress):
            #   ~0.1% split-screen images have correct detections on unlabeled side;
            #   label smoothing (e.g. 0.05) softens the false-negative penalty.
            # Images where task is absent → 0.0 (L_absent, unchanged).
            conf_targets = torch.zeros(B, n_inst, device=device)
            images_with_gt: set[int] = set()
            for b, (q_idx, _g_idx) in enumerate(task_matched):
                if len(q_idx) > 0:
                    conf_targets[b, q_idx] = 1.0
                    images_with_gt.add(b)

            if neg_smooth > 0:
                for b in images_with_gt:
                    unmatched = conf_targets[b] == 0.0
                    conf_targets[b, unmatched] = neg_smooth

            # Images that are task-known unlabeled FOR THIS TASK. They have no
            # match, so the zero-filled targets above read as "this task is
            # absent" — the opposite of what is known about them, and directly
            # opposed to L_mil. Excluded from the focal loss rather than
            # cancelled against it. Cross-task rows are untouched: an image of
            # task t genuinely is absent from every other task (L_absent).
            mil_rows = [
                b
                for b in range(B)
                if tdef.task_int in unlabeled_tasks[b] and tdef.task_int not in labeled_tasks[b]
            ]

            # Ranking supervision on top of per-slot calibration: focal loss says
            # "is this slot a hit?", ranking says "is this slot the BEST one?" —
            # and argmax(conf) at serving time depends only on the latter.
            if lambda_conf_rank > 0:
                total_conf_rank = total_conf_rank + conf_ranking_loss(
                    task_out.conf.squeeze(-1),
                    [list(q_idx) for q_idx, _ in task_matched],
                )

            conf_logits = task_out.conf.squeeze(-1)  # (B, n_inst)
            if mil_rows:
                keep = torch.ones(B, dtype=torch.bool, device=device)
                keep[mil_rows] = False
                if keep.any():
                    total_conf = total_conf + conf_focal_loss(
                        conf_logits[keep].reshape(-1),
                        conf_targets[keep].reshape(-1),
                    )
                    n_conf_terms += 1
            else:
                total_conf = total_conf + conf_focal_loss(
                    conf_logits.reshape(-1),
                    conf_targets.reshape(-1),
                )
                n_conf_terms += 1

            # -- L_mil: bag-level positive signal for task-known unlabeled data --
            # Exactly the rows dropped from the focal loss above: L_mil is the
            # only confidence supervision they receive for their own task.
            if lambda_mil > 0 and mil_rows:
                total_mil = total_mil + mil_loss(task_out.conf[mil_rows, :, 0])
                n_tasks_with_mil += 1

            # -- Bbox + landmark + param losses on matched pairs --
            # Collect numpy GT first, then one bulk CPU→GPU transfer per task
            # (avoids ~500 individual torch.tensor() calls per step)
            gt_boxes_np: list[np.ndarray] = []
            gt_lm_np: list[np.ndarray] = []
            gt_sup_np: list[np.ndarray] = []
            gt_vis_np: list[np.ndarray] = []
            gt_evi_np: list[np.ndarray] = []
            orig_hw_np: list[np.ndarray] = []
            b_indices: list[int] = []
            q_indices: list[int] = []

            for b, (q_idx, g_idx) in enumerate(task_matched):
                if len(q_idx) == 0:
                    continue
                for qi, gi in zip(q_idx, g_idx, strict=True):
                    inst = labeled_targets[b][gi]
                    gt_boxes_np.append(inst["bbox"])
                    gt_lm_np.append(inst["keypoints"])
                    gt_sup_np.append(inst["supervised_mask"])
                    gt_vis_np.append(inst["visible_mask"])
                    evi_default = np.zeros_like(inst["supervised_mask"], dtype=np.int64)
                    gt_evi_np.append(inst.get("evidence_mask", evi_default))
                    orig_hw_np.append(inst["original_hw"])
                    b_indices.append(b)
                    q_indices.append(qi)

            if not gt_boxes_np:
                continue

            n_tasks_with_gt += 1

            gt_lm = torch.tensor(np.stack(gt_lm_np), dtype=torch.float32, device=device)
            sup_mask = torch.tensor(np.stack(gt_sup_np), dtype=torch.bool, device=device)
            vis_mask = torch.tensor(np.stack(gt_vis_np), dtype=torch.bool, device=device)
            gt_boxes_cat = torch.tensor(np.stack(gt_boxes_np), dtype=torch.float32, device=device)

            pred_lm = task_out.landmarks[b_indices, q_indices]  # (N_matched, K_model, 2)

            # GT from collate may have supportive landmarks beyond the model's
            # prediction range. Slice to match for scored losses; keep full GT
            # for geometric consistency.
            if gt_lm.shape[1] > pred_lm.shape[1]:
                _gt_sup_for_geo = gt_lm[:, pred_lm.shape[1]:]
                _mask_sup_for_geo = (sup_mask & vis_mask)[:, pred_lm.shape[1]:]
                gt_lm = gt_lm[:, :pred_lm.shape[1]]
                sup_mask = sup_mask[:, :pred_lm.shape[1]]
                vis_mask = vis_mask[:, :pred_lm.shape[1]]
            else:
                _gt_sup_for_geo = None
                _mask_sup_for_geo = None

            if self.config.head.derive_bbox:
                pred_boxes_cat = derive_bbox_masked(
                    pred_lm, vis_mask, context_scale=tdef.bbox_context_scale
                )
            else:
                pred_boxes_cat = task_out.bbox[b_indices, q_indices]
            box_losses = bbox_loss(pred_boxes_cat, gt_boxes_cat)
            total_bbox = total_bbox + box_losses["loss_box_l1"] + box_losses["loss_box_ciou"]
            mask = sup_mask & vis_mask

            # Scored/supportive split: separate loss normalization
            _has_scored_mask = hasattr(self.model.tasks[tid], "scored_mask")
            if _has_scored_mask:
                scored_buf = self.model.tasks[tid].scored_mask  # (K_total,) bool
                scored_m = mask & scored_buf.unsqueeze(0)
                sup_lm_m = mask & ~scored_buf.unsqueeze(0)
            else:
                scored_m = mask
                sup_lm_m = torch.zeros_like(mask)

            loss_lm_scored = landmark_loss(
                pred_lm, gt_lm, scored_m,
                beta=self.config.loss.landmark_beta,
            )
            loss_lm_sup = landmark_loss(
                pred_lm, gt_lm, sup_lm_m,
                beta=self.config.loss.landmark_beta,
            )
            total_lm = total_lm + loss_lm_scored + lambda_supportive * loss_lm_sup

            total_param = total_param + param_loss(
                pred_lm,
                gt_lm,
                tid,
                vis_mask,
                beta=self.config.loss.landmark_beta,
            )

            lambda_angle_sign = self.config.loss.lambda_angle_sign
            if lambda_geo_constraint > 0 or lambda_angle_sign > 0:
                gc_gt = gt_lm if (not unlabeled_pass and lambda_angle_sign > 0) else None
                gc_components = geometric_constraint_loss(pred_lm, tid, gt_landmarks=gc_gt)
                # angle_sign has its own lambda; other constraints share lambda_geo_constraint
                angle_sign_val = gc_components.pop("angle_sign", None)
                if gc_components and lambda_geo_constraint > 0:
                    gc_sum = torch.stack(list(gc_components.values())).mean()
                    total_geo_constraint = total_geo_constraint + gc_sum
                    n_tasks_with_geo_constraint += 1
                if angle_sign_val is not None and lambda_angle_sign > 0:
                    total_geo_constraint = total_geo_constraint + lambda_angle_sign * angle_sign_val
                    total_gc_angle_sign = total_gc_angle_sign + angle_sign_val
                    n_tasks_with_geo_constraint = max(n_tasks_with_geo_constraint, 1)
                for k, v in gc_components.items():
                    if k == "ortho":
                        total_gc_ortho = total_gc_ortho + v
                    elif k == "axis_order":
                        total_gc_axis_order = total_gc_axis_order + v
                    elif k == "chamber_angle":
                        total_gc_chamber_angle = total_gc_chamber_angle + v

            if lambda_heatmap > 0 and task_out.heatmap is not None:
                pred_heat = task_out.heatmap[b_indices, q_indices]  # (N_matched, K_total, S)
                h_scored = heatmap_loss(
                    pred_heat, gt_lm, scored_m, spatial_shape,
                    sigma_cells=heatmap_sigma,
                )
                h_sup = heatmap_loss(
                    pred_heat, gt_lm, sup_lm_m, spatial_shape,
                    sigma_cells=heatmap_sigma,
                )
                total_heatmap = total_heatmap + h_scored + lambda_supportive * h_sup

            # [P2: SimCC readout] distribution-level supervision on 1D bins
            if lambda_simcc > 0 and task_out.simcc_logits is not None:
                sim_x, sim_y = task_out.simcc_logits
                s_scored = simcc_loss(
                    sim_x[b_indices, q_indices],
                    sim_y[b_indices, q_indices],
                    gt_lm, scored_m, simcc_n_bins,
                    sigma_bins=simcc_sigma,
                )
                s_sup = simcc_loss(
                    sim_x[b_indices, q_indices],
                    sim_y[b_indices, q_indices],
                    gt_lm, sup_lm_m, simcc_n_bins,
                    sigma_bins=simcc_sigma,
                )
                total_simcc = total_simcc + s_scored + lambda_supportive * s_sup

            if lambda_shape_cons > 0 and tid in self._canon_tasks:
                total_shape_cons = total_shape_cons + shape_consistency_loss(
                    pred_lm,
                    mask,
                    getattr(self, f"_canon_mean_{tid}"),
                    getattr(self, f"_canon_basis_{tid}"),
                )

            # Evidence loss: BCE on per-landmark evidence prediction
            if lambda_evidence > 0 and task_out.evidence is not None:
                gt_evi = torch.tensor(
                    np.stack(gt_evi_np), dtype=torch.float32, device=device,
                )
                pred_evi = task_out.evidence[b_indices, q_indices]
                evi_mask = sup_mask & vis_mask
                evi_bce = F.binary_cross_entropy_with_logits(
                    pred_evi, gt_evi, reduction="none",
                )
                total_evidence = total_evidence + (
                    (evi_bce * evi_mask.float()).sum()
                    / evi_mask.float().sum().clamp(min=1)
                )

            # Geometric consistency: compute supportive from predicted scored
            # landmarks via differentiable geometry, compare with GT supportive.
            # Gradient flows directly to scored predictions through the geometry.
            if lambda_geo_consistency > 0 and _gt_sup_for_geo is not None:
                derived_sup = compute_supportive_torch(tid, pred_lm)
                if derived_sup is not None and _mask_sup_for_geo.any():
                    total_geo = total_geo + landmark_loss(
                        derived_sup, _gt_sup_for_geo, _mask_sup_for_geo,
                        beta=self.config.loss.landmark_beta,
                    )
                    n_tasks_with_geo += 1

            # Accumulate metrics (val only — per-instance param computation is expensive)
            # Metrics use SCORED landmarks only (supportive are training-only)
            if stage == "val":
                K_scored = tdef.n_keypoints
                gt_lm_scored = gt_lm[:, :K_scored]
                mask_scored = mask[:, :K_scored]
                vis_scored = vis_mask[:, :K_scored]

                # Per-axis scale: normalized → original pixel space.
                # Letterbox is uniform scaling s = min(W/orig_w, H/orig_h),
                # so Δnorm * [W/s, H/s] gives exact original-pixel distances
                # (the padding offset cancels in pred−gt differences).
                _target_w, _target_h = self.config.backbone.input_size
                _orig_hw = np.stack(orig_hw_np)  # (N, 2) [H, W]
                _lb_scale = np.minimum(
                    _target_w / _orig_hw[:, 1], _target_h / _orig_hw[:, 0]
                )  # (N,)
                pixel_scale = torch.tensor(
                    np.stack([_target_w / _lb_scale, _target_h / _lb_scale], axis=1),
                    dtype=torch.float32,
                    device=device,
                )  # (N, 2)

                lm_all = task_out.landmarks[b_indices].detach()  # (N, n_inst, K_total, 2)
                if lm_all.shape[1] > 1:
                    f = lm_all.flatten(2)
                    pair = torch.cdist(f, f)  # (N, n_inst, n_inst)
                    eye = torch.eye(pair.shape[-1], device=pair.device) * 1e9
                    self._inst_sep.append((pair + eye).amin(dim=(1, 2)).cpu())

                serve_q = select_serving_query(task_out.conf[b_indices])
                serve_lm = task_out.landmarks[b_indices, serve_q]
                serve_lm_scored = serve_lm[:, :K_scored]
                serve_boxes = (
                    derive_bbox_masked(serve_lm, vis_mask, context_scale=tdef.bbox_context_scale)
                    if self.config.head.derive_bbox
                    else task_out.bbox[b_indices, serve_q]
                )
                self._accumulate_metrics(
                    tid,
                    serve_lm_scored,
                    gt_lm_scored,
                    mask_scored,
                    vis_scored,
                    serve_boxes,
                    gt_boxes_cat,
                    pixel_scale,
                    stage,
                )

                self._oracle_mre.update(
                    pred_lm[:, :K_scored] * pixel_scale[:, None, :],
                    gt_lm_scored * pixel_scale[:, None, :],
                    task_id=tdef.task_int,
                    mask=mask_scored,
                )

        denom = max(n_tasks_with_gt, 1)
        mil_denom = max(n_tasks_with_mil, 1)
        conf_denom = max(n_conf_terms, 1)
        geo_denom = max(n_tasks_with_geo, 1)
        gc_denom = max(n_tasks_with_geo_constraint, 1)
        total = (
            self.config.loss.lambda_bbox * total_bbox / denom
            + self.config.loss.lambda_conf * total_conf / conf_denom
            + lambda_conf_rank * total_conf_rank / denom
            + self.config.loss.lambda_land * total_lm / denom
            + self.config.loss.lambda_param * total_param / denom
            + lambda_mil * total_mil / mil_denom
            + lambda_heatmap * total_heatmap / denom
            + lambda_simcc * total_simcc / denom
            + lambda_shape_cons * total_shape_cons / denom
            + lambda_evidence * total_evidence / denom
            + lambda_geo_consistency * total_geo / geo_denom
            + lambda_geo_constraint * total_geo_constraint / gc_denom
        )

        # The three blocks below need no GT and would therefore silently start
        # acting on the unlabeled pass as well. Held to the labeled pass so that
        # enabling semi-supervision changes the objective by the semi-supervised
        # terms alone; extending any of them to unlabeled data is a deliberate
        # experiment, not a side effect of loading it.
        all_sample_losses = not unlabeled_pass

        # Shape residual loss — applied to ALL samples, not gated on GT
        total_shape = zero
        lambda_shape = self.config.loss.lambda_shape
        if lambda_shape > 0 and all_sample_losses:
            n_shape = 0
            for task_out in results.values():
                if task_out.residual is not None:
                    total_shape = total_shape + shape_residual_loss(task_out.residual)
                    n_shape += 1
            if n_shape > 0:
                total_shape = total_shape / n_shape
            total = total + lambda_shape * total_shape

        # AffineHead orthogonality regularization — applied to ALL samples
        total_ortho = zero
        lambda_ortho = self.config.loss.lambda_ortho
        if lambda_ortho > 0 and all_sample_losses:
            n_ortho = 0
            for task_out in results.values():
                if task_out.affine_T is not None:
                    total_ortho = total_ortho + ortho_regularization(task_out.affine_T)
                    n_ortho += 1
            if n_ortho > 0:
                total_ortho = total_ortho / n_ortho
            total = total + lambda_ortho * total_ortho

        # Instance repulsion: push UNMATCHED queries off the matched one. Matched
        # slots are detached so the penalty never drags a correct prediction away.
        total_repulsion = zero
        lambda_repulsion = self.config.loss.lambda_repulsion
        if lambda_repulsion > 0 and all_sample_losses:
            total_repulsion = instance_repulsion_loss(
                results,
                matched_queries={
                    tid: [list(q_idx) for q_idx, _ in matched[tid]] for tid in results
                },
            )
            total = total + lambda_repulsion * total_repulsion

        # MIRO: regularize fine-tuned backbone toward frozen pretrained features
        total_miro = zero
        if self._miro_enabled and stage == "train":
            from fubio.train.regularizer import miro_loss

            post_feats = model_out.backbone_out.features
            with torch.no_grad():
                pre_out = self._frozen_backbone(images)
            pre_feats = pre_out.features
            total_miro = miro_loss(pre_feats, post_feats, self._miro_encoders)
            total = total + self.config.miro.lambda_miro * total_miro

        sync = stage == "val"
        self.log(f"{stage}/loss", total, prog_bar=True, sync_dist=sync)
        self.log(f"{stage}/loss_bbox", total_bbox / denom, sync_dist=sync)
        self.log(f"{stage}/loss_conf", total_conf / conf_denom, sync_dist=sync)
        self.log(f"{stage}/loss_lm", total_lm / denom, sync_dist=sync)
        self.log(f"{stage}/loss_param", total_param / denom, sync_dist=sync)
        if lambda_conf_rank > 0:
            self.log(f"{stage}/loss_conf_rank", total_conf_rank / denom, sync_dist=sync)
        if lambda_mil > 0:
            self.log(f"{stage}/loss_mil", total_mil / mil_denom, sync_dist=sync)
        if lambda_heatmap > 0:
            self.log(f"{stage}/loss_heatmap", total_heatmap / denom, sync_dist=sync)
        if lambda_simcc > 0:
            self.log(f"{stage}/loss_simcc", total_simcc / denom, sync_dist=sync)
        if lambda_shape_cons > 0:
            self.log(f"{stage}/loss_shape_cons", total_shape_cons / denom, sync_dist=sync)
        if lambda_evidence > 0:
            self.log(f"{stage}/loss_evidence", total_evidence / denom, sync_dist=sync)
        if lambda_geo_consistency > 0:
            self.log(f"{stage}/loss_geo_consistency", total_geo / geo_denom, sync_dist=sync)
        if lambda_geo_constraint > 0:
            gc_val = total_geo_constraint / gc_denom
            self.log(f"{stage}/loss_geo_constraint", gc_val, sync_dist=sync)
            self.log(f"{stage}/loss_gc_ortho", total_gc_ortho / gc_denom, sync_dist=sync)
            self.log(
                f"{stage}/loss_gc_axis_order", total_gc_axis_order / gc_denom, sync_dist=sync,
            )
            self.log(
                f"{stage}/loss_gc_chamber", total_gc_chamber_angle / gc_denom, sync_dist=sync,
            )
            if total_gc_angle_sign > 0:
                self.log(
                    f"{stage}/loss_gc_angle_sign", total_gc_angle_sign / gc_denom, sync_dist=sync
                )
        if lambda_shape > 0:
            self.log(f"{stage}/loss_shape", total_shape, sync_dist=sync)
        if lambda_ortho > 0:
            self.log(f"{stage}/loss_ortho", total_ortho, sync_dist=sync)
        if lambda_repulsion > 0:
            self.log(f"{stage}/loss_repulsion", total_repulsion, sync_dist=sync)
        if self._miro_enabled:
            self.log(f"{stage}/loss_miro", total_miro, sync_dist=sync)

        if return_eq_refs:
            eq_refs: dict[str, Tensor] = {}
            for tid, task_out in results.items():
                eq_refs[tid] = task_out.landmarks.detach()
            return total, eq_refs
        return total

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _semi_step(
        self, batch: dict, eq_refs: dict[str, Tensor],
    ) -> Tensor:
        """Consolidated pseudo-label + equivariance on ONE shared warped view.

        4 forwards total (2 already done by caller):
          #1 student(labeled) — caller
          #2 student(unlabeled_clean) — caller, also provides eq_refs
          #3 teacher(unlabeled_clean) — this method [no_grad]
          #4 student(unlabeled_warped+aug) — this method [grad]

        Losses computed on the shared warped view:
          L_pseudo: conf-weighted L1 on landmark coords
          L_pseudo_param: L1 on clinical parameters derived from landmarks
          L_pseudo_geo: L1 on supportive landmarks (structural consistency)
          L_pseudo_gc: anatomical constraints (ortho, axis order, chamber angle)
          L_eq: equivariance consistency (student self-consistency)
        """
        from fubio.train.views import (
            _forward_to_inverse,
            make_rotation_affine,
            sample_affine,
            warp_image,
            warp_landmarks,
        )

        self._ensure_teacher_device()
        semi = self.config.semi

        images_01 = batch["image"].float().div_(255.0)
        images_norm = (images_01 - self._img_mean) / self._img_std
        targets: list[list[dict]] = batch["targets"]
        B = images_norm.shape[0]
        device = images_norm.device

        # Forward #3: teacher on clean image (+ optional TTA views)
        with torch.no_grad():
            teacher_out = self._teacher_model(images_norm)

            # TTA Teacher: average landmarks across rotated views.
            # TaskOutput.landmarks is read-only, so collect into a separate dict.
            if semi.tta_teacher_angles:
                tta_lm: dict[str, Tensor] = {
                    tid: teacher_out.task_outputs[tid].landmarks.clone()
                    for tid in teacher_out.task_outputs
                }
                for angle in semi.tta_teacher_angles:
                    mat_tta = make_rotation_affine(B, angle, device=device, dtype=images_01.dtype)
                    warped_tta = warp_image(images_01, mat_tta)
                    warped_norm = (warped_tta - self._img_mean) / self._img_std
                    out_tta = self._teacher_model(warped_norm)
                    inv_mat = _forward_to_inverse(mat_tta)
                    for tid in tta_lm:
                        lm_back = warp_landmarks(out_tta.task_outputs[tid].landmarks, inv_mat)
                        tta_lm[tid] = tta_lm[tid] + lm_back

                n_views = 1 + len(semi.tta_teacher_angles)
                for tid in tta_lm:
                    tta_lm[tid] = tta_lm[tid] / n_views
            else:
                tta_lm = None

        # Sample ONE affine (with flip) shared by pseudo + eq
        matrix = sample_affine(
            B,
            rotation_range=semi.eq_rotation_range,
            scale_range=semi.eq_scale_range,
            translate_range=semi.eq_translate_range,
            flip_prob=semi.flip_prob,
            device=device,
            dtype=images_01.dtype,
        )

        # Undo flip for no_flip_tasks: if a no-flip task's row got flipped
        # by sample_affine, reverse it by re-negating row 0.
        if semi.no_flip_tasks:
            no_flip_tids = set(
                TASKS[t].task_int for t in semi.no_flip_tasks if t in TASKS
            )
            for b in range(B):
                for inst in targets[b]:
                    if inst["task_id"] in no_flip_tids:
                        # Detect flip: determinant of 2x2 submatrix < 0
                        det = matrix[b, 0, 0] * matrix[b, 1, 1] - matrix[b, 0, 1] * matrix[b, 1, 0]
                        if det < 0:
                            matrix[b, 0, 0] *= -1
                            matrix[b, 0, 1] *= -1
                            matrix[b, 0, 2] = 1.0 - matrix[b, 0, 2]
                        break

        # Warp in [0,1], photometric aug, normalize
        warped_01 = warp_image(images_01, matrix)
        augmented_01 = self._photo_aug(warped_01)
        student_input = (augmented_01 - self._img_mean) / self._img_std

        # Forward #4: student on warped+augmented view
        student_out = self.model(student_input)

        # Build unlabeled task sets
        unlabeled_tasks: list[set[int]] = [set() for _ in range(B)]
        for b, instances in enumerate(targets):
            for inst in instances:
                if not inst["is_labeled"]:
                    unlabeled_tasks[b].add(inst["task_id"])

        # Per-task losses with separate denominators
        total_pseudo = torch.tensor(0.0, device=device)
        total_eq = torch.tensor(0.0, device=device)
        total_pseudo_param = torch.tensor(0.0, device=device)
        total_pseudo_geo = torch.tensor(0.0, device=device)
        total_pseudo_gc = torch.tensor(0.0, device=device)
        n_valid_total = torch.tensor(0.0, device=device)
        n_tasks = 0
        n_tasks_param = 0
        n_tasks_geo = 0
        n_tasks_gc = 0

        pseudo_ramp = self._semi_ramp()
        eq_ramp = self._eq_ramp()
        lambda_gc = self.config.loss.lambda_geo_constraint

        for tid, teacher_to in teacher_out.task_outputs.items():
            tdef = TASKS[tid]
            rows = [
                b for b in range(B)
                if tdef.task_int in unlabeled_tasks[b]
            ]
            if not rows:
                continue

            task_weight = semi.task_pseudo_lambda.get(tid, 1.0)
            student_lm = student_out.task_outputs[tid].landmarks[rows]
            K_scored = tdef.n_keypoints

            # --- L_pseudo: student(warped) vs T(teacher(clean)) ---
            if pseudo_ramp > 0 and semi.lambda_pseudo > 0:
                raw_lm = tta_lm[tid][rows] if tta_lm is not None else teacher_to.landmarks[rows]
                teacher_lm = raw_lm.detach()
                pseudo_target = warp_landmarks(teacher_lm, matrix[rows])
                pseudo_valid = (pseudo_target >= 0) & (pseudo_target <= 1)
                pseudo_valid = pseudo_valid.all(dim=-1)
                n_valid_total = n_valid_total + pseudo_valid.sum()
                teacher_conf = teacher_to.conf[rows]

                total_pseudo = total_pseudo + task_weight * pseudo_label_loss(
                    student_lm, pseudo_target, teacher_conf, pseudo_valid,
                )

                # --- L_pseudo_param: clinical parameters from landmarks ---
                if semi.lambda_pseudo_param > 0:
                    student_scored = student_lm[:, 0, :K_scored]
                    target_scored = pseudo_target[:, 0, :K_scored]
                    vis = pseudo_valid[:, 0, :K_scored]
                    pp = param_loss(
                        student_scored, target_scored, tid, vis,
                        beta=self.config.loss.landmark_beta,
                    )
                    total_pseudo_param = total_pseudo_param + task_weight * pp
                    n_tasks_param += 1

                # --- L_pseudo_geo: supportive consistency in warped frame ---
                if semi.lambda_pseudo_geo > 0:
                    teacher_sup = compute_supportive_torch(
                        tid, pseudo_target[:, 0, :K_scored],
                    )
                    student_sup = compute_supportive_torch(
                        tid, student_lm[:, 0, :K_scored],
                    )
                    if teacher_sup is not None and student_sup is not None:
                        geo = F.l1_loss(student_sup, teacher_sup.detach())
                        total_pseudo_geo = total_pseudo_geo + task_weight * geo
                        n_tasks_geo += 1

            # --- L_pseudo_gc: anatomical constraints on student warped view ---
            if pseudo_ramp > 0 and lambda_gc > 0:
                gc = geometric_constraint_loss(student_lm[:, 0], tid)
                if gc:
                    gc_sum = torch.stack(list(gc.values())).mean()
                    total_pseudo_gc = total_pseudo_gc + task_weight * gc_sum
                    n_tasks_gc += 1

            # --- L_eq: student(warped) vs T(detach(student(clean))) ---
            if eq_ramp > 0 and semi.lambda_eq > 0 and tid in eq_refs:
                eq_ref_lm = eq_refs[tid][rows]
                eq_conf = teacher_to.conf[rows]
                total_eq = total_eq + equivariance_loss(
                    eq_ref_lm, student_lm, matrix[rows], eq_conf,
                )

            n_tasks += 1

            with torch.no_grad():
                conf_mean = teacher_to.conf[rows].detach().squeeze(-1).sigmoid().mean()
                self.log(
                    f"semi/teacher_conf_{tid}", conf_mean,
                    on_step=False, on_epoch=True,
                )

        # Normalize by per-type denominators
        pseudo_loss = total_pseudo / max(n_tasks, 1)
        eq_loss = total_eq / max(n_tasks, 1)
        pp_loss = total_pseudo_param / max(n_tasks_param, 1)
        pg_loss = total_pseudo_geo / max(n_tasks_geo, 1)
        gc_loss = total_pseudo_gc / max(n_tasks_gc, 1)

        # Combine with lambdas and ramps
        loss = torch.tensor(0.0, device=device)
        if pseudo_ramp > 0:
            if semi.lambda_pseudo > 0:
                loss = loss + pseudo_ramp * semi.lambda_pseudo * pseudo_loss
            if semi.lambda_pseudo_param > 0:
                loss = loss + pseudo_ramp * semi.lambda_pseudo_param * pp_loss
            if semi.lambda_pseudo_geo > 0:
                loss = loss + pseudo_ramp * semi.lambda_pseudo_geo * pg_loss
            if lambda_gc > 0 and n_tasks_gc > 0:
                loss = loss + pseudo_ramp * lambda_gc * gc_loss
        if eq_ramp > 0 and semi.lambda_eq > 0:
            loss = loss + eq_ramp * semi.lambda_eq * eq_loss

        # Logging
        self.log("semi/loss_pseudo", pseudo_loss)
        self.log("semi/loss_eq", eq_loss)
        if n_tasks_param > 0:
            self.log("semi/loss_pseudo_param", pp_loss)
        if n_tasks_geo > 0:
            self.log("semi/loss_pseudo_geo", pg_loss)
        if n_tasks_gc > 0:
            self.log("semi/loss_pseudo_gc", gc_loss)
        self.log("semi/ramp_pseudo", pseudo_ramp)
        self.log("semi/ramp_eq", eq_ramp)
        self.log("semi/n_valid_landmarks", n_valid_total.float())
        loss_total = pseudo_loss + eq_loss + pp_loss + pg_loss + gc_loss
        self.log("semi/loss_total", loss_total, on_epoch=True)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        # semi.enabled → CombinedLoader hands over both streams keyed by tier;
        # otherwise a plain BatchDict, unchanged.
        if "labeled" in batch:
            loss = self._step(batch["labeled"], "train")
            # Check if semi losses are active this epoch BEFORE doing extra work
            pseudo_ramp = self._semi_ramp()
            eq_ramp = self._eq_ramp()
            semi_active = (
                (pseudo_ramp > 0 and self.config.semi.lambda_pseudo > 0)
                or (eq_ramp > 0 and self.config.semi.lambda_eq > 0)
            ) and hasattr(self, "_teacher_model")

            # Forward #2: student on clean unlabeled → MIL + L_absent
            # Also extract eq_refs if L_eq will be active
            need_eq_refs = semi_active and eq_ramp > 0 and self.config.semi.lambda_eq > 0
            step_result = self._step(
                batch["unlabeled"], "train_unlabeled", return_eq_refs=need_eq_refs,
            )
            if need_eq_refs:
                unlabeled_loss, eq_refs = step_result
            else:
                unlabeled_loss = step_result
                eq_refs = {}
            loss = loss + unlabeled_loss

            # Consolidated semi step: pseudo + eq on shared warped view
            if semi_active:
                loss = loss + self._semi_step(batch["unlabeled"], eq_refs)
            else:
                # Log zero so ModelCheckpoint can find semi/loss_total during burn-in
                self.log("semi/loss_total", 0.0, on_epoch=True)
        else:
            loss = self._step(batch, "train")
        if self._trainer is not None:
            opt = self.optimizers()
            if hasattr(opt, "param_groups"):
                # Log by group name (robust to LLRD's many backbone groups):
                # backbone shows the LR spread (top vs bottom layer), rest by name.
                bb_lrs = [
                    g["lr"]
                    for g in opt.param_groups
                    if str(g.get("name", "")).startswith("backbone")
                ]
                if bb_lrs:
                    self.log("lr/backbone", max(bb_lrs))
                    if len(bb_lrs) > 1:
                        self.log("lr/backbone_min", min(bb_lrs))
                for g in opt.param_groups:
                    if g.get("name") in ("neck", "decoder", "heads", "miro"):
                        self.log(f"lr/{g['name']}", g["lr"])
        return loss

    def on_train_epoch_start(self) -> None:
        if not self.config.head_tune.enabled:
            freeze_epochs = self.config.backbone.freeze_epochs
            if self.current_epoch < freeze_epochs:
                self.model.backbone.freeze()
            elif self.current_epoch == freeze_epochs:
                self.model.backbone.unfreeze()
                logger.info("Backbone unfrozen at epoch %d", self.current_epoch)

        dm = self.trainer.datamodule
        if hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

        self._train_mre.reset()
        self._train_param_mae.reset()
        self._train_bbox_prec.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self._val_mre.reset()
        self._val_mre_pixel.reset()
        self._val_param_mae.reset()
        self._val_bbox_prec.reset()
        self._oracle_mre.reset()
        self._inst_sep.clear()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _full_param_groups(
        self, opt_cfg: OptimizerConfig,
    ) -> tuple[list[dict], int]:
        """Standard param groups: backbone (LLRD) + neck + decoder + heads."""
        backbone_groups = self.model.backbone.param_groups(
            opt_cfg.lr_backbone,
            opt_cfg.layer_decay,
        )
        n_backbone_groups = len(backbone_groups)

        query_param_ids: set[int] = set()
        query_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for task_mod in self.model.tasks.values():
            qp = task_mod.query_params()
            query_params.extend(qp)
            query_param_ids.update(id(p) for p in qp)
        for task_mod in self.model.tasks.values():
            for p in task_mod.parameters():
                if id(p) not in query_param_ids:
                    head_params.append(p)

        neck_params = [p for n, p in self.model.named_parameters() if n.startswith("neck.")]
        decoder_params = [p for n, p in self.model.named_parameters() if n.startswith("decoder.")]
        _shared_names = ("loc_k_proj.", "fine_stem.", "fine_proj.")
        geo_params = [
            p
            for n, p in self.model.named_parameters()
            if any(n.startswith(x) for x in _shared_names)
        ]

        param_groups = [
            *backbone_groups,
            {"params": neck_params, "lr": opt_cfg.lr_neck, "name": "neck"},
            {
                "params": decoder_params + query_params,
                "lr": opt_cfg.lr_decoder,
                "name": "decoder",
            },
            {"params": head_params + geo_params, "lr": opt_cfg.lr_heads, "name": "heads"},
        ]

        grouped_ids = {id(p) for g in param_groups for p in g["params"]}
        all_ids = {id(p) for p in self.model.parameters()}
        assert grouped_ids == all_ids, (
            f"Parameter grouping mismatch: {len(grouped_ids)} grouped vs {len(all_ids)} total"
        )
        return param_groups, n_backbone_groups

    def _head_tune_param_groups(
        self, opt_cfg: OptimizerConfig, ht: HeadTuneConfig,
    ) -> tuple[list[dict], int]:
        """Per-task param groups for surgical head-tuning.

        Only trainable (requires_grad=True) params from tune_tasks' TaskModules
        are grouped. Each task gets two groups: queries at lr_decoder * scale,
        other params at lr_heads * scale.
        """
        param_groups: list[dict] = []

        for tid in ht.tune_tasks:
            task_mod = self.model.tasks[tid]
            scale = ht.lr_scale.get(tid, 1.0)

            qp = [p for p in task_mod.query_params() if p.requires_grad]
            qp_ids = {id(p) for p in qp}
            hp = [p for p in task_mod.parameters() if p.requires_grad and id(p) not in qp_ids]

            if qp:
                param_groups.append({
                    "params": qp,
                    "lr": opt_cfg.lr_decoder * scale,
                    "name": f"query_{tid}",
                })
            if hp:
                param_groups.append({
                    "params": hp,
                    "lr": opt_cfg.lr_heads * scale,
                    "name": f"heads_{tid}",
                })

        grouped_ids = {id(p) for g in param_groups for p in g["params"]}
        trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        assert grouped_ids == trainable_ids, (
            f"Head-tune grouping mismatch: {len(grouped_ids)} grouped "
            f"vs {len(trainable_ids)} trainable"
        )
        # No backbone groups in head-tune — freeze_steps irrelevant
        return param_groups, 0

    def configure_optimizers(self) -> dict:
        opt_cfg = self.config.optimizer
        ht = self.config.head_tune

        if ht.enabled:
            param_groups, n_backbone_groups = self._head_tune_param_groups(opt_cfg, ht)
        else:
            param_groups, n_backbone_groups = self._full_param_groups(opt_cfg)
        # MIRO variance encoder params — learn at neck LR
        if self._miro_enabled:
            param_groups.append(
                {
                    "params": list(self._miro_encoders.parameters()),
                    "lr": opt_cfg.lr_neck,
                    "name": "miro",
                }
            )
        optimizer = AdamW(
            param_groups,
            weight_decay=opt_cfg.weight_decay,
        )

        if self._trainer is None:
            return {"optimizer": optimizer}

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(1, int(total_steps) // opt_cfg.max_epochs)
        freeze_steps = self.config.backbone.freeze_epochs * steps_per_epoch

        scheduler = PhaseScheduler(
            optimizer=optimizer,
            warmup_steps=opt_cfg.warmup_steps,
            total_steps=int(total_steps),
            freeze_steps=freeze_steps,
            eta_min_ratio=opt_cfg.eta_min_ratio,
            n_backbone_groups=n_backbone_groups,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Shared metric helpers
    # ------------------------------------------------------------------

    def _get_metrics(
        self,
        stage: str,
    ) -> tuple[MREMetric, ParamMAEMetric, BBoxPrecisionMetric]:
        if stage == "train":
            return self._train_mre, self._train_param_mae, self._train_bbox_prec
        return self._val_mre, self._val_param_mae, self._val_bbox_prec

    @torch.no_grad()
    def _accumulate_metrics(
        self,
        task_id_str: str,
        pred_lm: Tensor,
        gt_lm: Tensor,
        mask: Tensor,
        vis_mask: Tensor,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
        pixel_scale: Tensor,
        stage: str,
    ) -> None:
        """Update MRE, ParamMAE, and BBoxPrecision for matched predictions."""
        mre, param_mae, bbox_prec = self._get_metrics(stage)
        tdef = TASKS[task_id_str]

        # MRE + ParamMAE: per-instance updates
        for i in range(pred_lm.shape[0]):
            p = pred_lm[i].unsqueeze(0).detach()
            g = gt_lm[i].unsqueeze(0)
            m = mask[i].unsqueeze(0)
            mre.update(p, g, task_id=tdef.task_int, mask=m)

            # MRE in original pixel space: normalized × [W/s, H/s] where
            # s = min(target_W/orig_w, target_H/orig_h). Letterbox is uniform
            # scaling so the padding offset cancels in pred−gt differences.
            s = pixel_scale[i]  # (2,) per-axis scale
            self._val_mre_pixel.update(p * s, g * s, task_id=tdef.task_int, mask=m)

            # Parameters in ORIGINAL PIXEL SPACE (p * s), matching the MRE line
            # above. These were previously computed on the normalized [0,1] canvas
            # coordinates, so every length and circumference was off by a
            # per-image factor of max(orig_h, orig_w) and the aggregate was not
            # comparable to anything the challenge scores.
            v = vis_mask[i].unsqueeze(0)
            pred_params = compute_params_for_evaluation(p * s, task_id_str, v)
            gt_params = compute_params_for_evaluation(g * s, task_id_str, v)
            param_mae.update(pred_params, gt_params)

        # BBox precision: batch update
        bbox_prec.update(pred_boxes, gt_boxes, task_id=tdef.task_int)

    def _log_epoch_metrics(self, stage: str) -> None:
        """Compute and log per-task MRE + ParamMAE + BBoxPrecision at epoch end."""
        mre, param_mae, bbox_prec = self._get_metrics(stage)
        sync = stage == "val"

        for key, val in mre.compute().items():
            self.log(f"{stage}/mre/{key}", val, sync_dist=sync)

        if stage == "val":
            pixel = self._val_mre_pixel.compute()
            for key, val in pixel.items():
                self.log(f"{stage}/mre_pixel/{key}", val, sync_dist=sync)

            # Flat aliases for ModelCheckpoint: Lightning turns each '/' in a
            # filename template into a directory level, and downstream tooling
            # (scripts/make_submission.py) parses checkpoint paths assuming the
            # one level that 'val/loss' produced.
            for key in ("overall", "worst_task"):
                if key in pixel:
                    self.log(f"{stage}_mre_{key}", pixel[key], sync_dist=sync)

            # Oracle gap = confidence-head ranking error. Diagnostic namespace so
            # it can never be mistaken for validation performance or monitored.
            oracle = self._oracle_mre.compute()
            for key, val in oracle.items():
                self.log(f"diagnostic/oracle_mre_pixel/{key}", val, sync_dist=sync)
            if self._inst_sep:
                sep = torch.cat(self._inst_sep)
                # min is an extreme statistic over the whole val set — one odd
                # image can drive it. p5 is the stable comparison stat; min is
                # the collapse alarm.
                self.log("diagnostic/instance_sep_min", sep.min(), sync_dist=sync)
                self.log("diagnostic/instance_sep_p5", sep.quantile(0.05), sync_dist=sync)
                self.log("diagnostic/instance_sep_mean", sep.mean(), sync_dist=sync)

            served = self._val_mre_pixel.compute().get("overall")
            if served is not None and "overall" in oracle:
                self.log(
                    "diagnostic/conf_ranking_cost",
                    served - oracle["overall"],
                    sync_dist=sync,
                )

        for key, val in param_mae.compute().items():
            self.log(f"{stage}/{key}", val, sync_dist=sync)

        for key, val in bbox_prec.compute().items():
            self.log(f"{stage}/{key}", val, sync_dist=sync)

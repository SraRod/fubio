"""Experiment configuration hierarchy — pydantic-settings.

All config sub-models are frozen (immutable after construction).
ExperimentConfig is the root, loadable from YAML + env vars.

Upstream: none (pure configuration).
Downstream: train/module.py, train/train.py consume ExperimentConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from fubio.data.mosaic import MosaicConfig
from fubio.data.transforms import TransformConfig

# ---------------------------------------------------------------------------
# Coordinate prediction mode — discriminated union
# ---------------------------------------------------------------------------


class GeoSimCCCoordConfig(BaseModel, frozen=True, extra="forbid"):
    """Geometry locates, statistics shape, local evidence refines.

    Stage 1 reads a coarse position off the patch grid (query·memory
    correlation + soft-argmax) — no feature→coordinate map is learned.
    Stage 2 places a Procrustes-aligned PCA shape via a closed-form similarity
    fitted from those points, so position comes from the image and shape from
    statistics. Stage 3 regresses a bounded offset from a patch of content
    features sampled at the current estimate.

    The stage-1 heat is returned as TaskOutput.heatmap, so `loss.lambda_heatmap`
    supervises it DIRECTLY (no prior in the sum). That direct accountability is
    the point: set lambda_heatmap > 0 or stage 1 is free to output anything and
    let stages 2-3 clean up.

    Requires shape_prior.json with canonical_mean/canonical_basis.
    """

    mode: Literal["geo_simcc"] = "geo_simcc"
    tau: float = 1.0
    # Offset bound, in patch-grid cells. One cell is ~50 original px at 224px
    # input and ~21.6 px at 518px, so 1.0 is already a wide correction window.
    window_cells: float = 1.0
    n_offset_bins: int = 32
    coarse_sigma_cells: float = 1.0  # L_coarse Gaussian width, in grid cells
    # Stage 3 scans a 1D profile along each landmark's measurement axis rather
    # than a square patch: the boundary it looks for is a 1D edge, and a square
    # patch averages the crossing away. Odd count keeps a sample on the centre.
    n_profile: int = 17
    # High-resolution map stage 3 samples, as a stride from the input. One
    # stage-1 token covers ~50 original px at 224px input while a myocardial
    # wall is ~10 px, so the offset head must read something finer than memory.
    fine_stride: int = 4
    d_fine: int = 64
    shape_weight: float = 0.5  # fixed geo/shape blend; a learned one can zero out geo
    d_local: int = 32


# Kept as a one-member discriminated union: checkpoint hyper_parameters carry
# {"mode": "geo_simcc", ...} and future readouts slot in as new members.
CoordConfig = Annotated[
    GeoSimCCCoordConfig,
    Field(discriminator="mode"),
]


class MatcherConfig(BaseModel, frozen=True):
    """Hungarian matching cost weights.

    cost_land dominates so assignment tracks landmark quality, not the
    (landmark-independent) bbox head — otherwise the matched query flips as
    the random-init bbox changes, making the landmark target a moving target
    that blocks overfitting. Set cost_land=0 to recover the legacy bbox-only
    behavior.
    """

    cost_conf: float = 1.0
    cost_box: float = 5.0
    cost_ciou: float = 2.0
    cost_land: float = 10.0


class BackboneConfig(BaseModel, frozen=True):
    """Backbone selection and freeze schedule."""

    name: str = "dinov2_vitb14"
    pretrained: bool = True
    freeze_epochs: int = 5
    input_size: tuple[int, int] = (518, 518)  # (W, H)

    @field_validator("input_size", mode="before")
    @classmethod
    def _coerce_input_size(cls, v: object) -> object:
        """Accept scalar int (backward compat) or [W, H] list/tuple."""
        if isinstance(v, int):
            return (v, v)
        return v


# ---------------------------------------------------------------------------
# Neck mode — discriminated union (Layer 2: backbone features → spatial memory)
# ---------------------------------------------------------------------------


class C2fNeckConfig(BaseModel, frozen=True):
    """C2f spatial fusion of multiple ViT blocks (RF-DETR pattern).

    Concat multi-layer features → C2f block (CSP bottleneck with 3x3 spatial
    convolutions) → d_model output. Cross-layer + cross-position nonlinear
    fusion, in place of a linear per-layer weighted sum.
    """

    mode: Literal["c2f"] = "c2f"
    layer_indices: list[int] = [2, 5, 8, 11]
    n_bottleneck: int = 2


# One-member discriminated union, same rationale as CoordConfig above.
NeckModeConfig = Annotated[
    C2fNeckConfig,
    Field(discriminator="mode"),
]


class DecoderConfig(BaseModel, frozen=True):
    """Layer 3: shared query decoder (cross-attention to spatial memory).

    Masked self-attn (within-task) + cross-attn (to memory) + FFN per layer.
    All tasks' queries share the same weights; different query embeddings
    produce different attention patterns.
    """

    n_layers: int = 3
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1


class HeadConfig(BaseModel, frozen=True):
    """Layers 4+5: per-task refiner + predictors.

    Task refiner (Layer 4): self-attention only, per-task weights.
    Predictors (Layer 5): bbox + conf + coord prediction.

    FFN width is separate from decoder because head layers are per-task (×9).
    DETR uses 2048 for shared decoder; 1024 keeps per-task cost reasonable.
    """

    n_layers: int = 2
    ffn_dim: int = 1024
    n_inst: int = 2
    dropout: float = 0.1
    derive_bbox: bool = False
    # Legacy keys, pinned to their disabled values: the AffineHead and the deep
    # confidence MLP were removed after ablation, but checkpoint
    # hyper_parameters still carry the fields.
    use_affine: Literal[False] = False
    conf_mlp_layers: Literal[1] = 1
    coord: CoordConfig = Field(default_factory=GeoSimCCCoordConfig)


class LossConfig(BaseModel, frozen=True):
    """Loss weights and toggles.

    Weights for the labeled objective. Terms that apply only to unlabeled data
    live in SemiConfig, next to the switch that loads it.

    - neg_label_smooth (L_suppress): ~0.1% of training images are split-screen
      where one side is annotated but the other side's correct detections get
      penalized as false positives. Label smoothing on unmatched conf targets
      (0→0.05) softens this penalty uniformly — simpler and more robust than
      per-sample ω estimation. Labeled-only, so it belongs here.
    """

    lambda_bbox: float = 1.0
    lambda_conf: float = 1.0
    lambda_land: float = 5.0
    lambda_param: float = 2.0
    neg_label_smooth: float = 0.0  # L_suppress: conf target for unmatched slots in labeled images
    lambda_heatmap: float = 1.0  # L_heatmap: Gaussian target CE on GeoSimCC stage 1
    lambda_shape_consistency: float = 0.0  # L_shape_cons: PCA subspace residual reg; 0 = disabled
    # Legacy keys, pinned to their disabled values. The implementations were
    # removed (falsified or superseded during the competition), but the
    # submitted checkpoint's hyper_parameters carry every field below, so the
    # names must stay accepted — and a config that tries to turn one back on
    # must fail loudly instead of silently training without the term.
    lambda_conf_rank: Literal[0.0] = 0.0
    landmark_beta: Literal[0.0] = 0.0  # pure L1, the submitted setting (paper Eq. 1)
    use_uncertainty: Literal[False] = False
    lambda_shape: Literal[0.0] = 0.0
    lambda_simcc: Literal[0.0] = 0.0
    lambda_ortho: Literal[0.0] = 0.0
    lambda_repulsion: Literal[0.0] = 0.0
    lambda_supportive: Literal[0.0] = 0.0
    lambda_evidence: Literal[0.0] = 0.0
    # L_geo: |compute_supportive(pred_scored) - GT_sup|; 0 = disabled
    lambda_geo_consistency: float = 0.0
    # L_geo_constraint: anatomical priors (ortho + axis order + chamber angle)
    lambda_geo_constraint: float = 0.0
    # L_angle_sign: penalizes A4C chambers whose predicted 上下-左右 angle crosses
    # 90° relative to GT. If GT is 85° (acute), pred at 91° (obtuse) gets heavy
    # penalty; 80° (same side) gets none from this term. Supervised only.
    lambda_angle_sign: float = 0.0
    shape_prior_path: Path = Path("data/shape_prior.json")  # canonical template source


class OptimizerConfig(BaseModel, frozen=True):
    """Optimizer and schedule parameters."""

    lr_backbone: float = 1e-5
    lr_neck: float = 1e-4
    lr_decoder: float = 1e-4
    lr_heads: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_epochs: int = 100
    eta_min_ratio: float = 0.0  # cosine 最低點 = base_lr × ratio；0 = 衰減到 0
    # Layer-wise LR decay for the ViT backbone: lr(layer) = lr_backbone × decay^(depth−layer).
    # 1.0 = disabled (single backbone group). <1 gives earlier layers a lower LR so the
    # pretrained low-level features adapt gently — the standard ViT fine-tuning recipe,
    # typically paired with freeze_epochs=0 (LLRD replaces the freeze phase).
    layer_decay: float = 1.0
    # Effective batch = data.batch_size × accumulate_grad_batches. Exists so a
    # resolution change, which is VRAM-bound and forces batch_size down, does not
    # also silently change the optimization regime. The LR schedule is unaffected:
    # it keys on trainer.estimated_stepping_batches, which already counts
    # optimizer steps rather than micro-batches.
    accumulate_grad_batches: int = 1


class MIROConfig(BaseModel, frozen=True):
    """Legacy key. MIRO was removed after ablation; the submitted checkpoint's
    hyper_parameters still carry this block (ExperimentConfig forbids unknown
    top-level keys), so it stays accepted — pinned to its disabled values."""

    lambda_miro: Literal[0.0] = 0.0
    init_variance: float = 0.1


class SWADConfig(BaseModel, frozen=True):
    """Legacy key. SWAD was removed after it regressed the prenatal tasks; the
    key stays accepted for checkpoint hyper_parameters, pinned to disabled."""

    enabled: Literal[False] = False
    n_converge: int = 3
    n_tolerance: int = 6
    tolerance_ratio: float = 0.3
    start_after_epoch: int = 0


class SemiConfig(BaseModel, frozen=True):
    """Everything semi-supervised: the switch, the loader, and its loss weights.

    enabled=False reproduces the fully-supervised path exactly: no second
    dataset is built and training_step receives a single batch.

    The weights live here rather than in LossConfig on purpose. They apply to
    exactly one data tier, they are no-ops without `enabled`, and splitting them
    across two blocks meant "is this a semi-supervised run, and with what?"
    could not be answered by reading one place.
    """

    enabled: bool = False
    # None → data.batch_size. Sized independently because the unlabeled branch
    # grows additional forward passes (teacher ref + student warped view),
    # and the labeled batch composition must not shrink to pay for it.
    batch_size: int | None = None
    # L_mil: at least one query slot must be confident about the task the image
    # is known to contain. The only positive confidence signal an unannotated
    # image can supply, since without landmarks there is no way to say WHICH
    # slot should fire. NOT absorbed by pseudo-labels — pseudo-label loss does
    # not supervise the student confidence head.
    lambda_mil: float = 0.0
    # L_eq: equivariance consistency (legacy R22, disabled by default).
    # Superseded by pseudo-label loss but kept for backward compat / ablation.
    lambda_eq: float = 0.0
    # L_pseudo: EMA teacher pseudo-label landmark loss. Teacher forward on
    # clean unlabeled image produces pseudo-targets; student forward on
    # geometrically + photometrically augmented view learns from them.
    # Confidence-weighted with fixed denominator (absolute attenuation).
    lambda_pseudo: float = 0.0
    # EMA teacher momentum. Updated every optimizer step via warm-up:
    # alpha_eff = min(1 - 1/(ema_step + 1), alpha_ema).
    # Converges to alpha_ema around step ~1/( 1 - alpha_ema).
    alpha_ema: float = 0.999
    # Pseudo-label loss ramp: 0 before ramp_start, linear 0→1 from
    # ramp_start to ramp_end, 1.0 after. In epochs.
    ramp_start: int = 2
    ramp_end: int = 7
    # Affine transform bounds for equivariance / pseudo-label views
    eq_rotation_range: float = 30.0
    eq_scale_range: tuple[float, float] = (0.8, 1.2)
    eq_translate_range: float = 0.1
    # L_eq ramp: equivariance consistency loss. Separate ramp so it starts
    # after pseudo-label has stabilized. eq_ramp_start/end are in epochs.
    eq_ramp_start: int = 7
    eq_ramp_end: int = 12
    # Per-task scale override for pseudo-label kornia affine. Tasks not listed
    # use eq_scale_range. Dict of task_id → [lo, hi].
    task_eq_scale_override: dict[str, tuple[float, float]] = {}
    # Horizontal flip probability for kornia views (pseudo + eq paths).
    # All tasks have empty flip_pairs (n_inst=1), so flip only mirrors
    # x coordinates without landmark index permutation.
    flip_prob: float = 0.0
    # Which manifest splits to include as unlabeled data.
    # Default: only train_unlabeled. Add "val" for transductive semi-supervised.
    unlabeled_splits: list[str] = ["train_unlabeled"]
    # Pseudo-param: L1 on clinical parameters derived from teacher vs student
    # landmarks. Directly targets MAE (50% of competition score).
    lambda_pseudo_param: float = 0.0
    # Pseudo geo_consistency: supportive landmarks computed from student's scored
    # predictions must match those from teacher's scored predictions (warped frame).
    lambda_pseudo_geo: float = 0.0
    # Per-task pseudo weight override. Default 1.0 for unlisted tasks.
    # Scales ALL pseudo-derived terms (landmark + param + geo) for that task.
    task_pseudo_lambda: dict[str, float] = {}
    # Tasks where kornia flip is forced off in pseudo/eq views.
    no_flip_tasks: list[str] = []
    # TTA Teacher: average teacher landmarks across multiple views for more
    # stable pseudo-targets. Empty list = single view (default). E.g. [5, -5]
    # runs original + ±5° rotation, averages in clean [0,1] space.
    # Cost: len(angles) extra teacher forwards (all no_grad, cheap).
    tta_teacher_angles: list[float] = []
    # Photometric aug (kornia ColorJiggle, applied in [0,1] range on student
    # input only). Teacher sees clean image.
    photo_brightness: float = 0.25
    photo_contrast: float = 0.25

    @model_validator(mode="after")
    def _weights_require_enabled(self) -> SemiConfig:
        """A weight set without the loader is a silent no-op, not a soft error."""
        has_weight = (
            self.lambda_mil > 0
            or self.lambda_eq > 0
            or self.lambda_pseudo > 0
            or self.lambda_pseudo_param > 0
            or self.lambda_pseudo_geo > 0
        )
        if has_weight and not self.enabled:
            raise ValueError(
                "Semi-supervised lambdas > 0 require semi.enabled = true."
            )
        return self


class HeadTuneConfig(BaseModel, frozen=True):
    """Surgical head-tuning: freeze shared components, retrain selected TaskModules.

    When tune_tasks is non-empty, backbone/neck/decoder/shared projections and
    all TaskModules NOT in tune_tasks are frozen. Only the listed tasks receive
    gradients. reinit_tasks (a subset) have their learned params reinitialized
    before training — buffers (shape prior, partner) are preserved.

    lr_scale overrides the LR for specific tasks relative to lr_heads.
    Tasks not listed default to 1.0.
    """

    tune_tasks: list[str] = []
    reinit_tasks: list[str] = []
    lr_scale: dict[str, float] = {}

    @model_validator(mode="after")
    def _reinit_subset_of_tune(self) -> HeadTuneConfig:
        if not set(self.reinit_tasks).issubset(set(self.tune_tasks)):
            raise ValueError(
                f"reinit_tasks {self.reinit_tasks} must be a subset of "
                f"tune_tasks {self.tune_tasks}"
            )
        for tid in self.lr_scale:
            if tid not in self.tune_tasks:
                raise ValueError(
                    f"lr_scale key '{tid}' not in tune_tasks {self.tune_tasks}"
                )
        return self

    @property
    def enabled(self) -> bool:
        return len(self.tune_tasks) > 0


class LoggingConfig(BaseModel, frozen=True):
    """Logging and experiment tracking."""

    project: str = "fubio"


class DataConfig(BaseModel, frozen=True):
    """Data loading, augmentation, and mosaic parameters."""

    transform: TransformConfig = TransformConfig()
    mosaic: MosaicConfig = MosaicConfig()
    batch_size: int = 32
    num_workers: int = 10
    pin_memory: bool = True
    val_fraction: float = 0.15
    # None = one epoch is the full sampler pass. Capping is a smoke-run tool and
    # must be asked for explicitly (config value or --limit-train-batches): as a
    # DEFAULT it silently starved any config that forgot to override it, and the
    # rules require all ~191K unlabeled images to be used — which a silent cap
    # would make impossible to demonstrate.
    limit_train_batches: int | float | None = None
    # Fold val_local into training for final push. No local GT validation —
    # use semi metrics (pseudo_loss + equivariance diagnostic) for checkpoint
    # selection, or online submissions.
    fold_val_local: bool = False
    da_open_epoch: int | None = None
    # Epoch at which BOTH flips stop (None = never). Mirrors mosaic's
    # close_epoch so training ends on the un-augmented distribution.
    flip_close_epoch: int | None = None


class ExperimentConfig(BaseSettings, frozen=True):  # type: ignore[reportGeneralTypeIssues]
    """Root config — composable from YAML + env vars via pydantic-settings."""

    d_model: int = 256
    backbone: BackboneConfig = BackboneConfig()
    neck: NeckModeConfig | None = None  # None → auto from backbone name
    decoder: DecoderConfig = DecoderConfig()
    head: HeadConfig = HeadConfig()
    loss: LossConfig = LossConfig()
    matcher: MatcherConfig = MatcherConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    miro: MIROConfig = MIROConfig()
    swad: SWADConfig = SWADConfig()
    semi: SemiConfig = SemiConfig()
    head_tune: HeadTuneConfig = HeadTuneConfig()
    logging: LoggingConfig = LoggingConfig()
    data: DataConfig = DataConfig()
    seed: int = 42
    precision: str = "bf16-mixed"
    cudnn_benchmark: bool = False
    neck_dropout: float = 0.1

    @model_validator(mode="after")
    def _default_neck(self) -> ExperimentConfig:
        """Default the neck when not explicitly configured."""
        if self.neck is None:
            object.__setattr__(self, "neck", C2fNeckConfig())
        return self

    @model_validator(mode="after")
    def _validate_heatmap_supervision(self) -> ExperimentConfig:
        # geo_simcc without lambda_heatmap is a silent failure mode: stage 1
        # would be free to emit anything and let stages 2-3 compensate, which
        # is the exact defect this readout exists to remove.
        if self.loss.lambda_heatmap <= 0:
            raise ValueError(
                "coord.mode = 'geo_simcc' requires lambda_heatmap > 0 — without it the "
                "geometric coarse stage is unsupervised and the readout is pointless"
            )
        return self

    @model_validator(mode="after")
    def _validate_input_size_matches_transform(self) -> ExperimentConfig:
        """Preprocessing size and model size are declared in two places; pin them.

        Training resizes via data.transform.target_size, but serving rebuilds a
        canvas from backbone.input_size (serving/predict.py). Nothing previously
        required the two to agree, so a checkpoint could encode preprocessing
        that inference would not reproduce — and the mismatch would surface only
        as quietly degraded predictions.
        """
        expected = tuple(self.backbone.input_size)  # (W, H)
        actual = tuple(self.data.transform.target_size)
        if actual != expected:
            raise ValueError(
                f"data.transform.target_size {actual} must equal "
                f"backbone.input_size = {expected}. Serving reconstructs a "
                f"canvas from backbone.input_size, so a mismatch means "
                f"inference silently preprocesses differently from training."
            )
        return self

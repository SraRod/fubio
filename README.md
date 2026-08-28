# FUBio — A Scalable Foundation for Nine-Task Ultrasound Biometry

Our entry to the **MICCAI 2026 Challenge on Foundation Model for Ultrasound
Biometry**: one model that performs 9 biometry tasks across prenatal,
intrapartum and cardiac ultrasound — 60 landmarks, 28 clinical parameters —
trained semi-supervised over the challenge's ~191K unlabeled images alongside
6,756 labeled ones. The rules require one unified model, so there are no
per-task networks here.

The method is described in the paper *"A Scalable Foundation for Nine-Task
Ultrasound Biometry: Domain Knowledge as Representation and Regularization"*
(in submission; see [Citation](#citation)). In short: one shared
representation (DINOv2 backbone + C2f neck, 107.11M of 145.62M parameters),
a compact 4.28M module per task whose queries attend to it, and domain
knowledge on both sides of learning — each task's statistical shape prior
initializes its queries and forms one readout stage
(GEO–SHAPE–FINE, `GeoSimCCPredictor`), while boxes, supportive points and
anatomical constraints are derived differentiably from the predicted
landmarks and the 28 clinical measurements are optimized directly.

**Final result: rank 6 of 31** (Board A), with the best test value in 6 of
the 18 per-task task–metric slots — more than any other team — and rank 2 of
31 on the normalized clinical-parameter dimension against rank 16 on landmark
localization. Team `sralin`, Codabench submission `890227`.

This README is a complete reproduction guide: environment, data, the
three-stage training that produced the submitted weights, weight extraction,
and the Docker image that was graded. An engineering reference for
`src/fubio/` follows it.

## 1. Environment

Python 3.12+, [uv](https://docs.astral.sh/uv/), dependencies pinned in
`uv.lock`:

```bash
uv sync
```

Every command below runs from the repository root (paths like `data/…` are
resolved relative to the working directory).

## 2. Data

Training needs the challenge dataset, distributed by the organizers under
CC BY-NC — it is not redistributed here. Arrange it as:

```
data/
├── Data/<task>/labeled/         # official download: 9 task folders with CSVs + images
├── images/<task>/               # the unlabeled pool: extract Data/<task>/unlabeled/*.zip here
├── val/<task>/                  # the validation-phase images: extract val_data.zip here
├── quality_flags.json           # committed — 12 excluded images (10 A4C, 2 PLAX)
└── shape_prior_v4.json          # committed — the exact prior the submitted weights use
```

with `<task>` ∈ {A4C, AOP, FA, FUGC, HC, IVC, PLAX, PSAX, fetal_femur}.

Then build the manifest, split, and prior:

```bash
# 9 heterogeneous CSVs -> one manifest.parquet. Applies quality_flags.json
# automatically (6,768 -> 6,756 labeled images) and handles the released
# data's quirks: GBK encoding, transposed FA columns, HC path prefixes.
uv run python -m fubio.data.build_manifest

# Group-aware stratified train_local/val_local split. Deterministic under
# --seed. (Skipping this is fine: training runs it on first launch.)
uv run python -m fubio.data.split stratified --seed 42

# Per-task PCA shape prior from the train_local annotations. Training
# verifies the prior's recorded manifest hash, so after building your own
# manifest you must also build your own prior — the committed
# data/shape_prior_v4.json is the exact prior of the SUBMITTED weights
# (byte-identical copy preserved in docker/data/), recorded against our
# manifest, which a rebuilt manifest cannot reproduce bit-for-bit.
uv run python -m fubio.data.build_shape_prior --output data/shape_prior_v4.json

# Optional but strongly recommended: pre-decode every image into a memmap
# cache (~80x faster dataloading; ~50 GB at 518px).
uv run python -m fubio.data.build_cache --manifest data/manifest.parquet \
    --data-root data --target-size 518 --resize-mode letterbox --workers 10
```

## 3. Experiment tracking (optional)

Training logs to [wandb](https://wandb.ai). No account is needed:

```bash
uv run python -m fubio.train.train fit --wandb-offline ...   # local logs only
```

or export `WANDB_MODE=offline` (or `disabled`). With an account, `wandb login`
once and drop the flag. Checkpoints land in
`wandb_logs/fubio/<run_id>/checkpoints/` either way, and
`wandb_logs/fubio/<run_id>/metrics.csv` mirrors the metrics without wandb.

## 4. Training — the three-stage lineage of the submitted weights

The submitted model is the end of a fixed lineage; each stage initializes
from the previous stage's checkpoint via `--init-weights`. Paper Table 4
reports the public-validation score after each stage.

| Stage | Config | Paper | What it does |
|---|---|---|---|
| 1 | `stage1.yaml` | row 1 | Supervised base: 160 epochs on the labeled set |
| 1b (optional) | `stage1-head-tune.yaml` | row 1 | Surgical repair of collapsed task modules |
| 2 | `stage2.yaml` | row 2 | Semi-supervised round 1: EMA teacher + pseudo-labels, 30 epochs |
| 3 | `stage3.yaml` | row 3 | Semi-supervised round 2: pseudo-weight ×2, val folded in, 15 epochs — **submitted checkpoint is epoch 007, EMA teacher** |

```bash
# Stage 1 — supervised base (DINOv2 init, nothing to load)
uv run python -m fubio.train.train fit --config configs/stage1.yaml

# Stage 1b — ONLY if stage 1 leaves individual task modules far behind the
# rest (in our lineage IVC and PSAX had collapsed to 94/25 px while every
# other task was at its best). Everything shared stays frozen; the tasks
# named in head_tune.tune_tasks are repaired. Skip when stage 1 is balanced.
uv run python -m fubio.train.train fit --config configs/stage1-head-tune.yaml \
    --init-weights wandb_logs/fubio/<stage1_run>/checkpoints/<best>.ckpt

# Stage 2 — semi-supervised round 1, from the stage-1 (or 1b) checkpoint
uv run python -m fubio.train.train fit --config configs/stage2.yaml \
    --init-weights wandb_logs/fubio/<stage1b_run>/checkpoints/<best>.ckpt

# Stage 3 — semi-supervised round 2, from the stage-2 checkpoint
uv run python -m fubio.train.train fit --config configs/stage3.yaml \
    --init-weights wandb_logs/fubio/<stage2_run>/checkpoints/<best>.ckpt
```

Checkpoint selection: stages 1–2 hold out `val_local` and name their
checkpoints by `val_mre_overall` — pick the lowest. Stage 3 folds the split
back into training (`fold_val_local: true`), so its checkpoints are named by
an unlabeled-loss statistic (`semi/loss_total`) instead; the submitted
checkpoint is epoch 007. Historical note: our stage 1 ran as 150 epochs plus
a 10-epoch resume; `stage1.yaml` reproduces the same recipe as a single
160-epoch launch (equivalent recipe, not step-identical learning-rate
history). `--epochs N` and `--limit-train-batches N` shrink any stage for a
smoke run.

## 5. Weight extraction — teacher or student

A Lightning checkpoint is ~2.2 GB (student + EMA teacher + optimizer). The
deployable file is ~583 MB and contains exactly what the container loads:
`hyper_parameters` + `teacher_state_dict` (bare `FUBioModel` keys):

```bash
# The submitted configuration: the EMA teacher
uv run python scripts/extract_weights.py \
    --ckpt "wandb_logs/fubio/<stage3_run>/checkpoints/<epoch7>.ckpt" \
    --output docker/best_model.pth

# The student weights, in the same container-loadable format
uv run python scripts/extract_weights.py --ckpt <ckpt> --source student --output student.pth
```

Applied to our archived stage-3 checkpoint, the extracted file is
bit-identical to the released `best_model.pth` (508/508 tensors equal,
identical hyper_parameters).

### Released weights

`best_model.pth` (~583 MB) — the EMA teacher of the final stage-3 run at
epoch 7, exactly the weights inside the graded container. Too large for git:
download it from this repository's **Releases** page and place it at
`docker/best_model.pth`.

## 6. Docker — building the graded image

`docker/predict.py` is the organizers' entry script, deliberately unmodified;
`docker/model.py` is our wrapper. The image runs fully offline — DINOv2 is
vendored under `docker/vendor_dinov2/` — and reads images from
`$GU_INPUT_DIR`, writing landmark predictions to `$GU_OUTPUT_DIR` per the
challenge contract.

```bash
# needs docker + nvidia-container-toolkit, and docker/best_model.pth in place
sudo bash docker/build_and_test.sh
```

The script assembles the build context (`docker/fubio/` is generated from
`src/fubio/` at build time, never versioned), builds
`sralin/fu-biometry:v1.0`, and — when `data/val/` is present — runs the
container on the validation images twice, checking output integrity and
determinism, and comparing against reference predictions when available. The
submitted image itself is identified by its Docker Hub digest, recorded at
submission time.

Two further scripts close the loop from a checkpoint without Docker:

```bash
# Re-run the validation loop on a checkpoint (regression guard)
uv run python scripts/validate_checkpoint.py --ckpt <ckpt> --device cuda

# Produce a submission.zip + prediction JSONs from a checkpoint
uv run python scripts/make_submission.py --ckpt <ckpt> --tag my-run --model-source teacher
```

## License

MIT — see `LICENSE`. DINOv2 is vendored under Apache-2.0; its provenance and
the reason the unused non-commercial license files are kept is documented in
[`docker/vendor_dinov2/VENDOR_NOTICE.md`](docker/vendor_dinov2/VENDOR_NOTICE.md).
The challenge data is CC BY-NC and is not part of this repository.

## Citation

The method paper is under submission; this section will carry the reference
once it is available.

---

The remainder of this document is the engineering reference for `src/fubio/`:
the layer structure, the data contract every stage of the pipeline honors, and
the design decisions behind them.

## Architecture

Five layers. One model, nine tasks — the challenge rules forbid per-task
networks, so task specificity is confined to layer 5.

| Layer | Module | Role | Per-task? |
|---|---|---|---|
| 1 · Backbone | `models/backbone.py` | DINOv2 ViT-B/14 at 518×518 native → 37×37 patch tokens from blocks [2, 5, 8, 11] | shared |
| 2 · Neck | `models/neck.py` | C2f fusion of the four block outputs into one spatial memory + sinusoidal 2D positional encoding | shared |
| 3 · Decoder | `models/decoder.py` | cross-attention from queries into that memory | shared |
| 4 · Refiner | `models/decoder.py` (`TaskRefinerLayer`) | self-attention among one task's queries | per-task |
| 5 · TaskModule | `models/heads.py` + `models/coord_predictors.py` | owns the task's query embeddings, anchor positions and the GEO–SHAPE–FINE readout | per-task |

Adding a task means registering one `TaskDef`; the model then creates its
`TaskModule` automatically. The shared decoder has no knowledge of tasks at
all.

The coordinate readout is `GeoSimCCPredictor` (paper Section 2.4): stage 1
(GEO) reads a coarse position off the patch grid by query–memory correlation
and soft-argmax; stage 2 (SHAPE) predicts PCA deformation coefficients and
places the resulting shape with a closed-form similarity fit to the stage-1
points, fused at a fixed 0.5 weight; stage 3 (FINE) samples a 1D profile of
high-resolution content features along each landmark's measurement axis and
adds a bounded offset. Six earlier readouts explored during development
(direct, ref-points, shape-prior, heatmap, SimCC, shape-SimCC) were retired
from the codebase; the git history of the development repository holds them.

### The canonical K=60 representation

Nine tasks own between 2 and 22 landmarks each. Rather than carry ragged
per-task arrays through the pipeline, every sample carries a **fixed `[60, 2]`
canonical array**, with each task occupying its own contiguous slice
(`data/task_registry.py`, `K = 60`). Three boolean masks disambiguate it:

| Mask | Meaning |
|---|---|
| `landmark_valid_mask` | slots this task owns |
| `landmark_supervised_mask` | owned **and** the CSV gives finite coordinates |
| `landmark_visible_mask` | still inside the frame after augmentation |

The cost is a padded tensor; the benefit is that collation is a plain stack,
augmentation is shape-invariant, and a mosaic tile from A4C and one from IVC
are the same shape and compose without special-casing. `local_to_canonical()`
does the conversion at the manifest boundary, so nothing downstream handles
per-task shapes.

## Module map

```
src/fubio/
├── data/            pure PyTorch, framework-independent — imports nothing from train/
│   ├── task_registry.py       canonical K=60 landmark schema — single source of truth
│   ├── ordering_schema.py     landmark ordering data models — pure schema, no registry dependency
│   ├── landmark_ordering.py   bidirectional competition <-> canonical landmark conversion
│   ├── types.py               shared type contracts for the pipeline
│   ├── build_manifest.py      9 heterogeneous CSVs -> one manifest.parquet
│   ├── build_cache.py         pre-decode and pre-resize every image into a numpy memmap
│   ├── build_shape_prior.py   PCA shape prior from the labeled training landmarks
│   ├── manifest.py            manifest -> decoded SampleDict (disk or memmap; ~80x apart)
│   ├── transforms.py          spatial + photometric augmentation, images stay HWC
│   ├── spatial.py             affine transforms — compose once, apply once
│   ├── mosaic.py              configurable R×C grid composition on a shared canvas
│   ├── sampler.py             balanced multi-task batch sampler for uneven task sizes
│   ├── collate.py             SampleDicts -> tensor batches
│   ├── split.py               train/val split builders
│   ├── shape_prior.py         PCA shape prior for per-task landmark structure
│   ├── supportive.py          supportive landmark computation from scored landmarks
│   ├── supportive_torch.py    the same, differentiable (the geo-consistency loss)
│   └── loaders.py             wires Dataset -> (Mosaic) -> Transform -> DataLoader
│
├── models/          pure nn.Module — imports nothing from Lightning
│   ├── backbone.py            DINOv2 wrapper producing multi-level token features
│   ├── neck.py                Layer 2 — C2f fusion -> spatial memory (+ 2D pos enc)
│   ├── decoder.py             Layers 3 and 4 — cross-attention decoder, per-task refiner
│   ├── queries.py             query position encoding utilities
│   ├── heads.py               TaskModule — query ownership, refinement, prediction
│   ├── coord_predictors.py    the GeoSimCC readout and its geometry helpers
│   └── model.py               FUBioModel — assembles the layers into per-task TaskOutputs
│
├── train/           Lightning layer — serving/ never imports this at runtime
│   ├── config.py              ExperimentConfig hierarchy (pydantic-settings, YAML + env)
│   ├── datamodule.py          real-data LightningDataModule over the data/ pipeline
│   ├── mock_datamodule.py     synthetic generator for end-to-end pipeline tests
│   ├── module.py              FUBioModule — single-pass multi-task training step + EMA teacher
│   ├── losses.py              landmark / detection / parameter / consistency losses
│   ├── matcher.py             per-task Hungarian matching, instance queries <-> GT
│   ├── views.py               differentiable geometric views for semi-supervised consistency
│   ├── schedule.py            phase-aware warmup + cosine, with backbone freeze
│   ├── callbacks.py           collapse guard, final-epoch checkpoint, CSV logging
│   ├── viz_callback.py        validation prediction grids
│   └── train.py               CLI entry point
│
├── evaluation/      shared by training and inference
│   ├── geometry.py            differentiable clinical parameter geometry, 0 learnable params
│   ├── metrics.py             MRE, parameter MAE, bbox precision
│   └── postprocessing.py      model output -> original pixel coordinates -> submission JSON
│
└── serving/         inference — no Lightning dependency on the graded path
    ├── predict.py             checkpoint + images -> submission JSON (TTA, rerouting)
    └── validate.py            submission integrity checks — every failure is loud
```

## Data flow

```
9 per-task CSVs (task-local landmark order)
  │
  ▼ data/build_manifest.py        GBK encoding, HC path prefixes, bracketed coordinates
manifest.parquet
  │
  ▼ data/manifest.py              local_to_canonical() happens here, and only here
SampleDict
  image             [H, W, 3] uint8, BGR->RGB
  keypoints         [1, 60, 2] float32, NaN for slots this task does not own
  transform_matrix  [1, 3, 3] identity
  landmark_{valid,supervised,visible}_mask, task_ids, is_labeled, original_hws
  │
  ▼ data/mosaic.py (optional)     tiles may come from different tasks — same shape, so they just stack
SampleDict with I instances on one canvas
  │
  ▼ data/transforms.py            albumentations; updates transform_matrix and visible_mask
SampleDict, augmented
  │
  ▼ data/collate.py
batch
  image      [B, 3, H, W]
  keypoints  [B, I_max, 60, 2] float32, NaN -> 0.0 (masks carry the truth)
```

`transform_matrix` accumulates every spatial operation, so
`evaluation/postprocessing.py` can invert the whole chain in one step and emit
coordinates in the organizers' original pixel space. Predictions are submitted
as **pixel coordinates only** — the organizers hold the pixel spacing and
convert to millimetres server-side, which is why the clinical parameter layer
is trained in pixel space and is scale-invariant.

## Repository layout

```
src/fubio/    the engine
configs/      the submitted lineage: stage1 / stage1-head-tune / stage2 / stage3
docker/       the inference container recipe (docker/fubio/ is generated at build time)
scripts/      extract_weights.py, validate_checkpoint.py, make_submission.py
tests/        pytest + hypothesis (equivariance property tests)
data/         committed: quality_flags.json, shape_prior_v4.json.
              Not committed: the ~19 GB challenge dataset (CC BY-NC)
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Landmark representation | Canonical `[60, 2]` + three masks | Uniform shape makes collation a stack and lets mosaic tiles from different tasks compose without special-casing. Padding is the price |
| Task extension | Register one `TaskDef` | The model auto-creates the `TaskModule`. No existing definition is touched |
| Architecture | Shared decoder + per-task TaskModule | The rules require one unified model. Task specificity is confined to layer 5, where it is cheapest |
| Backbone | DINOv2 ViT-B/14, 518×518 native | 37×37 patches at native resolution; self-supervised features transferred to ultrasound better than the ConvNeXt V2 line we also tried |
| Coordinate readout | GeoSimCC (GEO–SHAPE–FINE) | Position is read from grid geometry, shape is regularized toward a statistical subspace, and sub-patch detail is paid for only where needed |
| Clinical parameters | Differentiable, in pixel space | The organizers convert to millimetres server-side; gradients flow into the landmark layer |
| Detection matching | Per-task Hungarian | Instance queries are matched within a task, never across |
| Augmentation | albumentations in the data pipeline | Mature keypoint edge-case handling. kornia is reserved for the differentiable on-GPU path in `train/views.py` |
| Flips | Unswapped, p=0.25 each | Each landmark index names one consistent point, so a flip asks the model to find that point on mirrored anatomy — positional-shortcut disruption (paper Section 3.2) |
| Precision | bf16-mixed | Same throughput on A100 without loss scaling |
| Config | pydantic-settings, YAML + env | Validated at load; retired options are pinned `Literal` legacy keys, so old checkpoints reload and re-enabling them fails loudly |
| Dataframes | polars | Strict schemas catch dirty CSV values at the manifest boundary rather than at training time |

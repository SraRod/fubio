# FUBio — unified ultrasound biometry landmark detection

Our entry to the **MICCAI 2026 FU_Biometry (FUB) Challenge**: a single model
that detects biometry landmarks across prenatal, intrapartum and cardiac
ultrasound — 9 tasks, 60 landmarks, 28 clinical parameters — trained
semi-supervised over the challenge's ~191K unlabeled images alongside the
labeled set. The rules require one unified model, so there are no per-task
networks here.

**Preliminary result: rank 6 of 31** — rank 2 on the clinical-parameter error
dimension, rank 16 on the landmark error dimension. Team `sralin`, Codabench
submission `890227`.

## Installation

Python 3.12+, dependencies pinned in `uv.lock`:

```bash
uv sync
```

## Pretrained weights

`best_model.pth` (556 MB) — the EMA teacher weights of run `lkzjwmx6` epoch 7,
i.e. exactly the checkpoint inside the submitted container. It is too large for
git; download it from this repository's **Releases** page and put it at
`docker/best_model.pth`.

## Inference — the container that was graded

`docker/` is the submitted image, unchanged. `docker/predict.py` is the
organizers' entry script and is deliberately unmodified; `docker/model.py` is
our wrapper around it. The image runs fully offline — DINOv2 is vendored under
`docker/vendor_dinov2/` rather than fetched at runtime.

```bash
sudo bash docker/build_and_test.sh
```

It reads images from `$GU_INPUT_DIR` and writes landmark predictions to
`$GU_OUTPUT_DIR`, per the challenge's submission contract.

`docker/fubio/` is a byte-identical inference-only subset of `src/fubio/` — the
Dockerfile copies the 28 modules inference needs so the image carries no
training code. Two lazy imports inside that subset (`fubio.data.build_cache` in
`data/manifest.py`, `fubio.train.module` in `serving/predict.py`) therefore do
not resolve inside the image. Both sit on branches the container never
executes — the memmap cache path and `load_model()`, neither of which
`model.py` calls — so the image runs correctly. They are left in place because
this directory is the artifact that was graded, and it is shipped unedited.

## Training

```bash
uv run python -m fubio.data.build_manifest          # 9 heterogeneous CSVs -> one parquet
uv run python -m fubio.train.train fit --config configs/r45-vitb-518px.yaml
```

The submitted model is the end of a five-stage lineage; each stage initialises
from the previous one's checkpoint via `--init-weights` (see the comment header
in each config):

| Stage | Config | What it does |
|---|---|---|
| 1 | `r45-vitb-518px.yaml` | Supervised base — DINOv2-B/14 at 518px, GeoSimCC heads |
| 2 | `r45-resume.yaml` | Continue to epoch 160 |
| 3 | `r47-head-tune.yaml` | Per-task head repair — IVC/PSAX reinit, A4C refine |
| 4 | `r47-semi.yaml` | Semi-supervised round 1 over the unlabeled pool |
| 5 | `r48-semi.yaml` | Semi-supervised round 2 — **submitted model is epoch 7, EMA teacher** |

Training needs the challenge dataset, which the organizers distribute under
CC BY-NC; it is not redistributed here.

## Licence

MIT — see `LICENSE`. DINOv2 is vendored under Apache-2.0; its provenance and the
reason the unused non-commercial licence files are kept is documented in
[`docker/vendor_dinov2/VENDOR_NOTICE.md`](docker/vendor_dinov2/VENDOR_NOTICE.md).

## Citation

A method paper is in preparation; this section will carry the reference once it
is available.

---

The remainder of this document is the engineering reference for `src/fubio/`:
the layer structure, the data contract every stage of the pipeline honours, and
the design decisions behind them.

## Architecture

Five layers. One model, nine tasks — the challenge rules forbid per-task
networks, so task specificity is confined to layer 5.

| Layer | Module | Role | Per-task? |
|---|---|---|---|
| 1 · Backbone | `models/backbone.py` | DINOv2 ViT-B/14 at 518×518 native → 37×37 patch tokens | shared |
| 2 · Neck | `models/neck.py` | fuse backbone features into one spatial memory + sinusoidal 2D positional encoding | shared |
| 3 · Decoder | `models/decoder.py` | cross-attention from queries into that memory | shared |
| 4 · Refiner | `models/decoder.py` (`TaskRefinerLayer`) | self-attention among one task's queries | per-task |
| 5 · TaskModule | `models/heads.py` + `models/coord_predictors.py` | owns the task's query embeddings, anchor positions and coordinate predictor | per-task |

Adding a task means registering one `TaskDef`; the model then creates its
`TaskModule` automatically. The shared decoder has no knowledge of tasks at all.

Layer 5's coordinate predictor is swappable by config —
`DirectPredictor`, `RefPointsPredictor`, `ShapePriorPredictor`,
`HeatmapPredictor`, `SimCCPredictor`, `ShapeSimCCPredictor`,
`GeoSimCCPredictor`. The submitted model uses **GeoSimCC**.

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
augmentation is shape-invariant, and a mosaic tile from A4C and one from IVC are
the same shape and compose without special-casing. `local_to_canonical()` does
the conversion at the manifest boundary, so nothing downstream handles per-task
shapes.

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
│   ├── supportive.py          supportive landmark computation and evidence detection
│   ├── supportive_torch.py    the same, differentiable
│   └── loaders.py             wires Dataset -> (Mosaic) -> Transform -> DataLoader
│
├── models/          pure nn.Module — imports nothing from Lightning
│   ├── backbone.py            backbone wrappers producing structured multi-level features
│   ├── neck.py                Layer 2 — backbone features -> spatial memory (+ 2D pos enc)
│   ├── decoder.py             Layers 3 and 4 — cross-attention decoder, per-task refiner
│   ├── queries.py             query position encoding utilities
│   ├── heads.py               TaskModule — query ownership, refinement, prediction
│   ├── coord_predictors.py    the seven interchangeable coordinate predictors
│   └── model.py               FUBioModel — assembles the layers into per-task TaskOutputs
│
├── train/           Lightning layer — serving/ never imports this
│   ├── config.py              ExperimentConfig hierarchy (pydantic-settings, YAML + env)
│   ├── datamodule.py          real-data LightningDataModule over the data/ pipeline
│   ├── mock_datamodule.py     synthetic generator for end-to-end pipeline tests
│   ├── module.py              FUBioModule — single-pass multi-task training step
│   ├── losses.py              landmark / detection / parameter / consistency losses
│   ├── matcher.py             per-task Hungarian matching, instance queries <-> GT
│   ├── views.py               differentiable geometric views for semi-supervised consistency
│   ├── regularizer.py         MIRO — mutual information regularization with oracle
│   ├── schedule.py            phase-aware warmup + cosine, with backbone freeze
│   ├── callbacks.py           CSV logging and SWAD weight averaging
│   ├── viz_callback.py        validation prediction grids
│   └── train.py               CLI entry point
│
├── evaluation/      shared by training and inference
│   ├── geometry.py            differentiable clinical parameter geometry, 0 learnable params
│   ├── metrics.py             MRE, parameter MAE, bbox precision
│   └── postprocessing.py      model output -> original pixel coordinates -> submission JSON
│
├── serving/         inference — no Lightning dependency on the graded path
│   ├── predict.py             checkpoint + images -> submission JSON
│   ├── validate.py            submission integrity checks — every failure is loud
│   └── viewer_export.py       all-instance viewer cache from a trained checkpoint
│
└── viz/
    └── overlay.py             shared plotly overlay drawing for landmark geometry
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
convert to millimetres server-side, which is why the clinical parameter layer is
trained in pixel space and is scale-invariant.

## Repository layout

```
src/fubio/    the engine
configs/      training configs (pydantic-settings). Only the submitted model's lineage is here
docker/       the inference container, identical to the one that was graded
scripts/      make_submission.py, validate_checkpoint.py
tests/        pytest + hypothesis (equivariance property tests)
data/         ~18 GB of images — organizer-distributed under CC BY-NC, not redistributed here
```

The development environment also holds `notebooks/` (marimo exploration), a
streamlit instance viewer, and the organizers' baseline code. None of that is
part of the method, so none of it is released.

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Landmark representation | Canonical `[60, 2]` + three masks | Uniform shape makes collation a stack and lets mosaic tiles from different tasks compose without special-casing. Padding is the price |
| Task extension | Register one `TaskDef` | The model auto-creates the `TaskModule`. No existing definition is touched |
| Architecture | Shared decoder + per-task TaskModule | The rules require one unified model. Task specificity is confined to layer 5, where it is cheapest |
| Backbone | DINOv2 ViT-B/14, 518×518 native | 37×37 patches at native resolution; self-supervised features transfer to ultrasound better than the ConvNeXt V2 line we also tried |
| Coordinate head | GeoSimCC (of seven interchangeable predictors) | Config-swappable, so head choice is an experiment rather than a rewrite |
| Clinical parameters | Differentiable, in pixel space | The organizers convert to millimetres server-side; gradients flow into the landmark layer |
| Detection matching | Per-task Hungarian | Instance queries are matched within a task, never across |
| Augmentation | albumentations in the data pipeline | Mature keypoint edge-case handling. kornia is reserved for the differentiable on-GPU path in `train/views.py` |
| Flip | Disabled for symmetric two-point tasks | hflip/vflip swaps endpoints whose ground-truth order is otherwise stable, injecting label noise |
| Precision | bf16-mixed | Same throughput on A100 without loss scaling |
| Config | pydantic-settings, YAML + env | Validated at load; no dataclasses anywhere in the codebase |
| Dataframes | polars | Strict schemas catch dirty CSV values at the manifest boundary rather than at training time |

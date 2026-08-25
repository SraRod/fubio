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

The remainder of this document is the engineering reference for `src/fubio/` —
module responsibilities, the per-task instance data flow, and the design
decisions behind them.

## `src/fubio/` Module Structure

```
src/fubio/
├── data/                               # Pure PyTorch, framework-independent
│   │                                   # Lightning / serving 都 import 這裡，但這裡不 import 它們
│   │                                   # Task owns its landmarks — 不存在 global K constant
│   │
│   ├── task_registry.py                # Task-centric schema — single source of truth
│   │                                   #   TaskDef: task_id, task_int, domain, n_keypoints,
│   │                                   #     landmark_names, flip_pairs, allow_hflip
│   │                                   #   ParamDef: clinical parameters (formula + landmark indices)
│   │                                   #   TASKS: dict[str, TaskDef] — 9 tasks 的完整定義
│   │                                   #   擴增 task = 加一筆 TaskDef，不改任何既有定義
│   │                                   #   per_task_flip_permutation(task_id) → int[K_task]
│   │                                   #   [REMOVED] K=60, offset, local_to_canonical,
│   │                                   #     landmark_valid_mask, global_flip_permutation
│   │
│   ├── config.py                       # DataConfig(BaseModel)
│   │                                   #   image_size, augmentation toggles, loader params
│   │
│   ├── build_manifest.py               # 9 heterogeneous CSVs → unified parquet
│   │                                   #   處理 GBK 編碼、HC path prefix、bracket coordinates
│   │                                   #   輸出：data/manifest.parquet (198,557 records)
│   │
│   ├── manifest.py                     # ManifestDataset(Dataset) → SampleDict
│   │                                   #   每筆 sample 輸出一個 image + list[InstanceDict]
│   │                                   #   InstanceDict per instance:
│   │                                   #     task_id:          int
│   │                                   #     keypoints:        ndarray (K_task, 2) — per-task, 不 pad
│   │                                   #     supervised_mask:  ndarray (K_task,) — has finite coords?
│   │                                   #     visible_mask:     ndarray (K_task,) — init = supervised
│   │                                   #     bbox:             ndarray (4,) — cx,cy,w,h from kp extremes
│   │                                   #     transform_matrix: ndarray (3, 3) — identity
│   │                                   #     original_hw:      ndarray (2,) — (H, W) of loaded image
│   │                                   #     is_labeled:       bool
│   │                                   #     image_path:       str
│   │                                   #   不做 augmentation，不做 tensor 轉換
│   │                                   #   [REMOVED] landmark_valid_mask — per-task 天然全 valid
│   │
│   ├── transforms.py                   # TransformPipeline — albumentations, dict in → dict out
│   │                                   #   對 instances list 中每個 instance 的 (K_task, 2) 做變換
│   │                                   #   code 已 shape-agnostic — 不依賴固定 K
│   │                                   #   更新 transform_matrix, visible_mask, flip_applied
│   │                                   #   kornia 不在這裡 — 只用於 train/views.py 的 EC loss
│   │
│   ├── spatial.py                      # Spatial transform utilities
│   │                                   #   apply_spatial: shape-agnostic (K from runtime)
│   │                                   #   apply_flip_permutation: per-task permutation
│   │                                   #   map_to_original: dynamic K from input shape
│   │                                   #   [REMOVED] global_flip_permutation, hardcoded 60
│   │
│   ├── mosaic.py                       # MosaicWrapper(Dataset) — same-task / cross-task mosaic
│   │                                   #   各 tile 的 instances append 到 list — 不 stack、不 pad
│   │                                   #   cross-task 自然支援：A4C (16,2) + IVC (2,2) 是獨立 items
│   │                                   #   模擬 split-screen / PACS 截圖的臨床真實場景
│   │
│   ├── sampler.py                      # MixedTaskBatchSampler — batch-level task 分佈控制
│   │                                   #   balanced / proportional 策略
│   │                                   #   labeled / unlabeled 混合比例控制
│   │
│   ├── collate.py                      # collate_fn → batch dict
│   │                                   #   image:   Tensor (B, 3, H, W) — stack, 唯一的 tensor 合併
│   │                                   #   targets: list[list[InstanceDict]] — B images × instances
│   │                                   #     不做 K padding，不做 instance padding
│   │                                   #     每個 InstanceDict 保持 per-task shape (K_task, 2)
│   │                                   #   model / training loop 負責 group-by-task
│   │
│   ├── split.py                        # K-fold stratified split, patient-level isolation
│   │
│   └── loaders.py                      # build_train_loader(), build_val_loader()
│                                       #   組裝 Dataset + Sampler + collate → DataLoader
│
├── models/                             # Pure nn.Module — 不 import Lightning
│   │                                   # Unified DETR decoder, ED-Pose pattern
│   │                                   # 不依賴 FCOS / anchor / NMS — 全部 learned queries
│   │
│   ├── backbone.py                     # Backbone adapter → BackboneOutput (structured)
│   │                                   #   BackboneOutput: spatial_tokens, spatial_shape,
│   │                                   #     intermediate_features (for MIRO, None at inference)
│   │                                   #   DINOv2Backbone: torch.hub 'dinov2_vitb14'
│   │                                   #     input 518×518 (native, 37×37 = 1369 patches)
│   │                                   #     → (B, 1369, 768) patch tokens
│   │                                   #   [LATER] ConvNeXtV2Backbone
│   │                                   #   freeze() / unfreeze() / param_groups()
│   │
│   ├── projection.py                   # Backbone → Decoder 的唯一 parametric bridge
│   │                                   #   nn.Linear(C_backbone, D_model=256)
│   │                                   #   + sinusoidal 2D positional encoding
│   │                                   #   uses spatial_shape from BackboneOutput
│   │                                   #   沒有 FPN，沒有 Neck — 跟 original DETR 一樣
│   │
│   ├── decoder.py                      # Shared DETR TransformerDecoder
│   │                                   #   L=6, D=256, H=8, d_head=32, FFN=1024
│   │                                   #   forward(memory, queries) → (B, N_q, 256)
│   │                                   #   return_intermediate=True → (L, B, N_q, 256) for aux loss
│   │                                   #   同一份 weights — query 決定 output 語意
│   │                                   #   Layer: Self-Attn → Cross-Attn → FFN (DETR order)
│   │                                   #   per-layer positional injection (not just input)
│   │
│   ├── queries.py                      # Query embeddings + composition
│   │                                   #   DetectionQueries: nn.Embedding(N_det=20, 256)
│   │                                   #   LandmarkQueries: nn.ModuleDict — per-task
│   │                                   #     "A4C": Embedding(16, 256)
│   │                                   #     "PLAX": Embedding(22, 256) ... ×9 tasks
│   │                                   #     task owns its landmarks — 擴增 task 只需加 entry
│   │                                   #   build_landmark_queries(det_feat, task_id):
│   │                                   #     Q = point_emb[task_id] + det_feat
│   │                                   #     point_emb = "哪個 landmark" (what)
│   │                                   #     det_feat  = "task 在哪" (where, from Pass 1)
│   │
│   ├── heads.py                        # Output heads — small MLPs
│   │                                   #   DetectionHead: (N_det, 256) → bbox(4), task(9), conf(1)
│   │                                   #   LandmarkHead:  (K_task, 256) → (x, y) normalized [0,1]
│   │                                   #   UncertaintyHead: (K_task, 256) → σ > 0
│   │                                   #     config toggle, default OFF
│   │
│   └── model.py                        # FUBioModel → ModelOutput (structured)
│                                       #   forward(images, targets=None):
│                                       #     backbone → projection + pos_enc → memory
│                                       #     decoder(memory, Q_det)     → Pass 1: detection
│                                       #     group_by_task(targets)     → per-task batched
│                                       #     for each task:
│                                       #       idx = instances' image indices
│                                       #       decoder(memory[idx], Q_lm) → Pass 2
│                                       #   Per-task batched: 同 task 的 instances 合併成
│                                       #   一次 decoder call（≤9 calls），不是 per-instance
│
├── train/                              # Lightning layer — inference / serving 不 import 這裡
│   │
│   ├── config.py                       # ExperimentConfig (pydantic-settings)
│   │                                   #   BackboneConfig: name, pretrained, freeze_epochs
│   │                                   #   DecoderConfig: n_layers, d_model, n_heads, ffn_dim
│   │                                   #   LossConfig: λ_det, λ_land, λ_param, use_uncertainty
│   │                                   #   OptimizerConfig: lr_backbone, lr_decoder, warmup
│   │                                   #   MIROConfig: enabled, lambda, layers
│   │                                   #   從 YAML + env override 載入
│   │
│   ├── datamodule.py                   # FUBioDataModule(LightningDataModule)
│   │                                   #   thin adapter: 組裝 data/ 的 Dataset + Loader
│   │                                   #   fold setup, reproducible state
│   │
│   ├── module.py                       # FUBioModule(LightningModule)
│   │                                   #   training_step:
│   │                                   #     → _detection_loss (Hungarian → L1 + CIoU + CE)
│   │                                   #     → per-task _landmark_loss (Huber or GaussianNLL)
│   │                                   #     → per-task _param_loss (Huber on geometry)
│   │                                   #     → _miro_loss (when enabled)
│   │                                   #     → _aux_loss (intermediate decoder layers)
│   │                                   #   group_instances_by_task(targets) for per-task routing
│   │                                   #   configure_optimizers:
│   │                                   #     differential LR — backbone 1e-5, decoder/heads 1e-3
│   │                                   #   Phase control: freeze/unfreeze backbone
│   │                                   #   precision="bf16-mixed"
│   │
│   ├── losses.py                       # Loss functions
│   │                                   #   detection_loss: L1 + CIoU (bbox) + CE (task cls, 9-way)
│   │                                   #   landmark_loss: Huber (SmoothL1, β configurable)
│   │                                   #     or GaussianNLL when σ enabled
│   │                                   #   param_loss: Huber on differentiable geometry output
│   │
│   ├── matcher.py                      # Hungarian matching for detection queries
│   │                                   #   cost = λ_cls·CE + λ_box·(L1 + CIoU)
│   │                                   #   scipy.optimize.linear_sum_assignment
│   │                                   #   Ref: DETR (Carion et al., ECCV 2020)
│   │
│   ├── regularizer.py                  # MIRO regularizer
│   │                                   #   MeanEncoder, VarianceEncoder, MIROEncoders
│   │                                   #   build_miro_encoders(backbone, input_shape)
│   │                                   #   consumes BackboneOutput.intermediate_features
│   │                                   #   config-driven toggle — 不啟用時零成本
│   │
│   ├── callbacks.py                    # Lightning callbacks
│   │                                   #   CSVCallback: per-epoch val CSV ↔ checkpoint naming
│   │                                   #   SWADCallback: dense weight averaging + dead valley
│   │                                   #   CustomAveragedModel: step-tracked averaging
│   │
│   ├── schedule.py                     # LR scheduling
│   │                                   #   Linear warmup + cosine decay
│   │                                   #   Phase-aware freeze/unfreeze control
│   │
│   └── train.py                        # CLI entry point (typer)
│                                       #   load config → model → callbacks → Trainer.fit()
│                                       #   WandB logger, ModelCheckpoint
│
├── evaluation/                         # Training + inference 共用
│   │
│   ├── parameters.py                   # Clinical parameter geometry (differentiable, pixel space)
│   │                                   #   distance: ||p_i − p_j||          (25 params)
│   │                                   #   ellipse_c: Ramanujan             (HC, FA)
│   │                                   #   angle: arccos(dot/norms)         (AOP)
│   │                                   #   compute_params(landmarks, task_id) → dict
│   │                                   #   gradient flows through — used by L_param
│   │
│   ├── metrics.py                      # MRE (per-task + overall)
│   │                                   # Parameter MAE (per-param)
│   │                                   # Per-task + per-domain aggregation
│   │
│   └── postprocessing.py              # Prediction → submission format
│                                       #   coord inversion via transform_matrix
│                                       #   pixel clipping, JSON serialization
│                                       #   per-task landmark reordering
│
└── serving/                            # [LATER] Docker inference — 零 Lightning 依賴
    └── predict.py                      # checkpoint → model.eval() → forward()
                                        #   transform_matrix 反算 organizer pixel space
                                        #   輸出 JSON: landmarks (pixel coords)
```

## Data Flow — Per-Task Instance Design

```
CSV (per-task local coords, e.g. A4C: 16 points)
  │
  ▼ build_manifest.py
manifest.parquet (198,557 records)
  │
  ▼ ManifestDataset.__getitem__()
SampleDict:
  image:     ndarray (H, W, 3) uint8
  instances: [InstanceDict]              # list, length 1 for single-image
  │
  ▼ MosaicWrapper (optional)
SampleDict:
  image:     ndarray (H, W, 3)           # composited
  instances: [InstanceDict, ...]         # 1-4 instances, may be different tasks
  │                                      # A4C: (16, 2), IVC: (2, 2) — 不 pad, 不 stack
  │
  ▼ TransformPipeline
SampleDict:                              # same structure, augmented values
  image:     ndarray (tgt_H, tgt_W, 3) float32, normalized
  instances: [InstanceDict, ...]         # keypoints transformed, masks updated
  │
  ▼ collate_fubio()
batch dict:
  image:   Tensor (B, 3, H, W)          # stack — 唯一做 tensor 合併的
  targets: list[list[InstanceDict]]      # B × instances — 不 pad K, 不 pad I
```

InstanceDict (per instance, per-task shape):

| Key | Shape | Note |
|---|---|---|
| `task_id` | `int` | 0-8, from TaskDef.task_int |
| `keypoints` | `(K_task, 2)` | A4C=16, IVC=2, per-task |
| `supervised_mask` | `(K_task,)` | has finite GT coords? |
| `visible_mask` | `(K_task,)` | in bounds after aug? |
| `bbox` | `(4,)` | cx, cy, w, h from kp extremes |
| `transform_matrix` | `(3, 3)` | accumulated affine |
| `original_hw` | `(2,)` | source image (H, W) |
| `is_labeled` | `bool` | unlabeled → keypoints = NaN |
| `image_path` | `str` | relative path |
| `flip_applied` | `bool` | hflip state |

## Top-level Directories

```
src/fubio/       # the engine — data / models / train / evaluation / serving / viz
configs/         # 實驗 config YAML（pydantic-settings 載入）— 本 repo 只收提交模型的血緣鏈
docker/          # 提交用的推論容器，與評分時完全相同
scripts/         # make_submission.py / validate_checkpoint.py
tests/           # pytest + hypothesis（等變性 property tests）
data/            # ~18G 影像資料 — organizer 發布，CC BY-NC，不隨本 repo 散布
```

原始開發環境另有 `notebooks/`（marimo 探索）、`viewer/`（streamlit 檢視器）與
organizer baseline code；這些不屬於方法本身，未納入公開釋出。

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Data representation | Per-task InstanceDict, no global K | Task owns landmarks。list-of-dicts 不需要 pad K。擴增 task = 加 TaskDef |
| Architecture | Unified DETR decoder (ED-Pose pattern) | 一個 decoder 兩種 query — detection + landmark。消除 FCOS/anchor/NMS。Ref: Yang et al. ICLR 2023 |
| Backbone bridge | Linear projection only (no FPN/Neck) | 跟 original DETR 一樣。DINOv2 self-attention 已有全局 context |
| Landmark queries | Per-task nn.ModuleDict | Task 獨立 embedding，擴增 = 加 entry |
| Det→Lm conditioning | Q_lm = point_emb + det_feat | det_feat 攜帶 task identity + spatial position |
| Backbone | DINOv2-B/14 first, 518×518 native | 37×37 patches, 142M SSL。ConvNeXt V2-B 之後加 |
| Training framework | Lightning Trainer + LightningModule | SWAD / CSV-logging callbacks port cleanly onto Lightning |
| MIRO | Config toggle, off by default | 保護 pretrained features；Round 2 A/B vs LP-FT vs diff-LR |
| Augmentation | albumentations (data pipeline) | KeypointParams edge-case 成熟；kornia 僅用於 EC loss (Round 3) |
| Precision | bf16-mixed | A100 同 throughput，不需 loss scaling |
| Clinical params | pixel space, differentiable | organizer server-side px→mm；gradient flows for L_param |
| Config | pydantic-settings (YAML + env) | 禁 dataclass；取代 OmegaConf |
| Landmark loss | Huber / GaussianNLL | β configurable；σ 啟用時切 GaussianNLL |
| Detection loss | L1 + CIoU + CE + Hungarian | CE for 9-way mutually exclusive tasks (not BCE) |
| Collate | Stack images only, targets as list | 不 pad K、不 pad I — model 負責 group-by-task |
| Per-task batching | group_by_task → batched decoder call per task | ≤9 calls not N_instances — 同 task instances 合併 |
| Uncertainty | UncertaintyHead toggleable, default OFF | Conformal calibration 可開發但初始關閉 |

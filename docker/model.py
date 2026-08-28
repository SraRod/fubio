"""Docker submission wrapper — lean inference module.

Organizer's predict.py calls Model().predict(data_root, output_dir, batch_size).
This file is self-contained: InferenceModule (thin nn.Module wrapper around
FUBioModel) + Model class that wires checkpoint → inference → output.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("model")

APP_DIR = Path("/app")
SHAPE_PRIOR_PATH = APP_DIR / "data" / "shape_prior_v4.json"


class InferenceModule(nn.Module):
    """Lean wrapper: FUBioModel + ImageNet normalization buffers.

    Provides the exact interface that run_inference_tta expects:
      .model(tensor) → ModelOutput
      ._normalize_image(tensor) → tensor
      ._img_mean, ._img_std (registered buffers, used by TTA warp code)
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "_img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """uint8 (B,3,H,W) → float32 ImageNet-normalized."""
        return image.float().div_(255.0).sub_(self._img_mean).div_(self._img_std)


def _load_shape_prior(config):
    """Load the shape prior following FUBioModule.__init__ logic: GeoSimCC's
    SHAPE stage and the query anchors both need it."""
    from fubio.data.shape_prior import ShapePrior

    prior_path = config.loss.shape_prior_path
    if not prior_path.exists():
        raise FileNotFoundError(f"Shape prior not found: {prior_path}")
    return ShapePrior.model_validate_json(prior_path.read_text())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Model:
    """Competition interface. Organizer calls Model() then model.predict(...)."""

    def __init__(self) -> None:
        t0 = time.time()

        # --- Environment ---
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — this model requires a GPU")
        self._device = torch.device("cuda")

        logger.info("torch=%s  CUDA=%s  GPU=%s",
                     torch.__version__, torch.version.cuda,
                     torch.cuda.get_device_name(0))

        # --- Checkpoint: single deserialization ---
        ckpt_path = Path("/work/best_model.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Loading checkpoint: %s (sha256=%s)",
                     ckpt_path, _file_sha256(ckpt_path))

        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        if "hyper_parameters" not in raw:
            raise ValueError("Checkpoint missing 'hyper_parameters'")
        if "teacher_state_dict" not in raw or not raw["teacher_state_dict"]:
            raise ValueError("Checkpoint missing or empty 'teacher_state_dict'")

        hp = dict(raw["hyper_parameters"])
        teacher_sd = raw.pop("teacher_state_dict")
        n_teacher_keys = len(teacher_sd)
        del raw
        gc.collect()
        logger.info("Teacher state_dict: %d keys", n_teacher_keys)

        # --- Patch hparams ---
        loss = dict(hp.get("loss") or {})
        loss["shape_prior_path"] = str(SHAPE_PRIOR_PATH)
        hp["loss"] = loss

        coord_mode = hp.get("head", {}).get("coord", {}).get("mode", "")

        # --- Config ---
        from fubio.train.config import ExperimentConfig

        self._config = ExperimentConfig(**hp)
        logger.info("Backbone=%s  input_size=%s  coord=%s",
                     self._config.backbone.name,
                     self._config.backbone.input_size,
                     coord_mode)

        # --- Shape prior ---
        if not SHAPE_PRIOR_PATH.exists():
            raise FileNotFoundError(f"Shape prior not found: {SHAPE_PRIOR_PATH}")
        shape_prior = _load_shape_prior(self._config)
        logger.info("Shape prior loaded: %s (sha256=%s)",
                     SHAPE_PRIOR_PATH, _file_sha256(SHAPE_PRIOR_PATH))

        # --- Build FUBioModel directly (no LightningModule overhead) ---
        from fubio.models.model import FUBioModel

        model = FUBioModel(
            backbone_name=self._config.backbone.name,
            d_model=self._config.d_model,
            n_heads=self._config.decoder.n_heads,
            ffn_dim=self._config.decoder.ffn_dim,
            head_ffn_dim=self._config.head.ffn_dim,
            n_decoder_layers=self._config.decoder.n_layers,
            n_head_layers=self._config.head.n_layers,
            n_inst=self._config.head.n_inst,
            derive_bbox=self._config.head.derive_bbox,
            coord_config=self._config.head.coord,
            shape_prior=shape_prior,
            neck_config=self._config.neck,
            neck_dropout=self._config.neck_dropout,
            dropout=self._config.decoder.dropout,
        )

        # --- Load teacher weights (strict) ---
        result = model.load_state_dict(teacher_sd, strict=True)
        if result.missing_keys:
            raise RuntimeError(f"Missing keys in teacher state_dict: {result.missing_keys}")
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in teacher state_dict: {result.unexpected_keys}")
        del teacher_sd
        gc.collect()
        logger.info("Teacher weights loaded (strict=True, 0 missing, 0 unexpected)")

        # --- Wrap + freeze ---
        self._module = InferenceModule(model)
        self._module.to(self._device)
        self._module.eval()
        for p in self._module.parameters():
            p.requires_grad = False

        elapsed = time.time() - t0
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info("Model ready in %.1fs  VRAM=%.0fMB", elapsed, vram_mb)

    def predict(self, data_root: str, output_dir: str, batch_size: int = 8) -> None:
        t0 = time.time()
        data_root_p = Path(data_root)
        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)
        input_size = self._config.backbone.input_size

        from fubio.serving.predict import (
            discover_competition_images,
            run_inference_tta,
            save_outputs,
        )
        from fubio.serving.validate import (
            _check_normalized_range,
            _check_vector_lengths,
        )

        # --- Discover images ---
        samples = discover_competition_images(data_root_p, subdir="images")
        logger.info("Discovered %d images across %d tasks",
                     len(samples), len({s["task_id"] for s in samples}))

        if len(samples) == 0:
            raise RuntimeError(
                f"No images found under {data_root_p / 'images'}. "
                f"Check that the organizer's predict.py created the symlinks."
            )

        # --- Inference (batch_size=1 for determinism + min VRAM) ---
        torch.cuda.reset_peak_memory_stats()

        results = run_inference_tta(
            self._module, samples, data_root_p, input_size,
            self._device, batch_size=1,
            task_routing="high_conf",
            task_routing_threshold=0.9,
            tta_angles=[-5.0, 0.0, 5.0],
            ellipse_consensus=True,
        )

        # --- Validate ---
        if len(results) != len(samples):
            raise RuntimeError(
                f"Prediction count mismatch: got {len(results)}, expected {len(samples)}"
            )

        n_rerouted = 0
        for r in results:
            label = f"{r['task_id']}/{r.get('submission_path', r.get('image_path'))}"
            _check_vector_lengths(r, label)
            _check_normalized_range(r, label)
            img_task = r.get("submission_path", r["image_path"]).split("/")[0]
            if img_task != r["task_id"]:
                n_rerouted += 1

        tasks_present = {r["task_id"] for r in results}
        logger.info("Validation passed: %d predictions, %d tasks, %d rerouted",
                     len(results), len(tasks_present), n_rerouted)

        # --- Write output ---
        metadata = {
            "mode": "docker_submission",
            "model_source": "teacher",
            "batch_size": 1,
            "task_routing": "high_conf",
            "tta_angles": [-5.0, 0.0, 5.0],
            "ellipse_consensus": True,
        }
        save_outputs(results, output_dir_p, metadata, competition_ordering=True)

        # --- Final stats ---
        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(
            "Inference complete: %.1fs elapsed, peak VRAM=%.0fMB, "
            "%d predictions written to %s",
            elapsed, peak_vram, len(results), output_dir_p,
        )

        # Verify output files exist
        for name in ("regression_predictions.json", "landmark_predictions.json"):
            p = output_dir_p / name
            if not p.exists():
                raise RuntimeError(f"Output file missing: {p}")
            data = json.loads(p.read_text())
            logger.info("  %s: %d entries", name, len(data))

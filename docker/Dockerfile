FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System deps for OpenCV
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
       libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Inference-only Python deps (torch/torchvision/numpy already in base image)
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt
RUN python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

# DINOv2 vendor (pre-cloned, no network needed at runtime)
COPY vendor_dinov2/ /app/vendor/dinov2/

# Offline environment
ENV FUBIO_DINOV2_LOCAL=/app/vendor/dinov2
ENV TORCH_HOME=/app/.torch_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV WANDB_MODE=disabled

WORKDIR /app

# Application code — organizer entry + our wrapper
COPY predict.py model.py /app/

# fubio package — inference-only modules
COPY fubio/__init__.py /app/fubio/__init__.py

COPY fubio/data/__init__.py          /app/fubio/data/__init__.py
COPY fubio/data/landmark_ordering.py /app/fubio/data/landmark_ordering.py
COPY fubio/data/manifest.py          /app/fubio/data/manifest.py
COPY fubio/data/mosaic.py            /app/fubio/data/mosaic.py
COPY fubio/data/ordering_schema.py   /app/fubio/data/ordering_schema.py
COPY fubio/data/shape_prior.py       /app/fubio/data/shape_prior.py
COPY fubio/data/spatial.py           /app/fubio/data/spatial.py
COPY fubio/data/supportive.py        /app/fubio/data/supportive.py
COPY fubio/data/task_registry.py     /app/fubio/data/task_registry.py
COPY fubio/data/transforms.py        /app/fubio/data/transforms.py
COPY fubio/data/types.py             /app/fubio/data/types.py

COPY fubio/evaluation/__init__.py        /app/fubio/evaluation/__init__.py
COPY fubio/evaluation/postprocessing.py  /app/fubio/evaluation/postprocessing.py

COPY fubio/models/__init__.py         /app/fubio/models/__init__.py
COPY fubio/models/backbone.py         /app/fubio/models/backbone.py
COPY fubio/models/coord_predictors.py  /app/fubio/models/coord_predictors.py
COPY fubio/models/decoder.py          /app/fubio/models/decoder.py
COPY fubio/models/heads.py            /app/fubio/models/heads.py
COPY fubio/models/model.py            /app/fubio/models/model.py
COPY fubio/models/neck.py             /app/fubio/models/neck.py
COPY fubio/models/queries.py          /app/fubio/models/queries.py

COPY fubio/serving/__init__.py   /app/fubio/serving/__init__.py
COPY fubio/serving/predict.py    /app/fubio/serving/predict.py
COPY fubio/serving/validate.py   /app/fubio/serving/validate.py

COPY fubio/train/__init__.py  /app/fubio/train/__init__.py
COPY fubio/train/config.py    /app/fubio/train/config.py
COPY fubio/train/views.py     /app/fubio/train/views.py

# Data files
COPY data/shape_prior_v4.json /app/data/shape_prior_v4.json

# Checkpoint (stripped: teacher weights + hparams only, ~583MB)
COPY best_model.pth /app/best_model.pth

# Build-time smoke test
RUN python3 -c "\
import os; \
os.environ.setdefault('FUBIO_DINOV2_LOCAL', '/app/vendor/dinov2'); \
import torch; \
from fubio.models.model import FUBioModel; \
from fubio.models.backbone import DINOv2Backbone; \
b = DINOv2Backbone('dinov2_vitb14'); \
print(f'DINOv2 OK: embed_dim={b.embed_dim}, patch_size={b.patch_size}'); \
print(f'torch={torch.__version__} cuda={torch.version.cuda}'); \
print('Smoke test PASSED'); \
"

ENTRYPOINT ["python3", "/app/predict.py"]

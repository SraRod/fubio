"""Shared fixtures: a stub backbone and real-constructor model/module builders.

The model tests exercise the REAL FUBioModel / FUBioModule constructors — the
same code path training and serving run — with only the DINOv2 download
replaced by a stub. The shape prior is the repo's committed
data/shape_prior_v4.json, resolved relative to the repo root so pytest can be
invoked from anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from fubio.data.shape_prior import TaskShapePrior
from fubio.models.backbone import BackboneOutput

REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_PRIOR_V4 = REPO_ROOT / "data" / "shape_prior_v4.json"

D = 256
N_SPATIAL = 1369  # 37 * 37
C_BACKBONE = 768


class StubBackbone(nn.Module):
    """Mimics DINOv2Backbone's output contract without loading real weights."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = C_BACKBONE
        self.patch_size = 14
        self._linear = nn.Linear(3, C_BACKBONE)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def param_groups(self, lr: float, layer_decay: float = 1.0) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr, "name": "backbone"}]

    def forward(self, x: Tensor) -> BackboneOutput:
        b = x.shape[0]
        tokens = self._linear(x[:, :, :3, :3].permute(0, 2, 3, 1).reshape(b, -1, 3))
        tokens = torch.nn.functional.interpolate(
            tokens.permute(0, 2, 1),
            size=N_SPATIAL,
            mode="linear",
        ).permute(0, 2, 1)
        return BackboneOutput(features=[tokens], spatial_shape=(37, 37))


@pytest.fixture
def stub_backbone(monkeypatch: pytest.MonkeyPatch) -> type[StubBackbone]:
    """Patch build_backbone so FUBioModel.__init__ runs without torch.hub."""
    import fubio.models.model as _model_mod

    monkeypatch.setattr(
        _model_mod, "build_backbone", lambda **kw: StubBackbone()
    )
    return StubBackbone


def make_test_config(**overrides: object):
    """ExperimentConfig wired for stub tests: 1-level neck, committed v4 prior."""
    from fubio.train.config import ExperimentConfig

    base: dict = {
        "d_model": D,
        # StubBackbone returns one feature level, so the C2f neck must expect one.
        "neck": {"mode": "c2f", "layer_indices": [11], "n_bottleneck": 1},
        "loss": {"shape_prior_path": str(SHAPE_PRIOR_V4)},
        "head": {"n_inst": 1, "derive_bbox": True, "coord": {"mode": "geo_simcc"}},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return ExperimentConfig(**base)


def make_module(config=None, **overrides: object):
    """Real FUBioModule via the real constructor (stub_backbone must be active)."""
    from fubio.train.module import FUBioModule

    return FUBioModule(config if config is not None else make_test_config(**overrides))


def make_task_prior(K: int, M: int = 2, seed: int = 0) -> TaskShapePrior:
    """Synthetic-but-valid TaskShapePrior for arbitrary K (unit tests only)."""
    rng = np.random.default_rng(seed)
    M = min(M, 2 * K)
    basis = rng.normal(size=(M, 2 * K))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    mean_xy = rng.uniform(0.3, 0.7, size=(K, 2))
    canonical = mean_xy - mean_xy.mean(axis=0)
    canonical /= max(np.linalg.norm(canonical), 1e-6)
    return TaskShapePrior(
        K=K,
        M=M,
        n_samples=32,
        variance_explained=0.9,
        mean_logit=np.log(mean_xy / (1 - mean_xy)).tolist(),
        basis=basis.tolist(),
        eigenvalues=[1.0] * M,
        canonical_mean=canonical.tolist(),
        canonical_basis=basis.tolist(),
        mean_xy=mean_xy.tolist(),
        std_xy=(0.05 * np.ones((K, 2))).tolist(),
    )


def make_task_module(K: int, n_inst: int = 1, n_head_layers: int = 1, seed: int = 0):
    """TaskModule with a standalone GeoSimCC predictor (own keys, no fine map)."""
    from fubio.models.coord_predictors import GeoSimCCPredictor
    from fubio.models.heads import TaskModule

    prior = make_task_prior(K, seed=seed)
    return TaskModule(
        n_keypoints=K,
        d_model=D,
        n_inst=n_inst,
        n_head_layers=n_head_layers,
        task_shape_prior=prior,
        use_anchors=True,
        derive_bbox=True,
        coord_predictor=GeoSimCCPredictor(d_model=D, n_keypoints=K, prior=prior),
    )

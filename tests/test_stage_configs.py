"""The stage configs are the lineage of the released weights — pin that claim.

Every stage YAML must parse under the current schema, and stage3 (the config
behind the submitted run) must agree with the hyper_parameters stored inside
the released best_model.pth on every field that shapes the model or the
objective. The weights file is a Releases asset, so that half of the suite is
skipped when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fubio.train.config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "docker" / "best_model.pth"

STAGE_CONFIGS = ["stage1", "stage1-head-tune", "stage2", "stage3"]


@pytest.mark.parametrize("name", STAGE_CONFIGS)
def test_stage_config_parses(name: str) -> None:
    cfg = ExperimentConfig(**yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text()))
    assert cfg.head.coord.mode == "geo_simcc"
    assert cfg.neck is not None and cfg.neck.mode == "c2f"
    assert cfg.loss.lambda_heatmap > 0


def test_head_tune_stage_names_weak_tasks() -> None:
    cfg = ExperimentConfig(
        **yaml.safe_load((REPO_ROOT / "configs" / "stage1-head-tune.yaml").read_text())
    )
    assert cfg.head_tune.enabled
    assert set(cfg.head_tune.reinit_tasks) <= set(cfg.head_tune.tune_tasks)


def test_semi_stages_enable_the_teacher() -> None:
    for name in ("stage2", "stage3"):
        cfg = ExperimentConfig(
            **yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text())
        )
        assert cfg.semi.enabled and cfg.semi.lambda_pseudo > 0, name


requires_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="docker/best_model.pth not present (Releases asset)"
)


@requires_weights
def test_released_hparams_reload_under_current_schema() -> None:
    import torch

    raw = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig(**raw["hyper_parameters"])
    assert cfg.head.coord.mode == "geo_simcc"


@requires_weights
def test_stage3_matches_released_hyper_parameters() -> None:
    """stage3.yaml is the submitted run's config: the sections that shape the
    model or the objective must match the released weights' hyper_parameters
    exactly. (Schedule-free bookkeeping like logging is not compared.)"""
    import torch

    raw = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    released = ExperimentConfig(**raw["hyper_parameters"]).model_dump()
    ours = ExperimentConfig(
        **yaml.safe_load((REPO_ROOT / "configs" / "stage3.yaml").read_text())
    ).model_dump()

    for section in ("d_model", "backbone", "neck", "decoder", "head", "loss",
                    "matcher", "semi", "optimizer", "data", "precision"):
        assert ours[section] == released[section], f"{section} differs from the released run"

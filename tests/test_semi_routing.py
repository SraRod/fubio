"""Semi-supervised loss routing: which loss term sees which data tier.

The tier table in train/module.py is a contract, and every semi-supervised
result depends on it holding. A routing error does not crash and does not show
up in any logged loss — it presents only as "the experiment did not help",
which is indistinguishable from the hypothesis being wrong.

Covers the confidence path, the only tier-dependent one active before
equivariance lands: L_mil fires on task-known unlabeled data, the focal loss
does NOT fire on those same rows, cross-task absence still does, and the
labeled pass is unchanged.

Upstream: train/module.py (_step), train/losses.py (mil_loss).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from fubio.data.task_registry import TASKS
from fubio.train.config import ExperimentConfig
from fubio.train.losses import mil_loss
from fubio.train.module import FUBioModule

CONFIG_PATH = "configs/r22-semi-mil.yaml"


def _instance(task_str: str, *, labeled: bool) -> dict:
    """Minimal InstanceDict for one task in one image."""
    k = TASKS[task_str].n_keypoints
    return {
        "task_id": TASKS[task_str].task_int,
        "keypoints": np.full((k, 2), 0.5, dtype=np.float32),
        "supervised_mask": np.full(k, labeled, dtype=bool),
        "visible_mask": np.full(k, labeled, dtype=bool),
        "bbox": np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32),
        "transform_matrix": np.eye(3, dtype=np.float32),
        "original_hw": np.array([800, 600], dtype=np.int32),
        "is_labeled": labeled,
        "image_path": f"{task_str}.png",
    }


@pytest.fixture(scope="module")
def module() -> FUBioModule:
    with open(CONFIG_PATH) as f:
        config = ExperimentConfig(**yaml.safe_load(f))
    m = FUBioModule(config)
    m.eval()
    return m


@pytest.fixture
def run(module: FUBioModule, monkeypatch: pytest.MonkeyPatch):
    """Run _step and return (total, logged) — self.log is inert without a Trainer."""
    logged: dict[str, float] = {}

    def _capture(name: str, value, **_kwargs) -> None:
        logged[name] = float(value)

    monkeypatch.setattr(module, "log", _capture)
    size_w, size_h = module.config.backbone.input_size

    def _run(specs: list[tuple[str, bool]], stage: str) -> tuple[float, dict[str, float]]:
        logged.clear()
        torch.manual_seed(0)
        batch = {
            "image": torch.randint(0, 255, (len(specs), 3, size_h, size_w), dtype=torch.uint8),
            "targets": [[_instance(t, labeled=lab)] for t, lab in specs],
        }
        with torch.no_grad():
            total = module._step(batch, stage)
        return total.item(), dict(logged)

    return _run


def test_mil_loss_reduces_max_over_slots_mean_over_images() -> None:
    """Bag operator is max over slots; images are averaged, not summed."""
    logits = torch.tensor([[0.0, 5.0, -5.0], [0.0, -1.0, -2.0]])
    expected = -(F.logsigmoid(torch.tensor(5.0)) + F.logsigmoid(torch.tensor(0.0))) / 2
    assert torch.allclose(mil_loss(logits), expected)
    assert mil_loss(torch.zeros(0, 4)).item() == 0.0


def test_mil_fires_only_on_the_unlabeled_tier(run) -> None:
    """L_mil is zero for annotated data and positive for task-known unlabeled."""
    _, labeled = run([("HC", True), ("A4C", True)], "train")
    _, unlabeled = run([("HC", False), ("A4C", False)], "train_unlabeled")

    assert labeled["train/loss_mil"] == 0.0, "L_mil must not fire on annotated data"
    assert unlabeled["train_unlabeled/loss_mil"] > 0.0


def test_conf_loss_skips_the_own_task_of_an_unlabeled_image(run) -> None:
    """The focal loss must not drive an unlabeled image's own task toward zero.

    total_conf sums one term per task, so dropping HC's term makes the total
    strictly smaller than the same images with no tier information at all —
    where HC is treated as a plain negative and does contribute.
    """
    _, masked = run([("HC", False), ("HC", False)], "train_unlabeled")
    _, unmasked = run([("HC", True), ("HC", True)], "train")

    assert masked["train_unlabeled/loss_conf"] < unmasked["train/loss_conf"], (
        "HC's confidence term should be absent when both images are task-known "
        "unlabeled HC; it was not, so those rows still reach the focal loss"
    )


def test_cross_task_absence_is_still_supervised(run) -> None:
    """L_absent survives the mask: the other eight tasks are genuine negatives."""
    _, logged = run([("HC", False)], "train_unlabeled")
    assert logged["train_unlabeled/loss_conf"] > 0.0


def test_unlabeled_pass_contributes_confidence_terms_only(run) -> None:
    """No GT-free loss leaks into the unlabeled pass.

    lambda_ortho is 0.1 in this config and needs no GT, so without the tier gate
    it would start acting on unlabeled images — making the run differ from its
    anchor by more than the semi-supervised terms.
    """
    _, logged = run([("HC", False), ("A4C", False)], "train_unlabeled")

    assert logged.get("train_unlabeled/loss_ortho", 0.0) == 0.0
    for key in ("loss_bbox", "loss_lm", "loss_param"):
        assert logged[f"train_unlabeled/{key}"] == 0.0, f"{key} needs GT and must stay zero"


def test_labeled_pass_is_unchanged_by_a_preceding_unlabeled_pass(run) -> None:
    """Separate batches through separate _step calls is what buys this."""
    alone, _ = run([("HC", True), ("AOP", True)], "train")
    run([("A4C", False)], "train_unlabeled")
    after, _ = run([("HC", True), ("AOP", True)], "train")

    assert alone == pytest.approx(after, rel=1e-6)

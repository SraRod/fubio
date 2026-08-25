"""Smoke test: MIRO integration in FUBioModule.

Verifies that MIRO loss is computed and its raw magnitude relative to
other loss terms, for calibrating lambda_miro.
"""

from __future__ import annotations

import torch

from fubio.train.config import ExperimentConfig, MIROConfig
from fubio.train.module import FUBioModule


def _make_batch(batch_size: int, img_size: int, device: torch.device) -> dict:
    """Minimal batch with one labeled instance per image (task=A4C, K=16)."""
    images = torch.randint(
        0, 255, (batch_size, 3, img_size, img_size), device=device, dtype=torch.uint8
    )
    targets: list[list[dict]] = []
    for _ in range(batch_size):
        targets.append(
            [
                {
                    "task_id": 0,
                    "is_labeled": True,
                    "bbox": [0.3, 0.3, 0.7, 0.7],
                    "keypoints": [[0.5, 0.5]] * 16,
                    "supervised_mask": [True] * 16,
                    "visible_mask": [True] * 16,
                    "original_hw": [img_size, img_size],
                }
            ]
        )
    return {"image": images, "targets": targets}


def test_miro_loss_magnitude() -> None:
    """Check MIRO wiring works and print loss magnitudes for lambda calibration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(
        backbone={"name": "dinov2_vitb14", "input_size": 224, "freeze_epochs": 0},
        head={"n_inst": 2},
        miro=MIROConfig(lambda_miro=1.0, init_variance=0.1),
        optimizer={"max_epochs": 1},
        data={"transform": {"target_size": (224, 224)}},
    )

    module = FUBioModule(config).to(device)
    module.eval()

    batch = _make_batch(2, 224, device)

    with torch.no_grad():
        loss = module._step(batch, "train")

    print(f"\n{'=' * 50}")
    print("MIRO Integration Smoke Test — Loss Magnitudes")
    print(f"{'=' * 50}")
    print(f"total loss:    {loss.item():.4f}")

    # Extract logged metrics from the module's internal state
    # (Lightning logs aren't accessible directly, but the loss value confirms wiring)
    assert loss.isfinite(), "Loss is not finite!"

    # Now run a separate MIRO-only forward to see raw magnitude
    from fubio.train.regularizer import miro_loss

    images = module._normalize_image(batch["image"])
    model_out = module.model(images)
    post_feats = model_out.backbone_out.features
    assert len(post_feats) > 1, "features should have multiple levels with MIRO"

    with torch.no_grad():
        pre_out = module._frozen_backbone(images)
    pre_feats = pre_out.features

    raw_miro = miro_loss(pre_feats, post_feats, module._miro_encoders)
    print(f"raw MIRO loss: {raw_miro.item():.4f}")
    print(f"\nFeature shapes: {[f.shape for f in post_feats]}")

    # At init, frozen == training backbone, so MIRO loss should be small
    # (only non-zero because of the variance encoder's learned bias)
    print(f"MIRO at init (expect small): {raw_miro.item():.6f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    test_miro_loss_magnitude()

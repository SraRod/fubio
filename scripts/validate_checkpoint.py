"""Re-run the validation loop on an existing checkpoint and print its metrics.

Exists as a regression guard: after changing anything in the evaluation stack,
a known checkpoint must still produce the value it produced before. R20 epoch 087
(`j3xi8ip6`) logged val_mre_overall = 27.515241622924805, and that number is the
reference every later comparison is anchored to.

Usage:
    uv run python scripts/validate_checkpoint.py --ckpt <path> [--device cpu]

Upstream: train/module.py, train/datamodule.py.
Downstream: none — prints to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging

import lightning as L
import torch

from fubio.train.config import ExperimentConfig
from fubio.train.datamodule import FUBioDataModule
from fubio.train.module import FUBioModule

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-validate a checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--expect", type=float, default=None, help="Reference val_mre_overall")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**raw["hyper_parameters"])
    module = FUBioModule.load_from_checkpoint(
        args.ckpt, config=config, map_location="cpu", weights_only=False
    )

    dm = FUBioDataModule(data_config=config.data, seed=config.seed)
    # FUBioDataModule.setup only builds datasets for stage in ("fit", None);
    # trainer.validate passes stage="validate", so val_dataloader would assert.
    # Prepared here rather than by widening the datamodule, so this regression
    # guard cannot perturb the training data path it is meant to check.
    dm.setup("fit")

    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        precision="32-true" if args.device == "cpu" else config.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.validate(module, datamodule=dm)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    key = "val_mre_overall"
    print(json.dumps({k: v for k, v in sorted(metrics.items()) if "mre" in k}, indent=2))

    if args.expect is not None:
        got = metrics.get(key)
        if got is None:
            raise SystemExit(f"FAIL: {key} not produced")
        delta = abs(got - args.expect)
        print(f"\n{key}: got {got!r}  expected {args.expect!r}  |delta| = {delta:.3e}")
        # bf16 vs fp32 and GPU reduction order both perturb the low bits, so this
        # is a tolerance check, not equality against the rounded 27.515.
        if delta > args.tol:
            raise SystemExit(f"FAIL: differs by {delta:.3e} > tol {args.tol}")
        print("PASS: evaluation stack reproduces the reference value")


if __name__ == "__main__":
    main()

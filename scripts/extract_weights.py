"""Extract deployable weights from a Lightning training checkpoint.

A training checkpoint carries the student (under ``state_dict``, keys prefixed
``model.``), the EMA teacher (under ``teacher_state_dict``, bare FUBioModel
keys), and optimizer/scheduler state — 2.2 GB in all. The inference container
needs none of the training state: ``docker/model.py`` requires exactly two keys,
``hyper_parameters`` and ``teacher_state_dict`` (bare FUBioModel keys, loaded
with ``strict=True``). This script produces that file (~580 MB).

``--source student`` strips the ``model.`` prefix and stores the student
weights under the same ``teacher_state_dict`` key — the key names the
container's contract, not the weights' provenance. The submitted model used
the teacher (paper Section 3.5).

Usage:
    uv run python scripts/extract_weights.py --ckpt <path> --output docker/best_model.pth
    uv run python scripts/extract_weights.py --ckpt <path> --source student --output student.pth

Upstream: train/module.py (on_save_checkpoint writes teacher_state_dict).
Downstream: docker/model.py (load contract), docker/build_and_test.sh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def extract(ckpt_path: Path, source: str) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in raw:
        raise SystemExit(f"{ckpt_path} has no hyper_parameters — not a FUBioModule checkpoint?")

    if source == "teacher":
        sd = raw.get("teacher_state_dict")
        if not sd:
            raise SystemExit(
                "Checkpoint has no teacher_state_dict — the EMA teacher only exists "
                "when training ran with semi.enabled and semi.lambda_pseudo > 0. "
                "Use --source student for a supervised-phase checkpoint."
            )
    elif source == "student":
        prefix = "model."
        sd = {
            k.removeprefix(prefix): v
            for k, v in raw["state_dict"].items()
            if k.startswith(prefix)
        }
        if not sd:
            raise SystemExit("state_dict has no 'model.'-prefixed keys — unexpected layout.")
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(source)

    return {"hyper_parameters": raw["hyper_parameters"], "teacher_state_dict": sd}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract deployable weights from a .ckpt")
    ap.add_argument("--ckpt", type=Path, required=True, help="Lightning checkpoint")
    ap.add_argument("--output", type=Path, required=True, help="Output .pth path")
    ap.add_argument(
        "--source",
        choices=["teacher", "student"],
        default="teacher",
        help="Which weights to keep (default: teacher, as submitted)",
    )
    args = ap.parse_args()

    out = extract(args.ckpt, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    n_params = sum(v.numel() for v in out["teacher_state_dict"].values())
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"{args.source}: {len(out['teacher_state_dict'])} tensors, "
        f"{n_params / 1e6:.2f}M params -> {args.output} ({size_mb:.0f} MB)"
    )


if __name__ == "__main__":
    main()

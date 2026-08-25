"""Build PCA shape prior from labeled training landmarks.

Reads manifest.parquet (train_local split only), computes per-task PCA
in logit space, writes shape_prior.json.

Pipeline position: build_manifest → split → **build_shape_prior** → build_cache → train.

Usage:
    uv run python -m fubio.data.build_shape_prior
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from fubio.data.shape_prior import compute_shape_prior

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Annotated[Path, typer.Option(help="Path to manifest.parquet")] = Path(
        "data/manifest.parquet"
    ),
    variance_threshold: Annotated[
        float, typer.Option(help="Cumulative variance threshold for M selection")
    ] = 0.85,
    m_cap: Annotated[int, typer.Option(help="Hard cap on components per task")] = 10,
    output: Annotated[Path, typer.Option(help="Output JSON path")] = Path("data/shape_prior.json"),
    no_supportive: Annotated[
        bool, typer.Option("--no-supportive", help="K_scored only (no supportive landmarks)")
    ] = False,
) -> None:
    """Compute PCA shape prior for all tasks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}\n"
            f"Build with: uv run python -m fubio.data.build_manifest"
        )

    include_sup = not no_supportive
    logger.info("Computing shape prior from %s", manifest)
    logger.info(
        "  variance_threshold=%.2f  m_cap=%d  supportive=%s",
        variance_threshold, m_cap, include_sup,
    )

    prior = compute_shape_prior(
        manifest_path=manifest,
        variance_threshold=variance_threshold,
        m_cap=m_cap,
        include_supportive=include_sup,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prior.model_dump_json(indent=2))

    logger.info("Shape prior written to %s (%d tasks)", output, len(prior.tasks))


if __name__ == "__main__":
    app()

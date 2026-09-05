"""P1 Prepare CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("compute-sample-descriptives")
def cli_compute_sample_descriptives(
    universe_csv: Path = typer.Option(
        Path("data/universe.csv"), help="Universe metadata (Symbol, Sector, Industry)."
    ),
    embeddings_dir: Path = typer.Option(
        Path("data/shared/embeddings/qwen3-embedding-8b"),
        help="Directory of per-ticker embedding parquets.",
    ),
    returns_dir: Path = typer.Option(
        Path("data/shared/returns"), help="Directory of per-ticker return parquets."
    ),
    distance_artifact_dir: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        help="Validated rooted W2 artifact directory.",
    ),
    output_file: Path = typer.Option(
        Path("data/papers/paper1/sample_descriptives/summary.yaml"),
        help="Output YAML summary path.",
    ),
    provider_id: str = typer.Option(
        "qwen3-embedding-8b",
        help="Encoder identity asserted against the distance artifact.",
    ),
    representation_id: str = typer.Option(
        "qwen3-embedding-8b-unit",
        help="Representation identity asserted against the distance artifact.",
    ),
    distance_id: str = typer.Option(
        "wasserstein_w2",
        help="Distance identity asserted against the distance artifact.",
    ),
) -> None:
    """Compute per-sector and overall sample descriptives.

    Include firm and article counts, return volatility, within/cross-sector
    rooted W2 distance, and the full ticker roster.
    """
    own_float64()
    from pipeline.stages.papers.paper1.sample_descriptives import (
        run_sample_descriptives,
    )

    run_sample_descriptives(
        universe_csv=universe_csv,
        embeddings_dir=embeddings_dir,
        returns_dir=returns_dir,
        distance_artifact_dir=distance_artifact_dir,
        output_file=output_file,
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id=distance_id,
    )

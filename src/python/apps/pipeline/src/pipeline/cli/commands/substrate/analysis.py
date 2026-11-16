"""Substrate Analysis CLI command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("compute-sample-descriptives")
def cli_compute_sample_descriptives(
    universe_csv: Annotated[
        Path, typer.Option(help="Universe metadata (Symbol, Sector, Industry).")
    ],
    embeddings_dir: Annotated[
        Path, typer.Option(help="Directory of per-ticker embedding parquets.")
    ],
    returns_dir: Annotated[
        Path, typer.Option(help="Directory of per-ticker return parquets.")
    ],
    distance_artifact_dir: Annotated[
        Path, typer.Option(help="Validated rooted W2 artifact directory.")
    ],
    output_file: Annotated[Path, typer.Option(help="Results JSON (results.json).")],
    provider_id: Annotated[
        str,
        typer.Option(help="Encoder identity asserted against the distance artifact."),
    ],
    representation_id: Annotated[
        str,
        typer.Option(
            help="Representation identity asserted against the distance artifact."
        ),
    ],
    distance_id: Annotated[
        str,
        typer.Option(help="Distance identity asserted against the distance artifact."),
    ],
) -> None:
    """Compute per-sector and overall sample descriptives.

    Include firm and article counts, return volatility, within/cross-sector
    rooted W2 distance, and the full ticker roster.
    """
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.shared.sample_descriptives import run_sample_descriptives

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


@commands.command("compute-co-mentions")
def cli_compute_co_mentions(
    events_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    universe_csv: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)],
    start_date: Annotated[str, typer.Option()],
    end_date: Annotated[str, typer.Option()],
    min_persistence_years: Annotated[int, typer.Option(min=1)],
) -> None:
    """Rebuild the comparator from published exact-two-firm event metadata."""
    from pipeline.stages.papers.paper5.co_mentions import compute_co_mentions

    compute_co_mentions(
        events_file,
        universe_csv,
        output_dir,
        start_date,
        end_date,
        min_persistence_years,
    )

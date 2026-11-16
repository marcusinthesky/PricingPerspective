"""Paper 1 ETNIC command adapters: the frozen extract and the alignment test."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-etnic-alignment")
def cli_paper1_etnic_alignment(
    extract: Annotated[Path, typer.Option(help="Roster-restricted ETNIC extract.")],
    crosswalk: Annotated[Path, typer.Option(help="Ticker/CIK/gvkey crosswalk.")],
    universe_csv: Annotated[Path, typer.Option(help="Ticker-sector metadata.")],
    distance_root: Annotated[
        Path,
        typer.Option(
            help="Typed-distance directory holding one subdirectory per geometry."
        ),
    ],
    geometry: Annotated[
        list[str], typer.Option(help="Geometries to Mantel-test; repeatable.")
    ],
    primary_geometry: Annotated[str, typer.Option(help="Paper 1 geometry.")],
    permutations: Annotated[int, typer.Option(min=1, help="Mantel permutations.")],
    bootstrap_draws: Annotated[int, typer.Option(min=1, help="Firm bootstrap draws.")],
    seed: Annotated[int, typer.Option(help="Permutation and bootstrap seed.")],
    output_summary: Annotated[
        Path, typer.Option(help="Alignment results JSON (results.json).")
    ],
    output_pairs: Annotated[Path, typer.Option(help="Per-pair ETNIC parquet.")],
) -> None:
    """Validate Paper 1 news geometry against the ETNIC doc2vec network."""
    own_float64()
    from pipeline.stages.papers.paper1.etnic_alignment import (
        EtnicAlignmentConfig,
        run_etnic_alignment,
    )

    run_etnic_alignment(
        EtnicAlignmentConfig(
            extract=extract,
            crosswalk=crosswalk,
            universe_csv=universe_csv,
            distance_root=distance_root,
            geometries=tuple(geometry),
            primary_geometry=primary_geometry,
            permutations=permutations,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
            output_summary=output_summary,
            output_pairs=output_pairs,
        )
    )
    typer.echo(f"Saved: {output_summary}")
    typer.echo(f"Saved: {output_pairs}")

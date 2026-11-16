"""Paper 1 neighbour outcome-split command adapters: the split and its exhibit."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-neighbour-outcome-split")
def cli_paper1_neighbour_outcome_split(
    distance_artifact_dir: Annotated[
        Path,
        typer.Option(help="Shared 2018-2020 windowed rooted W2 distance artifact."),
    ],
    returns_dir: Annotated[
        Path, typer.Option(help="Per-ticker simple-return directory.")
    ],
    universe_csv: Annotated[Path, typer.Option(help="Ticker-sector metadata.")],
    params_file: Annotated[
        Path,
        typer.Option(
            help=(
                "params.yaml; reads typed_analysis to resolve the selection "
                "geometry and its window."
            )
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            help=(
                "Result folder: pair_paths.parquet, selection.json, results.json, "
                "provenance.json."
            )
        ),
    ],
) -> None:
    """Run Paper 1's preregistered original-space neighbour outcome split."""
    # The shared W2 artifact is materialized by JAX-backed transport code, so
    # this consumer owns the repository's float64 precision policy as well.
    own_float64()
    from pipeline.stages.papers.paper1.neighbour_outcome_split import (
        run_neighbour_outcome_split,
    )

    run_neighbour_outcome_split(
        distance_artifact_dir=distance_artifact_dir,
        returns_dir=returns_dir,
        universe_csv=universe_csv,
        params_file=params_file,
        output_dir=output_dir,
    )


@commands.command("render-paper1-neighbour-outcome-split")
def cli_render_paper1_neighbour_outcome_split(
    results_file: Annotated[
        Path, typer.Option(help="paper1-neighbour-outcome-split results JSON.")
    ],
    provenance_file: Annotated[Path, typer.Option(help="Split provenance JSON.")],
    selection_file: Annotated[Path, typer.Option(help="Split selection JSON.")],
    pair_paths_file: Annotated[
        Path, typer.Option(help="Selected-pair cumulative return paths parquet.")
    ],
    numbers_path: Annotated[
        Path,
        typer.Option(help="Generated numbers.tex the displayed values are checked in."),
    ],
    figure_path: Annotated[Path, typer.Option(help="Output PGF figure.")],
    contract_path: Annotated[
        Path, typer.Option(help="Output registered semantic contract JSON.")
    ],
) -> None:
    """Render the Paper 1 neighbour holdout and registered semantic contract."""
    from pipeline.figures.paper1.neighbour_outcome_split import (
        NeighbourOutcomeSplitRenderPaths,
        render_paper1_neighbour_outcome_split,
    )

    render_paper1_neighbour_outcome_split(
        NeighbourOutcomeSplitRenderPaths(
            results_file=results_file,
            provenance_file=provenance_file,
            selection_file=selection_file,
            pair_paths_file=pair_paths_file,
            numbers_path=numbers_path,
            figure_path=figure_path,
            contract_path=contract_path,
        )
    )

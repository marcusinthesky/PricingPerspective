"""Paper 3 point-in-time squared-W2 archive command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper3-pit-w2")
def cli_paper3_pit_w2(
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads paper3.pit_vintage_grid and typed_analysis.",
        ),
    ],
    out_path: Annotated[
        Path,
        typer.Option(dir_okay=False, help="Output .npz of PIT squared-W2 matrices."),
    ],
    distance_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Root of the window-scoped typed statistical-distance tree.",
        ),
    ],
) -> None:
    """Assemble point-in-time squared-W2 matrices from typed artifacts."""
    from pipeline.stages.papers.paper3.pit_distance import compute_paper3_pit_distance

    compute_paper3_pit_distance(params_file, out_path, distance_root)

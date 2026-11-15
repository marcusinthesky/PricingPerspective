"""Paper 5 conceptual illustrations command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper5-illustrations")
def cli_render_paper5_illustrations(
    output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            help=(
                "Folder receiving spatial_exposure_arc.pgf and "
                "w2_operator_construction.pgf."
            ),
        ),
    ],
) -> None:
    """Render Paper 5's numerically honest conceptual illustrations."""
    from pipeline.figures.paper5.illustrations import render_paper5_illustrations

    render_paper5_illustrations(output_dir)

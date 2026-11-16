"""Paper 3 data-free illustration command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper3-illustrations")
def cli_render_paper3_illustrations(
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Manuscript images folder.")
    ],
) -> None:
    """Render the data-free conceptual pipeline illustration."""
    from pipeline.figures.paper3.exhibits import render_paper3_illustrations

    render_paper3_illustrations(output_dir)

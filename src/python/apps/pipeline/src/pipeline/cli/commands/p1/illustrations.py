"""Paper 1 analytic illustration command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper1-illustrations")
def cli_render_paper1_illustrations(
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for the schematic figure.")
    ],
) -> None:
    """Render Paper 1's analytic bound-construction schematic.

    Non-data-driven: the figure is a closed-form illustration with no data
    inputs, so this stage is separate from ``render-paper1-pcoa``.
    """
    from pipeline.figures.paper1.illustrations import render_paper1_illustrations

    render_paper1_illustrations(output_dir=output_dir)

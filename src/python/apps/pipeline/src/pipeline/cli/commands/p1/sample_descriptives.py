"""Paper 1 renderer for the shared sample-descriptives results."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper1-sample-descriptives")
def cli_render_paper1_sample_descriptives(
    results_file: Annotated[
        Path, typer.Option(help="compute-sample-descriptives results JSON.")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Generated per-sector table fragment.")
    ],
) -> None:
    """Render Paper 1's per-sector sample-descriptives table."""
    from pipeline.figures.paper1.sample_descriptives import (
        render_sample_descriptives_table,
    )

    render_sample_descriptives_table(summary_file=results_file, output_file=output_file)

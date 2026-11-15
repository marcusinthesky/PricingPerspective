"""Paper 5 empirical target-anchored W2 barycentric mixture figure."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper5-barycentric-mixture")
def cli_render_paper5_barycentric_mixture(
    barycentre_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Validated leave-one-out W2 target-projection artifact.",
        ),
    ],
    universe_csv: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Universe metadata used to validate the plotted ticker roster.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            help="Output PGF map of the target-anchored W2 mixture weights.",
        ),
    ],
) -> None:
    """Render Paper 5's empirical target-anchored W2 barycentric mixtures."""
    from pipeline.figures.paper5.figures import (
        BarycentricMixturePaths,
        render_barycentric_mixture_map,
    )

    render_barycentric_mixture_map(
        BarycentricMixturePaths(
            barycentre_dir=barycentre_dir,
            universe_csv=universe_csv,
            output_file=output_file,
        )
    )

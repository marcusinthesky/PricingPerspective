"""Paper 3 generated-numbers render command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper3-numbers")
def cli_render_paper3_numbers(
    numbers_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated numbers.tex to write.")
    ],
    certificate_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Certificate result folder."),
    ],
    validation_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Certificate-validation result folder."
        ),
    ],
    anatomy_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Anatomy result folder."),
    ],
) -> None:
    """Emit Paper 3 generated number bindings."""
    from pipeline.figures.paper3.numbers import Paper3NumberPaths, render_paper3_numbers

    render_paper3_numbers(
        Paper3NumberPaths(
            numbers_path=numbers_path,
            certificate_dir=certificate_dir,
            validation_dir=validation_dir,
            anatomy_dir=anatomy_dir,
        )
    )

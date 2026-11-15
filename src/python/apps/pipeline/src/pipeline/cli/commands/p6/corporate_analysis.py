"""Replay the retained full-sample corporate directional projection."""

import tempfile
from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper6-corporate-analysis")
def corporate_analysis(
    params_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "params.yaml"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "data/papers/paper6/corporate_analysis"
    ),
    workers: Annotated[int, typer.Option(min=1, max=3)] = 3,
    threads: Annotated[int, typer.Option(min=1)] = 1,
    scratch: Annotated[Path, typer.Option(file_okay=False)] = Path(
        tempfile.gettempdir(), "paper6-corporate-analysis"
    ),
    draws: Annotated[
        int | None,
        typer.Option(
            min=1, help="Explicit pilot override; use a separate output namespace."
        ),
    ] = None,
) -> None:
    """Rebuild point estimates and paired article redraws from frozen inputs."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper6.corporate_analysis.campaign import (
        read_config,
        run,
    )

    config = read_config(params_file)
    if draws is not None:
        config = config.model_copy(update={"draws": draws})
    result = run(output, config, workers=workers, threads=threads, scratch=scratch)
    typer.echo(
        f"Paper 6 corporate replay: {result['status']}; "
        f"{result['draws']} article redraws"
    )

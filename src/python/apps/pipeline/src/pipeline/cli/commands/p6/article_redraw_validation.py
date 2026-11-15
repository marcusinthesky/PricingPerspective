"""Nested calibration of the retained corporate article-selection procedure."""

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper6-article-redraw-validation")
def validate(
    design: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/papers/paper6/article_redraw_design.json"
    ),
    params_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "params.yaml"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "data/papers/paper6/article_redraw_validation"
    ),
    workers: Annotated[
        str, typer.Option(help="Bounded process count, or auto (at most three). ")
    ] = "auto",
    threads: Annotated[int, typer.Option(min=1)] = 1,
    pilot_only: Annotated[  # noqa: FBT002 - Typer option signature
        bool,
        typer.Option(
            help="Run four disjoint pilot archives and check serial/spawn agreement."
        ),
    ] = False,
) -> None:
    """Validate conditional article sensitivity on the fixed generated-law design."""
    from pipeline._kernels.paper6.article_redraw_validation import (
        ArticleRedrawValidationConfig,
    )
    from pipeline.io.params import load_params
    from pipeline.stages.papers.paper6.article_redraw_validation import run_validation

    config = ArticleRedrawValidationConfig.model_validate(
        load_params(params_file)["paper6"]["article_redraw_validation"]
    )
    result = run_validation(
        output,
        design=design,
        config=config,
        workers=workers,
        threads=threads,
        pilot_only=pilot_only,
    )
    typer.echo(f"Paper 6 article-redraw validation: {result['status']}")
    if result["status"] == "execution_failed":
        raise typer.Exit(code=1)


@commands.command("render-paper6-article-redraw-validation")
def render(
    results: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/papers/paper6/article_redraw_validation/results.json"
    ),
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "src/latex/projects/06_information_motion"
    ),
) -> None:
    """Project saved nested calibration results into their dedicated exhibit."""
    from pipeline.figures.paper6.article_redraw_validation import render as publish

    publish(results, project)

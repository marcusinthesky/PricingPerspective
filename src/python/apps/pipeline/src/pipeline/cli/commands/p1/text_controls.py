"""Lexical and encoder-control computations from published numeric inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()

InputDir = Annotated[Path, typer.Option(exists=True, file_okay=False)]

InputFile = Annotated[Path, typer.Option(exists=True, dir_okay=False)]

OutputDir = Annotated[Path, typer.Option(file_okay=False)]


@commands.command("paper1-tfidf-controls")
def cli_tfidf_controls(
    numeric_inputs: InputDir, params_file: InputFile, output_dir: OutputDir
) -> None:
    """Compute untuned firm cosine and document W2 with separate TF-IDF fits."""
    own_float64()
    from pipeline.stages.papers.paper1.text_controls import run_tfidf_controls

    run_tfidf_controls(
        numeric_inputs=numeric_inputs, params_file=params_file, output_dir=output_dir
    )


@commands.command("paper1-pretraining-distances")
def cli_pretraining_distances(
    embeddings_dir: InputDir,
    articles_dir: InputDir,
    model_key: Annotated[str, typer.Option()],
    params_file: InputFile,
    output_dir: OutputDir,
) -> None:
    """Compute canonical article-normalized W2 for a local encoder arm."""
    own_float64()
    from pipeline.stages.papers.paper1.text_controls import run_pretraining_distances

    run_pretraining_distances(
        embeddings_dir=embeddings_dir,
        articles_dir=articles_dir,
        model_key=model_key,
        params_file=params_file,
        output_dir=output_dir,
    )


@commands.command("paper1-text-controls")
def cli_text_controls(
    pairs_file: InputFile,
    node_counts_file: InputFile,
    control_dir: Annotated[list[Path], typer.Option(exists=True, file_okay=False)],
    output_dir: OutputDir,
) -> None:
    """Save comparison estimates and draws independently of publication."""
    own_float64()
    from pipeline.stages.papers.paper1.text_controls import run_text_controls

    run_text_controls(
        pairs_file=pairs_file,
        node_counts_file=node_counts_file,
        control_dirs=control_dir,
        output_dir=output_dir,
    )


@commands.command("render-paper1-text-controls")
def cli_render_text_controls(results_file: InputFile, output_dir: OutputDir) -> None:
    """Render the supplementary comparison table from saved results."""
    from pipeline.figures.paper1.text_controls import render_paper1_text_controls

    render_paper1_text_controls(results_file, output_dir)

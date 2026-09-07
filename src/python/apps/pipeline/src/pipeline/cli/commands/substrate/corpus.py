"""Substrate Corpus CLI command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.cli.options import CliOptions, model_command, options_from_context

commands = typer.Typer()


@commands.command("normalize-corpus")
def cli_normalize_corpus(
    input_path: Annotated[
        Path,
        typer.Option("--input", help="Path to nasdaq.jsonlines (raw corpus)."),
    ] = Path("data/raw/nasdaq.jsonlines"),
    output_parquet: Annotated[
        Path,
        typer.Option(help="Output parquet path."),
    ] = Path("data/shared/corpus/articles.parquet"),
    output_summary: Annotated[
        Path,
        typer.Option(help="Output summary YAML path."),
    ] = Path("data/shared/corpus/summary.yaml"),
    params: Annotated[
        Path,
        typer.Option(help="Path to params.yaml for ticker universe."),
    ] = Path("params.yaml"),
) -> None:
    """Deduplicate, date-parse, and filter NASDAQ data to the ticker universe."""
    from pipeline.stages.substrate.corpus import normalize_corpus

    normalize_corpus(input_path, output_parquet, output_summary, params)


class SubsampleCliOptions(CliOptions):
    """Inputs for a single-ticker corpus subsample."""

    symbol: Annotated[str, typer.Option(help="Ticker symbol (case-insensitive).")]
    input_path: Annotated[
        Path,
        typer.Option("--input", help="Path to normalized corpus parquet."),
    ] = Path("data/shared/corpus/articles.parquet")
    output: Annotated[
        Path,
        typer.Option(help="Output parquet path."),
    ] = Path("data/shared/subsampled_headlines/OUT.parquet")
    start_date: Annotated[
        str,
        typer.Option(help="Start date inclusive (YYYY-MM-DD)."),
    ] = "2018-01-01"
    end_date: Annotated[
        str,
        typer.Option(help="End date inclusive (YYYY-MM-DD)."),
    ] = "2022-12-31"
    min_body_chars: Annotated[
        int,
        typer.Option(help="Minimum body length in characters."),
    ] = 200
    n_samples: Annotated[
        int,
        typer.Option(help="Maximum rows to keep per ticker."),
    ] = 128


subsample_command = model_command(SubsampleCliOptions)


@commands.command("subsample", cls=subsample_command)
def cli_subsample(ctx: typer.Context) -> None:
    """Subsample normalized corpus for a single ticker."""
    from pipeline.stages.substrate.corpus import SubsampleOptions, subsample

    options = options_from_context(ctx, SubsampleCliOptions)
    subsample(
        options.symbol,
        options.input_path,
        options.output,
        SubsampleOptions(
            start_date=options.start_date,
            end_date=options.end_date,
            min_body_chars=options.min_body_chars,
            n_samples=options.n_samples,
        ),
    )

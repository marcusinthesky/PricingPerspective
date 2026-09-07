"""Substrate Embeddings CLI command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context

commands = typer.Typer()


class GenerateEmbeddingsCliOptions(CliOptions):
    """Inputs for one ticker's embedding request."""

    symbol: str = Field(description="Ticker symbol to process.")
    parquet_path: Path = Field(description="Path to input parquet (subsample).")
    model_key: str = Field(description="Model key from params.yaml embedding.models.")
    output_file: Path = Field(description="Output parquet file path.")
    params_file: Path = Field(
        default=Path("params.yaml"),
        description="Path to params.yaml.",
    )
    cache_dir: Path = Field(
        default=Path("data/.emb_cache"),
        description="Cache directory for embedding results.",
    )
    skip_existing: Annotated[
        bool, typer.Option(help="Skip if output file already exists.")
    ] = False


generate_embeddings_command = model_command(GenerateEmbeddingsCliOptions)


@commands.command("generate-embeddings", cls=generate_embeddings_command)
def cli_generate_embeddings(ctx: typer.Context) -> None:
    """Generate embeddings via OpenRouter for a single ticker subsample parquet."""
    from pipeline.stages.substrate.embeddings import (
        GenerateEmbeddingOptions,
        generate_embeddings,
    )

    options = options_from_context(ctx, GenerateEmbeddingsCliOptions)
    generate_embeddings(
        options.symbol,
        options.parquet_path,
        GenerateEmbeddingOptions(
            model_key=options.model_key,
            output_file=options.output_file,
            params_file=options.params_file,
            cache_dir=options.cache_dir,
            skip_existing=options.skip_existing,
        ),
    )


@commands.command("validate-coverage")
def cli_validate_coverage(
    model_key: str = typer.Option(
        ..., help="Model key from params.yaml embedding.models."
    ),
    embeddings_dir: Path = typer.Option(
        ..., help="Directory containing per-ticker parquets."
    ),
    output_dir: Path = typer.Option(
        ..., help="Output directory for summary.yaml and coverage.csv."
    ),
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml."),
) -> None:
    """Validate embedding coverage and run energy-distance sanity checks."""
    from pipeline.stages.substrate.embeddings import validate_coverage

    validate_coverage(model_key, embeddings_dir, output_dir, params_file)

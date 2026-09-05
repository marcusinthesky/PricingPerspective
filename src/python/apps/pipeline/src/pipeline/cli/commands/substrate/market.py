"""Substrate Market CLI command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.cli.options import CliOptions, model_command, options_from_context

commands = typer.Typer()


@commands.command("download-market-data")
def cli_download_market_data(
    start_date: Annotated[str, typer.Option(help="Start date in YYYY-MM-DD format.")],
    end_date: Annotated[str, typer.Option(help="End date in YYYY-MM-DD format.")],
    ticker: Annotated[str, typer.Option(help="Ticker symbol to download.")],
    output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            writable=True,
            help="Path to the output directory.",
        ),
    ] = Path("data/shared/market_data"),
    universe_csv: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            readable=True,
            help="Universe CSV supplying MarketSymbol/MarketSource per firm.",
        ),
    ] = None,
    archive_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            help="Root for recorded histories of MarketSource='archive' firms.",
        ),
    ] = None,
) -> None:
    """Download total return data for a given ticker."""
    from pipeline.stages.substrate.market import download_market_data

    download_market_data(
        start_date, end_date, ticker, output_dir, universe_csv, archive_dir
    )


@commands.command("compute-returns")
def cli_compute_returns(
    market_data: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to market data parquet file for a single ticker.",
        ),
    ],
    start_date: Annotated[
        str,
        typer.Option("--start-date", help="Start date for filtering (YYYY-MM-DD)."),
    ],
    end_date: Annotated[
        str,
        typer.Option("--end-date", help="End date for filtering (YYYY-MM-DD)."),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            writable=True,
            help="Output parquet file to save returns.",
        ),
    ],
) -> None:
    """Compute daily returns from market data for a single ticker."""
    from pipeline.stages.substrate.market import compute_returns

    compute_returns(market_data, start_date, end_date, output_file)


class ComputeCovarianceCliOptions(CliOptions):
    """Inputs and canonical panel contract for covariance estimation."""

    returns_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing returns parquet files.",
        ),
    ]
    output_file: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            writable=True,
            help="Output parquet file to save covariance matrix.",
        ),
    ]
    min_observations: Annotated[
        int,
        typer.Option(min=1, help="Minimum number of common observations required."),
    ] = 100
    expected_tickers: Annotated[
        int,
        typer.Option(
            min=1,
            help="Required number of ordered ticker columns in the canonical panel.",
        ),
    ] = 100
    expected_observations: Annotated[
        int,
        typer.Option(
            min=1,
            help="Required number of complete dates in the canonical panel.",
        ),
    ] = 1207
    expected_start_date: Annotated[
        str,
        typer.Option(help="Required first complete-case date (YYYY-MM-DD)."),
    ] = "2018-03-19"
    expected_end_date: Annotated[
        str,
        typer.Option(help="Required last complete-case date (YYYY-MM-DD)."),
    ] = "2022-12-30"


compute_covariance_command = model_command(ComputeCovarianceCliOptions)


@commands.command("compute-covariance", cls=compute_covariance_command)
def cli_compute_covariance(ctx: typer.Context) -> None:
    """Compute variance-covariance matrix from returns data."""
    from pipeline.stages.substrate.market import (
        ReturnPanelContract,
        compute_covariance,
    )

    options = options_from_context(ctx, ComputeCovarianceCliOptions)
    compute_covariance(
        options.returns_dir,
        options.output_file,
        ReturnPanelContract(
            min_observations=options.min_observations,
            expected_tickers=options.expected_tickers,
            expected_observations=options.expected_observations,
            expected_start_date=options.expected_start_date,
            expected_end_date=options.expected_end_date,
        ),
    )

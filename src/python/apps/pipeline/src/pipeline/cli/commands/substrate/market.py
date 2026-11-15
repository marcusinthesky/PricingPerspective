"""CLI covariance computation from frozen return inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("compute-covariance")
def cli_compute_covariance(
    returns_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing returns parquet files.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            writable=True,
            help="Output parquet file to save covariance matrix.",
        ),
    ],
    min_observations: Annotated[
        int,
        typer.Option(min=1, help="Minimum number of common observations required."),
    ],
    expected_tickers: Annotated[
        int,
        typer.Option(
            min=1,
            help="Required number of ordered ticker columns in the canonical panel.",
        ),
    ],
    expected_observations: Annotated[
        int,
        typer.Option(
            min=1,
            help="Required number of complete dates in the canonical panel.",
        ),
    ],
    expected_start_date: Annotated[
        str,
        typer.Option(help="Required first complete-case date (YYYY-MM-DD)."),
    ],
    expected_end_date: Annotated[
        str,
        typer.Option(help="Required last complete-case date (YYYY-MM-DD)."),
    ],
) -> None:
    """Compute variance-covariance matrix from returns data."""
    from pipeline.stages.substrate.market import (
        ReturnPanelContract,
        compute_covariance,
    )

    compute_covariance(
        returns_dir,
        output_file,
        ReturnPanelContract(
            min_observations=min_observations,
            expected_tickers=expected_tickers,
            expected_observations=expected_observations,
            expected_start_date=expected_start_date,
            expected_end_date=expected_end_date,
        ),
    )

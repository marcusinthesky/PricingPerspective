"""Market data, returns, and covariance computation.

Provides:
- ``download_market_data`` — download OHLCV + adjusted close via yfinance.
- ``compute_returns`` — compute daily simple returns from adjusted close.
- ``compute_covariance`` — compute variance-covariance matrix from returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# Runtime import, not TYPE_CHECKING: `download_market_data` constructs the
# default archive root as a real Path when MarketSource is 'archive'.
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yfinance as yf

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Fallback contract for the in-sample panel when a caller supplies no explicit
# expectation. The authoritative values are the per-window `expected_*` keys in
# params.yaml:return_panels, which dvc.yaml passes on every stage invocation;
# these constants only cover direct library use.
CANONICAL_RETURN_PANEL_TICKERS = 100
CANONICAL_RETURN_PANEL_OBSERVATIONS = 1207
CANONICAL_RETURN_PANEL_START_DATE = "2018-03-19"
CANONICAL_RETURN_PANEL_END_DATE = "2022-12-30"


@dataclass(frozen=True)
class ReturnPanelContract:
    """Required size and date coverage for a complete-case return panel."""

    min_observations: int = 100
    expected_tickers: int | None = CANONICAL_RETURN_PANEL_TICKERS
    expected_observations: int | None = CANONICAL_RETURN_PANEL_OBSERVATIONS
    expected_start_date: str | None = CANONICAL_RETURN_PANEL_START_DATE
    expected_end_date: str | None = CANONICAL_RETURN_PANEL_END_DATE


DEFAULT_RETURN_PANEL_CONTRACT = ReturnPanelContract()


@dataclass(frozen=True)
class ReturnPanelSummary:
    """Observed complete-case date bounds and row count."""

    first_date: str
    last_date: str
    observations: int


class _MissingDateColumnError(ValueError):
    def __init__(self) -> None:
        super().__init__("return panel query did not include its Date column")


class _EmptyTickerPanelError(ValueError):
    def __init__(self) -> None:
        super().__init__("return panel has no ticker columns")


class _MissingTickerColumnError(ValueError):
    def __init__(self, ticker: str) -> None:
        super().__init__(f"return panel is missing ordered ticker column {ticker!r}")


class _ReturnPanelShapeError(ValueError):
    def __init__(self, shape: tuple[int, ...], dates: int, tickers: int) -> None:
        super().__init__(
            "complete-case return panel shape does not match dates/ticker ordering: "
            f"matrix={shape}, dates={dates}, tickers={tickers}"
        )


class _NonfiniteReturnPanelError(ValueError):
    def __init__(self) -> None:
        super().__init__("complete-case return panel still contains nonfinite values")


class _EmptyMarketDownloadError(RuntimeError):
    def __init__(self, ticker: str, start_date: str, end_date: str) -> None:
        super().__init__(
            f"yfinance returned no data for {ticker!r} over {start_date}..{end_date}; "
            "refusing to write an empty market-data parquet"
        )


class _UnknownMarketSourceError(ValueError):
    def __init__(self, ticker: str, source: str) -> None:
        super().__init__(
            f"universe.csv declares MarketSource {source!r} for {ticker!r}; "
            "supported sources are 'yfinance' and 'archive'"
        )


class _MissingUniverseRowError(ValueError):
    def __init__(self, ticker: str, universe_csv: Path) -> None:
        super().__init__(f"{universe_csv} has no row for ticker {ticker!r}")


class _MissingArchiveError(RuntimeError):
    def __init__(self, ticker: str, archive_path: Path) -> None:
        super().__init__(
            f"{ticker!r} is declared MarketSource='archive' in universe.csv because "
            f"its listing ended and a live provider request returns nothing, but "
            f"{archive_path} does not exist. Supply the recorded price history at "
            "that path; do not substitute a successor ticker or forward-fill."
        )


class _MissingReturnCountError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("DuckDB returned no row for the return count")


class _MissingCombinedReturnCountError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("DuckDB returned no row for the combined return count")


class _TickerCountContractError(ValueError):
    def __init__(self, expected: int, observed: int) -> None:
        super().__init__(
            f"canonical return panel requires {expected} tickers, found {observed}"
        )


class _ObservationCountContractError(ValueError):
    def __init__(self, expected: int, observed: int) -> None:
        super().__init__(
            f"canonical return panel requires {expected} observations, found {observed}"
        )


class _StartDateContractError(ValueError):
    def __init__(self, expected: str, observed: str) -> None:
        super().__init__(
            f"canonical return panel must start {expected}, found {observed}"
        )


class _EndDateContractError(ValueError):
    def __init__(self, expected: str, observed: str) -> None:
        super().__init__(
            f"canonical return panel must end {expected}, found {observed}"
        )


def _quote_duckdb_identifier(identifier: str) -> str:
    """Quote an identifier according to DuckDB's SQL identifier rules."""
    return '"' + identifier.replace('"', '""') + '"'


def _complete_case_return_panel(
    result: Mapping[str, object], ticker_list: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve nullable-column masks and return finite complete-case rows.

    DuckDB's ``fetchnumpy`` represents nullable numeric columns as masked arrays.
    ``np.column_stack`` discards those per-column masks and exposes the masked
    buffer values (zeros in the affected canonical ZS cells). Mask-aware stacking
    must therefore happen before any ndarray conversion.
    """
    if "Date" not in result:
        raise _MissingDateColumnError
    if not ticker_list:
        raise _EmptyTickerPanelError

    columns: list[np.ma.MaskedArray] = []
    for ticker in ticker_list:
        if ticker not in result:
            raise _MissingTickerColumnError(ticker)
        column = np.ma.asarray(result[ticker], dtype=np.float64)
        data = np.asarray(column.data, dtype=np.float64)
        mask = np.ma.getmaskarray(column) | ~np.isfinite(data)
        columns.append(np.ma.array(data, mask=mask, copy=False))

    masked_matrix = np.ma.column_stack(columns)
    row_has_missing = np.ma.getmaskarray(masked_matrix).any(axis=1)
    clean_dates = np.asarray(result["Date"])[~row_has_missing]
    clean_data = np.asarray(masked_matrix[~row_has_missing].filled(np.nan))

    if clean_data.shape != (clean_dates.size, len(ticker_list)):
        raise _ReturnPanelShapeError(
            clean_data.shape, clean_dates.size, len(ticker_list)
        )
    if not np.isfinite(clean_data).all():
        raise _NonfiniteReturnPanelError
    return clean_dates, clean_data


def _resolve_market_route(ticker: str, universe_csv: Path | None) -> tuple[str, str]:
    """Resolve ``(market_symbol, market_source)`` for one roster ticker.

    With no universe file the historical behaviour is preserved exactly: fetch
    the ticker itself from yfinance.
    """
    if universe_csv is None:
        return ticker, "yfinance"

    with duckdb.connect() as con:
        row = con.execute(
            """
            SELECT UPPER(MarketSymbol), MarketSource
            FROM read_csv_auto(?)
            WHERE UPPER(Symbol) = ?
            """,
            [str(universe_csv), ticker.upper()],
        ).fetchone()
    if row is None:
        raise _MissingUniverseRowError(ticker, universe_csv)
    market_symbol, market_source = str(row[0]), str(row[1]).strip().lower()
    if market_source not in {"yfinance", "archive"}:
        raise _UnknownMarketSourceError(ticker, market_source)
    return market_symbol, market_source


def download_market_data(
    start_date: str,
    end_date: str,
    ticker: str,
    output_dir: Path,
    universe_csv: Path | None = None,
    archive_dir: Path | None = None,
) -> None:
    """Materialize total return data for one ticker from its declared source.

    ``universe.csv`` carries a ``MarketSymbol``/``MarketSource`` pair per firm.
    ``yfinance`` fetches live; ``archive`` copies a recorded history for a firm
    whose listing has ended, so a delisting is an explicit, reviewable artifact
    rather than an empty response indistinguishable from a network failure.

    ``auto_adjust=True`` keeps the schema consistent across tickers (adjusted
    ``Close``; no separate ``Adj Close`` column). A transient yfinance/network
    failure returns an *empty* frame; we retry once and then fail loudly rather
    than write an empty parquet, which would silently break ``compute_returns``
    downstream (the empty frame has no ``Date`` column).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.parquet"
    market_symbol, market_source = _resolve_market_route(ticker, universe_csv)

    if market_source == "archive":
        archive_root = archive_dir or Path("data/archive/market_data")
        archive_path = archive_root / f"{market_symbol}.parquet"
        if not archive_path.is_file():
            raise _MissingArchiveError(ticker, archive_path)
        archived = pq.read_table(archive_path)
        pq.write_table(archived, output_path)
        return

    # One retry — transient yfinance/network failures return an empty frame.
    data = yf.download(
        market_symbol, start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if data is None or data.empty:
        data = yf.download(
            market_symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    if data is None or data.empty:
        raise _EmptyMarketDownloadError(market_symbol, start_date, end_date)
    data.to_parquet(output_path)


def compute_returns(
    market_data: Path,
    start_date: str,
    end_date: str,
    output_file: Path,
) -> None:
    """Compute daily simple returns from market data for a single ticker.

    Reads a market data parquet file, computes daily returns, and saves
    to a parquet file with columns: Date, ticker, return.

    The vendor close series is deliberately not carried into the output: it is
    identical to the withheld ``market_data`` Close column, and the return
    panels are published (see ``scripts/strip_close_price``).

    One consequence: this stage sees a single window, so the first row's return
    is null and the cross-window seam return cannot be produced here -- the OOS
    window's ``market_data`` does not contain the last in-sample close. The
    published panels carry that value in the OOS leading row, backfilled by the
    operator above. ``compute_returns`` is therefore frozen in ``dvc.yaml``;
    unfreezing and re-running yields a null seam return until this stage is
    given the prior close as an explicit input.
    """
    ticker = market_data.stem
    typer.echo(f"Computing returns for {ticker}")
    typer.echo(f"Date range: {start_date} to {end_date}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not market_data.exists():
        typer.echo(f"Error: Market data not found at {market_data}", err=True)
        raise typer.Exit(code=1)

    close_col = _quote_duckdb_identifier(f"('Close', '{ticker}')")

    with duckdb.connect(":memory:") as con:
        returns_query = Template("""
        CREATE OR REPLACE TABLE returns AS
        SELECT
            Date,
            ? as ticker,
            ($close_column / LAG($close_column) OVER (ORDER BY Date) - 1) as return
        FROM read_parquet(?)
        WHERE Date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        ORDER BY Date;
        """).substitute(close_column=close_col)

        typer.echo("Computing returns...")
        con.execute(returns_query, [ticker, str(market_data), start_date, end_date])

        count_result = con.execute("SELECT COUNT(*) FROM returns").fetchone()
        if count_result is None:
            raise _MissingReturnCountError
        typer.echo(f"Total return records: {count_result[0]}")

        summary = con.execute(
            """
            SELECT
                ticker,
                COUNT(*) as days,
                MIN(Date) as first_date,
                MAX(Date) as last_date,
                AVG(return) as mean_return,
                STDDEV(return) as std_return
            FROM returns
            WHERE return IS NOT NULL
            GROUP BY ticker
            """
        ).fetchone()

        if summary:
            typer.echo(f"\nSummary for {ticker}:")
            typer.echo(f"  Days: {summary[1]}")
            typer.echo(f"  Date range: {summary[2]} to {summary[3]}")
            typer.echo(f"  Mean return: {summary[4]:.6f}")
            typer.echo(f"  Std return: {summary[5]:.6f}")

        typer.echo(f"\nSaving to {output_file}")
        con.table("returns").write_parquet(str(output_file))

    typer.echo("Returns computation complete!")


def _load_wide_return_panel(
    returns_dir: Path,
) -> tuple[list[str], Mapping[str, object]]:
    """Load per-ticker returns and pivot them into a date-indexed wide panel."""
    return_files = list(returns_dir.glob("*.parquet"))
    if not return_files:
        typer.echo("Error: No return files found", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Found {len(return_files)} ticker return files")

    with duckdb.connect(":memory:") as con:
        con.execute(
            """
            CREATE OR REPLACE TABLE all_returns AS
            SELECT Date, ticker, return
            FROM read_parquet(?)
            WHERE return IS NOT NULL
            """,
            [[str(return_file) for return_file in return_files]],
        )
        count_row = con.execute("SELECT COUNT(*) FROM all_returns").fetchone()
        if count_row is None:
            raise _MissingCombinedReturnCountError
        typer.echo(f"Total return observations: {count_row[0]}")

        tickers = con.execute(
            "SELECT DISTINCT ticker FROM all_returns ORDER BY ticker"
        ).fetchall()
        ticker_list = [str(row[0]) for row in tickers]
        typer.echo(f"Unique tickers: {len(ticker_list)}")
        typer.echo("Pivoting returns to wide format...")
        con.execute(
            """
            CREATE OR REPLACE TABLE returns_wide AS
            PIVOT all_returns
            ON ticker
            USING FIRST(return)
            GROUP BY Date
            ORDER BY Date
            """
        )
        ticker_columns = ", ".join(map(_quote_duckdb_identifier, ticker_list))
        projection = f"Date, {ticker_columns}"
        result = (
            con.table("returns_wide").project(projection).order("Date").fetchnumpy()
        )
    return ticker_list, result


def _validate_return_panel_contract(
    clean_dates: np.ndarray,
    clean_data: np.ndarray,
    ticker_list: list[str],
    contract: ReturnPanelContract,
) -> tuple[str, str]:
    """Validate complete-case dimensions and return the observed date bounds."""
    observations = clean_data.shape[0]
    if (
        contract.expected_tickers is not None
        and len(ticker_list) != contract.expected_tickers
    ):
        raise _TickerCountContractError(contract.expected_tickers, len(ticker_list))
    if (
        contract.expected_observations is not None
        and observations != contract.expected_observations
    ):
        raise _ObservationCountContractError(
            contract.expected_observations, observations
        )
    first_date = str(clean_dates.min())[:10]
    last_date = str(clean_dates.max())[:10]
    if (
        contract.expected_start_date is not None
        and first_date != contract.expected_start_date
    ):
        raise _StartDateContractError(contract.expected_start_date, first_date)
    if (
        contract.expected_end_date is not None
        and last_date != contract.expected_end_date
    ):
        raise _EndDateContractError(contract.expected_end_date, last_date)
    if observations < contract.min_observations:
        typer.echo(
            f"Error: Only {observations} common observations, "
            f"need at least {contract.min_observations}",
            err=True,
        )
        raise typer.Exit(code=1)
    return first_date, last_date


def _covariance_output_table(
    ticker_list: list[str],
    covariance: np.ndarray,
    correlation: np.ndarray,
    panel: ReturnPanelSummary,
) -> pa.Table:
    """Build the long covariance table and attach return-panel provenance."""
    variances = np.diag(covariance)
    ticker_i, ticker_j = np.meshgrid(ticker_list, ticker_list, indexing="ij")
    table = pa.table(
        {
            "ticker_i": ticker_i.ravel().tolist(),
            "ticker_j": ticker_j.ravel().tolist(),
            "covariance": covariance.ravel().tolist(),
            "correlation": correlation.ravel().tolist(),
            "variance_i": np.repeat(variances, len(ticker_list)).tolist(),
            "variance_j": np.tile(variances, len(ticker_list)).tolist(),
        }
    )
    return table.replace_schema_metadata(
        {
            b"return_panel_policy": b"complete-case; masks preserved; no imputation",
            b"return_panel_rows": str(panel.observations).encode(),
            b"return_panel_columns": str(len(ticker_list)).encode(),
            b"return_panel_start_date": panel.first_date.encode(),
            b"return_panel_end_date": panel.last_date.encode(),
            b"return_panel_tickers": ",".join(ticker_list).encode(),
        }
    )


def compute_covariance(
    returns_dir: Path,
    output_file: Path,
    contract: ReturnPanelContract = DEFAULT_RETURN_PANEL_CONTRACT,
) -> None:
    """Compute variance-covariance matrix from returns data.

    Reads individual ticker return files, pivots to wide format, and computes
    the variance-covariance matrix. Saves in long format with columns:
    ticker_i, ticker_j, covariance, correlation, variance_i, variance_j.
    """
    typer.echo(f"Loading returns from {returns_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    ticker_list, result = _load_wide_return_panel(returns_dir)
    clean_dates, clean_data = _complete_case_return_panel(result, ticker_list)

    n_obs = clean_data.shape[0]
    typer.echo(f"Computing covariance on {n_obs} common observations")
    first_date, last_date = _validate_return_panel_contract(
        clean_dates, clean_data, ticker_list, contract
    )

    cov_matrix = np.cov(clean_data, rowvar=False)
    corr_matrix = np.corrcoef(clean_data, rowvar=False)

    variances = np.diag(cov_matrix)

    typer.echo(f"Covariance matrix shape: {cov_matrix.shape}")
    typer.echo(f"Mean variance: {variances.mean():.6f}")
    mean_correlation = corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean()
    typer.echo(f"Mean correlation: {mean_correlation:.6f}")

    n_tickers = len(ticker_list)
    typer.echo("Converting to long format...")
    table = _covariance_output_table(
        ticker_list,
        cov_matrix,
        corr_matrix,
        ReturnPanelSummary(
            first_date=first_date,
            last_date=last_date,
            observations=n_obs,
        ),
    )

    typer.echo(f"Saving to {output_file}")
    pq.write_table(table, output_file)

    typer.echo("\n=== Summary Statistics ===")
    typer.echo(f"Number of tickers: {n_tickers}")
    typer.echo(f"Common observations: {n_obs}")
    typer.echo(f"Total pairs: {n_tickers * n_tickers}")
    typer.echo("\nVariance statistics:")
    typer.echo(f"  Min: {variances.min():.6f}")
    typer.echo(f"  Max: {variances.max():.6f}")
    typer.echo(f"  Mean: {variances.mean():.6f}")
    typer.echo(f"  Median: {np.median(variances):.6f}")

    off_diag_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    typer.echo("\nCorrelation statistics (off-diagonal):")
    typer.echo(f"  Min: {off_diag_corr.min():.6f}")
    typer.echo(f"  Max: {off_diag_corr.max():.6f}")
    typer.echo(f"  Mean: {off_diag_corr.mean():.6f}")
    typer.echo(f"  Median: {np.median(off_diag_corr):.6f}")

    typer.echo("\nCovariance matrix computation complete!")

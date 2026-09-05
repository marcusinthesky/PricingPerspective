"""Mantel test between a typed distance and covariance-derived structure.

Provides ``run_mantel_tests`` — compute a Mantel test between a validated
typed-distance artifact and a return-correlation matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd
import typer
from jcor.association.mantel import mantel_test

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

logger = logging.getLogger(__name__)
MIN_MANTEL_TICKERS = 3
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class MantelOptions:
    """Permutation and persistence options for a Mantel analysis."""

    method: Literal["pearson", "spearman"] = "pearson"
    permutations: int = 10_000
    seed: int = 42
    null_output_file: Path | None = None


DEFAULT_MANTEL_OPTIONS = MantelOptions()


class InvalidCovarianceColumnError(ValueError):
    """Raised when a matrix loader receives an unsupported value column."""

    def __init__(self, column: str) -> None:
        """Describe the unsupported column name."""
        super().__init__(
            f"value_col must be 'covariance' or 'correlation', got {column!r}"
        )


class NoCommonTickersError(ValueError):
    """Raised when two matrices have no shared ticker universe."""

    def __init__(self) -> None:
        """Initialize the fixed no-overlap diagnostic."""
        super().__init__("No common tickers between matrices")


# ---------------------------------------------------------------------------
# Shared matrix loaders
# ---------------------------------------------------------------------------


def _load_covariance_matrix(
    parquet_path: Path,
    value_col: str = "covariance",
) -> tuple[list[str], np.ndarray]:
    """Load covariance (or correlation) matrix."""
    if value_col not in {"covariance", "correlation"}:
        raise InvalidCovarianceColumnError(value_col)
    selected_column = "covariance" if value_col == "covariance" else "correlation"
    query = """
        SELECT ticker_i, ticker_j, covariance
        FROM read_parquet(?)
        ORDER BY ticker_i, ticker_j
    """
    if selected_column == "correlation":
        query = """
            SELECT ticker_i, ticker_j, correlation
            FROM read_parquet(?)
            ORDER BY ticker_i, ticker_j
        """
    con = duckdb.connect(":memory:")
    df = con.execute(query, [str(parquet_path)]).df()
    con.close()

    tickers = sorted(set(df["ticker_i"].unique()) | set(df["ticker_j"].unique()))
    n = len(tickers)

    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    matrix = np.zeros((n, n))

    for _, row in df.iterrows():
        i = ticker_to_idx[row["ticker_i"]]
        j = ticker_to_idx[row["ticker_j"]]
        matrix[i, j] = row[selected_column]
        matrix[j, i] = row[selected_column]

    return tickers, matrix


# ---------------------------------------------------------------------------
# Mantel test
# ---------------------------------------------------------------------------
#
# The permutation test itself (jointly permuting rows/columns of one distance
# matrix, comparing upper-triangular correlation against the null) lives in
# jcor.mantel_test — a generic, domain-agnostic primitive with no ticker/
# covariance coupling. This module keeps only the DuckDB loading, ticker
# alignment, and reporting glue around it.


def _align_matrices(
    tickers1: list[str],
    matrix1: np.ndarray,
    tickers2: list[str],
    matrix2: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align two matrices to have the same tickers in the same order."""
    common_tickers = sorted(set(tickers1) & set(tickers2))
    if not common_tickers:
        raise NoCommonTickersError

    idx1 = [tickers1.index(t) for t in common_tickers]
    idx2 = [tickers2.index(t) for t in common_tickers]

    aligned1 = matrix1[np.ix_(idx1, idx1)]
    aligned2 = matrix2[np.ix_(idx2, idx2)]

    return common_tickers, aligned1, aligned2


def _load_typed_distance_metric(
    artifact_dir: Path,
) -> tuple[list[str], np.ndarray, str]:
    """Load a validated typed statistical-distance artifact."""
    frame, summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity={},
    )
    distance_id = str(summary["distance_id"])
    item_ids = summary["item_ids"]
    if not isinstance(item_ids, list):
        message = "typed distance summary item_ids must be a list"
        raise TypeError(message)
    tickers = [str(item) for item in item_ids]
    functional = (
        frame["value"].to_numpy(dtype=np.float64).reshape(len(tickers), len(tickers))
    )
    return tickers, functional, distance_id


def run_mantel_tests(
    distance_artifact_dir: Path,
    covariance_matrix: Path,
    output_file: Path,
    options: MantelOptions = DEFAULT_MANTEL_OPTIONS,
) -> None:
    """Compute a Mantel test between a typed distance and covariance matrix.

    Tests whether the distance structure of news embeddings correlates with the
    distance structure of stock returns derived from correlations.
    """
    method = options.method
    permutations = options.permutations
    seed = options.seed
    typer.echo("Loading typed distance results...")
    distance_tickers, distance_dist, distance_id = _load_typed_distance_metric(
        distance_artifact_dir
    )
    typer.echo(f"  {distance_id}: {len(distance_tickers)} tickers")

    typer.echo("Loading covariance matrix...")
    cov_tickers, cov_corr = _load_covariance_matrix(covariance_matrix, "correlation")
    typer.echo(f"  Covariance: {len(cov_tickers)} tickers")

    typer.echo("Aligning matrices...")
    common_tickers, aligned_distance, aligned_corr = _align_matrices(
        distance_tickers, distance_dist, cov_tickers, cov_corr
    )
    typer.echo(f"  Common tickers: {len(common_tickers)}")

    if len(common_tickers) < MIN_MANTEL_TICKERS:
        typer.echo(
            f"Error: Need at least {MIN_MANTEL_TICKERS} common tickers", err=True
        )
        raise typer.Exit(code=1)

    # Convert correlation to distance
    aligned_cov_dist = np.sqrt(2 * (1 - aligned_corr))

    typer.echo(f"\nComputing Mantel test ({method}, {permutations} permutations)...")
    result = mantel_test(
        aligned_distance,
        aligned_cov_dist,
        method=method,
        num_permutations=permutations,
        seed=seed,
    )
    correlation, p_value = result.correlation, result.pvalue

    typer.echo("\n=== Mantel Test Results ===")
    typer.echo(f"Method: {method}")
    typer.echo(f"Correlation: {correlation:.6f}")
    typer.echo(f"P-value: {p_value:.6f}")
    significant = "Yes" if p_value < SIGNIFICANCE_LEVEL else "No"
    typer.echo(f"Significant: {significant} (α={SIGNIFICANCE_LEVEL})")

    indices = np.triu_indices(len(common_tickers), k=1)
    distance_vals = aligned_distance[indices]
    cov_vals = aligned_cov_dist[indices]

    typer.echo("\n=== Matrix Statistics ===")
    typer.echo(f"Common tickers: {len(common_tickers)}")
    typer.echo(f"Pairwise comparisons: {len(distance_vals)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "method": [method],
        "correlation": [correlation],
        "pvalue": [p_value],
        "permutations": [permutations],
        "seed": [seed],
        "distance_id": [distance_id],
        "n_tickers": [len(common_tickers)],
        "n_pairs": [len(distance_vals)],
        "distance_mean": [distance_vals.mean()],
        "distance_std": [distance_vals.std()],
        "return_dist_mean": [cov_vals.mean()],
        "return_dist_std": [cov_vals.std()],
        "null_hypothesis": [
            "firm-label exchangeability conditional on the fixed return-distance matrix"
        ],
        "null_exceedance_count": [
            int(np.sum(result.null_distribution >= result.correlation))
        ],
        "null_q025": [float(np.percentile(result.null_distribution, 2.5))],
        "null_q50": [float(np.percentile(result.null_distribution, 50.0))],
        "null_q975": [float(np.percentile(result.null_distribution, 97.5))],
    }

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    null_path = (
        options.null_output_file or output_file.parent / "null_distribution.parquet"
    )
    pd.DataFrame(
        {
            "permutation_id": np.arange(permutations, dtype=np.int32),
            "correlation": result.null_distribution,
        }
    ).to_parquet(null_path, index=False)

    typer.echo("\nMantel test complete!")

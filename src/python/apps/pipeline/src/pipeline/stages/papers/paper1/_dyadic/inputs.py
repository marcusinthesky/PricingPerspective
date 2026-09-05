"""Paper-1 file loading and dyadic-control construction."""

from __future__ import annotations

import logging
from string import Template
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd
from jcor.model.dyadic import DyadicInputError, DyadicSample
from scipy.stats import pearsonr, spearmanr

from pipeline._kernels.arrays import (
    InvalidNormalizationRowsError,
    l2_normalize_rows,
)
from pipeline.stages.papers.paper1._dyadic.contracts import (
    PreparedDyadicInputs,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact
from pipeline.stages.substrate.energy_metric import _energy_metric_from_functional
from pipeline.stages.substrate.energy_shared import load_energy_functional_pairs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline.stages.papers.paper1._dyadic.contracts import DyadicConfoundConfig

_CRISIS_START = "2020-02-01"
_CRISIS_END = "2020-04-30"


def load_covariance_long(covariance_matrix: Path) -> pd.DataFrame:
    """Load the canonical upper-triangle return covariance panel."""
    con = duckdb.connect(":memory:")
    frame = con.execute(
        """
        SELECT ticker_i, ticker_j, covariance, correlation
        FROM read_parquet(?)
        WHERE ticker_i < ticker_j
        ORDER BY ticker_i, ticker_j
        """,
        [str(covariance_matrix)],
    ).df()
    con.close()
    return frame


def load_w2_long(distance_artifact_dir: Path) -> pd.DataFrame:
    """Load the governed Qwen3-Embedding 8B W2 artifact as dyads."""
    frame, summary = read_typed_distance_artifact(
        distance_artifact_dir,
        expected_identity={"distance_id": "wasserstein_w2"},
    )
    provider = summary.get("provider_id")
    representation = summary.get("representation_id")
    if provider != "qwen3-embedding-8b" or representation != "qwen3-embedding-8b-unit":
        message = (
            "Paper 1's primary W2 regression requires the "
            "qwen3-embedding-8b unit representation"
        )
        raise DyadicInputError(message)
    return frame.rename(
        columns={"item_i": "ticker_i", "item_j": "ticker_j", "value": "w2_distance"}
    ).loc[lambda values: values["ticker_i"] < values["ticker_j"]]


def load_energy_long(distance_artifact_dir: Path, embeddings_dir: Path) -> pd.DataFrame:
    """Load matched Qwen3-Embedding 8B energy distances and counts."""
    frame = load_energy_functional_pairs(
        distance_artifact_dir,
        provider_id="qwen3-embedding-8b",
        representation_id="qwen3-embedding-8b-unit",
        distance_id="energy_v",
    )
    frame = frame.loc[frame["symbol1"] < frame["symbol2"]].rename(
        columns={
            "symbol1": "ticker_i",
            "symbol2": "ticker_j",
            "energy_distance": "energy_functional",
        }
    )
    tickers = set(frame["ticker_i"]) | set(frame["ticker_j"])
    missing = sorted(
        str(ticker)
        for ticker in tickers
        if not (embeddings_dir / f"{ticker}.parquet").is_file()
    )
    if missing:
        message = f"embeddings_dir={embeddings_dir} is missing tickers: {missing}"
        raise DyadicInputError(message)
    counts = {
        ticker: len(pd.read_parquet(embeddings_dir / f"{ticker}.parquet"))
        for ticker in tickers
    }
    frame["n_samples_i"] = frame["ticker_i"].map(counts)
    frame["n_samples_j"] = frame["ticker_j"].map(counts)
    frame = frame.sort_values(["ticker_i", "ticker_j"], ignore_index=True)
    frame["energy_distance"] = _energy_metric_from_functional(
        frame["energy_functional"].to_numpy(dtype=float)
    )
    return frame


def embedding_dispersions(
    embeddings_dir: Path,
    tickers: list[str],
) -> dict[str, float] | None:
    r"""Compute row-normalized per-firm V-statistic dispersion.

    ``dispersions[i]`` is

    .. math:: B_i = \frac{1}{N_i^2} \sum_n \sum_{n'} d(e_{in}, e_{in'}).

    The ordered-pair sum includes the zero diagonal, matching the upstream
    V-statistic.  ``None`` denotes an absent, incomplete, or degenerate
    embedding panel so the Paper-1 ladder can omit affected specifications.
    """
    if not embeddings_dir.exists():
        return None
    con = duckdb.connect(":memory:")
    dispersions: dict[str, float] = {}
    for ticker in tickers:
        path = embeddings_dir / f"{ticker}.parquet"
        if not path.exists():
            con.close()
            return None
        embedding_frame = con.execute(
            "SELECT embedding FROM read_parquet(?)", [str(path)]
        ).df()
        if embedding_frame.empty:
            con.close()
            return None
        stacked: NDArray[np.float64] = np.stack(
            list(embedding_frame["embedding"])
        ).astype(np.float64, copy=False)
        try:
            rows = l2_normalize_rows(stacked)
        except InvalidNormalizationRowsError:
            con.close()
            return None
        gram: NDArray[np.float64] = rows @ rows.T
        squared_distance = np.maximum(2.0 - 2.0 * gram, 0.0)
        dispersions[ticker] = float(np.sqrt(squared_distance).mean())
    con.close()
    return dispersions


def liquidity_size_proxy(
    market_data_dir: Path,
    tickers: list[str],
) -> dict[str, float]:
    """Compute each firm's mean log dollar-volume proxy."""
    con = duckdb.connect(":memory:")
    proxy: dict[str, float] = {}
    for ticker in tickers:
        path = market_data_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        close_column = quote_duckdb_identifier(f"('Close', '{ticker}')")
        volume_column = quote_duckdb_identifier(f"('Volume', '{ticker}')")
        query = Template(
            """
            SELECT avg(ln($close_column * $volume_column)) AS proxy
            FROM read_parquet(?)
            WHERE $close_column > 0 AND $volume_column > 0
            """
        ).substitute(close_column=close_column, volume_column=volume_column)
        row = con.execute(query, [str(path)]).fetchone()
        if row is not None and row[0] is not None:
            proxy[ticker] = float(row[0])
    con.close()
    return proxy


def crisis_indicator(returns_dir: Path, tickers: list[str]) -> dict[str, int]:
    """Return whether each firm's sample spans the COVID-2020Q1 window."""
    con = duckdb.connect(":memory:")
    flags: dict[str, int] = {}
    for ticker in tickers:
        path = returns_dir / f"{ticker}.parquet"
        if not path.exists():
            flags[ticker] = 0
            continue
        row = con.execute(
            """
            SELECT count(*) FROM read_parquet(?)
            WHERE "Date" BETWEEN ? AND ?
            """,
            [str(path), _CRISIS_START, _CRISIS_END],
        ).fetchone()
        flags[ticker] = 1 if row is not None and row[0] and row[0] > 0 else 0
    con.close()
    return flags


def quote_duckdb_identifier(value: str) -> str:
    """Quote one dynamic DuckDB identifier without changing its spelling."""
    return '"' + value.replace('"', '""') + '"'


def prepare_dyadic_inputs(config: DyadicConfoundConfig) -> PreparedDyadicInputs:
    """Load dyadic inputs and construct controls on their common universe."""
    merged = load_energy_long(
        config.energy_comparator_artifact_dir, config.embeddings_dir
    ).merge(
        load_covariance_long(config.covariance_matrix),
        on=["ticker_i", "ticker_j"],
        how="inner",
    )
    merged = merged.merge(
        load_w2_long(config.distance_artifact_dir),
        on=["ticker_i", "ticker_j"],
        how="inner",
    )
    if merged.empty:
        message = "No overlapping dyads between energy and covariance data"
        raise DyadicInputError(message)
    universe = pd.read_csv(config.universe_csv)
    sectors = dict(zip(universe["Symbol"], universe["Sector"], strict=False))
    tickers = sorted(
        str(ticker) for ticker in set(merged["ticker_i"]) | set(merged["ticker_j"])
    )
    size_proxy = liquidity_size_proxy(config.market_data_dir, tickers)
    crisis_flag = crisis_indicator(config.returns_dir, tickers)
    # Dropping a firm here changes the estimation sample, so say so. A firm
    # lacking a proxy usually has no positive Volume in its market-data file --
    # e.g. an archive-sourced series whose vendor export carried no volume
    # column -- and the run would otherwise succeed having quietly analysed a
    # smaller universe than the roster declares.
    without_proxy = sorted(set(tickers) - set(size_proxy))
    if without_proxy:
        logger.warning(
            "dropping %d firm(s) with no dollar-volume proxy from the dyadic "
            "frame: %s. Check that their market data carries positive Volume.",
            len(without_proxy),
            ", ".join(without_proxy),
        )
    merged = merged[
        merged["ticker_i"].isin(size_proxy) & merged["ticker_j"].isin(size_proxy)
    ].reset_index(drop=True)
    tickers = sorted(
        str(ticker) for ticker in set(merged["ticker_i"]) | set(merged["ticker_j"])
    )
    firm_i: NDArray[np.str_] = merged["ticker_i"].to_numpy(dtype=str)
    firm_j: NDArray[np.str_] = merged["ticker_j"].to_numpy(dtype=str)
    log_count_i = np.log(merged["n_samples_i"].to_numpy(dtype=float))
    log_count_j = np.log(merged["n_samples_j"].to_numpy(dtype=float))
    columns: dict[str, np.ndarray] = {
        "energy_distance": merged["energy_distance"].to_numpy(),
        "w2_distance": merged.get(
            "w2_distance", pd.Series(np.nan, index=merged.index)
        ).to_numpy(dtype=float),
        "same_sector": (
            merged["ticker_i"].map(sectors) == merged["ticker_j"].map(sectors)
        ).to_numpy(dtype=float),
        "size_gap": (
            merged["ticker_i"].map(size_proxy) - merged["ticker_j"].map(size_proxy)
        )
        .abs()
        .to_numpy(),
        "crisis": (
            (merged["ticker_i"].map(crisis_flag) > 0)
            & (merged["ticker_j"].map(crisis_flag) > 0)
        ).to_numpy(dtype=float),
        "abs_log_n_diff": np.abs(log_count_i - log_count_j),
        "min_log_n": np.minimum(log_count_i, log_count_j),
    }
    correlation: NDArray[np.float64] = np.clip(
        merged["correlation"].to_numpy(dtype=np.float64),
        -1.0,
        1.0,
    )
    target = np.sqrt(2 * (1 - correlation))
    # The random-exposure diagnostic works in standardized total-return geometry
    # (v_i = v_j = 1), so the raw variances are deliberately not carried: widening
    # this projection also widens the OOS covariance panel `dyadic_oos` reads.
    columns["return_correlation"] = correlation
    columns["return_chord_squared"] = target**2
    dispersions = embedding_dispersions(config.embeddings_dir, tickers)
    sample = DyadicSample[str](target, firm_i, firm_j)
    if dispersions is None:
        note = (
            f"embeddings_dir={config.embeddings_dir} missing or incomplete for the "
            "dyad universe; within-firm dispersion specification omitted"
        )
        logger.warning(note)
        return PreparedDyadicInputs(tickers, columns, sample, None, note, None)
    columns["disp_sum"] = np.array(
        [dispersions[i] + dispersions[j] for i, j in zip(firm_i, firm_j, strict=True)]
    )
    columns["disp_gap"] = np.array(
        [
            abs(dispersions[i] - dispersions[j])
            for i, j in zip(firm_i, firm_j, strict=True)
        ]
    )
    counts: dict[str, int] = {}
    for row in merged.itertuples(index=False):
        counts[str(row.ticker_i)] = int(float(str(row.n_samples_i)))
        counts[str(row.ticker_j)] = int(float(str(row.n_samples_j)))
    correction = np.array(
        [
            dispersions[i] / max(counts[i] - 1, 1)
            + dispersions[j] / max(counts[j] - 1, 1)
            for i, j in zip(firm_i, firm_j, strict=True)
        ]
    )
    u_functional: NDArray[np.float64] = (
        merged["energy_functional"].to_numpy(dtype=np.float64) - correction
    )
    u_metric = _energy_metric_from_functional(u_functional)
    sensitivity: dict[str, object] = {
        "estimator": "U-statistic functional, projected to sqrt metric",
        "pearson_return_distance": float(pearsonr(u_metric, target).statistic),
        "spearman_return_distance": float(spearmanr(u_metric, target).statistic),
        "minimum_u_functional": float(u_functional.min()),
        "mean_v_to_u_correction": float(correction.mean()),
        "mean_correction_share_of_v_functional": float(
            correction.mean() / merged["energy_functional"].mean()
        ),
    }
    return PreparedDyadicInputs(
        tickers, columns, sample, dispersions, None, sensitivity
    )

"""Tests for nullable return panels and covariance artifact contracts."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline.stages.substrate.market import (
    CANONICAL_RETURN_PANEL_END_DATE,
    CANONICAL_RETURN_PANEL_OBSERVATIONS,
    CANONICAL_RETURN_PANEL_START_DATE,
    CANONICAL_RETURN_PANEL_TICKERS,
    ReturnPanelContract,
    _complete_case_return_panel,
    compute_covariance,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_nullable_mask_survives_stack_instead_of_becoming_zero() -> None:
    """Nullable return masks survive stacking and drive complete-case filtering."""
    dates = np.asarray(
        ["2018-01-02", "2018-01-03", "2018-01-04"], dtype="datetime64[D]"
    )
    complete = np.asarray([0.1, 0.2, 0.3])
    incomplete = np.ma.array(
        [0.0, 0.4, 0.5], mask=[True, False, False], fill_value=1e20
    )

    legacy = np.column_stack([complete, incomplete])
    assert np.ma.getmaskarray(legacy).sum() == 0
    assert np.equal(legacy[0, 1], 0.0)

    clean_dates, clean_data = _complete_case_return_panel(
        {"Date": dates, "AAA": complete, "ZZZ": incomplete}, ["AAA", "ZZZ"]
    )

    assert clean_dates.tolist() == dates[1:].tolist()
    assert clean_data.shape == (2, 2)
    assert np.array_equal(clean_data, np.asarray([[0.2, 0.4], [0.3, 0.5]]))
    assert np.isfinite(clean_data).all()


def _write_returns(
    path: Path, ticker: str, dates: list[str], values: list[float | None]
) -> None:
    table = pa.table(
        {
            "Date": pa.array([date.fromisoformat(value) for value in dates]),
            "ticker": [ticker] * len(dates),
            "return": values,
        }
    )
    pq.write_table(table, path)


def test_covariance_fixture_enforces_dates_shape_finiteness_and_order(
    tmp_path: Path,
) -> None:
    """Covariance output preserves panel ordering, dates, and finite values."""
    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    dates = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"]
    _write_returns(returns_dir / "ZZZ.parquet", "ZZZ", dates, [None, 0.4, 0.7, 0.8])
    _write_returns(returns_dir / "AAA.parquet", "AAA", dates, [0.1, 0.2, 0.5, 0.9])
    _write_returns(returns_dir / "MMM.parquet", "MMM", dates, [0.2, 0.6, 0.4, 0.3])
    output = tmp_path / "covariance.parquet"

    compute_covariance(
        returns_dir,
        output,
        ReturnPanelContract(
            min_observations=3,
            expected_tickers=3,
            expected_observations=3,
            expected_start_date="2018-01-03",
            expected_end_date="2018-01-05",
        ),
    )

    table = pq.read_table(output)
    frame = table.to_pandas()
    assert frame[["ticker_i", "ticker_j"]].values.tolist() == [
        ["AAA", "AAA"],
        ["AAA", "MMM"],
        ["AAA", "ZZZ"],
        ["MMM", "AAA"],
        ["MMM", "MMM"],
        ["MMM", "ZZZ"],
        ["ZZZ", "AAA"],
        ["ZZZ", "MMM"],
        ["ZZZ", "ZZZ"],
    ]
    expected_panel = np.asarray(
        [
            [0.2, 0.6, 0.4],
            [0.5, 0.4, 0.7],
            [0.9, 0.3, 0.8],
        ]
    )
    expected_covariance = np.cov(expected_panel, rowvar=False)
    expected_correlation = np.corrcoef(expected_panel, rowvar=False)
    observed_covariance = frame["covariance"].to_numpy().reshape(3, 3)
    observed_correlation = frame["correlation"].to_numpy().reshape(3, 3)
    assert np.allclose(observed_covariance, expected_covariance, atol=1e-15)
    assert np.allclose(observed_correlation, expected_correlation, atol=1e-15)
    assert np.isfinite(observed_covariance).all()
    assert np.isfinite(observed_correlation).all()
    metadata = table.schema.metadata
    assert metadata is not None
    assert metadata[b"return_panel_rows"] == b"3"
    assert metadata[b"return_panel_columns"] == b"3"
    assert metadata[b"return_panel_start_date"] == b"2018-01-03"
    assert metadata[b"return_panel_end_date"] == b"2018-01-05"
    assert metadata[b"return_panel_tickers"] == b"AAA,MMM,ZZZ"
    assert b"no imputation" in metadata[b"return_panel_policy"]


def test_canonical_panel_contract_is_exact() -> None:
    """Canonical return-panel constants remain pinned to governed values."""
    actual = (
        CANONICAL_RETURN_PANEL_TICKERS,
        CANONICAL_RETURN_PANEL_OBSERVATIONS,
        CANONICAL_RETURN_PANEL_START_DATE,
        CANONICAL_RETURN_PANEL_END_DATE,
    )
    expected = (100, 1207, "2018-03-19", "2022-12-30")
    assert actual == expected

"""Tests for Paper 3's portfolio-anatomy decomposition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline._kernels.paper3.validation import solve_certificate
from pipeline.figures.paper3.numbers import Paper3NumberInputs, _write_paper3_numbers
from pipeline.figures.paper3.portfolio_anatomy import (
    _render_holdings_table,
    _ticker_label_positions,
)
from pipeline.stages.papers.paper3.portfolio_anatomy import (
    PortfolioAnatomyError,
    build_portfolio_anatomy,
)

if TYPE_CHECKING:
    from pathlib import Path

_FINAL_YEAR = 2022


def test_ticker_labels_use_alternating_positions_without_dropping_data() -> None:
    """Keep the 100-firm axes legible while retaining all plotted values."""
    positions = _ticker_label_positions(100)
    np.testing.assert_array_equal(positions, np.arange(0, 100, 2))
    assert positions.tolist() == list(range(0, 100, 2))


def _squared_distances(points: np.ndarray) -> np.ndarray:
    return np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)


def _inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metadata = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "name": ["Alpha", "Beta", "Gamma", "Delta"],
            "sector": ["One", "One", "Two", "Two"],
        }
    )
    final = _squared_distances(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.2, 0.0],
                [0.0, 0.0, 1.5],
            ]
        )
    )
    earlier = _squared_distances(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.0, 1.1],
            ]
        )
    )
    weights = solve_certificate(final, cap=1.0).weights
    return (
        metadata,
        weights,
        final,
        {
            "2018-12-31": 1.2 * earlier,
            "2019-12-31": 1.1 * earlier,
            "2020-12-31": earlier,
            "2021-12-31": earlier,
            "2022-12-31": final,
        },
    )


def test_anatomy_decompositions_close_exactly() -> None:
    """Firm, pair, and sector credits reproduce one governed objective."""
    metadata, weights, final, vintages = _inputs()
    artifacts = build_portfolio_anatomy(metadata, weights, final, vintages)
    certificate = artifacts.summary["certificate"]
    assert isinstance(certificate, dict)
    credit = float(certificate["certificate_credit"])
    assert artifacts.firms["certificate_credit"].sum() == pytest.approx(credit)
    assert artifacts.pairs["certificate_credit"].sum() == pytest.approx(credit)
    assert artifacts.sectors["certificate_credit_attribution"].sum() == pytest.approx(
        credit
    )
    assert float(certificate["within_sector_share"]) + float(
        certificate["cross_sector_share"]
    ) == pytest.approx(1.0)


def test_firm_certificate_share_is_the_kkt_weight_identity() -> None:
    """The proposed firm-share scatter is redundant at the optimum."""
    metadata, weights, final, vintages = _inputs()
    artifacts = build_portfolio_anatomy(metadata, weights, final, vintages)
    np.testing.assert_allclose(
        artifacts.firms["certificate_share"],
        artifacts.firms["weight"],
        atol=5e-6,
        rtol=0.0,
    )
    assert artifacts.pairs.iloc[0]["certificate_credit"] > 0.0


def test_vintage_outputs_use_one_way_turnover_and_reproduce_final_weights() -> None:
    """The expanding-cutoff leaf reports stable, auditable decision changes."""
    metadata, weights, final, vintages = _inputs()
    artifacts = build_portfolio_anatomy(metadata, weights, final, vintages)
    summary = artifacts.vintage_summary
    assert pd.isna(summary.iloc[0]["one_way_turnover"])
    first = artifacts.vintage_weights.query("year == 2021")["weight"].to_numpy()
    second = artifacts.vintage_weights.loc[
        artifacts.vintage_weights["year"] == _FINAL_YEAR, "weight"
    ].to_numpy()
    terminal = summary.loc[summary["year"] == _FINAL_YEAR].iloc[0]
    assert float(terminal["one_way_turnover"]) == pytest.approx(
        0.5 * np.abs(second - first).sum()
    )
    np.testing.assert_allclose(second, weights, atol=1e-8, rtol=0.0)


def test_final_vintage_must_match_the_canonical_allocation() -> None:
    """The heatmap cannot silently end at a different full-sample decision."""
    metadata, weights, final, vintages = _inputs()
    vintages["2022-12-31"] = vintages["2021-12-31"]
    with pytest.raises(PortfolioAnatomyError, match="does not reproduce"):
        build_portfolio_anatomy(metadata, weights, final, vintages)


def test_anatomy_numbers_and_holdings_table_are_generated(tmp_path: Path) -> None:
    """Publication prose consumes governed macros and a generated table."""
    metadata, weights, final, vintages = _inputs()
    artifacts = build_portfolio_anatomy(metadata, weights, final, vintages)
    numbers_path = tmp_path / "numbers.tex"
    _write_paper3_numbers(
        numbers_path,
        Paper3NumberInputs(
            summary={},
            results=pd.DataFrame(),
            anatomy_summary=artifacts.summary,
        ),
    )
    numbers = numbers_path.read_text(encoding="utf-8")
    portfolio = artifacts.summary["portfolio"]
    assert isinstance(portfolio, dict)
    expected_active = portfolio["active_firms"]
    assert f"\\ppDeclareValue{{anatomy-active-firms}}{{{expected_active}}}" in numbers
    assert "anatomy-cross-sector-credit-pct" in numbers

    table_path = tmp_path / "holdings.tex"
    _render_holdings_table(table_path, artifacts.firms)
    table = table_path.read_text(encoding="utf-8")
    assert r"\label{tab:p3-top-holdings}" in table
    assert "Alpha" in table
    assert all(line == line.rstrip() for line in table.splitlines())

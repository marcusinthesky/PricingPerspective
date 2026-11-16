"""Tests for Paper 3's information-certificate kernel."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pipeline._kernels.paper3.certificate import (
    certificate_credit,
    certificate_surface,
    equal_weights,
    exposure_floor,
    inverse_volatility_weights,
)
from pipeline.stages.papers.paper3.certificate import (
    _news_only_benchmark,
    _reference_draws,
    _reference_percentiles,
    _ReferenceDesign,
    _standardize_returns,
)


def _distance() -> np.ndarray:
    return np.array(
        [[0.0, 1.0, 0.8], [1.0, 0.0, 0.6], [0.8, 0.6, 0.0]],
        dtype=np.float64,
    )


def test_certificate_is_monotone_in_scale_and_slack() -> None:
    """More distortion or slack cannot increase the certificate."""
    portfolios = {"equal_weight": equal_weights(3)}
    cells = certificate_surface(_distance(), portfolios, (1.0, 1.5), (0.0, 0.1))
    by_scenario = {
        (cell.lipschitz_inverse_scale, cell.common_slack): cell.certificate_credit
        for cell in cells
    }
    assert by_scenario[1.0, 0.0] >= by_scenario[1.5, 0.0]
    assert by_scenario[1.0, 0.0] >= by_scenario[1.0, 0.1]
    assert by_scenario[1.5, 0.0] >= by_scenario[1.5, 0.1]


def test_zero_weight_asset_is_ignored() -> None:
    """An asset outside the portfolio contributes no pairwise credit."""
    floor = exposure_floor(_distance(), 1.0, 0.0)
    assert certificate_credit(floor, np.array([0.5, 0.5, 0.0])) == pytest.approx(0.25)


def test_inverse_volatility_weights_are_normalized() -> None:
    """Inverse-volatility construction returns positive simplex weights."""
    returns = np.array([[0.01, 0.02], [-0.01, 0.04], [0.02, -0.02]], dtype=np.float64)
    weights = inverse_volatility_weights(returns)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)


@pytest.mark.parametrize(
    ("scale", "slack"),
    [(0.0, 0.0), (-1.0, 0.0), (1.0, -0.01)],
)
def test_invalid_transmission_scenarios_fail(scale: float, slack: float) -> None:
    """Invalid scale or slack declarations fail before computation."""
    with pytest.raises(ValueError, match="finite"):
        exposure_floor(_distance(), scale, slack)


def test_malformed_distance_matrix_fails_closed() -> None:
    """A non-symmetric artifact is rejected rather than silently repaired."""
    malformed = _distance()
    malformed[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        exposure_floor(malformed, 1.0, 0.0)


def test_zero_volatility_fails_closed() -> None:
    """Inverse-volatility weights reject undefined reciprocal risks."""
    with pytest.raises(ValueError, match="positive volatility"):
        inverse_volatility_weights(np.ones((3, 2)))


@pytest.mark.parametrize(
    "returns",
    [
        np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),
        np.array([[1.0, 2.0], [np.nan, 3.0], [2.0, 4.0]]),
    ],
)
def test_news_standardization_rejects_invalid_inputs(returns: np.ndarray) -> None:
    """Missing and zero-volatility standardization inputs fail closed."""
    with pytest.raises(ValueError, match=r"finite|positive"):
        _standardize_returns(returns)


def test_news_standardization_and_benchmark_are_deterministic() -> None:
    """The descriptive news allocation has a valid full-sample contract."""
    returns = np.array(
        [
            [0.01, 0.02, -0.01],
            [-0.02, 0.01, 0.03],
            [0.03, -0.01, 0.02],
            [0.01, 0.04, -0.02],
            [-0.01, 0.00, 0.01],
            [0.02, -0.02, 0.04],
        ]
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    squared_w2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    tickers = ["A", "B", "C"]
    designs = (
        _ReferenceDesign("uniform", 1.0, 0.60),
        _ReferenceDesign("concentrated", 0.5, 0.80),
    )
    weights, summary = _news_only_benchmark(
        tickers, returns, squared_w2, designs, 2000, 42
    )
    assert weights["ticker"].tolist() == tickers
    assert np.isfinite(weights["weight"]).all()
    assert (weights["weight"] >= 0.0).all()
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert summary["status"] == "full_sample_descriptive"
    objective = cast("dict[str, object]", summary["objective"])
    certificate_credit = cast("float", objective["certificate_credit"])
    equal_credit = cast("float", objective["equal_weight_certificate_credit"])
    assert certificate_credit + 1e-12 >= equal_credit

    reference = cast("dict[str, object]", summary["reference_population"])
    rows = cast("list[dict[str, object]]", reference["designs"])
    assert [row["label"] for row in rows] == ["uniform", "concentrated"]
    for row, design in zip(rows, designs, strict=True):
        draws = cast("np.ndarray", row["retained_draws"])
        assert int(draws) > 0
        percentile = cast("dict[str, float]", row["percentile"])
        # A percentile is a share, so it must stay inside the unit interval for
        # every candidate under every declared law.
        assert all(0.0 <= float(value) <= 1.0 for value in percentile.values())
        effective_n = cast("float", row["mean_effective_number_assets"])
        assert 1.0 <= float(effective_n) <= len(tickers)
        assert float(design.maximum_weight) * len(tickers) >= 1.0


def test_reference_draws_respect_their_declared_cap() -> None:
    """Renormalizing after a cap can breach it, so the draws must be re-checked."""
    n_assets = 52
    design = _ReferenceDesign("tight", 1.0, 0.05)
    sample = _reference_draws(design, n_assets, 4000, 42)
    assert sample.shape[1] == n_assets
    assert sample.max() <= design.maximum_weight + 1e-8
    assert np.allclose(sample.sum(axis=1), 1.0)


def test_reference_design_rejects_an_infeasible_cap() -> None:
    """A cap below 1/n cannot fill the simplex and must fail loudly."""
    with pytest.raises(ValueError, match="cannot fill the simplex"):
        _reference_draws(_ReferenceDesign("impossible", 1.0, 0.01), 52, 100, 42)


def test_reference_percentiles_reject_candidate_outside_design_cap() -> None:
    """A candidate cannot be described as inside a law whose cap it violates."""
    covariance = np.eye(3)
    candidates = {"outside": np.array([0.7, 0.2, 0.1])}
    designs = (_ReferenceDesign("capped", 1.0, 0.6),)
    with pytest.raises(ValueError, match="candidate outside exceeds"):
        _reference_percentiles(covariance, candidates, designs, 2000, 42)


def test_reference_percentiles_annotate_candidate_outside_design_cap() -> None:
    """Cross-representation tables retain an infeasible cell without ranking it."""
    covariance = np.eye(3)
    candidates = {
        "outside": np.array([0.7, 0.2, 0.1]),
        "inside": np.array([0.5, 0.3, 0.2]),
    }
    designs = (_ReferenceDesign("capped", 1.0, 0.6),)
    rows = _reference_percentiles(
        covariance,
        candidates,
        designs,
        2000,
        42,
        cap_policy="annotate",
    )
    percentile = cast("dict[str, float | None]", rows[0]["percentile"])
    assert rows[0]["percentile"] == {
        "outside": None,
        "inside": percentile["inside"],
    }
    assert rows[0]["candidate_status"] == {
        "outside": "infeasible_cap",
        "inside": "computed",
    }

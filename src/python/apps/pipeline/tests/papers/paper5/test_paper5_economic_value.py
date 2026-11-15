"""Tests for Paper 5's factor-conditioned economic-value diagnostics."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pipeline.stages.papers.paper5.economic_value import (
    _FIELD_MODELS,
    Paper5EconomicValueConfig,
    _bootstrap_intervals,
    _centroid_weights,
    _Inputs,
    _ledoit_wolf_covariance,
    _ledoit_wolf_daily_score,
    _paired_score_contrast,
    _peer_rows,
)
from pipeline.stages.papers.paper5.estimate import sar_factor_fit_at_rho
from pipeline.stages.papers.paper5.field_operators import rbf_weights as _rbf_weights
from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel
from statadic.estimand.core import Covariance, Estimand
from statadic.estimation.covariance import LedoitWolf


def test_rbf_field_is_row_stochastic_and_zero_diagonal() -> None:
    """Distance weighting retains the interaction-matrix contract."""
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]],
        dtype=np.float64,
    )
    weights = _rbf_weights(distances, 1.0)
    assert_allclose(np.diag(weights), 0.0)
    assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)


def test_centroid_reconstruction_is_leave_one_out() -> None:
    """The Euclidean centroid comparator never assigns self-weight."""
    centroids = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    weights = _centroid_weights(centroids)
    assert_allclose(np.diag(weights), 0.0, atol=1e-12)
    assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(weights >= 0.0)
    assert_allclose(weights[0, 1], 0.5)
    assert_allclose(weights[0, 2], 0.5)


def test_peer_panel_is_deterministic_and_reports_governed_flags() -> None:
    """Peer rows retain ranks, classifications, and co-mention indicators."""
    tickers = ["AMD", "AAA", "BBB"]
    weights = np.array(
        [[0.0, 0.8, 0.2], [0.4, 0.0, 0.6], [0.7, 0.3, 0.0]],
        dtype=np.float64,
    )
    squared_distances = np.array(
        [[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    co_mentions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    rows, summary = _peer_rows(
        tickers,
        weights,
        squared_distances,
        co_mentions,
        dict.fromkeys(tickers, "industry"),
        dict.fromkeys(tickers, "sector"),
        1e-8,
    )
    assert isinstance(rows, pd.DataFrame)
    assert set(rows.columns) >= {
        "target",
        "candidate",
        "weight",
        "distance_rank",
        "same_sector",
        "same_industry",
        "co_mention",
        "effective_source_count",
        "incoming_mass",
    }
    assert summary["descriptive_only"] is True
    amd = rows.loc[rows["target"] == "AMD"].sort_values("weight", ascending=False)
    assert amd.iloc[0]["candidate"] == "AAA"
    assert bool(amd.iloc[0]["co_mention"])


def _synthetic_panel(
    n_dates: int, n_assets: int, n_factors: int, seed: int
) -> _AlignedFactorPanel:
    """Build a factor-complete panel for the refit bootstrap tests."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(scale=0.01, size=(n_dates, n_factors))
    coefficients = rng.normal(scale=0.3, size=(1 + n_factors, n_assets))
    excess = np.column_stack((np.ones(n_dates), factors)) @ coefficients + rng.normal(
        scale=0.01, size=(n_dates, n_assets)
    )
    return _AlignedFactorPanel(
        dates=np.datetime64("2018-01-01") + np.arange(n_dates, dtype="timedelta64[D]"),
        raw_returns=excess,
        excess_returns=excess,
        factors=factors,
        dropped_return_dates=0,
    )


def _uniform_operator(n_assets: int) -> np.ndarray:
    """Return the row-stochastic zero-diagonal equal-weight field."""
    matrix = np.full((n_assets, n_assets), 1.0 / (n_assets - 1), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_bootstrap_intervals_refit_the_calibration_fit_on_every_draw() -> None:
    """Comparator intervals carry calibration uncertainty, not score draws alone."""
    n_assets = 4
    calibration = _synthetic_panel(120, n_assets, 2, seed=20260920)
    evaluation = _synthetic_panel(60, n_assets, 2, seed=20260921)
    distances = np.abs(np.subtract.outer(np.arange(n_assets), np.arange(n_assets)))
    squared = distances.astype(np.float64) ** 2
    inputs = _Inputs(
        tickers=[f"T{index}" for index in range(n_assets)],
        matrices={"w1": _uniform_operator(n_assets), "w2": _uniform_operator(n_assets)},
        squared_distances=squared,
        calibration=calibration,
        evaluation=evaluation,
        co_mentions=_uniform_operator(n_assets),
        industry=dict.fromkeys((f"T{index}" for index in range(n_assets)), "A"),
        sector=dict.fromkeys((f"T{index}" for index in range(n_assets)), "A"),
    )
    matrices = {model: _uniform_operator(n_assets) for model in _FIELD_MODELS}
    eigenvalues = {
        model: np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128)
        for model, matrix in matrices.items()
    }
    config = Paper5EconomicValueConfig(n_bootstrap=5, block_length=3, seed=4)
    draws, intervals, provenance = _bootstrap_intervals(
        inputs, matrices, eigenvalues, config
    )
    assert len(draws) == 5 * 9
    assert provenance["refit_parameters"] is True
    assert provenance["architecture"] == "paired_refit_stationary_block"
    assert provenance["seed_offsets"] == {
        "calibration": 0,
        "evaluation_fit": 1,
        "evaluation_score": 2,
    }
    for model in ("w2", "ledoit_wolf"):
        gains = draws.loc[draws["model"] == model, "score_gain"].to_numpy()
        assert np.all(np.isfinite(gains))
        assert float(np.std(gains)) > 0.0
        assert intervals[model][0] <= intervals[model][1]


def test_w2_centroid_contrast_uses_common_bootstrap_draws() -> None:
    """The W2-centroid interval compares paired held-out score draws."""
    draws = pd.DataFrame(
        {
            "draw_id": [0, 0, 1, 1, 2, 2],
            "model": ["w2", "centroid"] * 3,
            "score_gain": [0.20, 0.15, 0.10, 0.12, 0.30, 0.25],
        }
    )
    contrast = _paired_score_contrast(
        draws,
        candidate="w2",
        reference="centroid",
        point_estimate=0.05,
        confidence_level=0.95,
    )
    assert contrast["candidate"] == "w2"
    assert contrast["reference"] == "centroid"
    assert_allclose(contrast["estimate"], 0.05)
    assert_allclose(contrast["bootstrap_mean"], 0.0266666667)
    expected_draws = 3
    assert contrast["bootstrap_valid_draws"] == expected_draws
    assert_allclose(contrast["positive_share"], 2 / expected_draws)


def test_direct_ledoit_wolf_covariance_is_positive_definite_and_scores() -> None:
    """The generic residual benchmark is finite and fixed-fit scoreable."""
    rng = np.random.default_rng(20260920)
    n_dates, n_assets, n_factors = 80, 5, 2
    factors = rng.normal(scale=0.01, size=(n_dates, n_factors))
    coefficients = rng.normal(scale=0.3, size=(1 + n_factors, n_assets))
    returns = np.column_stack((np.ones(n_dates), factors)) @ coefficients + rng.normal(
        scale=0.01, size=(n_dates, n_assets)
    )
    panel = _AlignedFactorPanel(
        dates=np.datetime64("2020-01-01") + np.arange(n_dates, dtype="timedelta64[D]"),
        raw_returns=returns,
        excess_returns=returns,
        factors=factors,
        dropped_return_dates=0,
    )
    weights = np.zeros((n_assets, n_assets), dtype=np.float64)
    for row in range(n_assets):
        peers = np.delete(np.arange(n_assets), row)
        weights[row, peers] = 1.0 / len(peers)
    fit = sar_factor_fit_at_rho(
        returns,
        weights,
        factors,
        0.0,
        eigvals_w=np.asarray(np.linalg.eigvals(weights), dtype=np.complex128),
    )

    covariance = _ledoit_wolf_covariance(panel, fit)
    scores = _ledoit_wolf_daily_score(fit, panel, covariance)

    assert covariance.shape == (n_assets, n_assets)
    assert np.all(np.isfinite(covariance))
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert scores.shape == (n_dates,)
    assert np.all(np.isfinite(scores))


def test_ledoit_wolf_benchmark_equals_the_statadic_owner_on_factor_residuals() -> None:
    """The stage covariance is the owner's estimate of the factor residuals."""
    rng = np.random.default_rng(20260921)
    n_dates, n_assets, n_factors = 70, 4, 2
    factors = rng.normal(scale=0.01, size=(n_dates, n_factors))
    coefficients = rng.normal(scale=0.3, size=(1 + n_factors, n_assets))
    returns = np.column_stack((np.ones(n_dates), factors)) @ coefficients + rng.normal(
        scale=0.01, size=(n_dates, n_assets)
    )
    panel = _AlignedFactorPanel(
        dates=np.datetime64("2020-01-01") + np.arange(n_dates, dtype="timedelta64[D]"),
        raw_returns=returns,
        excess_returns=returns,
        factors=factors,
        dropped_return_dates=0,
    )
    weights = (np.ones((n_assets, n_assets)) - np.eye(n_assets)) / (n_assets - 1)
    fit = sar_factor_fit_at_rho(
        returns,
        weights,
        factors,
        0.0,
        eigvals_w=np.asarray(np.linalg.eigvals(weights), dtype=np.complex128),
    )
    residuals = returns - np.column_stack((np.ones(n_dates), factors)) @ (
        fit.coefficients
    )
    with jax.enable_x64(new_val=True):
        native = np.asarray(
            Estimand(Covariance(), estimator=LedoitWolf()).estimate(residuals).sigma,
            dtype=np.float64,
        )
    assert_allclose(
        _ledoit_wolf_covariance(panel, fit), 0.5 * (native + native.T), rtol=1e-12
    )

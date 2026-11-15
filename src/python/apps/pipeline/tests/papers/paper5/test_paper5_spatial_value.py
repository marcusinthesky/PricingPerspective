"""Tests for Paper 5 factor-only versus spatial-value comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pipeline.stages.papers.paper5.spatial_value import (
    Paper5SpatialValueConfig,
    _fit_models,
    _information_criteria,
    _paired_bootstrap,
    _qlr_bootstrap,
)
from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pipeline.stages.papers.paper5.estimate import FactorSarFit
    from pipeline.stages.papers.paper5.spatial_value import VarianceModel


def _weights(rng: np.random.Generator, n_assets: int) -> NDArray[np.float64]:
    weights = rng.uniform(0.1, 1.0, size=(n_assets, n_assets))
    np.fill_diagonal(weights, 0.0)
    return cast("NDArray[np.float64]", weights / weights.sum(axis=1, keepdims=True))


def _panel(
    *, seed: int, dates: int, weights: NDArray[np.float64], rho: float
) -> _AlignedFactorPanel:
    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    factors = rng.normal(scale=0.01, size=(dates, 2))
    loadings = rng.normal(size=(2, n_assets))
    scales = np.linspace(0.005, 0.012, n_assets)
    filtered = factors @ loadings + rng.normal(size=(dates, n_assets)) * scales
    returns = np.linalg.solve(
        np.eye(n_assets, dtype=np.float64) - rho * weights, filtered.T
    ).T
    return _AlignedFactorPanel(
        dates=np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")[:dates],
        raw_returns=cast("NDArray[np.float64]", returns),
        excess_returns=cast("NDArray[np.float64]", returns),
        factors=cast("NDArray[np.float64]", factors),
        dropped_return_dates=0,
    )


def test_information_criteria_disclose_both_panel_conventions() -> None:
    """Working AIC, HQC, and both BIC conventions use declared sample sizes."""
    result = _information_criteria(-100.0, 10, 50, 4)

    np.testing.assert_allclose(result["aic"], 220.0)
    np.testing.assert_allclose(
        result["hqc_dates"], 200.0 + 2.0 * 10.0 * np.log(np.log(50))
    )
    np.testing.assert_allclose(
        result["hqc_firm_days"], 200.0 + 2.0 * 10.0 * np.log(np.log(200))
    )
    np.testing.assert_allclose(result["bic_dates"], 200.0 + np.log(50) * 10)
    np.testing.assert_allclose(result["bic_firm_days"], 200.0 + np.log(200) * 10)


def test_hannan_quinn_requires_a_valid_log_log_sample_size() -> None:
    """HQC rejects sample sizes where the nested logarithm is undefined."""
    with pytest.raises(ValueError, match="HQC requires"):
        _information_criteria(-100.0, 10, 1, 4)


def test_paired_and_restricted_null_bootstraps_share_model_contract() -> None:
    """Paired score and restricted-null draws cover every governed model."""
    rng = np.random.default_rng(601)
    n_assets = 4
    w2 = _weights(rng, n_assets)
    w1 = 0.97 * w2 + 0.03 * np.roll(w2, shift=1, axis=1)
    np.fill_diagonal(w1, 0.0)
    w1 /= w1.sum(axis=1, keepdims=True)
    matrices = {"w1": cast("NDArray[np.float64]", w1), "w2": w2}
    eigenvalues = {
        order: cast("NDArray[np.complex128]", np.linalg.eigvals(weights))
        for order, weights in matrices.items()
    }
    calibration = _panel(seed=602, dates=100, weights=w2, rho=0.30)
    evaluation = _panel(seed=603, dates=80, weights=w2, rho=0.30)
    config = Paper5SpatialValueConfig(n_bootstrap=3, block_length=7, seed=9)

    paired = _paired_bootstrap(calibration, evaluation, matrices, eigenvalues, config)
    observed: dict[VarianceModel, dict[str, FactorSarFit]] = {
        variance: _fit_models(evaluation, matrices, eigenvalues, variance)
        for variance in ("spherical", "diagonal")
    }
    qlr_draws, qlr = _qlr_bootstrap(evaluation, matrices, eigenvalues, config, observed)

    assert len(paired) == config.n_bootstrap
    assert paired["all_solvers_converged"].all()
    np.testing.assert_allclose(
        paired["score_delta_spherical_w2"],
        paired["score_spherical_w2"] - paired["score_spherical_none"],
    )
    assert len(qlr_draws) == config.n_bootstrap * 4
    assert qlr_draws["all_solvers_converged"].all()
    assert set(qlr) == {
        ("spherical", "w1"),
        ("spherical", "w2"),
        ("diagonal", "w1"),
        ("diagonal", "w2"),
    }
    assert all(
        0.0 < cast("float", result["bootstrap_p_value"]) <= 1.0
        for result in qlr.values()
    )

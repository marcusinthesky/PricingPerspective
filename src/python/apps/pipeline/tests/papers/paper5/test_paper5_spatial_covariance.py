"""Tests for the exploratory Paper 5 spatial covariance stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pytest
import yaml
from pipeline.stages.papers.paper5.estimate import (
    sar_factor_fit_at_rho,
    sar_factor_qmle,
)
from pipeline.stages.papers.paper5.spatial_covariance import (
    Paper5SpatialCovarianceConfig,
    _spatialize_covariance,
    spatial_ledoit_wolf_covariance,
)
from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel
from statadic.estimand.core import Covariance, Estimand
from statadic.estimation.covariance import LedoitWolf

if TYPE_CHECKING:
    from numpy.typing import NDArray


_PSD_TOLERANCE = 1e-12


def _panel(
    *, seed: int, dates: int, weights: NDArray[np.float64], rho: float
) -> _AlignedFactorPanel:
    rng = np.random.default_rng(seed)
    n_assets = weights.shape[0]
    factors = rng.normal(scale=0.01, size=(dates, 2))
    coefficients = rng.normal(scale=0.4, size=(3, n_assets))
    innovations = rng.normal(
        scale=np.linspace(0.004, 0.01, n_assets), size=(dates, n_assets)
    )
    design = np.column_stack((np.ones(dates), factors))
    filtered = design @ coefficients + innovations
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


def _weights(n_assets: int) -> NDArray[np.float64]:
    weights = np.zeros((n_assets, n_assets), dtype=np.float64)
    for row in range(n_assets):
        candidates = [column for column in range(n_assets) if column != row]
        weights[row, candidates] = np.linspace(0.2, 1.0, n_assets - 1)
        weights[row] /= weights[row].sum()
    return cast("NDArray[np.float64]", weights)


def test_ledoit_wolf_covariance_is_spatialized_and_psd() -> None:
    """Both covariance choices preserve the fitted filter and PSD structure."""
    weights = _weights(5)
    panel = _panel(seed=710, dates=160, weights=weights, rho=0.35)
    fit = sar_factor_qmle(
        panel.excess_returns,
        weights,
        panel.factors,
        eigvals_w=cast("NDArray[np.complex128]", np.linalg.eigvals(weights)),
    )

    result = spatial_ledoit_wolf_covariance(fit, panel, weights)
    system = np.eye(5) - fit.rho_hat * weights
    expected = np.linalg.solve(system, result.innovation_qmle)
    expected = np.linalg.solve(system, expected.T).T
    np.testing.assert_allclose(result.qmle, 0.5 * (expected + expected.T))
    for covariance in (result.qmle, result.ledoit_wolf):
        np.testing.assert_allclose(covariance, covariance.T)
        assert np.linalg.eigvalsh(covariance).min() > -_PSD_TOLERANCE
    assert 0.0 <= result.shrinkage <= 1.0


def test_innovation_ledoit_wolf_equals_the_statadic_owner() -> None:
    """Shrinkage and innovation covariance are the owner's estimate."""
    weights = _weights(5)
    panel = _panel(seed=711, dates=160, weights=weights, rho=0.35)
    fit = sar_factor_qmle(
        panel.excess_returns,
        weights,
        panel.factors,
        eigvals_w=cast("NDArray[np.complex128]", np.linalg.eigvals(weights)),
    )
    result = spatial_ledoit_wolf_covariance(fit, panel, weights)
    design = np.column_stack((np.ones(160), panel.factors))
    innovations = panel.excess_returns - fit.rho_hat * (
        panel.excess_returns @ weights.T
    )
    innovations = innovations - design @ fit.coefficients
    with jax.enable_x64(new_val=True):
        native = Estimand(Covariance(), estimator=LedoitWolf()).estimate(innovations)
    sigma = np.asarray(native.sigma, dtype=np.float64)
    np.testing.assert_allclose(
        result.innovation_ledoit_wolf, 0.5 * (sigma + sigma.T), rtol=1e-12
    )
    assert result.shrinkage == native.shrinkage


def test_total_covariance_reconstructs_systematic_and_residual_components() -> None:
    """Total covariance is exactly the declared factor-risk reconstruction."""
    weights = _weights(5)
    panel = _panel(seed=712, dates=160, weights=weights, rho=0.35)
    fit = sar_factor_qmle(
        panel.excess_returns,
        weights,
        panel.factors,
        eigvals_w=cast("NDArray[np.complex128]", np.linalg.eigvals(weights)),
    )
    result = spatial_ledoit_wolf_covariance(fit, panel, weights)
    system = np.eye(5) - fit.rho_hat * weights
    reduced_loadings = np.linalg.solve(system, np.asarray(fit.coefficients)[1:].T)
    factor_covariance = np.cov(panel.factors, rowvar=False, ddof=1)
    expected_systematic = reduced_loadings @ factor_covariance @ reduced_loadings.T
    np.testing.assert_allclose(
        result.systematic, expected_systematic, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(result.total_qmle, result.qmle + result.systematic)
    np.testing.assert_allclose(
        result.total_ledoit_wolf, result.ledoit_wolf + result.systematic
    )


def test_factor_only_control_has_identity_spatial_mapping() -> None:
    """The matched rho=0 control uses OLS residuals and the same target rule."""
    weights = _weights(5)
    panel = _panel(seed=713, dates=160, weights=weights, rho=0.0)
    fit = sar_factor_fit_at_rho(
        panel.excess_returns,
        weights,
        panel.factors,
        0.0,
        eigvals_w=cast("NDArray[np.complex128]", np.linalg.eigvals(weights)),
    )
    result = spatial_ledoit_wolf_covariance(fit, panel, np.eye(5, dtype=np.float64))
    np.testing.assert_allclose(result.qmle, result.innovation_qmle)
    np.testing.assert_allclose(result.total_qmle, result.qmle + result.systematic)


def test_ols_total_covariance_reconstructs_sample_return_covariance() -> None:
    """Centered OLS factors plus residuals recover total excess covariance."""
    weights = _weights(5)
    panel = _panel(seed=714, dates=160, weights=weights, rho=0.0)
    fit = sar_factor_fit_at_rho(
        panel.excess_returns,
        weights,
        panel.factors,
        0.0,
        eigvals_w=cast("NDArray[np.complex128]", np.linalg.eigvals(weights)),
    )
    design = np.column_stack((np.ones(len(panel.dates)), panel.factors))
    residual = panel.excess_returns - design @ fit.coefficients
    beta = np.asarray(fit.coefficients)[1:].T
    reconstructed = beta @ np.cov(panel.factors, rowvar=False, ddof=1) @ beta.T
    reconstructed += np.cov(residual, rowvar=False, ddof=1)
    np.testing.assert_allclose(
        reconstructed,
        np.cov(panel.excess_returns, rowvar=False, ddof=1),
        rtol=0.0,
        atol=1e-12,
    )


def test_unshrunk_spatial_residual_mapping_cancels_exactly() -> None:
    """Reversing the spatial transform recovers ordinary residual covariance."""
    rng = np.random.default_rng(715)
    n_dates, n_assets = 160, 5
    operator = _weights(n_assets)
    system = np.eye(n_assets) - 0.35 * operator
    residual = rng.normal(scale=0.01, size=(n_dates, n_assets))
    spatial_innovations = residual @ system.T
    mapped = _spatialize_covariance(
        system,
        np.cov(spatial_innovations, rowvar=False, ddof=1),
    )
    np.testing.assert_allclose(
        mapped, np.cov(residual, rowvar=False, ddof=1), atol=1e-12
    )


def test_nonconverged_fit_cannot_emit_a_covariance() -> None:
    """Covariance output cannot hide an unsuccessful upstream numerical fit."""
    weights = _weights(4)
    panel = _panel(seed=711, dates=120, weights=weights, rho=0.2)
    fit = sar_factor_qmle(
        panel.excess_returns,
        weights,
        panel.factors,
        eigvals_w=cast("NDArray[np.complex128]", np.linalg.eigvals(weights)),
    )
    with pytest.raises(RuntimeError, match="nonconverged"):
        spatial_ledoit_wolf_covariance(replace(fit, converged=False), panel, weights)


def test_dvc_registers_isolated_covariance_stage_without_paper_outputs() -> None:
    """The exploratory stage has dedicated outputs and no manuscript outputs."""
    root = Path(__file__).parents[7]
    document = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))
    stage = document["stages"]["p5_spatial_covariance"]
    assert "paper5-spatial-covariance" in stage["cmd"]
    assert any(
        "spatial_value_comparison/results.json" in str(dep) for dep in stage["deps"]
    )
    outputs = " ".join(str(output) for output in stage["outs"])
    assert "spatial_covariance" in outputs
    assert "src/latex" not in outputs


def test_config_rejects_overlapping_windows() -> None:
    """Calibration and evaluation covariance windows must be disjoint."""
    with pytest.raises(ValueError, match="ordered and disjoint"):
        Paper5SpatialCovarianceConfig(
            calibration_end="2023-01-01",
            evaluation_start="2023-01-01",
        ).validate()

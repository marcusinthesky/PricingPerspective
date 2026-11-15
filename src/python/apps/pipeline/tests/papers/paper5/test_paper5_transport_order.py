"""Tests for Paper 5's factor-conditioned transport-order comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import pytest
from pipeline.io.fama_french import FrenchFactorPanel
from pipeline.stages.papers.paper5 import estimate as estimate_module
from pipeline.stages.papers.paper5.estimate import (
    NUMERICAL_FLOOR,
    _global_scalar_minimize,
    factor_sar_parameter_count,
    sar_factor_daily_logscore,
    sar_factor_fit_at_rho,
    sar_factor_qmle,
    sar_factor_reduced_form_coefficients,
)
from pipeline.stages.papers.paper5.transport_order import (
    Paper5TransportOrderConfig,
    _align_factor_panel,
    _AlignedFactorPanel,
    _operator_separation,
    _paired_bootstrap,
    _validate_nested_panel_identity,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

RHO_RECOVERY_TOLERANCE = 0.06
FLOAT32_ROUNDING_TOLERANCE = 1e-6
LIKELIHOOD_IDENTITY_TOLERANCE = 1e-8
BOOTSTRAP_DRAWS = 4


def _weight_matrix(rng: np.random.Generator, n_assets: int) -> NDArray[np.float64]:
    """Return a dense directed row-stochastic matrix with zero diagonal."""
    weights = rng.uniform(0.2, 1.0, size=(n_assets, n_assets))
    np.fill_diagonal(weights, 0.0)
    return cast("NDArray[np.float64]", weights / weights.sum(axis=1, keepdims=True))


def _factor_sar_panel(
    *,
    seed: int,
    n_dates: int,
    n_assets: int,
    n_factors: int,
    rho: float,
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a spatial factor model with firm-specific loadings."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(scale=0.01, size=(n_dates, n_factors))
    loadings = rng.normal(size=(n_factors, n_assets))
    intercepts = rng.normal(scale=0.0002, size=n_assets)
    innovations = rng.normal(scale=0.008, size=(n_dates, n_assets))
    filtered_returns = intercepts + factors @ loadings + innovations
    returns = np.linalg.solve(
        np.eye(n_assets, dtype=np.float64) - rho * weights,
        filtered_returns.T,
    ).T
    return (
        cast("NDArray[np.float64]", returns),
        cast("NDArray[np.float64]", factors),
    )


def test_factor_alignment_subtracts_rf_and_preserves_nested_sample() -> None:
    """Every factor set uses the same non-overlapping excess-return panels."""
    calibration_dates = pd.date_range("2018-01-02", periods=40, freq="B")
    evaluation_dates = pd.date_range("2023-01-03", periods=40, freq="B")
    factor_dates = calibration_dates.append(evaluation_dates)
    frame = pd.DataFrame(
        {
            "Date": factor_dates,
            "MKT_RF": 0.002,
            "SMB": -0.001,
            "HML": 0.003,
            "RF": 0.0001,
        }
    )
    factor_panel = FrenchFactorPanel(
        model_id="ff3",
        frame=frame,
        factor_columns=("MKT_RF", "SMB", "HML"),
        source_hashes={"ff3": "test"},
        dropped_dates={"ff3": 0},
    )
    calibration_returns = np.full((40, 3), 0.01, dtype=np.float64)
    evaluation_returns = np.full((40, 3), 0.02, dtype=np.float64)
    calibration = _align_factor_panel(
        calibration_dates.to_numpy(dtype="datetime64[ns]"),
        calibration_returns,
        factor_panel,
    )
    evaluation = _align_factor_panel(
        evaluation_dates.to_numpy(dtype="datetime64[ns]"),
        evaluation_returns,
        factor_panel,
    )

    np.testing.assert_allclose(calibration.excess_returns, 0.0099)
    np.testing.assert_allclose(evaluation.excess_returns, 0.0199)
    _validate_nested_panel_identity(
        {"ff3": calibration, "ff5_mom": calibration},
        {"ff3": evaluation, "ff5_mom": evaluation},
        "ff5_mom",
    )


def test_factor_qmle_profiles_loadings_and_recovers_rho() -> None:
    """The exact factor profile recovers a synthetic spatial coefficient."""
    rng = np.random.default_rng(123)
    n_assets = 6
    true_rho = 0.45
    weights = _weight_matrix(rng, n_assets)
    returns, factors = _factor_sar_panel(
        seed=124,
        n_dates=800,
        n_assets=n_assets,
        n_factors=2,
        rho=true_rho,
        weights=weights,
    )

    fit = sar_factor_qmle(returns, weights, factors)
    scores = sar_factor_daily_logscore(fit, returns, weights, factors)

    assert abs(fit.rho_hat - true_rho) < RHO_RECOVERY_TOLERANCE
    assert fit.coefficients.shape == (3, n_assets)
    assert fit.coefficients.dtype == np.float64
    assert fit.converged
    np.testing.assert_allclose(
        np.sum(scores),
        fit.loglik,
        rtol=0.0,
        atol=LIKELIHOOD_IDENTITY_TOLERANCE,
    )


def test_scalar_profile_search_scans_and_refines_multiple_minima() -> None:
    """The guarded scalar search does not treat one Brent run as global."""

    def objective(value: float) -> float:
        return min((value - 0.23) ** 2 + 0.02, (value - 0.77) ** 2)

    result = _global_scalar_minimize(objective, (-1.0, 1.0), grid_points=41)

    assert result.x == pytest.approx(0.77, abs=1e-5)
    assert result.value == pytest.approx(0.0, abs=1e-10)
    assert result.converged


def test_scalar_profile_search_refines_both_endpoint_adjacent_intervals() -> None:
    """A minimum just inside either endpoint is not lost by the grid scan."""
    left = _global_scalar_minimize(lambda value: (value + 0.99) ** 2, (-0.999, 0.999))
    right = _global_scalar_minimize(lambda value: (value - 0.99) ** 2, (-0.999, 0.999))

    assert left.x == pytest.approx(-0.99, abs=1e-6)
    assert right.x == pytest.approx(0.99, abs=1e-6)
    assert left.value == pytest.approx(0.0, abs=1e-12)
    assert right.value == pytest.approx(0.0, abs=1e-12)
    assert left.refinement_gain > 0.0
    assert right.refinement_gain > 0.0


def test_scalar_profile_search_preserves_a_failed_refinement_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed local solve cannot be converted into a successful search."""

    class FailedResult:
        success = False
        x = float("nan")

    def failed_minimize_scalar(*_args: object, **_kwargs: object) -> FailedResult:
        return FailedResult()

    monkeypatch.setattr(estimate_module, "minimize_scalar", failed_minimize_scalar)
    result = _global_scalar_minimize(lambda value: value**2, (-1.0, 1.0), grid_points=5)

    assert result.x == pytest.approx(0.0)
    assert not result.converged


def test_factor_qmle_owns_float64_for_nearby_input_dtypes() -> None:
    """Input storage precision does not change the fitted decision or output dtype."""
    rng = np.random.default_rng(321)
    n_assets = 5
    weights = _weight_matrix(rng, n_assets)
    returns, factors = _factor_sar_panel(
        seed=322,
        n_dates=500,
        n_assets=n_assets,
        n_factors=2,
        rho=0.35,
        weights=weights,
    )

    fit64 = sar_factor_qmle(returns, weights, factors)
    rounded = sar_factor_qmle(
        returns.astype(np.float32),
        weights.astype(np.float32),
        factors.astype(np.float32),
    )

    assert fit64.rho_hat > 0.0
    assert rounded.rho_hat > 0.0
    assert rounded.coefficients.dtype == np.float64
    assert abs(fit64.rho_hat - rounded.rho_hat) < FLOAT32_ROUNDING_TOLERANCE


def test_diagonal_factor_qmle_recovers_rho_and_scores_exactly() -> None:
    """Firm-specific innovation variances preserve spatial recovery and density."""
    rng = np.random.default_rng(501)
    n_assets = 6
    n_dates = 1200
    true_rho = 0.40
    weights = _weight_matrix(rng, n_assets)
    factors = rng.normal(scale=0.01, size=(n_dates, 2))
    loadings = rng.normal(size=(2, n_assets))
    scales = np.linspace(0.004, 0.014, n_assets)
    innovations = rng.normal(size=(n_dates, n_assets)) * scales
    filtered = factors @ loadings + innovations
    returns = np.linalg.solve(
        np.eye(n_assets, dtype=np.float64) - true_rho * weights,
        filtered.T,
    ).T

    fit = sar_factor_qmle(returns, weights, factors, variance_model="diagonal")
    scores = sar_factor_daily_logscore(fit, returns, weights, factors)

    assert fit.variance_model == "diagonal"
    assert abs(fit.rho_hat - true_rho) < RHO_RECOVERY_TOLERANCE
    assert np.ptp(fit.innovation_variances) > 0.0
    np.testing.assert_allclose(np.sum(scores), fit.loglik, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("variance_model", ["spherical", "diagonal"])
def test_floor_binding_factor_profile_matches_daily_scores(
    variance_model: Literal["spherical", "diagonal"],
) -> None:
    """Variance clipping preserves the fitted-density score identity."""
    rng = np.random.default_rng(502)
    n_dates, n_assets, n_factors = 100, 4, 2
    factors = rng.normal(size=(n_dates, n_factors))
    design = np.column_stack((np.ones(n_dates), factors))
    coefficients = rng.normal(scale=0.01, size=(n_factors + 1, n_assets))
    returns = design @ coefficients + rng.normal(scale=1e-10, size=(n_dates, n_assets))
    weights = np.zeros((n_assets, n_assets), dtype=np.float64)

    fit = sar_factor_fit_at_rho(
        returns,
        weights,
        factors,
        0.0,
        variance_model=variance_model,
    )
    scores = sar_factor_daily_logscore(fit, returns, weights, factors)

    np.testing.assert_allclose(fit.innovation_variances, NUMERICAL_FLOOR)
    np.testing.assert_allclose(np.sum(scores), fit.loglik, atol=1e-8, rtol=0.0)


def test_reduced_form_factor_coefficients_equal_firmwise_ols() -> None:
    """The spatial structural coefficients map exactly to reduced-form OLS."""
    rng = np.random.default_rng(511)
    n_assets = 5
    weights = _weight_matrix(rng, n_assets)
    returns, factors = _factor_sar_panel(
        seed=512,
        n_dates=400,
        n_assets=n_assets,
        n_factors=3,
        rho=0.35,
        weights=weights,
    )
    fit = sar_factor_qmle(returns, weights, factors, variance_model="diagonal")
    design = np.column_stack((np.ones(len(returns)), factors))
    ols_coefficients, *_ = np.linalg.lstsq(design, returns, rcond=None)

    reduced = sar_factor_reduced_form_coefficients(fit, weights)

    np.testing.assert_allclose(reduced, ols_coefficients, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        design @ reduced,
        design @ ols_coefficients,
        atol=1e-10,
        rtol=1e-10,
    )
    assert factor_sar_parameter_count(fit, spatial=False) == n_assets * 4 + n_assets
    assert factor_sar_parameter_count(fit, spatial=True) == n_assets * 4 + n_assets + 1


def test_paired_bootstrap_refits_factor_models_and_predictive_parameters() -> None:
    """Every draw refits both fields on joint calibration and evaluation blocks."""
    rng = np.random.default_rng(77)
    n_assets = 4
    w1 = _weight_matrix(rng, n_assets)
    w2 = 0.98 * w1 + 0.02 * np.roll(w1, shift=1, axis=1)
    np.fill_diagonal(w2, 0.0)
    w2 /= w2.sum(axis=1, keepdims=True)
    matrices = {"w1": w1, "w2": cast("NDArray[np.float64]", w2)}
    eigenvalues = {
        order: cast("NDArray[np.complex128]", np.linalg.eigvals(weights))
        for order, weights in matrices.items()
    }
    calibration_returns, calibration_factors = _factor_sar_panel(
        seed=78,
        n_dates=120,
        n_assets=n_assets,
        n_factors=2,
        rho=0.30,
        weights=w2,
    )
    evaluation_returns, evaluation_factors = _factor_sar_panel(
        seed=79,
        n_dates=90,
        n_assets=n_assets,
        n_factors=2,
        rho=0.30,
        weights=w2,
    )
    calibration = _AlignedFactorPanel(
        dates=np.arange("2018-01-01", "2018-05-01", dtype="datetime64[D]")[:120],
        raw_returns=calibration_returns,
        excess_returns=calibration_returns,
        factors=calibration_factors,
        dropped_return_dates=0,
    )
    evaluation = _AlignedFactorPanel(
        dates=np.arange("2023-01-01", "2023-04-01", dtype="datetime64[D]")[:90],
        raw_returns=evaluation_returns,
        excess_returns=evaluation_returns,
        factors=evaluation_factors,
        dropped_return_dates=0,
    )

    draws = _paired_bootstrap(
        calibration,
        evaluation,
        matrices,
        eigenvalues,
        draws=BOOTSTRAP_DRAWS,
        block_length=7,
        seed=4,
    )

    assert len(draws) == BOOTSTRAP_DRAWS
    assert draws["all_solvers_converged"].all()
    np.testing.assert_allclose(
        draws["rho_w2_minus_w1"],
        draws["rho_w2"] - draws["rho_w1"],
    )
    assert np.isfinite(draws["predictive_score_w2_minus_w1"]).all()


def test_weak_separation_rule_blocks_orthogonalization() -> None:
    """Nearly identical induced fields trigger the frozen two-part support rule."""
    rng = np.random.default_rng(91)
    n_assets = 5
    w1 = _weight_matrix(rng, n_assets)
    w2 = 0.995 * w1 + 0.005 * np.roll(w1, shift=1, axis=1)
    np.fill_diagonal(w2, 0.0)
    w2 /= w2.sum(axis=1, keepdims=True)
    returns = rng.normal(scale=0.02, size=(200, n_assets))

    result = _operator_separation(
        w1,
        cast("NDArray[np.float64]", w2),
        cast("NDArray[np.float64]", returns),
        Paper5TransportOrderConfig(),
    )

    assert result["weakly_separated"] is True
    assert result["decision"] == "weakly_separated_do_not_orthogonalize"
    assert result["orthogonalization"] == (
        "not_performed_joint_unchanged_field_model_reported_separately"
    )

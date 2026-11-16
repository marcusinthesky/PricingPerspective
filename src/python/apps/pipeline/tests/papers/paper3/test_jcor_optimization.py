"""Independent regression evidence for Statadic-backed portfolio optimization."""

from __future__ import annotations

import numpy as np
import pytest
from pipeline._kernels.paper3.validation import (
    CertificateValidationError,
    solve_certificate,
)
from pipeline.stages.papers.paper3.certificate_validation import (
    GMV_KKT_TOLERANCE,
    _gmv,
)
from pipeline.stages.papers.paper3.representation_ablation import (
    _return_bootstrap_draws,
)
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _scipy_gmv(covariance: np.ndarray) -> np.ndarray:
    """Return an independent SLSQP benchmark under the original normalization."""
    dimension = covariance.shape[0]
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.full(dimension, 1.0 / dimension),
        jac=lambda weights: 2.0 * covariance @ weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * dimension,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert result.success
    return result.x


def _check_gmv_optimality(
    covariance: np.ndarray,
    weights: np.ndarray,
    expected: np.ndarray,
    *,
    objective_tolerance: float,
) -> None:
    """Check feasibility, an independent objective and a global first-order bound."""
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert float(weights @ covariance @ weights) == pytest.approx(
        float(expected @ covariance @ expected), abs=objective_tolerance, rel=0.0
    )
    quadratic = 2.0 * covariance
    gradient = (quadratic / np.linalg.norm(quadratic, ord=np.inf)) @ weights
    gap = float(gradient @ weights - gradient.min())
    # Each of 2n complementary-slackness terms contributes at most the KKT
    # tolerance; stationarity contributes at most 2 * tolerance because the
    # L1 distance between feasible probability vectors is bounded by two.
    gap_bound = (2 * len(weights) + 2) * GMV_KKT_TOLERANCE
    assert -GMV_KKT_TOLERANCE <= gap <= gap_bound


@pytest.mark.reference
def test_sample_gmv_matches_analytic_and_scipy_objectives() -> None:
    """The backend change preserves the long-only minimum-variance target."""
    covariance = np.diag([1.0, 2.0, 3.0])
    weights = _gmv(covariance)
    _check_gmv_optimality(
        covariance,
        weights,
        np.array([6 / 11, 3 / 11, 2 / 11]),
        objective_tolerance=1e-12,
    )
    generator = np.random.default_rng(20261025)
    panel = generator.normal(size=(30, 4))
    covariance = np.cov(panel, rowvar=False, ddof=1)
    weights = _gmv(covariance)
    reference = _scipy_gmv(covariance)
    assert float(weights @ covariance @ weights) == pytest.approx(
        float(reference @ covariance @ reference), abs=1e-11
    )


@pytest.mark.reference
@pytest.mark.parametrize("covariance_multiplier", [1e-6, 1e4, 1e6])
def test_sample_gmv_is_invariant_to_currency_units(
    covariance_multiplier: float,
) -> None:
    """Daily correlated risks keep the same allocation when currency units change."""
    volatility = np.array([0.008, 0.012, 0.020])
    correlation = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    covariance = correlation * np.outer(volatility, volatility)
    inverse_ones = np.linalg.solve(covariance, np.ones(3))
    expected = inverse_ones / inverse_ones.sum()
    assert np.all(expected > 0.0)
    baseline = _gmv(covariance)
    rescaled = _gmv(covariance * covariance_multiplier)
    # The unconstrained closed-form optimum is feasible, so it independently
    # identifies the long-only solution for this positive-definite covariance.
    _check_gmv_optimality(covariance, baseline, expected, objective_tolerance=1e-14)
    _check_gmv_optimality(covariance, rescaled, expected, objective_tolerance=1e-14)
    np.testing.assert_allclose(rescaled, baseline, atol=2e-12, rtol=0.0)


def test_sample_gmv_retains_a_finite_uniform_zero_risk_solution() -> None:
    """Objective normalization must not divide the zero covariance by zero."""
    np.testing.assert_array_equal(_gmv(np.zeros((4, 4))), np.full(4, 0.25))


@pytest.mark.reference
def test_certificate_preserves_credit_and_strict_centered_rank_gate() -> None:
    """Conditional negative definiteness permits the QP but never waives rank."""
    generator = np.random.default_rng(20261025)
    # Reproduce the pre-migration fixture sequence: returns preceded geometry.
    generator.normal(size=(30, 4))
    points = generator.normal(size=(4, 3))
    squared = cdist(points, points) ** 2
    result = solve_certificate(squared, 0.4)
    kkt_tolerance = 1e-6
    # Pre-migration SLSQP fixture, with one free and two upper-active weights.
    np.testing.assert_allclose(result.weights, [0.0, 0.4, 0.4, 0.2], atol=1e-7)
    assert result.credit == pytest.approx(1.2796362236883831, abs=1e-7)
    assert result.kkt_residual <= kkt_tolerance
    collinear = cdist(np.arange(4.0)[:, None], np.arange(4.0)[:, None]) ** 2
    with pytest.raises(CertificateValidationError, match="not positive definite"):
        solve_certificate(collinear, 0.4)


def test_certificate_preserves_the_unique_boundary_support() -> None:
    """The right triangle's unique certificate excludes its origin vertex."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    solution = solve_certificate(cdist(points, points) ** 2, 1.0)
    support_threshold = 1e-8
    np.testing.assert_allclose(
        solution.weights, [0.0, 0.5, 0.5], atol=support_threshold, rtol=0
    )
    np.testing.assert_array_equal(
        solution.weights > support_threshold, [False, True, True]
    )
    assert solution.credit == pytest.approx(0.5, abs=1e-12)
    kkt_tolerance = 1e-6
    assert solution.kkt_residual <= kkt_tolerance


@pytest.mark.reference
def test_supplied_return_draws_preserve_refits_pairing_and_failed_positions() -> None:
    """Re-standardization and ddof=1 agree per supplied draw with a host oracle."""
    generator = np.random.default_rng(654321)
    panel = generator.normal(size=(24, 4))
    panel[0] = 0.0
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    first = generator.integers(1, len(panel), size=len(panel))
    second = generator.integers(1, len(panel), size=len(panel))
    indices = np.stack([first, np.zeros(len(panel), dtype=int), second, first])
    original_indices = indices.copy()
    draws, count = _return_bootstrap_draws(panel, weights, indices)
    assert count == len(indices) - 1
    assert np.isnan(draws[1])
    np.testing.assert_array_equal(indices, original_indices)
    assert draws[0] == draws[3]
    for draw_id in (0, 2, 3):
        sampled = panel[indices[draw_id]]
        standardized = sampled / np.std(sampled, axis=0, ddof=1)
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        reference = _scipy_gmv(covariance)
        expected = (
            100.0
            * np.var(standardized @ weights, ddof=1)
            / np.var(standardized @ reference, ddof=1)
        )
        assert draws[draw_id] == pytest.approx(expected, abs=1e-8)


def test_return_bootstrap_retains_empty_draw_behavior() -> None:
    """The host adapter can report a deliberately empty requested schedule."""
    draws, count = _return_bootstrap_draws(
        np.ones((3, 2)), np.full(2, 0.5), np.empty((0, 3), dtype=int)
    )
    assert draws.shape == (0,)
    assert count == 0

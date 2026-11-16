"""Tests for Paper 5's factor-conditioned joint W1/W2 likelihood."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pipeline.stages.papers.paper5.bootstrap import (
    independent_stationary_bootstrap_indices,
)
from pipeline.stages.papers.paper5.estimate import sar_factor_qmle
from pipeline.stages.papers.paper5.factor_two_field import (
    FACTOR_TWO_FIELD_DRAW_COLUMNS,
    _null_profile_stats,
    _resampled_stats,
    _restricted_factor_residuals,
    build_factor_two_field_problem,
    factor_nested_qlr_bootstrap,
    factor_two_field_bootstrap,
    factor_two_field_daily_logscore,
    factor_two_field_qmle,
    factor_two_field_with_grid,
)
from pipeline.stages.papers.paper5.two_field import TwoFieldGrid, two_field_qmle

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pipeline.stages.papers.paper5.factor_two_field import FactorTwoFieldProblem

type Float = NDArray[np.float64]

_SEED = 20260903
_RHO_TOLERANCE = 0.10
_COEFFICIENT_RECONCILIATION_TOLERANCE = 1e-6
_FLOAT32_RHO_TOLERANCE = 1e-6
_DRAW_COLUMN_COUNT = 8


def _row_stochastic(
    rng: np.random.Generator,
    n_assets: int,
    *,
    support_size: int,
) -> Float:
    """Draw one directed, zero-diagonal row-stochastic field."""
    weights = np.zeros((n_assets, n_assets), dtype=np.float64)
    for row in range(n_assets):
        candidates = np.delete(np.arange(n_assets), row)
        support = rng.choice(candidates, size=support_size, replace=False)
        weights[row, support] = rng.dirichlet(np.ones(support_size))
    return weights


def _simulate_factor_two_field(
    rng: np.random.Generator,
    w1: Float,
    w2: Float,
    *,
    n_dates: int,
    n_factors: int,
    rho_w1: float,
    rho_w2: float,
) -> tuple[Float, Float]:
    """Simulate the exact factor-spatial working model."""
    n_assets = w1.shape[0]
    factors = rng.normal(scale=0.01, size=(n_dates, n_factors))
    loadings = rng.normal(size=(n_factors, n_assets))
    intercepts = rng.normal(scale=0.0002, size=n_assets)
    innovations = rng.normal(scale=0.007, size=(n_dates, n_assets))
    filtered = intercepts + factors @ loadings + innovations
    operator = np.eye(n_assets, dtype=np.float64) - rho_w1 * w1 - rho_w2 * w2
    returns = np.linalg.solve(operator, filtered.T).T
    return cast("Float", returns), cast("Float", factors)


def _problems(
    returns: Float,
    factors: Float,
    w1: Float,
    w2: Float,
) -> tuple[FactorTwoFieldProblem, FactorTwoFieldProblem, FactorTwoFieldProblem]:
    """Build one joint profile and its two pinned boundaries."""
    joint_grid = TwoFieldGrid.build(theta_points=81, rho_points=301)
    joint = build_factor_two_field_problem(returns, w1, w2, factors, joint_grid)
    w1_only = factor_two_field_with_grid(
        joint,
        TwoFieldGrid.build(
            theta_points=1,
            rho_points=301,
            theta_lower=1.0,
            theta_upper=1.0,
        ),
    )
    w2_only = factor_two_field_with_grid(
        joint,
        TwoFieldGrid.build(
            theta_points=1,
            rho_points=301,
            theta_lower=0.0,
            theta_upper=0.0,
        ),
    )
    return joint, w1_only, w2_only


def test_factor_two_field_recovers_distinct_positive_channels() -> None:
    """The profiled likelihood recovers a separated synthetic joint signal."""
    rng = np.random.default_rng(_SEED)
    n_assets = 7
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=4)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=900,
        n_factors=3,
        rho_w1=0.25,
        rho_w2=0.35,
    )
    joint_grid = TwoFieldGrid.build(theta_points=101, rho_points=401)

    fit = factor_two_field_qmle(
        build_factor_two_field_problem(returns, w1, w2, factors, joint_grid)
    )

    assert abs(fit.rho_w1 - 0.25) < _RHO_TOLERANCE
    assert abs(fit.rho_w2 - 0.35) < _RHO_TOLERANCE
    assert fit.rho_total < 1.0
    assert fit.coefficients.shape == (4, n_assets)
    assert fit.coefficients.dtype == np.float64
    assert fit.converged


def test_factor_two_field_daily_logscore_is_finite_on_fixed_fit() -> None:
    """The held-out scorer returns one finite score for every evaluation date."""
    rng = np.random.default_rng(_SEED + 20)
    n_assets = 5
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=3)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=240,
        n_factors=2,
        rho_w1=0.20,
        rho_w2=0.25,
    )
    problem = build_factor_two_field_problem(
        returns,
        w1,
        w2,
        factors,
        TwoFieldGrid.build(theta_points=21, rho_points=81),
    )
    fit = factor_two_field_qmle(problem)

    scores = factor_two_field_daily_logscore(fit, returns, w1, w2, factors)

    assert scores.shape == (returns.shape[0],)
    assert np.all(np.isfinite(scores))


def test_factor_boundaries_reproduce_single_field_likelihoods() -> None:
    """Pinned W1/W2 profiles equal the existing factor single-field fits."""
    rng = np.random.default_rng(_SEED + 1)
    n_assets = 6
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=4)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=600,
        n_factors=2,
        rho_w1=0.40,
        rho_w2=0.0,
    )
    _joint, w1_only, w2_only = _problems(returns, factors, w1, w2)

    for weights, problem in ((w1, w1_only), (w2, w2_only)):
        expected = sar_factor_qmle(returns, weights, factors, compute_se=False)
        actual = factor_two_field_qmle(problem)
        assert actual.rho_total == pytest.approx(expected.rho_hat, abs=1e-4)
        assert actual.loglik == pytest.approx(expected.loglik, abs=1e-6)
        np.testing.assert_allclose(
            actual.coefficients,
            expected.coefficients,
            rtol=0.0,
            atol=_COEFFICIENT_RECONCILIATION_TOLERANCE,
        )


def test_joint_fit_contains_both_factor_conditioned_boundaries() -> None:
    """The unchanged-field joint fit cannot fall below either nested boundary."""
    rng = np.random.default_rng(_SEED + 2)
    n_assets = 6
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=4)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=500,
        n_factors=2,
        rho_w1=0.20,
        rho_w2=0.30,
    )
    joint, w1_only, w2_only = _problems(returns, factors, w1, w2)
    fit_joint = factor_two_field_qmle(joint)

    assert fit_joint.loglik >= factor_two_field_qmle(w1_only).loglik - 1e-7
    assert fit_joint.loglik >= factor_two_field_qmle(w2_only).loglik - 1e-7


def test_joint_date_bootstrap_reprofiles_factor_nuisances() -> None:
    """A resampled fit equals rebuilding the factor profile on that draw."""
    rng = np.random.default_rng(_SEED + 3)
    n_assets = 5
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=3)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=140,
        n_factors=2,
        rho_w1=0.20,
        rho_w2=0.25,
    )
    grid = TwoFieldGrid.build(theta_points=41, rho_points=151)
    problem = build_factor_two_field_problem(returns, w1, w2, factors, grid)
    index = rng.integers(0, len(returns), size=len(returns), dtype=np.int64)

    draw = factor_two_field_bootstrap(problem, index[None, :])[0]
    rebuilt = factor_two_field_qmle(
        build_factor_two_field_problem(returns[index], w1, w2, factors[index], grid)
    )

    np.testing.assert_allclose(
        draw[:5],
        [
            rebuilt.rho_w1,
            rebuilt.rho_w2,
            rebuilt.rho_total,
            rebuilt.theta_w1,
            rebuilt.loglik,
        ],
        rtol=0.0,
        atol=1e-8,
    )
    direct_core = two_field_qmle(
        replace(problem.core, stats=_resampled_stats(problem, index))
    )
    assert draw[0] == pytest.approx(direct_core["rho_b"], abs=1e-10)


def test_null_factor_projection_matches_explicit_panel_refit() -> None:
    """The fast null path is the exact full-panel nuisance refit in residual space."""
    rng = np.random.default_rng(_SEED + 4)
    n_assets = 5
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=3)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=160,
        n_factors=2,
        rho_w1=0.35,
        rho_w2=0.0,
    )
    _joint, restricted, _w2_only = _problems(returns, factors, w1, w2)
    fit = factor_two_field_qmle(restricted)
    residuals = _restricted_factor_residuals(restricted, fit, 1.0)
    operator = np.eye(n_assets, dtype=np.float64) - fit.rho_total * w1
    multiplier = cast("Float", np.linalg.inv(operator))
    basis = cast("Float", np.linalg.qr(restricted.design, mode="reduced")[0])

    fast = _null_profile_stats(residuals, basis, multiplier, w1, w2)
    fitted = restricted.design @ fit.coefficients
    explicit_returns = (fitted + residuals) @ multiplier.T
    explicit = build_factor_two_field_problem(
        cast("Float", explicit_returns),
        w1,
        w2,
        factors,
        restricted.core.grid,
    )

    np.testing.assert_allclose(fast.gram, explicit.core.stats.gram, rtol=1e-10)
    np.testing.assert_allclose(fast.totals, explicit.core.stats.totals, atol=1e-12)


def test_boundary_bootstrap_is_unit_for_identical_fields() -> None:
    """Indistinguishable W1/W2 fields have zero incremental QLR and p one."""
    rng = np.random.default_rng(_SEED + 5)
    n_assets = 5
    weights = _row_stochastic(rng, n_assets, support_size=3)
    returns, factors = _simulate_factor_two_field(
        rng,
        weights,
        weights,
        n_dates=180,
        n_factors=2,
        rho_w1=0.40,
        rho_w2=0.0,
    )
    joint, w1_only, _w2_only = _problems(returns, factors, weights, weights)
    indices = independent_stationary_bootstrap_indices(
        len(returns),
        n_assets,
        draws=8,
        expected_block_length=5,
        seed=_SEED + 5,
    )

    test = factor_nested_qlr_bootstrap(joint, w1_only, indices)

    assert test.statistic == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(test.draws, 0.0, atol=1e-7)
    assert test.exceedances == len(indices)
    assert test.p_value == pytest.approx(1.0)


def test_factor_two_field_owns_float64_for_float32_storage() -> None:
    """Storage dtype rounding preserves the inferential branch and float64 outputs."""
    rng = np.random.default_rng(_SEED + 6)
    n_assets = 5
    w1 = _row_stochastic(rng, n_assets, support_size=2)
    w2 = _row_stochastic(rng, n_assets, support_size=3)
    returns, factors = _simulate_factor_two_field(
        rng,
        w1,
        w2,
        n_dates=400,
        n_factors=2,
        rho_w1=0.20,
        rho_w2=0.30,
    )
    grid = TwoFieldGrid.build(theta_points=61, rho_points=251)
    fit64 = factor_two_field_qmle(
        build_factor_two_field_problem(returns, w1, w2, factors, grid)
    )
    fit32 = factor_two_field_qmle(
        build_factor_two_field_problem(
            returns.astype(np.float32),
            w1.astype(np.float32),
            w2.astype(np.float32),
            factors.astype(np.float32),
            grid,
        )
    )

    assert fit32.coefficients.dtype == np.float64
    assert abs(fit64.rho_w1 - fit32.rho_w1) < _FLOAT32_RHO_TOLERANCE
    assert abs(fit64.rho_w2 - fit32.rho_w2) < _FLOAT32_RHO_TOLERANCE
    assert len(FACTOR_TWO_FIELD_DRAW_COLUMNS) == _DRAW_COLUMN_COUNT

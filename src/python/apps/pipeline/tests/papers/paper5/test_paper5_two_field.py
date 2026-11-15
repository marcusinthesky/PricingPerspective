"""Tests for Paper 5's two-field spatial QMLE and its nested boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5 import two_field as two_field_module
from pipeline.stages.papers.paper5.bootstrap import (
    independent_stationary_bootstrap_indices,
)
from pipeline.stages.papers.paper5.two_field import (
    TwoFieldEstimationError,
    TwoFieldGrid,
    _aggregate,
    _mixture_boundary_index,
    _null_panel,
    _point_loglik,
    _profile_loglik,
    _residual_sum_of_squares,
    _restricted_residuals,
    build_two_field_problem,
    channel_intensity,
    matrix_log_determinant,
    nested_qlr_bootstrap,
    quasi_likelihood_ratio,
    spectrum_log_determinant,
    two_field_bootstrap,
    two_field_panel_stats,
    two_field_qmle,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5.two_field import TwoFieldProblem

Float = NDArray[np.float64]
Fields = tuple[Float, Float, Float]

_SEED = 20260823
_N_ASSETS = 6
_N_DATES = 220
_RHO_GRID = 2001
_RECONCILIATION_TOLERANCE = 1e-4
_QLR_DRAWS = 8
_SMALL_N_DATES = 4


def _row_stochastic(rng: np.random.Generator, n_assets: int, *, sparsity: int) -> Float:
    """Draw a nonnegative zero-diagonal matrix with unit row sums."""
    weights = np.zeros((n_assets, n_assets), dtype=np.float64)
    for row in range(n_assets):
        candidates = np.delete(np.arange(n_assets), row)
        support = rng.choice(candidates, size=sparsity, replace=False)
        weights[row, support] = rng.dirichlet(np.ones(sparsity))
    return weights


def _simulate_panel(
    rng: np.random.Generator,
    weights: Float,
    *,
    rho: float,
    n_dates: int,
) -> Float:
    """Simulate a spatial-lag panel so the fitted coefficient is well separated."""
    n_assets = weights.shape[0]
    multiplier = np.linalg.inv(np.eye(n_assets) - rho * weights)
    innovations = rng.normal(scale=0.01, size=(n_dates, n_assets))
    return np.asarray(innovations @ multiplier.T, dtype=np.float64)


@pytest.fixture(name="fields")
def fixture_fields() -> Fields:
    """Build a simulated panel and two distinct admissible interaction fields."""
    rng = np.random.default_rng(_SEED)
    w_b = _row_stochastic(rng, _N_ASSETS, sparsity=2)
    w_n = _row_stochastic(rng, _N_ASSETS, sparsity=4)
    returns = _simulate_panel(rng, w_b, rho=0.6, n_dates=_N_DATES)
    return returns, w_b, w_n


def _pinned_problem(
    returns: Float, w_b: Float, w_n: Float, theta: float
) -> TwoFieldProblem:
    """Assemble a problem whose mixture weight is pinned to one field."""
    grid = TwoFieldGrid.build(
        theta_points=1,
        rho_points=_RHO_GRID,
        theta_lower=theta,
        theta_upper=theta,
        rho_lower=-0.999,
        rho_upper=0.999,
    )
    return build_two_field_problem(returns, w_b, w_n, grid)


def test_pinned_first_field_reproduces_single_field_qmle(
    fields: Fields,
) -> None:
    """At the first boundary the two-field fit is the published single-field fit."""
    returns, w_b, w_n = fields
    reference = est.sar_qmle(returns, w_b, compute_se=False)
    fit = two_field_qmle(_pinned_problem(returns, w_b, w_n, 1.0))

    assert fit["rho_b"] == pytest.approx(
        reference["rho_hat"], abs=_RECONCILIATION_TOLERANCE
    )
    assert fit["rho_n"] == pytest.approx(0.0, abs=1e-12)
    assert fit["loglik"] == pytest.approx(reference["loglik"], rel=1e-9)
    assert fit["alpha_hat"] == pytest.approx(reference["alpha_hat"], abs=1e-8)
    assert fit["sigma2_hat"] == pytest.approx(reference["sigma2_hat"], rel=1e-6)


def test_pinned_second_field_reproduces_single_field_qmle(
    fields: Fields,
) -> None:
    """At the second boundary the fit reproduces the comparator field's QMLE."""
    returns, w_b, w_n = fields
    reference = est.sar_qmle(returns, w_n, compute_se=False)
    fit = two_field_qmle(_pinned_problem(returns, w_b, w_n, 0.0))

    assert fit["rho_n"] == pytest.approx(
        reference["rho_hat"], abs=_RECONCILIATION_TOLERANCE
    )
    assert fit["rho_b"] == pytest.approx(0.0, abs=1e-12)
    assert fit["loglik"] == pytest.approx(reference["loglik"], rel=1e-9)


def test_gram_form_equals_direct_residual_sum_of_squares(
    fields: Fields,
) -> None:
    """The 3x3 quadratic form reproduces the concentrated residual sum of squares."""
    returns, w_b, w_n = fields
    stats = two_field_panel_stats(returns, w_b, w_n)
    aggregate = _aggregate(stats, None)
    theta, rho = 0.35, 0.42
    rho_b, rho_n = rho * theta, rho * (1.0 - theta)

    residual_pre = returns - rho_b * (returns @ w_b.T) - rho_n * (returns @ w_n.T)
    direct = float(np.sum((residual_pre - residual_pre.mean()) ** 2))

    assert _residual_sum_of_squares(aggregate.gram, theta, rho) == pytest.approx(
        direct, rel=1e-10
    )


def test_vectorized_profile_matches_pointwise_objective(
    fields: Fields,
) -> None:
    """The grid profile agrees with the exact pointwise objective it vectorizes."""
    returns, w_b, w_n = fields
    grid = TwoFieldGrid.build(theta_points=11, rho_points=21)
    problem = build_two_field_problem(returns, w_b, w_n, grid)
    aggregate = _aggregate(problem.stats, None)
    profile = _profile_loglik(aggregate, problem)

    for theta_index in (0, 5, 10):
        for rho_index in (1, 9, 20):
            expected = _point_loglik(
                aggregate,
                problem,
                float(grid.theta[theta_index]),
                float(grid.rho[rho_index]),
            )
            assert profile[theta_index, rho_index] == pytest.approx(expected, rel=1e-9)


def test_both_log_determinant_routes_agree(fields: Fields) -> None:
    """The tabulated spectral Jacobian equals the pointwise LU one.

    The grid path tabulates the spectrum once; the polish path factorizes at an
    arbitrary point. They must agree, or a polished estimate would be optimizing
    a different objective from the grid that bracketed it.
    """
    returns, w_b, w_n = fields
    grid = TwoFieldGrid.build(theta_points=7, rho_points=9)
    problem = build_two_field_problem(returns, w_b, w_n, grid)

    for theta_index, theta in enumerate(grid.theta):
        for rho in grid.rho:
            spectral = spectrum_log_determinant(problem.eigenvalues[theta_index], rho)
            factorized = matrix_log_determinant(
                w_b, w_n, rho * theta, rho * (1.0 - theta)
            )
            assert spectral == pytest.approx(factorized, abs=1e-9)


def test_per_date_statistics_are_additive_over_a_resample(
    fields: Fields,
) -> None:
    """Aggregating a date subset equals building statistics from that subpanel."""
    returns, w_b, w_n = fields
    stats = two_field_panel_stats(returns, w_b, w_n)
    index = np.array([3, 3, 7, 11, 11, 11, 40], dtype=np.int64)

    resampled = _aggregate(stats, index)
    direct = _aggregate(two_field_panel_stats(returns[index], w_b, w_n), None)

    assert resampled.t_obs == direct.t_obs
    np.testing.assert_allclose(resampled.gram, direct.gram, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(resampled.column, direct.column, rtol=1e-10, atol=1e-16)


def test_joint_fit_never_falls_below_its_nested_boundaries(
    fields: Fields,
) -> None:
    """The joint model contains both single-field fits, so it cannot fit worse."""
    returns, w_b, w_n = fields
    grid = TwoFieldGrid.build(theta_points=51, rho_points=401)
    joint = two_field_qmle(build_two_field_problem(returns, w_b, w_n, grid))
    first = two_field_qmle(_pinned_problem(returns, w_b, w_n, 1.0))
    second = two_field_qmle(_pinned_problem(returns, w_b, w_n, 0.0))

    assert joint["loglik"] >= first["loglik"] - 1e-6
    assert joint["loglik"] >= second["loglik"] - 1e-6
    assert 0.0 <= joint["rho_b"] + joint["rho_n"] < 1.0


def test_bootstrap_returns_one_admissible_row_per_draw(
    fields: Fields,
) -> None:
    """Every joint-date refit stays inside the economic region."""
    returns, w_b, w_n = fields
    grid = TwoFieldGrid.build(theta_points=21, rho_points=101)
    problem = build_two_field_problem(returns, w_b, w_n, grid)
    rng = np.random.default_rng(_SEED)
    indices = rng.integers(0, _N_DATES, size=(8, _N_DATES)).astype(np.int64)

    draws = two_field_bootstrap(problem, indices)

    assert draws.shape == (8, 5)
    assert np.all(draws[:, 0] >= 0.0)
    assert np.all(draws[:, 1] >= 0.0)
    assert np.all(draws[:, 0] + draws[:, 1] < 1.0)
    assert np.all(np.isin(draws[:, 3:], [0.0, 1.0]))


def test_failed_interior_search_stays_nonconverged_when_endpoint_wins(
    fields: Fields,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful endpoint does not erase a failed interior search."""
    returns, w_b, w_n = fields
    grid = TwoFieldGrid.build(theta_points=3, rho_points=51)
    problem = build_two_field_problem(returns, w_b, w_n, grid)

    def failed_refine(
        *_args: object, **_kwargs: object
    ) -> tuple[float, float, float, bool]:
        return 0.5, 0.4, 100_000.0, False

    def successful_endpoint_refine(
        *_args: object, **_kwargs: object
    ) -> tuple[float, float, bool]:
        return 0.4, 100_001.0, True

    monkeypatch.setattr(two_field_module, "_refine", failed_refine)
    monkeypatch.setattr(
        two_field_module, "_refine_fixed_theta", successful_endpoint_refine
    )

    fit = two_field_qmle(problem)

    assert fit["loglik"] == pytest.approx(100_001.0)
    assert not fit["converged"]


def test_two_field_bootstrap_rejects_a_failed_fit(
    fields: Fields,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unadjusted bootstrap output never silently drops failed fit status."""
    returns, w_b, w_n = fields
    problem = build_two_field_problem(
        returns, w_b, w_n, TwoFieldGrid.build(theta_points=3, rho_points=51)
    )
    indices = np.zeros((1, _N_DATES), dtype=np.int64)

    def failed_qmle(*_args: object, **_kwargs: object) -> dict[str, float]:
        return {"converged": 0.0}

    monkeypatch.setattr(two_field_module, "two_field_qmle", failed_qmle)

    with pytest.raises(TwoFieldEstimationError, match="bootstrap draw 0"):
        two_field_bootstrap(problem, indices)


def test_quasi_likelihood_ratio_truncates_instead_of_reflecting() -> None:
    """A failed nested optimization is rejected rather than reflected."""
    assert quasi_likelihood_ratio(12.0, 10.5) == pytest.approx(3.0)
    with pytest.raises(TwoFieldEstimationError, match="materially lower"):
        quasi_likelihood_ratio(10.5, 12.0)


def test_boundary_flags_use_final_coefficients_not_grid_index() -> None:
    """A polished near-endpoint weight is not silently relabeled as a boundary."""
    assert _mixture_boundary_index(1.0, 0.5) == 1
    assert _mixture_boundary_index(0.0, 0.5) == 0
    assert _mixture_boundary_index(0.999, 0.5) is None
    assert _mixture_boundary_index(1.0, 0.0) is None


def test_identity_resample_reconstructs_the_restricted_panel(fields: Fields) -> None:
    """Restricted innovations and the SAR multiplier invert each other."""
    returns, w_b, w_n = fields
    problem = _pinned_problem(returns, w_b, w_n, 1.0)
    fit = two_field_qmle(problem)
    residuals = _restricted_residuals(
        returns,
        w_b,
        rho=fit["rho_total"],
        alpha=fit["alpha_hat"],
    )
    multiplier = np.asarray(
        np.linalg.inv(np.eye(_N_ASSETS, dtype=np.float64) - fit["rho_total"] * w_b),
        dtype=np.float64,
    )
    reconstructed = _null_panel(
        residuals,
        multiplier,
        alpha=fit["alpha_hat"],
        asset_date_index=np.broadcast_to(
            np.arange(_N_DATES, dtype=np.int64)[:, None],
            residuals.shape,
        ),
    )

    np.testing.assert_allclose(reconstructed, returns, rtol=1e-10, atol=1e-12)


def test_nested_qlr_uses_independent_assetwise_paths() -> None:
    """Null paths have an explicit draw-by-asset-by-date contract."""
    indices = independent_stationary_bootstrap_indices(
        _SMALL_N_DATES,
        2,
        draws=3,
        expected_block_length=2,
        seed=_SEED,
    )
    assert indices.shape == (3, 2, _SMALL_N_DATES)
    assert np.all((indices >= 0) & (indices < _SMALL_N_DATES))


def test_nested_qlr_null_bootstrap_respects_an_indistinguishable_field(
    fields: Fields,
) -> None:
    """Identical fields yield no nested gain and a unit bootstrap p-value."""
    returns, w_b, _w_n = fields
    joint_grid = TwoFieldGrid.build(theta_points=11, rho_points=101)
    restricted_grid = TwoFieldGrid.build(
        theta_points=1,
        rho_points=101,
        theta_lower=1.0,
        theta_upper=1.0,
    )
    joint = build_two_field_problem(returns, w_b, w_b, joint_grid)
    restricted = build_two_field_problem(returns, w_b, w_b, restricted_grid)
    indices = independent_stationary_bootstrap_indices(
        _N_DATES,
        _N_ASSETS,
        draws=_QLR_DRAWS,
        expected_block_length=5,
        seed=_SEED,
    )

    test = nested_qlr_bootstrap(returns, joint, restricted, indices)

    assert test.statistic == pytest.approx(0.0, abs=1e-8)
    assert test.draws == pytest.approx(np.zeros(_QLR_DRAWS), abs=1e-8)
    assert test.exceedances == _QLR_DRAWS
    assert test.p_value == pytest.approx(1.0)


def test_polished_refits_leave_the_grid(fields: Fields) -> None:
    """Polishing moves refits off the grid nodes an unpolished fit is pinned to.

    Without it every resampled coefficient is a grid node and the reported
    intervals inherit the grid spacing, so this guards the resolution of the
    published intervals rather than the point estimate.
    """
    returns, w_b, w_n = fields
    grid = TwoFieldGrid.build(theta_points=11, rho_points=51)
    problem = build_two_field_problem(returns, w_b, w_n, grid)
    rng = np.random.default_rng(_SEED)
    indices = rng.integers(0, _N_DATES, size=(12, _N_DATES)).astype(np.int64)

    coarse = two_field_bootstrap(problem, indices, refine=False)
    polished = two_field_bootstrap(problem, indices, refine=True)

    on_grid = np.isin(coarse[:, 2], grid.theta)
    assert np.all(on_grid)
    assert not np.all(np.isin(polished[:, 2], grid.theta))
    assert np.all(polished[:, 0] + polished[:, 1] < 1.0)


def test_channel_intensity_inverts_the_two_field_closure() -> None:
    """Channel intensities round-trip through the closure's coefficient map."""
    lambda_b, lambda_n = 2.5, 0.75
    total = 1.0 + lambda_b + lambda_n
    rho_b, rho_n = lambda_b / total, lambda_n / total

    assert channel_intensity(rho_b, rho_b + rho_n) == pytest.approx(lambda_b)
    assert channel_intensity(rho_n, rho_b + rho_n) == pytest.approx(lambda_n)


def test_channel_intensity_rejects_a_nonstationary_total() -> None:
    """A total at or beyond the boundary has no adjustment-intensity reading."""
    with pytest.raises(TwoFieldEstimationError, match="stationarity region"):
        channel_intensity(0.5, 1.0)


def test_grid_rejects_a_feedback_range_outside_the_interior() -> None:
    """The profile refuses a grid that leaves the stationary interior."""
    with pytest.raises(TwoFieldEstimationError, match="stationary interior"):
        TwoFieldGrid.build(theta_points=3, rho_points=3, rho_upper=1.0)


def test_panel_statistics_reject_a_mismatched_field() -> None:
    """A field that does not match the panel cross-section is refused."""
    rng = np.random.default_rng(_SEED)
    returns = rng.normal(size=(10, 4))
    with pytest.raises(TwoFieldEstimationError, match="square and match"):
        two_field_panel_stats(returns, np.eye(4), np.eye(3))

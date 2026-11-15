"""Factor-profiled two-field spatial QMLE and boundary-null QLR calibration.

The module lifts Paper 5's nonnegative two-field likelihood to the excess-return
factor specification

    ``(I - rho_1 W_1 - rho_2 W_2) r_t^e = alpha + A f_t + epsilon_t``.

Both fixed fields enter unchanged.  Firm-specific intercepts and factor
loadings are profiled by residualizing the returns and both spatial-lag panels
against ``[1, f_t]``.  The resulting three-panel Gram matrix can then use the
same spatial-Jacobian and mixture-profile engine as the unadjusted two-field
model.

The nested tests impose either ``rho_2 = 0`` or ``rho_1 = 0``.  Their null law
is generated from the corresponding restricted fit, not from a chi-square
reference: standardized asset-level innovation blocks are resampled, fitted
variance is restored, the restricted spatial inverse maps them back to returns,
and the factor nuisance space is projected out anew in every simulated panel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np

from pipeline.stages.papers.paper5.estimate import NUMERICAL_FLOOR
from pipeline.stages.papers.paper5.two_field import (
    NestedQLRBootstrap,
    TwoFieldEstimationError,
    TwoFieldPanelStats,
    TwoFieldProblem,
    log_determinant_table,
    mixture_eigenvalues,
    quasi_likelihood_ratio,
    two_field_qmle,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pipeline.stages.papers.paper5.two_field import TwoFieldGrid

type Float = NDArray[np.float64]
type FloatInput = NDArray[np.floating[Any]]

_MATRIX_NDIM = 2
_INDEX_NDIM = 3
_MINIMUM_OBSERVATIONS = 2

FACTOR_TWO_FIELD_DRAW_COLUMNS = (
    "rho_w1",
    "rho_w2",
    "rho_total",
    "theta_w1",
    "loglik",
    "at_w1_boundary",
    "at_w2_boundary",
    "converged",
)


@dataclass(frozen=True, slots=True)
class _FactorProjection:
    """Factor-profiled panels and nuisance coefficients for one sample."""

    stats: TwoFieldPanelStats
    design: Float
    returns: Float
    lag_w1: Float
    lag_w2: Float
    coefficient_returns: Float
    coefficient_lag_w1: Float
    coefficient_lag_w2: Float


@dataclass(frozen=True, slots=True)
class FactorTwoFieldProblem:
    """A factor-profiled sample paired with a two-field likelihood grid."""

    core: TwoFieldProblem
    design: Float
    returns: Float
    lag_w1: Float
    lag_w2: Float
    factors: Float
    coefficient_returns: Float
    coefficient_lag_w1: Float
    coefficient_lag_w2: Float

    @property
    def n_obs(self) -> int:
        """Return the number of dates in the profiled panel."""
        return self.returns.shape[0]

    @property
    def n_assets(self) -> int:
        """Return the number of firms in the profiled panel."""
        return self.returns.shape[1]


@dataclass(frozen=True, slots=True)
class FactorTwoFieldFit:
    """Materialized joint fit after profiling firm-level factor coefficients."""

    rho_w1: float
    rho_w2: float
    rho_total: float
    theta_w1: float
    coefficients: Float
    sigma2_hat: float
    loglik: float
    converged: bool
    n_obs: int
    n_assets: int
    n_factors: int
    at_w1_boundary: bool
    at_w2_boundary: bool


def _raise(message: str) -> NoReturn:
    """Raise the shared two-field domain error."""
    raise TwoFieldEstimationError(message)


def _validate_grid(grid: TwoFieldGrid) -> None:
    """Require the nonnegative mixture region used by the incremental tests."""
    if (
        float(np.min(grid.theta)) < 0.0
        or float(np.max(grid.theta)) > 1.0
        or float(np.min(grid.rho)) < 0.0
    ):
        _raise("factor two-field QMLE requires a nonnegative mixture grid")


def _validate_components(
    returns: Float,
    lag_w1: Float,
    lag_w2: Float,
    factors: Float,
) -> None:
    """Reject panels that cannot share one firm-specific factor projection."""
    if returns.ndim != _MATRIX_NDIM or returns.shape[0] < _MINIMUM_OBSERVATIONS:
        _raise("factor two-field returns must be a nonempty date-by-asset panel")
    if lag_w1.shape != returns.shape or lag_w2.shape != returns.shape:
        _raise("factor two-field spatial lags must match the return panel")
    if factors.ndim != _MATRIX_NDIM or factors.shape[0] != returns.shape[0]:
        _raise("factor two-field factors must be date aligned with returns")
    if not all(
        bool(np.all(np.isfinite(values)))
        for values in (returns, lag_w1, lag_w2, factors)
    ):
        _raise("factor two-field inputs must be finite")


def _stats_from_residual_panels(
    returns: Float,
    lag_w1: Float,
    lag_w2: Float,
) -> TwoFieldPanelStats:
    """Build the core engine's per-date Gram blocks from profiled panels."""
    stacked = np.stack((returns, lag_w1, lag_w2), axis=-1)
    return TwoFieldPanelStats(
        gram=cast("Float", np.einsum("tna,tnb->tab", stacked, stacked)),
        totals=cast("Float", stacked.sum(axis=1)),
        n_assets=int(returns.shape[1]),
    )


def _factor_projection_from_components(
    returns: FloatInput,
    lag_w1: FloatInput,
    lag_w2: FloatInput,
    factors: FloatInput,
) -> _FactorProjection:
    """Profile one common factor design from returns and both spatial lags."""
    returns64 = np.asarray(returns, dtype=np.float64)
    lag_w1_64 = np.asarray(lag_w1, dtype=np.float64)
    lag_w2_64 = np.asarray(lag_w2, dtype=np.float64)
    factors64 = np.asarray(factors, dtype=np.float64)
    _validate_components(returns64, lag_w1_64, lag_w2_64, factors64)

    design = np.column_stack((np.ones(returns64.shape[0], dtype=np.float64), factors64))
    responses = np.concatenate((returns64, lag_w1_64, lag_w2_64), axis=1)
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design, responses, rcond=None
    )
    if rank < design.shape[1]:
        _raise("factor two-field design matrix is rank deficient")

    n_assets = returns64.shape[1]
    profiled = responses - design @ coefficients
    residual_returns = profiled[:, :n_assets]
    residual_lag_w1 = profiled[:, n_assets : 2 * n_assets]
    residual_lag_w2 = profiled[:, 2 * n_assets :]
    return _FactorProjection(
        stats=_stats_from_residual_panels(
            cast("Float", residual_returns),
            cast("Float", residual_lag_w1),
            cast("Float", residual_lag_w2),
        ),
        design=cast("Float", design),
        returns=cast("Float", returns64),
        lag_w1=cast("Float", lag_w1_64),
        lag_w2=cast("Float", lag_w2_64),
        coefficient_returns=cast("Float", coefficients[:, :n_assets]),
        coefficient_lag_w1=cast("Float", coefficients[:, n_assets : 2 * n_assets]),
        coefficient_lag_w2=cast("Float", coefficients[:, 2 * n_assets :]),
    )


def _core_problem(
    stats: TwoFieldPanelStats,
    w1: Float,
    w2: Float,
    grid: TwoFieldGrid,
) -> TwoFieldProblem:
    """Combine factor-profiled statistics with the fixed spatial Jacobian."""
    eigenvalues = mixture_eigenvalues(w1, w2, grid.theta)
    return TwoFieldProblem(
        stats=stats,
        w_b=w1,
        w_n=w2,
        grid=grid,
        eigenvalues=eigenvalues,
        log_determinant=log_determinant_table(eigenvalues, grid.rho),
    )


def build_factor_two_field_problem(
    excess_returns: FloatInput,
    w1: FloatInput,
    w2: FloatInput,
    factors: FloatInput,
    grid: TwoFieldGrid,
) -> FactorTwoFieldProblem:
    """Build a float64 factor-profiled W1/W2 likelihood problem."""
    _validate_grid(grid)
    returns64 = np.asarray(excess_returns, dtype=np.float64)
    w1_64 = np.asarray(w1, dtype=np.float64)
    w2_64 = np.asarray(w2, dtype=np.float64)
    factors64 = np.asarray(factors, dtype=np.float64)
    if returns64.ndim != _MATRIX_NDIM:
        _raise("factor two-field returns must be two dimensional")
    n_assets = returns64.shape[1]
    if w1_64.shape != (n_assets, n_assets) or w2_64.shape != (n_assets, n_assets):
        _raise("factor two-field operators must match the asset dimension")
    if not bool(np.all(np.isfinite(w1_64))) or not bool(np.all(np.isfinite(w2_64))):
        _raise("factor two-field operators must be finite")

    lag_w1 = returns64 @ w1_64.T
    lag_w2 = returns64 @ w2_64.T
    projection = _factor_projection_from_components(
        returns64, lag_w1, lag_w2, factors64
    )
    return FactorTwoFieldProblem(
        core=_core_problem(projection.stats, w1_64, w2_64, grid),
        design=projection.design,
        returns=projection.returns,
        lag_w1=projection.lag_w1,
        lag_w2=projection.lag_w2,
        factors=cast("Float", factors64),
        coefficient_returns=projection.coefficient_returns,
        coefficient_lag_w1=projection.coefficient_lag_w1,
        coefficient_lag_w2=projection.coefficient_lag_w2,
    )


def factor_two_field_with_grid(
    problem: FactorTwoFieldProblem,
    grid: TwoFieldGrid,
) -> FactorTwoFieldProblem:
    """Reuse one factor projection under a joint or pinned mixture grid."""
    _validate_grid(grid)
    return replace(
        problem,
        core=_core_problem(
            problem.core.stats,
            problem.core.w_b,
            problem.core.w_n,
            grid,
        ),
    )


def factor_two_field_qmle(
    problem: FactorTwoFieldProblem,
    *,
    refine: bool = True,
) -> FactorTwoFieldFit:
    """Fit both channel coefficients and materialize factor nuisances."""
    fit = two_field_qmle(problem.core, refine=refine)
    rho_w1 = float(fit["rho_b"])
    rho_w2 = float(fit["rho_n"])
    coefficients = (
        problem.coefficient_returns
        - rho_w1 * problem.coefficient_lag_w1
        - rho_w2 * problem.coefficient_lag_w2
    )
    return FactorTwoFieldFit(
        rho_w1=rho_w1,
        rho_w2=rho_w2,
        rho_total=float(fit["rho_total"]),
        theta_w1=float(fit["theta"]),
        coefficients=cast("Float", np.asarray(coefficients, dtype=np.float64)),
        sigma2_hat=float(fit["sigma2_hat"]),
        loglik=float(fit["loglik"]),
        converged=bool(fit["converged"]),
        n_obs=int(fit["t_obs"]),
        n_assets=int(fit["n_assets"]),
        n_factors=problem.factors.shape[1],
        at_w1_boundary=bool(fit["at_field_b_boundary"]),
        at_w2_boundary=bool(fit["at_field_n_boundary"]),
    )


def factor_two_field_daily_logscore(
    fit: FactorTwoFieldFit,
    excess_returns: FloatInput,
    w1: FloatInput,
    w2: FloatInput,
    factors: FloatInput,
) -> Float:
    """Score held-out dates under a fixed factor-conditioned two-field fit."""
    returns64 = np.asarray(excess_returns, dtype=np.float64)
    w1_64 = np.asarray(w1, dtype=np.float64)
    w2_64 = np.asarray(w2, dtype=np.float64)
    factors64 = np.asarray(factors, dtype=np.float64)
    _validate_components(
        returns64,
        returns64 @ w1_64.T,
        returns64 @ w2_64.T,
        factors64,
    )
    design = np.column_stack((np.ones(returns64.shape[0], dtype=np.float64), factors64))
    filtered = (
        returns64
        - fit.rho_w1 * (returns64 @ w1_64.T)
        - fit.rho_w2 * (returns64 @ w2_64.T)
    )
    residual = filtered - design @ fit.coefficients
    operator = (
        np.eye(fit.n_assets, dtype=np.float64) - fit.rho_w1 * w1_64 - fit.rho_w2 * w2_64
    )
    sign, logdet = np.linalg.slogdet(operator)
    if sign <= 0.0 or not np.isfinite(logdet):
        _raise("factor two-field held-out operator has invalid log determinant")
    variance = max(float(fit.sigma2_hat), NUMERICAL_FLOOR)
    scores = float(logdet) - 0.5 * (
        fit.n_assets * np.log(2.0 * np.pi * variance)
        + np.sum(np.square(residual), axis=1) / variance
    )
    return cast("Float", np.asarray(scores, dtype=np.float64))


def _resampled_stats(
    problem: FactorTwoFieldProblem,
    date_index: NDArray[np.int64],
) -> TwoFieldPanelStats:
    """Refit the factor projection after selecting one joint-date draw."""
    projection = _factor_projection_from_components(
        problem.returns[date_index],
        problem.lag_w1[date_index],
        problem.lag_w2[date_index],
        problem.factors[date_index],
    )
    return projection.stats


def factor_two_field_problem_at(
    problem: FactorTwoFieldProblem,
    date_index: NDArray[np.int64],
) -> FactorTwoFieldProblem:
    """Reprofile one factor two-field problem on a resampled date draw.

    The grid, both operators, and the panel geometry are carried over; only the
    factor projection and the profiled sufficient statistics are recomputed, so
    a refitted draw is the same estimator applied to the resampled dates.
    """
    if date_index.ndim != 1:
        _raise("factor two-field date draw must be one dimensional")
    projection = _factor_projection_from_components(
        problem.returns[date_index],
        problem.lag_w1[date_index],
        problem.lag_w2[date_index],
        problem.factors[date_index],
    )
    return replace(
        problem,
        core=replace(problem.core, stats=projection.stats),
        design=projection.design,
        returns=projection.returns,
        lag_w1=projection.lag_w1,
        lag_w2=projection.lag_w2,
        factors=cast("Float", problem.factors[date_index]),
        coefficient_returns=projection.coefficient_returns,
        coefficient_lag_w1=projection.coefficient_lag_w1,
        coefficient_lag_w2=projection.coefficient_lag_w2,
    )


def factor_two_field_bootstrap(
    problem: FactorTwoFieldProblem,
    indices: NDArray[np.int64],
    *,
    refine: bool = True,
) -> Float:
    """Refit factors and both spatial coefficients on each joint-date draw."""
    if indices.ndim != _MATRIX_NDIM or indices.shape[1] != problem.n_obs:
        _raise("factor two-field bootstrap indices must cover the full date panel")
    draws = np.empty(
        (indices.shape[0], len(FACTOR_TWO_FIELD_DRAW_COLUMNS)), dtype=np.float64
    )
    for draw, date_index in enumerate(indices):
        core = replace(problem.core, stats=_resampled_stats(problem, date_index))
        fit = two_field_qmle(core, refine=refine)
        draws[draw] = (
            fit["rho_b"],
            fit["rho_n"],
            fit["rho_total"],
            fit["theta"],
            fit["loglik"],
            fit["at_field_b_boundary"],
            fit["at_field_n_boundary"],
            fit["converged"],
        )
    return cast("Float", draws)


def _restricted_factor_residuals(
    problem: FactorTwoFieldProblem,
    fit: FactorTwoFieldFit,
    theta_null: float,
) -> Float:
    """Recover centered firm innovations from one pinned factor fit."""
    weights = theta_null * problem.core.w_b + (1.0 - theta_null) * problem.core.w_n
    filtered = problem.returns - fit.rho_total * (problem.returns @ weights.T)
    residuals = filtered - problem.design @ fit.coefficients
    residuals = residuals - np.mean(residuals, axis=0, keepdims=True)
    return cast("Float", residuals)


def _standardize_factor_residuals(residuals: Float) -> Float:
    """Standardize centered firm innovations for a common-variance null."""
    scale = np.sqrt(np.maximum(np.mean(np.square(residuals), axis=0), NUMERICAL_FLOOR))
    return cast("Float", residuals / scale)


def _null_profile_stats(
    innovations: Float,
    factor_basis: Float,
    spatial_multiplier: Float,
    w1: Float,
    w2: Float,
) -> TwoFieldPanelStats:
    """Profile nuisances from one restricted-null innovation panel.

    The restricted conditional mean lies in the factor-design column space.
    Projecting the resampled innovations against that space and then applying
    the spatial inverse is therefore algebraically identical to generating the
    full null return panel and re-estimating every firm coefficient.
    """
    factor_residual = innovations - factor_basis @ (factor_basis.T @ innovations)
    residual_returns = factor_residual @ spatial_multiplier.T
    return _stats_from_residual_panels(
        cast("Float", residual_returns),
        cast("Float", residual_returns @ w1.T),
        cast("Float", residual_returns @ w2.T),
    )


def factor_nested_qlr_bootstrap(
    joint_problem: FactorTwoFieldProblem,
    restricted_problem: FactorTwoFieldProblem,
    indices: NDArray[np.int64],
) -> NestedQLRBootstrap:
    """Calibrate one factor-conditioned channel QLR under its boundary null."""
    if restricted_problem.core.grid.theta.size != 1:
        _raise("the factor-conditioned nested QLR null must pin mixture weight")
    if (
        indices.ndim != _INDEX_NDIM
        or indices.shape[1] != joint_problem.n_assets
        or indices.shape[2] != joint_problem.n_obs
    ):
        _raise("factor-conditioned QLR indices must provide one path per asset")
    if not (
        np.array_equal(joint_problem.returns, restricted_problem.returns)
        and np.array_equal(joint_problem.factors, restricted_problem.factors)
        and np.array_equal(joint_problem.core.w_b, restricted_problem.core.w_b)
        and np.array_equal(joint_problem.core.w_n, restricted_problem.core.w_n)
    ):
        _raise("joint and restricted factor QLR problems must share data and fields")

    unrestricted = factor_two_field_qmle(joint_problem)
    restricted = factor_two_field_qmle(restricted_problem)
    if not unrestricted.converged or not restricted.converged:
        _raise("factor-conditioned nested QLR observed fits did not converge")
    statistic = quasi_likelihood_ratio(unrestricted.loglik, restricted.loglik)
    theta_null = float(restricted_problem.core.grid.theta[0])
    weights = (
        theta_null * restricted_problem.core.w_b
        + (1.0 - theta_null) * restricted_problem.core.w_n
    )
    residuals = _standardize_factor_residuals(
        _restricted_factor_residuals(restricted_problem, restricted, theta_null)
    )
    operator = (
        np.eye(weights.shape[0], dtype=np.float64) - restricted.rho_total * weights
    )
    multiplier = cast("Float", np.asarray(np.linalg.inv(operator), dtype=np.float64))
    factor_basis = cast(
        "Float", np.asarray(np.linalg.qr(joint_problem.design, mode="reduced")[0])
    )

    draws = np.empty(indices.shape[0], dtype=np.float64)
    for draw in range(indices.shape[0]):
        asset_date_index = np.asarray(indices[draw].T, dtype=np.int64)
        innovations = cast(
            "Float", np.take_along_axis(residuals, asset_date_index, axis=0)
        )
        stats = _null_profile_stats(
            innovations * np.sqrt(max(restricted.sigma2_hat, NUMERICAL_FLOOR)),
            factor_basis,
            multiplier,
            joint_problem.core.w_b,
            joint_problem.core.w_n,
        )
        unrestricted_draw = two_field_qmle(replace(joint_problem.core, stats=stats))
        restricted_draw = two_field_qmle(replace(restricted_problem.core, stats=stats))
        if not bool(unrestricted_draw["converged"]) or not bool(
            restricted_draw["converged"]
        ):
            _raise(f"factor-conditioned nested QLR null draw {draw} did not converge")
        draws[draw] = quasi_likelihood_ratio(
            unrestricted_draw["loglik"], restricted_draw["loglik"]
        )

    exceedances = int(np.count_nonzero(draws >= statistic))
    return NestedQLRBootstrap(
        statistic=statistic,
        p_value=float((1 + exceedances) / (draws.size + 1)),
        exceedances=exceedances,
        draws=draws,
    )


__all__ = [
    "FACTOR_TWO_FIELD_DRAW_COLUMNS",
    "FactorTwoFieldFit",
    "FactorTwoFieldProblem",
    "build_factor_two_field_problem",
    "factor_nested_qlr_bootstrap",
    "factor_two_field_bootstrap",
    "factor_two_field_daily_logscore",
    "factor_two_field_problem_at",
    "factor_two_field_qmle",
    "factor_two_field_with_grid",
]

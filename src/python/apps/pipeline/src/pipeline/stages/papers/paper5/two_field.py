"""Two-field spatial QMLE over a pair of frozen interaction fields.

The exposure model prices peer misalignment against two admissible interaction
fields at nonnegative intensities ``lambda_b`` and ``lambda_n``. Because both
fields are row stochastic with a zero diagonal, their convex mixture

    ``W(theta) = theta * W_b + (1 - theta) * W_n``

is again admissible and the two-field spatial operator factors exactly,

    ``I - rho_b W_b - rho_n W_n = I - rho_total W(theta)``,

with ``rho_total = rho_b + rho_n`` and ``theta = rho_b / rho_total``. The
concentrated quasi-likelihood of :func:`pipeline.stages.papers.paper5.estimate.sar_qmle`
therefore applies verbatim at the mixture, and the separate single-field fits
are the ``theta = 1`` and ``theta = 0`` boundaries of one nested family.

Two facts make a two-dimensional profile cheap enough for 2,000 dependent-date
bootstrap refits. First, the concentrated residual sum of squares is a quadratic
form in ``(1, -rho_b, -rho_n)`` against the 3x3 Gram matrix of the stacked
panels ``r``, ``W_b r`` and ``W_n r``; that Gram matrix and the matching column
sums are additive over dates, so a joint-date resample is a sum of precomputed
per-date blocks rather than a fresh pass over the panel. Second, the spatial
Jacobian ``log|det(I - rho_total W(theta))|`` depends only on the parameters and
never on the returns, so it is tabulated once for the whole stage.

``estimate.sar_qmle`` is deliberately left untouched: it produces the published
single-field values, and this module is reconciled against it by test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar

from pipeline.stages.papers.paper5.estimate import NUMERICAL_FLOOR

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MATRIX_NDIM = 2
_PANEL_NDIM = 2
_MIN_THETA_POINTS = 1
_MIN_RHO_POINTS = 2
# Interior margin on the total feedback, matching the clip in ``estimate.sar_qmle``.
_RHO_INTERIOR_EPS = 1e-3
# Bracketing tolerance for the polish step, two orders of magnitude finer than any
# grid this stage runs, so a reported coefficient is not a grid node.
_REFINE_TOLERANCE = 1e-6
# Draw-level columns: the two channel coefficients, the mixture weight, and the
# two grid-boundary indicators.
_DRAW_COLUMNS = 5


class TwoFieldEstimationError(ValueError):
    """Report an invalid two-field configuration or an unusable fit."""


def _raise(message: str) -> None:
    """Raise the module's domain error without embedding literals at call sites."""
    raise TwoFieldEstimationError(message)


@dataclass(frozen=True)
class TwoFieldGrid:
    """Profile grid for the mixture weight and the total spatial feedback."""

    theta: NDArray[np.float64]
    rho: NDArray[np.float64]

    @classmethod
    def build(
        cls,
        *,
        theta_points: int,
        rho_points: int,
        theta_lower: float = 0.0,
        theta_upper: float = 1.0,
        rho_lower: float = 0.0,
        rho_upper: float = 1.0 - _RHO_INTERIOR_EPS,
    ) -> TwoFieldGrid:
        """Build an evenly spaced profile grid.

        Args:
            theta_points: Mixture-weight grid points; one pins a single field.
            rho_points: Total-feedback grid points.
            theta_lower: Smallest mixture weight retained.
            theta_upper: Largest mixture weight retained.
            rho_lower: Smallest total feedback retained.
            rho_upper: Largest total feedback retained.

        Returns:
            The profile grid.

        Raises:
            TwoFieldEstimationError: If either range is degenerate or the
                feedback range leaves the stationary interior.

        """
        if theta_points < _MIN_THETA_POINTS or rho_points < _MIN_RHO_POINTS:
            _raise("the profile needs one mixture point and two feedback points")
        if not theta_lower <= theta_upper:
            _raise("the mixture-weight range must be nondecreasing")
        if not -1.0 < rho_lower < rho_upper < 1.0:
            _raise("the total-feedback grid must lie inside the stationary interior")
        return cls(
            theta=np.linspace(theta_lower, theta_upper, theta_points, dtype=np.float64),
            rho=np.linspace(rho_lower, rho_upper, rho_points, dtype=np.float64),
        )


@dataclass(frozen=True)
class TwoFieldPanelStats:
    """Per-date sufficient statistics for the concentrated two-field likelihood.

    ``gram`` holds each date's raw Gram matrix of the stacked cross-sections
    ``(r_t, W_b r_t, W_n r_t)`` and ``totals`` their raw column sums. Both are
    additive over dates, so a joint-date resample is a sum of blocks.
    """

    gram: NDArray[np.float64]
    totals: NDArray[np.float64]
    n_assets: int


@dataclass(frozen=True)
class TwoFieldProblem:
    """Everything a two-field fit needs that does not depend on the resample."""

    stats: TwoFieldPanelStats
    w_b: NDArray[np.float64]
    w_n: NDArray[np.float64]
    grid: TwoFieldGrid
    eigenvalues: NDArray[np.complex128]
    log_determinant: NDArray[np.float64]


@dataclass(frozen=True)
class NestedQLRBootstrap:
    """Null-imposed bootstrap calibration for one nested field comparison."""

    statistic: float
    p_value: float
    exceedances: int
    draws: NDArray[np.float64]


@dataclass(frozen=True)
class _Aggregate:
    """Demeaned Gram matrix and column sums for one set of dates."""

    gram: NDArray[np.float64]
    column: NDArray[np.float64]
    t_obs: int
    n_assets: int

    @property
    def observations(self) -> int:
        """Return the number of scalar residuals behind the aggregate."""
        return self.t_obs * self.n_assets


def _validate_fields(
    returns: NDArray[np.float64],
    w_b: NDArray[np.float64],
    w_n: NDArray[np.float64],
) -> None:
    """Reject a panel or field pair that cannot define a two-field profile."""
    if returns.ndim != _PANEL_NDIM or returns.shape[0] < 1:
        _raise("the return panel must be a nonempty date-by-asset matrix")
    n_assets = returns.shape[1]
    for field in (w_b, w_n):
        if field.ndim != _MATRIX_NDIM or field.shape != (n_assets, n_assets):
            _raise("both interaction fields must be square and match the panel")
    if not (
        np.all(np.isfinite(returns))
        and np.all(np.isfinite(w_b))
        and np.all(np.isfinite(w_n))
    ):
        _raise("the panel and both interaction fields must be finite")


def two_field_panel_stats(
    returns: NDArray[np.float64],
    w_b: NDArray[np.float64],
    w_n: NDArray[np.float64],
) -> TwoFieldPanelStats:
    """Reduce a return panel and two interaction fields to per-date statistics.

    Args:
        returns: Return panel, shape ``(T, n)``.
        w_b: First interaction field, shape ``(n, n)``.
        w_n: Second interaction field, shape ``(n, n)``.

    Returns:
        Per-date Gram blocks and column sums.

    """
    _validate_fields(returns, w_b, w_n)
    stacks = np.stack([returns, returns @ w_b.T, returns @ w_n.T], axis=-1)
    return TwoFieldPanelStats(
        gram=np.einsum("tna,tnb->tab", stacks, stacks),
        totals=stacks.sum(axis=1),
        n_assets=int(returns.shape[1]),
    )


def mixture_eigenvalues(
    w_b: NDArray[np.float64],
    w_n: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return the eigenvalues of every grid mixture of the two fields.

    Args:
        w_b: First interaction field.
        w_n: Second interaction field.
        theta_grid: Mixture weights.

    Returns:
        Complex eigenvalues, shape ``(len(theta_grid), n)``.

    """
    return np.stack([_mixture_spectrum(w_b, w_n, float(theta)) for theta in theta_grid])


def _mixture_spectrum(
    w_b: NDArray[np.float64],
    w_n: NDArray[np.float64],
    theta: float,
) -> NDArray[np.complex128]:
    """Return one mixture's spectrum; a real asymmetric field may be complex."""
    return np.asarray(
        np.linalg.eigvals(theta * w_b + (1.0 - theta) * w_n),
        dtype=np.complex128,
    )


def log_determinant_table(
    eigenvalues: NDArray[np.complex128],
    rho_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Tabulate ``log|det(I - rho W(theta))|`` over the profile grid.

    A real asymmetric mixture may have complex-conjugate eigenvalue pairs, so
    the imaginary parts stay inside the modulus exactly as in the single-field
    Jacobian.

    Args:
        eigenvalues: Mixture eigenvalues, shape ``(K, n)``.
        rho_grid: Total-feedback grid, shape ``(R,)``.

    Returns:
        Log-determinant table, shape ``(K, R)``.

    """
    table = np.empty((eigenvalues.shape[0], rho_grid.size), dtype=np.float64)
    with np.errstate(divide="ignore"):
        for index, values in enumerate(eigenvalues):
            table[index] = np.sum(
                np.log(np.abs(1.0 - np.outer(rho_grid, values))), axis=1
            )
    return table


def build_two_field_problem(
    returns: NDArray[np.float64],
    w_b: NDArray[np.float64],
    w_n: NDArray[np.float64],
    grid: TwoFieldGrid,
) -> TwoFieldProblem:
    """Precompute every resample-independent input to the two-field profile.

    Args:
        returns: Return panel, shape ``(T, n)``.
        w_b: First interaction field.
        w_n: Second interaction field.
        grid: Profile grid.

    Returns:
        The assembled problem.

    """
    eigenvalues = mixture_eigenvalues(w_b, w_n, grid.theta)
    return TwoFieldProblem(
        stats=two_field_panel_stats(returns, w_b, w_n),
        w_b=w_b,
        w_n=w_n,
        grid=grid,
        eigenvalues=eigenvalues,
        log_determinant=log_determinant_table(eigenvalues, grid.rho),
    )


def _aggregate(
    stats: TwoFieldPanelStats,
    date_index: NDArray[np.int64] | None,
) -> _Aggregate:
    """Sum per-date statistics and remove the concentrated common intercept."""
    gram = stats.gram if date_index is None else stats.gram[date_index]
    totals = stats.totals if date_index is None else stats.totals[date_index]
    t_obs = int(gram.shape[0])
    observations = t_obs * stats.n_assets
    column = totals.sum(axis=0)
    return _Aggregate(
        gram=gram.sum(axis=0) - np.outer(column, column) / observations,
        column=column,
        t_obs=t_obs,
        n_assets=stats.n_assets,
    )


def _residual_sum_of_squares(
    gram: NDArray[np.float64],
    theta: float,
    rho: float,
) -> float:
    """Evaluate the concentrated residual sum of squares at one profile point."""
    coefficients = np.array([1.0, -rho * theta, -rho * (1.0 - theta)], dtype=np.float64)
    return float(coefficients @ gram @ coefficients)


def _loglik_from_rss(
    rss: NDArray[np.float64] | float,
    observations: int,
) -> NDArray[np.float64] | float:
    """Concentrate ``sigma^2`` out of the Gaussian working log-likelihood."""
    sigma2 = np.maximum(np.asarray(rss) / observations, NUMERICAL_FLOOR)
    return -0.5 * observations * np.log(2.0 * np.pi * sigma2) - rss / (2.0 * sigma2)


def _profile_loglik(
    aggregate: _Aggregate,
    problem: TwoFieldProblem,
) -> NDArray[np.float64]:
    """Evaluate the concentrated quasi-log-likelihood on the whole profile grid."""
    gram = aggregate.gram
    theta = problem.grid.theta
    rho = problem.grid.rho
    cross = theta * gram[0, 1] + (1.0 - theta) * gram[0, 2]
    quadratic = (
        theta**2 * gram[1, 1]
        + 2.0 * theta * (1.0 - theta) * gram[1, 2]
        + (1.0 - theta) ** 2 * gram[2, 2]
    )
    rss = gram[0, 0] - 2.0 * np.outer(cross, rho) + np.outer(quadratic, rho**2)
    return _loglik_from_rss(rss, aggregate.observations) + (
        aggregate.t_obs * problem.log_determinant
    )


def spectrum_log_determinant(
    eigenvalues: NDArray[np.complex128],
    rho: float,
) -> float:
    """Return ``log|det(I - rho W)|`` from the spectrum of ``W``."""
    with np.errstate(divide="ignore"):
        return float(np.sum(np.log(np.abs(1.0 - rho * eigenvalues))))


def matrix_log_determinant(
    w_b: NDArray[np.float64],
    w_n: NDArray[np.float64],
    rho_b: float,
    rho_n: float,
) -> float:
    """Return ``log|det(I - rho_b W_b - rho_n W_n)|`` by LU factorization.

    At a single point this is an order of magnitude cheaper than an
    eigendecomposition, which is what makes polishing affordable inside every
    bootstrap refit. The tabulated grid path still uses the spectrum; the two
    routes agree to floating-point tolerance and a test holds them together.

    Args:
        w_b: First interaction field.
        w_n: Second interaction field.
        rho_b: First channel coefficient.
        rho_n: Second channel coefficient.

    Returns:
        The log absolute determinant, negative infinity at a singular operator.

    """
    operator = np.eye(w_b.shape[0], dtype=np.float64) - rho_b * w_b - rho_n * w_n
    _sign, log_absolute = np.linalg.slogdet(operator)
    # A singular operator yields negative infinity here, exactly as the spectral
    # route does when a factor vanishes; the caller treats it as inadmissible.
    return float(log_absolute)


def _point_loglik(
    aggregate: _Aggregate,
    problem: TwoFieldProblem,
    theta: float,
    rho: float,
) -> float:
    """Evaluate the concentrated quasi-log-likelihood at one profile point."""
    rss = _residual_sum_of_squares(aggregate.gram, theta, rho)
    log_determinant = matrix_log_determinant(
        problem.w_b, problem.w_n, rho * theta, rho * (1.0 - theta)
    )
    return float(
        _loglik_from_rss(rss, aggregate.observations)
        + aggregate.t_obs * log_determinant
    )


def _bracket(values: NDArray[np.float64], index: int) -> tuple[float, float]:
    """Return the grid cell around one index, collapsing on a singleton axis."""
    lower = float(values[max(index - 1, 0)])
    upper = float(values[min(index + 1, values.size - 1)])
    return lower, upper


def _refine(
    aggregate: _Aggregate,
    problem: TwoFieldProblem,
    theta_index: int,
    rho_index: int,
    tolerance: float = _REFINE_TOLERANCE,
) -> tuple[float, float, float, bool]:
    """Polish the grid maximum inside its own cell on the exact objective."""
    theta_lower, theta_upper = _bracket(problem.grid.theta, theta_index)
    rho_lower, rho_upper = _bracket(problem.grid.rho, rho_index)

    def fit_rho(theta: float) -> tuple[float, float]:
        result = minimize_scalar(
            lambda rho: -_point_loglik(aggregate, problem, theta, rho),
            bounds=(rho_lower, rho_upper),
            method="bounded",
            options={"xatol": tolerance},
        )
        return float(result.x), float(-result.fun)

    if theta_lower >= theta_upper:
        rho_hat, loglik = fit_rho(theta_lower)
        return theta_lower, rho_hat, loglik, True
    outer = minimize_scalar(
        lambda theta: -fit_rho(theta)[1],
        bounds=(theta_lower, theta_upper),
        method="bounded",
        options={"xatol": tolerance},
    )
    theta_hat = float(outer.x)
    rho_hat, loglik = fit_rho(theta_hat)
    return theta_hat, rho_hat, loglik, bool(outer.success)


def two_field_qmle(
    problem: TwoFieldProblem,
    *,
    date_index: NDArray[np.int64] | None = None,
    refine: bool = True,
) -> dict[str, float]:
    """Fit the two-field concentrated QMLE by profiling over the grid.

    The grid maximum is polished inside its own cell on the exact objective. The
    polished point is accepted only when it does not lose likelihood against the
    grid, so a joint fit can never report a worse fit than a nested boundary
    evaluated on the same grid.

    The two boundary indicators read the grid argmax rather than the polished
    weight. Bounded polishing approaches a bracket endpoint without reaching it,
    so testing the polished weight against the endpoint would silently report a
    boundary optimum as interior.

    Args:
        problem: Precomputed resample-independent inputs.
        date_index: Dates to fit; ``None`` uses the whole panel.
        refine: Whether to polish the grid maximum on the exact objective.

    Returns:
        Dict with ``theta``, ``rho_total``, ``rho_b``, ``rho_n``, ``loglik``,
        ``alpha_hat``, ``sigma2_hat``, ``t_obs``, ``n_assets``, ``converged``,
        and the two mixture-boundary indicators.

    """
    aggregate = _aggregate(problem.stats, date_index)
    profile = _profile_loglik(aggregate, problem)
    theta_index, rho_index = np.unravel_index(int(np.argmax(profile)), profile.shape)
    theta = float(problem.grid.theta[theta_index])
    rho_total = float(problem.grid.rho[rho_index])
    loglik = float(profile[theta_index, rho_index])
    converged = True
    if refine:
        theta_hat, rho_hat, refined, converged = _refine(
            aggregate, problem, int(theta_index), int(rho_index)
        )
        if refined >= loglik:
            theta, rho_total, loglik = theta_hat, rho_hat, refined

    rho_b = rho_total * theta
    rho_n = rho_total * (1.0 - theta)
    coefficients = np.array([1.0, -rho_b, -rho_n], dtype=np.float64)
    alpha_hat = float(aggregate.column @ coefficients) / aggregate.observations
    sigma2_hat = max(
        _residual_sum_of_squares(aggregate.gram, theta, rho_total)
        / aggregate.observations,
        NUMERICAL_FLOOR,
    )
    return {
        "theta": theta,
        "rho_total": rho_total,
        "rho_b": rho_b,
        "rho_n": rho_n,
        "loglik": loglik,
        "alpha_hat": alpha_hat,
        "sigma2_hat": sigma2_hat,
        "t_obs": float(aggregate.t_obs),
        "n_assets": float(aggregate.n_assets),
        "converged": float(converged),
        "at_field_b_boundary": float(int(theta_index) == problem.grid.theta.size - 1),
        "at_field_n_boundary": float(int(theta_index) == 0),
    }


def two_field_profile_surface(
    problem: TwoFieldProblem,
    *,
    date_index: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the profiled log-likelihood in channel-coefficient coordinates.

    The grid is regular in ``(theta, rho_total)`` and therefore scattered once
    mapped to ``(rho_b, rho_n)``; consumers should contour it as scattered data
    rather than assume a rectangular mesh.

    Args:
        problem: Precomputed resample-independent inputs.
        date_index: Dates to profile; ``None`` uses the whole panel.

    Returns:
        Flattened ``(rho_b, rho_n, loglik)`` over the profile grid.

    """
    aggregate = _aggregate(problem.stats, date_index)
    profile = _profile_loglik(aggregate, problem)
    theta = problem.grid.theta[:, None]
    rho = problem.grid.rho[None, :]
    return (
        (rho * theta).reshape(-1),
        (rho * (1.0 - theta)).reshape(-1),
        profile.reshape(-1),
    )


def two_field_bootstrap(
    problem: TwoFieldProblem,
    indices: NDArray[np.int64],
    *,
    refine: bool = True,
) -> NDArray[np.float64]:
    """Refit the profile on every joint-date bootstrap draw.

    Refits are polished by default. Without it every resampled coefficient is a
    grid node, which quantizes the reported intervals at the grid spacing; the
    LU-based Jacobian keeps polishing cheap enough to do on all draws.

    Args:
        problem: Precomputed resample-independent inputs.
        indices: Joint-date resample indices, shape ``(draws, T)``.
        refine: Whether to polish each refit inside its own grid cell.

    Returns:
        Draw-level ``(rho_b, rho_n, theta, at_b_boundary, at_n_boundary)``,
        shape ``(draws, 5)``.

    """
    draws = np.empty((indices.shape[0], _DRAW_COLUMNS), dtype=np.float64)
    for draw, date_index in enumerate(indices):
        fit = two_field_qmle(problem, date_index=date_index, refine=refine)
        draws[draw] = (
            fit["rho_b"],
            fit["rho_n"],
            fit["theta"],
            fit["at_field_b_boundary"],
            fit["at_field_n_boundary"],
        )
    return draws


def quasi_likelihood_ratio(
    unrestricted_loglik: float,
    restricted_loglik: float,
) -> float:
    """Return twice the nonnegative gain of a nested quasi-likelihood fit.

    The unrestricted family contains the restricted boundary, so a negative
    numerical difference is truncation or optimizer noise rather than evidence
    against nesting. Truncating at zero preserves that geometry; taking an
    absolute value would turn a failed nested fit into positive evidence.
    """
    return 2.0 * max(unrestricted_loglik - restricted_loglik, 0.0)


def _restricted_residuals(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    rho: float,
    alpha: float,
) -> NDArray[np.float64]:
    """Recover and recenter full cross-sectional innovations under one null."""
    residuals = returns - rho * (returns @ weights.T) - alpha
    return np.asarray(residuals - np.mean(residuals), dtype=np.float64)


def _null_panel(
    residuals: NDArray[np.float64],
    multiplier: NDArray[np.float64],
    *,
    alpha: float,
    asset_date_index: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Generate one restricted-null panel from asset-wise residual blocks."""
    if asset_date_index.shape != residuals.shape:
        _raise("nested QLR asset-date indices must match the residual panel")
    innovations = np.take_along_axis(residuals, asset_date_index, axis=0)
    return np.asarray((alpha + innovations) @ multiplier.T, dtype=np.float64)


def _assetwise_date_index(
    indices: NDArray[np.int64],
    *,
    draw: int,
    n_assets: int,
) -> NDArray[np.int64]:
    """Assign a different existing stationary-bootstrap path to each asset.

    Cycling through the shared draw bank uses every precomputed path equally
    often, preserves each asset's temporal blocks, and removes the excluded
    field's contemporaneous residual alignment under the imposed null.
    """
    draw_rows = (draw + np.arange(n_assets, dtype=np.int64)) % indices.shape[0]
    return np.asarray(indices[draw_rows].T, dtype=np.int64)


def nested_qlr_bootstrap(
    returns: NDArray[np.float64],
    joint_problem: TwoFieldProblem,
    restricted_problem: TwoFieldProblem,
    indices: NDArray[np.int64],
) -> NestedQLRBootstrap:
    """Calibrate one nested-channel QLR by a restricted residual bootstrap.

    The restricted problem must pin the mixture weight to one value. Each
    asset's fitted innovations are resampled on a different path from the shared
    stationary-bootstrap draw bank. This preserves asset-level temporal blocks
    while removing the excluded field's contemporaneous residual alignment,
    which would otherwise carry the tested channel into every null panel. Every
    null panel is then fit by the same joint and pinned objectives used for the
    observed statistic, so the draw distribution absorbs the nonnegative-channel
    boundary without imposing a chi-square reference.

    Args:
        returns: Original return panel, shape ``(T, n)``.
        joint_problem: Unrestricted two-field problem.
        restricted_problem: Same field pair with a singleton theta grid.
        indices: Stationary-bootstrap date indices, shape ``(draws, T)``.

    Returns:
        Observed QLR statistic, finite-sample p-value, exceedance count, and
        null draw distribution.

    Raises:
        TwoFieldEstimationError: If the restricted problem is not pinned or the
            resample shape does not match the panel.

    """
    if restricted_problem.grid.theta.size != 1:
        _raise("the nested QLR null must pin the mixture weight")
    if indices.ndim != _PANEL_NDIM or indices.shape[1] != returns.shape[0]:
        _raise("nested QLR indices must have one full date panel per draw")

    restricted = two_field_qmle(restricted_problem)
    unrestricted = two_field_qmle(joint_problem)
    statistic = quasi_likelihood_ratio(unrestricted["loglik"], restricted["loglik"])

    theta_null = float(restricted_problem.grid.theta[0])
    weights = (
        theta_null * restricted_problem.w_b
        + (1.0 - theta_null) * restricted_problem.w_n
    )
    rho_null = restricted["rho_total"]
    alpha_null = restricted["alpha_hat"]
    residuals = _restricted_residuals(
        returns,
        weights,
        rho=rho_null,
        alpha=alpha_null,
    )
    operator = np.eye(weights.shape[0], dtype=np.float64) - rho_null * weights
    multiplier = np.asarray(np.linalg.inv(operator), dtype=np.float64)

    draws = np.empty(indices.shape[0], dtype=np.float64)
    for draw in range(indices.shape[0]):
        null_returns = _null_panel(
            residuals,
            multiplier,
            alpha=alpha_null,
            asset_date_index=_assetwise_date_index(
                indices,
                draw=draw,
                n_assets=returns.shape[1],
            ),
        )
        stats = two_field_panel_stats(
            null_returns,
            joint_problem.w_b,
            joint_problem.w_n,
        )
        unrestricted_draw = two_field_qmle(replace(joint_problem, stats=stats))
        restricted_draw = two_field_qmle(replace(restricted_problem, stats=stats))
        draws[draw] = quasi_likelihood_ratio(
            unrestricted_draw["loglik"], restricted_draw["loglik"]
        )

    exceedances = int(np.count_nonzero(draws >= statistic))
    p_value = float((1 + exceedances) / (draws.size + 1))
    return NestedQLRBootstrap(
        statistic=statistic,
        p_value=p_value,
        exceedances=exceedances,
        draws=draws,
    )


def channel_intensity(rho_channel: float, rho_total: float) -> float:
    """Map one channel's spatial feedback to its quadratic adjustment intensity.

    The two-field closure sets ``rho_k = lambda_k / (1 + lambda_b + lambda_n)``,
    which inverts to ``lambda_k = rho_k / (1 - rho_b - rho_n)`` and is strictly
    increasing in the channel coefficient on the stationary region.

    Args:
        rho_channel: One channel's spatial feedback coefficient.
        rho_total: The sum of both channel coefficients.

    Returns:
        The channel's model-implied adjustment intensity.

    Raises:
        TwoFieldEstimationError: If the total leaves the region on which the
            quadratic model has an adjustment-intensity representation.

    """
    if not 0.0 <= rho_total < 1.0:
        message = (
            "total spatial feedback outside the stationarity region has no "
            f"adjustment-intensity representation: rho_total={rho_total!r}"
        )
        raise TwoFieldEstimationError(message)
    return rho_channel / (1.0 - rho_total)


__all__ = [
    "NestedQLRBootstrap",
    "TwoFieldEstimationError",
    "TwoFieldGrid",
    "TwoFieldPanelStats",
    "TwoFieldProblem",
    "build_two_field_problem",
    "channel_intensity",
    "log_determinant_table",
    "matrix_log_determinant",
    "mixture_eigenvalues",
    "nested_qlr_bootstrap",
    "quasi_likelihood_ratio",
    "spectrum_log_determinant",
    "two_field_bootstrap",
    "two_field_panel_stats",
    "two_field_profile_surface",
    "two_field_qmle",
]

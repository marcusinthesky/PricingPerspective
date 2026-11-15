"""SAR QMLE/GMM estimation of ``rho`` under a fixed interaction matrix ``W``.

Model (disclosed simplification — no cross-sectional covariates beyond a
common intercept, consistent with the S-CAPM ``bar-alpha = 0`` pricing
restriction under test in ``hypotheses.py``):

    r_t = rho * W r_t + alpha * 1 + e_t,   e_t ~ iid(0, sigma^2 I_n),

for each cross-section ``t = 1..T`` in a fold, ``W`` FIXED (not estimated
jointly).  This is the Ord (1975) concentrated-likelihood SAR spatial-lag
model, degenerate along the time dimension (the same ``rho``, ``alpha``
pool information across all ``T`` cross-sections, as in a spatial panel
with time-invariant ``W``).

QMLE: concentrated log-likelihood in ``rho`` (Ord's Jacobian term via the
eigenvalues of ``W``), optimized by a coarse global scan followed by bounded
local refinements within ``(-1+eps, 1-eps)``.

GMM: Kelejian-Prucha-style two-stage spatial instrumental-variables
estimator, using ``W^2 1`` as an instrument for ``Wr`` (best linear
predictor of the endogenous spatial lag from the exogenous instrument set
``{1, W1, W^2 1}``), then OLS of ``r`` on ``[Wr_hat, 1]``.

QMLE searches the strict interior. The legacy intercept-only IV design fails
identification for row-stochastic W and raises rather than reporting a clipped
minimum-norm coefficient as an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from collections.abc import Callable

Float = NDArray[np.float64]
FloatInput = NDArray[np.floating[Any]]
Complex = NDArray[np.complex128]
type FactorVarianceModel = Literal["spherical", "diagonal"]

_RHO_EPS = 1e-3
NUMERICAL_FLOOR = 1e-12
_MATRIX_DIMENSIONS = 2
_SCALAR_SCAN_POINTS = 65
_SCALAR_XATOL = 1e-8
_MIN_GRID_POINTS = 2


class SarEstimationError(ValueError):
    """Report an invalid or unstable SAR estimate."""


@dataclass(frozen=True, slots=True)
class FactorSarFit:
    """Profiled spatial factor-model fit.

    ``coefficients`` has shape ``(1 + k, n)``.  Its first row contains
    firm-specific intercepts and the remaining rows contain firm-specific
    loadings in the same order as the supplied factor columns.
    """

    rho_hat: float
    coefficients: Float
    sigma2_hat: float
    innovation_variances: Float
    variance_model: FactorVarianceModel
    loglik: float
    converged: bool
    n_obs: int
    n_assets: int
    n_factors: int
    rho_se: float
    refinement_gain: float = 0.0


@dataclass(frozen=True, slots=True)
class _FactorProfile:
    """Sufficient statistics for profiling factor loadings over ``rho``."""

    returns: Float
    spatial_lag: Float
    design: Float
    coefficient_returns: Float
    coefficient_lag: Float
    residual_returns: Float
    residual_lag: Float
    rss_returns: Float
    cross_returns_lag: Float
    rss_lag: Float
    eigvals_w: Complex
    bounds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _ScalarSearch:
    """Result of a scanned and locally refined one-dimensional search."""

    x: float
    value: float
    converged: bool
    refinement_gain: float


def _global_scalar_minimize(
    objective: Callable[[float], float],
    bounds: tuple[float, float],
    *,
    grid_points: int = _SCALAR_SCAN_POINTS,
    xatol: float = _SCALAR_XATOL,
) -> _ScalarSearch:
    """Guard bounded scalar optimization with a scan and local refinements.

    ``refinement_gain`` records the improvement over the best finite grid
    value.  It is a diagnostic of local polishing, not a bound on the
    remaining distance to a global optimum.
    """
    lower, upper = bounds
    if not lower < upper:
        message = f"invalid scalar-search bounds: {bounds}"
        raise SarEstimationError(message)
    if grid_points < _MIN_GRID_POINTS:
        message = "scalar-search grid must contain at least two points"
        raise SarEstimationError(message)
    grid = np.linspace(lower, upper, grid_points, dtype=np.float64)
    values = np.asarray([objective(float(point)) for point in grid], dtype=np.float64)
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        message = "scalar-search objective is non-finite on its grid"
        raise SarEstimationError(message)

    candidate_indices = {int(np.argmin(np.where(finite, values, np.inf)))}
    candidate_indices.update((0, grid_points - 1))
    candidate_indices.update(
        index
        for index in range(1, grid_points - 1)
        if finite[index]
        and values[index] <= values[index - 1]
        and values[index] <= values[index + 1]
    )

    candidates: list[tuple[float, float]] = [
        (float(point), float(value))
        for point, value, is_finite in zip(grid, values, finite, strict=True)
        if is_finite
    ]
    refinement_brackets = {
        (
            max(index - 1, 0),
            min(index + 1, grid_points - 1),
        )
        for index in candidate_indices
    }
    refinement_statuses: list[bool] = []
    for lower_index, upper_index in sorted(refinement_brackets):
        if lower_index >= upper_index:
            continue
        result = minimize_scalar(
            objective,
            bounds=(float(grid[lower_index]), float(grid[upper_index])),
            method="bounded",
            options={"xatol": xatol},
        )
        x = float(result.x)
        value = float(objective(x)) if np.isfinite(x) else float("nan")
        success = bool(result.success) and np.isfinite(x) and np.isfinite(value)
        refinement_statuses.append(success)
        if success:
            candidates.append((x, value))

    if lower < 0.0 < upper:
        zero_value = float(objective(0.0))
        if np.isfinite(zero_value):
            candidates.append((0.0, zero_value))
    if not candidates:
        message = "scalar-search produced no finite candidate"
        raise SarEstimationError(message)

    best_x, best_value = min(candidates, key=lambda item: item[1])
    converged = all(refinement_statuses)
    grid_best = float(np.min(values[finite]))
    refinement_gain = max(0.0, grid_best - best_value)
    return _ScalarSearch(
        x=best_x,
        value=best_value,
        converged=converged,
        refinement_gain=refinement_gain,
    )


def _log_det_jacobian(rho: float, eigvals_w: Complex) -> float:
    """Return Ord's Jacobian term ``log|det(I - rho W)|``.

    A real asymmetric ``W`` may have complex-conjugate eigenvalue pairs. Their
    imaginary components must remain in ``|1 - rho * lambda_k|``; discarding
    them before the logarithm does not equal the matrix log-absolute-
    determinant. A singular factor deliberately contributes negative infinity.
    """
    with np.errstate(divide="ignore"):
        log_abs_terms = np.log(np.abs(1.0 - rho * eigvals_w))
    return float(np.sum(log_abs_terms))


@dataclass(frozen=True, slots=True)
class _SarProfile:
    """Common-intercept profile shared by point fitting and curvature."""

    returns: Float
    spatial_lag: Float
    spectrum: Complex
    bounds: tuple[float, float]

    def nuisance(self, rho: float) -> tuple[float, float, float]:
        """Return intercept, profiled innovation scale and residual sum of squares."""
        residual_pre = self.returns - rho * self.spatial_lag
        alpha = float(np.mean(residual_pre))
        rss = float(np.sum((residual_pre - alpha) ** 2))
        return alpha, max(rss / self.returns.size, NUMERICAL_FLOOR), rss

    def objective(self, rho: float) -> float:
        """Return the negative Gaussian profile likelihood, including constants."""
        _, sigma2, rss = self.nuisance(rho)
        count = self.returns.size
        ll = -0.5 * count * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
        ll += self.returns.shape[0] * _log_det_jacobian(rho, self.spectrum)
        return float(-ll)


def _sar_profile(returns: Float, w: Float) -> _SarProfile:
    """Prepare one spatial lag, full complex spectrum and admissible bounds."""
    spectrum = cast("Complex", np.linalg.eigvals(w))
    rho_max = 1.0 / max(np.max(np.abs(spectrum)), 1e-8)
    limit = min(0.999, 0.999 * rho_max)
    return _SarProfile(returns, returns @ w.T, spectrum, (-limit, limit))


def _sar_profile_se(
    profile: _SarProfile,
    rho_hat: float,
    bounds: tuple[float, float],
    step: float,
) -> float:
    """Evaluate the retained central-curvature rule on the fitted objective."""
    lo, hi = bounds
    h = step
    if rho_hat + h >= hi or rho_hat - h <= lo:
        h = min(h, 0.4 * (hi - rho_hat), 0.4 * (rho_hat - lo))
        if h <= 0.0:
            return float("nan")
    curvature = (
        profile.objective(rho_hat + h)
        - 2.0 * profile.objective(rho_hat)
        + profile.objective(rho_hat - h)
    ) / (h * h)
    if not np.isfinite(curvature) or curvature <= 0.0:
        return float("nan")
    return float(np.sqrt(1.0 / curvature))


def sar_qmle(
    returns: Float,
    w: Float,
    *,
    compute_se: bool = True,
) -> dict[str, float]:
    """Concentrated-likelihood QMLE of ``rho`` (and ``alpha``, ``sigma^2``).

    Args:
        returns: Return panel for the fold, shape ``(T, n)``.
        w: Fixed interaction matrix, shape ``(n, n)``.
        compute_se: Whether to evaluate numerical-Hessian uncertainty. Bootstrap
            refits may disable it when only the point estimate is consumed.

    Returns:
        Dict with ``rho_hat``, ``alpha_hat``, ``sigma2_hat``, ``loglik``,
        ``converged``, and the scan-to-refinement ``refinement_gain``. The
        gain is not a bound on remaining optimization error.
        ``rho_se`` is numerical-Hessian uncertainty when ``compute_se`` is true
        and ``nan`` for resampled point-only fits.

    """
    t_obs, n = returns.shape
    profile = _sar_profile(returns, w)
    lo, hi = profile.bounds
    result = _global_scalar_minimize(profile.objective, profile.bounds)
    rho_hat = float(result.x)
    alpha_hat, sigma2_hat, _ = profile.nuisance(rho_hat)

    rho_hat = float(np.clip(rho_hat, -1.0 + _RHO_EPS, 1.0 - _RHO_EPS))
    if not abs(rho_hat) < 1.0:
        message = f"SAR QMLE produced |rho_hat|>=1: {rho_hat}"
        raise SarEstimationError(message)

    rho_se = (
        _sar_profile_se(profile, rho_hat, (lo, hi), 1e-2)
        if compute_se
        else float("nan")
    )

    return {
        "rho_hat": rho_hat,
        "alpha_hat": alpha_hat,
        "sigma2_hat": sigma2_hat,
        "loglik": float(-result.value),
        "converged": bool(result.converged),
        "n_obs": int(t_obs),
        "n_assets": int(n),
        "rho_se": rho_se,
        "refinement_gain": float(result.refinement_gain),
    }


def sar_rho_se(
    returns: Float,
    w: Float,
    rho_hat: float,
    bounds: tuple[float, float] | None = None,
    step: float = 1e-2,
) -> float:
    """Compute the numerical-Hessian standard error of ``rho_hat``.

    Reuses the exact ``neg_loglik(rho)`` objective evaluated by
    :func:`sar_qmle`'s scalar search and takes a central second difference at
    ``rho_hat`` to approximate the observed information
    ``-d^2 loglik/drho^2``.  ``SE = sqrt(1 / (-d^2 loglik/drho^2))``.

    Falls back to ``float("nan")`` (never silently to a fixed constant) if the
    curvature is non-positive or the step lands outside the admissible
    ``|rho| < 1`` support; the caller is responsible for a loud, disclosed
    fallback (see ``run.py``).

    Args:
        returns: Return panel for the fold, shape ``(T, n)``.
        w: Fixed interaction matrix, shape ``(n, n)``.
        rho_hat: The QMLE point estimate to evaluate curvature at.
        bounds: Optional ``(lo, hi)`` admissible range for ``rho`` (defaults
            to the same eigenvalue-derived bounds used by ``sar_qmle``).
        step: Central finite-difference step in ``rho`` (default ``1e-2``).

    Returns:
        The numerical-Hessian SE, or ``nan`` if curvature is unusable.

    """
    profile = _sar_profile(returns, w)
    return _sar_profile_se(
        profile, rho_hat, profile.bounds if bounds is None else bounds, step
    )


def _factor_profile(
    returns: FloatInput,
    w: FloatInput,
    factors: FloatInput,
    eigvals_w: Complex | None = None,
) -> _FactorProfile:
    """Build exact profile-likelihood sufficient statistics in float64."""
    returns64 = np.asarray(returns, dtype=np.float64)
    weights64 = np.asarray(w, dtype=np.float64)
    factors64 = np.asarray(factors, dtype=np.float64)
    if returns64.ndim != _MATRIX_DIMENSIONS:
        message = "factor SAR returns must be a two-dimensional date-by-asset panel"
        raise SarEstimationError(message)
    t_obs, n_assets = returns64.shape
    if weights64.shape != (n_assets, n_assets):
        message = (
            "factor SAR weights must match the asset dimension: "
            f"expected {(n_assets, n_assets)}, received {weights64.shape}"
        )
        raise SarEstimationError(message)
    if factors64.ndim != _MATRIX_DIMENSIONS or factors64.shape[0] != t_obs:
        message = (
            "factor SAR factors must be a date-aligned matrix: "
            f"expected first dimension {t_obs}, received {factors64.shape}"
        )
        raise SarEstimationError(message)
    if not all(
        bool(np.all(np.isfinite(values)))
        for values in (returns64, weights64, factors64)
    ):
        message = "factor SAR inputs must be finite"
        raise SarEstimationError(message)
    design = np.column_stack((np.ones(t_obs, dtype=np.float64), factors64))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        message = "factor SAR design matrix is rank deficient"
        raise SarEstimationError(message)
    spatial_lag = returns64 @ weights64.T
    coefficient_returns, *_ = np.linalg.lstsq(design, returns64, rcond=None)
    coefficient_lag, *_ = np.linalg.lstsq(design, spatial_lag, rcond=None)
    residual_returns = returns64 - design @ coefficient_returns
    residual_lag = spatial_lag - design @ coefficient_lag
    rss_returns = np.sum(np.square(residual_returns), axis=0)
    cross_returns_lag = np.sum(residual_returns * residual_lag, axis=0)
    rss_lag = np.sum(np.square(residual_lag), axis=0)
    eigenvalues = (
        cast("Complex", np.linalg.eigvals(weights64))
        if eigvals_w is None
        else np.asarray(eigvals_w, dtype=np.complex128)
    )
    if eigenvalues.shape != (n_assets,) or not bool(np.all(np.isfinite(eigenvalues))):
        message = "factor SAR eigenvalues do not match the supplied weights"
        raise SarEstimationError(message)
    rho_max = 1.0 / max(float(np.max(np.abs(eigenvalues))), 1e-8)
    bounds = (-min(0.999, 0.999 * rho_max), min(0.999, 0.999 * rho_max))
    return _FactorProfile(
        returns=returns64,
        spatial_lag=spatial_lag,
        design=cast("Float", design),
        coefficient_returns=cast("Float", coefficient_returns),
        coefficient_lag=cast("Float", coefficient_lag),
        residual_returns=cast("Float", residual_returns),
        residual_lag=cast("Float", residual_lag),
        rss_returns=cast("Float", rss_returns),
        cross_returns_lag=cast("Float", cross_returns_lag),
        rss_lag=cast("Float", rss_lag),
        eigvals_w=cast("Complex", eigenvalues),
        bounds=bounds,
    )


def _factor_profile_loglik(
    profile: _FactorProfile,
    rho: float,
    variance_model: FactorVarianceModel,
) -> tuple[float, Float]:
    """Evaluate the factor-profiled Gaussian quasi-likelihood at ``rho``."""
    rss = (
        profile.rss_returns
        - 2.0 * rho * profile.cross_returns_lag
        + rho * rho * profile.rss_lag
    )
    rss = np.maximum(rss, 0.0)
    t_obs = profile.returns.shape[0]
    n_assets = profile.returns.shape[1]
    if variance_model == "spherical":
        total_rss = float(np.sum(rss))
        common_variance = max(total_rss / (t_obs * n_assets), NUMERICAL_FLOOR)
        innovation_variances = np.full(n_assets, common_variance, dtype=np.float64)
        quadratic = total_rss / common_variance
        log_scale = float(n_assets * np.log(2.0 * np.pi * common_variance))
    elif variance_model == "diagonal":
        innovation_variances = np.maximum(rss / t_obs, NUMERICAL_FLOOR)
        quadratic = float(np.sum(rss / innovation_variances))
        log_scale = float(np.sum(np.log(2.0 * np.pi * innovation_variances)))
    else:
        message = f"unsupported factor SAR variance model: {variance_model}"
        raise SarEstimationError(message)
    loglik = -0.5 * t_obs * log_scale - 0.5 * quadratic
    loglik += t_obs * _log_det_jacobian(rho, profile.eigvals_w)
    return float(loglik), cast("Float", innovation_variances)


def _profile_rho_se(
    profile: _FactorProfile,
    rho_hat: float,
    variance_model: FactorVarianceModel,
    step: float = 1e-2,
) -> float:
    """Return observed-information uncertainty for a factor-profiled fit."""
    lo, hi = profile.bounds
    h = min(step, 0.4 * (hi - rho_hat), 0.4 * (rho_hat - lo))
    if h <= 0.0:
        return float("nan")
    f0 = -_factor_profile_loglik(profile, rho_hat, variance_model)[0]
    f_plus = -_factor_profile_loglik(profile, rho_hat + h, variance_model)[0]
    f_minus = -_factor_profile_loglik(profile, rho_hat - h, variance_model)[0]
    curvature = (f_plus - 2.0 * f0 + f_minus) / (h * h)
    if not np.isfinite(curvature) or curvature <= 0.0:
        return float("nan")
    return float(np.sqrt(1.0 / curvature))


def _factor_fit_from_profile(
    profile: _FactorProfile,
    rho: float,
    *,
    converged: bool,
    compute_se: bool,
    variance_model: FactorVarianceModel,
    refinement_gain: float = 0.0,
) -> FactorSarFit:
    """Materialize nuisance estimates from a profiled value of ``rho``."""
    loglik, innovation_variances = _factor_profile_loglik(profile, rho, variance_model)
    coefficients = profile.coefficient_returns - rho * profile.coefficient_lag
    t_obs, n_assets = profile.returns.shape
    return FactorSarFit(
        rho_hat=rho,
        coefficients=cast("Float", coefficients),
        sigma2_hat=float(np.mean(innovation_variances)),
        innovation_variances=innovation_variances,
        variance_model=variance_model,
        loglik=loglik,
        converged=converged,
        n_obs=t_obs,
        n_assets=n_assets,
        n_factors=profile.design.shape[1] - 1,
        rho_se=(
            _profile_rho_se(profile, rho, variance_model)
            if compute_se
            else float("nan")
        ),
        refinement_gain=refinement_gain,
    )


def sar_factor_qmle(
    excess_returns: FloatInput,
    w: FloatInput,
    factors: FloatInput,
    *,
    compute_se: bool = True,
    eigvals_w: Complex | None = None,
    variance_model: FactorVarianceModel = "spherical",
) -> FactorSarFit:
    """Profile firm intercepts and factor loadings inside spatial QMLE.

    The working equation is
    ``(I - rho W) r_t^e = alpha + A f_t + epsilon_t``.  Because the residual
    maker for ``[1, f_t]`` is linear, both the transformed returns and their
    spatial lag are residualized once.  Every likelihood evaluation then uses
    their exact affine combination, avoiding repeated least-squares solves.
    """
    profile = _factor_profile(excess_returns, w, factors, eigvals_w)
    lo, hi = profile.bounds
    result = _global_scalar_minimize(
        lambda rho: -_factor_profile_loglik(profile, float(rho), variance_model)[0],
        (lo, hi),
    )
    rho_hat = float(np.clip(result.x, -1.0 + _RHO_EPS, 1.0 - _RHO_EPS))
    if not abs(rho_hat) < 1.0:
        message = f"factor SAR QMLE produced |rho_hat|>=1: {rho_hat}"
        raise SarEstimationError(message)
    return _factor_fit_from_profile(
        profile,
        rho_hat,
        converged=bool(result.converged),
        compute_se=compute_se,
        variance_model=variance_model,
        refinement_gain=result.refinement_gain,
    )


def sar_factor_fit_at_rho(
    excess_returns: FloatInput,
    w: FloatInput,
    factors: FloatInput,
    rho: float,
    *,
    eigvals_w: Complex | None = None,
    variance_model: FactorVarianceModel = "spherical",
) -> FactorSarFit:
    """Profile factor coefficients at one prespecified spatial coefficient."""
    profile = _factor_profile(excess_returns, w, factors, eigvals_w)
    lo, hi = profile.bounds
    if not lo < rho < hi:
        message = (
            f"factor SAR rho={rho} lies outside the admissible interval {(lo, hi)}"
        )
        raise SarEstimationError(message)
    return _factor_fit_from_profile(
        profile,
        float(rho),
        converged=True,
        compute_se=False,
        variance_model=variance_model,
    )


def sar_factor_daily_logscore(
    fit: FactorSarFit,
    excess_returns: FloatInput,
    w: FloatInput,
    factors: FloatInput,
    *,
    eigvals_w: Complex | None = None,
) -> Float:
    """Score held-out dates without refitting any fitted parameter."""
    returns64 = np.asarray(excess_returns, dtype=np.float64)
    weights64 = np.asarray(w, dtype=np.float64)
    factors64 = np.asarray(factors, dtype=np.float64)
    if returns64.ndim != _MATRIX_DIMENSIONS or returns64.shape[1] != fit.n_assets:
        message = "held-out returns do not match the fitted asset dimension"
        raise SarEstimationError(message)
    if factors64.shape != (returns64.shape[0], fit.n_factors):
        message = "held-out factors do not match the fitted factor dimension"
        raise SarEstimationError(message)
    if weights64.shape != (fit.n_assets, fit.n_assets):
        message = "held-out weights do not match the fitted asset dimension"
        raise SarEstimationError(message)
    design = np.column_stack((np.ones(returns64.shape[0], dtype=np.float64), factors64))
    residual = returns64 - fit.rho_hat * (returns64 @ weights64.T)
    residual -= design @ fit.coefficients
    eigenvalues = (
        cast("Complex", np.linalg.eigvals(weights64))
        if eigvals_w is None
        else np.asarray(eigvals_w, dtype=np.complex128)
    )
    if eigenvalues.shape != (fit.n_assets,):
        message = "held-out eigenvalues do not match the fitted asset dimension"
        raise SarEstimationError(message)
    logdet = _log_det_jacobian(fit.rho_hat, cast("Complex", eigenvalues))
    variance_terms = np.log(2.0 * np.pi * fit.innovation_variances)
    constant = logdet - 0.5 * float(np.sum(variance_terms))
    scores = constant - 0.5 * np.sum(
        np.square(residual) / fit.innovation_variances[None, :], axis=1
    )
    return cast("Float", np.asarray(scores, dtype=np.float64))


def sar_factor_reduced_form_coefficients(
    fit: FactorSarFit,
    w: FloatInput,
) -> Float:
    """Map structural factor coefficients back to observed-return coefficients."""
    weights64 = np.asarray(w, dtype=np.float64)
    if weights64.shape != (fit.n_assets, fit.n_assets):
        message = "reduced-form weights do not match the fitted asset dimension"
        raise SarEstimationError(message)
    spatial_filter = np.eye(fit.n_assets, dtype=np.float64) - fit.rho_hat * weights64
    reduced = np.linalg.solve(spatial_filter, fit.coefficients.T).T
    return cast("Float", np.asarray(reduced, dtype=np.float64))


def factor_sar_parameter_count(fit: FactorSarFit, *, spatial: bool = True) -> int:
    """Count profiled coefficients for working information criteria."""
    regression_parameters = fit.n_assets * (fit.n_factors + 1)
    variance_parameters = 1 if fit.variance_model == "spherical" else fit.n_assets
    return regression_parameters + variance_parameters + int(spatial)


def sar_gmm(returns: Float, w: Float) -> dict[str, float]:
    """Kelejian-Prucha-style spatial 2SLS/GMM estimate of ``rho``, ``alpha``.

    Instrument set ``{1, W1, W^2 1}`` (best linear predictor of the
    endogenous spatial lag ``Wr`` from exogenous instruments), first-stage
    OLS predicting ``Wr`` from the instruments, second-stage OLS of ``r`` on
    ``[Wr_hat, 1]``.

    Structural instrument degeneracy (disclosed, not fixed by construction
    -- see t14 validation): for any row-stochastic ``w`` (row sums exactly
    ``1``, as every ``W`` this project builds is), ``W1 = 1`` identically,
    so ``W^2 1 = 1`` too -- the three-column instrument matrix ``z`` above
    collapses to three copies of the constant column *regardless of ``n``*.
    This is not a small-``n`` weak-instrument phenomenon (it does not
    improve with more assets or more time periods); it is an exact
    algebraic consequence of pairing the Kelejian-Prucha ``{1, W1, W^2 1}``
    instrument set with an intercept-only model (no exogenous covariate
    whose spatial lag would vary across assets). ``pysal.spreg.GM_Lag``
    hits the identical degeneracy on the same inputs (see
    ``validate_estimators.py``'s ``_spreg_estimates``): it raises on an
    exactly singular ``Z'H`` design in most cases, and in the rare case it
    does not raise, is numerically unstable (t14 empirical comparison
    produced a ``GM_Lag`` estimate outside ``[-1, 1]``). Unlike ``spreg``,
    ``numpy.linalg.lstsq`` does not raise on the (near-)singular design; it
    silently returns the minimum-norm solution, which puts negligible
    weight on the informative direction and drives ``rho_hat`` toward
    ``0`` regardless of the true ``rho``. Rank failure therefore raises
    before any least-squares coefficient can be reported.

    Args:
        returns: Return panel for the fold, shape ``(T, n)``.
        w: Fixed interaction matrix, shape ``(n, n)``.

    Returns:
        Dict with ``rho_hat``, ``alpha_hat`` and rank diagnostics, only for
        an identified design inside the admissible support.

    Raises:
        SarEstimationError: Instruments are rank deficient or the estimate
            lies outside the support. Row-stochastic intercept-only inputs
            always fail identification.

    """
    t_obs, n = returns.shape
    ones = np.ones(n)
    w1 = w @ ones
    w2_1 = w @ w1
    z = np.column_stack([ones, w1, w2_1])  # (n, 3) instruments

    instrument_rank_deficient = bool(np.linalg.matrix_rank(z) < z.shape[1])
    if instrument_rank_deficient:
        message = (
            "SAR IV is unidentified: {1, W1, W²1} are rank deficient. "
            "Provide actual exogenous covariates and instruments to a spatial IV model."
        )
        raise SarEstimationError(message)

    wr = returns @ w.T  # (T, n)
    r_flat = returns.reshape(-1)
    wr_flat = wr.reshape(-1)
    z_stacked = np.tile(z, (t_obs, 1))  # (T*n, 3)

    # First stage: predict Wr from instruments (pooled OLS).
    coef1, *_ = np.linalg.lstsq(z_stacked, wr_flat, rcond=None)
    wr_hat = z_stacked @ coef1

    # Second stage: r on [Wr_hat, 1].
    x2 = np.column_stack([wr_hat, np.ones(r_flat.shape[0])])
    coef2, *_ = np.linalg.lstsq(x2, r_flat, rcond=None)
    rho_hat = float(coef2[0])
    alpha_hat = float(coef2[1])

    if not abs(rho_hat) < 1.0:
        message = f"SAR GMM produced |rho_hat|>=1: {rho_hat}"
        raise SarEstimationError(message)

    return {
        "rho_hat": rho_hat,
        "alpha_hat": alpha_hat,
        "instrument_rank_deficient": instrument_rank_deficient,
    }

"""SAR QMLE/GMM estimation of ``rho``, Moran's I, and ``W`` baselines.

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
eigenvalues of ``W``), optimized by bounded scalar search
(``scipy.optimize.minimize_scalar``, Brent within ``(-1+eps, 1-eps)``).

GMM: Kelejian-Prucha-style two-stage spatial instrumental-variables
estimator, using ``W^2 1`` as an instrument for ``Wr`` (best linear
predictor of the endogenous spatial lag from the exogenous instrument set
``{1, W1, W^2 1}``), then OLS of ``r`` on ``[Wr_hat, 1]``.

Every estimate is post-hoc clipped/asserted ``|rho_hat| < 1`` (required by
convergence of the Leontief series; ``W`` row-stochastic or sub-stochastic
so the strict interior is the meaningful support).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

Float = NDArray[np.float64]
Complex = NDArray[np.complex128]

_RHO_EPS = 1e-3
NUMERICAL_FLOOR = 1e-12


class SarEstimationError(ValueError):
    """Report an invalid or unstable SAR estimate."""


def moran_i(returns: Float, w: Float) -> float:
    """Moran's I spatial-autocorrelation statistic on a return panel.

    ``I = (n / S0) * (e' W e) / (e' e)`` with ``e`` the panel-demeaned,
    time-averaged residual (cross-sectional mean return per asset,
    demeaned), ``S0 = sum_ij W_ij``.

    Args:
        returns: Return panel, shape ``(T, n)``.
        w: Interaction matrix, shape ``(n, n)``.

    Returns:
        Moran's I statistic.

    """
    x = returns.mean(axis=0)
    e = x - x.mean()
    s0 = float(np.sum(w))
    denom = float(e @ e)
    if abs(s0) <= 0.0 or denom <= 0.0:
        return float("nan")
    n = w.shape[0]
    return float(n / s0 * (e @ w @ e) / denom)


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
        ``converged``. ``rho_se`` is numerical-Hessian uncertainty when
        ``compute_se`` is true and ``nan`` for resampled point-only fits.

    """
    t_obs, n = returns.shape
    eigvals_w = cast("Complex", np.linalg.eigvals(w))
    rho_max = 1.0 / max(np.max(np.abs(eigvals_w)), 1e-8)
    lo = -min(0.999, 0.999 * rho_max)
    hi = min(0.999, 0.999 * rho_max)

    def neg_loglik(rho: float) -> float:
        wr = returns @ w.T  # (T, n): (W r_t) for each t, stacked
        resid_pre = returns - rho * wr  # (I - rho W) r_t
        alpha_hat = float(np.mean(resid_pre))
        resid = resid_pre - alpha_hat
        rss = float(np.sum(resid**2))
        sigma2 = rss / (t_obs * n)
        sigma2 = max(sigma2, NUMERICAL_FLOOR)
        logdet = _log_det_jacobian(rho, eigvals_w)
        ll = -0.5 * t_obs * n * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
        ll += t_obs * logdet
        return -ll

    res = minimize_scalar(neg_loglik, bounds=(lo, hi), method="bounded")
    rho_hat = float(res.x)
    wr = returns @ w.T
    resid_pre = returns - rho_hat * wr
    alpha_hat = float(np.mean(resid_pre))
    resid = resid_pre - alpha_hat
    rss = float(np.sum(resid**2))
    sigma2_hat = max(rss / (t_obs * n), NUMERICAL_FLOOR)

    rho_hat = float(np.clip(rho_hat, -1.0 + _RHO_EPS, 1.0 - _RHO_EPS))
    if not abs(rho_hat) < 1.0:
        message = f"SAR QMLE produced |rho_hat|>=1: {rho_hat}"
        raise SarEstimationError(message)

    rho_se = (
        sar_rho_se(returns, w, rho_hat, bounds=(lo, hi)) if compute_se else float("nan")
    )

    return {
        "rho_hat": rho_hat,
        "alpha_hat": alpha_hat,
        "sigma2_hat": sigma2_hat,
        "loglik": float(-res.fun),
        "converged": bool(res.success),
        "n_obs": int(t_obs),
        "n_assets": int(n),
        "rho_se": rho_se,
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
    t_obs, n = returns.shape
    eigvals_w = cast("Complex", np.linalg.eigvals(w))
    if bounds is None:
        rho_max = 1.0 / max(np.max(np.abs(eigvals_w)), 1e-8)
        bounds = (-min(0.999, 0.999 * rho_max), min(0.999, 0.999 * rho_max))
    lo, hi = bounds

    def neg_loglik(rho: float) -> float:
        wr = returns @ w.T
        resid_pre = returns - rho * wr
        alpha_hat = float(np.mean(resid_pre))
        resid = resid_pre - alpha_hat
        rss = float(np.sum(resid**2))
        sigma2 = max(rss / (t_obs * n), NUMERICAL_FLOOR)
        logdet = _log_det_jacobian(rho, eigvals_w)
        ll = -0.5 * t_obs * n * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
        ll += t_obs * logdet
        return -ll

    h = step
    rho_plus = rho_hat + h
    rho_minus = rho_hat - h
    if rho_plus >= hi or rho_minus <= lo:
        # Shrink the step to stay within the admissible interior; if that is
        # not possible (rho_hat pinned at the boundary), give up cleanly.
        h = min(h, 0.4 * (hi - rho_hat), 0.4 * (rho_hat - lo))
        if h <= 0.0:
            return float("nan")
        rho_plus = rho_hat + h
        rho_minus = rho_hat - h

    f0 = neg_loglik(rho_hat)
    f_plus = neg_loglik(rho_plus)
    f_minus = neg_loglik(rho_minus)
    curvature = (f_plus - 2.0 * f0 + f_minus) / (h * h)
    if not np.isfinite(curvature) or curvature <= 0.0:
        return float("nan")
    return float(np.sqrt(1.0 / curvature))


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
    ``0`` regardless of the true ``rho``. ``instrument_rank_deficient``
    below makes that failure mode visible instead of masking it as "no
    spatial correlation".

    Args:
        returns: Return panel for the fold, shape ``(T, n)``.
        w: Fixed interaction matrix, shape ``(n, n)``.

    Returns:
        Dict with ``rho_hat``, ``alpha_hat``, and
        ``instrument_rank_deficient`` (``True`` iff the ``{1, W1, W^2 1}``
        instrument matrix is (near-)rank-deficient, in which case
        ``rho_hat`` reflects the minimum-norm least-squares solution rather
        than an identified GMM estimate).

    """
    t_obs, n = returns.shape
    ones = np.ones(n)
    w1 = w @ ones
    w2_1 = w @ w1
    z = np.column_stack([ones, w1, w2_1])  # (n, 3) instruments

    instrument_rank_deficient = bool(np.linalg.matrix_rank(z) < z.shape[1])
    if instrument_rank_deficient:
        logger.warning(
            "sar_gmm: {1, W1, W^2 1} instrument matrix is rank-deficient "
            "(rank=%d of %d) -- W is row-stochastic so W1=W^2 1=1; rho_hat "
            "reflects the minimum-norm least-squares solution, not an "
            "identified GMM estimate (see sar_gmm docstring).",
            int(np.linalg.matrix_rank(z)),
            z.shape[1],
        )

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

    rho_hat = float(np.clip(rho_hat, -1.0 + _RHO_EPS, 1.0 - _RHO_EPS))
    if not abs(rho_hat) < 1.0:
        message = f"SAR GMM produced |rho_hat|>=1: {rho_hat}"
        raise SarEstimationError(message)

    return {
        "rho_hat": rho_hat,
        "alpha_hat": alpha_hat,
        "instrument_rank_deficient": instrument_rank_deficient,
    }


# ---------------------------------------------------------------------------
# Baseline W constructions
# ---------------------------------------------------------------------------


def industry_adjacency_w(tickers: list[str], sector_of: dict[str, str]) -> Float:
    """Row-normalized zero-diagonal industry-adjacency baseline ``W``.

    ``W_ij = 1`` if ``i != j`` and ``sector(i) == sector(j)``, else ``0``,
    row-normalized.

    Args:
        tickers: Ticker order.
        sector_of: Dict ``{ticker: sector}`` (from ``universe.csv``).

    Returns:
        ``(n, n)`` row-normalized matrix.

    """
    n = len(tickers)
    w = np.zeros((n, n))
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i != j and sector_of.get(ti) == sector_of.get(tj):
                w[i, j] = 1.0
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return w / row_sums


def correlation_knn_w(returns: Float, k: int = 5) -> Float:
    """Row-normalized zero-diagonal correlation-kNN baseline ``W``.

    ``W_ij`` nonzero for ``j`` among asset ``i``'s ``k`` highest-correlation
    neighbors (Pearson correlation of returns), weighted by the correlation
    itself (clipped to be nonnegative), row-normalized.

    Args:
        returns: Return panel, shape ``(T, n)``.
        k: Number of neighbors per row.

    Returns:
        ``(n, n)`` row-normalized matrix, zero diagonal.

    """
    n = returns.shape[1]
    corr = cast("Float", np.corrcoef(returns, rowvar=False))
    np.fill_diagonal(corr, -np.inf)
    w = np.zeros((n, n))
    for i in range(n):
        order = np.argsort(corr[i])[::-1][:k]
        vals = np.clip(corr[i, order], 0.0, None)
        w[i, order] = vals
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return w / row_sums


def zero_w(n: int) -> Float:
    """Return the ``rho = 0`` no-interaction baseline (all-zero ``W``)."""
    return np.zeros((n, n))

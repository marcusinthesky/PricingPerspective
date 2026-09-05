"""Frontier estimators and certified-cap conversion for the H1 pilot."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jax.typing import ArrayLike  # noqa: TC002  # runtime array contract
from simulation.bounds import certified_floor


class FrontierResult(NamedTuple):
    """Point estimates of the identification-frontier Lipschitz constants.

    Attributes:
        ell_hat: Lower Lipschitz constant, the D-weighted ``tau_lo`` quantile
            of the ratios r_p = ||beta_i - beta_j|| / D_ij.
        L_hat: Upper Lipschitz constant, the D-weighted ``tau_hi`` quantile.
        breach: Fraction of pairs whose ratio falls outside
            ``[ell_hat, L_hat]`` (should be approx tau_lo + (1 - tau_hi)).
        breach_unweighted: Breach fraction for the unweighted band.
        ratios: The (P,) vector of pairwise ratios r_p.
        weights: The (P,) vector of x-weights x_p = D_ij.
        n_skipped: Number of pairs skipped because D_ij < eps.

    """

    ell_hat: float
    L_hat: float
    breach: float
    breach_unweighted: float
    ratios: np.ndarray
    weights: np.ndarray
    n_skipped: int


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """x-weighted quantile via the cumulative-weight step function.

    Sorts ``values`` ascending, accumulates ``weights``, and returns the first
    value whose cumulative weight reaches ``q * total_weight``. This is the
    through-origin quantile estimator: c_hat(tau) is the x-weighted tau-quantile
    of the ratios, which minimises the weighted check-loss of the constrained
    regression ||beta_i - beta_j|| = c * D_ij (see ROADMAP §2.0).

    Args:
        values: (P,) array of ratios r_p.
        weights: (P,) array of non-negative weights x_p.
        q: Quantile level in [0, 1].

    Returns:
        The weighted quantile as a float.

    """
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0.0:
        return float("nan")
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, q * total, side="left"))
    idx = min(idx, v.shape[0] - 1)
    return float(v[idx])


def frontier_quantiles(
    betas: np.ndarray,
    d: ArrayLike,
    tau_lo: float = 0.05,
    tau_hi: float = 0.95,
    eps: float = 1e-12,
) -> FrontierResult:
    """Estimate the lower/upper Lipschitz constants over all pairs.

    For every unordered pair (i, j) with D_ij >= eps, form the ratio
    r_p = ||beta_i - beta_j|| / D_ij and weight x_p = D_ij. The lower constant
    ell_hat is the x-weighted ``tau_lo`` quantile of the ratios and the upper
    constant L_hat is the x-weighted ``tau_hi`` quantile. As tau_lo -> 0 and
    tau_hi -> 1 these recover the hard min/max Lipschitz constants of
    ``simulation.bounds.lipschitz_constants``.

    Args:
        betas: (n, k) array of per-ticker exposure loadings.
        d: (n, n) symmetric Hilbertian distance matrix (zero diagonal).
        tau_lo: Lower quantile level (default 0.05).
        tau_hi: Upper quantile level (default 0.95).
        eps: Pairs with D_ij < eps are skipped.

    Returns:
        A :class:`FrontierResult`.

    """
    d = np.asarray(d)
    n = betas.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diffs = betas[iu] - betas[ju]
    y = np.linalg.norm(diffs, axis=1)
    x = d[iu, ju]

    valid = x >= eps
    n_skipped = int((~valid).sum())
    y = y[valid]
    x = x[valid]
    ratios = y / x

    ell_hat = _weighted_quantile(ratios, x, tau_lo)
    upper = _weighted_quantile(ratios, x, tau_hi)
    breach = float(np.mean((ratios < ell_hat) | (ratios > upper)))

    ell_uw = float(np.quantile(ratios, tau_lo))
    upper_uw = float(np.quantile(ratios, tau_hi))
    breach_uw = float(np.mean((ratios < ell_uw) | (ratios > upper_uw)))

    return FrontierResult(
        ell_hat=ell_hat,
        L_hat=upper,
        breach=breach,
        breach_unweighted=breach_uw,
        ratios=ratios,
        weights=x,
        n_skipped=n_skipped,
    )


class CapReductionResult(NamedTuple):
    """Cap reduction driven by the lower Lipschitz constant ell.

    The certified variance cap on ``w'(bb')w`` is
    ``cap(ell) = sum_i w_i ||beta_i||^2 - 0.5 * ell^2 * w' E w`` (feeding the
    LOWER Lipschitz constant into the generic ``certified_floor`` yields the
    upper bound / cap). At ``ell = 0`` this is the trivial cap
    ``cap0 = sum_i w_i ||beta_i||^2``. The reduction the frontier buys is

        reduction = cap0 - cap(ell) = 0.5 * ell^2 * w' E w,

    and ``pct = reduction / cap0`` is the fraction of the trivial cap removed.

    Attributes:
        cap0: Trivial cap ``sum_i w_i ||beta_i||^2`` (ell = 0).
        cap: Certified cap ``cap(ell)``.
        reduction: Absolute reduction ``cap0 - cap``.
        pct: Reduction as a fraction of ``cap0``.

    """

    cap0: float
    cap: float
    reduction: float
    pct: float


def cap_reduction_pct(
    w: np.ndarray,
    betas: np.ndarray,
    d2: ArrayLike,
    ell: float,
) -> CapReductionResult:
    """Cap reduction a lower Lipschitz constant buys for one weight scheme.

    Uses ``simulation.bounds.certified_floor`` (the Lean-certified JAX
    formula, binding to the non-homogeneous envelope
    ``portfolio_variance_upper_bound``, Energy.lean:585) as the single source
    of truth: the cap is that function evaluated with ``L_max = ell``, and
    the trivial cap is the same with ``L_max = 0``.  The H1 stage owns JAX
    float64 before entering this certificate path.

    Args:
        w: (n,) portfolio weights (need not be simplex; used as given).
        betas: (n, k) exposure loadings.
        d2: (n, n) raw energy-distance matrix E (D squared).
        ell: Lower Lipschitz constant (use the conservative ``ell_lo``).

    Returns:
        A :class:`CapReductionResult`.

    """
    cap0 = certified_floor(w, betas, d2, 0.0)
    cap = certified_floor(w, betas, d2, ell)
    reduction = cap0 - cap
    pct = reduction / cap0 if abs(cap0) > 0.0 else float("nan")
    return CapReductionResult(cap0=cap0, cap=cap, reduction=reduction, pct=pct)

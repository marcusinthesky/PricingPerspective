"""Minimum-variance optimizer and the covariance estimators it is fed."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import numpy as np
from jcor.model.covariance import (
    ledoit_wolf_sample,
    plug_in_lambda_dist,
    sample_covariance,
    shrink_to_dist,
)
from jcor.operators.covariance import sigma_dist
from jcor.optimize import ridge_psd
from jcor.optimize.psd import nearest_covariance
from simulation.optimize import pgd_min_variance

from pipeline.stages.papers.paper3._empirical.panel import calibrate_kappa

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipeline.stages.papers.paper3._empirical.contracts import Float


@functools.lru_cache(maxsize=4)
def _jit_pgd(n_steps: int) -> Callable[[jax.Array], jax.Array]:
    """Return a jitted long-only PGD min-variance solver (cached by n_steps).

    Compiling once per ``n_steps`` and reusing across the ~hundreds of
    rebalance dates avoids per-call JAX tracing (the eager-tracing blowup that
    made a naive daily loop intractable).
    """
    jax.config.update("jax_enable_x64", val=True)
    return jax.jit(functools.partial(pgd_min_variance, n_steps=n_steps))


def _mv_weights(sigma: Float, *, long_only: bool, n_steps: int = 1500) -> Float:
    """Minimum-variance weights, long-only (simplex PGD) or unconstrained.

    Args:
        sigma: Covariance matrix, shape ``(n, n)``.
        long_only: If ``True`` use the jitted simplex PGD solver; else the
            analytic ``Σ⁻¹ 1 / (1ᵀΣ⁻¹ 1)`` weights (ridge-regularised).
        n_steps: PGD iterations for the long-only solver.

    Returns:
        Weight vector, shape ``(n,)`` summing to 1.

    """
    n = sigma.shape[0]
    # Keep the repaired covariance on the JAX side for the jitted solver.  The
    # previous NumPy round trip forced a device/host synchronization on every
    # rebalance before immediately copying the matrix back to JAX.
    sig = ridge_psd(jax.numpy.asarray(sigma), eps=1e-8)
    if long_only:
        w = _jit_pgd(n_steps)(sig)
        return np.asarray(w, dtype=np.float64)
    sig_np = np.asarray(sig, dtype=np.float64)
    inv1 = np.linalg.solve(sig_np, np.ones(n))
    return inv1 / inv1.sum()


def _nonlinear_lw_cov(train: Float) -> Float:
    """Simplified analytic nonlinear (eigenvalue-dependent) shrinkage.

    Approximates the spirit of Ledoit-Wolf (2012) "nonlinear shrinkage of
    large-dimensional covariance matrices": sample-covariance eigenvalues are
    shrunk toward the cross-sectional grand mean with intensity increasing in
    the concentration ratio ``c = n / T`` (more shrinkage on short/wide
    windows). This is a documented SIMPLIFICATION, not the full QuEST kernel
    density estimator of the eigenvalue spectrum from the original paper —
    that requires a numerical fixed-point solve that is out of scope here.

    Args:
        train: Training return window, shape ``(T_train, n)``.

    Returns:
        Shrunk covariance, shape ``(n, n)``.

    """
    sample_covariance_matrix = sample_covariance(train, ddof=1)
    t_obs, n = train.shape
    eigval, eigvec = np.linalg.eigh(
        0.5 * (sample_covariance_matrix + sample_covariance_matrix.T)
    )
    eigval = np.clip(eigval, 1e-12, None)
    c = n / max(t_obs, 1)
    grand_mean = float(eigval.mean())
    shrunk = grand_mean + (eigval - grand_mean) / (1.0 + c)
    shrunk = np.clip(shrunk, 1e-10, None)
    return eigvec @ np.diag(shrunk) @ eigvec.T


def _ff5_cov(train: Float, ff5_factors: Float | None) -> Float | None:
    """FF5 factor-implied covariance, or ``None`` if no factor data is wired.

    Args:
        train: Training return window, shape ``(T_train, n)``.
        ff5_factors: Fama-French 5-factor daily return panel aligned to
            ``train``, shape ``(T_train, 5)``, or ``None`` if unavailable.

    Returns:
        ``B @ Sigma_f @ B.T + diag(idio_var)`` factor-implied covariance, or
        ``None`` when ``ff5_factors`` is not supplied — the FF5 column is a
        RECORDED BLOCKER (no FF5 factor-return data is on-graph in this repo
        and this stage does not fetch network data); see
        ``.context/plan/paper3/tasks/t15-backtest-table-ff5-rolling-kappa/README.md``.

    """
    if ff5_factors is None:
        return None
    f = np.asarray(ff5_factors, dtype=np.float64)
    fc = f - f.mean(axis=0, keepdims=True)
    sigma_f = np.cov(fc, rowvar=False, ddof=1)
    betas, *_ = np.linalg.lstsq(
        fc, train - train.mean(axis=0, keepdims=True), rcond=None
    )
    resid = (train - train.mean(axis=0, keepdims=True)) - fc @ betas
    idio_var = np.var(resid, axis=0, ddof=1)
    return betas.T @ sigma_f @ betas + np.diag(idio_var)


def _cov_estimators(
    train: Float, squared_distances: Float, ff5_factors: Float | None = None
) -> dict[str, Float | float]:
    """Build the covariance estimators on a training window.

    κ is recalibrated *from the training window itself* (no look-ahead): the
    full-panel κ is never used in the OOS backtest path. On short/early windows
    where the off-diagonal regression is degenerate, :func:`calibrate_kappa`
    returns ``κ=0``, in which case the distance term drops out and ``Σ_dist``
    reduces to the variance-average form ``½(σ_i² + σ_j²)`` (diagonal = sample
    variances exactly) rather than crashing.

    Args:
        train: Training return window, shape ``(T_train, n)``.
        squared_distances: Squared W2 matrix, shape ``(n, n)``.
        ff5_factors: Optional FF5 factor-return window aligned to ``train``;
            when ``None`` the ``ff5`` key is omitted entirely (recorded
            blocker — see :func:`_ff5_cov`).

    Returns:
        Dict ``{name: Σ}`` for ``sample``, ``ledoit_wolf``, ``sigma_dist``,
        ``sigma_shrink``, ``nonlinear_lw`` and, when ``ff5_factors`` is
        supplied, ``ff5``. ``kappa`` (the training-window-local calibration)
        is also returned under the ``_kappa`` key for the caller to log.

    """
    kappa = calibrate_kappa(train, squared_distances)[0]
    sigma_sq = np.diag(sample_covariance(train, ddof=1))
    # The PSD repair is traced, so Σ_dist stays on device through it and the
    # host boundary is one materialisation rather than two. x64 is required
    # for the nearest-matrix contract (see jcor.optimize.psd's precision
    # policy) and is what the surrounding stage already runs under.
    with jax.enable_x64(new_val=True):
        sd = np.array(
            nearest_covariance(
                sigma_dist(squared_distances, sigma_sq, kappa=kappa)
            ).matrix,
            dtype=np.float64,
        )
    # ``sd`` uses variances and κ estimated from ``train`` itself, so the
    # fixed-target Ledoit–Wolf derivation is invalid here.  Retain the historical
    # blend as an explicitly labelled same-window plug-in heuristic.
    lam, _, _ = plug_in_lambda_dist(train, sd)
    shr = shrink_to_dist(train, sd, lam=lam)
    out: dict[str, Float | float] = {
        "sample": np.asarray(sample_covariance(train, ddof=1)),
        "ledoit_wolf": np.asarray(ledoit_wolf_sample(train).sigma),
        "sigma_dist": sd,
        "sigma_shrink": np.asarray(shr.sigma),
        "nonlinear_lw": _nonlinear_lw_cov(train),
    }
    out["_lambda"] = shr.lam
    ff5 = _ff5_cov(train, ff5_factors)
    if ff5 is not None:
        out["ff5"] = ff5
    out["_kappa"] = kappa  # scalar smuggled for caller logging
    return out


def _portfolio_diagnostics(
    sigma: Float, w: Float, *, rank_aware: bool = False
) -> dict[str, float]:
    """Condition number, effective #assets and HHI for a weight vector.

    Args:
        sigma: Covariance used to form ``w`` (for its condition number).
        w: Weight vector, shape ``(n,)``.
        rank_aware: If true, report an infinite condition number when the
            symmetric covariance is rank-deficient at machine precision. This
            is used for the sample covariance, whose exact rank is bounded by
            ``T − 1`` after demeaning.

    Returns:
        Dict with ``cond``, ``eff_n``, ``hhi``.

    """
    ev = np.abs(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)))
    rank_tol = np.finfo(np.float64).eps * max(sigma.shape) * ev.max()
    singular = ev.min() <= rank_tol
    cond = (
        float("inf")
        if ev.min() <= 0.0 or (rank_aware and singular)
        else float(ev.max() / ev.min())
    )
    hhi = float(np.sum(w**2))
    eff_n = float(1.0 / hhi) if hhi > 0 else 0.0
    return {"cond": cond, "eff_n": eff_n, "hhi": hhi}

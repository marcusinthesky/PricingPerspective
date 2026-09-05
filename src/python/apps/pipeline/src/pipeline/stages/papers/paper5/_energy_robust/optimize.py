"""Robust and WDRO minimum-variance optimizers over a fixed ambiguity set."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jcor.optimize import ridge_psd
from jcor.optimize.psd import nearest_covariance
from optax.projections import projection_simplex
from simulation.optimize import pgd_min_variance

from pipeline.stages.papers.paper5._energy_robust.contracts import (
    RobustPortfolioComputationError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipeline.stages.papers.paper5._energy_robust.contracts import (
        Float,
    )

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _jit_pgd(n_steps: int) -> Callable[[jax.Array], jax.Array]:
    jax.config.update("jax_enable_x64", val=True)
    return jax.jit(functools.partial(pgd_min_variance, n_steps=n_steps))


def _nearest_psd_host(sigma: Float) -> Float:
    """Materialise the traced nearest-PSD repair for a host NumPy caller.

    :func:`jcor.optimize.psd.nearest_covariance` states its nearest-matrix
    contract at float64 only, so the x64 scope is part of the call rather than
    an ambient assumption. Prefer keeping the covariance on device where the
    consumer is also traced; this exists for the host-side dictionaries the
    emit path builds.

    The diagonal check is the host half of that kernel's documented contract.
    A traced kernel cannot raise, so a nonpositive variance there yields NaN
    rather than an exception — and this function's caller
    (:func:`higham_corner_weights`) is handed ``Sigma_hi``, which
    :func:`robust_corner_weights` advertises as "need not be PSD". Its diagonal
    is in fact ``sigma_sq`` and so positive for any traded ticker, but that is
    a property of the data, not of the type, and the result reaches a published
    column. A crash beats a NaN in a paper table.
    """
    diagonal = np.diag(np.asarray(sigma, dtype=np.float64))
    if not np.all(diagonal > 0.0):
        message = (
            "nearest-PSD repair requires a strictly positive diagonal; got "
            f"min={diagonal.min():.3e}"
        )
        raise RobustPortfolioComputationError(message)
    with jax.enable_x64(new_val=True):
        return np.array(
            nearest_covariance(jnp.asarray(sigma, dtype=jnp.float64)).matrix,
            dtype=np.float64,
        )


def _pgd_weights(sigma: Float, n_steps: int = 1500) -> Float:
    """Long-only simplex PGD weights on a numerically-safe PSD-ridged sigma."""
    # The consumer is jitted; keep the repaired covariance as a JAX value and
    # avoid a host synchronization followed by an immediate device copy.
    sig = ridge_psd(jnp.asarray(sigma), eps=1e-8)
    w = _jit_pgd(n_steps)(sig)
    return np.asarray(w, dtype=np.float64)


def robust_corner_weights(sigma_hi: Float, n_steps: int = 1500) -> Float:
    """Long-only min-variance at the box corner ``U = Sigma_hi`` (T3).

    ``Sigma_hi`` need not be PSD (box cap PSD subset argument, plan
    "backtracking decisions"); :func:`_pgd_weights` applies only the minimal
    numerical ridge needed for the solver, NOT a full projection.
    """
    return _pgd_weights(sigma_hi, n_steps)


def higham_corner_weights(sigma_hi: Float, n_steps: int = 1500) -> Float:
    """Corner weights after a full Higham nearest-PSD projection (heuristic).

    Distinct from :func:`robust_corner_weights`: this moves entries in BOTH
    directions to reach the PSD cone (conservatism gap vs. the exact corner
    is a plan-flagged open question, MC-3 -- not evaluated here).
    """
    return _pgd_weights(_nearest_psd_host(sigma_hi), n_steps)


def robust_ridge_weights(sigma_hat: Float, r: float, n_steps: int = 1500) -> Float:
    """Compute ridge-robust minimum-variance weights."""
    n = sigma_hat.shape[0]
    return _pgd_weights(sigma_hat + r * np.eye(n), n_steps)


def sdp_corner_weights(
    sigma_lo: Float,
    sigma_hi: Float,
    cap: float = 1.0,
    solver: str | None = None,
) -> Float:
    """Exact box-cap-PSD min-max robust weights via a single SDP (T3, tight case).

    Solves ``min_{w in simplex, w<=cap} max_{Sigma_lo<=Sigma<=Sigma_hi, Sigma>=0}
    w'Sigma w`` exactly, i.e. the worst case is restricted to ``box cap PSD``
    (a strict subset of :func:`robust_corner_weights`'s plain box, hence
    weakly TIGHTER -- box cap PSD subseteq box makes the corner-only bound
    valid but conservative; this closes that conservatism gap, MC-3).

    Derivation (standard SDP-duality robust-portfolio result, e.g. Lobo &
    Boyd 2000): for fixed ``w``, ``f(w) = max_Sigma <ww', Sigma>`` over the
    box-cap-PSD set is itself a linear SDP in ``Sigma``; strong duality (the
    box-cap-PSD set has a Slater point, e.g. any strictly-interior-PSD matrix
    in the box) gives
        ``f(w) = min_{Lhi,Llo>=0} <Lhi,Sigma_hi> - <Llo,Sigma_lo>``
        ``s.t. Lhi - Llo - ww' >= 0``
    (elementwise ``Lhi,Llo>=0``, LMI PSD constraint). ``f`` is convex in
    ``w``, so the outer minimisation over ``w`` and the dual variables
    ``(Lhi, Llo)`` is a SINGLE joint SDP (the ``ww'>=0`` LMI constraint is
    linearised via its Schur complement ``[[M, w],[w', 1]] >= 0`` where
    ``M = Lhi - Llo``).

    Args:
        sigma_lo: Box lower bracket, shape ``(n, n)``.
        sigma_hi: Box upper bracket, shape ``(n, n)`` (need not be PSD).
        cap: Per-asset weight cap (long-only box constraint on ``w``).
        solver: cvxpy solver name; defaults to CLARABEL, falling back to SCS.

    Returns:
        Long-only, box-capped weights, shape ``(n,)``, exactly minimising
        the box-cap-PSD worst-case variance.

    """
    n = sigma_lo.shape[0]
    w = cp.Variable(n)
    l_hi = cp.Variable((n, n), symmetric=True)
    l_lo = cp.Variable((n, n), symmetric=True)
    m = l_hi - l_lo
    schur = cp.bmat(
        [
            [m, cp.reshape(w, (n, 1), order="F")],
            [cp.reshape(w, (1, n), order="F"), np.ones((1, 1))],
        ]
    )

    constraints = [
        l_hi >= 0,
        l_lo >= 0,
        schur >> 0,
        w >= 0,
        w <= cap,
        cp.sum(w) == 1,
    ]
    objective = cp.Minimize(
        cp.sum(cp.multiply(l_hi, sigma_hi)) - cp.sum(cp.multiply(l_lo, sigma_lo))
    )
    problem = cp.Problem(objective, cast("Any", constraints))
    solvers = [solver] if solver is not None else ["CLARABEL", "SCS"]
    last_exc: Exception | None = None
    for s in solvers:
        try:
            problem.solve(solver=s)
        except cp.error.SolverError as exc:
            logger.exception(
                "SDP solver %s failed for %d assets with cap=%g; trying next solver",
                s,
                n,
                cap,
            )
            last_exc = exc
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.clip(np.asarray(w.value, dtype=np.float64), 0.0, cap)
    message = (
        f"sdp_corner_weights: no solver reached optimality (status={problem.status})"
    )
    raise RobustPortfolioComputationError(message) from last_exc


@functools.lru_cache(maxsize=4)
def _jit_wdro(n_steps: int) -> Callable[[jax.Array, float], jax.Array]:
    def run(sigma: Float, rho: float) -> Float:
        n = sigma.shape[0]
        eigvals = jnp.linalg.eigvalsh(sigma)
        lam_max = jnp.maximum(jnp.max(eigvals), 1e-10)
        step = 1.0 / (2.0 * lam_max + rho)
        w_init = jnp.ones(n) / n

        def body(_i: int, w: Float) -> Float:
            norm = jnp.sqrt(jnp.sum(w**2) + 1e-12)
            grad = 2.0 * sigma @ w + rho * w / norm
            return projection_simplex(w - step * grad)

        return jax.lax.fori_loop(0, n_steps, body, w_init)

    return jax.jit(run)


def wdro_weights(sigma_hat: Float, rho: float, n_steps: int = 1500) -> Float:
    """Blanchet WDRO min-variance baseline: norm-regularised MV, no OT solver.

    Uses the Blanchet, Kang & Murthy (2019) equivalence for Wasserstein-DRO
    quadratic loss: the type-2 Wasserstein worst-case risk with ambiguity
    radius ``rho`` reduces in closed form to a norm-regularised empirical
    objective -- ``min_w w'Sigma_hat w + rho*||w||_2`` here, a convex penalty
    proportional to ``rho`` with NO inner transport-plan optimisation (no OT
    solver is instantiated at any point; this is the point of the
    equivalence theorem). Solved in pure JAX via subgradient PGD on the
    simplex (``||w||_2`` is non-smooth only at ``w=0``, unreachable on the
    simplex since ``sum(w)=1``).

    Args:
        sigma_hat: Point covariance estimate, shape ``(n, n)``.
        rho: Wasserstein ambiguity radius (regularisation strength).
        n_steps: PGD iteration count.

    Returns:
        Long-only weights, shape ``(n,)``.

    """
    sig = np.asarray(ridge_psd(jnp.asarray(sigma_hat), eps=1e-8), dtype=np.float64)
    w = _jit_wdro(n_steps)(jnp.asarray(sig), float(rho))
    return np.asarray(w, dtype=np.float64)

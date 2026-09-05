"""Portfolio-domain PGD optimizers, plus energy-kernel glue for mixture weights.

``pgd_min_variance``/``pgd_max_quadratic`` are thin, domain-named wrappers
over :mod:`jcor`'s generic simplex-constrained quadratic-form PGD solvers
(:func:`jcor.optimize.pgd_minimize_quadratic_form` /
:func:`jcor.optimize.pgd_maximize_quadratic_form`)
— the names document what the quadratic form *means* at each call site
(portfolio variance; PF-MAX-SPREAD's ``D²`` spread objective), which the
generic jcor primitives deliberately don't encode. ``ridge_psd`` moved to
:mod:`jcor` outright (no domain meaning added by a wrapper); import it from
there directly.

``mixture_weights_qp``/``energy_barycentre_weights_qp`` are unrelated: thin
wrappers over :mod:`jcor.discrepancy.balancing`'s energy-distance QP/PGD
solvers for mixture/barycentre weight selection, kept here because they are
simulation's own glue over jcor, not generic primitives in their own right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Runtime import, deliberately not TYPE_CHECKING-guarded. With `from __future__
# import annotations` a guarded `Float`/`Array` is an unresolvable name at
# runtime; jaxtyping's `_destring_annotation` swallows that and hands beartype
# `Any` for the whole annotation instead of failing, so every shape string below
# would be decoration under `tests/conftest.py`'s `install_import_hook`. The
# probe is recorded in `jcor.core.typing`'s module docstring; not re-derived here.
from jaxtyping import Array, Float  # noqa: TC002
from jcor.discrepancy.balancing import energy_distance_kernel

# `EnergyBarycentreDiagnostics` is un-guarded for the same reason: it shares one
# compound return annotation with `Float[Array, " K"]`, and the fallback above is
# per *annotation*, not per name — guarded, it would keep that shape site dead.
# It rides the statement that already imports `energy_barycentre_weights` at
# runtime, so it costs nothing at import time.
from jcor.geometry import (  # noqa: TC002  # runtime annotation; see comment above
    EnergyBarycentreDiagnostics,
    energy_barycentre_weights,
)
from jcor.optimize import pgd_maximize_quadratic_form, pgd_minimize_quadratic_form

if TYPE_CHECKING:
    # These two stay guarded on purpose: each is a *standalone* parameter
    # annotation, so leaving it unresolvable costs no shape site.
    # `GroundDistanceSelection` is additionally a PEP 695 alias over a protocol
    # union, and un-guarding it would newly enforce the metric vocabulary at
    # simulation's boundary — a behaviour change this module has not measured.
    import jax.numpy as jnp
    from jcor.ground import GroundDistanceSelection


def pgd_min_variance(
    sigma: Float[Array, "K K"],
    n_steps: int = 5000,
) -> Float[Array, " K"]:
    """Minimise portfolio variance ``wᵀΣw`` over the simplex via PGD.

    Thin domain-named wrapper over :func:`jcor.optimize.pgd_minimize_quadratic_form`.

    Args:
        sigma: PSD covariance matrix of shape ``(K, K)``.
        n_steps: Number of PGD iterations.

    Returns:
        Approximate minimiser of shape ``(K,)`` on the simplex.

    """
    return pgd_minimize_quadratic_form(sigma, n_steps=n_steps)


def pgd_max_quadratic(
    q: Float[Array, "K K"],
    n_steps: int = 5000,
) -> Float[Array, " K"]:
    """Maximise ``wᵀQw`` over the simplex via PGD ascent.

    Used for the PF-MAX-SPREAD portfolio with ``Q = D²`` (element-wise
    square of the energy-distance matrix).  Thin domain-named wrapper over
    :func:`jcor.optimize.pgd_maximize_quadratic_form`.

    Args:
        q: Symmetric matrix of shape ``(K, K)``.
        n_steps: Number of PGD ascent iterations.

    Returns:
        Approximate maximiser of shape ``(K,)`` on the simplex.

    """
    return pgd_maximize_quadratic_form(q, n_steps=n_steps)


def mixture_weights_qp(
    target: Float[Array, "m d"],
    candidates: Float[Array, "K m d"] | list[Float[Array, "m d"]],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    solver: str = "boxosqp",
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[Float[Array, " K"], bool]:
    """Find mixture weights minimising energy distance to a target.

    Thin wrapper over :func:`jcor.discrepancy.balancing.energy_distance_kernel` that
    returns ``(weights, success)`` — always ``success=True`` since the PGD
    solver does not raise on convergence failure (it runs for ``maxiter``
    iterations and returns the best iterate).

    Source: hedging script's ``run_target`` in
    ``validation/hedging_error.py``.

    Args:
        target: Target distribution sample cloud, shape ``(m, d)``.
        candidates: ``K`` candidate sample clouds.  Either a stacked array
            ``(K, m, d)`` or a list of ``(m, d)`` arrays (same-size clouds).
        metric: Distance metric for the energy kernel.
        exponent: Distance exponent α ∈ (0, 2).
        solver: Solver alias (``"boxosqp"``, ``"pgd"``, ``"eqcp"`` all map
            to PGD; see :func:`jcor.discrepancy.balancing.energy_distance_kernel`).
        tol: PGD convergence tolerance.
        maxiter: Maximum PGD iterations.

    Returns:
        Tuple ``(weights, success)`` where ``weights`` has shape ``(K,)``
        and ``success`` is always ``True``.

    """
    weights = energy_distance_kernel(
        target,
        candidates,
        metric=metric,
        exponent=exponent,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
    )
    return weights, True


def energy_barycentre_weights_qp(
    target: Float[Array, "m d"],
    # A candidate cloud carries its own sample count, not the target's: the list
    # form is documented below as supporting unequal sample sizes, and
    # `tests/test_optimize.py:99` passes a (2, 1) target against two (1, 1)
    # candidates. Only the ambient dimension `d` is shared, so the stacked form
    # binds `n` and the list form an anonymous axis. The former `"K m d"` /
    # `list[Float[Array, "m d"]]` spelling bound both to the target's `m` and was
    # false — it was simply never checked while the import stayed guarded.
    candidates: Float[Array, "K n d"] | list[Float[Array, "_ d"]],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    *,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[Float[Array, " K"], EnergyBarycentreDiagnostics]:
    """Find the exact finite-mixture energy barycentre.

    This wrapper deliberately differs from :func:`mixture_weights_qp`, which
    preserves the historical kernel-balancing surrogate.  The exact solver
    minimises

    ``2 cᵀw - wᵀQw - b`` over the probability simplex — via a deterministic
    primal-dual interior-point method on the tangent-reduced convex program —
    and returns the numerical diagnostics needed by certificate-producing
    simulation stages.

    Args:
        target: Target distribution sample cloud, shape ``(m, d)``.
        candidates: ``K`` candidate sample clouds, stacked as ``(K, n, d)`` or
            supplied as a list of ``(·, d)`` arrays (which also supports unequal
            sample sizes; only the ambient dimension ``d`` must match
            ``target``).
        metric: Ground distance used in the energy objective.
        exponent: Distance exponent α ∈ (0, 2).
        solver: Solver name.  ``"boxosqp"``, ``"eqcp"`` and ``"pgd"`` remain
            accepted as configuration aliases for the exact interior-point
            solver (``"qp"``) so existing DVC parameters do not silently select
            the legacy objective.
        tol: Frank–Wolfe-gap convergence tolerance.
        maxiter: Maximum projected-gradient iterations.
        validate_cnd: Whether to reject a candidate Gram matrix that is not
            conditionally negative definite on the simplex tangent space.
        cnd_tol: Relative tolerance for the CND numerical check.
        precomputed_components: Optional ``(XX, YX)`` pair to reuse instead
            of recomputing the Gram/cross terms from scratch; see
            :func:`jcor.geometry.energy_barycentre_weights`.

    Returns:
        Pair ``(weights, diagnostics)``.  Callers must inspect
        ``diagnostics.converged`` before using the weights in a certificate.

    """
    exact_solver = "qp" if solver in {"boxosqp", "eqcp", "pgd"} else solver
    weights, diagnostics = energy_barycentre_weights(
        target,
        candidates,
        metric=metric,
        exponent=exponent,
        solver=exact_solver,
        tol=tol,
        maxiter=maxiter,
        validate_cnd=validate_cnd,
        cnd_tol=cnd_tol,
        return_diagnostics=True,
        precomputed_components=precomputed_components,
    )
    return weights, diagnostics

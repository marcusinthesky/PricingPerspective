"""Projected-gradient descent on the probability simplex.

Cross-cutting substrate (t46 stage rank 0): importable by every stage. Pure
JAX, :func:`jax.jit`- and :func:`jax.vmap`-friendly.

**Two deliberately different solvers, one module.** Neither subsumes the other;
picking the wrong one changes both the answer and the cost model.

======================================  =====================  ================
entry point                             objective              loop
======================================  =====================  ================
:func:`pgd_minimize_quadratic_form`     ``min wᵀAw``           ``lax.fori_loop``
:func:`pgd_maximize_quadratic_form`     ``max wᵀAw``           fixed ``n_steps``
:func:`pgd_simplex_affine`              ``min ½wᵀQw + cᵀw``    ``lax.while_loop``
                                                               early stop
:func:`batched_nonnegative_quadratic_`  ``min ½xᵀQx + cᵀx``    fixed cyclic
``program``                             ``x >= 0``              coordinate sweeps
======================================  =====================  ================

* The **pure quadratic** pair steps by ``1/(2·λ_max(A))`` — the Lipschitz
  constant of ``∇_w wᵀAw = 2Aw`` — and runs a *static* iteration count, so the
  whole call traces with no data-dependent control flow. Callers that warm-start
  from the simplex barycentre and do not need early stopping use it
  (minimum-variance and maximum-spread portfolio construction).
* The **affine quadratic** solver carries a linear term, so its gradient is
  ``Qw + c`` and its step is ``1/λ_max(Q)``. It stops early on
  ``||w_new - w||^2 < tol²`` inside a ``lax.while_loop``, which is what makes it
  affordable at ``maxiter`` in the thousands — the energy-barycentre and
  kernel-balancing callers need the tolerance, not a fixed budget.

Both warm-start at uniform ``1/K`` and project with
:func:`optax.projections.projection_simplex`.

History (t46.10): the pure-quadratic half was ``jcor/optimize.py``; the affine
half was ``jcor.kernels._common._pgd_simplex``, parked during wave B at
``jcor.discrepancy._kernels.simplex`` and promoted to a public name here. PSD
repair (``ridge_psd``) moved to :mod:`jcor.optimize.psd`.
"""

from __future__ import annotations

import math
from functools import partial
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from optax.projections import projection_simplex

from jcor.core.typing import (  # noqa: TC001  # runtime; see docstring
    Array,
    Bool,
    Float,
    Int,
    Static,
)

__all__ = [
    "NonnegativeQPSolution",
    "batched_nonnegative_quadratic_program",
    "pgd_maximize_quadratic_form",
    "pgd_minimize_quadratic_form",
    "pgd_simplex_affine",
]


class NonnegativeQPSolution(NamedTuple):
    """Array-only result of :func:`batched_nonnegative_quadratic_program`.

    Every field admits leading batch axes, so the result remains a pytree under
    nested :func:`jax.jit` and :func:`jax.vmap` transforms. ``active_mask`` uses
    the caller's absolute ``active_tolerance`` and marks *binding* coordinates
    (``x <= active_tolerance``). ``projected_gradient_norm`` is

    ``||x - max(x - (Qx + c), 0)||_inf``.

    ``flat_mask`` identifies coordinate directions whose diagonal curvature is
    negligible relative to ``Q``. A flat direction with a negative linear cost
    is an unbounded-like input and sets ``bounded=False``; a flat NNLS column
    with zero cost is valid and is held deterministically at zero. ``bounded``
    is deliberately a coordinate-flat diagnostic, not a general recession-cone
    certificate for every singular convex QP. Eager callers must inspect all
    validity and convergence flags before consuming an iterate.

    Attributes:
        primal: Deterministic nonnegative iterate.
        objective: ``0.5 * x.T @ Q @ x + c.T @ x``; NaN for rejected lanes.
        active_mask: Coordinates at or below ``active_tolerance``.
        projected_gradient_norm: Infinity norm of the projected-KKT residual.
        sweeps: First converged full coordinate sweep, or ``max_sweeps``.
        converged: Whether the final scaled projected-KKT check passed.
        input_finite: Whether both ``Q`` and the lane's ``c`` are finite.
        convex: Whether symmetric ``Q`` is positive semidefinite to tolerance.
        bounded: Whether no rejected coordinate-flat descent direction exists.
        flat_mask: Coordinate-flat directions of symmetric ``Q``.

    """

    primal: Float[Array, "*batch K"]
    objective: Float[Array, "*batch"]
    active_mask: Bool[Array, "*batch K"]
    projected_gradient_norm: Float[Array, "*batch"]
    sweeps: Int[Array, "*batch"]
    converged: Bool[Array, "*batch"]
    input_finite: Bool[Array, "*batch"]
    convex: Bool[Array, "*batch"]
    bounded: Bool[Array, "*batch"]
    flat_mask: Bool[Array, "*batch K"]


def _projected_kkt_norm(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, " K"],
    primal: Float[Array, " K"],
) -> Float[Array, ""]:
    """Return the infinity norm of the nonnegative-QP projected residual."""
    gradient = quadratic @ primal + linear
    residual = primal - jnp.maximum(primal - gradient, 0.0)
    return jnp.max(jnp.abs(residual))


def _lane_linear_terms(
    flat_mask: Bool[Array, " K"],
    quadratic_finite: Bool[Array, ""],
    convex: Bool[Array, ""],
    linear: Float[Array, " K"],
    linear_tolerance: Static[float],
) -> tuple[Bool[Array, ""], Bool[Array, ""], Float[Array, " K"]]:
    """Return one lane's finiteness flag, boundedness flag and sanitized cost."""
    linear_finite = jnp.all(jnp.isfinite(linear))
    input_finite = jnp.logical_and(quadratic_finite, linear_finite)
    safe_linear = jnp.where(linear_finite, linear, jnp.zeros_like(linear))
    linear_scale = 1.0 + jnp.max(jnp.abs(safe_linear))
    flat_descent = jnp.any(
        jnp.logical_and(flat_mask, safe_linear < -linear_tolerance * linear_scale)
    )
    bounded = jnp.logical_and(
        jnp.logical_and(input_finite, convex),
        jnp.logical_not(flat_descent),
    )
    # Rejected lane-specific linear terms become zero. The shared quadratic was
    # sanitized once before ``vmap``; keeping it lane-invariant avoids materializing
    # one ``K x K`` copy per Monte Carlo draw.
    return (
        input_finite,
        bounded,
        jnp.where(bounded, safe_linear, jnp.zeros_like(safe_linear)),
    )


def _lane_solution(
    quadratic: Float[Array, "K K"],
    safe_linear: Float[Array, " K"],
    iterate: Float[Array, " K"],
    first_sweep: Int[Array, ""],
    flat_mask: Bool[Array, " K"],
    input_finite: Bool[Array, ""],
    convex: Bool[Array, ""],
    valid: Bool[Array, ""],
    max_sweeps: Static[int],
    tolerance: Static[float],
    active_tolerance: Static[float],
) -> NonnegativeQPSolution:
    """Return the lane result and KKT diagnostics for a finished iterate."""
    residual = _projected_kkt_norm(quadratic, safe_linear, iterate)
    threshold = tolerance * (
        1.0 + jnp.max(jnp.abs(safe_linear)) + jnp.max(jnp.abs(iterate))
    )
    converged = jnp.logical_and(valid, residual <= threshold)
    objective = 0.5 * iterate @ quadratic @ iterate + safe_linear @ iterate
    safe_objective = jnp.where(
        valid, objective, jnp.asarray(jnp.nan, dtype=objective.dtype)
    )
    safe_residual = jnp.where(
        valid, residual, jnp.asarray(jnp.inf, dtype=residual.dtype)
    )
    primal = jnp.where(valid, iterate, jnp.zeros_like(iterate))
    return NonnegativeQPSolution(
        primal=primal,
        objective=safe_objective,
        active_mask=primal <= active_tolerance,
        projected_gradient_norm=safe_residual,
        sweeps=jnp.where(converged, first_sweep, max_sweeps),
        converged=converged,
        input_finite=input_finite,
        convex=convex,
        bounded=valid,
        flat_mask=flat_mask,
    )


def _solve_nonnegative_quadratic_program_lane(
    quadratic: Float[Array, "K K"],
    diagonal: Float[Array, " K"],
    flat_mask: Bool[Array, " K"],
    quadratic_finite: Bool[Array, ""],
    convex: Bool[Array, ""],
    linear: Float[Array, " K"],
    max_sweeps: Static[int],
    tolerance: Static[float],
    active_tolerance: Static[float],
    linear_tolerance: Static[float],
) -> NonnegativeQPSolution:
    """Solve one lane with fixed-work cyclic projected coordinate descent."""
    input_finite, valid, safe_linear = _lane_linear_terms(
        flat_mask, quadratic_finite, convex, linear, linear_tolerance
    )
    dimension = quadratic.shape[0]

    primal0 = jnp.zeros_like(safe_linear)
    residual0 = _projected_kkt_norm(quadratic, safe_linear, primal0)
    threshold0 = tolerance * (1.0 + jnp.max(jnp.abs(safe_linear)))
    initially_converged = residual0 <= threshold0
    first_sweep0 = jnp.where(initially_converged, 0, max_sweeps)

    def sweep(
        sweep_index: Int[Array, ""],
        state: tuple[Float[Array, " K"], Int[Array, ""]],
    ) -> tuple[Float[Array, " K"], Int[Array, ""]]:
        primal, first_sweep = state

        def coordinate(
            coordinate_index: Int[Array, ""],
            current: Float[Array, " K"],
        ) -> Float[Array, " K"]:
            gradient = (
                quadratic[coordinate_index] @ current + safe_linear[coordinate_index]
            )
            candidate = jnp.maximum(
                0.0,
                current[coordinate_index] - gradient / diagonal[coordinate_index],
            )
            candidate = jnp.where(flat_mask[coordinate_index], 0.0, candidate)
            return current.at[coordinate_index].set(candidate)

        updated = lax.fori_loop(0, dimension, coordinate, primal)
        residual = _projected_kkt_norm(quadratic, safe_linear, updated)
        threshold = tolerance * (
            1.0 + jnp.max(jnp.abs(safe_linear)) + jnp.max(jnp.abs(updated))
        )
        converged_now = residual <= threshold
        first_sweep = jnp.where(
            jnp.logical_and(first_sweep == max_sweeps, converged_now),
            sweep_index + 1,
            first_sweep,
        )
        return updated, first_sweep

    primal, first_sweep = lax.fori_loop(
        0,
        max_sweeps,
        sweep,
        (primal0, jnp.asarray(first_sweep0, dtype=jnp.int32)),
    )
    return _lane_solution(
        quadratic,
        safe_linear,
        primal,
        first_sweep,
        flat_mask,
        input_finite,
        convex,
        valid,
        max_sweeps,
        tolerance,
        active_tolerance,
    )


class _SanitizedQuadratic(NamedTuple):
    """Lane-invariant quadratic data prepared once before the batched ``vmap``.

    Attributes:
        quadratic: Symmetrized quadratic, replaced by the identity when the
            input is nonfinite or fails the PSD check.
        diagonal: Coordinate curvatures used as exact-minimizer denominators,
            replaced by ones wherever they are unusable.
        flat_mask: Coordinate-flat directions of the symmetrized quadratic.
        finite: Whether the symmetrized quadratic is finite.
        convex: Whether the symmetrized quadratic is PSD to tolerance.

    """

    quadratic: Float[Array, "K K"]
    diagonal: Float[Array, " K"]
    flat_mask: Bool[Array, " K"]
    finite: Bool[Array, ""]
    convex: Bool[Array, ""]


def _sanitize_quadratic(
    quadratic: Float[Array, "K K"],
    curvature_tolerance: Static[float],
) -> _SanitizedQuadratic:
    """Return the symmetrized quadratic, its curvatures and its validity flags."""
    symmetric = 0.5 * (quadratic + quadratic.T)
    quadratic_finite = jnp.all(jnp.isfinite(symmetric))
    safe_symmetric = jnp.where(
        quadratic_finite,
        symmetric,
        jnp.eye(symmetric.shape[0], dtype=symmetric.dtype),
    )
    curvature_scale = jnp.maximum(1.0, jnp.max(jnp.abs(safe_symmetric)))
    curvature_floor = curvature_tolerance * curvature_scale
    diagonal = jnp.diag(safe_symmetric)
    flat_mask = jnp.abs(diagonal) <= curvature_floor
    minimum_eigenvalue = jnp.min(jnp.linalg.eigvalsh(safe_symmetric))
    convex = jnp.logical_and(
        quadratic_finite,
        minimum_eigenvalue >= -curvature_floor,
    )
    quadratic_usable = jnp.logical_and(quadratic_finite, convex)
    return _SanitizedQuadratic(
        quadratic=jnp.where(
            quadratic_usable,
            safe_symmetric,
            jnp.eye(symmetric.shape[0], dtype=symmetric.dtype),
        ),
        diagonal=jnp.where(
            jnp.logical_and(quadratic_usable, jnp.logical_not(flat_mask)),
            diagonal,
            jnp.ones_like(diagonal),
        ),
        flat_mask=flat_mask,
        finite=quadratic_finite,
        convex=convex,
    )


@partial(
    jax.jit,
    static_argnames=(
        "max_sweeps",
        "tolerance",
        "active_tolerance",
        "curvature_tolerance",
        "linear_tolerance",
    ),
)
def _batched_nonnegative_quadratic_program_jit(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, "*batch K"],
    *,
    max_sweeps: Static[int],
    tolerance: Static[float],
    active_tolerance: Static[float],
    curvature_tolerance: Static[float],
    linear_tolerance: Static[float],
) -> NonnegativeQPSolution:
    """Jitted implementation of :func:`batched_nonnegative_quadratic_program`."""
    prepared = _sanitize_quadratic(quadratic, curvature_tolerance)

    dimension = quadratic.shape[0]
    batch_shape = linear.shape[:-1]
    flat_linear = linear.reshape((-1, dimension))
    flat_solution = jax.vmap(
        _solve_nonnegative_quadratic_program_lane,
        in_axes=(None, None, None, None, None, 0, None, None, None, None),
    )(
        prepared.quadratic,
        prepared.diagonal,
        prepared.flat_mask,
        prepared.finite,
        prepared.convex,
        flat_linear,
        max_sweeps,
        tolerance,
        active_tolerance,
        linear_tolerance,
    )
    return NonnegativeQPSolution(
        primal=flat_solution.primal.reshape((*batch_shape, dimension)),
        objective=flat_solution.objective.reshape(batch_shape),
        active_mask=flat_solution.active_mask.reshape((*batch_shape, dimension)),
        projected_gradient_norm=flat_solution.projected_gradient_norm.reshape(
            batch_shape
        ),
        sweeps=flat_solution.sweeps.reshape(batch_shape),
        converged=flat_solution.converged.reshape(batch_shape),
        input_finite=flat_solution.input_finite.reshape(batch_shape),
        convex=flat_solution.convex.reshape(batch_shape),
        bounded=flat_solution.bounded.reshape(batch_shape),
        flat_mask=flat_solution.flat_mask.reshape((*batch_shape, dimension)),
    )


def batched_nonnegative_quadratic_program(
    quadratic: Float[Array, "K K"],
    linear: Float[Array, "*batch K"],
    *,
    max_sweeps: Static[int] = 256,
    tolerance: Static[float] = 1e-10,
    active_tolerance: Static[float] = 1e-10,
    curvature_tolerance: Static[float] = 1e-12,
    linear_tolerance: Static[float] = 1e-12,
) -> NonnegativeQPSolution:
    """Solve batched convex nonnegative QPs with a fixed transformable budget.

    For every leading batch lane in ``linear``, solve

    ``minimize 0.5 * x.T @ quadratic @ x + linear.T @ x, subject to x >= 0``.

    One full sweep applies the exact projected minimizer for each coordinate in
    order (cyclic Gauss--Seidel coordinate descent). The loop always executes
    ``max_sweeps`` sweeps, while ``sweeps`` records the first scaled
    projected-KKT success. Static configuration and array-only diagnostics keep
    the contract composable under ``make_jaxpr``, ``jit``, and an outer ``vmap``.
    This solver is piecewise differentiable away from active-set transitions;
    no gradient guarantee is made at those nonsmooth boundaries.

    Args:
        quadratic: Shared symmetric positive-semidefinite matrix, shape ``(K, K)``.
        linear: Linear terms with shape ``(*batch, K)``; ``(K,)`` is one lane.
        max_sweeps: Static number of complete cyclic coordinate sweeps.
        tolerance: Relative projected-KKT infinity-norm tolerance.
        active_tolerance: Absolute threshold for the returned binding mask.
        curvature_tolerance: Relative threshold for PSD and flat-coordinate checks.
        linear_tolerance: Relative threshold for flat negative-cost detection.

    Returns:
        :class:`NonnegativeQPSolution` with the same leading batch shape as
        ``linear``. The compiled core never raises; eager inference adapters
        decide whether invalid or unconverged lanes are usable.

    Raises:
        ValueError: If a static configuration is non-positive or array shapes
            do not describe one square quadratic and matching linear terms.

    """
    expected_rank = 2
    if (
        quadratic.ndim != expected_rank
        or quadratic.shape[0] != quadratic.shape[1]
        or quadratic.shape[0] < 1
    ):
        message = "quadratic must be a nonempty square rank-2 array."
        raise ValueError(message)
    if linear.ndim < 1 or linear.shape[-1] != quadratic.shape[0]:
        message = "linear must have shape (*batch, quadratic.shape[0])."
        raise ValueError(message)
    if (
        isinstance(max_sweeps, bool)
        or not isinstance(max_sweeps, Integral)
        or max_sweeps < 1
    ):
        message = "max_sweeps must be a positive integer."
        raise ValueError(message)
    tolerances = (
        tolerance,
        active_tolerance,
        curvature_tolerance,
        linear_tolerance,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0.0
        for value in tolerances
    ):
        message = "solver tolerances must be finite and strictly positive."
        raise ValueError(message)
    return _batched_nonnegative_quadratic_program_jit(
        quadratic,
        linear,
        max_sweeps=int(max_sweeps),
        tolerance=float(tolerance),
        active_tolerance=float(active_tolerance),
        curvature_tolerance=float(curvature_tolerance),
        linear_tolerance=float(linear_tolerance),
    )


@partial(jax.jit, static_argnums=(1, 2))
def _pgd_simplex_quadratic(
    a: Float[Array, "K K"],
    n_steps: int,
    sign: float,
) -> Float[Array, " K"]:
    """Shared PGD loop for :func:`pgd_minimize_quadratic_form`/`_maximize_`.

    Step size is ``1 / (2 · λ_max(A))``, the Lipschitz-optimal constant step
    for the gradient ``∇_w wᵀAw = 2Aw``.  Warm-start: uniform ``1/K``.

    Returns:
        Approximate optimum of shape ``(K,)`` on the simplex.

    """
    k = a.shape[0]
    eigvals = jnp.linalg.eigvalsh(a)
    lam_max = jnp.maximum(jnp.max(eigvals), 1e-10)
    step = 1.0 / (2.0 * lam_max)
    w_init = jnp.ones(k) / k

    # `_i` is the fori_loop-traced counter, not a Python int (t46.1).
    def body(_i: Int[Array, ""], w: Float[Array, " K"]) -> Float[Array, " K"]:
        grad = 2.0 * a @ w
        return projection_simplex(w + sign * step * grad)

    return jax.lax.fori_loop(0, n_steps, body, w_init)


def pgd_minimize_quadratic_form(
    a: Float[Array, "K K"],
    n_steps: int = 5000,
) -> Float[Array, " K"]:
    """Minimise ``wᵀAw`` over the probability simplex via projected gradient descent.

    Args:
        a: PSD matrix of shape ``(K, K)``.
        n_steps: Number of PGD iterations (fixed; no early stopping).

    Returns:
        Approximate minimiser of shape ``(K,)`` on the simplex.

    """
    return _pgd_simplex_quadratic(a, n_steps, sign=-1.0)


def pgd_maximize_quadratic_form(
    a: Float[Array, "K K"],
    n_steps: int = 5000,
) -> Float[Array, " K"]:
    """Maximise ``wᵀAw`` over the probability simplex via projected gradient ascent.

    Args:
        a: Symmetric matrix of shape ``(K, K)``.
        n_steps: Number of PGD iterations (fixed; no early stopping).

    Returns:
        Approximate maximiser of shape ``(K,)`` on the simplex.

    """
    return _pgd_simplex_quadratic(a, n_steps, sign=+1.0)


def pgd_simplex_affine(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int = 1000,
    tol: float = 1e-6,
) -> Float[Array, " K"]:
    """Minimise ``½ wᵀQw + cᵀw`` over the probability simplex, stopping on step norm.

    Step size is ``1/λ_max(Q)`` (the gradient Lipschitz constant).  Projection
    onto the probability simplex (``sum = 1``, ``w >= 0``) uses
    :func:`optax.projections.projection_simplex`.

    Distinct from :func:`pgd_minimize_quadratic_form`, which drops the linear
    term ``c`` and runs a fixed iteration count instead of stopping on the step
    norm — see the module docstring.

    Args:
        quadratic: Positive-semi-definite matrix of shape ``(K, K)``.
        c: Linear cost vector of shape ``(K,)``.
        maxiter: Maximum number of PGD iterations.
        tol: Convergence tolerance (L2 norm of the step).

    Returns:
        Approximate minimiser of shape ``(K,)`` satisfying the simplex constraints.

    """
    # Convergence criterion is step-norm based (||w_new - w||² < tol²); this may
    # cause early termination without converging to the true optimum on extremely
    # ill-conditioned Q (condition number λ_max/λ_min ≫ 1e4), where oscillation
    # can produce a small step norm before reaching the simplex-projected minimum.
    return _pgd_simplex_affine_jit(quadratic, c, int(maxiter), float(tol))


@partial(jax.jit, static_argnums=(2, 3))
def _pgd_simplex_affine_jit(
    quadratic: Float[Array, "K K"],
    c: Float[Array, " K"],
    maxiter: int,
    tol: float,
) -> Float[Array, " K"]:
    """Jitted PGD-on-simplex loop (see :func:`pgd_simplex_affine`).

    The iteration, step size, warm start and step-norm stopping rule are
    identical to the reference eager loop; running it inside ``lax.while_loop``
    removes the per-iteration host sync (the old ``float(...)`` convergence
    check blocked on the device every step) and the Python dispatch overhead,
    which dominated runtime once ``maxiter`` reached the thousands.

    Returns:
        Approximate minimiser of shape ``(K,)`` on the probability simplex.

    """
    eigvals = jnp.linalg.eigvalsh(quadratic)
    lam_max = jnp.maximum(jnp.max(eigvals), 1e-10)
    step = 1.0 / lam_max

    candidate_count = quadratic.shape[0]
    w0 = jnp.ones(candidate_count) / candidate_count
    tol_sq = tol**2

    def cond(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        _, i, step_sq = state
        return jnp.logical_and(i < maxiter, step_sq >= tol_sq)

    def body(
        state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        w, i, _ = state
        w_new = projection_simplex(w - step * (quadratic @ w + c))
        return w_new, i + 1, jnp.sum((w_new - w) ** 2)

    w, _, _ = lax.while_loop(cond, body, (w0, 0, jnp.array(jnp.inf)))
    return w

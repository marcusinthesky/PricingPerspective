"""Shared orthant-projection refinement for the Wolak and GMS solver boundaries.

Both :mod:`jcor.inference.wolak` and :mod:`jcor.inference.gms` project onto the
non-negative orthant in an inverse-covariance metric, and both consume
:func:`jcor.optimize.simplex.batched_nonnegative_quadratic_program`. That shared
solver runs a *fixed* number of cyclic coordinate sweeps under ``lax.fori_loop``,
so its convergence is linear at rate ``~(1 - 1/cond(Q))`` per sweep with no early
exit.

**Why that is not enough.** The retired ``scipy.optimize.nnls`` path was
Lawson--Hanson, a *finite* active-set method that terminates at the exact
solution. Replacing it with fixed-budget coordinate descent lost that property,
and on real moment covariances no affordable budget recovers it: on the Paper 3
chi-bar calibration (``m = 15``, 3000 draws, ``cond(Q) = 1840``) the compiled
solver leaves 940/3000 lanes above the projected-KKT tolerance at 4096 sweeps,
and needs 16384 to reach 0/3000.

**What this is not.** The defect is *not* squared conditioning: for SPD ``S``,
``cond(S⁻¹) = cond(S)`` identically, and the measured value is 1840, not a
blow-up. Jacobi (diagonal) preconditioning was implemented and rejected on
measurement --- ``cond(diag(Q)^-1/2 Q diag(Q)^-1/2) = 1884.8`` is marginally
*worse* than ``cond(Q) = 1840.4``, because the conditioning is not diagonal. Do
not re-attempt it.

This module restores finite active-set termination as a *refinement* on top of
the compiled iterate, at the eager inference boundary, leaving
``jcor.optimize.simplex``'s traced contract untouched so other consumers are
unaffected.

The refinement is a deliberate eager host step: it runs a bounded Python loop
over batch ``jnp.linalg.solve`` calls. Its float64 inputs are preserved by the
caller owning ``jax_enable_x64``, not by any scope this module opens. It is not
itself ``jit``-traced, but it returns plain JAX leaves so the eager consumers
own exactly one host materialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax  # noqa: TC002  # runtime: beartype resolves `jax.Array` annotations
import jax.numpy as jnp
from jax import lax

from jcor.core.typing import ArrayLike  # noqa: TC001  # public signature

if TYPE_CHECKING:
    from jcor.optimize.simplex import NonnegativeQPSolution

#: Relative projected-KKT tolerance requested from the shared solver.
#:
#: Passed explicitly by callers rather than defaulted, so that the convergence
#: verdict recomputed by :func:`refine_projection_solution` uses the identical
#: threshold the compiled solver would have applied. Nothing here loosens that
#: criterion.
PROJECTION_TOLERANCE = 1e-10

#: Active-set refinement rounds applied to the compiled solver's iterate.
#:
#: Two rounds suffice for the transformable Wolak Paper 3 MC kernel, but eager
#: Wolak inference keeps this conservative default because Paper 2 includes
#: less-well-scaled one-lane frontier projections. The GMS QLR projection also
#: needs the full budget because its slack recentering
#: (``standardized[:, slack] += 1e6 * max(sig)``) makes many free-face solves
#: infeasible, so most rounds spend their step on a ratio test rather than
#: reaching the face minimum. Measured on the real 3001-lane GMS batch: 8
#: rounds leave 3 lanes, 16 leave 3, and 32 reach 0. 32 costs ~340 ms there.
POLISH_ROUNDS = 32

#: Relative threshold for releasing a binding coordinate whose multiplier is
#: negative. Matches the shared solver's ``linear_tolerance`` scale convention.
RELEASE_TOLERANCE = 1e-12


def projected_kkt_residual(
    quadratic: ArrayLike,
    linear: ArrayLike,
    primal: ArrayLike,
) -> jax.Array:
    """Batched ``||x - max(x - (Qx + c), 0)||_inf``, the solver's own residual.

    Args:
        quadratic: Symmetric quadratic form, shape ``(K, K)``.
        linear: Per-lane linear terms, shape ``(B, K)``.
        primal: Per-lane iterates, shape ``(B, K)``.

    Returns:
        Per-lane projected-KKT infinity norm, shape ``(B,)``.

    """
    quadratic = jnp.asarray(quadratic)
    linear = jnp.asarray(linear)
    primal = jnp.asarray(primal)
    gradient = primal @ quadratic + linear
    return jnp.max(jnp.abs(primal - jnp.maximum(primal - gradient, 0.0)), axis=-1)


def _ratio_test_candidate(
    current: jax.Array,
    free: jax.Array,
    face_minimum: jax.Array,
) -> jax.Array:
    """Take one Lawson--Hanson ratio-test step toward the free-face minimum.

    Advance toward the face minimum only as far as the orthant allows, so a
    blocking coordinate lands exactly on zero. JAX never raises on a zero
    denominator; the resulting inf/nan is filtered by the ``isfinite`` step
    below, matching the retired ``np.errstate`` semantics without a NumPy
    context manager.

    Args:
        current: Per-lane iterates, shape ``(B, K)``.
        free: Per-lane free-set mask, shape ``(B, K)``.
        face_minimum: Per-lane free-face minimizers, shape ``(B, K)``.

    Returns:
        The feasible candidate iterate, shape ``(B, K)``.

    """
    blocking = free & (face_minimum < 0.0)
    ratios = jnp.where(
        blocking & (current > 0.0),
        current / (current - face_minimum),
        jnp.inf,
    )
    step = jnp.min(ratios, axis=-1)
    step = jnp.where(jnp.isfinite(step), step, 1.0)
    direction = jnp.where(free, face_minimum - current, 0.0)
    return jnp.maximum(current + step[:, None] * direction, 0.0)


def polish_active_set(
    quadratic: ArrayLike,
    linear: ArrayLike,
    primal: ArrayLike,
    flat_mask: ArrayLike,
    *,
    rounds: int = POLISH_ROUNDS,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Finish a cyclic-coordinate-descent iterate with exact free-face solves.

    Each round

    1. estimates the free set as coordinates that are strictly positive **or**
       binding with a negative multiplier (the *release* rule -- omitting it
       makes the free set monotonically shrink and the iteration stall, measured
       as a stall at 34/3000 lanes);
    2. solves the equality-constrained subproblem ``Q_FF x_F = -c_F`` exactly on
       that face via one batched ``jnp.linalg.solve`` over ``(B, K, K)`` -- no
       Python loop over lanes -- holding the complement at zero. A coordinate
       already at zero that the face solve wants to push negative is simply not
       released this round, and the face is re-solved without it. That decision
       is deliberately *memoryless*: a persistent pin set over-constrains and
       leaves a lane permanently stuck (measured: 774 -> 1 stalled lane with a
       persistent pin, 774 -> 0 without it);
    3. takes a **Lawson--Hanson ratio-test step** toward that face minimum
       rather than clipping to it. Clipping is what stalls: on the real GMS
       batch *every one* of the 774 stalled lanes had a negative free-face
       component, so the clipped point was no better and was rejected, forever.
       The ratio test instead advances only as far as feasibility allows,
       driving the blocking coordinate exactly to zero, which is guaranteed
       progress;
    4. keeps the result only if it *strictly reduces* the projected-KKT
       residual.

    Acceptance is graded on the residual rather than the objective on purpose.
    Near the optimum the objective improvement falls below float64 resolution
    (the objective is ``O(1)`` while the remaining gap is ``<< 1e-16``), so an
    objective-based test rejects strictly better iterates and stalls.

    Because step 4 is monotone, this can never return a worse iterate than the
    solver's, and it solves the same mathematical program: the feasible set, the
    quadratic form, and the linear term are all untouched.

    A singular free face is handled explicitly: a lane whose face system is
    exactly singular makes the shared ``jnp.linalg.solve`` return non-finite
    values (the same LAPACK ``dgesv`` that ``np.linalg.solve`` surfaces as a
    ``LinAlgError`` for the whole batch), so the round bails and refinement
    stops with whatever iterate was best so far. That is fail-safe rather than
    fail-open --- the caller's eager check still grades the unrefined iterate
    and rejects it if it does not meet the tolerance.

    Args:
        quadratic: Symmetric positive-semidefinite form, shape ``(K, K)``.
        linear: Per-lane linear terms, shape ``(B, K)``.
        primal: Compiled solver's iterate, shape ``(B, K)``.
        flat_mask: Zero-curvature coordinates, held at zero, shape ``(B, K)``.
        rounds: Static upper bound on active-set refinement rounds. The
            measured transformable Wolak MC kernel uses two; eager Wolak and
            GMS retain the conservative shared default of 32.

    Returns:
        Tuple of the refined primal, its projected-KKT residual, and a mask of
        the lanes the refinement actually improved.

    """
    quadratic = jnp.asarray(quadratic)
    linear = jnp.asarray(linear)
    primal = jnp.asarray(primal)
    flat_mask = jnp.asarray(flat_mask, dtype=bool)
    identity = jnp.eye(quadratic.shape[0], dtype=jnp.float64)
    release_floor = RELEASE_TOLERANCE * (
        1.0 + jnp.max(jnp.abs(linear), axis=-1, keepdims=True)
    )

    def solve_face(free: jax.Array) -> jax.Array:
        face = free[:, :, None] & free[:, None, :]
        system = jnp.where(face, quadratic[None, :, :], identity[None, :, :])
        rhs = jnp.where(free, -linear, 0.0)
        return jnp.linalg.solve(system, rhs[:, :, None])[:, :, 0]

    def estimate_free_face(
        current: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        gradient = current @ quadratic + linear
        free = jnp.logical_and(
            jnp.logical_or(current > 0.0, gradient < -release_floor),
            jnp.logical_not(flat_mask),
        )
        face_minimum = solve_face(free)
        finite = jnp.all(jnp.isfinite(face_minimum))
        # Do not release a coordinate that is already at zero and that the
        # face minimum would drive negative; re-solve without it. Memoryless
        # by design -- it may be released again on a later round.
        dropped = free & (face_minimum < 0.0) & (current <= 0.0)
        reduced_free = free & ~dropped
        reduced_minimum = solve_face(reduced_free)
        use_reduced = jnp.any(dropped)
        free = jnp.where(use_reduced, reduced_free, free)
        face_minimum = jnp.where(use_reduced, reduced_minimum, face_minimum)
        return free, face_minimum, finite & jnp.all(jnp.isfinite(face_minimum))

    best = primal
    best_residual = projected_kkt_residual(quadratic, linear, best)
    refined = jnp.zeros(best_residual.shape, dtype=bool)
    current = primal

    def polish_round(
        _index: jax.Array,
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        best, best_residual, refined, current, running = carry
        free, face_minimum, finite = estimate_free_face(current)
        # Lawson--Hanson ratio test toward that face minimum.
        candidate = _ratio_test_candidate(current, free, face_minimum)
        active = running & finite
        candidate = jnp.where(active, candidate, current)
        residual = projected_kkt_residual(quadratic, linear, candidate)
        improved = active & (residual < best_residual)
        best = jnp.where(improved[:, None], candidate, best)
        best_residual = jnp.where(improved, residual, best_residual)
        refined = refined | improved
        return best, best_residual, refined, candidate, active

    best, best_residual, refined, _, _ = lax.fori_loop(
        0,
        rounds,
        polish_round,
        (best, best_residual, refined, current, jnp.ones((), dtype=bool)),
    )
    return best, best_residual, refined


def refine_projection_solution(
    solution: NonnegativeQPSolution,
    quadratic: ArrayLike,
    linear: ArrayLike,
    *,
    active_tolerance: float,
    polish_rounds: int = POLISH_ROUNDS,
) -> NonnegativeQPSolution:
    """Re-derive the primal, mask, residual, and verdict after active-set polish.

    ``input_finite``, ``convex``, ``bounded``, and ``flat_mask`` are the compiled
    solver's and are carried through untouched: the polish refines an *iterate*,
    it does not re-adjudicate whether the *problem* was admissible. A lane the
    solver rejected as nonfinite, nonconvex, or unbounded-like keeps its
    deterministic zero iterate and infinite residual and can never be
    resurrected here.

    ``converged`` is recomputed with the *same* formula and the *same*
    :data:`PROJECTION_TOLERANCE` the solver applies
    (``residual <= tolerance * (1 + ||c||_inf + ||x||_inf)``). This grades a
    strictly better iterate by an unchanged criterion; it does not loosen one.
    The solver's verdict is overturned only for a lane the refinement measurably
    improved, so an injected or genuine non-convergence on an already-optimal
    iterate still fails closed.

    Args:
        solution: Result of the compiled batched nonnegative-QP solve.
        quadratic: The symmetric quadratic form actually solved, shape ``(K, K)``.
        linear: Per-lane linear terms, shape ``(B, K)``.
        active_tolerance: Absolute binding threshold for the returned mask.
        polish_rounds: Static active-set refinement budget passed to
            :func:`polish_active_set`.

    Returns:
        The solution with refined primal and re-derived diagnostics.

    """
    quadratic = jnp.asarray(quadratic)
    linear = jnp.asarray(linear)
    valid = (
        jnp.asarray(solution.input_finite, dtype=bool)
        & jnp.asarray(solution.convex, dtype=bool)
        & jnp.asarray(solution.bounded, dtype=bool)
    )
    primal, residual, refined = polish_active_set(
        quadratic,
        linear,
        solution.primal,
        solution.flat_mask,
        rounds=polish_rounds,
    )
    primal = jnp.where(valid[:, None], primal, 0.0)
    residual = jnp.where(valid, residual, jnp.inf)
    threshold = PROJECTION_TOLERANCE * (
        1.0 + jnp.max(jnp.abs(linear), axis=-1) + jnp.max(jnp.abs(primal), axis=-1)
    )
    converged = jnp.asarray(solution.converged, dtype=bool)
    converged = valid & (residual <= threshold) & (converged | refined)
    return solution._replace(  # NamedTuple public API
        primal=primal,
        objective=jnp.where(
            valid,
            0.5 * jnp.einsum("bi,ij,bj->b", primal, quadratic, primal)
            + jnp.einsum("bi,bi->b", linear, primal),
            jnp.nan,
        ),
        active_mask=primal <= active_tolerance,
        projected_gradient_norm=residual,
        converged=converged,
    )

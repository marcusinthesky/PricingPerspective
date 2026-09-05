"""Pure-JAX empirical Wasserstein barycentre solvers.

The functions in this module operate on one dense array of source supports:
``(n_sources, n_points, n_dimensions)``.  They deliberately do not know about
cloud IDs, registries, or artifact schemas.  The pipeline owns those concerns;
this module owns only the alternating Hungarian geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

import jax
import jax.numpy as jnp
import optax

from jcor.core.axioms import Law  # noqa: TC001  # runtime contract
from jcor.core.typing import Array, ArrayLike, Float  # noqa: TC001
from jcor.ground.config import (  # noqa: TC001  # runtime contract
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.ground.metrics import EUCLIDEAN, GroundDistance, cdist  # noqa: TC001

__all__ = [
    "WassersteinBarycentreResult",
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
]

_MINIMUM_SOURCES: Final = 2
_TWO_SOURCES: Final = 2
_SOURCE_ARRAY_NDIM: Final = 3
_WASSERSTEIN_ORDER_ONE: Final = 1
_WASSERSTEIN_ORDER_TWO: Final = 2
_DEFAULT_MAX_ITERATIONS: Final = 128
_SIMPLEX_ATOL: Final = 1e-6


@dataclass(frozen=True, slots=True)
class WassersteinBarycentreResult:
    """Support and diagnostics returned by an empirical barycentre solver."""

    support_values: Float[Array, "n d"]
    source_weights: Float[Array, " k"]
    objective: float
    initial_objective: float
    optimality_gap: float
    constraint_violation: float
    converged: bool
    iterations: int
    objective_semantics: str
    #: Sup-norm distance from the returned support to the alternating map's
    #: fixed point: exactly ``0.0`` once the permutations repeat (one more sweep
    #: then reproduces the support), otherwise the last observed displacement as
    #: a proxy.  This is the honest residual; ``optimality_gap`` is only the
    #: improvement over the starting support and certifies nothing about
    #: optimality, global or local.  Zero for the closed-form two-source solver,
    #: which never iterates.
    support_shift: float = 0.0


def _invalid(message: str) -> NoReturn:
    """Raise one uniform eager-boundary validation error."""
    raise ValueError(message)


def _validated_inputs(
    sources: ArrayLike,
    source_weights: ArrayLike,
    *,
    minimum_sources: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Validate the eager array boundary shared by both solvers."""
    source_array = jnp.asarray(sources)
    dtype = jnp.result_type(source_array.dtype, jnp.float32)
    source_array = source_array.astype(dtype)
    weight_array = jnp.asarray(source_weights, dtype=dtype)
    if (
        source_array.ndim != _SOURCE_ARRAY_NDIM
        or source_array.shape[0] < minimum_sources
    ):
        _invalid(
            "Wasserstein sources must have shape "
            "(n_sources, n_points, n_dimensions) "
            f"with at least {minimum_sources} sources"
        )
    if source_array.shape[1] == 0 or source_array.shape[2] == 0:
        _invalid("Wasserstein sources must have non-empty support and dimension")
    if weight_array.ndim != 1 or weight_array.shape[0] != source_array.shape[0]:
        _invalid("Wasserstein source weights must match the source count")
    if not bool(jnp.all(jnp.isfinite(source_array))):
        _invalid("Wasserstein sources must be finite")
    if not bool(jnp.all(jnp.isfinite(weight_array))):
        _invalid("Wasserstein source weights must be finite")
    if bool(jnp.any(weight_array < 0.0)) or not bool(
        jnp.isclose(jnp.sum(weight_array), 1.0, atol=_SIMPLEX_ATOL, rtol=0.0)
    ):
        _invalid("Wasserstein source weights must be a simplex")
    return source_array, weight_array


def _validated_order(p: float) -> float:
    """Return a finite Wasserstein order in the metric range."""
    order = float(p)
    if not jnp.isfinite(order) or order < 1.0:
        _invalid("Wasserstein order p must be finite and at least 1")
    return order


def _metric(metric: GroundDistanceSelection) -> GroundDistance[Law]:
    """Resolve a serialized metric once, outside the numerical loop."""
    return resolve_ground_distance(metric)


def _assignment(cost: Float[Array, "n n"]) -> jnp.ndarray:
    """Return a support-row-aligned Hungarian permutation."""
    rows, columns = optax.assignment.hungarian_algorithm(cost)
    permutation = jnp.zeros((cost.shape[0],), dtype=columns.dtype)
    return permutation.at[rows].set(columns)


def _objective_and_assignments(
    support: Float[Array, "n d"],
    sources: Float[Array, "k n d"],
    weights: Float[Array, " k"],
    *,
    p: float,
    metric: GroundDistance[Law],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the fixed-support objective and its optimal permutations.

    The matched costs are read back out of ``costs`` rather than recomputed from
    the matched points.  Recomputing them with a Euclidean norm would silently
    report a different quantity than the one the assignment minimized whenever
    ``metric`` is not Euclidean -- for unit rows under ``cosine`` the two differ
    by an order of magnitude -- and the gather is cheaper than a second distance
    evaluation besides.
    """
    costs = jax.vmap(lambda source: cdist(support, source, metric=metric))(sources)
    permutations = jax.vmap(lambda cost: _assignment(cost**p))(costs)
    matched_costs = jnp.take_along_axis(costs, permutations[:, :, None], axis=2)[..., 0]
    point_costs = matched_costs**p
    objective = jnp.mean(jnp.sum(weights[:, None] * point_costs, axis=0))
    return objective, permutations


def solve_wasserstein_measure_barycentre(
    sources: ArrayLike,
    source_weights: ArrayLike,
    *,
    p: float = 2.0,
    metric: GroundDistanceSelection = "euclidean",
) -> WassersteinBarycentreResult:
    """Solve the exact two-source displacement barycentre.

    The assignment is solved on ``d**p`` and the returned support interpolates
    each matched pair at geodesic parameter ``w[1]``.

    Exactness is narrower than "assignment plus interpolation" suggests, and the
    solver refuses the cases where it does not hold rather than returning a
    non-minimal support under ``converged=True``:

    * Along the geodesic the objective is ``(w0 t**p + w1 (1-t)**p) * Wp**p``,
      minimized at ``t* = w1**(1/(p-1)) / (w0**(1/(p-1)) + w1**(1/(p-1)))``.
      The interpolation parameter used here is ``w1``, and ``t* == w1`` exactly
      when ``p == 2`` or the weights are equal.  At ``p == 1`` the objective is
      linear in ``t``, so the minimum sits at the source carrying the *larger*
      weight and equals ``min(w0, w1) * W1``; only equal weights make the whole
      geodesic -- including the midpoint -- optimal.
    * The interpolant is a straight line in the ambient coordinates, which is
      the geodesic only under the Euclidean ground metric.  Under ``cosine``,
      ``angular``, or ``normalized_euclidean`` it leaves the sphere the metric
      is defined on.
    """
    source_array, weights = _validated_inputs(
        sources,
        source_weights,
        minimum_sources=_TWO_SOURCES,
    )
    if source_array.shape[0] != _TWO_SOURCES:
        _invalid("the exact two-source Wasserstein solver requires two sources")
    order = _validated_order(p)
    ground = _metric(metric)
    if ground is not EUCLIDEAN:
        _invalid(
            "the two-source displacement barycentre interpolates linearly, which "
            "is the geodesic only under the Euclidean ground metric; use the "
            "free-support solver for a non-Euclidean ground distance"
        )
    equal_weights = bool(
        jnp.isclose(weights[0], weights[1], atol=_SIMPLEX_ATOL, rtol=0.0)
    )
    if order != _WASSERSTEIN_ORDER_TWO and not equal_weights:
        _invalid(
            f"interpolating at parameter w[1] minimizes the order-{order:g} "
            "barycentre objective only when p == 2 or the source weights are "
            f"equal; weights {float(weights[0]):g}/{float(weights[1]):g} at "
            f"order {order:g} are minimized elsewhere on the geodesic "
            + (
                "(at order 1, at the source carrying the larger weight)"
                if order == _WASSERSTEIN_ORDER_ONE
                else "(at t* != w[1])"
            )
        )
    costs = cdist(source_array[0], source_array[1], metric=ground)
    permutation = _assignment(costs**order)
    matched_right = source_array[1][permutation]
    support = weights[0] * source_array[0] + weights[1] * matched_right
    objective, _ = _objective_and_assignments(
        support,
        source_array,
        weights,
        p=order,
        metric=ground,
    )
    initial = jnp.mean(
        weights[1] * costs[jnp.arange(costs.shape[0]), permutation] ** order
    )
    objective_value = float(objective)
    initial_value = float(initial)
    return WassersteinBarycentreResult(
        support_values=support,
        source_weights=weights,
        objective=objective_value,
        initial_objective=initial_value,
        optimality_gap=max(initial_value - objective_value, 0.0),
        constraint_violation=float(jnp.abs(jnp.sum(weights) - 1.0)),
        converged=True,
        iterations=1,
        objective_semantics=f"W{order:g}_powered_transport_objective",
    )


def solve_wasserstein_free_support_barycentre(
    sources: ArrayLike,
    source_weights: ArrayLike,
    *,
    p: float = 2.0,
    metric: GroundDistanceSelection = "euclidean",
    tol: float = 1e-6,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> WassersteinBarycentreResult:
    """Solve a K-source free-support barycentre by JAX alternating minimization.

    Each iteration performs one exact Hungarian assignment per source and then
    updates the support with the weighted mean of matched points.  The update
    is the exact fixed-assignment minimizer for ``p=2``; for ``p=1`` the same
    mean update is retained as a useful heuristic and is reported as
    unconverged.

    ``converged`` is set by the *discrete* fixed point -- the permutations
    stopping changing -- not by the support settling below ``tol``.  Once every
    assignment repeats, the next weighted mean reproduces the current support
    exactly, so this is a checkable certificate rather than a threshold; and a
    support that has drifted below ``tol`` while the assignments are still
    moving is not a fixed point at all.  ``tol`` therefore acts as a budget
    guard that stops the loop without claiming convergence, alongside
    ``max_iterations``.  ``support_shift`` reports the final displacement either
    way.

    The objective is non-convex in the support, so a fixed point is local.  It
    also depends on where the iteration starts, and the start is ``sources[0]``:
    a deterministic, transport-meaningful support.  Selecting the start by
    weight (``argmax``) would make the result depend on the last bit of a
    hand-written weight vector -- ``[1/3, 1/3, 1/3]`` and its float64 rounding
    ``[0.3333333333333333, 0.3333333333333333, 0.3333333333333334]`` choose
    different sources -- and different starts land on objectives that differ in
    the third significant figure.
    """
    sources_array, weights = _validated_inputs(
        sources,
        source_weights,
        minimum_sources=_MINIMUM_SOURCES,
    )
    order = _validated_order(p)
    if max_iterations < 1:
        _invalid("max_iterations must be positive")
    if tol < 0.0 or not jnp.isfinite(tol):
        _invalid("tol must be finite and non-negative")
    ground = _metric(metric)
    support = sources_array[0]
    initial, permutations = _objective_and_assignments(
        support,
        sources_array,
        weights,
        p=order,
        metric=ground,
    )

    def condition(state: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        _, _, _, iteration, stable, shift = state
        return (iteration < max_iterations) & ~stable & (shift > tol)

    def body(state: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        current, _, current_permutations, iteration, _, _ = state
        matched = jnp.take_along_axis(
            sources_array,
            jnp.broadcast_to(current_permutations[..., None], sources_array.shape),
            axis=1,
        )
        candidate = jnp.sum(weights[:, None, None] * matched, axis=0)
        candidate_objective, candidate_permutations = _objective_and_assignments(
            candidate,
            sources_array,
            weights,
            p=order,
            metric=ground,
        )
        return (
            candidate,
            candidate_objective,
            candidate_permutations,
            iteration + 1,
            jnp.all(candidate_permutations == current_permutations),
            jnp.max(jnp.abs(candidate - current)),
        )

    support, objective, _, iterations, stable, shift = jax.lax.while_loop(
        condition,
        body,
        (
            support,
            initial,
            permutations,
            jnp.asarray(0),
            jnp.zeros((), dtype=bool),
            jnp.asarray(jnp.inf, dtype=sources_array.dtype),
        ),
    )
    objective_value = float(objective)
    initial_value = float(initial)
    reached_fixed_point = bool(stable)
    return WassersteinBarycentreResult(
        support_values=support,
        source_weights=weights,
        objective=objective_value,
        initial_objective=initial_value,
        optimality_gap=max(initial_value - objective_value, 0.0),
        constraint_violation=float(jnp.abs(jnp.sum(weights) - 1.0)),
        converged=reached_fixed_point and order == _WASSERSTEIN_ORDER_TWO,
        iterations=int(iterations),
        objective_semantics=f"W{order:g}_powered_transport_objective",
        support_shift=0.0 if reached_fixed_point else float(shift),
    )

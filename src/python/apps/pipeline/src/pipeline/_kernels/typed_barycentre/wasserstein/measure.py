"""Typed adapters over the pure-JAX source-measure barycentre solvers.

The geometry lives in :mod:`jcor.geometry._barycentre.wasserstein.measure`,
which knows only dense ``(k, n, d)`` arrays.  Everything here is the pipeline's
half of that boundary: registry-vocabulary validation, support-ID minting,
error translation, and the objective-semantics string the artifact schema
expects.  The module deliberately carries the same name as its jcor counterpart
so the two halves of one computation are found under one word.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jcor.geometry import (
    solve_wasserstein_free_support_barycentre as _solve_jax_free_support_barycentre,
)
from jcor.geometry import (
    solve_wasserstein_measure_barycentre as _solve_jax_measure_barycentre,
)

from pipeline._kernels.typed_barycentre._common import (
    MINIMUM_CLOUDS,
    TWO_SOURCE_COUNT,
    _barycentre_error,
    _reraise_as_barycentre_error,
    _stacked_sources,
)
from pipeline._kernels.typed_barycentre._diagnostics import (
    MeasureBarycentreKernelResult,
)

if TYPE_CHECKING:
    from jcor.geometry import WassersteinBarycentreResult

    from pipeline._kernels.typed_geometry import EmbeddingCloud

_ESTIMATORS = {"balanced_wasserstein_1", "balanced_wasserstein_2"}


class _ArmIdentity(NamedTuple):
    """The three declaration strings an arm carries into its result.

    Grouped rather than passed separately because they travel together and mean
    nothing apart: every one is copied verbatim from the ``params.yaml`` arm.
    """

    solver: str
    feasible_set: str
    estimator: str

    @property
    def order(self) -> float:
        """The transport order jcor takes for this arm's estimator."""
        if self.estimator not in _ESTIMATORS:
            _barycentre_error(f"unsupported Wasserstein estimator {self.estimator!r}")
        return 2.0 if self.estimator == "balanced_wasserstein_2" else 1.0

    @property
    def objective_semantics(self) -> str:
        """The artifact vocabulary for this arm's reported objective.

        Restated rather than passed through from jcor, which reports
        ``W2_powered_transport_objective`` where the artifact schema and its
        readers speak ``squared_statistical_distance``.
        """
        if self.estimator == "balanced_wasserstein_2":
            return "squared_statistical_distance"
        return "statistical_distance"


def _measure_result(
    result: WassersteinBarycentreResult,
    items: tuple[EmbeddingCloud, ...],
    support_ids: tuple[str, ...],
    arm: _ArmIdentity,
) -> MeasureBarycentreKernelResult:
    """Translate one jcor certificate into the artifact-boundary result."""
    return MeasureBarycentreKernelResult(
        support_ids=support_ids,
        support_values=np.asarray(result.support_values),
        source_ids=tuple(item.item_id for item in items),
        source_weights=tuple(
            float(weight) for weight in np.asarray(result.source_weights)
        ),
        objective=result.objective,
        initial_objective=result.initial_objective,
        optimality_gap=result.optimality_gap,
        constraint_violation=result.constraint_violation,
        converged=result.converged,
        iterations=result.iterations,
        solver=arm.solver,
        feasible_set=arm.feasible_set,
        objective_semantics=arm.objective_semantics,
        support_shift=result.support_shift,
    )


def solve_wasserstein_measure_barycentre(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    source_weights: tuple[float, ...],
    *,
    solver: str = "wasserstein_ot",
    feasible_set: str = "prescribed_source_weights",
    estimator: str = "balanced_wasserstein_1",
) -> MeasureBarycentreKernelResult:
    """Solve a two-source empirical Wasserstein barycentre by optimal matching.

    An optimal assignment followed by interpolation along each matched transport
    segment is the exact geodesic barycentre, but only under conditions the
    shared solver enforces rather than assumes.  Order 2 is exact at any source
    weights.  Order 1 is exact only at *equal* weights: the objective is linear
    along the geodesic, so an asymmetric order-1 barycentre is attained at the
    source carrying the larger weight and the interpolant is not a minimizer at
    all -- ``balanced_wasserstein_1`` with unequal weights is therefore rejected
    instead of returned under ``converged=True``.  See
    :func:`jcor.geometry.solve_wasserstein_measure_barycentre` for the
    derivation.

    More than two source measures need
    :func:`solve_wasserstein_free_support_barycentre`; they are rejected here
    rather than silently approximated.

    The reported objective is ``sum_k w_k W_p^p(b, mu_k)``, which for two
    sources collapses to ``w_0 w_1 W_p^p(mu_0, mu_1)`` up to the order-1 factor
    of two the linear cost carries.
    """
    items = tuple(clouds)
    if len(items) != TWO_SOURCE_COUNT or len(source_weights) != TWO_SOURCE_COUNT:
        _barycentre_error(
            "the shared Wasserstein measure solver currently requires two sources"
        )
    arm = _ArmIdentity(solver, feasible_set, estimator)
    sources = _stacked_sources(items)
    try:
        result = _solve_jax_measure_barycentre(
            jnp.asarray(sources),
            jnp.asarray(source_weights),
            p=arm.order,
            metric="euclidean",
        )
    except (TypeError, ValueError) as error:
        _reraise_as_barycentre_error(error)
    support_ids = tuple(
        f"{items[0].item_id}::{items[1].item_id}::{index:06d}"
        for index in range(len(np.asarray(result.support_values)))
    )
    return _measure_result(result, items, support_ids, arm)


def solve_wasserstein_free_support_barycentre(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    source_weights: tuple[float, ...],
    *,
    solver: str = "wasserstein_free_support",
    feasible_set: str = "prescribed_source_weights",
    estimator: str = "balanced_wasserstein_2",
    max_iterations: int = 128,
    tol: float = 1e-6,
) -> MeasureBarycentreKernelResult:
    """Solve a K-source free-support empirical Wasserstein barycentre.

    Generalizes :func:`solve_wasserstein_measure_barycentre` past two sources by
    alternating minimization on ``min_b sum_k w_k W_p^p(b, mu_k)``:

    1. **Assignment step** — with the support fixed, solve one exact assignment
       per source. The objective separates over sources at fixed support, so
       this step is globally optimal in the assignments.
    2. **Update step** — with the assignments fixed, the objective is a
       separable weighted least-squares problem in the support points whose
       minimizer is the weighted mean of the matched points. For order 1 the
       weighted *median* would be the exact minimizer; the mean update is used
       for both orders and is exact only at order 2, so order 1 is reported as
       an unconverged heuristic rather than silently labelled optimal.

    Each step is non-increasing in the objective and the objective is bounded
    below, so the iteration converges. It converges to a *local* minimum: the
    free-support objective is non-convex in the support, unlike the two-source
    case where an optimal assignment followed by displacement interpolation is
    exactly the geodesic barycentre.

    ``converged`` is the *discrete* fixed point -- the assignments repeating, so
    that one more sweep reproduces the support exactly -- and ``support_shift``
    is the residual, ``0.0`` once that holds. ``tol`` is a budget guard that
    stops the loop without claiming convergence, as ``max_iterations`` does.
    Read none of these as global optimality; ``optimality_gap`` in particular is
    only the improvement over the starting support.

    Equal support sizes across sources are required, as in the two-source
    solver. The typed spine fixes ``sample_size`` per distance, so this holds
    by construction for its declared arms.
    """
    items = tuple(clouds)
    if len(items) < MINIMUM_CLOUDS or len(source_weights) != len(items):
        _barycentre_error(
            "the free-support Wasserstein solver needs at least two sources and "
            "one weight per source"
        )
    arm = _ArmIdentity(solver, feasible_set, estimator)
    sources = _stacked_sources(items)
    try:
        result = _solve_jax_free_support_barycentre(
            jnp.asarray(sources),
            jnp.asarray(source_weights),
            p=arm.order,
            metric="euclidean",
            tol=tol,
            max_iterations=max_iterations,
        )
    except (TypeError, ValueError) as error:
        _reraise_as_barycentre_error(error)
    support_ids = tuple(
        "::".join(item.item_id for item in items) + f"::{index:06d}"
        for index in range(len(np.asarray(result.support_values)))
    )
    return _measure_result(result, items, support_ids, arm)


__all__ = [
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
]

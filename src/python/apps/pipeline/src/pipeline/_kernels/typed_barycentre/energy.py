"""Energy target projections on the simplex and the signed feasible sets.

The simplex arm delegates its numerics to
:func:`jcor.geometry.energy_barycentre_weights`; the signed arms are a
least-squares solve small enough to keep here.  What this module owns either way
is the declared-arm validation and the certificate assembly -- registry
vocabulary ``jcor`` has no business knowing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jcor.geometry import energy_barycentre_weights

from pipeline._kernels.typed_barycentre._common import (
    ACTIVE_WEIGHT_EPS_FACTOR,
    MINIMUM_LEAVE_ONE_OUT_CLOUDS,
    _barycentre_error,
    _component_values,
    _constraint_violation,
)
from pipeline._kernels.typed_barycentre._diagnostics import (
    BarycentreDiagnostics,
    BarycentreResult,
    _certificate_scalars,
)
from pipeline._kernels.typed_geometry import compute_mean_ground_distance_component

if TYPE_CHECKING:
    from pipeline._kernels.typed_geometry import (
        EmbeddingCloud,
        MeanGroundDistanceMatrix,
    )
    from pipeline.io.typed_analysis import (
        BarycentreArm,
        PointDistanceSpec,
        StatisticalDistanceSpec,
    )


def _validate_target_projection(
    items: tuple[EmbeddingCloud, ...],
    geometry: StatisticalDistanceSpec,
    arm: BarycentreArm,
    ground: PointDistanceSpec,
) -> tuple[float, int]:
    """Check the energy projection's geometry, ground, and solver budget.

    The ground distance and the solver budget are registry facts.  Neither is
    inferred from an ID string nor defaulted here, so an arm cannot quietly
    solve a different geometry or at a different tolerance than it declares.
    """
    if geometry.family != "energy":
        _barycentre_error("energy target solver requires an energy geometry")
    if geometry.estimator != "v_statistic":
        _barycentre_error("energy target solver requires the V-statistic")
    if arm.problem != "target_projection":
        _barycentre_error("energy target solver requires target_projection")
    if ground.distance_id != geometry.ground_distance:
        _barycentre_error(
            f"{geometry.distance_id!r} declares ground distance "
            f"{geometry.ground_distance!r}, received {ground.distance_id!r}"
        )
    if arm.tol is None or arm.maxiter is None:
        _barycentre_error(f"arm {arm.arm_id!r} has no declared solver budget")
    if len(items) < MINIMUM_LEAVE_ONE_OUT_CLOUDS:
        _barycentre_error("leave-one-out projections require at least three clouds")
    return arm.tol, arm.maxiter


def _solve_signed(
    gram: jnp.ndarray, cross: jnp.ndarray, feasible_set: str
) -> np.ndarray:
    q = np.asarray(gram, dtype=np.float64)
    c = np.asarray(cross, dtype=np.float64)
    if feasible_set == "free_signed":
        weights = np.linalg.lstsq(q, c, rcond=None)[0]
    else:
        ones = np.ones((len(c), 1), dtype=np.float64)
        system = np.block([[q, ones], [ones.T, np.zeros((1, 1))]])
        rhs = np.concatenate([c, [1.0]])
        weights = np.linalg.lstsq(system, rhs, rcond=None)[0][:-1]
    if not np.all(np.isfinite(weights)):
        _barycentre_error("signed energy projection produced non-finite weights")
    return weights


def _energy_objective(
    weights: np.ndarray, cross: np.ndarray, gram: np.ndarray, target_self: float
) -> float:
    return float(2.0 * cross @ weights - weights @ gram @ weights - target_self)


def solve_energy_target_projections(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    geometry: StatisticalDistanceSpec,
    arm: BarycentreArm,
    ground: PointDistanceSpec,
    component: MeanGroundDistanceMatrix | None = None,
) -> BarycentreResult:
    """Solve every declared target projection against the other clouds."""
    items = tuple(clouds)
    tol, maxiter = _validate_target_projection(items, geometry, arm, ground)
    mean_distances = (
        compute_mean_ground_distance_component(items, geometry, ground).values
        if component is None
        else _component_values(component, items, "mean_ground_distance")
    )
    target_ids: list[str] = []
    candidate_ids: list[tuple[str, ...]] = []
    weight_rows: list[np.ndarray] = []
    diagnostics: list[BarycentreDiagnostics] = []
    feasible_set = arm.feasible_set
    for target_index, target in enumerate(items):
        candidate_indices = tuple(
            index for index in range(len(items)) if index != target_index
        )
        candidates = tuple(items[index] for index in candidate_indices)
        gram = jnp.asarray(mean_distances[np.ix_(candidate_indices, candidate_indices)])
        cross = jnp.asarray(mean_distances[target_index, candidate_indices])
        target_self = float(mean_distances[target_index, target_index])
        projected_gradient_norm = 0.0
        cnd_tangent_max_eigenvalue = 0.0
        tangent_curvature_min = 0.0
        simplex_sum_error = 0.0
        min_weight = 0.0
        if feasible_set == "simplex_nonnegative":
            weights_jax, certificate = energy_barycentre_weights(
                jnp.asarray(target.values),
                [jnp.asarray(candidate.values) for candidate in candidates],
                metric=ground.metric,
                exponent=geometry.exponent,
                solver="qp" if arm.solver == "energy_qp" else "pgd",
                tol=tol,
                maxiter=maxiter,
                precomputed_components=(gram, cross),
                return_diagnostics=True,
            )
            weights = np.asarray(weights_jax)
            objective = float(certificate.energy)
            initial = float(certificate.initial_energy)
            gap = float(certificate.frank_wolfe_gap)
            converged = bool(certificate.converged)
            iterations = int(certificate.iterations)
            (
                projected_gradient_norm,
                cnd_tangent_max_eigenvalue,
                tangent_curvature_min,
                simplex_sum_error,
                min_weight,
            ) = _certificate_scalars(certificate)
        elif feasible_set in {"affine_signed", "free_signed"}:
            weights = _solve_signed(gram, cross, feasible_set)
            objective = _energy_objective(
                weights, np.asarray(cross), np.asarray(gram), target_self
            )
            initial_weights = np.full(len(candidates), 1.0 / len(candidates))
            initial = _energy_objective(
                initial_weights, np.asarray(cross), np.asarray(gram), target_self
            )
            gap = 0.0
            converged = True
            iterations = 1
        else:
            _barycentre_error(
                f"feasible set {feasible_set!r} is not an energy target projection"
            )
        constraint_violation = _constraint_violation(weights, feasible_set)
        target_ids.append(target.item_id)
        candidate_ids.append(tuple(candidate.item_id for candidate in candidates))
        weight_rows.append(weights)
        diagnostics.append(
            BarycentreDiagnostics(
                target_id=target.item_id,
                objective=objective,
                initial_objective=initial,
                optimality_gap=gap,
                constraint_violation=constraint_violation,
                active_support_size=int(
                    np.count_nonzero(
                        np.abs(weights)
                        > ACTIVE_WEIGHT_EPS_FACTOR * np.finfo(weights.dtype).eps
                    )
                ),
                converged=converged,
                iterations=iterations,
                solver=arm.solver,
                feasible_set=feasible_set,
                projected_gradient_norm=projected_gradient_norm,
                cnd_tangent_max_eigenvalue=cnd_tangent_max_eigenvalue,
                tangent_curvature_min=tangent_curvature_min,
                simplex_sum_error=simplex_sum_error,
                min_weight=min_weight,
            )
        )
    return BarycentreResult(
        target_ids=tuple(target_ids),
        candidate_ids=tuple(candidate_ids),
        weights=tuple(weight_rows),
        diagnostics=tuple(diagnostics),
        arm_id=arm.arm_id,
        geometry_id=geometry.distance_id,
    )


__all__ = ["solve_energy_target_projections"]

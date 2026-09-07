"""V-statistic MMD target projections on the nonnegative simplex.

Numerics live in :func:`jcor.geometry.kernel_barycentre_weights`; this module
owns the declared-arm validation, the kernel resolution against the typed
registry, and the certificate assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from jcor.geometry import kernel_barycentre_weights

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
)
from pipeline._kernels.typed_geometry import (
    compute_kernel_mean_component,
    resolve_mmd_kernel,
)

if TYPE_CHECKING:
    from pipeline._kernels.typed_geometry import EmbeddingCloud, KernelMeanMatrix
    from pipeline.io.typed_analysis import (
        BarycentreArm,
        PointDistanceSpec,
        StatisticalDistanceSpec,
    )


def _validate_mmd_target_projection(
    items: tuple[EmbeddingCloud, ...],
    geometry: StatisticalDistanceSpec,
    arm: BarycentreArm,
    ground: PointDistanceSpec,
) -> tuple[float, int]:
    """Validate the declared V-statistic MMD projection arm."""
    if geometry.family != "mmd" or geometry.estimator != "v_statistic":
        _barycentre_error("MMD target solver requires a V-statistic MMD geometry")
    if geometry.kernel is None:
        _barycentre_error("MMD target solver requires a declared kernel")
    if arm.problem != "target_projection" or arm.target_policy != "leave_one_out":
        _barycentre_error("MMD target solver requires leave-one-out target_projection")
    if arm.solver != "mmd_qp" or arm.feasible_set != "simplex_nonnegative":
        _barycentre_error(
            "MMD target solver requires mmd_qp on the nonnegative simplex"
        )
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


def _mmd_kernel_means(
    items: tuple[EmbeddingCloud, ...],
    geometry: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
) -> tuple[np.ndarray, Any]:
    """Compute V-statistic kernel means using the declared point geometry."""
    component = compute_kernel_mean_component(items, geometry, ground)
    kernel, _ = resolve_mmd_kernel(geometry, ground, items)
    return component.values, kernel


def solve_mmd_target_projections(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    geometry: StatisticalDistanceSpec,
    arm: BarycentreArm,
    ground: PointDistanceSpec,
    component: KernelMeanMatrix | None = None,
) -> BarycentreResult:
    """Solve typed-kernel MMD projections in deterministic leave-one-out order."""
    items = tuple(clouds)
    tol, maxiter = _validate_mmd_target_projection(items, geometry, arm, ground)
    if component is None:
        kernel_means, kernel = _mmd_kernel_means(items, geometry, ground)
    else:
        kernel_means = _component_values(component, items, "kernel_mean")
        kernel, _ = resolve_mmd_kernel(
            geometry,
            ground,
            items,
            bandwidth_override=component.metadata.bandwidth,
        )
    target_ids: list[str] = []
    candidate_ids: list[tuple[str, ...]] = []
    weight_rows: list[np.ndarray] = []
    diagnostics: list[BarycentreDiagnostics] = []
    for target_index, target in enumerate(items):
        candidate_indices = tuple(i for i in range(len(items)) if i != target_index)
        candidates = tuple(items[i] for i in candidate_indices)
        candidate_gram = jnp.asarray(
            kernel_means[np.ix_(candidate_indices, candidate_indices)]
        )
        cross = jnp.asarray(kernel_means[target_index, candidate_indices])
        target_self = jnp.asarray(kernel_means[target_index, target_index])
        weights_jax, certificate = kernel_barycentre_weights(
            jnp.asarray(target.values),
            [jnp.asarray(candidate.values) for candidate in candidates],
            kernel=kernel,
            solver="qp",
            tol=tol,
            maxiter=maxiter,
            validate_psd=True,
            precomputed_components=(candidate_gram, cross, target_self),
            return_diagnostics=True,
        )
        weights = np.asarray(weights_jax)
        target_ids.append(target.item_id)
        candidate_ids.append(tuple(candidate.item_id for candidate in candidates))
        weight_rows.append(weights)
        diagnostics.append(
            BarycentreDiagnostics(
                target_id=target.item_id,
                objective=float(certificate.mmd_squared),
                initial_objective=float(certificate.initial_mmd_squared),
                optimality_gap=float(certificate.frank_wolfe_gap),
                constraint_violation=_constraint_violation(
                    weights, "simplex_nonnegative"
                ),
                active_support_size=int(
                    np.count_nonzero(
                        weights > ACTIVE_WEIGHT_EPS_FACTOR * np.finfo(weights.dtype).eps
                    )
                ),
                converged=bool(certificate.converged),
                iterations=int(certificate.iterations),
                solver=arm.solver,
                feasible_set=arm.feasible_set,
                projected_gradient_norm=float(certificate.projected_gradient_norm),
                tangent_curvature_min=float(certificate.tangent_curvature_min),
                simplex_sum_error=float(certificate.simplex_sum_error),
                min_weight=float(certificate.min_weight),
                psd_tangent_min_eigenvalue=float(
                    certificate.psd_tangent_min_eigenvalue
                ),
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


__all__ = ["solve_mmd_target_projections"]

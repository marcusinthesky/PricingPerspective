"""Target-anchored empirical W1/W2 barycentre projections.

Why this one stays in the pipeline while its measure-barycentre sibling moved to
``jcor``: the solve is ``scipy.optimize.minimize(method="SLSQP")`` over the
candidate simplex, on top of ``scipy.optimize.linear_sum_assignment`` couplings.
``jcor`` retired its last runtime SciPy call at t65 -- SciPy is a test-only
dependency of that package now -- so promoting this needs a JAX reimplementation
of constrained SLSQP, not a move.  Until then the anchored solver lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist as scipy_cdist

from pipeline._kernels.arrays import project_simplex
from pipeline._kernels.typed_barycentre._common import (
    ACTIVE_WEIGHT_EPS_FACTOR,
    MINIMUM_LEAVE_ONE_OUT_CLOUDS,
    _barycentre_error,
)
from pipeline._kernels.typed_barycentre._diagnostics import (
    BarycentreDiagnostics,
    BarycentreResult,
)

if TYPE_CHECKING:
    from pipeline._kernels.typed_barycentre._common import GroundMetric
    from pipeline._kernels.typed_geometry import EmbeddingCloud
    from pipeline.io.typed_analysis import (
        BarycentreArm,
        PointDistanceSpec,
        StatisticalDistanceSpec,
    )


def _transport_permutation(
    left: np.ndarray, right: np.ndarray, *, order_two: bool, metric: GroundMetric
) -> tuple[np.ndarray, float]:
    """Return the optimal assignment and its exact second-best cost gap."""
    if metric != "euclidean":
        _barycentre_error("Wasserstein target couplings require Euclidean metric")
    cost = scipy_cdist(left, right, metric="euclidean")
    assignment_cost = cost**2 if order_two else cost
    rows, columns = linear_sum_assignment(assignment_cost)
    permutation = np.empty(len(left), dtype=np.int64)
    permutation[rows] = columns
    best = float(np.sum(assignment_cost[rows, columns]))
    second_best = float("inf")
    for row, column in zip(rows, columns, strict=True):
        excluded = assignment_cost.copy()
        excluded[row, column] = np.inf
        alternative_rows, alternative_columns = linear_sum_assignment(excluded)
        alternative = float(np.sum(excluded[alternative_rows, alternative_columns]))
        second_best = min(second_best, alternative)
    gap = max(second_best - best, 0.0)
    return permutation, gap


def _target_anchored_couplings(
    clouds: tuple[EmbeddingCloud, ...],
    *,
    order_two: bool,
    metric: GroundMetric,
) -> tuple[tuple[tuple[np.ndarray, ...], ...], np.ndarray]:
    """Build one deterministic target-aligned coupling for every cloud pair.

    The coupling is solved once per unordered pair and inverted for the reverse
    direction.  The resulting tuple has an empty diagonal; entry ``(i, j)`` is
    the index vector that re-indexes cloud ``j`` onto target cloud ``i``, so the
    aligned support is ``clouds[j].values[entry]``.  Fixing these couplings is
    the declared restricted-support step in the Wasserstein target projection:
    the subsequent simplex solve is an exact barycentre solve conditional on
    these transport maps, not an unrestricted multi-marginal OT claim.

    Only the PERMUTATIONS are retained, not the permuted supports. Materializing
    the latter costs ``n**2`` copies of an ``(m, d)`` cloud: at m=128, d=4096,
    float64 that is ~10 GiB for a 52-item panel and ~39 GiB for 100 items, which
    the OOM killer reaches before the solve starts. An index vector is ~1 KB, so
    the grid is a few MB at any panel size and callers apply it per target. The
    numerics are unchanged -- the same permutation applied to the same values,
    just later.
    """
    if any(cloud.values.shape != clouds[0].values.shape for cloud in clouds):
        _barycentre_error("Wasserstein target projections require equal cloud shapes")
    empty = np.empty(0, dtype=np.intp)
    permutations: list[list[np.ndarray]] = [[empty for _ in clouds] for _ in clouds]
    gaps = np.full((len(clouds), len(clouds)), np.inf, dtype=np.float64)
    for left_index in range(len(clouds)):
        for right_index in range(left_index + 1, len(clouds)):
            permutation, gap = _transport_permutation(
                np.asarray(clouds[left_index].values),
                np.asarray(clouds[right_index].values),
                order_two=order_two,
                metric=metric,
            )
            permutations[left_index][right_index] = permutation
            gaps[left_index, right_index] = gap
            gaps[right_index, left_index] = gap
            inverse = np.empty_like(permutation)
            inverse[permutation] = np.arange(len(permutation))
            permutations[right_index][left_index] = inverse
    return tuple(tuple(row) for row in permutations), gaps


def _anchored_objective_and_gradient(
    weights: np.ndarray,
    target: np.ndarray,
    matched_candidates: np.ndarray,
    *,
    order_two: bool,
) -> tuple[float, np.ndarray]:
    """Evaluate the anchored W1/W2 barycentre objective and a valid gradient."""
    support = np.einsum("k,kmd->md", weights, matched_candidates)
    residual = target - support
    if order_two:
        objective = float(np.mean(np.sum(residual * residual, axis=1)))
        gradient = -2.0 * np.einsum("kmd,md->k", matched_candidates, residual)
        gradient /= float(len(target))
        return objective, gradient
    norms = np.linalg.norm(residual, axis=1)
    objective = float(np.mean(norms))
    normalized = np.divide(
        residual,
        norms[:, None],
        out=np.zeros_like(residual),
        where=norms[:, None] > 0.0,
    )
    gradient = -np.einsum("kmd,md->k", matched_candidates, normalized)
    gradient /= float(len(target))
    return objective, gradient


def _anchored_diagnostics(
    *,
    target_id: str,
    weights: np.ndarray,
    objective: float,
    initial_objective: float,
    iterations: int,
    converged: bool,
    solver: str,
    feasible_set: str,
    gradient: np.ndarray,
    order_two: bool,
    curvature: tuple[float, float, float],
    assignment_gap_min: float,
    assignment_support_size: int,
) -> BarycentreDiagnostics:
    """Create the common target-projection certificate for W1 and W2."""
    if order_two:
        cnd_max, tangent_min, psd_min = curvature
    else:
        cnd_max = tangent_min = psd_min = 0.0
    return BarycentreDiagnostics(
        target_id=target_id,
        objective=objective,
        initial_objective=initial_objective,
        optimality_gap=max(initial_objective - objective, 0.0),
        constraint_violation=max(
            abs(float(np.sum(weights)) - 1.0), 0.0, -float(np.min(weights))
        ),
        active_support_size=int(
            np.count_nonzero(
                weights > ACTIVE_WEIGHT_EPS_FACTOR * np.finfo(weights.dtype).eps
            )
        ),
        converged=converged,
        iterations=iterations,
        solver=solver,
        feasible_set=feasible_set,
        projected_gradient_norm=float(np.linalg.norm(gradient - np.mean(gradient))),
        cnd_tangent_max_eigenvalue=cnd_max,
        tangent_curvature_min=tangent_min,
        simplex_sum_error=abs(float(np.sum(weights)) - 1.0),
        min_weight=float(np.min(weights)),
        psd_tangent_min_eigenvalue=psd_min,
        assignment_gap_min=assignment_gap_min,
        assignment_support_size=assignment_support_size,
    )


def _solve_anchored_wasserstein_row(
    target: EmbeddingCloud,
    matched_candidates: np.ndarray,
    *,
    order_two: bool,
    tol: float,
    maxiter: int,
    solver: str,
    feasible_set: str,
    assignment_gap_min: float,
) -> tuple[np.ndarray, BarycentreDiagnostics]:
    """Solve one candidate-simplex row conditional on target-aligned maps."""
    candidate_count = matched_candidates.shape[0]
    initial_weights = np.full(candidate_count, 1.0 / candidate_count, dtype=np.float64)
    initial_objective, _ = _anchored_objective_and_gradient(
        initial_weights,
        np.asarray(target.values, dtype=np.float64),
        matched_candidates,
        order_two=order_two,
    )

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return _anchored_objective_and_gradient(
            np.asarray(weights, dtype=np.float64),
            np.asarray(target.values, dtype=np.float64),
            matched_candidates,
            order_two=order_two,
        )

    result = minimize(
        lambda weights: objective(weights)[0],
        initial_weights,
        jac=lambda weights: objective(weights)[1],
        method="SLSQP",
        bounds=[(0.0, 1.0)] * candidate_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"ftol": tol, "maxiter": maxiter, "disp": False},
    )
    weights = project_simplex(np.asarray(result.x, dtype=np.float64))
    value, gradient = objective(weights)
    curvature = (0.0, 0.0, 0.0)
    if order_two:
        residual_candidates = matched_candidates - np.asarray(target.values)[None, :, :]
        gram = np.einsum("kmd,lmd->kl", residual_candidates, residual_candidates)
        gram /= float(len(target.values))
        gram = (gram + gram.T) * 0.5
        tangent = np.column_stack(
            [
                np.eye(candidate_count)[:, index] - np.eye(candidate_count)[:, -1]
                for index in range(candidate_count - 1)
            ]
        )
        basis, _ = np.linalg.qr(tangent, mode="reduced")
        eigenvalues = np.linalg.eigvalsh(2.0 * (basis.T @ gram @ basis))
        curvature = (
            float(np.max(eigenvalues)),
            float(np.min(eigenvalues)),
            float(np.min(eigenvalues)),
        )
    diagnostics = _anchored_diagnostics(
        target_id=target.item_id,
        weights=weights,
        objective=value,
        initial_objective=initial_objective,
        iterations=int(getattr(result, "nit", 0) or 0),
        converged=bool(
            result.success and np.isfinite(value) and np.all(np.isfinite(weights))
        ),
        solver=solver,
        feasible_set=feasible_set,
        gradient=gradient,
        order_two=order_two,
        curvature=curvature,
        assignment_gap_min=assignment_gap_min,
        assignment_support_size=len(target.values),
    )
    return weights, diagnostics


def solve_wasserstein_target_projections(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    geometry: StatisticalDistanceSpec,
    arm: BarycentreArm,
    ground: PointDistanceSpec,
) -> BarycentreResult:
    """Solve leave-one-out anchored empirical W1/W2 barycentre projections.

    Each target row first computes an optimal balanced transport map from the
    target cloud to every candidate cloud.  The candidate clouds are then
    displaced through those fixed maps and a simplex optimization chooses the
    barycentre mixture.  For W2 this is a convex quadratic program in the
    mixture weights; for W1 it is a convex sum-of-norms program solved by SLSQP.
    The result is therefore an exact solve of the declared target-anchored
    restriction, not an unrestricted multi-marginal Wasserstein claim.
    """
    items = tuple(clouds)
    if geometry.family != "wasserstein" or arm.problem != "target_projection":
        _barycentre_error("Wasserstein target solver requires a target projection")
    if arm.target_policy != "leave_one_out":
        _barycentre_error("Wasserstein target solver requires leave_one_out")
    if arm.feasible_set != "simplex_nonnegative":
        _barycentre_error("Wasserstein target solver requires the nonnegative simplex")
    if arm.solver not in {"wasserstein_w1_target", "wasserstein_w2_target"}:
        _barycentre_error(f"unsupported Wasserstein target solver {arm.solver!r}")
    if ground.metric != "euclidean":
        _barycentre_error(
            "Wasserstein target solver requires Euclidean ground distance"
        )
    if arm.tol is None or arm.maxiter is None:
        _barycentre_error("Wasserstein target solver requires a solver budget")
    if len(items) < MINIMUM_LEAVE_ONE_OUT_CLOUDS:
        _barycentre_error("leave-one-out projections require at least three clouds")
    order_two = geometry.estimator == "balanced_wasserstein_2"
    expected_solver = "wasserstein_w2_target" if order_two else "wasserstein_w1_target"
    if arm.solver != expected_solver:
        _barycentre_error(
            f"{geometry.estimator} requires solver {expected_solver!r}, "
            f"got {arm.solver!r}"
        )
    permutations, assignment_gaps = _target_anchored_couplings(
        items, order_two=order_two, metric=ground.metric
    )
    target_ids: list[str] = []
    candidate_ids: list[tuple[str, ...]] = []
    weight_rows: list[np.ndarray] = []
    diagnostics: list[BarycentreDiagnostics] = []
    for target_index, target in enumerate(items):
        candidate_indices = tuple(
            index for index in range(len(items)) if index != target_index
        )
        # Apply the stored couplings here rather than retaining n**2 permuted
        # supports: only this target's (n-1, m, d) stack is live at a time.
        matched = np.stack(
            [
                np.asarray(items[candidate_index].values)[
                    permutations[target_index][candidate_index]
                ]
                for candidate_index in candidate_indices
            ],
            axis=0,
        ).astype(np.float64, copy=False)
        weights, diagnostic = _solve_anchored_wasserstein_row(
            target,
            matched,
            order_two=order_two,
            tol=arm.tol,
            maxiter=arm.maxiter,
            solver=arm.solver,
            feasible_set=arm.feasible_set,
            assignment_gap_min=float(np.min(assignment_gaps[target_index])),
        )
        target_ids.append(target.item_id)
        candidate_ids.append(tuple(items[index].item_id for index in candidate_indices))
        weight_rows.append(weights)
        diagnostics.append(diagnostic)
    return BarycentreResult(
        target_ids=tuple(target_ids),
        candidate_ids=tuple(candidate_ids),
        weights=tuple(weight_rows),
        diagnostics=tuple(diagnostics),
        arm_id=arm.arm_id,
        geometry_id=geometry.distance_id,
    )


__all__ = ["solve_wasserstein_target_projections"]

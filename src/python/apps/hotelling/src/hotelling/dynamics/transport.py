"""Weighted squared transport with explicit finite-precision LP diagnostics.

The scientific readout is W2 squared. HiGHS runs in float64 because its
feasibility tolerances and reported dual certificate define this API's accuracy.
No JAX precision settings or global numerical configuration are changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
MATRIX_AXES = 2
FEASIBILITY_LIMIT = 2e-9
OBJECTIVE_TARGET = 1e-9
AXIS_SIGNS = 2


@dataclass(frozen=True)
class TransportResult:
    """Squared transport cost and numerical feasibility/optimality diagnostics."""

    cost: float
    marginal_error: float
    dual_gap: float
    error_bound: float


@lru_cache(maxsize=16)
def _constraints(rows: int, columns: int) -> csr_matrix:
    """Cache the two sparse marginal constraints for each support shape."""
    variables = np.arange(rows * columns)
    row_ids = np.repeat(np.arange(rows), columns)
    col_ids = rows + np.tile(np.arange(columns), rows)
    return csr_matrix(
        (
            np.ones(2 * rows * columns),
            (np.concatenate((row_ids, col_ids)), np.tile(variables, 2)),
        ),
        shape=(rows + columns, rows * columns),
    )


@lru_cache(maxsize=16)
def _costs(
    x_bytes: bytes, y_bytes: bytes, rows: int, columns: int, dimension: int
) -> FloatArray:
    x = np.frombuffer(x_bytes, dtype=np.float64).reshape(rows, dimension)
    y = np.frombuffer(y_bytes, dtype=np.float64).reshape(columns, dimension)
    result = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    result.setflags(write=False)
    return result


def _law(points: FloatArray, weights: FloatArray) -> tuple[FloatArray, FloatArray]:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        points.ndim != MATRIX_AXES
        or not points.shape[0]
        or not points.shape[1]
        or weights.shape != (points.shape[0],)
        or not np.isfinite(points).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        msg = "finite points and nonnegative weights with matching support required"
        raise ValueError(msg)
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12):
        msg = "transport weights must define a probability law"
        raise ValueError(msg)
    # Normalize only the accepted floating-point probability-sum discrepancy.
    weights = weights / weights.sum()
    return points, weights


def _round_coupling(
    plan: FloatArray, left: FloatArray, right: FloatArray
) -> FloatArray:
    """Round an approximately feasible nonnegative plan to both marginals."""
    plan = np.maximum(plan, 0)
    rows = plan.sum(axis=1)
    scale = np.minimum(np.divide(left, rows, out=np.ones_like(rows), where=rows > 0), 1)
    plan *= scale[:, None]
    columns = plan.sum(axis=0)
    scale = np.minimum(
        np.divide(right, columns, out=np.ones_like(columns), where=columns > 0), 1
    )
    plan *= scale[None, :]
    missing_left = np.maximum(left - plan.sum(axis=1), 0)
    missing_right = np.maximum(right - plan.sum(axis=0), 0)
    mass = float(missing_right.sum())
    if mass > 0:
        plan += np.outer(missing_left, missing_right) / mass
    return plan


def weighted_w2(
    x: FloatArray,
    wx: FloatArray,
    y: FloatArray,
    wy: FloatArray,
    *,
    deadline: float | None = None,
) -> TransportResult:
    """Solve weighted squared W2 using sparse HiGHS and report a dual bound.

    Zero-weight support points are valid and retained. A failed LP or a numerical
    feasibility failure raises ArithmeticError; it is never recorded as zero
    movement. ``error_bound`` uses a rounded primal plan and dual potentials
    shifted to feasibility, with a floating-point marginal allowance.
    """
    x, wx = _law(x, wx)
    y, wy = _law(y, wy)
    if x.shape[1] != y.shape[1]:
        msg = "transport supports must have the same article dimension"
        raise ValueError(msg)
    cost = _costs(x.tobytes(), y.tobytes(), len(x), len(y), x.shape[1])
    if not np.isfinite(cost).all():
        msg = "squared transport distances overflowed"
        raise ArithmeticError(msg)
    marginals = np.concatenate((wx, wy))
    options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }
    if deadline is not None:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            msg = "weighted transport exceeded the campaign time limit"
            raise TimeoutError(msg)
        options["time_limit"] = remaining
    solved = linprog(
        cost.ravel(),
        A_eq=_constraints(len(x), len(y)),
        b_eq=marginals,
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if deadline is not None and (perf_counter() >= deadline or solved.status == 1):
        msg = "weighted transport exceeded the campaign time limit"
        raise TimeoutError(msg)
    if not solved.success:
        msg = f"weighted transport LP failed: {solved.message}"
        raise ArithmeticError(msg)
    plan = np.asarray(solved.x, dtype=np.float64).reshape(cost.shape)
    dual = np.asarray(solved.eqlin.marginals, dtype=np.float64)
    marginal_error = float(
        max(
            np.max(np.abs(plan.sum(axis=1) - wx)),
            np.max(np.abs(plan.sum(axis=0) - wy)),
            0.0,
            -float(np.min(plan)),
        )
    )
    if not np.isfinite(plan).all() or not np.isfinite(dual).all():
        msg = "weighted transport LP returned nonfinite primal or dual values"
        raise ArithmeticError(msg)
    if marginal_error > FEASIBILITY_LIMIT:
        msg = "weighted transport LP exceeded its marginal feasibility tolerance"
        raise ArithmeticError(msg)
    rounded = _round_coupling(plan, wx, wy)
    value = float(np.sum(rounded * cost))
    dual_value = float(marginals @ dual)
    violation = max(
        0.0, float(np.max(dual[: len(x), None] + dual[None, len(x) :] - cost))
    )
    allowance = 64 * np.finfo(np.float64).eps * max(1.0, float(np.max(cost)))
    lower_bound = dual_value - violation
    error_bound = max(0.0, value - lower_bound) + allowance
    if not np.isfinite(error_bound) or error_bound > OBJECTIVE_TARGET * max(
        1.0, float(np.max(cost))
    ):
        msg = "weighted transport exceeded its objective uncertainty budget"
        raise ArithmeticError(msg)
    return TransportResult(
        value,
        marginal_error,
        abs(value - dual_value),
        error_bound,
    )


def axis_w2(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return exact squared transport for symmetric orthogonal axis mixtures.

    Component probabilities, rather than the duplicated signed-point weights,
    enter this identity. For two activities this equals 2*abs(p_left-p_right),
    including paths that cross any fixed reference law.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape[-1:] != right.shape[-1:] or not left.shape[-1:]:
        msg = "matching nonempty activity dimensions required"
        raise ValueError(msg)
    for shares in (left, right):
        if (
            not np.isfinite(shares).all()
            or np.any(shares < 0)
            or not np.allclose(shares.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)
        ):
            msg = "axis mixtures must be probability vectors"
            raise ValueError(msg)
    return np.asarray(np.sum(np.abs(left - right), axis=-1), dtype=np.float64)


def axis_point_w2(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return squared W2 for arbitrary masses on adjacent +/- orthonormal axes.

    This identity permits sampled sign imbalances. First match common point
    masses at zero cost. Residual mass R travels at cost two except between
    opposite points, whose cost is four. The graph of cost-two edges omits just
    each point's unique antipode. Its only possible positive Hall deficit is a
    singleton surplus plus its forbidden deficit minus R. Thus the cost is
    2*R plus twice that unavoidable opposite-point mass. The cost matrix obeys
    the triangle inequality, so matching common masses first is optimal.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.shape[-1:] != right.shape[-1:]
        or not left.shape[-1:]
        or not left.shape[-1]
        or left.shape[-1] % AXIS_SIGNS
    ):
        msg = "matching nonempty pairs of adjacent +/- point masses required"
        raise ValueError(msg)
    for weights in (left, right):
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)
        ):
            msg = "axis point weights must be probability vectors"
            raise ValueError(msg)
    difference = left - right
    surplus = np.maximum(difference, 0)
    deficit = np.maximum(-difference, 0)
    residual = surplus.sum(axis=-1)
    opposite = np.arange(left.shape[-1], dtype=np.int64) ^ 1
    unavoidable = np.maximum(
        np.max(surplus + deficit[..., opposite], axis=-1) - residual, 0
    )
    return np.asarray(2 * (residual + unavoidable), dtype=np.float64)

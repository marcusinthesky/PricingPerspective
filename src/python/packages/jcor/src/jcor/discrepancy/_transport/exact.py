"""S3 — exact balanced empirical Wasserstein-p via the Hungarian algorithm.

Position
--------
rank 3 · ``discrepancy``

Consumes
--------
two equal-cardinality row samples, read as equal-weight empirical measures

Produces
--------
a scalar ``W_p`` and the square ground-cost matrix it is solved from

Order
-----
``W_p`` assigns on the ground cost raised to ``p``, averages the matched
powered costs, and takes the ``p``-th root. Different orders generally solve
different assignment problems, so the order cannot be applied after the fact
to a matching optimized for another order.

Why this sits beside ``discrepancy.transport`` rather than inside it
--------------------------------------------------------------------
:mod:`jcor.discrepancy.transport` is closed-form: exact in one dimension and
sliced (a mean over random one-dimensional projections) in higher ones, with no
linear program anywhere. This module is the **exact-LP slot**: it forms the full
``n x n`` ground cost and solves the assignment problem exactly with
:func:`optax.assignment.hungarian_algorithm`, the same JAX implementation used
by :func:`ott.tools.unreg.hungarian`. They answer the same question by opposite
means, so they are siblings, not duplicates.

Transform boundary
------------------
The assignment core is exact, fixed-shape, and JAX-transformable. The eager
NumPy doors remain for stored float64 artifacts and validation; they delegate
to the same JAX Hungarian core under a door-owned x64 context. Callers entering
the JAX core directly own the process precision context.

Axioms — measured, and weaker than the name suggests
-----------------------------------------------------
``W_p`` for finite ``p >= 1`` over a metric ground cost is a metric on
``P_p``. The cloud convenience API imports Euclidean sphere-chord distance
from :mod:`jcor.ground.metrics` and requires **already unit-normalized rows**.
Normalization is a pipeline transformation that changes the estimand, so the
eager doors reject off-sphere rows instead of silently projecting them. The
exported array-level declaration remains a pseudometric because independent row
permutations represent the same empirical measure. On equal-cardinality
empirical measures on the unit sphere it is a genuine metric.

``PSEUDOMETRIC`` is off the linear marker chain, so the brand drops to
:class:`jcor.core.axioms.Premetric`; see the brand-down rule in
:mod:`jcor.core.axioms`.
"""

from numbers import Real
from typing import Final

import jax.numpy as jnp
import numpy as np
import optax

from jcor.core.axioms import Axioms, Premetric
from jcor.core.typing import Array, ArrayLike, Float  # noqa: TC001  # runtime contract
from jcor.ground.metrics import sphere_chord_distance

# Re-exported verbatim by the ``jcor.discrepancy.exact`` facade, which binds this
# list as its own ``__all__``. Before the private-module split the facade *was*
# this module and carried no ``__all__``, so a star import published every
# public name; this list reproduces that surface exactly rather than narrowing
# it, keeping ``pipeline._kernels.wasserstein`` and the package ``__init__``
# importing the same objects they always did.
__all__ = [
    "EXACT_BALANCED_W1_AXIOMS",
    "EXACT_BALANCED_W1_BRAND",
    "EXACT_BALANCED_W2_AXIOMS",
    "EXACT_BALANCED_W2_BRAND",
    "EXACT_BALANCED_WASSERSTEIN_AXIOMS",
    "EXACT_BALANCED_WASSERSTEIN_BRAND",
    "EXPECTED_ARRAY_DIMENSIONS",
    "UNIT_NORM_ATOL",
    "WASSERSTEIN_ORDER_ONE",
    "WASSERSTEIN_ORDER_TWO",
    "EmptyOrNonMatrixSampleError",
    "InvalidCostMatrixError",
    "InvalidWassersteinOrderError",
    "NegativeCostError",
    "NonfiniteCostError",
    "NonfiniteSampleError",
    "OffUnitSphereSampleError",
    "UnbalancedSampleShapeError",
    "ZeroNormSampleError",
    "balanced_wasserstein1_from_cost",
    "balanced_wasserstein1_from_cost_jax",
    "balanced_wasserstein2_from_cost",
    "balanced_wasserstein2_from_cost_jax",
    "balanced_wasserstein_from_cost",
    "balanced_wasserstein_from_cost_jax",
    "chord_cost_matrix",
    "chord_cost_matrix_jax",
    "exact_balanced_wasserstein",
    "exact_balanced_wasserstein1",
    "exact_balanced_wasserstein1_jax",
    "exact_balanced_wasserstein2",
    "exact_balanced_wasserstein2_jax",
    "exact_balanced_wasserstein_jax",
]

EXPECTED_ARRAY_DIMENSIONS = 2
UNIT_NORM_ATOL: Final = 1e-6
WASSERSTEIN_ORDER_ONE: Final = 1.0
WASSERSTEIN_ORDER_TWO: Final = 2.0

#: Axioms :func:`exact_balanced_wasserstein1` satisfies on ``R^{n x d}`` row
#: samples of equal shape. Flags, never an alias name — see
#: :mod:`jcor.core.axioms`. ``IDENTITY`` is absent by measurement, not by
#: caution: row permutations identify the same empirical measure.
EXACT_BALANCED_W1_AXIOMS: Final = (
    Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY | Axioms.TRIANGLE
)

#: The strongest static marker :data:`EXACT_BALANCED_W1_AXIOMS` implies, for a
#: caller branding a :class:`jcor.core.typing.DMat` assembled from this
#: dissimilarity. ``Semimetric`` would promise ``IDENTITY``, which does not hold.
EXACT_BALANCED_W1_BRAND: Final[type[Premetric]] = Premetric

#: Axioms :func:`exact_balanced_wasserstein2` satisfies, on the same row samples
#: and for the same measured reason as :data:`EXACT_BALANCED_W1_AXIOMS`. The
#: order enters the cost, not the axiom set: ``W2`` over a metric ground cost is
#: a metric on ``P_2``, and the Minkowski inequality carries the triangle
#: inequality through the quadratic aggregation.
EXACT_BALANCED_W2_AXIOMS: Final = (
    Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY | Axioms.TRIANGLE
)

#: The strongest static marker :data:`EXACT_BALANCED_W2_AXIOMS` implies. Same
#: brand-down reason as :data:`EXACT_BALANCED_W1_BRAND`.
EXACT_BALANCED_W2_BRAND: Final[type[Premetric]] = Premetric

#: Shared array-level law for every finite order ``p >= 1`` under the
#: sphere-chord ground metric. W1/W2 names remain compatibility projections.
EXACT_BALANCED_WASSERSTEIN_AXIOMS: Final = EXACT_BALANCED_W1_AXIOMS
EXACT_BALANCED_WASSERSTEIN_BRAND: Final[type[Premetric]] = Premetric


class EmptyOrNonMatrixSampleError(ValueError):
    """Raised when a sample is not a nonempty matrix."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("sample must be a nonempty two-dimensional array")


class NonfiniteSampleError(ValueError):
    """Raised when a sample contains a nonfinite coordinate."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("sample contains nonfinite values")


class ZeroNormSampleError(ValueError):
    """Raised when a sample contains a zero row."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("sample contains a zero-norm row")


class OffUnitSphereSampleError(ValueError):
    """Raised when a caller passes rows the pipeline did not normalize."""

    def __init__(self, worst_deviation: float) -> None:
        """Build the error with the measured unit-norm deviation."""
        super().__init__(
            "sample rows must already be L2-normalized by the calling pipeline; "
            f"worst |norm - 1| is {worst_deviation:.3e}"
        )


class UnbalancedSampleShapeError(ValueError):
    """Raised when balanced samples have different shapes."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__(
            "balanced Wasserstein requires equal-cardinality samples with equal "
            "feature width"
        )


class InvalidCostMatrixError(ValueError):
    """Raised when a balanced-Wasserstein cost matrix is structurally invalid."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("balanced Wasserstein requires a nonempty square cost matrix")


class NonfiniteCostError(ValueError):
    """Raised when a cost matrix contains a nonfinite entry."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("cost matrix contains nonfinite values")


class NegativeCostError(ValueError):
    """Raised when a ground-cost matrix contains a negative entry."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("cost matrix contains negative values")


class InvalidWassersteinOrderError(ValueError):
    """Raised when an eager caller requests an order outside ``[1, inf)``."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__("Wasserstein order p must be finite and at least 1")


def _checked_unit_rows(sample: ArrayLike) -> Float[Array, "m d"]:
    """Validate and return a sample whose rows are already unit norm.

    Args:
        sample: A nonempty ``(n, d)`` array of observations.

    Returns:
        The same rows in the active caller dtype, unchanged.

    Raises:
        EmptyOrNonMatrixSampleError: If ``sample`` is not a nonempty matrix.
        NonfiniteSampleError: If any coordinate is not finite.
        ZeroNormSampleError: If any row has zero norm.
        OffUnitSphereSampleError: If a row is not unit norm within tolerance.

    """
    rows = jnp.asarray(sample)
    if (
        rows.ndim != EXPECTED_ARRAY_DIMENSIONS
        or rows.shape[0] == 0
        or rows.shape[1] == 0
    ):
        raise EmptyOrNonMatrixSampleError
    if not bool(jnp.all(jnp.isfinite(rows))):
        raise NonfiniteSampleError
    norms = jnp.linalg.norm(rows, axis=1)
    if bool(jnp.any(norms <= 0.0)):
        raise ZeroNormSampleError
    deviations = jnp.abs(norms - 1.0)
    if bool(jnp.any(deviations > UNIT_NORM_ATOL)):
        raise OffUnitSphereSampleError(float(deviations.max()))
    return rows


# Compatibility alias: one callable object and one ground-cost implementation.
chord_cost_matrix_jax = sphere_chord_distance


def balanced_wasserstein_from_cost_jax(
    cost: Float[Array, "n n"],
    p: float | Float[Array, ""],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Solve uniform balanced ``W_p`` with the exact Hungarian core.

    Preconditions are owned by the eager door: ``cost`` is a finite,
    nonnegative square matrix and ``p >= 1`` is finite. When ``cost**p``
    would overflow in the active dtype, scaling by the largest cost leaves the
    assignment unchanged; the final root then restores the original scale.
    Ordinary finite W1/W2 inputs keep the unscaled path and its established
    rounding behavior.
    """
    is_w1 = jnp.equal(p, WASSERSTEIN_ORDER_ONE)
    is_w2 = jnp.equal(p, WASSERSTEIN_ORDER_TWO)
    unscaled_power = jnp.where(
        is_w1,
        cost,
        jnp.where(is_w2, jnp.square(cost), jnp.power(cost, p)),
    )
    needs_rescaling = jnp.any(~jnp.isfinite(unscaled_power))
    scale = jnp.max(cost)
    safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    normalized = cost / safe_scale
    scaled_power = jnp.where(
        is_w1,
        normalized,
        jnp.where(is_w2, jnp.square(normalized), jnp.power(normalized, p)),
    )
    powered = jnp.where(needs_rescaling, scaled_power, unscaled_power)
    row_ind, col_ind = optax.assignment.hungarian_algorithm(powered)
    matched_mean = jnp.mean(powered[row_ind, col_ind])
    rooted = jnp.where(
        is_w1,
        matched_mean,
        jnp.where(is_w2, jnp.sqrt(matched_mean), jnp.power(matched_mean, 1.0 / p)),
    )
    return jnp.where(needs_rescaling, scale * rooted, rooted)


def balanced_wasserstein1_from_cost_jax(
    cost: Float[Array, "n n"],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Compatibility specialization of uniform balanced ``W_p`` at ``p=1``."""
    return balanced_wasserstein_from_cost_jax(cost, WASSERSTEIN_ORDER_ONE)


def exact_balanced_wasserstein_jax(
    left: Float[Array, "n d"],
    right: Float[Array, "n d"],
    p: float | Float[Array, ""],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Compute exact ``W_p`` on checked unit rows under sphere-chord cost."""
    costs = sphere_chord_distance(left, right)
    return balanced_wasserstein_from_cost_jax(costs, p)


def exact_balanced_wasserstein1_jax(
    left: Float[Array, "n d"],
    right: Float[Array, "n d"],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Compute exact W1 over already-checked unit-sphere rows."""
    return exact_balanced_wasserstein_jax(left, right, WASSERSTEIN_ORDER_ONE)


def balanced_wasserstein2_from_cost_jax(
    cost: Float[Array, "n n"],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Solve uniform balanced W2 with the exact transformable Hungarian core.

    The assignment runs on the squared ground cost, which is the objective W2
    minimizes; squaring after a W1 assignment would report a feasible but
    suboptimal transport plan.
    """
    return balanced_wasserstein_from_cost_jax(cost, WASSERSTEIN_ORDER_TWO)


def exact_balanced_wasserstein2_jax(
    left: Float[Array, "n d"],
    right: Float[Array, "n d"],
) -> Float[Array, ""]:  # jaxtyping scalar shape
    """Compute exact W2 over already-checked unit-sphere rows."""
    return exact_balanced_wasserstein_jax(left, right, WASSERSTEIN_ORDER_TWO)


def chord_cost_matrix(left: ArrayLike, right: ArrayLike) -> Float[Array, "m n"]:
    """Return the float64 chord-cost matrix between two row samples.

    The samples must have the same positive cardinality and feature width.
    Chord distance between unit rows is
    ``sqrt(2 - 2 * dot(left_i, right_j))``.

    Args:
        left: An ``(n, d)`` sample.
        right: An ``(n, d)`` sample with the same shape as ``left``.

    Returns:
        The ``(n, n)`` ground-cost matrix of chord distances between the
        already-unit rows of ``left`` and of ``right``.

    Raises:
        UnbalancedSampleShapeError: If the two samples do not share a shape.

    """
    left_shape = jnp.shape(left)
    right_shape = jnp.shape(right)
    if (
        len(left_shape) == EXPECTED_ARRAY_DIMENSIONS
        and len(right_shape) == EXPECTED_ARRAY_DIMENSIONS
        and left_shape != right_shape
    ):
        raise UnbalancedSampleShapeError
    left_unit = _checked_unit_rows(left)
    right_unit = _checked_unit_rows(right)
    return sphere_chord_distance(left_unit, right_unit)


def _checked_order(p: object) -> float:
    """Return a finite Wasserstein order on the metric range ``[1, inf)``."""
    if isinstance(p, bool) or not isinstance(p, Real):
        raise InvalidWassersteinOrderError
    order = float(p)
    if not np.isfinite(order) or order < 1.0:
        raise InvalidWassersteinOrderError
    return order


def _checked_cost_matrix(cost: ArrayLike) -> Float[Array, "n n"]:
    """Validate and return a floating square ground-cost matrix.

    Integral inputs are promoted before entering the JAX computation.
    Caller-provided floating dtypes are retained; the shared core scales finite
    costs before applying its power, avoiding avoidable active-dtype overflow.

    Args:
        cost: A nonempty square ``(n, n)`` matrix of nonnegative ground costs.

    Returns:
        The validated matrix in its caller-provided floating dtype, or float32
        for integral inputs.

    Raises:
        InvalidCostMatrixError: If ``cost`` is empty or not square.
        NonfiniteCostError: If an entry is nonfinite.
        NegativeCostError: If any entry is negative.

    """
    raw = np.asarray(cost)
    if (
        raw.ndim != EXPECTED_ARRAY_DIMENSIONS
        or raw.shape[0] == 0
        or raw.shape[0] != raw.shape[1]
    ):
        raise InvalidCostMatrixError
    if not np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(np.float32)
    matrix = jnp.asarray(raw)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise NonfiniteCostError
    if bool(jnp.any(matrix < 0.0)):
        raise NegativeCostError
    return matrix


def balanced_wasserstein_from_cost(cost: ArrayLike, p: object) -> float:
    """Solve exact uniform balanced ``W_p`` from a ground-cost matrix.

    Args:
        cost: A nonempty square matrix of nonnegative ground costs.
        p: A finite Wasserstein order at least one.

    Returns:
        The rooted optimal mean ``p``-power cost between two uniform empirical
        measures of equal cardinality.

    """
    order = _checked_order(p)
    matrix = _checked_cost_matrix(cost)
    return float(balanced_wasserstein_from_cost_jax(matrix, order))


def balanced_wasserstein1_from_cost(cost: ArrayLike) -> float:
    """Solve uniform balanced W1 exactly from a square ground-cost matrix.

    Args:
        cost: A nonempty square ``(n, n)`` matrix of nonnegative ground costs.

    Returns:
        The mean cost of the optimal assignment — the exact ``W1`` between two
        uniform empirical measures of ``n`` atoms each.

    """
    return balanced_wasserstein_from_cost(cost, WASSERSTEIN_ORDER_ONE)


def balanced_wasserstein2_from_cost(cost: ArrayLike) -> float:
    """Solve uniform balanced W2 exactly from a square ground-cost matrix.

    Args:
        cost: A nonempty square ``(n, n)`` matrix of nonnegative ground costs.

    Returns:
        The root-mean squared cost of the assignment optimal for the squared
        cost — the exact ``W2`` between two uniform empirical measures of ``n``
        atoms each.

    """
    return balanced_wasserstein_from_cost(cost, WASSERSTEIN_ORDER_TWO)


def exact_balanced_wasserstein(
    left: ArrayLike,
    right: ArrayLike,
    p: object,
) -> float:
    """Exact uniform empirical ``W_p`` under sphere-chord ground cost.

    The samples must have equal cardinality and width, contain finite nonzero
    rows, and already lie on the L2 unit sphere. The eager boundary validates
    those conditions and ``p >= 1``.
    """
    order = _checked_order(p)
    return balanced_wasserstein_from_cost(chord_cost_matrix(left, right), order)


def exact_balanced_wasserstein1(left: ArrayLike, right: ArrayLike) -> float:
    """Exact empirical balanced W1 under uniform weights and chord ground cost.

    Satisfies :data:`EXACT_BALANCED_W1_AXIOMS` — ``NONNEGATIVE | SYMMETRY |
    TRIANGLE``. Rows must already be normalized by the caller. It is a metric
    on equal-cardinality empirical measures on the unit sphere.

    Args:
        left: An ``(n, d)`` sample, read as ``n`` equally-weighted atoms.
        right: An ``(n, d)`` sample with the same shape as ``left``.

    Returns:
        The exact ``W1`` between the two empirical measures, invariant to the
        row order of either argument.

    """
    return exact_balanced_wasserstein(left, right, WASSERSTEIN_ORDER_ONE)


def exact_balanced_wasserstein2(left: ArrayLike, right: ArrayLike) -> float:
    """Exact empirical balanced W2 under uniform weights and chord ground cost.

    Satisfies :data:`EXACT_BALANCED_W2_AXIOMS` — ``NONNEGATIVE | SYMMETRY |
    TRIANGLE``. Rows must already be normalized by the caller. It is a metric
    on equal-cardinality empirical measures on the unit sphere, and dominates
    :func:`exact_balanced_wasserstein1` by Cauchy--Schwarz.

    Args:
        left: An ``(n, d)`` sample, read as ``n`` equally-weighted atoms.
        right: An ``(n, d)`` sample with the same shape as ``left``.

    Returns:
        The exact ``W2`` between the two empirical measures, invariant to the
        row order of either argument.

    """
    return exact_balanced_wasserstein(left, right, WASSERSTEIN_ORDER_TWO)

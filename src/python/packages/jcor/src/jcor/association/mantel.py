"""Mantel test — permutation test of correlation between two distance matrices.

Position
--------
S4 · association. Consumes two distance matrices (waist 1) and returns a
*coefficient*. A Mantel statistic is not itself a distance, which is why it
sits a stage above ``discrepancy`` rather than beside it.

Precondition
------------
``Axioms.SYMMETRY`` only. The statistic reads the strict upper triangle of both
matrices, so symmetry is what makes that a faithful summary; nothing here needs
the identity of indiscernibles, the triangle inequality, or negative type. Per
:mod:`jcor.core.axioms`, the precondition is stated as a **flag**, never as an
alias name.

The parameters are *not* branded :class:`jcor.core.typing.DMat`. Measured under
pyrefly: ``DMat.values`` is ``Float[Array, ...]`` (a JAX array), and
``DMat(values=<numpy array>)`` is a ``bad-argument-type`` error — so a NumPy
boundary module cannot carry the brand without a cast that lies. See t46.6's
``Publishes`` block.

Tests whether two ``(n, n)`` distance (or dissimilarity) matrices are
correlated beyond what is expected under a random relabeling of the objects.
Standard tool in ecology, genetics, and spatial statistics for comparing two
independently-derived distance structures on the same ``n`` objects; the
permutation scheme (jointly permuting rows/columns of one matrix) correctly
accounts for the non-independence of a distance matrix's ``n(n-1)/2`` entries,
which a naive elementwise permutation test would not.

The numerical statistic and relabeling null are JAX kernels. Eager public doors
own validation, percentile reports, and NumPy artifact conversion; compiled
callers use :func:`mantel_correlation_kernel` or
:func:`mantel_permutation_kernel` directly.
"""

# ruff: noqa: F722, F821, UP037  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata

from jcor.core.random import cell_key, permutation_batch
from jcor.core.typing import Array, ArrayLike, Int, Static, as_index  # noqa: TC001
from jcor.core.typing import Float as JaxFloat

#: Host-facing aliases retained as *names* so the ~10 signatures below read the
#: same, but now resolving to JAX. t65.3 removed their `NDArray[np.float64]` /
#: `NDArray[np.intp]` definitions — the alias-typed returns the import fence
#: cannot see. `np.intp` is one of only two symbols in the package with no `jnp`
#: counterpart; index vectors are `int32` here, which is what
#: `jax.random.permutation` and `jnp.triu_indices` already produce.
type Float = JaxFloat[Array, "..."]
type Index = Int[Array, "..."]
MantelMethod = Literal["pearson", "spearman"]
_MATRIX_NDIM = 2
_MIN_TRIANGLE_ENTRIES = 2

__all__ = [
    "MantelTestResult",
    "PairedMantelBootstrapResult",
    "mantel_bootstrap_ci",
    "mantel_correlation_kernel",
    "mantel_permutation_kernel",
    "mantel_test",
    "paired_mantel_bootstrap",
]


class MantelTestResult(NamedTuple):
    """Result of :func:`mantel_test`.

    Attributes:
        correlation: Observed correlation between the upper-triangular
            entries of the two input matrices.
        pvalue: One-sided right-tail permutation p-value with add-one
            smoothing, ``(1 + #{null >= observed}) / (1 + num_permutations)``
            — appropriate for the directional (positive-association)
            hypothesis under test. Floored at ``1 / (1 + num_permutations)``.
            The bootstrap confidence-interval path is a separate procedure
            and does not use this p-value.
        method: Correlation method used (``"pearson"`` or ``"spearman"``).
        num_permutations: Number of row/column permutations performed.
        null_distribution: Permuted correlations, shape
            ``(num_permutations,)``.

    """

    correlation: float
    pvalue: float
    method: MantelMethod
    num_permutations: int
    null_distribution: Float


class PairedMantelBootstrapResult(NamedTuple):
    """Paired bootstrap summary for a difference of Mantel correlations."""

    gap_mean: float
    gap_ci_low: float
    gap_ci_high: float
    gap_prob_le_zero: float
    gap_n_boot: int


def _upper_triangle(x: Float, indices: tuple[Index, Index]) -> Float:
    return x[indices]


@partial(jax.jit, static_argnames=("method",))
def mantel_correlation_kernel(
    x: JaxFloat[Array, "p"],
    y: JaxFloat[Array, "p"],
    method: Static[MantelMethod] = "pearson",
) -> JaxFloat[Array, ""]:
    """Return one Pearson or tie-aware Spearman coefficient on graph."""
    x_values = jnp.asarray(x)
    y_values = jnp.asarray(y)
    if method == "spearman":
        x_values = rankdata(x_values, method="average")
        y_values = rankdata(y_values, method="average")
    x_centered = x_values - jnp.mean(x_values)
    y_centered = y_values - jnp.mean(y_values)
    denominator = jnp.linalg.norm(x_centered) * jnp.linalg.norm(y_centered)
    return jnp.sum(x_centered * y_centered) / denominator


@partial(jax.jit, static_argnames=("method",))
def mantel_permutation_kernel(
    x: JaxFloat[Array, "n n"],
    y: JaxFloat[Array, "n n"],
    permutations: Int[Array, "b n"],
    method: Static[MantelMethod] = "pearson",
) -> tuple[JaxFloat[Array, ""], JaxFloat[Array, "b"]]:
    """Return the observed Mantel coefficient and one vmapped relabeling null."""
    n = x.shape[0]
    triangle = jnp.triu_indices(n, k=1)
    x_values = x[triangle]
    y_values = y[triangle]
    observed = mantel_correlation_kernel(x_values, y_values, method)

    def permuted_correlation(
        permutation: Int[Array, "n"],
    ) -> JaxFloat[Array, ""]:
        permuted = y[permutation[:, None], permutation[None, :]]
        return mantel_correlation_kernel(x_values, permuted[triangle], method)

    return observed, jax.vmap(permuted_correlation)(permutations)


def _require_method(method: MantelMethod) -> None:
    """Reject an unsupported correlation method at every public entry.

    Every public door calls this. Until the bootstrap loops became traceable the
    private ``_correlate`` helper carried the only runtime spelling of this
    guard, and ``mantel_test`` — which never called it — had none: an untyped
    caller passing ``method="kendall"`` silently received a Pearson coefficient,
    because the kernel's ``if method == "spearman"`` simply fell through. The
    guard now lives where the contract is documented instead of where an
    implementation detail happened to put it.
    """
    if method not in ("pearson", "spearman"):
        message = f"Unknown method '{method}'. Choose 'pearson' or 'spearman'."
        raise ValueError(message)


def _masked_correlation(
    x: JaxFloat[Array, "p"],
    y: JaxFloat[Array, "p"],
    mask: JaxFloat[Array, "p"],
    method: Static[MantelMethod],
) -> tuple[JaxFloat[Array, ""], JaxFloat[Array, ""]]:
    """Correlate ``x`` against ``y`` over the kept entries only, at fixed shape.

    ``mask`` is 1.0 on kept entries and 0.0 elsewhere. Dropped pairs are
    neutralised rather than compacted, so every bootstrap draw presents the same
    shape and the whole reduction stays ``vmap``/``scan``-safe — the same
    fixed-shape masking the GEL subspace uses to keep a data-dependent rank
    traceable.

    Returns:
        Tuple ``(coefficient, denominator)``. ``denominator`` is the product of
        the two centred norms; it is zero exactly when one side is constant over
        the kept entries, which is the traced spelling of the eager ``std == 0``
        skip the callers used to apply per draw.

    """
    if method == "spearman":
        # Rank among kept entries only. Sending dropped entries to +inf parks
        # them in the tail, so the kept ranks come out 1..k — identical to what
        # rankdata would return on the compacted vector.
        x = rankdata(jnp.where(mask > 0.0, x, jnp.inf), method="average")
        y = rankdata(jnp.where(mask > 0.0, y, jnp.inf), method="average")
    count = jnp.sum(mask)
    safe_count = jnp.where(count > 0.0, count, 1.0)
    x_centered = mask * (x - jnp.sum(mask * x) / safe_count)
    y_centered = mask * (y - jnp.sum(mask * y) / safe_count)
    denominator = jnp.linalg.norm(x_centered) * jnp.linalg.norm(y_centered)
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    coefficient = jnp.sum(x_centered * y_centered) / safe_denominator
    return coefficient, denominator


@partial(jax.jit, static_argnames=("method", "min_distinct_pairs", "chunk"))
def mantel_bootstrap_kernel(
    matrices: JaxFloat[Array, "k n n"],
    node_indices: Int[Array, "b n"],
    left: Int[Array, "q"],
    right: Int[Array, "q"],
    method: Static[MantelMethod] = "pearson",
    min_distinct_pairs: Static[int] = _MIN_TRIANGLE_ENTRIES,
    chunk: Static[int] = 1,
) -> tuple[JaxFloat[Array, "b q"], JaxFloat[Array, "b"]]:
    """Correlate every ``(left, right)`` matrix pair under every node resample.

    One compiled scan replaces the host loop that dispatched a kernel and synced
    a scalar back per draw. ``chunk`` is the number of draws vmapped inside each
    scan step: the batched intermediate is ``chunk x k x p``, so it is chosen by
    the caller from the triangle size rather than fixed — a blanket ``vmap`` over
    a 1000-draw schedule on a few-hundred-node matrix would materialise several
    hundred megabytes at once.

    Returns:
        Tuple ``(coefficients, valid)`` with ``coefficients`` shape ``(b, q)``
        and ``valid`` the ``(b,)`` 0/1 flag marking draws that cleared both the
        distinct-pair floor and the constant-input guard. Invalid draws still
        occupy a row; the eager door drops them.

    """
    n_nodes = matrices.shape[-1]
    triangle_i, triangle_j = jnp.triu_indices(n_nodes, k=1)

    def one_draw(
        node_index: Int[Array, "n"],
    ) -> tuple[JaxFloat[Array, "q"], JaxFloat[Array, ""]]:
        rows = node_index[triangle_i]
        columns = node_index[triangle_j]
        # A resampled pair naming the same original node twice is discarded
        # rather than contributing an artificial (0, 0) observation.
        mask = (rows != columns).astype(matrices.dtype)
        values = matrices[:, rows, columns]
        coefficients, denominators = jax.vmap(
            lambda a, b: _masked_correlation(values[a], values[b], mask, method)
        )(left, right)
        valid = (jnp.sum(mask) >= min_distinct_pairs) & jnp.all(denominators > 0.0)
        return coefficients, valid.astype(matrices.dtype)

    # Prefer a single vmap. `lax.map(batch_size=len)` is not equivalent to it —
    # the scan wrapper it emits even for one chunk measured several times slower
    # than the plain vmap it should reduce to. The scan earns its overhead only
    # when the batched intermediate would not fit.
    if node_indices.shape[0] <= chunk:
        return jax.vmap(one_draw)(node_indices)
    return jax.lax.map(one_draw, node_indices, batch_size=chunk)


#: Cap on the ``chunk x k x p`` intermediate one scan step may materialise.
#: Bounds peak memory independently of the resample count and node count.
_BOOTSTRAP_CHUNK_ELEMENTS = 1 << 22


def _run_bootstrap(
    matrices: tuple[ArrayLike, ...],
    indices: Index,
    pairs: tuple[tuple[int, int], ...],
    method: MantelMethod,
    min_distinct_pairs: int,
) -> Float:
    """Eager float64 door: run the whole resample schedule in one traced pass.

    Returns:
        Shape ``(kept, len(pairs))`` — one row per draw that cleared both the
        distinct-pair floor and the constant-input guard, one column per
        requested matrix pair. Dropping the failed draws here keeps the
        skip semantics of the retired per-draw ``continue`` in one place.

    """
    # Convert before reading `.shape`: `ArrayLike` admits bare scalars, so the
    # attribute access is only well-typed once the value is an array.
    stacked = jnp.stack([jnp.asarray(matrix) for matrix in matrices])
    index_matrix = jnp.asarray(indices)
    n_nodes = stacked.shape[1]
    span = max(n_nodes * (n_nodes - 1) // 2 * len(matrices), 1)
    chunk = max(1, min(int(index_matrix.shape[0]), _BOOTSTRAP_CHUNK_ELEMENTS // span))
    left = jnp.asarray([pair[0] for pair in pairs])
    right = jnp.asarray([pair[1] for pair in pairs])
    coefficients, valid = mantel_bootstrap_kernel(
        stacked,
        index_matrix,
        left,
        right,
        method,
        min_distinct_pairs,
        chunk,
    )
    kept = jnp.asarray(valid) > 0.0
    return jnp.asarray(coefficients)[kept]


def _validate_triangle(matrix: Float, *, name: str) -> Float:
    """Validate one finite symmetric matrix and return its strict triangle."""
    if not bool(jnp.all(jnp.isfinite(matrix))):
        message = f"{name} must contain only finite values."
        raise ValueError(message)
    if not bool(jnp.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)):
        message = f"{name} must be symmetric."
        raise ValueError(message)
    indices = jnp.triu_indices(matrix.shape[0], k=1)
    triangle = _upper_triangle(matrix, indices)
    if triangle.size < _MIN_TRIANGLE_ENTRIES:
        message = f"{name} must have at least two strict upper-triangle entries."
        raise ValueError(message)
    if bool(jnp.all(triangle == triangle[0])):
        message = f"{name}'s strict upper triangle must be nonconstant."
        raise ValueError(message)
    return triangle


def _validate_bootstrap_matrices(
    matrices: tuple[ArrayLike, ...],
    bootstrap_indices: ArrayLike,
) -> tuple[tuple[Float, ...], Index]:
    """Validate aligned square matrices and a fixed node-resample schedule."""
    converted = tuple(jnp.asarray(matrix) for matrix in matrices)
    first_shape = converted[0].shape
    if (
        any(matrix.ndim != _MATRIX_NDIM for matrix in converted)
        or len({matrix.shape for matrix in converted}) != 1
        or first_shape[0] != first_shape[1]
    ):
        message = "bootstrap matrices must be equal-shaped square matrices."
        raise ValueError(message)
    if any(
        not bool(jnp.all(jnp.isfinite(matrix)))
        or not bool(jnp.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12))
        for matrix in converted
    ):
        message = "bootstrap matrices must be finite and symmetric."
        raise ValueError(message)
    indices = jnp.asarray(bootstrap_indices)
    n_nodes = first_shape[0]
    message = (
        "bootstrap_indices must be a nonempty integer matrix with shape "
        f"(resamples, {n_nodes}) and values in [0, {n_nodes})."
    )
    if indices.ndim != _MATRIX_NDIM:
        raise ValueError(message)
    if indices.shape[0] < 1 or indices.shape[1] != n_nodes:
        raise ValueError(message)
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError(message)
    if bool(jnp.any(indices < 0)) or bool(jnp.any(indices >= n_nodes)):
        raise ValueError(message)
    return converted, indices


def mantel_bootstrap_ci(
    x: ArrayLike,
    y: ArrayLike,
    bootstrap_indices: ArrayLike,
    method: MantelMethod = "pearson",
    *,
    min_distinct_pairs: int = 2,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> tuple[float, float]:
    """Return a node-bootstrap interval for one Mantel correlation.

    The caller supplies the resample schedule so artifact adapters can preserve
    an existing RNG stream while numerical code remains RNG-agnostic. Pairs
    whose resampled endpoints name the same original node are discarded rather
    than adding artificial ``(0, 0)`` observations.
    """
    (x_matrix, y_matrix), indices = _validate_bootstrap_matrices(
        (x, y),
        bootstrap_indices,
    )
    _require_method(method)
    correlations = _run_bootstrap(
        (x_matrix, y_matrix),
        indices,
        ((0, 1),),
        method,
        min_distinct_pairs,
    )[:, 0]
    if correlations.size == 0:
        return float("nan"), float("nan")
    return (
        float(jnp.percentile(correlations, lower_percentile)),
        float(jnp.percentile(correlations, upper_percentile)),
    )


def paired_mantel_bootstrap(
    focal: ArrayLike,
    comparator: ArrayLike,
    target: ArrayLike,
    bootstrap_indices: ArrayLike,
    method: MantelMethod = "pearson",
    *,
    min_distinct_pairs: int = 2,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> PairedMantelBootstrapResult:
    """Bootstrap ``corr(comparator, target) - corr(focal, target)`` by node.

    Both correlations use the same node resample in each draw. This retains
    their shared sampling noise and estimates the paired difference rather than
    subtracting two independent intervals.
    """
    (focal_matrix, comparator_matrix, target_matrix), indices = (
        _validate_bootstrap_matrices(
            (focal, comparator, target),
            bootstrap_indices,
        )
    )
    _require_method(method)
    # Pair 0 correlates focal against target, pair 1 comparator against target.
    # Both arms read the same draw, which is the point of the paired design.
    kept = _run_bootstrap(
        (focal_matrix, comparator_matrix, target_matrix),
        indices,
        ((0, 2), (1, 2)),
        method,
        min_distinct_pairs,
    )
    if kept.shape[0] == 0:
        nan = float("nan")
        return PairedMantelBootstrapResult(nan, nan, nan, nan, 0)
    values = kept[:, 1] - kept[:, 0]
    return PairedMantelBootstrapResult(
        gap_mean=float(values.mean()),
        gap_ci_low=float(jnp.percentile(values, lower_percentile)),
        gap_ci_high=float(jnp.percentile(values, upper_percentile)),
        gap_prob_le_zero=float((values <= 0.0).mean()),
        gap_n_boot=int(values.size),
    )


def mantel_test(
    x: ArrayLike,
    y: ArrayLike,
    method: MantelMethod = "pearson",
    num_permutations: int = 10_000,
    seed: int = 42,
) -> MantelTestResult:
    """Mantel test: permutation test of correlation between two distance matrices.

    Computes the (Pearson or Spearman) correlation between the upper-triangular
    entries of two symmetric ``(n, n)`` matrices, then tests significance by
    repeatedly permuting the rows/columns of ``y`` (jointly, preserving its
    internal structure) and recomputing the correlation.

    Args:
        x: Symmetric matrix, shape ``(n, n)``.
        y: Symmetric matrix, shape ``(n, n)``.  Must match ``x``'s shape.
        method: Correlation statistic — ``"pearson"`` or ``"spearman"``.
        num_permutations: Number of row/column permutations of ``y``.
        seed: RNG seed for the permutation draws.

    Returns:
        :class:`MantelTestResult`.

    Raises:
        ValueError: If ``x`` and ``y`` are not equal-shaped square matrices,
            if either strict upper triangle is nonfinite or constant, if either
            matrix is asymmetric, if too few upper-triangle entries exist, if
            ``num_permutations`` is not positive, or if ``method`` is not
            ``"pearson"``/``"spearman"``.

    """
    x_matrix = jnp.asarray(x)
    y_matrix = jnp.asarray(y)
    if (
        x_matrix.ndim != _MATRIX_NDIM
        or y_matrix.ndim != _MATRIX_NDIM
        or x_matrix.shape != y_matrix.shape
        or x_matrix.shape[0] != x_matrix.shape[1]
    ):
        message = (
            "x and y must be equal-shaped square matrices, got "
            f"{x_matrix.shape} and {y_matrix.shape}"
        )
        raise ValueError(message)
    if as_index(num_permutations) is None or num_permutations <= 0:
        message = "num_permutations must be a positive integer."
        raise ValueError(message)
    _require_method(method)

    n = x_matrix.shape[0]
    _validate_triangle(x_matrix, name="x")
    _validate_triangle(y_matrix, name="y")
    key = cell_key(seed, "association", "mantel_test")
    permutations = permutation_batch(key, jnp.arange(n), num_permutations)
    observed_array, null_array = mantel_permutation_kernel(
        jnp.asarray(x_matrix),
        jnp.asarray(y_matrix),
        permutations,
        method,
    )
    observed = float(observed_array)
    null_distribution = jnp.asarray(null_array)
    if not math.isfinite(observed):
        message = "the observed Mantel correlation is undefined."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(null_distribution))):
        message = "the Mantel permutation null contains undefined correlations."
        raise ValueError(message)

    # One-sided right-tail p-value with add-one smoothing (Davison & Hinkley,
    # 1997): avoids a reported p-value of exactly zero and matches the
    # directional (positive-association) hypothesis under test. Floored at
    # 1 / (1 + num_permutations).
    pvalue = float(
        (1 + jnp.sum(null_distribution >= observed)) / (1 + num_permutations)
    )

    return MantelTestResult(
        correlation=observed,
        pvalue=pvalue,
        method=method,
        num_permutations=num_permutations,
        null_distribution=null_distribution,
    )

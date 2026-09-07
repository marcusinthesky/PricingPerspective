"""Raw MMD² estimate matrices and branded V-statistic distance matrices."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax import jit, lax, vmap

from jcor.core.typing import Array, Float  # noqa: TC001  # runtime contract
from jcor.discrepancy._mmd.contracts import (
    _require_u_sample_size,
    _require_v_distance,
)
from jcor.discrepancy._mmd.scalar import _shared_reference
from jcor.ground.similarities import (
    LINEAR_KERNEL,
    SimilarityStrategy,
    gram,
)
from jcor.sample.packed import (  # noqa: TC001  # runtime contract
    PackedClouds,
    masked_row_mask,
    masked_sum,
    pack,
    pair_chunks,
)

__all__ = ["mmd_matrix_values", "mmd_squared_matrix"]

if TYPE_CHECKING:
    from jcor.ground.similarities import PositiveSemidefiniteKernelLaw

_EXPECTED_SAMPLE_DIMENSIONS = 2
_FEATURE_AXIS = 1


def _clouds(sample_sets: list[Array]) -> list[Array]:
    """Validate the list-of-clouds convention and coerce to JAX arrays.

    Row counts may differ across clouds; every cloud must be rank two and share
    one feature width.  The width is the only shape fact the estimator algebra
    needs, because each divisor is a product of the clouds' own row counts.
    """
    if not sample_sets:
        message = "sample_sets must contain at least one cloud"
        raise ValueError(message)
    clouds = [jnp.asarray(sample) for sample in sample_sets]
    for cloud in clouds:
        if len(cloud.shape) != _EXPECTED_SAMPLE_DIMENSIONS:
            message = f"sample clouds must be two-dimensional, got {cloud.shape}"
            raise ValueError(message)
    width = clouds[0].shape[_FEATURE_AXIS]
    for cloud in clouds[1:]:
        if cloud.shape[_FEATURE_AXIS] != width:
            message = (
                f"sample clouds must share feature width {width}, "
                f"got {cloud.shape[_FEATURE_AXIS]}"
            )
            raise ValueError(message)
    return clouds


def _require_nonempty_clouds(clouds: list[Array]) -> None:
    """Reject a zero-row cloud before it reaches the packed traversal.

    A zero count would make both the ``n_i n_j`` cross divisor and the
    ``n_i^2`` within divisor zero, and :func:`jcor.sample.packed.pack` has no
    row to pad from.
    """
    counts = [cloud.shape[0] for cloud in clouds]
    if any(count == 0 for count in counts):
        message = (
            f"sample clouds must each contain at least one row, got row counts {counts}"
        )
        raise ValueError(message)


@partial(
    jit,
    static_argnames=("kernel", "unbiased"),
)
def _mmd_squared_matrix_packed(
    packed: PackedClouds,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: Float[Array, " d"],
    *,
    unbiased: bool,
) -> Float[Array, "sets sets"]:
    """Evaluate the pairwise MMD² grid over zero-padded clouds of any sizes.

    This is the producer's only core. Cross terms walk the shared
    upper-triangle chunking (:func:`jcor.sample.packed.pair_chunks`), which is
    what the sibling energy producers use, and within-cloud terms are computed
    once per cloud under ``vmap``.

    Every divisor is a product of true row counts, so estimator weights match
    the per-pair scalar calls exactly. Padded rows are zero vectors, which the
    cosine kernel and the angular/cosine grounds map to ``NaN``; every reduction
    here therefore goes through :func:`jcor.sample.packed.masked_sum` or an
    explicit :func:`jax.numpy.where`, never through a multiplicative mask, since
    ``NaN * 0`` is ``NaN``. Equal row counts are not a special case: ``pack``
    then yields ``m_max == m`` and writes no padding at all, so every mask is
    all-true and the reduction is exact.
    """
    data = packed.data
    counts = packed.counts
    n_sets, m_max = data.shape[0], data.shape[1]

    # Inner closures take vmap/lax.map tracers; they carry the array *type* but
    # no shape string, because the batching rule rewrites the rank the hook sees.
    def within_terms(sample: Array, count: Array) -> tuple[Array, Array]:
        kxx = gram(sample, sample, kernel, reference=reference)
        total = masked_sum(kxx, count, count)
        size = count.astype(kxx.dtype)
        zero = jnp.zeros((), dtype=kxx.dtype)
        active = masked_row_mask(m_max, count)
        trace = jnp.sum(jnp.where(active, jnp.diagonal(kxx), zero))
        return total / (size * size), (total - trace) / (size * (size - 1.0))

    within_v, within_u = vmap(within_terms)(data, counts)
    within = within_u if unbiased else within_v

    row_indices, column_indices, pair_count, pair_grid = pair_chunks(n_sets, k=1)

    def pair_value(pair: Array) -> Array:
        left, right = pair[0], pair[1]
        kxy = gram(data[left], data[right], kernel, reference=reference)
        kyx = gram(data[right], data[left], kernel, reference=reference)
        denominator = counts[left].astype(kxy.dtype) * counts[right].astype(kxy.dtype)
        forward = masked_sum(kxy, counts[left], counts[right])
        reverse = masked_sum(kyx, counts[right], counts[left])
        cross = 0.5 * (forward + reverse) / denominator
        return within[left] + within[right] - 2.0 * cross

    pair_values = lax.map(vmap(pair_value), jnp.asarray(pair_grid)).reshape(-1)[
        :pair_count
    ]
    # The V-statistic diagonal is exactly zero; the U-statistic diagonal is the
    # eager path's ``i == j`` cell, 2 * (U_ii - V_ii), rebuilt without a second
    # Gram block.
    diagonal = 2.0 * (within_u - within_v) if unbiased else jnp.zeros_like(within_v)
    estimates = (
        jnp.zeros((n_sets, n_sets), dtype=pair_values.dtype)
        .at[row_indices, column_indices]
        .set(pair_values)
        .at[column_indices, row_indices]
        .set(pair_values)
    )
    return estimates.at[jnp.diag_indices(n_sets)].set(
        diagonal.astype(pair_values.dtype)
    )


def mmd_squared_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: bool = False,
) -> Float[Array, "sets sets"]:
    """Construct an unbranded symmetric matrix of pairwise MMD² estimates.

    Unlike :func:`mmd_matrix`, this function permits ``unbiased=True`` and
    returns the signed U-statistic estimates without a square root, clamp, or
    ``DMat`` brand.  The U-statistic diagonal need not be zero.

    Args:
        sample_sets: Nonempty list of rank-two ``(m_k, d)`` clouds. Row counts
            ``m_k`` may differ; the feature width ``d`` must be shared, and no
            cloud may have zero rows.
        kernel: Declared PSD strategy with parameters and law closed.
        reference: Reference shared by every distance-induced Gram block.
        unbiased: Drop the within-sample diagonals.

    Returns:
        Symmetric raw-estimate matrix of shape ``(sets, sets)``.

    Raises:
        InsufficientUStatisticSampleError: If a U-statistic cloud is singleton.
        TypeError: If a bare string or callable bypasses declaration.
        ValueError: If the list is empty, a cloud is not rank two, the feature
            widths differ, or any cloud has zero rows.

    Notes:
        Normalizing kernels and grounds require the evidence named by their
        strategy. This transformable array door leaves an out-of-domain value
        visible as ``NaN``; checked carrier doors reject eagerly.

        Every input, equal-size or ragged, routes through
        :func:`jcor.sample.packed.pack` and one compiled core that walks the
        shared upper-triangle chunking. Reductions are ``where``-based, so the
        ``NaN`` a padded zero row induces under a normalizing kernel or ground
        cannot contaminate a result, and divisors are products of true counts.

        The single traversal supersedes the pre-t62 eager pair loop. Measured
        over 24 cases (``K=5, m=11, d=4`` and ``K=9, m=37, d=9`` x six kernel
        families x U/V), equal-size results move by at most
        ``|delta| = 2.1316e-14`` in float64; 12 of the 24 are still bit-identical
        and every difference is batched ``dot_general``/``lax.map`` tiling, not a
        weight change. The author accepted that drift in exchange for the shared
        traversal; ``tests/discrepancy/test_mmd.py`` pins it at
        ``atol=1e-13, rtol=1e-12`` against a reference eager implementation.

    """
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    clouds = _clouds(sample_sets)
    # Before ``_shared_reference``, which reads ``clouds[0][0]``.
    _require_nonempty_clouds(clouds)
    if unbiased:
        for cloud in clouds:
            _require_u_sample_size(cloud.shape[0])
    return _mmd_squared_matrix_packed(
        pack(clouds, require_positive_counts=True),
        kernel,
        _shared_reference(clouds[0], reference),
        unbiased=unbiased,
    )


def mmd_matrix_values(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: Literal[False] = False,
) -> Float[Array, "sets sets"]:
    """Construct the raw rooted V-statistic MMD value matrix.

    Args:
        sample_sets: Nonempty list of rank-two ``(m_k, d)`` clouds, admitted on
            exactly the terms :func:`mmd_squared_matrix` admits: row counts
            ``m_k`` may differ, the feature width ``d`` may not, and no cloud
            may have zero rows.
        kernel: Declared PSD strategy with parameters and law closed.
        reference: Reference shared by every distance-induced Gram block.
        unbiased: Must remain ``False``; retained to reject old calls clearly.

    Returns:
        Symmetric rooted MMD values with a zero diagonal. Semantic brands are
        attached only by :func:`rooted_mmd_matrix`.

    Raises:
        UnbiasedMmdDistanceError: If ``unbiased=True`` is supplied at runtime.
        ValueError: If :func:`mmd_squared_matrix` rejects the cloud list.

    """
    _require_v_distance(unbiased=unbiased)
    squared = mmd_squared_matrix(
        sample_sets,
        kernel=kernel,
        reference=reference,
        unbiased=False,
    )
    values = jnp.sqrt(jnp.maximum(squared, 0.0))
    return values.at[jnp.diag_indices(values.shape[0])].set(0.0)

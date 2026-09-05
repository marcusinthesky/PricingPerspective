"""Batched matrix estimators for the energy-discrepancy family.

Position
--------
rank 3 (WAIST 1) · private implementation behind
:mod:`jcor.discrepancy.energy`
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit, lax, vmap

from jcor.core.axioms import Law, PremetricLaw  # noqa: TC001  # runtime contract
from jcor.core.domains import (  # noqa: TC001  # runtime contract
    DirectOrigin,
    EmpiricalDistribution,
)
from jcor.core.matrices import (  # noqa: TC001  # runtime contract
    DMat,
    EnergyFunctional,
    PairwiseEstimateMatrix,
    VStatisticScheme,
    _dmat,
    _pairwise_estimate,
)
from jcor.core.statistics import (  # noqa: TC001  # runtime contract
    VStatisticEstimator,
)
from jcor.core.typing import (  # noqa: TC001  # runtime; see jcor.core.typing
    Array,
    Float,
)
from jcor.discrepancy._mmd.components import _triu_pair_chunks
from jcor.discrepancy._mmd.energy.scalar import _resolve_distribution_inputs
from jcor.discrepancy.provenance import (  # noqa: TC001  # runtime contract
    ExpectedGroundDistance,
)
from jcor.ground.config import (  # noqa: TC001  # runtime contract
    GroundDistanceSelection,
)
from jcor.ground.metrics import GroundDistance, cdist  # noqa: TC001  # runtime contract
from jcor.sample.packed import masked_mean, pack

if TYPE_CHECKING:
    from jcor.sample.packed import PackedClouds

__all__ = [
    "energy_distance_matrix",
    "energy_functional_matrix",
    "mean_distance_estimate_matrix",
    "mean_distance_matrix",
]

_CLOUD_DIMENSIONS = 2


def _shared_cloud_shape(sample_sets: list[Array]) -> tuple[int, ...] | None:
    """Return the one shape every cloud shares, or ``None`` when they differ.

    Mirrors the shape-uniformity branch of
    :func:`jcor.discrepancy._mmd.components._compute_energy_components` rather than
    inventing a second dispatch pattern.

    Args:
        sample_sets: Candidate clouds, not yet validated.

    Returns:
        The single shared shape tuple, or ``None`` for an empty or ragged list.

    """
    shapes = {tuple(jnp.shape(sample_set)) for sample_set in sample_sets}
    return shapes.pop() if len(shapes) == 1 else None


def _require_stackable_shape(shape: tuple[int, ...]) -> None:
    """Reject a uniform shape that the stacked equal-size core cannot consume.

    The ragged branch inherits these laws from :func:`jcor.sample.packed.pack`;
    a uniform list never reaches ``pack``, so the same two conditions are
    enforced here.

    Args:
        shape: The shape shared by every cloud.

    Raises:
        ValueError: If the clouds are not rank two, or carry zero rows.

    """
    if len(shape) != _CLOUD_DIMENSIONS:
        message = f"every cloud must be two-dimensional, got shape {shape}"
        raise ValueError(message)
    if shape[0] == 0:
        message = f"every cloud must have at least one row, got shape {shape}"
        raise ValueError(message)


def _normalized_clouds(sample_sets: list[Array]) -> list[Array]:
    """Promote ragged clouds to one global result dtype before packing.

    :func:`jcor.sample.packed.pack` preserves the *first* cloud's dtype, which
    would silently truncate a mixed-precision list. Promoting eagerly keeps the
    packed buffer at the dtype every input can be represented in.

    Args:
        sample_sets: Clouds to promote; may be empty.

    Returns:
        The same clouds as JAX arrays sharing one promoted dtype.

    """
    arrays = [jnp.asarray(sample_set) for sample_set in sample_sets]
    if not arrays:
        return arrays
    dtype = jnp.result_type(*arrays)
    return [array.astype(dtype) for array in arrays]


@partial(jit, static_argnums=(1, 2))
def _energy_distance_matrix_core(
    a: Float[Array, "sets m d"],
    exponent: float,
    metric: GroundDistance[Law],
) -> Float[Array, "sets sets"]:
    """Batched symmetric energy-distance matrix over stacked equal-size samples.

    Args:
        a: Stacked samples of shape (n_sets, m, d).
        exponent: Distance exponent applied to the metric.
        metric: Declared ground-distance strategy.

    Returns:
        Symmetric (n_sets, n_sets) matrix with a zero diagonal.

    """
    n = a.shape[0]
    size_sq = a.shape[1] * a.shape[1]

    # Compute only the i<=j cross sums (upper triangle + diagonal) and mirror;
    # the discarded lower triangle is redundant matmul work (see
    # _triu_pair_chunks). The diagonal supplies the within-set terms.
    ii, jj, n_pairs, pair_chunks = _triu_pair_chunks(n, 0)

    def _one_pair(pair: Array) -> Array:
        i, j = pair[0], pair[1]
        return jnp.sum(cdist(a[i], a[j], metric=metric) ** exponent)

    sums = lax.map(vmap(_one_pair), jnp.asarray(pair_chunks))
    sums = sums.reshape(-1)[:n_pairs]
    cross = (
        jnp.zeros((n, n), dtype=sums.dtype).at[ii, jj].set(sums).at[jj, ii].set(sums)
    )
    within = jnp.diagonal(cross)
    # Replicate the scalar core term-by-term
    # (2·Σxy/nm - Σxx/nn - Σyy/mm); all sets share size m, so every divisor is
    # size_sq. Separate divisions and subtraction order match the scalar path.
    full = (
        (2.0 * cross) / size_sq - within[:, None] / size_sq - within[None, :] / size_sq
    )
    upper = jnp.triu(full, 1)
    return upper + upper.T


@partial(jit, static_argnums=(1, 2))
def _energy_distance_matrix_packed_core(
    clouds: PackedClouds,
    exponent: float,
    metric: GroundDistance[Law],
) -> Float[Array, "sets sets"]:
    """Symmetric energy-distance matrix over zero-padded unequal-size clouds.

    Each ``i <= j`` block is reduced with
    :func:`jcor.sample.packed.masked_mean`, so every divisor is the product of
    the two **true** counts rather than the padded ``m_max``: the estimator
    weights are exactly those of the per-pair scalar call. The reduction is
    ``jnp.where``-based, never a multiplicative mask, because zero padding rows
    make angular/cosine grounds evaluate to ``NaN`` outside the active block and
    ``NaN * 0`` is still ``NaN``.

    Args:
        clouds: Padded clouds and their true row counts.
        exponent: Distance exponent applied to the metric.
        metric: Declared ground-distance strategy.

    Returns:
        Symmetric ``(K, K)`` matrix with an exactly-zero diagonal.

    """
    data = clouds.data
    counts = clouds.counts
    n = data.shape[0]

    ii, jj, n_pairs, pair_chunks = _triu_pair_chunks(n, 0)

    def _one_pair(pair: Array) -> Array:
        i, j = pair[0], pair[1]
        distances = cdist(data[i], data[j], metric=metric) ** exponent
        return masked_mean(distances, counts[i], counts[j])

    means = lax.map(vmap(_one_pair), jnp.asarray(pair_chunks))
    means = means.reshape(-1)[:n_pairs]
    mean_block = (
        jnp.zeros((n, n), dtype=means.dtype).at[ii, jj].set(means).at[jj, ii].set(means)
    )
    within = jnp.diagonal(mean_block)
    # Replicate the scalar core term-by-term (2·E[dxy] - E[dxx] - E[dyy]); the
    # division by true counts already happened inside each masked_mean, so no
    # single shared size_sq divisor appears (and none would be correct here).
    full = 2.0 * mean_block - within[:, None] - within[None, :]
    upper = jnp.triu(full, 1)
    return upper + upper.T


def energy_distance_matrix(
    sample_sets: list[Array],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "angular",
) -> Float[Array, "sets sets"]:
    """Symmetric matrix of pairwise energy distances between samples.

    Batched, numerically-equivalent replacement for calling
    :func:`jcor.discrepancy.energy.energy_distance` on every ordered pair
    ``(i, j)``. Each within-sample term ``E[||X-X'||^α]`` is computed once per
    set rather than ``O(n)`` times, and the whole matrix is produced in a single
    compiled pass (``lax.map`` over rows, keeping peak memory to one
    ``(n_sets, m, m)`` block).

    Unequal row counts are admitted (t62). Clouds sharing one shape ``(m, d)``
    take the stacked equal-size core unchanged; a ragged list is zero-padded
    through :func:`jcor.sample.packed.pack` and reduced with masked means whose
    divisors are the products of the **true** counts, so every entry keeps the
    estimator weights of the corresponding scalar
    :func:`jcor.discrepancy.energy.energy_distance` call. Zero padding rows make
    angular/cosine grounds produce ``NaN`` outside the active block; the packed
    reduction is ``jnp.where``-based and therefore cannot propagate them.

    Still required, and still rejected with a reason: every cloud must be rank
    two and share the feature width ``d`` (there is no meaningful ground
    distance between clouds of different widths), the list must be nonempty and
    no cloud may have zero rows (the ``counts[i] * counts[j]`` divisor would be
    zero, so no estimator is defined).

    Args:
        sample_sets: List of ``n_sets`` arrays of shape ``(m_k, d)``; ``m_k``
            may vary across clouds, ``d`` may not. Mixed dtypes are promoted to
            one shared result dtype on the ragged path.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        Symmetric ``(n_sets, n_sets)`` matrix ``D`` with ``D[i, j]`` the energy
        distance between ``sample_sets[i]`` and ``sample_sets[j]`` and an
        exactly-zero diagonal.

    Raises:
        ValueError: If exponent is not in (0, 2), if the list is empty, if a
            cloud is not rank two or has zero rows, or if feature widths differ.
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    Examples:
        >>> a = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> b = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        >>> d = energy_distance_matrix([a, b], exponent=1.0, metric="euclidean")
        >>> d.shape
        (2, 2)
        >>> bool(d[0, 1] == d[1, 0]) and bool(d[0, 0] == 0.0)
        True

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    shared_shape = _shared_cloud_shape(sample_sets)
    if shared_shape is not None:
        _require_stackable_shape(shared_shape)
        a = jnp.stack([jnp.asarray(s) for s in sample_sets])
        return _energy_distance_matrix_core(
            a,
            validated_exponent.value,
            resolved_metric,
        )

    if not sample_sets:
        message = "energy_distance_matrix requires at least one cloud"
        raise ValueError(message)
    packed = pack(_normalized_clouds(sample_sets), require_positive_counts=True)
    return _energy_distance_matrix_packed_core(
        packed,
        validated_exponent.value,
        resolved_metric,
    )


@partial(jit, static_argnums=(1, 2))
def _mean_distance_matrix_core(
    a: Float[Array, "sets m d"],
    exponent: float,
    metric: GroundDistance[Law],
) -> Float[Array, "sets sets"]:
    """Batched full pairwise mean-distance matrix, including the diagonal.

    Args:
        a: Stacked samples of shape (n_sets, m, d).
        exponent: Distance exponent applied to the metric.
        metric: Declared ground-distance strategy.

    Returns:
        ``(n_sets, n_sets)`` matrix ``M`` with ``M[i, j] = mean(cdist(a[i],
        a[j], metric=metric) ** exponent)``, including the within-set
        diagonal terms ``M[i, i]``. Only the ``i <= j`` entries are computed
        directly; ``j > i`` entries are mirrored from ``i < j`` (author-approved
        contract relaxation 2026-07-18). Diagonal and upper-triangle entries
        are bit-identical to computing the same pair in isolation via
        :func:`jcor.discrepancy._mmd.components._compute_energy_components`;
        lower-triangle entries equal them only up to summation-order/XLA-tiling
        tolerance (~1e-8 float64 / ~1e-6 float32).

    """
    n = a.shape[0]
    ii, jj, n_pairs, pair_chunks = _triu_pair_chunks(n, 0)

    def _one_pair(pair: Array) -> Array:
        i, j = pair[0], pair[1]
        return jnp.mean(cdist(a[i], a[j], metric=metric) ** exponent)

    sums = lax.map(vmap(_one_pair), jnp.asarray(pair_chunks))
    sums = sums.reshape(-1)[:n_pairs]
    return jnp.zeros((n, n), dtype=sums.dtype).at[ii, jj].set(sums).at[jj, ii].set(sums)


def mean_distance_matrix(
    sample_sets: list[Float[Array, "m d"]],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "angular",
) -> Float[Array, "sets sets"]:
    """Full pairwise mean-distance matrix (including diagonal) over equal-size samples.

    Unlike :func:`energy_distance_matrix` (which combines terms into the
    energy-distance statistic and zeroes the diagonal), this returns the raw
    per-pair mean distance ``M[i, j] = mean(cdist(a[i], a[j]) ** exponent)``
    for every ``(i, j)`` including ``i == j``. Only the ``i <= j`` entries are
    computed directly; ``j > i`` entries are mirrored from ``i < j`` to skip the
    redundant lower-triangle matmul work (author-approved contract relaxation
    2026-07-18). Slicing a diagonal or upper-triangle entry out of ``M`` is
    bit-identical to calling
        :func:`jcor.discrepancy._mmd.components._compute_energy_components` on the
    corresponding leave-one-out target/candidate split; lower-triangle entries
    match only up to summation-order/XLA-tiling tolerance (~1e-8 float64 /
    ~1e-6 float32).

    Unlike :func:`energy_distance_matrix`, this producer stays **equal-shape by
    decision**: t62 widened only the energy statistic, and a ragged mean-distance
    matrix has no consumer yet, so unequal row counts remain rejected by the
    stacked path rather than silently padded.

    Args:
        sample_sets: List of ``n_sets`` arrays, each of shape ``(m, d)``.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        ``(n_sets, n_sets)`` matrix ``M``, see above.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    a = jnp.stack([jnp.asarray(s) for s in sample_sets])
    return _mean_distance_matrix_core(
        a,
        validated_exponent.value,
        resolved_metric,
    )


def energy_functional_matrix(
    sample_sets: list[Array],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "angular",
) -> DMat[EmpiricalDistribution, PremetricLaw, EnergyFunctional[VStatisticScheme]]:
    """Brand :func:`energy_distance_matrix` with its domain, law, and origin.

    The typed successor to the unbranded producer: same numbers, same ragged
    admission, and same masked count arithmetic, with the provenance the array
    form cannot carry. Only this form can enter
    :func:`jcor.core.matrices.metrize_energy`.

    What the three parameters claim, and what they deliberately do not:

    * ``EmpiricalDistribution`` — the compared objects are finitely-supported
      empirical measures. Never ``PopulationLaw``: a sample is not a law.
    * ``PremetricLaw`` — nonnegative and symmetric with a zero diagonal, and
      **not** separating. The energy functional separates laws only over a
      ground of *strong* negative type, which is a population hypothesis about
      the ground and travels as :func:`metrize_energy`'s ``ground`` argument,
      not as a law brand on the matrix. It is also unconditional in ``metric``:
      the default ``angular`` ground cannot see scale at all.
    * ``EnergyFunctional[VStatisticScheme]`` — the biased plug-in ``S``, not the
      population functional and not its root. This is the origin
      :func:`metrize_energy` demands, so a transport or checked-door matrix with
      identical law capabilities is a static error there.

    Args:
        sample_sets: Clouds admitted exactly as :func:`energy_distance_matrix`
            admits them; row counts may differ, feature widths may not.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        The branded energy functional matrix, ready for
        :func:`jcor.core.matrices.metrize_energy`.

    Raises:
        ValueError: Whatever :func:`energy_distance_matrix` raises.

    Examples:
        >>> a = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> b = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        >>> energy_functional_matrix([a, b], metric="euclidean").values.shape
        (2, 2)

    """
    branded: DMat[
        EmpiricalDistribution, PremetricLaw, EnergyFunctional[VStatisticScheme]
    ] = _dmat(energy_distance_matrix(sample_sets, exponent, metric))
    return branded


def mean_distance_estimate_matrix(
    sample_sets: list[Float[Array, "m d"]],  # jaxtyping shape
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "angular",
) -> PairwiseEstimateMatrix[
    EmpiricalDistribution,
    ExpectedGroundDistance,
    VStatisticEstimator,
    DirectOrigin,
]:
    """Brand :func:`mean_distance_matrix` as a realized estimate, not a distance.

    A ``PairwiseEstimateMatrix`` rather than a ``DMat``, and that is the whole
    content of the signature: ``M[i, i] = E[d(X, X')^α]`` is the within-cloud
    dispersion, which does not vanish, so no zero-diagonal, identity, or
    metrization law applies. Branding it a dissimilarity would let it flow into
    a consumer that assumes one.

    ``DirectOrigin``, not ``PoweredGroundOrigin``: the exponent is a runtime
    float that may be exactly one, so the wider origin is the claim this
    producer can actually support.

    Args:
        sample_sets: Equal-shape clouds, exactly as
            :func:`mean_distance_matrix` requires — t62 widened only the energy
            statistic, and this producer inherits that decision unchanged.
        exponent: Distance exponent α. Must be in (0, 2).
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        The branded pairwise mean-distance estimates, diagonal included.

    Raises:
        ValueError: Whatever :func:`mean_distance_matrix` raises.

    """
    branded: PairwiseEstimateMatrix[
        EmpiricalDistribution,
        ExpectedGroundDistance,
        VStatisticEstimator,
        DirectOrigin,
    ] = _pairwise_estimate(mean_distance_matrix(sample_sets, exponent, metric))
    return branded

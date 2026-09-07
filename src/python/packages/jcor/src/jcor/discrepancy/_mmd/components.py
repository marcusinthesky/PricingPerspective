"""Shared numerical primitives for energy-statistic reductions.

This internal boundary owns the packed upper-triangle traversal used by both
the public discrepancy layer (:mod:`jcor.discrepancy.energy`) and the kernel
layer. Keeping it below those layers avoids an import cycle while retaining
their historical private compatibility names, which
``jcor/_energy_components.py`` still re-exports through waves A-B.

Array-annotation convention (t46.5, published for all of wave B)
----------------------------------------------------------------
``Float[Array, "…"]`` throughout — jaxtyping on both sides, never
``NDArray[np.float64]``, which carries a dtype but no shape and so can never be
load-bearing under the t46.1 import hook.

**The NumPy half of this convention is retired (t65.4).** The index grids were
annotated ``Int[np.ndarray, "…"]`` because
:func:`jcor.sample.packed.pair_chunks` returned NumPy, and the two spellings are
mutually exclusive at runtime — measured: ``Float[Array, …]`` rejects an
``np.ndarray`` and ``Float[np.ndarray, …]`` rejects a ``jax.Array``. That forced
``import numpy`` to sit at runtime rather than under ``if TYPE_CHECKING:``,
since beartype silently *skips* an annotation whose names it cannot resolve and
a guarded import would have made the shape string inert (t46.1 Finding 1). Now
that ``pair_chunks`` returns ``jnp``, the grids are ``Int[Array, …]`` like
everything else and the import is gone — which is why this module had to be the
last one converted.

The cost is one ``F722`` per-file-ignore in ``packages/jcor/pyproject.toml``
per module — ruff parses a space-delimited shape string as a forward reference.
That is a shared file, so t46.5 landed the entries for every wave-B module the
convention touches, including t46.9's two.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax, vmap

from jcor.core.axioms import Law  # noqa: TC001  # runtime annotations
from jcor.core.typing import Array, Float, Int  # noqa: TC001  # runtime; see docstring
from jcor.discrepancy._mmd.scalar import _shared_reference
from jcor.ground.metrics import (  # noqa: TC001  # runtime annotations
    EUCLIDEAN,
    GroundDistance,
    cdist,
)
from jcor.ground.similarities import gram
from jcor.sample.packed import DEFAULT_PAIR_CHUNK, masked_mean, pack
from jcor.sample.packed import pair_chunks as _packed_pair_chunks

if TYPE_CHECKING:
    from jcor.ground._similarity_strategy import (
        PositiveSemidefiniteKernelLaw,
        SimilarityStrategy,
    )
    from jcor.sample.packed import PackedClouds

_PAIR_CHUNK = DEFAULT_PAIR_CHUNK


def _triu_pair_chunks(
    n: int, k: int
) -> tuple[
    Int[Array, " pairs"],
    Int[Array, " pairs"],
    int,
    Int[Array, "chunks chunk 2"],
]:
    """Return padded chunks of upper-triangle index pairs.

    Args:
        n: Matrix side length.
        k: Diagonal offset passed to :func:`numpy.triu_indices`.

    Returns:
        Unpadded row and column indices, their count, and the padded pair grid.

    """
    return _packed_pair_chunks(n, k=k, chunk=_PAIR_CHUNK)


@partial(jax.jit, static_argnums=(2, 3))
def _compute_energy_components_batched(
    target: Float[Array, "n d"],
    stacked_candidates: Float[Array, "k m d"],
    metric: GroundDistance[Law],
    exponent: float,
) -> tuple[Float[Array, "k k"], Float[Array, " k"]]:
    """Compute candidate Gram and target-cross terms for equal-size clouds."""
    candidate_count = stacked_candidates.shape[0]
    row_indices, column_indices, pair_count, pair_grid = _triu_pair_chunks(
        candidate_count, 0
    )

    # Inner closures take vmap/lax.map tracers; they carry the array *type* but
    # no shape string, because the batching rule rewrites the rank the hook sees.
    def pair_mean(pair: Array) -> Array:
        row_index, column_index = pair[0], pair[1]
        distances = cdist(
            stacked_candidates[row_index],
            stacked_candidates[column_index],
            metric=metric,
        )
        return jnp.mean(distances**exponent)

    pair_means = lax.map(vmap(pair_mean), jnp.asarray(pair_grid)).reshape(-1)[
        :pair_count
    ]
    gram = (
        jnp.zeros((candidate_count, candidate_count), dtype=pair_means.dtype)
        .at[row_indices, column_indices]
        .set(pair_means)
        .at[column_indices, row_indices]
        .set(pair_means)
    )
    target_cross = vmap(
        lambda candidate: jnp.mean(cdist(target, candidate, metric=metric) ** exponent)
    )(stacked_candidates)
    return gram, target_cross


@partial(jax.jit, static_argnums=(2, 3))
def _compute_energy_components_packed(
    target: Float[Array, "n d"],
    packed_candidates: PackedClouds,
    metric: GroundDistance[Law],
    exponent: float,
) -> tuple[Float[Array, "k k"], Float[Array, " k"]]:
    """Compute energy components for unequal-size, padded candidate clouds."""
    candidate_data = packed_candidates.data
    candidate_sizes = packed_candidates.counts
    candidate_count = candidate_data.shape[0]
    row_indices, column_indices, pair_count, pair_grid = _triu_pair_chunks(
        candidate_count, 0
    )

    def pair_mean(pair: Array) -> Array:
        row_index, column_index = pair[0], pair[1]
        distances = (
            cdist(
                candidate_data[row_index],
                candidate_data[column_index],
                metric=metric,
            )
            ** exponent
        )
        return masked_mean(
            distances, candidate_sizes[row_index], candidate_sizes[column_index]
        )

    pair_means = lax.map(vmap(pair_mean), jnp.asarray(pair_grid)).reshape(-1)[
        :pair_count
    ]
    gram = (
        jnp.zeros((candidate_count, candidate_count), dtype=pair_means.dtype)
        .at[row_indices, column_indices]
        .set(pair_means)
        .at[column_indices, row_indices]
        .set(pair_means)
    )
    target_size = jnp.asarray(target.shape[0])

    def target_mean(candidate: Array, candidate_size: Array) -> Array:
        distances = cdist(target, candidate, metric=metric) ** exponent
        return masked_mean(distances, target_size, candidate_size)

    target_cross = vmap(target_mean)(candidate_data, candidate_sizes)
    return gram, target_cross


def _compute_energy_components(
    target: Float[Array, "n d"],
    candidates: list[Array],
    metric: GroundDistance[Law] = EUCLIDEAN,
    exponent: float = 1.0,
) -> tuple[Float[Array, "k k"], Float[Array, " k"]]:
    """Compute candidate Gram and target-cross mean-distance components.

    Equal-size candidates use the stacked JAX path. Unequal-size candidates
    use a masked packed representation without changing estimator weights.
    """
    candidate_shapes = {tuple(candidate.shape) for candidate in candidates}
    if len(candidate_shapes) == 1:
        return _compute_energy_components_batched(
            target, jnp.stack(candidates), metric, exponent
        )
    return _compute_energy_components_packed(
        target, pack(list(candidates)), metric, exponent
    )


def _compute_mmd_components(
    target: Float[Array, "n d"],
    candidates: list[Float[Array, "m d"]],
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: Float[Array, " d"] | None = None,
) -> tuple[
    Float[Array, "k k"],
    Float[Array, " k"],
    Float[Array, ""],
]:
    """Compute kernel Gram means for a target-directed MMD barycentre.

    The returned values are the candidate--candidate Gram means ``G``, the
    target--candidate Gram means ``c``, and the target self mean ``b``. The
    same reference is passed through every block so distance-induced kernels
    retain the MMD cancellation convention.
    """
    shared_reference = _shared_reference(target, reference)
    candidate_gram = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.mean(
                        gram(
                            left,
                            right,
                            kernel,
                            reference=shared_reference,
                        )
                    )
                    for right in candidates
                ]
            )
            for left in candidates
        ]
    )
    candidate_gram = (candidate_gram + candidate_gram.T) / 2.0
    target_cross = jnp.stack(
        [
            jnp.mean(
                gram(
                    target,
                    candidate,
                    kernel,
                    reference=shared_reference,
                )
            )
            for candidate in candidates
        ]
    )
    target_self = jnp.mean(gram(target, target, kernel, reference=shared_reference))
    return candidate_gram, target_cross, target_self

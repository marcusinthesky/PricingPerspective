"""Grouped block reduction shared by every replicate-driven resampling loop.

Three loops in this package re-index a pooled distance matrix once per
replicate — ``distances[perm][:, perm]`` in
:func:`jcor.inference.permutation.energy_permutation_test_result` and
:func:`jcor.inference._permutation.split_runner`'s null driver, and
``distances[idx][:, idx]`` in
:func:`jcor.inference.bootstrap._bootstrap_energy_ci_result`. Each then reduces
that gathered ``(N, N)`` view to group blocks ``B = Mᵀ D M``.

The gather is unnecessary. Every quantity those loops consume is a **bilinear
form in the group indicator**, so the replicate can be pushed into the
membership matrix instead of the distance matrix. Writing ``C`` for the
``(groups, pooled)`` matrix that counts how many replicate slots in group ``g``
were drawn from pooled index ``k``::

    B = C D Cᵀ            block sums, (groups, groups)
    S = D Cᵀ              per-pooled-point sums to each group, (pooled, groups)

For a permutation ``C`` is a 0/1 indicator; for a with-replacement bootstrap it
holds multiplicities. Both are produced by :func:`replicate_counts`, and the
identity holds in either case because

    (Mᵀ D[idx][:, idx] M)[a, b] = Σ_{i,j} M[i,a] M[j,b] D[idx[i], idx[j]]
                                = Σ_{k,l} C[a,k] C[b,l] D[k,l]

with ``C[a,k] = Σ_i M[i,a]·1{idx[i] = k}``, which is exactly what the scatter-add
in :func:`replicate_counts` accumulates. The rewrite is therefore *algebraic*,
not an approximation: at float64 it reproduces the gather form to ~1e-15
relative, the ordinary summation-order floor (the same class of drift the
``B = Mᵀ D M`` reduction identity in ``tests/migration/_parity.py`` is exempt
from only because it uses exactly-representable integer entries).

Why it is faster: the gather form runs ``P`` sequential ``(N, N)`` gathers, each
memory-bound with essentially no arithmetic intensity. The reduction form is one
batched ``dot_general`` over the whole replicate chunk, which XLA lowers to a
single GEMM. Measured end-to-end through
``energy_permutation_test_result`` (CPU, f32, N pooled points, P replicates):
10.1x at N=100/P=999, 8.5x at N=400/P=999, 4.3x at N=200/P=9999, with p-values
unchanged in every case.

Memory: the reduction form's working set is the replicate counts, ``O(chunk ·
groups · pooled)``, against the gather form's ``O(pooled²)``. That trade is only
safe because the replicate axis stays chunked — hence
:func:`map_replicates`, which keeps the ``lax.map`` (scan) memory bound the
original loops were deliberately written for, while still handing XLA a
``batch_size``-wide slab to vectorise. Do not replace it with a bare ``vmap``:
that reinstates the peak-memory blowup those loops avoid.

einsum policy (see :mod:`jcor.sample.packed`): the two reductions here are
two-operand matmuls and stay as ``@``, not ``einsum``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike  # noqa: TC002  # runtime jaxtyping contract

from jcor.core.typing import Array, Float, Int  # noqa: TC001  # runtime annotations

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_REPLICATE_CHUNK",
    "grouped_block_reduction",
    "map_replicates",
    "replicate_counts",
]

#: Replicates per :func:`jax.lax.map` step. Mirrors the role of
#: ``jcor.sample.packed.DEFAULT_PAIR_CHUNK``: large enough that XLA sees a
#: worthwhile GEMM, small enough that the ``(chunk, groups, pooled)`` counts
#: buffer stays far below the ``(pooled, pooled)`` distance matrix it replaces.
#: Callers may override via the ``chunk`` argument.
DEFAULT_REPLICATE_CHUNK = 64


def replicate_counts(
    indices: Int[Array, "slots"],  # noqa: F821, UP037  # jaxtyping shape
    group_ids: Int[Array, "slots"],  # noqa: F821, UP037  # jaxtyping shape
    n_groups: int,
    n_pooled: int,
    dtype: DTypeLike,
) -> Float[Array, "groups pooled"]:  # noqa: F722  # jaxtyping shape
    """Count replicate slots per (group, pooled index) pair.

    ``indices[i]`` is the pooled row that replicate slot ``i`` draws, and
    ``group_ids[i]`` is the group that slot occupies. The result is the ``C``
    of this module's docstring: a 0/1 indicator when ``indices`` is a
    permutation, a multiplicity matrix when it is a with-replacement draw.

    Args:
        indices: Pooled index drawn by each replicate slot.
        group_ids: Group label of each replicate slot, in the same slot order.
        n_groups: Static number of groups (``K+1`` in the mixture-test sense).
        n_pooled: Static number of pooled points.
        dtype: Result dtype; pass the distance matrix's dtype so the downstream
            matmuls need no promotion.

    Returns:
        The ``(n_groups, n_pooled)`` count matrix ``C``.

    """
    return jnp.zeros((n_groups, n_pooled), dtype).at[group_ids, indices].add(1.0)


def grouped_block_reduction(
    counts: Float[Array, "groups pooled"],  # noqa: F722  # jaxtyping shape
    distances: Float[Array, "pooled pooled"],  # noqa: F722  # jaxtyping shape
) -> tuple[
    Float[Array, "groups groups"],  # noqa: F722  # jaxtyping shape
    Float[Array, "pooled groups"],  # noqa: F722  # jaxtyping shape
]:
    """Reduce a pooled distance matrix to group blocks without gathering it.

    Computes ``S = D Cᵀ`` once and contracts it a second time for
    ``B = C S = C D Cᵀ``, so the shared ``(pooled, groups)`` intermediate is
    paid for only once. Callers that need only ``B`` may discard ``S``; the
    studentized jackknife in :mod:`jcor.inference.studentized` needs both.

    Args:
        counts: Replicate counts ``C`` from :func:`replicate_counts`.
        distances: Pooled, pre-exponentiated ``(N, N)`` distance matrix.

    Returns:
        Tuple ``(block_sums, pooled_group_sums)`` where ``block_sums[a, b] =
        Σ_{i∈a, j∈b} d(i, j)`` over the replicate's groups and
        ``pooled_group_sums[k, g]`` is the total distance from pooled point
        ``k`` to every replicate slot in group ``g``.

    """
    pooled_group_sums = distances @ counts.T
    return counts @ pooled_group_sums, pooled_group_sums


def map_replicates[T](
    fn: Callable[[Array], T],
    xs: Array,
    chunk: int = DEFAULT_REPLICATE_CHUNK,
) -> T:
    """Drive a replicate loop in memory-bounded, vectorised chunks.

    ``lax.map``'s ``batch_size`` splits the leading axis into ``chunk``-wide
    slabs, ``vmap``s ``fn`` within a slab, and scans across slabs. That is the
    native form of the chunked-vmap idiom these loops want: XLA gets a batched
    ``dot_general`` to fuse, and peak memory stays proportional to ``chunk``
    rather than to the full replicate count.

    Args:
        fn: Per-replicate function, applied to one slice of ``xs``.
        xs: Replicate driver array, replicates along the leading axis.
        chunk: Replicates per step.

    Returns:
        ``fn`` mapped over the leading axis of ``xs``.

    """
    return lax.map(fn, xs, batch_size=chunk)

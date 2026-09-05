# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""S4 — cross-model nearest-neighbour stability diagnostics.

Position
--------
rank 4 · ``geometry``

Consumes
--------
one baseline distance matrix and one or more comparator distance matrices over
the same objects, identified by caller-assigned integer ids (waist 1)

Produces
--------
per-comparator aggregate persistence summaries and the complete directed
anchor-neighbour edge list they are reconstructible from, keyed by matrix row
index

Entity neutrality — what t48.3 changed
--------------------------------------
Until t48.3 the entry point took ``Sequence[str]`` tickers and broke distance
ties with ``np.lexsort`` over ``np.asarray(ticker_list, dtype=object)``. The
object dtype was the symptom; the cause was that jcor knew what a ticker was.
The tie-break itself is a legitimate determinism device and survives — it now
sorts on the caller's **integer object id**, which is exactly as total an order
and needs no object dtype or string handling. Label mapping moved to
``pipeline/_kernels/neighbour_stability.py``.

**The caller owns the ordering, and it is now the caller's determinism.** The
ids are the tie-break, so ids assigned by iteration over a ``dict`` or a DuckDB
result make the output non-reproducible even though nothing here is random.
Assign them from an explicitly pinned rule — sorted labels → rank is what the
pipeline adapter does, and it reproduces the retired string tie-break exactly,
because rank in the sorted label list induces the same order as the labels.

Axioms required of the input
-----------------------------
Only :data:`NEIGHBOUR_INPUT_AXIOMS` — ``SYMMETRY``. Neighbour
ranking is invariant to any strictly increasing transform of the distances, so
``TRIANGLE`` is never used; ``IDENTITY`` is never used either, because the
anchor is excluded from its own candidate set and ties are broken
deterministically by object id. ``NONNEGATIVE`` is not required: translating a
symmetric dissimilarity matrix by a constant preserves every row ordering. The
eager boundary validates symmetry so the declared precondition is enforced
rather than merely documented. That makes this module safe on the
dissimilarities t46.1 measured as scale-blind (``cosine_distance``,
``angular_distance``, ``energy_distance`` over an angular ground metric), which
is precisely why the requirement is declared in flags rather than inherited
from a marker name. The integer-id re-sign changes what the caller passes, not
what the *matrix* must satisfy, so the constant is unchanged by t48.3.

Why the input is not a ``DMat``
-------------------------------
:class:`jcor.core.typing.DMat` wraps ``Float[Array, "*batch n n"]``, a branded
carrier. The callers here hold raw matrices produced by MDS and by parquet
reads, so the entry annotation stays ``ArrayLike`` — narrowing it to ``DMat``
would change the public signature and reject every current caller at the
jaxtyping/beartype import hook. The axiom requirement therefore travels as the
module constant below — which is what :mod:`jcor.core.axioms` mandates anyway
("preconditions declare flags, never alias names"). A caller that *does* hold a
branded matrix passes ``matrix.values``.

Note the asymmetry, which is deliberate: ``ArrayLike`` **in** so NumPy-holding
callers need no boxing, and a shaped ``Int[Array, "n n-1"]`` **out** so the
rank and the anchor exclusion are contracts rather than prose.

Why the ordering core is JAX now — t65, measured then reprioritised
-------------------------------------------------------------------
The ordering core *is* jnp-convertible: ``jnp.lexsort`` exists on the pinned jax
0.10.2, the anchor exclusion is expressible as a leading sort key plus a static
slice (no data-dependent shape), and the JAX path reproduces the NumPy
permutation exactly under ``jax.enable_x64``. t48.3 measured it 4-10× slower at
every size (520 µs vs 58 µs at n=52, 4.2-10.3× at n=256/1024/2048 on the CPU
backend) and declined it per the t48 binding to benchmark before shipping.
t65 re-measured with a warmed JIT and ``.block_until_ready()`` (n=52: 4.8×,
n=256: 5.4×, n=1024: 4.8×, n=2048: 3.0×) and **accepted that regression** to
advance the JAX conversion; the follow-on optimisation (control-flow
restructuring, ``jit``/``vmap``, per-shape JIT-cache reuse) is deferred to a
later task. The ``backend="numpy"`` escape hatch that carried the old path was
retired once the conversion had settled (t65.4); the bit-identical NumPy
permutation now lives only in the test module, as an independent oracle rather
than as a shipped second backend.

Why the guards stay eager
-------------------------
Only the raise-semantics validation guards run on the host now:
``_validate_object_ids``'s uniqueness check and ``_validate_distance_matrix``'s
finiteness/symmetry checks exist to ``raise`` on malformed input. Raise-semantics
are not a jittable kernel — a traced actor would either trace a ``raise`` or
silently turn it into a padded result — so they stay eager by design (t65 D1/P4)
and cross the device boundary through ``bool()``. They are ``jnp`` rather than
``np`` since t65.4, which retired the module's last NumPy import; that changed
the library, not the boundary. The results still leave through a Python record
list consumed by pandas, so this is a host door either way.

What t48.3 *did* remove is the per-anchor Python loop: one vectorised
``lexsort`` over all anchors replaces ``n`` calls, measured 419 µs → 58 µs at
n=52 (7.2×) with a bit-identical permutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

import jax.numpy as jnp

from jcor.core.axioms import Axioms
from jcor.core.typing import Array, ArrayLike, Float, Int  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "NEIGHBOUR_INPUT_AXIOMS",
    "FloatMatrix",
    "NeighbourStabilityEdge",
    "NeighbourStabilitySummary",
    "cross_encoder_neighbour_stability",
    "neighbour_order",
    "neighbour_order_kernel",
]

type FloatMatrix = Float[Array, "n n"]

#: One row per anchor listing every *other* object, so the trailing column of
#: the ``(n, n)`` lexsort — the anchor itself, pushed last by the indicator key
#: — is sliced off. Spelled symbolically rather than as two free axes because
#: the ``n - 1`` is the whole point: a return that kept the anchor would be a
#: silent contract break, and t65 shipped one (`Int[Array, " n"]`, a rank-1
#: annotation on a rank-2 return) that beartype caught only at test time.
type NeighbourRanking = Int[Array, "n n-1"]

#: Axioms :func:`cross_encoder_neighbour_stability` requires of every distance
#: matrix it is handed. Deliberately weak — see the module docstring.
NEIGHBOUR_INPUT_AXIOMS: Final = Axioms.SYMMETRY


class NeighbourStabilitySummary(TypedDict):
    """One comparator's aggregate neighbour-persistence diagnostics.

    ``comparator_index`` indexes ``comparison_distances`` as the caller ordered
    it; naming the comparator is the caller's business.
    """

    comparator_index: int
    n_objects: int
    top_k: int
    mean_top_k_overlap: float
    exact_nearest_agreement: int
    exact_nearest_agreement_share: float


class NeighbourStabilityEdge(TypedDict):
    """One directed anchor-neighbour comparison with complete ranks.

    ``anchor_index`` and ``neighbour_index`` are **row indices** into the
    distance matrices, not the caller's ids; the caller already holds the
    row-ordered labels it built the ids from.
    """

    comparator_index: int
    anchor_index: int
    neighbour_index: int
    baseline_distance: float
    comparator_distance: float
    baseline_rank: int
    comparator_rank: int
    in_baseline_top_k: bool
    in_comparator_top_k: bool


def _validate_object_ids(
    object_ids: Sequence[int] | ArrayLike,
) -> Int[Array, " n"]:
    """Return the caller's ids as a unique 1-D int64 array.

    Args:
        object_ids: One integer id per matrix row, in row order.

    Returns:
        The ids as ``int64``.

    Raises:
        ValueError: If the ids are not one-dimensional, not integral, empty, or
            not unique — uniqueness is what makes the tie-break a total order.

    """
    ids = jnp.asarray(object_ids)
    if ids.ndim != 1 or ids.size == 0:
        message = f"object_ids must be a non-empty 1-D sequence; got shape {ids.shape}"
        raise ValueError(message)
    if not jnp.issubdtype(ids.dtype, jnp.integer):
        message = f"object_ids must be integral; got dtype {ids.dtype}"
        raise ValueError(message)
    ids = ids.astype(jnp.int32)
    # Deliberate *eager* guard (t65 D1/P4): this `unique` exists to `raise` on a
    # duplicate id, and raise-semantics are not a jittable kernel. `jnp.unique`
    # needs a static `size=` under trace precisely because its output shape is
    # data-dependent; leaving it eager is what keeps the `raise` a `raise`
    # instead of a padded result. Being jnp rather than np changes the library,
    # not the boundary — this function still runs on the host.
    if jnp.unique(ids).size != ids.size:
        message = "object ids must be unique"
        raise ValueError(message)
    return ids


def _validate_distance_matrix(
    label: str,
    distances: ArrayLike,
    n_objects: int,
) -> FloatMatrix:
    """Return a finite symmetric matrix of the required size.

    The caller's width is preserved rather than promoted: under t56.7 precision
    is caller-owned, so a float32 input stays float32 and it is the caller's
    ``jax.enable_x64`` scope that decides otherwise.

    Args:
        label: Positional name of the matrix, used in error messages.
        distances: The candidate distance matrix.
        n_objects: The required side length.

    Returns:
        The same matrix, at the caller's width.

    Raises:
        ValueError: If the matrix is not ``(n_objects, n_objects)``, contains
            a non-finite entry, or is not symmetric.

    """
    matrix = jnp.asarray(distances)
    if matrix.shape != (n_objects, n_objects):
        message = (
            f"{label} distance matrix has shape {matrix.shape}; "
            f"expected {(n_objects, n_objects)}"
        )
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        message = f"{label} distance matrix contains non-finite values"
        raise ValueError(message)
    if not bool(jnp.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)):
        message = f"{label} distance matrix must satisfy symmetry"
        raise ValueError(message)
    return matrix


def neighbour_order_kernel(
    object_ids: ArrayLike,
    distances: ArrayLike,
) -> NeighbourRanking:
    """JAX ``lexsort`` core: rank every row's neighbours on the JAX graph.

    The transformable twin of :func:`neighbour_order` (t65 P4). It reproduces
    the NumPy permutation exactly — anchor pushed last by a leading key, then
    distance, then object id — and is ``jit``/``vmap``-compatible, always
    returning ``(n, n - 1)``. It is the path :func:`neighbour_order` takes: the
    measured 4.2-10.3× CPU regression against the retired NumPy implementation
    (see the module docstring) was accepted by t65, and the optimisation that
    would close it is deferred rather than hedged behind a second backend.

    Args:
        object_ids: One unique integer id per matrix row, in row order.
        distances: The ``(n, n)`` distance matrix over those objects.

    Returns:
        The ``(n, n - 1)`` JAX integer array of neighbours per anchor.

    """
    ids = jnp.asarray(object_ids)
    matrix = jnp.asarray(distances)
    n_objects = ids.shape[0]
    is_anchor = jnp.eye(n_objects, dtype=matrix.dtype)
    id_key = jnp.broadcast_to(ids, matrix.shape)
    # lexsort applies the LAST key first: anchor-last, then distance, then id.
    ordered = jnp.lexsort((id_key, matrix, is_anchor), axis=-1)
    return jnp.asarray(ordered[:, : n_objects - 1], dtype=jnp.int64)


def neighbour_order(
    object_ids: Sequence[int] | ArrayLike,
    distances: ArrayLike,
) -> NeighbourRanking:
    """Rank every non-anchor object by distance, breaking ties by object id.

    One vectorised ``lexsort`` over all anchors: the anchor is pushed to the
    end of its own row by a leading indicator key rather than removed by a
    boolean mask, so every shape is static and the trailing column can be
    sliced off. That is what makes the core jnp-shaped.

    The ordering runs on :func:`neighbour_order_kernel` (t65) and is
    bit-identical to the NumPy permutation t48.3 shipped; t65 accepted the
    4-10× regression to advance the JAX conversion, and t65.4 retired the
    ``backend="numpy"`` escape hatch once no caller used it. Only the
    raise-semantics guards below remain on NumPy.

    Args:
        object_ids: One unique integer id per matrix row, in row order.
        distances: The ``(n, n)`` distance matrix over those objects.

    Returns:
        An ``(n, n - 1)`` array whose row ``a`` lists every other row index
        ascending in distance from ``a``, ties broken by ascending object id.

    Raises:
        ValueError: If the ids are invalid or the matrix does not match them.

    """
    ids = _validate_object_ids(object_ids)
    n_objects = ids.size
    matrix = _validate_distance_matrix("input", distances, n_objects)
    return neighbour_order_kernel(ids, matrix)


def _cross_encoder_comparator(
    *,
    comparator_index: int,
    baseline: FloatMatrix,
    baseline_orders: NeighbourRanking,
    comparator: FloatMatrix,
    comparator_orders: NeighbourRanking,
    n_objects: int,
    top_k: int,
) -> tuple[NeighbourStabilitySummary, list[NeighbourStabilityEdge]]:
    """Build one comparator's aggregate and reconstructible edge rows."""
    overlap_shares: list[float] = []
    exact_nearest = 0
    edge_rows: list[NeighbourStabilityEdge] = []
    for anchor_index in range(n_objects):
        baseline_order = baseline_orders[anchor_index]
        comparator_order = comparator_orders[anchor_index]
        baseline_top = set(baseline_order[:top_k].tolist())
        comparator_top = set(comparator_order[:top_k].tolist())
        overlap_shares.append(len(baseline_top & comparator_top) / top_k)
        exact_nearest += int(baseline_order[0] == comparator_order[0])
        baseline_rank = {
            int(index): rank + 1 for rank, index in enumerate(baseline_order)
        }
        comparator_rank = {
            int(index): rank + 1 for rank, index in enumerate(comparator_order)
        }
        for neighbour in baseline_order:
            neighbour_index = int(neighbour)
            edge_rows.append(
                {
                    "comparator_index": comparator_index,
                    "anchor_index": anchor_index,
                    "neighbour_index": neighbour_index,
                    "baseline_distance": float(baseline[anchor_index, neighbour_index]),
                    "comparator_distance": float(
                        comparator[anchor_index, neighbour_index]
                    ),
                    "baseline_rank": baseline_rank[neighbour_index],
                    "comparator_rank": comparator_rank[neighbour_index],
                    "in_baseline_top_k": neighbour_index in baseline_top,
                    "in_comparator_top_k": neighbour_index in comparator_top,
                }
            )
    return {
        "comparator_index": comparator_index,
        "n_objects": n_objects,
        "top_k": top_k,
        "mean_top_k_overlap": float(jnp.mean(jnp.asarray(overlap_shares))),
        "exact_nearest_agreement": exact_nearest,
        "exact_nearest_agreement_share": exact_nearest / n_objects,
    }, edge_rows


def cross_encoder_neighbour_stability(
    *,
    object_ids: Sequence[int] | ArrayLike,
    baseline_distances: ArrayLike,
    comparison_distances: Sequence[ArrayLike],
    top_k: int = 5,
) -> tuple[list[NeighbourStabilitySummary], list[NeighbourStabilityEdge]]:
    """Compare directed original-space neighbour rankings across models.

    The summary reports, for each comparator, the mean anchor-level overlap
    between its top-``k`` set and the baseline set plus exact-nearest
    agreement. The edge record retains complete ranks and distances so the
    aggregate can be reconstructed without refitting embeddings or MDS.

    Every distance matrix must satisfy :data:`NEIGHBOUR_INPUT_AXIOMS`; nothing
    stronger is used.

    Args:
        object_ids: One unique integer id per matrix row, in row order. Ties in
            distance break on this id, so the caller owns reproducibility — see
            the module docstring.
        baseline_distances: The reference ``(n, n)`` distance matrix.
        comparison_distances: Comparator matrices, in the caller's order; the
            summaries and edges carry that position as ``comparator_index``.
        top_k: Neighbourhood size, in ``[1, n - 1]``.

    Returns:
        The per-comparator summaries and the complete directed edge list, both
        ordered by comparator position, then by anchor row, then by baseline
        rank.

    Raises:
        ValueError: If the ids are not unique, ``top_k`` is out of range, or no
            comparator matrix was supplied.

    """
    ids = _validate_object_ids(object_ids)
    n_objects = ids.size
    if not 0 < top_k < n_objects:
        message = f"top_k must lie in [1, {n_objects - 1}], got {top_k}"
        raise ValueError(message)
    comparators = list(comparison_distances)
    if not comparators:
        message = "at least one comparison matrix is required"
        raise ValueError(message)

    baseline = _validate_distance_matrix("baseline", baseline_distances, n_objects)
    baseline_orders = neighbour_order(ids, baseline)

    summaries: list[NeighbourStabilitySummary] = []
    edge_rows: list[NeighbourStabilityEdge] = []
    for comparator_index, comparator_distances in enumerate(comparators):
        comparator = _validate_distance_matrix(
            f"comparator {comparator_index}",
            comparator_distances,
            n_objects,
        )
        comparator_orders = neighbour_order(ids, comparator)
        summary, comparator_edges = _cross_encoder_comparator(
            comparator_index=comparator_index,
            baseline=baseline,
            baseline_orders=baseline_orders,
            comparator=comparator,
            comparator_orders=comparator_orders,
            n_objects=n_objects,
            top_k=top_k,
        )
        summaries.append(summary)
        edge_rows.extend(comparator_edges)
    return summaries, edge_rows

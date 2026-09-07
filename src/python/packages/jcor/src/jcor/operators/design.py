"""Rank-clean regression-design operators for undirected dyadic panels.

These functions construct or residualize a design matrix.  They deliberately
know nothing about a fitted model, bootstrap schedule, paper-specific regressor
names, or artifact format; those concerns begin at :mod:`jcor.model.dyadic`.
"""

from __future__ import annotations

import operator
from collections.abc import Hashable, Iterator  # noqa: TC003  # runtime Protocol
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # noqa: TC002  # runtime jaxtyping contract

__all__ = [
    "ArrayView",
    "NodeIndex",
    "drop_collinear_columns",
    "fwl_coefficient",
    "symmetric_dyadic_effects",
]

_MATRIX_DIMENSIONS = 2
_MIN_ENDPOINTS = 2


@runtime_checkable
class ArrayView(Protocol):
    """Structural view of an aligned dyadic column (NumPy or JAX).

    Deliberately narrower than ``jax.typing.ArrayLike``, which admits bare
    scalars and so erases the shape and indexing guarantees every reader
    depends on. Deliberately wider than ``jax.Array``, because ``apps/pipeline``
    supplies NumPy panels read off parquet, and because endpoint labels are
    arbitrary hashables (strings, ints, mixed ``object`` columns) that never
    enter a JAX tensor. Only the container surface is constrained here, never
    the dtype.
    """

    # `Any` is load-bearing on the element-typed members: NumPy and JAX
    # disagree on their index and element types, and an endpoint column's
    # elements are arbitrary hashables. Constraining them here would reject
    # one backend or the other.

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the column's dimensions."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        ...

    def __getitem__(self, key: Any, /) -> Any:  # noqa: ANN401
        """Return one element or slice."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the leading axis."""
        ...

    def __len__(self) -> int:
        """Return the length of the leading axis."""
        ...

    def tolist(self) -> Any:  # noqa: ANN401
        """Return the contents as nested host lists."""
        ...


def _as_id_sequence(value: ArrayView) -> list[Any]:
    """Return an endpoint label array as a flat host list."""
    return list(value.tolist())


def _node_sort_key(node: Hashable) -> tuple[str, str]:
    """Order unlike ID types; cross-process stability follows their ``repr``."""
    node_type = type(node)
    return f"{node_type.__module__}.{node_type.__qualname__}", repr(node)


def _validate_node_id[NodeIdT: Hashable](node: NodeIdT) -> NodeIdT:
    """Return one usable entity ID or reject missing/non-hashable values."""
    if node is None:
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message)
    try:
        hash(node)
    except (TypeError, ValueError) as error:
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message) from error
    try:
        reflexive = operator.eq(node, node)
    except (TypeError, ValueError) as error:
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message) from error
    if not isinstance(reflexive, bool) or not bool(reflexive):
        message = "dyadic endpoint IDs must be non-missing hashable scalar values"
        raise ValueError(message)
    return node


@dataclass(frozen=True)
class NodeIndex[NodeIdT: Hashable = Hashable]:
    """Canonical entity IDs and their shared dyadic-design column mapping.

    Canonical ordering uses an ID's qualified type and representation, rather
    than requiring IDs of unrelated types to implement a common ``<`` order.
    It is cross-process deterministic for primitive IDs and custom IDs whose
    qualified type name and ``repr`` are themselves stable; arbitrary custom
    representations can only promise process-local ordering. The representation
    key must distinguish unequal IDs, so ambiguous keys fail closed instead of
    making endpoint orientation choose the omitted effect column.
    """

    node_ids: tuple[NodeIdT, ...]

    def __post_init__(self) -> None:
        """Validate, deduplicate, and canonically order entity identifiers."""
        validated = tuple(_validate_node_id(node) for node in self.node_ids)
        try:
            unique = tuple(dict.fromkeys(validated))
        except (TypeError, ValueError) as error:
            message = "dyadic endpoint IDs must have scalar equality semantics"
            raise ValueError(message) from error
        if len(unique) != len(validated):
            message = "node index IDs must be unique"
            raise ValueError(message)
        if len(unique) < _MIN_ENDPOINTS:
            message = "dyadic data require at least two distinct endpoint IDs"
            raise ValueError(message)
        keyed = [(_node_sort_key(node), node) for node in unique]
        if len({key for key, _node in keyed}) != len(keyed):
            message = (
                "unequal endpoint IDs must have distinct canonical representations"
            )
            raise ValueError(message)
        canonical = tuple(
            node for _key, node in sorted(keyed, key=lambda item: item[0])
        )
        object.__setattr__(self, "node_ids", canonical)

    def __len__(self) -> int:
        """Return the number of indexed entities."""
        return len(self.node_ids)

    @classmethod
    def from_endpoints(
        cls,
        endpoint_i: ArrayView,
        endpoint_j: ArrayView,
    ) -> NodeIndex[Hashable]:
        """Build the canonical index for one aligned distinct-node dyadic panel."""
        diagonal_i = endpoint_i
        diagonal_j = endpoint_j
        if (
            diagonal_i.ndim != 1
            or diagonal_j.ndim != 1
            or diagonal_i.shape != diagonal_j.shape
        ):
            message = "dyadic endpoint arrays must be aligned one-dimensional vectors"
            raise ValueError(message)
        values = [*_as_id_sequence(endpoint_i), *_as_id_sequence(endpoint_j)]
        validated = [_validate_node_id(value) for value in values]
        try:
            unique = tuple(dict.fromkeys(validated))
        except (TypeError, ValueError) as error:
            message = "dyadic endpoint IDs must have scalar equality semantics"
            raise ValueError(message) from error
        index = cls(unique)
        i_position = index.positions_for(endpoint_i)
        j_position = index.positions_for(endpoint_j)
        if bool(jnp.any(i_position == j_position)):
            message = "undirected dyadic samples cannot contain self-dyads"
            raise ValueError(message)
        return index

    def positions_for(self, endpoint_ids: ArrayView) -> jax.Array:
        """Return integer positions for IDs under this exact canonical mapping."""
        if endpoint_ids.ndim != 1:
            message = "dyadic endpoint IDs must be a one-dimensional array"
            raise ValueError(message)
        lookup = {node: position for position, node in enumerate(self.node_ids)}
        positions: list[int] = []
        for raw_node in _as_id_sequence(endpoint_ids):
            node = _validate_node_id(raw_node)
            try:
                positions.append(lookup[node])
            except (KeyError, TypeError, ValueError) as error:
                message = f"endpoint ID is absent from the node index: {node!r}"
                raise ValueError(message) from error
        return jnp.asarray(positions, dtype=jnp.int64)


def _rank_profile_greedy(xf: jax.Array, threshold: float) -> list[int]:
    """Return the declared-order rank profile by masked greedy rank growth.

    Keep a column when adding it to the already-kept prefix raises the rank.
    That definition, not any particular factorization, is the contract.

    Columns are masked to zero rather than sliced out.  Zeroing column ``k``
    leaves ``xf @ xf.T`` -- and therefore every nonzero singular value --
    identical to deleting it, so the rank is exact.  Holding the width fixed is
    what makes it *fast*: the sliced candidate changed shape on every
    iteration, so the eager ``matrix_rank`` kernel missed its shape-keyed cache
    and recompiled once per column.  Measured on one rank-deficient 1326x74
    dyadic design, sliced against masked: 26.4 s and 742 compilations against
    0.67 s and 4, falling to 0.48 s and 0 once the two shapes are warm.

    Requires caller-owned ``jax_enable_x64=True``; ``xf`` must already be
    float64.  jcor opens no scope of its own.
    """
    width = xf.shape[1]
    flags = [0.0] * width
    keep: list[int] = []
    rank = 0
    for column in range(width):
        flags[column] = 1.0
        candidate = xf * jnp.asarray(flags, dtype=jnp.float64)[None, :]
        candidate_rank = int(jnp.linalg.matrix_rank(candidate, tol=threshold))
        if candidate_rank > rank:
            keep.append(column)
            rank = candidate_rank
        else:
            flags[column] = 0.0
    return keep


@jax.jit
def _rank_profile_gram_schmidt(xf: jax.Array, threshold: jax.Array) -> jax.Array:
    """Return the declared-order rank profile in one traced pass.

    Same greedy question as :func:`_rank_profile_greedy` -- does this column
    leave the span of its accepted predecessors? -- answered by carrying an
    orthonormal basis instead of re-factorizing a growing candidate.  Rejected
    slots of ``basis`` stay exactly zero, so ``basis @ (basis.T @ column)``
    projects onto the accepted prefix and nothing else; no masking is needed.
    The reprojection is Kahan's "twice is enough": one repeat restores
    orthogonality to machine precision after cancellation, which plain
    Gram-Schmidt loses.

    ``lax.scan`` keeps every shape fixed, so this is one compilation and, unlike
    the ``matrix_rank`` loop, carries no per-column device synchronization --
    the whole profile is traceable.  ``threshold`` is a traced argument rather
    than a closed-over Python float on purpose: as a constant it lands in the
    jaxpr and re-keys the cache on every call, which cost one recompilation per
    invocation (measured 20 compilations across 20 calls) and gave back most of
    what the single pass buys.

    The accept test is *not* the same predicate as growing ``matrix_rank``.
    Greedy asks whether ``sigma_{k+1}`` clears the threshold; this asks whether
    the orthogonal residual norm does, and ``sigma_min([A a]) <= dist(a,
    span(A))``, so this is the more permissive of the two.  They can only
    disagree when a singular value lands near the threshold, which is a regime
    the callers do not reach -- see :func:`drop_collinear_columns`.

    Requires caller-owned ``jax_enable_x64=True``; ``xf`` must already be
    float64.  jcor opens no scope of its own.
    """
    rows, width = xf.shape
    capacity = min(rows, width)
    if capacity == 0:
        # ``scan`` still traces its body once at zero length, and indexing an
        # empty basis raises rather than being skipped.
        return jnp.zeros((width,), dtype=bool)

    def step(
        carry: tuple[jax.Array, jax.Array], column: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        basis, filled = carry
        residual = column - basis @ (basis.T @ column)
        residual = residual - basis @ (basis.T @ residual)
        norm = jnp.linalg.norm(residual)
        # ``filled < capacity`` cannot bind on a real design -- no more than
        # ``capacity`` columns can be independent -- but without it a numerical
        # accept past the last slot would clamp onto, and destroy, an existing
        # basis vector rather than being ignored.
        accept = (norm > threshold) & (filled < capacity)
        unit = jnp.where(accept, residual / jnp.where(accept, norm, 1.0), 0.0)
        index = jnp.minimum(filled, capacity - 1)
        return (
            jnp.where(accept, basis.at[:, index].set(unit), basis),
            filled + accept.astype(filled.dtype),
        ), accept

    initial = (jnp.zeros((rows, capacity), dtype=xf.dtype), jnp.asarray(0))
    _carry, accepted = jax.lax.scan(step, initial, xf.T)
    return accepted


def drop_collinear_columns(x: ArrayLike, tol: float = 1e-10) -> jax.Array:
    """Return an ordered maximal linearly independent set of column indices.

    Greedy rank growth keeps the first representative of each collinear column
    family, so the declared column order is preserved and a named regressor is
    never dropped in favour of a generated effect that follows it.

    :func:`_rank_profile_gram_schmidt` answers that in one traced pass;
    :func:`_rank_profile_greedy` is the ``matrix_rank`` reference the parity
    sweep in ``tests.operators.test_design`` holds it to.  The two use different
    numerical predicates and agree on every structural, random, and wide case in
    that sweep; they diverge only when a singular value sits within about two
    orders of magnitude of ``tol * ||x||_2``.  Across 600 real node-bootstrap
    designs -- both the weighted full design and the effect-only design of
    ``model._dyadic.inference`` -- no singular value landed in that band: the
    closest retained one was 2.6e8 times the threshold and the closest dropped
    nonzero one 3.6e-6 times it, because collinearity here is structural (a
    zero-weight node zeroes a column exactly) rather than marginal.  Condition
    number is *not* the diagnostic to check this with -- every one of those
    designs exceeded 1e9, and the worst reached 1e99, purely from exact zeros.

    Neither QR variant can replace the loop, and the tests pin both
    counterexamples:

    * **Unpivoted**, reading the profile off ``|diag(R)|``, under-selects.  A
      dependent column consumes its diagonal row whether or not it adds rank,
      which shifts every later independent column off the diagonal.  On
      ``[[1, 1, 0], [0, 0, 1]]`` the duplicate at index 1 burns row 1 and the
      diagonal reads ``[1, 0]``, yielding ``[0]`` where the rank-2 contract
      wants ``[0, 2]``.
    * **Column-pivoted** (``geqp3``, ``jax.lax.linalg.qr(..., pivoting=True)``)
      reorders by norm, so it returns *a* maximal independent set rather than
      *the* declared-order one.  On ``[[1, 0, 1], [0, 1, 1]]`` it prefers the
      largest-norm column and yields ``[0, 2]`` where the contract wants
      ``[0, 1]``.

    Requires caller-owned ``jax_enable_x64=True``.  The 1e9 conditioning above
    is not a curiosity: under float32 the threshold test has no signal left, and
    a full
    ``dvc repro`` with the caller flag off rejected every one of 1999
    node-bootstrap draws in ``p1_dyadic_confound``.  An array born under x64 also
    keeps its dtype *label* outside the scope while later ops silently compute
    in float32 (measured 2.5e-8 relative error on the SSE path), so a dtype
    assertion alone does not detect the loss.

    Args:
        x: Two-dimensional design matrix.
        tol: Relative matrix-rank tolerance, scaled by ``max(||x||_2, 1)``.

    Returns:
        Integer indices of retained columns, in their original order.

    """
    xa = jnp.asarray(x)
    if xa.ndim != _MATRIX_DIMENSIONS:
        message = f"design must be two-dimensional; received shape {xa.shape}"
        raise ValueError(message)
    xf = jnp.asarray(x).astype(jnp.float64)
    scale = max(float(jnp.linalg.norm(xf, ord=2)), 1.0)
    threshold = jnp.asarray(tol * scale, dtype=jnp.float64)
    accepted = _rank_profile_gram_schmidt(xf, threshold)
    return jnp.flatnonzero(accepted).astype(jnp.int64)


def symmetric_dyadic_effects(
    endpoint_i: ArrayView,
    endpoint_j: ArrayView,
    *,
    name_prefix: str = "node_effect",
    node_index: NodeIndex[Hashable] | None = None,
) -> tuple[jax.Array, list[str]]:
    """Construct additive undirected endpoint effects with one base omitted.

    Row ``r`` contains ``1{endpoint_i[r] = k} + 1{endpoint_j[r] = k}`` for
    every endpoint ``k`` except the first endpoint in the canonical node index.

    Args:
        endpoint_i: First endpoint label for each dyad.
        endpoint_j: Second endpoint label for each dyad.
        name_prefix: Prefix used for the non-base endpoint-effect labels.
        node_index: Optional validated mapping shared with fit/bootstrap code.

    Returns:
        Effect matrix and aligned ``<name_prefix>[...]`` column labels.

    Raises:
        ValueError: If endpoint arrays are not aligned or name fewer than two
            distinct endpoints.

    """
    if not isinstance(name_prefix, str) or not name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise ValueError(message)
    index = node_index or NodeIndex.from_endpoints(endpoint_i, endpoint_j)
    i_position = index.positions_for(endpoint_i)
    j_position = index.positions_for(endpoint_j)
    observed_positions = set(i_position.tolist()) | set(j_position.tolist())
    if observed_positions != set(range(len(index))):
        message = "node index must exactly match the observed dyadic endpoint IDs"
        raise ValueError(message)
    if bool(jnp.any(i_position == j_position)):
        message = "undirected dyadic samples cannot contain self-dyads"
        raise ValueError(message)
    columns = [
        (i_position == position).astype(jnp.float64)
        + (j_position == position).astype(jnp.float64)
        for position in range(1, len(index))
    ]
    effects = jnp.column_stack(columns)
    return effects, [f"{name_prefix}[{endpoint}]" for endpoint in index.node_ids[1:]]


def fwl_coefficient(x: ArrayLike, y: ArrayLike, focal_column: int) -> float:
    """Return one OLS coefficient by Frisch--Waugh--Lovell residualization.

    Args:
        x: Full-rank design matrix.
        y: Aligned outcome vector.
        focal_column: Column of ``x`` whose coefficient is requested.

    Returns:
        The residual-on-residual slope, or ``NaN`` when the focal regressor has
        no variation after nuisance projection.

    """
    xa = jnp.asarray(x)
    ya = jnp.asarray(y)
    if xa.ndim != _MATRIX_DIMENSIONS or ya.ndim != 1 or xa.shape[0] != ya.shape[0]:
        message = "design and outcome must be an aligned matrix/vector pair"
        raise ValueError(message)
    xf = jnp.asarray(x).astype(jnp.float64)
    yf = jnp.asarray(y).astype(jnp.float64)
    nuisance = jnp.delete(xf, focal_column, axis=1)
    focal = xf[:, focal_column]
    y_residual = yf - nuisance @ jnp.linalg.lstsq(nuisance, yf, rcond=None)[0]
    focal_residual = focal - nuisance @ jnp.linalg.lstsq(nuisance, focal, rcond=None)[0]
    denominator = float(focal_residual @ focal_residual)
    if denominator <= 0.0:
        return float("nan")
    return float((focal_residual @ y_residual) / denominator)

"""PackedClouds: padded-pack + masked-reduction substrate for sample clouds.

Standardizes the padded-pack idiom used by the package's equal-size cloud
stacks and triangle-halving pair chunking
(`jcor.discrepancy._mmd.components._triu_pair_chunks`, t11).
Those operations previously re-derived padding/counts/chunking locally; this
module gives them one shared implementation.

``PackedClouds`` is a registered frozen dataclass. Its ``__post_init__`` checks
only structural facts that remain static while JAX traces a program: JAX-array
leaves, numeric/integer dtypes, ranks, the shared leading batch axes, and the
shared cloud axis. Concrete count values deliberately live at an eager boundary:
use :func:`checked_packed_clouds` for externally assembled buffers, or
:func:`pack` for a list of clouds. Jaxtyping's test import hook independently
checks the same field annotations, but production safety does not depend on that
hook. Registration is adopted for structural enforcement, **not** as a
performance lever: t13's rejection of custom registration as a way to speed up
kernels still stands. The dataclass has exactly the same two dynamic leaves, in
the same order, as the former ``NamedTuple``.

Both fields admit the same leading ``*batch`` axes. ``pack`` itself is a host-side,
unjitted adapter and returns the unbatched ``(k, m, d)`` / ``(k,)`` form; the batch
axes are admitted so ``vmap`` and ``lax.map`` can rebuild transformed outputs by
calling the checked dataclass constructor. This follows the measured ``DMat``
discipline in :mod:`jcor.core.typing` and does not add a physical axis, metadata
leaf, or semantic brand to an ordinary packed value.

``pack`` is an eager host adapter. It uses one tree-level
``jax.device_get(tuple(clouds))`` call before the NumPy build and one tree-level
``jax.device_put((data, counts))`` call afterwards. These are API-call and
synchronisation boundaries, not a claim that a two-leaf pytree needs one physical
copy: each array leaf can still transfer separately.

einsum policy: use ``jnp.einsum`` for NEW multi-operand contractions where
index structure clarifies intent over a chain of `sum`/broadcast calls (see
the quadratic forms in ``jcor.inference._projection``). Do NOT rewrite
working two-operand matmuls into einsum form purely for style — same XLA
``dot_general``, ~1e-8 parity churn for zero gain (rejected 2026-07-18).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jcor.core.typing import Array, Int, Num  # noqa: TC001  # runtime fields

_PACKED_DATA_AXES = 3
_CLOUD_DIMENSIONS = 2


def _require_jax_array(value: Array, *, field_name: str) -> None:
    """Require an immutable JAX array or a transform-time tracer leaf."""
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return
    message = (
        f"PackedClouds.{field_name} must be a JAX array; use pack or "
        "checked_packed_clouds to construct immutable leaves"
    )
    raise TypeError(message)


def _require_packed_structure(data: Array, counts: Array) -> None:
    """Validate only shape/dtype metadata, which remains concrete under tracing."""
    _require_jax_array(data, field_name="data")
    _require_jax_array(counts, field_name="counts")

    data_array = data
    counts_array = counts
    if data_array.ndim < _PACKED_DATA_AXES:
        message = (
            "PackedClouds.data must have shape (*batch, K, m_max, d); "
            f"got {data_array.shape}"
        )
        raise ValueError(message)
    expected_count_rank = data_array.ndim - (_PACKED_DATA_AXES - 1)
    if counts_array.ndim != expected_count_rank:
        message = (
            "PackedClouds.counts must have shape (*batch, K) matching data; "
            f"got data {data_array.shape} and counts {counts_array.shape}"
        )
        raise ValueError(message)
    if (
        data_array.shape[:-_PACKED_DATA_AXES] != counts_array.shape[:-1]
        or data_array.shape[-_PACKED_DATA_AXES] != counts_array.shape[-1]
    ):
        message = (
            "PackedClouds data/counts batch and cloud axes must match; "
            f"got data {data_array.shape} and counts {counts_array.shape}"
        )
        raise ValueError(message)
    if not jnp.issubdtype(data_array.dtype, jnp.number):
        message = f"PackedClouds.data must have a numeric dtype, got {data_array.dtype}"
        raise TypeError(message)
    if not jnp.issubdtype(counts_array.dtype, jnp.signedinteger):
        message = (
            "PackedClouds.counts must have a signed integer dtype, "
            f"got {counts_array.dtype}"
        )
        raise TypeError(message)


def _require_count_values(
    counts: Int[Array, " k"],  # noqa: F722  # jaxtyping shape
    *,
    m_max: int,
    require_positive_counts: bool,
) -> None:
    """Validate concrete count bounds at an eager host boundary."""
    minimum = 1 if require_positive_counts else 0
    if require_positive_counts and counts.size == 0:
        message = "positive-count PackedClouds require at least one cloud"
        raise ValueError(message)
    if bool(jnp.any(counts < minimum)) or bool(jnp.any(counts > m_max)):
        interval = f"[1, {m_max}]" if require_positive_counts else f"[0, {m_max}]"
        message = f"PackedClouds counts must lie in {interval}, got {counts.tolist()}"
        raise ValueError(message)


def _require_boolean_flag(value: object, *, name: str) -> bool:
    """Validate an eager boolean option without relying on the import hook."""
    if not isinstance(value, bool):
        message = f"{name} must be a bool, got {type(value).__name__}"
        raise TypeError(message)
    return bool(value)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PackedClouds:
    """Padded stack of ragged sample clouds plus their true sizes.

    A registered dataclass pytree: ``data`` and ``counts`` are its only leaves,
    so ``jit``/``vmap``/``lax.map`` traverse the same dynamic structure as the
    former ``NamedTuple`` carrier. The raw constructor checks tracer-safe
    structure only; it cannot inspect count values while JAX is tracing. Use
    :func:`checked_packed_clouds` when counts come from an external source.

    Attributes:
        data: Padded clouds, shape ``(*batch, K, m_max, d)``. :func:`pack`
            writes zero padding; external inactive entries are ignored by the
            masked reducers and are not value-checked.
        counts: True (unpadded) size of each cloud, shape ``(*batch, K)``.

    """

    data: Num[Array, "*batch k m d"]  # noqa: F722  # jaxtyping shape
    counts: Int[Array, "*batch k"]  # noqa: F722  # jaxtyping shape

    def __post_init__(self) -> None:
        """Enforce the carrier's tracer-safe structural contract in production."""
        _require_packed_structure(self.data, self.counts)


def checked_packed_clouds(
    data: Num[Array, "*batch k m d"],  # noqa: F722  # jaxtyping shape
    counts: Int[Array, "*batch k"],  # noqa: F722  # jaxtyping shape
    *,
    require_positive_counts: bool = False,
) -> PackedClouds:
    """Construct a packed carrier after eager validation of count values.

    This is the external-buffer door. It is intentionally host-only: count
    values cannot be tested with Python control flow while they are JAX
    tracers. It proves count bounds only; inactive padding values are ignored,
    not inspected or claimed to be zero. The raw :class:`PackedClouds`
    constructor remains available inside ``jit``/``vmap``/``lax.map`` and
    enforces structure only.

    Args:
        data: Padded JAX array of shape ``(*batch, K, m_max, d)``.
        counts: True row extents, shape ``(*batch, K)``.
        require_positive_counts: Require every extent to be at least one and
            reject an empty carrier. Statistical consumers such as T62 should
            enable this; the generic carrier retains zero-count compatibility.

    Returns:
        A structurally and value-checked :class:`PackedClouds`.

    Raises:
        TypeError: If called while ``counts`` is a JAX tracer, or if the boolean
            option or structural leaf/dtype contract is invalid.
        ValueError: If structural axes or a count value are outside their contracts.

    """
    positive_counts = _require_boolean_flag(
        require_positive_counts,
        name="require_positive_counts",
    )
    packed = PackedClouds(data=data, counts=counts)
    if isinstance(packed.counts, jax.core.Tracer):
        message = (
            "checked_packed_clouds is an eager value-law boundary; construct "
            "PackedClouds directly inside JAX transformations"
        )
        raise TypeError(message)
    host_counts = jnp.asarray(jax.device_get(packed.counts))
    _require_count_values(
        host_counts,
        m_max=int(packed.data.shape[-2]),
        require_positive_counts=positive_counts,
    )
    return packed


def pack(
    clouds: list[jnp.ndarray],
    *,
    require_positive_counts: bool = False,
) -> PackedClouds:
    """Zero-pad a ragged list of ``(m_k, d)`` clouds into a ``PackedClouds``.

    Args:
        clouds: List of ``K`` arrays, each of shape ``(m_k, d)``; ``m_k`` may
            vary across clouds, but ``d`` must be shared. Mixed dtypes retain
            the historical first-cloud dtype selection.
        require_positive_counts: Require a nonempty list and ``m_k > 0`` for
            every cloud. Leave false for the historical ``pack([])`` result and
            for generic carriers that deliberately admit empty clouds.

    Returns:
        ``PackedClouds`` with ``data`` shape ``(K, m_max, d)`` (``m_max`` the
        largest ``m_k``) and ``counts`` shape ``(K,)``. The counts are derived
        from the original cloud shapes rather than supplied by the caller.

    Raises:
        TypeError: If ``require_positive_counts`` is not a boolean.
        ValueError: If a cloud is not rank two, feature widths differ, or the
            positive-count door receives an empty list/cloud.
        OverflowError: If a derived row count cannot fit the preserved int32 dtype.

    """
    positive_counts = _require_boolean_flag(
        require_positive_counts,
        name="require_positive_counts",
    )
    k = len(clouds)
    for index, cloud in enumerate(clouds):
        if not hasattr(cloud, "shape") or len(cloud.shape) != _CLOUD_DIMENSIONS:
            shape = getattr(cloud, "shape", None)
            message = f"cloud {index} must be two-dimensional, got shape {shape}"
            raise ValueError(message)
    dims = {int(cloud.shape[1]) for cloud in clouds}
    if len(dims) > 1:
        message = f"inconsistent cloud feature widths: {dims}"
        raise ValueError(message)
    d = dims.pop() if dims else 0
    sizes = [int(cloud.shape[0]) for cloud in clouds]
    if any(size > jnp.iinfo(jnp.int32).max for size in sizes):
        message = "cloud row counts must fit the preserved int32 count dtype"
        raise OverflowError(message)
    counts = jnp.asarray(sizes, dtype=jnp.int32)
    m_max = max(sizes) if k else 0
    _require_count_values(
        counts,
        m_max=m_max,
        require_positive_counts=positive_counts,
    )
    dtype = clouds[0].dtype if k else jnp.float32
    # One fused scatter, not a per-cloud loop. `m_max` comes from Python
    # `.shape[0]` ints, so every index below is static; the rows are
    # concatenated once and written in a single `.at[].set`, which measured
    # ~9x faster than the NumPy buffer-and-slice-assign it replaces and
    # bit-identical to it. The obvious alternative — `jnp.pad` each cloud then
    # `stack` — measured ~12x *slower* than NumPy, because it emits one XLA op
    # per cloud instead of one for the batch.
    if k == 0 or m_max == 0:
        data = jnp.zeros((k, m_max, d), dtype=dtype)
    else:
        flat_rows = jnp.concatenate([jnp.asarray(cloud) for cloud in clouds], axis=0)
        row_targets = jnp.asarray(
            [i * m_max + j for i, size in enumerate(sizes) for j in range(size)],
            dtype=jnp.int32,
        )
        # `flat_rows.dtype`, not `clouds[0].dtype`: the clouds may be NumPy, and
        # feeding a NumPy dtype to `jnp.zeros` requests a width JAX may not be
        # configured for (a float64 request warns and truncates under the
        # default config). Taking it from the already-converted rows asks for
        # the width the scatter will actually write.
        data = (
            jnp.zeros((k * m_max, d), dtype=flat_rows.dtype)
            .at[row_targets]
            .set(flat_rows)
            .reshape(k, m_max, d)
        )
    return PackedClouds(data=data, counts=counts)


def masked_row_mask(m_max: int, count: jnp.ndarray) -> jnp.ndarray:
    """Boolean/float validity mask ``ar < count`` for one padded row.

    Args:
        m_max: Padded row length.
        count: True (unpadded) size, scalar or shape ``(...,)``.

    Returns:
        Boolean array of shape ``(..., m_max)``, ``True`` where the row
        position is within the true (unpadded) extent.

    """
    ar = jnp.arange(m_max)
    return ar < count[..., None] if count.ndim else ar < count


def masked_sum(
    values: jnp.ndarray, row_count: jnp.ndarray, col_count: jnp.ndarray
) -> jnp.ndarray:
    """Sum of ``values`` over the valid ``(row_count, col_count)`` leading block.

    Zeroes the padded region before summing, matching the leading-block
    masked-reduction pattern used by the package's pairwise kernels.

    Args:
        values: Pairwise values, shape ``(m_max, n_max)``.
        row_count: Valid row extent (scalar).
        col_count: Valid column extent (scalar).

    Returns:
        Scalar sum over ``values[:row_count, :col_count]``.

    """
    m_max, n_max = values.shape
    row_mask = jnp.arange(m_max) < row_count
    col_mask = jnp.arange(n_max) < col_count
    valid = row_mask[:, None] & col_mask[None, :]
    return jnp.sum(jnp.where(valid, values, jnp.zeros((), dtype=values.dtype)))


def masked_mean(
    values: jnp.ndarray, row_count: jnp.ndarray, col_count: jnp.ndarray
) -> jnp.ndarray:
    """Mean of ``values`` over the valid ``(row_count, col_count)`` leading block.

    Args:
        values: Pairwise values, shape ``(m_max, n_max)``.
        row_count: Valid row extent (scalar).
        col_count: Valid column extent (scalar).

    Returns:
        Scalar mean over ``values[:row_count, :col_count]``.

    """
    total = masked_sum(values, row_count, col_count)
    denom = row_count.astype(values.dtype) * col_count.astype(values.dtype)
    return total / denom


# Pairs per lax.map step for chunked upper-triangle pair kernels. Callers may
# override the shared default via `chunk`.
DEFAULT_PAIR_CHUNK = 16


def pair_chunks(
    n: int, k: int = 0, chunk: int = DEFAULT_PAIR_CHUNK
) -> tuple[
    Int[Array, " p"],  # noqa: F722  # jaxtyping shape
    Int[Array, " p"],  # noqa: F722  # jaxtyping shape
    int,
    Int[Array, "c q 2"],  # noqa: F722  # jaxtyping shape
]:
    """Upper-triangle ``(i, j)`` index pairs, padded and chunked for ``lax.map``.

    Produces the ``i <= j`` (``k=0``) or ``i < j`` (``k=1``) entries needed by
    upper-triangle cloud reductions, since the lower triangle is redundant
    matmul work in exact arithmetic (author-approved contract relaxation
    2026-07-18; mirrored entries are equal only up to
    summation-order/XLA-tiling tolerance).

    Args:
        n: Matrix side length (number of clouds).
        k: Diagonal offset for :func:`numpy.triu_indices` (0 keeps the
            diagonal, 1 drops it).
        chunk: Pairs per ``lax.map`` step.

    Returns:
        Tuple ``(ii, jj, n_pairs, chunks)`` where ``ii``/``jj`` are the
        unpadded row/column indices, ``n_pairs`` their count, and ``chunks``
        the zero-padded ``(n_chunks, chunk, 2)`` pair grid.

    """
    # `jnp.triu_indices` returns int64 under x64, and the padded grid is int32
    # by contract (it indexes clouds, and `PackedClouds` pins the count dtype).
    # NumPy downcast that silently on assignment; a JAX scatter warns today and
    # is documented to become an error, so the narrowing is explicit here.
    ii, jj = (index.astype(jnp.int32) for index in jnp.triu_indices(n, k=k))
    n_pairs = int(ii.shape[0])
    n_pad = (-n_pairs) % chunk
    pairs = jnp.zeros((n_pairs + n_pad, 2), dtype=jnp.int32)
    pairs = pairs.at[:n_pairs, 0].set(ii)
    pairs = pairs.at[:n_pairs, 1].set(jj)
    return ii, jj, n_pairs, pairs.reshape(-1, chunk, 2)

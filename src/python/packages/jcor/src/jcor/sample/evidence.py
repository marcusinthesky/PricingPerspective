"""Checked empirical clouds — value evidence minted at eager doors.

Position
--------
rank 1 · wraps :mod:`jcor.sample.packed`, adds no field to it

What this module is for
-----------------------
:mod:`jcor.core.domains` says *which* value facts exist and how they imply one
another. :class:`~jcor.sample.packed.PackedClouds` carries the numbers. Neither
one measures anything: the packed carrier checks structure (and, at
:func:`~jcor.sample.packed.checked_packed_clouds`, count *bounds*) but never
inspects a value, and the evidence markers are freely constructible.

This module is where the two meet. :class:`CheckedClouds` is a thin wrapper
holding a packed carrier plus the evidence a door actually established by
looking at the numbers. The wrapper adds **no field to ``PackedClouds``** and no
leaf to the tree: its nested dynamic leaves remain exactly ``data`` and
``counts``, and the evidence markers travel as *static* (``meta_fields``)
metadata, so no provenance object can enter a compiled array tree.

Only active rows are validated
------------------------------
The packed carrier is ragged: ``data`` has shape ``(*batch, K, m_max, d)`` and
only the first ``counts[..., k]`` rows of cloud ``k`` are data. Padding is not
data. Every door here masks to the active block before it tests anything, so a
zero-padded row cannot fail a nonzero-norm check and a ``NaN``-padded external
buffer cannot fail a finiteness check. The masking is ``where``-based rather
than multiplicative for the same reason as
:func:`jcor.sample.packed.masked_sum`: ``NaN * 0 = NaN`` would leak an inactive
value into an active reduction.

The two evidence axes are separate on purpose
---------------------------------------------
``support`` and ``pooled`` are **two fields minted by two checks**, never one
"fully checked" fact. :class:`~jcor.core.domains.UnitNormSupport` and
:class:`~jcor.core.domains.NonzeroPooledMean` are ``@final`` nominal siblings
because the antipodal cloud ``{x, -x}`` has unit rows and pools to *exactly*
zero. A type joining them would assert an implication that is false; only
:func:`checked_nonzero_pooled_mean`, which measures ``‖mean‖``, may mint the
pooled fact. ``tests/sample/test_evidence.py`` witnesses the antipodal cloud at
runtime rather than merely declaring it.

Doors reject; they never repair
-------------------------------
A zero row makes ``N(x) = x / ‖x‖`` undefined. JCOR direct-carrier doors
validate and never transform; explicitly selected pullback doors perform only
their declared map. Pipeline policy adapters likewise reject undefined rows
rather than substitute a norm or fabricate unit-sphere evidence.

What a finite sample still does not prove
-----------------------------------------
Nothing in this module reaches a population claim. Finite observed values are
not a finite population moment, checked observed rows are not an almost-sure
support policy, and an observed statistic is not consistency or strong negative
type. :class:`PopulationHypotheses` therefore exists only behind
:func:`declare_population_hypotheses`, whose arguments are supplied by the
caller as hypotheses. **There is deliberately no function from
:class:`CheckedClouds` to :class:`PopulationHypotheses`**, and that absence is
the deliverable; the static fixtures in the test module pin it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Generic, TypeVar, final

import jax
import jax.numpy as jnp

from jcor.core.domains import (  # noqa: TC001  # runtime dataclass fields
    AlmostSureNonzeroSupport,
    CheckedEvidence,
    ComputedEvidence,
    DomainT,
    EvidenceOrigin,
    FiniteMoment,
    FiniteValues,
    MomentOrder,
    MomentT,
    NonzeroNormalizationDomain,
    NonzeroPooledMean,
    NonzeroVector,
    ObjectDomain,
    RawVector,
    UnitNormSupport,
    UnitSphereValue,
    UnsafeAssumption,
    ValueEvidence,
)
from jcor.core.typing import Array, ArrayLike  # noqa: TC001
from jcor.sample.packed import (
    PackedClouds,
    pack,
)

__all__ = [
    "CheckedClouds",
    "PopulationHypotheses",
    "Unchecked",
    "add_checked_clouds",
    "assume_cloud_evidence",
    "checked_finite_clouds",
    "checked_nonzero_clouds",
    "checked_nonzero_pooled_mean",
    "checked_unit_norm_clouds",
    "computed_l2_normalize_clouds",
    "concatenate_checked_clouds",
    "declare_population_hypotheses",
    "forget_cloud_evidence",
    "permute_checked_rows",
    "pool_checked_clouds",
    "project_checked_clouds",
    "scale_checked_clouds",
    "select_checked_clouds",
    "truncate_checked_features",
]

#: Default absolute tolerance for ``|‖x‖ - 1| <= atol`` at the unit-norm door.
#: Loose enough for a float32 carrier normalized upstream, tight enough that an
#: unnormalized row cannot pass. Callers working in float64 may tighten it.
DEFAULT_UNIT_NORM_ATOL = 1e-6
_UNBATCHED_PACKED_NDIM = 3
_MATRIX_NDIM = 2


@final
@dataclass(frozen=True, slots=True)
class Unchecked(ValueEvidence):
    """The absence of evidence on an axis, as a value rather than ``None``.

    A direct child of :class:`~jcor.core.domains.ValueEvidence`, hence an
    unrelated sibling of every checked fact: a carrier whose ``pooled`` slot is
    ``Unchecked`` cannot be passed where
    :class:`~jcor.core.domains.NonzeroPooledMean` is required. It is a frozen
    zero-field dataclass so it compares and hashes by value, which is what makes
    it legal ``meta_fields`` metadata — an identity-hashed sentinel would give
    two structurally identical carriers different treedefs and split the ``jit``
    cache.
    """


#: **Covariant** value evidence about the *rows* of a checked carrier. A door
#: that established unit norms satisfies a requirement for finite values or a
#: defined normalization domain; the reverse never type-checks. The explicit
#: ``TypeVar`` is required for the same reason as
#: :data:`jcor.core.domains.DomainT`: the parameter appears only in a field
#: whose type *is* the parameter, and PEP 695 settles on invariant.
SupportT = TypeVar("SupportT", bound=ValueEvidence, covariant=True)  # noqa: PLC0105

#: **Covariant** evidence about the *pooled mean* of a checked carrier. Held
#: separately from :data:`SupportT` because no row-level fact implies it.
PooledT = TypeVar("PooledT", bound=ValueEvidence, covariant=True)  # noqa: PLC0105

#: **Covariant** provenance of the value evidence. Existing annotations that
#: omit this fourth parameter widen to :class:`EvidenceOrigin`; doors and new
#: consumers should request the exact checked/computed/unsafe origin.
ValueOriginT = TypeVar(  # noqa: PLC0105
    "ValueOriginT",
    bound=EvidenceOrigin,
    covariant=True,
    default=EvidenceOrigin,
)


@final
@dataclass(frozen=True, slots=True)
class _EvidenceMint:
    """Private token required by the carrier constructor and JAX unflatten."""


_EVIDENCE_MINT = _EvidenceMint()


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["clouds"],
    meta_fields=["support", "pooled", "origin", "_mint"],
)
@dataclass(frozen=True)
class CheckedClouds(
    Generic[DomainT, SupportT, PooledT, ValueOriginT]  # noqa: UP046  # variance
):
    """A packed carrier plus the value evidence a door measured on it.

    ``DomainT`` is phantom — it records which object domain the *rows* were
    checked to inhabit (raw vectors, finite-nonzero vectors, unit-sphere
    values) and appears in no field. ``support`` and ``pooled`` are static
    metadata, so ``jit``/``vmap`` traverse exactly the nested ``data`` and
    ``counts`` leaves of the wrapped :class:`~jcor.sample.packed.PackedClouds`.

    There is no ``__post_init__`` value check, and that is deliberate:
    ``tree_unflatten`` rebuilds a registered dataclass by *calling* ``__init__``
    with traced leaves, so a value law here would break every transform. Value
    laws live at the eager doors below, which reject tracers outright. The raw
    constructor requires a private mint token. Ordinary callers therefore use
    a checked/computed/unsafe door, and JAX unflatten receives the same static
    token through the registered metadata.

    Attributes:
        clouds: The wrapped ragged carrier; the only dynamic child.
        support: Row-level evidence established by a door — finiteness, a
            defined normalization domain, or unit norms.
        pooled: Pooled-mean evidence, established only by
            :func:`checked_nonzero_pooled_mean`.
        origin: Whether the facts were computed, checked, or unsafely assumed.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jcor.sample.packed import pack
        >>> checked = checked_finite_clouds(pack([jnp.ones((2, 3))]))
        >>> [leaf.shape for leaf in jax.tree_util.tree_leaves(checked)]
        [(1, 2, 3), (1,)]

    """

    clouds: PackedClouds
    support: SupportT
    pooled: PooledT
    origin: ValueOriginT
    _mint: _EvidenceMint

    def __post_init__(self) -> None:
        """Reject direct evidence construction outside the private producers."""
        if self._mint is not _EVIDENCE_MINT:
            message = (
                "CheckedClouds evidence is opaque; use a checked/computed door or "
                "the explicit assume_cloud_evidence escape hatch"
            )
            raise TypeError(message)


def _mint_checked_clouds[
    DomainKind: ObjectDomain,
    SupportKind: ValueEvidence,
    PooledKind: ValueEvidence,
    OriginKind: EvidenceOrigin,
](
    clouds: PackedClouds,
    *,
    support: SupportKind,
    pooled: PooledKind,
    origin: OriginKind,
) -> CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind]:
    """Private common producer; public doors own all policy and validation."""
    return CheckedClouds(
        clouds=clouds,
        support=support,
        pooled=pooled,
        origin=origin,
        _mint=_EVIDENCE_MINT,
    )


@final
@dataclass(frozen=True, slots=True)
class PopulationHypotheses(Generic[DomainT, MomentT]):  # noqa: UP046  # see variance
    """Population-level conditions a caller **declares**, never a sample proves.

    A normalization pushforward ``N#P`` needs ``P`` to put no mass at the
    origin, and the energy identification results need a finite moment at the
    exponent in use. Neither is checkable from finitely many observed rows, so
    this container is unreachable from :class:`CheckedClouds` by construction —
    no function in this module maps one to the other.

    Not a pytree and not registered: it holds no array and must never be
    threaded through a transform.

    Attributes:
        support: The declared almost-sure nonzero-support policy.
        moment: The declared finite-moment condition at order ``MomentT``.

    """

    support: AlmostSureNonzeroSupport
    moment: FiniteMoment[MomentT]


def _host_arrays(clouds: PackedClouds) -> tuple[Array, Array]:
    """Fetch concrete ``(data, counts)``, refusing traced carriers.

    Args:
        clouds: The packed carrier to inspect.

    Returns:
        Host copies of ``data`` and ``counts``.

    Raises:
        TypeError: If either leaf is a JAX tracer, since a value law cannot be
            decided with Python control flow while JAX is tracing.

    """
    if isinstance(clouds.data, jax.core.Tracer) or isinstance(
        clouds.counts, jax.core.Tracer
    ):
        message = (
            "jcor.sample.evidence doors are eager value-law boundaries; mint "
            "evidence outside jit/vmap and carry it as static metadata"
        )
        raise TypeError(message)
    data, counts = jax.device_get((clouds.data, clouds.counts))
    return jnp.asarray(data), jnp.asarray(counts)


def _active_mask(counts: Array, m_max: int) -> Array:
    """Boolean mask of the active (unpadded) rows, shape ``(*batch, K, m_max)``.

    Args:
        counts: True row extents, shape ``(*batch, K)``.
        m_max: Padded row length.

    Returns:
        ``True`` exactly where a row position lies inside its cloud's extent.

    """
    return jnp.arange(m_max) < counts[..., None]


def _active_row_norms(data: Array, mask: Array) -> Array:
    """Euclidean norms of the active rows only, as a flat vector.

    Padded rows are dropped *before* the norm is compared rather than zeroed
    afterwards, mirroring the ``where``-based masking of
    :func:`jcor.sample.packed.masked_sum`: a multiplicative mask would turn a
    ``NaN`` padding row into a ``NaN`` norm.

    Args:
        data: Padded rows, shape ``(*batch, K, m_max, d)``.
        mask: Active-row mask, shape ``(*batch, K, m_max)``.

    Returns:
        Norms of the active rows, shape ``(n_active,)``.

    """
    return jnp.linalg.norm(data[mask], axis=-1)


def _require_finite_rows(data: Array, mask: Array) -> None:
    """Reject a carrier whose active block holds ``NaN`` or ``±inf``.

    Args:
        data: Padded rows, shape ``(*batch, K, m_max, d)``.
        mask: Active-row mask, shape ``(*batch, K, m_max)``.

    Raises:
        ValueError: If any active entry is not finite.

    """
    active = data[mask]
    if not bool(jnp.all(jnp.isfinite(active))):
        offending = int(jnp.count_nonzero(~jnp.isfinite(active).all(axis=-1)))
        message = (
            "checked clouds require finite active values; "
            f"{offending} active row(s) contain NaN or infinity"
        )
        raise ValueError(message)


def _require_tolerance(atol: float) -> float:
    """Validate an eager absolute tolerance.

    Args:
        atol: Candidate tolerance.

    Returns:
        The tolerance as a float.

    Raises:
        TypeError: If ``atol`` is not a real number.
        ValueError: If ``atol`` is negative or not finite.

    """
    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        message = f"atol must be a real number, got {type(atol).__name__}"
        raise TypeError(message)
    value = float(atol)
    if not math.isfinite(value) or value < 0.0:
        message = f"atol must be finite and nonnegative, got {value}"
        raise ValueError(message)
    return value


def checked_finite_clouds(
    clouds: PackedClouds,
) -> CheckedClouds[RawVector, FiniteValues, Unchecked, CheckedEvidence]:
    """Check that every **active** value is finite.

    The widest door: it admits zero rows, so the carrier stays a
    :class:`~jcor.core.domains.RawVector` domain on which a normalizing
    strategy is *not* total.

    Args:
        clouds: Packed carrier, validated only over its active rows.

    Returns:
        The same carrier branded with :class:`~jcor.core.domains.FiniteValues`.

    Raises:
        TypeError: If the carrier's leaves are JAX tracers.
        ValueError: If an active entry is ``NaN`` or infinite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jcor.sample.packed import pack
        >>> checked_finite_clouds(pack([jnp.ones((2, 2))])).support
        FiniteValues()

    """
    data, counts = _host_arrays(clouds)
    mask = _active_mask(counts, int(data.shape[-2]))
    _require_finite_rows(data, mask)
    return _mint_checked_clouds(
        clouds,
        support=FiniteValues(),
        pooled=Unchecked(),
        origin=CheckedEvidence(),
    )


def checked_nonzero_clouds(
    clouds: PackedClouds,
) -> CheckedClouds[
    NonzeroVector,
    NonzeroNormalizationDomain,
    Unchecked,
    CheckedEvidence,
]:
    """Check that every active row is finite with a **nonzero** norm.

    This is the door that makes ``N(x) = x / ‖x‖`` total on the carrier. It
    rejects a zero row; it does not repair one. Choosing a replacement, a drop
    policy, or a documented total map is the calling pipeline's decision, and
    jcor must not make it silently.

    Args:
        clouds: Packed carrier, validated only over its active rows.

    Returns:
        The carrier branded
        :class:`~jcor.core.domains.NonzeroNormalizationDomain` over the
        :class:`~jcor.core.domains.NonzeroVector` domain.

    Raises:
        TypeError: If the carrier's leaves are JAX tracers.
        ValueError: If an active entry is nonfinite or an active row has zero
            norm.

    """
    data, counts = _host_arrays(clouds)
    mask = _active_mask(counts, int(data.shape[-2]))
    _require_finite_rows(data, mask)
    norms = _active_row_norms(data, mask)
    zero_rows = int(jnp.count_nonzero(norms <= 0.0))
    if zero_rows:
        message = (
            "normalization is undefined on a zero row: "
            f"{zero_rows} active row(s) have zero norm; normalize or drop them "
            "in the calling stage, jcor does not repair carriers"
        )
        raise ValueError(message)
    return _mint_checked_clouds(
        clouds,
        support=NonzeroNormalizationDomain(),
        pooled=Unchecked(),
        origin=CheckedEvidence(),
    )


def checked_unit_norm_clouds(
    clouds: PackedClouds,
    *,
    atol: float = DEFAULT_UNIT_NORM_ATOL,
) -> CheckedClouds[
    UnitSphereValue,
    UnitNormSupport,
    Unchecked,
    CheckedEvidence,
]:
    """Check that every active row satisfies ``‖x‖ = 1`` within ``atol``.

    The only domain on which the chord formula ``‖N(x) - N(y)‖`` separates
    without further evidence. Note what this door does **not** establish: unit
    rows say nothing about the pooled mean, because ``{x, -x}`` averages to
    exactly zero. Use :func:`checked_nonzero_pooled_mean` for that fact.

    Args:
        clouds: Packed carrier, validated only over its active rows.
        atol: Absolute tolerance on ``|‖x‖ - 1|``.

    Returns:
        The carrier branded :class:`~jcor.core.domains.UnitNormSupport` over
        the :class:`~jcor.core.domains.UnitSphereValue` domain, with its pooled
        axis still :class:`Unchecked`.

    Raises:
        TypeError: If the carrier's leaves are JAX tracers or ``atol`` is not a
            real number.
        ValueError: If ``atol`` is negative/nonfinite, or an active row is
            nonfinite or off the unit sphere.

    """
    tolerance = _require_tolerance(atol)
    data, counts = _host_arrays(clouds)
    mask = _active_mask(counts, int(data.shape[-2]))
    _require_finite_rows(data, mask)
    norms = _active_row_norms(data, mask)
    deviations = jnp.abs(norms - 1.0)
    if norms.size and bool(jnp.any(deviations > tolerance)):
        worst = float(jnp.max(deviations))
        offending = int(jnp.count_nonzero(deviations > tolerance))
        message = (
            f"unit-norm evidence requires |‖x‖ - 1| <= {tolerance}; {offending} "
            f"active row(s) deviate, worst {worst}"
        )
        raise ValueError(message)
    return _mint_checked_clouds(
        clouds,
        support=UnitNormSupport(),
        pooled=Unchecked(),
        origin=CheckedEvidence(),
    )


def checked_nonzero_pooled_mean[
    DomainKind: ObjectDomain,
    SupportKind: FiniteValues,
    OriginKind: EvidenceOrigin,
](
    checked: CheckedClouds[DomainKind, SupportKind, ValueEvidence, OriginKind],
    *,
    atol: float = 0.0,
) -> CheckedClouds[DomainKind, SupportKind, NonzeroPooledMean, CheckedEvidence]:
    """Measure ``‖mean‖`` per cloud and mint the pooled-mean fact.

    The **only** producer of :class:`~jcor.core.domains.NonzeroPooledMean`. It
    consumes a carrier whose rows are already known finite and returns the same
    carrier with its row-level evidence preserved and its pooled axis filled in;
    the two axes never merge, because an antipodal unit cloud satisfies the
    first and fails this one.

    The mean is taken over each cloud's active rows only, with ``where``-based
    masking and the true counts from t62 as the denominator. The pooled vector
    itself is measured and discarded — storing it would add a third leaf.

    Args:
        checked: A carrier with at least :class:`~jcor.core.domains.FiniteValues`
            row evidence.
        atol: Norms at or below this bound are rejected; the default ``0.0``
            rejects only an exactly zero pooled mean, matching the existing
            ``ZeroNormMeanEmbeddingError`` boundary.

    Returns:
        The same carrier, with :class:`~jcor.core.domains.NonzeroPooledMean` on
        its pooled axis.

    Raises:
        TypeError: If the carrier's leaves are JAX tracers or ``atol`` is not a
            real number.
        ValueError: If ``atol`` is negative/nonfinite, a cloud is empty (an
            undefined mean), or a pooled mean is nonfinite or has norm at most
            ``atol``.

    """
    tolerance = _require_tolerance(atol)
    data, counts = _host_arrays(checked.clouds)
    if counts.size == 0 or bool(jnp.any(counts <= 0)):
        message = (
            "a pooled mean is undefined for an empty cloud; "
            f"got counts {counts.tolist()}"
        )
        raise ValueError(message)
    mask = _active_mask(counts, int(data.shape[-2]))
    contributions = jnp.where(
        mask[..., None],
        data,
        jnp.zeros((), dtype=data.dtype),
    )
    means = contributions.sum(axis=-2) / counts[..., None].astype(data.dtype)
    norms = jnp.linalg.norm(means, axis=-1)
    if not bool(jnp.all(jnp.isfinite(norms))):
        message = "pooled means must be finite; a cloud summed to NaN or infinity"
        raise ValueError(message)
    offending = int(jnp.count_nonzero(norms <= tolerance))
    if offending:
        message = (
            f"nonzero-pooled-mean evidence requires ‖mean‖ > {tolerance}; "
            f"{offending} cloud(s) pool to (near) zero — an antipodal cloud has "
            "unit rows and still averages to exactly zero"
        )
        raise ValueError(message)
    return _mint_checked_clouds(
        checked.clouds,
        support=checked.support,
        pooled=NonzeroPooledMean(),
        origin=CheckedEvidence(),
    )


def assume_cloud_evidence[
    DomainKind: ObjectDomain,
    SupportKind: ValueEvidence,
    PooledKind: ValueEvidence,
](
    domain: type[DomainKind],
    clouds: PackedClouds,
    *,
    support: SupportKind,
    pooled: PooledKind,
) -> CheckedClouds[DomainKind, SupportKind, PooledKind, UnsafeAssumption]:
    """Explicitly assert value evidence without checking or computing it.

    This is the only public escape hatch around the eager doors.  Its return
    type permanently advertises :class:`UnsafeAssumption`, so a consumer can
    refuse assumed evidence even when the support marker itself is compatible.
    """
    del domain
    return _mint_checked_clouds(
        clouds,
        support=support,
        pooled=pooled,
        origin=UnsafeAssumption(),
    )


def computed_l2_normalize_clouds(
    checked: CheckedClouds[
        NonzeroVector,
        NonzeroNormalizationDomain,
        ValueEvidence,
        EvidenceOrigin,
    ],
) -> CheckedClouds[
    UnitSphereValue,
    UnitNormSupport,
    Unchecked,
    ComputedEvidence,
]:
    """Normalize active rows and mint L2-unit evidence by construction.

    The operation is intentionally readable and eager. It checks the actual
    values even when its input came through the unsafe door, rejects integral
    storage rather than silently changing dtype, writes zero padding, and
    erases pooled-mean evidence because rowwise normalization changes a cloud's
    mean.
    """
    data, counts = _host_arrays(checked.clouds)
    if not jnp.issubdtype(data.dtype, jnp.floating):
        message = (
            "computed L2 normalization requires a floating carrier so its dtype "
            f"is explicit; got {data.dtype}"
        )
        raise TypeError(message)
    mask = _active_mask(counts, int(data.shape[-2]))
    _require_finite_rows(data, mask)
    norms = _active_row_norms(data, mask)
    if bool(jnp.any(norms <= 0.0)):
        message = "computed L2 normalization is undefined on an active zero row"
        raise ValueError(message)
    # Was a mutable masked write into a NumPy buffer. `.at[mask].set(...)` is the
    # immutable equivalent and writes the same active rows, leaving padding zero.
    unit_rows = data[mask] / norms[:, None].astype(data.dtype)
    normalized = jnp.zeros_like(data).at[mask].set(unit_rows)
    clouds = PackedClouds(data=normalized, counts=checked.clouds.counts)
    return _mint_checked_clouds(
        clouds,
        support=UnitNormSupport(),
        pooled=Unchecked(),
        origin=ComputedEvidence(),
    )


def select_checked_clouds[
    DomainKind: ObjectDomain,
    SupportKind: ValueEvidence,
    PooledKind: ValueEvidence,
    OriginKind: EvidenceOrigin,
](
    checked: CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind],
    indices: tuple[int, ...],
) -> CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind]:
    """Select whole unbatched clouds, preserving every per-cloud fact."""
    if (
        checked.clouds.data.ndim != _UNBATCHED_PACKED_NDIM
        or checked.clouds.counts.ndim != 1
    ):
        message = "select_checked_clouds currently requires an unbatched carrier"
        raise ValueError(message)
    k = int(checked.clouds.data.shape[0])
    if any(index < 0 or index >= k for index in indices):
        message = f"cloud indices must lie in [0, {k}), got {indices}"
        raise IndexError(message)
    index = jnp.asarray(indices, dtype=jnp.int32)
    clouds = PackedClouds(
        data=jnp.take(checked.clouds.data, index, axis=0),
        counts=jnp.take(checked.clouds.counts, index, axis=0),
    )
    return _mint_checked_clouds(
        clouds,
        support=checked.support,
        pooled=checked.pooled,
        origin=checked.origin,
    )


def permute_checked_rows[
    DomainKind: ObjectDomain,
    SupportKind: ValueEvidence,
    PooledKind: ValueEvidence,
    OriginKind: EvidenceOrigin,
](
    checked: CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind],
    permutations: ArrayLike,
) -> CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind]:
    """Permute rows within each active prefix, preserving support and means."""
    data, counts = _host_arrays(checked.clouds)
    if data.ndim != _UNBATCHED_PACKED_NDIM or counts.ndim != 1:
        message = "permute_checked_rows currently requires an unbatched carrier"
        raise ValueError(message)
    order = jnp.asarray(permutations)
    expected_shape = data.shape[:2]
    if order.shape != expected_shape or not jnp.issubdtype(order.dtype, jnp.integer):
        message = (
            f"permutations must be an integer array of shape {expected_shape}, "
            f"got {order.shape} and {order.dtype}"
        )
        raise ValueError(message)
    m_max = int(data.shape[1])
    for cloud_index, count in enumerate(counts.tolist()):
        active = int(count)
        if sorted(order[cloud_index, :active].tolist()) != list(range(active)):
            message = f"cloud {cloud_index} active indices are not a permutation"
            raise ValueError(message)
        if sorted(order[cloud_index, active:].tolist()) != list(range(active, m_max)):
            message = f"cloud {cloud_index} padding indices must remain in padding"
            raise ValueError(message)
    # The per-cloud gather replaces a `np.empty_like` buffer filled row block by
    # row block. `take_along_axis` expresses the same permutation as one op over
    # the whole batch, after the loop above has validated every cloud's order.
    permuted = jnp.take_along_axis(data, order[..., None], axis=1)
    clouds = PackedClouds(data=permuted, counts=checked.clouds.counts)
    return _mint_checked_clouds(
        clouds,
        support=checked.support,
        pooled=checked.pooled,
        origin=checked.origin,
    )


def concatenate_checked_clouds[
    DomainKind: ObjectDomain,
    SupportKind: ValueEvidence,
    PooledKind: ValueEvidence,
    OriginKind: EvidenceOrigin,
](
    left: CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind],
    right: CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind],
) -> CheckedClouds[DomainKind, SupportKind, PooledKind, OriginKind]:
    """Concatenate unbatched clouds carrying identical static evidence."""
    if (left.support, left.pooled, left.origin) != (
        right.support,
        right.pooled,
        right.origin,
    ):
        message = "concatenation preserves evidence only when metadata are identical"
        raise TypeError(message)
    left_data, left_counts = _host_arrays(left.clouds)
    right_data, right_counts = _host_arrays(right.clouds)
    if (
        left_data.ndim != _UNBATCHED_PACKED_NDIM
        or right_data.ndim != _UNBATCHED_PACKED_NDIM
    ):
        message = "concatenate_checked_clouds currently requires unbatched carriers"
        raise ValueError(message)
    if left_data.shape[-1] != right_data.shape[-1]:
        message = "concatenated carriers must share a feature width"
        raise ValueError(message)
    rows = [
        jnp.asarray(data[index, : int(count)])
        for data, counts in ((left_data, left_counts), (right_data, right_counts))
        for index, count in enumerate(counts.tolist())
    ]
    clouds = pack(rows)
    return _mint_checked_clouds(
        clouds,
        support=left.support,
        pooled=left.pooled,
        origin=left.origin,
    )


def forget_cloud_evidence(
    checked: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
) -> PackedClouds:
    """Explicitly erase every semantic claim while retaining the arrays."""
    return checked.clouds


def scale_checked_clouds(
    checked: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
    factor: float,
) -> PackedClouds:
    """Scale values and conservatively erase all evidence, even for factor one."""
    return PackedClouds(
        data=checked.clouds.data * factor,
        counts=checked.clouds.counts,
    )


def add_checked_clouds(
    left: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
    right: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
) -> PackedClouds:
    """Add aligned values and erase all evidence."""
    if left.clouds.data.shape != right.clouds.data.shape:
        message = "added checked carriers must have equal data shapes"
        raise ValueError(message)
    left_counts, right_counts = jax.device_get(
        (left.clouds.counts, right.clouds.counts)
    )
    if not bool(jnp.array_equal(left_counts, right_counts)):
        message = "added checked carriers must have equal active row counts"
        raise ValueError(message)
    return PackedClouds(
        data=left.clouds.data + right.clouds.data,
        counts=left.clouds.counts,
    )


def truncate_checked_features(
    checked: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
    width: int,
) -> PackedClouds:
    """Truncate feature coordinates and erase support/pooled evidence."""
    if isinstance(width, bool) or not isinstance(width, int):
        message = f"feature width must be an integer, got {type(width).__name__}"
        raise TypeError(message)
    d = int(checked.clouds.data.shape[-1])
    if width <= 0 or width > d:
        message = f"feature width must lie in [1, {d}], got {width}"
        raise ValueError(message)
    return PackedClouds(
        data=checked.clouds.data[..., :width],
        counts=checked.clouds.counts,
    )


def project_checked_clouds(
    checked: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
    projection: jax.Array,
) -> PackedClouds:
    """Apply a general linear projection and erase every value-law claim."""
    if (
        projection.ndim != _MATRIX_NDIM
        or projection.shape[0] != checked.clouds.data.shape[-1]
    ):
        message = (
            "projection must have shape (input_width, output_width), got "
            f"{projection.shape} for input width {checked.clouds.data.shape[-1]}"
        )
        raise ValueError(message)
    return PackedClouds(
        data=checked.clouds.data @ projection,
        counts=checked.clouds.counts,
    )


def pool_checked_clouds(
    checked: CheckedClouds[
        ObjectDomain,
        ValueEvidence,
        ValueEvidence,
        EvidenceOrigin,
    ],
) -> jax.Array:
    """Return active-row means as bare arrays, carrying no rowwise evidence."""
    data = checked.clouds.data
    counts = checked.clouds.counts
    mask = jnp.arange(data.shape[-2]) < counts[..., None]
    total = jnp.sum(jnp.where(mask[..., None], data, 0), axis=-2)
    return total / counts[..., None]


def declare_population_hypotheses[DomainKind: ObjectDomain, Order: MomentOrder](
    domain: type[DomainKind],
    *,
    support: AlmostSureNonzeroSupport,
    moment: FiniteMoment[Order],
) -> PopulationHypotheses[DomainKind, Order]:
    """Record population conditions the caller asserts, checking nothing.

    Named for what it is. No sample can establish either argument, so this
    function inspects no data and has no eager counterpart; the caller owns the
    justification exactly as it owns a law declaration in
    :mod:`jcor.core.axioms`.

    Args:
        domain: The object domain the law lives on, supplied as the class so the
            phantom parameter is inferred rather than annotated at every call.
        support: Declared almost-sure nonzero-support policy for ``N#P``.
        moment: Declared finite-moment condition at the order in use. The order
            is *not* checked against a ground exponent here — ``sample`` is rank
            1 and cannot see the ``discrepancy`` exponent regime, so that
            compatibility check belongs at the ``discrepancy`` door.

    Returns:
        The declared hypotheses, carried statically.

    Examples:
        >>> from jcor.core.domains import FirstMomentOrder, PopulationLaw
        >>> declared = declare_population_hypotheses(
        ...     PopulationLaw,
        ...     support=AlmostSureNonzeroSupport(),
        ...     moment=FiniteMoment[FirstMomentOrder](),
        ... )
        >>> declared.support
        AlmostSureNonzeroSupport()

    """
    del domain
    return PopulationHypotheses(support=support, moment=moment)

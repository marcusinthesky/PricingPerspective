r"""Opaque DISCO component types: a typed decomposition with its provenance.

Position
--------
rank 4 · association. Types only, plus the one typed door that mints them.

What this module is for
-----------------------
:mod:`jcor.association.dispersion` owns only the traceable five-scalar numerical
kernel. This module is the public mathematical door: it validates the grouped
design, requires a law- and origin-bearing matrix, and returns typed DISCO
components. Calibration composes above it in :mod:`jcor.inference.disco`.

The three prohibitions, and where each is enforced
--------------------------------------------------
Acceptance for t55.3 asks that a DISCO decomposition cannot be built from
arbitrary scalars or an incompatible matrix, and that its statistic cannot
stand in for a distance or for a calibrated test. All three are **static**:

* **Arbitrary scalars.** Every field of :class:`DiscoDecomposition` demands a
  :class:`~jcor.core.statistics.RealizedEstimate` at its own ``@final``
  estimand, or a :class:`~jcor.core.statistics.RealizedStatistic` at its own
  ``@final`` kind. A bare ``float``, a raw array, and a within-dispersion
  estimate supplied as the total are three distinct ``bad-argument-type``\ s.
* **An incompatible matrix.** :func:`disco_components` consumes
  ``DMat[Domain, DispersionDecomposableLaw, Origin]``. A
  :class:`~jcor.core.matrices.PairwiseEstimateMatrix` is not a ``DMat`` at all;
  a ``MetricLaw`` matrix is short the negative-type capability; and the output
  of :func:`~jcor.core.matrices.checked_dmat` carries only
  :class:`~jcor.core.matrices.ObservedSquareLaw`, so an observed nonnegative
  symmetric array cannot be promoted into a dispersion decomposition. That last
  rejection is node defect 3 — an observed finite-sample matrix is not a
  population property — made unwritable rather than documented.
* **Substitution.** The index and the F-ratio are
  :class:`~jcor.core.statistics.RealizedStatistic`, which carries no law
  parameter and cannot acquire one: :data:`~jcor.core.statistics.StatisticT` is
  bounded at :class:`~jcor.core.statistics.StatisticKind`, so
  ``RealizedStatistic[float, MetricLaw, ...]`` is a ``bad-specialization``.
  And this decomposition is **pure** — it holds no p-value, no null sample and
  no calibration axis, so it never satisfies a
  :class:`~jcor.core.statistics.CalibratedTest` requirement.

Why negative type rides in the type and not in ``@requires``
------------------------------------------------------------
:math:`S_B = T - S_W` is nonnegative because the ground dissimilarity is of
negative type, not because its numbers happened to come out nonnegative on one
sample. Negative type is a :class:`~jcor.core.axioms.LawProperties` member, not
an :class:`~jcor.core.axioms.Axioms` flag, so it cannot be spelled in
``@requires``; the decorator therefore declares the strictly smaller,
observable half of the precondition, exactly as
:func:`~jcor.core.matrices.metrize_energy` does. Do not "complete" it with a
runtime check: inventing one for a population property is the error t55.3
exists to prevent.

What this module is *not*
-------------------------
It owns no calibration. There is no p-value, no permutation null, and no
``Hypothesis`` or ``Calibration`` parameter anywhere below: ``inference`` owns
exact, Monte-Carlo add-one, and asymptotic calibration, and composes
:class:`~jcor.core.statistics.CalibratedTest` around the F-ratio this module
hands it. It delegates arithmetic to
:func:`jcor.association.dispersion.disco_decomposition`; the split is a normal
kernel/contract boundary, not two competing public procedures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol

import jax
import jax.numpy as jnp

from jcor.association.dispersion import disco_decomposition
from jcor.core.axioms import (  # runtime: Protocol bases + decorator
    Axioms,
    NegativeTypeLaw,
    PremetricLaw,
    Symmetric,
    requires,
)
from jcor.core.domains import (  # runtime: Generic bases and bounds
    ConstructionOrigin,
    DomainT,
    ObjectDomain,
    OriginT,
)
from jcor.core.matrices import DMat  # noqa: TC001  # runtime: annotation is checked
from jcor.core.statistics import (  # noqa: TC001  # runtime: Generic bases and fields
    AdditiveDispersionDecomposition,
    DecompositionOrigin,
    DispersionIndex,
    EstimateProvenance,
    GroupDesign,
    RealizedEstimate,
    RealizedStatistic,
    StatisticProvenance,
    ValueT,
    VarianceRatioStatistic,
    VStatisticEstimator,
)
from jcor.core.typing import Array, ArrayLike, Float  # noqa: TC001  # see typing

__all__ = [
    "DiscoDecomposition",
    "DiscoOrigin",
    "DispersionDecomposableLaw",
    "disco_components",
    "group_design",
]

_MIN_GROUPS = 2


# --- The law a dispersion decomposition may be taken over. ---


class DispersionDecomposableLaw(PremetricLaw, Symmetric, NegativeTypeLaw, Protocol):
    r"""Nonnegative, zero-diagonal, symmetric, and of negative type.

    The first three are what the arithmetic reads: :math:`\bar a_{gg}` is a
    V-statistic that keeps the zero diagonal and divides by :math:`n_g^2`, and
    the masked bilinear form ``mask @ d @ mask`` counts each ordered pair once
    in each direction. The fourth is what makes the *result* meaningful:
    without conditional negative definiteness of the ground there is no reason
    for :math:`S_B = T - S_W` to be nonnegative, and a negative between-group
    dispersion is not a dispersion at all.

    Structural, like every law in :mod:`jcor.core.axioms`, so
    :class:`~jcor.core.matrices.HilbertianMetricLaw` — what
    :func:`~jcor.core.matrices.metrize_energy` produces — satisfies it in one
    step. A bare :class:`~jcor.core.axioms.MetricLaw` does not: a genuine
    metric that separates every pair of points still need not be of negative
    type, which is the ``l1`` counterexample recorded as node defect 6.
    """


# --- Construction origin. A refinement of the generic decomposition origin. ---


class DiscoOrigin(DecompositionOrigin, Generic[OriginT]):  # noqa: UP046  # see `OriginT`
    """The DISCO decomposition of a dissimilarity built by ``OriginT``.

    Generic in the origin of the matrix that was decomposed, mirroring
    :class:`jcor.core.matrices.EnergyFunctional`: the provenance of the input
    survives the decomposition rather than being flattened into "some
    decomposition". A DISCO over a rooted energy matrix and a DISCO over a
    transport matrix are therefore different types, and a consumer written for
    one does not silently accept the other.

    Widens to :class:`~jcor.core.statistics.DecompositionOrigin` and thence to
    :class:`~jcor.core.domains.ConstructionOrigin`, so one origin parameter
    still carries matrix and scalar provenance through the whole lattice.
    """


#: The two statistic descriptors this module produces, named once. Carriers take
#: descriptors (see :class:`jcor.core.statistics.StatisticProvenance`), so the
#: kind and the origin are combined here rather than spelled at each field, and
#: an axis added to ``StatisticProvenance`` changes these two lines only.
type DiscoIndexProvenance[OriginA: ConstructionOrigin] = StatisticProvenance[
    DispersionIndex,
    DiscoOrigin[OriginA],
]
type DiscoFRatioProvenance[OriginA: ConstructionOrigin] = StatisticProvenance[
    VarianceRatioStatistic,
    DiscoOrigin[OriginA],
]


def group_design(codes: ArrayLike, *, n_groups: int | None = None) -> GroupDesign:
    """Validate contiguous group codes at an eager boundary."""
    host = jnp.asarray(codes)
    if host.ndim != 1 or host.size == 0:
        message = "group codes must be a nonempty vector"
        raise ValueError(message)
    if not jnp.issubdtype(host.dtype, jnp.integer):
        message = "group codes must be integers"
        raise TypeError(message)
    # Deliberate host guard (t65 D1/P4): this `unique` exists to `raise` on a
    # non-contiguous encoding, and raise-semantics are not a jittable kernel.
    # t65.3 moved it to `jnp` without changing that: `jnp.unique` returns a
    # data-dependent length, which is legal eagerly and would fail under `jit` —
    # the guard is still a door, it simply no longer needs NumPy to be one.
    unique = jnp.unique(host)
    expected = jnp.arange(unique.size)
    if not bool(jnp.array_equal(unique, expected)):
        message = "group codes must be contiguous from zero"
        raise ValueError(message)
    resolved = int(unique.size) if n_groups is None else n_groups
    if resolved != unique.size or resolved < _MIN_GROUPS or resolved >= host.size:
        message = "group design requires 2 <= n_groups < n with matching codes"
        raise ValueError(message)
    return GroupDesign(codes=host, n_groups=resolved)


# --- The decomposition. Pure: estimates and statistics, and nothing else. ---


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscoDecomposition(
    Generic[ValueT, DomainT, OriginT],  # noqa: UP046  # PEP 695 loses variance
):
    r"""``T = S_W + S_B`` with the DISCO index and F-ratio, all typed.

    Composition of :class:`~jcor.core.statistics.AdditiveDispersionDecomposition`
    with the two derived statistics, exactly as that class's docstring reserves.
    The estimator scheme is pinned to
    :class:`~jcor.core.statistics.VStatisticEstimator` because the cited DISCO
    :math:`g_\alpha(A, A)` keeps its zero diagonal and divides by
    :math:`n_g^2`; a diagonal-debiased U-statistic is a different quantity and
    has no route into this container.

    ``DomainT`` and ``OriginT`` are phantom: they record which matrix was
    decomposed without adding a field, so ``tree_leaves`` sees exactly the six
    numeric leaves — the three estimates, the group codes, the index, and the
    F-ratio. No estimand, provenance, or law object can reach a compiled array
    tree, because none of them is a field anywhere in this tree.

    There is deliberately **no** p-value, null sample, or calibration axis:
    this carrier is what ``inference`` calibrates, not the calibrated result.

    Attributes:
        components: The additive ``T = S_W + S_B`` estimates and the grouping.
        index: The DISCO index :math:`R^2_E = S_B / T`, a share of total
            dispersion.
        f_ratio: The DISCO F-ratio
            :math:`[S_B/(K-1)] / [S_W/(N-K)]`, a degrees-of-freedom-scaled
            ratio and a *sibling* kind of the index, so neither may be passed
            where the other is required.

    """

    components: AdditiveDispersionDecomposition[
        ValueT, VStatisticEstimator, DiscoOrigin[OriginT]
    ]
    index: RealizedStatistic[ValueT, DiscoIndexProvenance[OriginT]]
    f_ratio: RealizedStatistic[ValueT, DiscoFRatioProvenance[OriginT]]


# Flags, never alias names (see `requires`): the observable, finite-sample half
# of the contract. It is deliberately NOT the whole precondition —
# `DispersionDecomposableLaw` additionally demands negative type, which is a
# `LawProperties` member and has no `Axioms` spelling, and which no inspection
# of one observed matrix could establish anyway (node §Requirements 7, defect 3).
# `metrize_energy` splits its contract the same way and for the same reason.
@requires(Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY)
def disco_components[DomainA: ObjectDomain, OriginA: ConstructionOrigin](
    matrix: DMat[DomainA, DispersionDecomposableLaw, OriginA],
    design: GroupDesign,
) -> DiscoDecomposition[Float[Array, ...], DomainA, OriginA]:  # jaxtyping
    """Decompose a branded dissimilarity over a grouped design, keeping provenance.

    The typed door onto the DISCO components. It performs no arithmetic of its
    own: :func:`jcor.association.dispersion.disco_decomposition` is called
    unchanged, so Paper 1's committed numbers are bit-identical, and this
    function's content is the signature — which matrices may be decomposed, and
    what the resulting numbers are allowed to be mistaken for.

    The input matrix's domain and origin are preserved into the result, the
    latter under :class:`DiscoOrigin`, so a decomposition of a rooted energy
    matrix stays distinguishable from a decomposition of a transport matrix.

    Args:
        matrix: The square dissimilarity, of negative type, to decompose.
        design: The group labels, with the group count as static metadata.

    Returns:
        The typed decomposition: total, within, and between estimates, plus the
        DISCO index and F-ratio, at ``DiscoOrigin[OriginA]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jcor.core.matrices import unsafe_assume_dmat
        >>> d = 1.0 - jnp.eye(4)
        >>> decomposition = disco_components(
        ...     unsafe_assume_dmat(d),
        ...     GroupDesign(codes=jnp.array([0, 0, 1, 1]), n_groups=2),
        ... )
        >>> float(decomposition.components.total.value)
        1.5

    """
    within, between, total, index, f_ratio = disco_decomposition(
        matrix.values, design.codes, design.n_groups
    )
    return DiscoDecomposition(
        components=AdditiveDispersionDecomposition(
            total=RealizedEstimate(value=total, provenance=EstimateProvenance()),
            within=RealizedEstimate(value=within, provenance=EstimateProvenance()),
            between=RealizedEstimate(value=between, provenance=EstimateProvenance()),
            design=design,
        ),
        index=RealizedStatistic(value=index, provenance=StatisticProvenance()),
        f_ratio=RealizedStatistic(value=f_ratio, provenance=StatisticProvenance()),
    )

"""Opaque matrix containers, construction doors, and the energy scheme axis.

Position
--------
rank 0 · importable by every stage

What this module is for
-----------------------
:mod:`jcor.core.axioms` says *which axioms* a dissimilarity satisfies and
:mod:`jcor.core.domains` says *on what* they hold. Neither says **how the
numbers were built**, and that third question decides whether a square array may
be metrized at all. The energy functional ``S(P, Q)`` is not the measure metric;
``sqrt(S(P, Q))`` is, and only under strong negative type with the matching
finite moment. Its empirical V-statistic plug-in is nonnegative under negative
type, while a diagonal-debiased U-statistic estimate is **signed** — so it is
not a dissimilarity at all and must not be spelled as one.

Descriptors carry axes; carriers carry descriptors
--------------------------------------------------
:class:`SpaceDMat` names six semantic axes through **two** parameters, because
five of them travel together and are bundled into :class:`MatrixContract`. That
is the same move :data:`jcor.core.domains.L2UnitSphereSpace` makes for carrier,
equality and structure, and :data:`jcor.core.transforms.L2NormalizationContract`
makes for source, target, map, totality and origin. The rule it follows is
recorded in ``AGENTS.md`` and exercised by the public typing and contract tests:
a *carrier* —
a pytree holding arrays — takes at most three type parameters, one value type plus two
descriptors, and a new semantic axis is added inside a descriptor rather than
onto the carrier. Without that budget, generalising the package multiplies every
signature by every axis, which is the cost that turns a 27k-line library into a
180k-line one — and it is invisible one axis at a time, which is what makes it a
durable design convention.

Two containers, deliberately siblings
-------------------------------------
:class:`DMat` carries ``(Domain, Law, Origin)`` and means "a dissimilarity over
``Domain`` declared to satisfy ``Law``, built by ``Origin``".
:class:`PairwiseEstimateMatrix` carries ``(Domain, Estimand, Estimator,
Origin)`` and means "a realized pairwise *estimate*". It is **not** a subclass:
``mean_distance_matrix`` and a signed U-statistic MMD² produce values that may
be negative and whose diagonal has no reason to vanish, so no distance law
applies to them. Rooted V-statistic MMD and compatible energy-function matrices
stay on :class:`DMat`, with their estimator recorded in the origin.

Three doors, and what each may conclude
---------------------------------------
* :func:`_dmat` / :func:`_pairwise_estimate` — the private producer
  constructors. In-package producers that *know* what they built import these;
  they are not part of the public surface.
* :func:`checked_dmat` / :func:`checked_pairwise_estimate` — the checked
  external door for persisted or third-party arrays. It verifies **observable
  finite-sample axioms only** and therefore returns the widest possible brand:
  ``Domain=ObjectDomain``, ``Origin=ConstructionOrigin``, and a law limited to
  :class:`ObservedSquareLaw`. It cannot reconstruct how the array was built, it
  cannot promote an observed conditionally-negative-definite Gram to population
  strong negative type, and it cannot infer separation — separation is a
  statement about *indexed objects* under a domain-aware equality, and this door
  sees only numbers.
* :func:`unsafe_assume_dmat` / :func:`unsafe_assume_pairwise_estimate` — the
  single place an unsupported three-axis assertion is allowed to be written. The
  name is the review trigger.

The checked doors are **eager host doors**. They call ``bool()`` on array
comparisons, which raises ``TracerBoolConversionError`` under a transform, for
the same reason ``core/results.py`` runs no ``__post_init__`` validation and
``discrepancy/metrize.py`` splits its JAX and NumPy forms. Validate before you
enter ``jit``, never inside it.

Construction enforcement
------------------------
The public dataclass constructors require a private, static producer token.
JAX carries that singleton in the treedef and supplies it again during
unflatten, so the carrier remains a one-array-leaf pytree while ordinary code
cannot write ``DMat(values=...)``. In-package producers use the private doors;
external assertions use the loudly named ``unsafe_assume_*`` doors.

Migration note
--------------
This supersedes the one-axis :class:`jcor.core.typing.DMat`, whose public
constructor lets any caller assert a brand. Both names exist during t55.4's
consumer migration; import the three-axis container from **this** module by its
full path when the distinction matters. T55.5 removes the legacy container along
with the nominal ``Premetric`` marker chain it is parameterized by.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar, final

import jax
import jax.numpy as jnp

from jcor.core.axioms import (  # noqa: TC001  # runtime: Protocol bounds + Generic bases
    Axioms,
    Law,
    MetricLaw,
    NegativeTypeLaw,
    PremetricLaw,
    PseudometricLaw,
    StrongNegativeTypeLaw,
    Symmetric,
    requires,
)
from jcor.core.domains import (  # noqa: TC001  # runtime: Generic bases and bounds
    AuthorityT,
    ConstructionOrigin,
    DomainLaw,
    DomainT,
    EnergyOrigin,
    EvidenceOriginT,
    FiniteMoment,
    MomentOrder,
    ObjectDomain,
    OriginT,
    SpaceDescriptor,
    SpaceT,
    ValueEvidence,
)
from jcor.core.transforms import TransformDescriptor
from jcor.core.typing import (  # noqa: TC001  # runtime; see jcor.core.typing docstring
    Array,
    Float,
    LawT,
)

__all__ = [
    "AsymmetricMatrixError",
    "CheckedMatrixError",
    "DMat",
    "EnergyFunctional",
    "EnergyScheme",
    "EstimandT",
    "EstimatorT",
    "HilbertianMetricLaw",
    "HilbertianPseudometricLaw",
    "MatrixContract",
    "MatrixContractT",
    "MatrixDescriptor",
    "MatrixShapeError",
    "NegativeEntryError",
    "NonfiniteMatrixError",
    "NonnegativeEnergyScheme",
    "NonzeroDiagonalError",
    "ObservedSquareLaw",
    "PairwiseEstimateMatrix",
    "PopulationScheme",
    "RootEnergyMetric",
    "SchemeT",
    "SpaceDMat",
    "SquaredDistanceNegativeTypeLaw",
    "UStatisticScheme",
    "VStatisticScheme",
    "checked_dmat",
    "checked_pairwise_estimate",
    "metrize_energy",
    "unsafe_assume_dmat",
    "unsafe_assume_pairwise_estimate",
    "unsafe_assume_space_dmat",
]


# --- The estimation scheme axis. Sign guarantee, not merely a label. ---


class EnergyScheme:
    """Root of the estimation-scheme axis for energy-type constructions.

    Splits by the property downstream code actually depends on — whether the
    produced values are guaranteed nonnegative — rather than by the name of the
    estimator.
    """


class NonnegativeEnergyScheme(EnergyScheme):
    """Schemes whose values are nonnegative under negative type.

    The bound on :data:`SchemeT`, hence the gate on :class:`EnergyFunctional`.
    Only these schemes may be spelled as a dissimilarity and rooted into a
    measure metric.
    """


@final
class PopulationScheme(NonnegativeEnergyScheme):
    """The exact population functional ``S(P, Q)``, not an estimate."""


@final
class VStatisticScheme(NonnegativeEnergyScheme):
    """The empirical V-statistic plug-in.

    Biased, and nonnegative under negative type of the ground. A *sibling* of
    :class:`PopulationScheme`: a plug-in on a finite sample is not the
    population functional, and node defect 3 is exactly the promotion that must
    stay unavailable.
    """


@final
class UStatisticScheme(EnergyScheme):
    """The diagonal-debiased U-statistic estimate, which may be **signed**.

    An unrelated sibling of :class:`NonnegativeEnergyScheme`, and that is the
    whole point: it can never appear inside :class:`EnergyFunctional`, so
    ``DMat[..., MetricLaw, EnergyFunctional[UStatisticScheme]]`` is a static
    error at the type application rather than a convention nobody enforces. A
    U-statistic estimate lives in :class:`PairwiseEstimateMatrix`.
    """


#: **Covariant** nonnegative estimation scheme. Bounded at
#: :class:`NonnegativeEnergyScheme`, which is what excludes the signed
#: U-statistic from every dissimilarity-shaped container. Explicit ``TypeVar``
#: for the reason given in :mod:`jcor.core.domains`: phantom parameters give
#: PEP 695 nothing to infer variance from.
SchemeT = TypeVar("SchemeT", bound=NonnegativeEnergyScheme, covariant=True)  # noqa: PLC0105

#: **Covariant** estimand — *what* is being estimated. Deliberately unbounded
#: until ``core/statistics.py`` lands the estimand root; the bound tightens
#: there without changing any call site, because every marker it defines is
#: already assignable to the unbounded parameter.
EstimandT = TypeVar("EstimandT", covariant=True)  # noqa: PLC0105

#: **Covariant** estimator scheme — *how* it was estimated. Unbounded for the
#: same reason as :data:`EstimandT`; :class:`UStatisticScheme` and
#: :class:`VStatisticScheme` are the two markers this module contributes.
EstimatorT = TypeVar("EstimatorT", covariant=True)  # noqa: PLC0105
MatrixTransformT = TypeVar(  # noqa: PLC0105
    "MatrixTransformT",
    bound=TransformDescriptor,
    covariant=True,
)
MatrixEvidenceT = TypeVar(  # noqa: PLC0105
    "MatrixEvidenceT",
    bound=ValueEvidence,
    covariant=True,
)


# --- The matrix contract. Five axes here, so the carrier can take one token. ---


@dataclass(frozen=True, slots=True)
class MatrixDescriptor:
    """Non-generic root for a complete static matrix claim-and-warrant."""


class MatrixContract(
    MatrixDescriptor,
    Generic[  # noqa: UP046  # explicit TypeVars retain covariance
        LawT,
        MatrixTransformT,
        AuthorityT,
        MatrixEvidenceT,
        EvidenceOriginT,
    ],
):
    """What is claimed of a matrix's values, and on what warrant.

    The five axes are the law itself, the transform that built the values, the
    authority behind the law, the input evidence that licensed the computation,
    and how that evidence was obtained. They travel together at every call site
    that mentions any of them, which is why they are one descriptor rather than
    five carrier parameters.

    **Descriptors carry axes; carriers carry descriptors.** A new axis is added
    here and changes no carrier signature, no producer arity, and no annotation
    that names the combination through a ``type`` alias. That is the property
    the parameter budget in ``AGENTS.md`` exists to keep, and it is why
    :class:`SpaceDMat` takes two parameters rather than six.

    Every parameter is covariant, so the sound widenings still compose in one
    step: a metric contract satisfies a pseudometric requirement, and checked
    unit-norm evidence satisfies a finite-values requirement, exactly as when
    the axes sat directly on the carrier.
    """


#: **Covariant** matrix contract, the second parameter :class:`SpaceDMat` takes.
MatrixContractT = TypeVar(  # noqa: PLC0105
    "MatrixContractT",
    bound=MatrixDescriptor,
    covariant=True,
)


# --- Energy construction origins. Refinements of `EnergyOrigin`. ---


class EnergyFunctional(EnergyOrigin, Generic[SchemeT]):  # noqa: UP046  # see `SchemeT`
    """The energy functional ``S(P, Q)`` itself, at a nonnegative scheme.

    Nonnegative under negative type of the ground, and **not** the measure
    metric: it fails the triangle inequality, which is what :func:`metrize_energy`
    repairs. Widens to :class:`jcor.core.domains.EnergyOrigin`, so a generic
    energy consumer accepts it, while a transport or direct origin never
    satisfies a requirement written as ``EnergyFunctional[...]`` no matter how
    strong its declared law is.
    """


class RootEnergyMetric(EnergyOrigin, Generic[SchemeT]):  # noqa: UP046  # see `SchemeT`
    """``sqrt(S(P, Q))`` — the rooted energy, at the scheme it was rooted from.

    A *sibling* of :class:`EnergyFunctional`, never a subtype in either
    direction: a consumer wanting the functional must not be handed its root,
    and a consumer wanting the metric must not be handed the un-rooted
    semimetric. The scheme parameter is preserved by :func:`metrize_energy`, so
    a rooted V-statistic stays distinguishable from a rooted population
    functional.
    """


# --- Laws this module needs and `core/axioms.py` does not already name. ---


class ObservedSquareLaw(PremetricLaw, Symmetric, Protocol):
    """Exactly what a checked numeric door can verify on a finite matrix.

    Nonnegative, zero-diagonal, and symmetric. Deliberately **not** separating:
    ``d(i, j) == 0`` for ``i != j`` is only an error once you know the objects
    at ``i`` and ``j`` differ under a domain-aware equality, and the door sees
    no objects. Deliberately not of negative type either — that is computable
    on the observed matrix, but observed conditional negative definiteness on
    one finite sample is not population strong negative type (node defect 3).
    """


class SquaredDistanceNegativeTypeLaw(Law, Protocol):
    """Capability required by classical scaling: ``D²`` is CND."""

    @property
    def squared_distance_negative_type(self) -> Literal[True]:
        """Declare that ``D²`` is conditionally negative definite."""
        ...


class HilbertianPseudometricLaw(
    PseudometricLaw,
    NegativeTypeLaw,
    SquaredDistanceNegativeTypeLaw,
    Protocol,
):
    """A possibly nonseparating Hilbert-norm pullback."""


class HilbertianMetricLaw(
    MetricLaw,
    NegativeTypeLaw,
    SquaredDistanceNegativeTypeLaw,
    Protocol,
):
    """A metric that is additionally of negative type — a Hilbert embedding.

    What :func:`metrize_energy` produces: on a ground of strong negative type,
    ``sqrt(S)`` is the norm distance in the associated Hilbert space, hence a
    metric on measures *and* of negative type. Note the asymmetry the return
    type keeps: the root is of ordinary negative type; **strong** negative type
    of the root is not claimed.
    """


# --- The containers. One array leaf each; every other axis is phantom. ---


@final
@dataclass(frozen=True, slots=True)
class _ProducerToken:
    """Private constructor token carried as static pytree metadata."""


_PRODUCER_TOKEN = _ProducerToken()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DMat(Generic[DomainT, LawT, OriginT]):  # noqa: UP046  # PEP 695 loses variance
    """A square dissimilarity branded with its domain, law, and construction.

    All three parameters are *phantom*: they appear in no field, cost nothing at
    runtime, and never reach a JAX transform. ``tree_leaves`` sees exactly one
    leaf, the matrix, so ``jit`` and ``vmap`` are unaffected.

    Covariance is inherited verbatim from the axes rather than re-derived, so
    the sound widenings compose in one step: a matrix over checked unit-sphere
    values, declared a metric, built by an energy functional, satisfies a
    requirement for a nonzero-vector pseudometric of generic energy origin. The
    reverse never type-checks, and neither does swapping a transport origin in
    where an energy origin is required.

    Attributes:
        values: The dissimilarity, shape ``(n, n)`` — or ``(*batch, n, n)`` once
            a transform has batched it. The shape lives on this field; the
            semantics live on the class parameters. Never nest them.

    Examples:
        >>> from jcor.core.axioms import MetricLaw
        >>> from jcor.core.domains import DirectOrigin, UnitSphereValue
        >>> d: DMat[UnitSphereValue, MetricLaw, DirectOrigin] = unsafe_assume_dmat(
        ...     jnp.zeros((3, 3))
        ... )
        >>> d.values.shape
        (3, 3)

    """

    values: Float[Array, "*batch n n"]  # noqa: F722  # jaxtyping shape, not a forward ref
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpaceDMat(
    Generic[SpaceT, MatrixContractT],  # noqa: UP046  # explicit TypeVars retain variance
):
    """A matrix over one mathematical space, at one declared contract.

    Two parameters, carrying six axes. ``SpaceT`` is *what the values are over*
    — carrier, equality and structure together — and
    :class:`MatrixContract` is *what is claimed about them and why*: law,
    transform, authority, input evidence, and how that evidence was obtained.

    Unlike the migration-era :class:`DMat`, the space includes the equality and
    carrier structure. Every axis is phantom; ``values`` remains the only
    numerical leaf, so ``tree_leaves`` sees exactly one array and ``jit``/
    ``vmap`` are unaffected.

    Examples:
        >>> from jcor.core.axioms import MetricLaw
        >>> from jcor.core.domains import (
        ...     CheckedEvidence,
        ...     GovernedReferenceAuthority,
        ...     L2UnitSphereSpace,
        ...     UnitNormSupport,
        ... )
        >>> from jcor.core.transforms import L2NormalizationContract
        >>> type Chord = MatrixContract[
        ...     MetricLaw,
        ...     L2NormalizationContract,
        ...     GovernedReferenceAuthority,
        ...     UnitNormSupport,
        ...     CheckedEvidence,
        ... ]
        >>> d: SpaceDMat[L2UnitSphereSpace, Chord] = unsafe_assume_space_dmat(
        ...     jnp.zeros((3, 3))
        ... )
        >>> d.values.shape
        (3, 3)

    """

    values: Float[Array, "*batch n n"]  # noqa: F722  # jaxtyping shape
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PairwiseEstimateMatrix(
    Generic[DomainT, EstimandT, EstimatorT, OriginT],  # noqa: UP046  # loses variance
):
    """A realized pairwise estimate — a **sibling** of :class:`DMat`, not a subtype.

    Carries no distance law, because a signed realized estimate is not a
    dissimilarity: a diagonal-debiased U-statistic MMD² is routinely negative,
    and a mean-distance matrix has no reason to vanish on its diagonal.
    Declaring either a ``DMat`` would let it flow into a consumer that assumes
    nonnegativity, a zero diagonal, or metrizability.

    The values are ``(*batch, m, n)`` rather than square: the container must be
    able to hold a rectangular block of cross-cloud estimates without a shape
    that implies matrix-level symmetry.

    Attributes:
        values: The realized estimates, shape ``(m, n)`` or ``(*batch, m, n)``.

    Examples:
        >>> from jcor.core.domains import EmpiricalDistribution, MmdOrigin
        >>> squared_mmd: PairwiseEstimateMatrix[
        ...     EmpiricalDistribution, object, UStatisticScheme, MmdOrigin
        ... ] = unsafe_assume_pairwise_estimate(jnp.zeros((2, 2)))
        >>> float(squared_mmd.values.sum())
        0.0

    """

    values: Float[Array, "*batch m n"]  # noqa: F722  # jaxtyping shape, not a forward ref
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


# --- Typed errors raised by the checked doors. Eager host boundary only. ---


class CheckedMatrixError(ValueError):
    """Base class for every rejection raised by a checked matrix door."""


class MatrixShapeError(CheckedMatrixError):
    """The array is not a matrix, or not square where squareness is required."""


class NonfiniteMatrixError(CheckedMatrixError):
    """The array contains ``NaN`` or ``+/-inf``."""


class NegativeEntryError(CheckedMatrixError):
    """A dissimilarity entry is negative."""


class NonzeroDiagonalError(CheckedMatrixError):
    """A dissimilarity has a nonzero diagonal entry."""


class AsymmetricMatrixError(CheckedMatrixError):
    """A dissimilarity is not symmetric within the requested tolerance."""


# --- Doors. ---


def _dmat[DomainA: ObjectDomain, LawA: Law, OriginA: ConstructionOrigin](
    values: Float[Array, "*batch n n"],  # noqa: F722  # jaxtyping shape
) -> DMat[DomainA, LawA, OriginA]:
    """Construct a branded matrix from an in-package producer that knows its origin.

    The private producer door. The brand is read from the annotation at the call
    site, so a producer writes its own guarantee once, in its return type, and
    the value cannot be re-branded downstream without crossing
    :func:`unsafe_assume_dmat`.

    Args:
        values: The square dissimilarity the producer computed.

    Returns:
        The matrix branded with the parameters the call site requires.

    """
    return DMat(values=values, _token=_PRODUCER_TOKEN)


def _space_dmat[SpaceA: SpaceDescriptor, ContractA: MatrixDescriptor](
    values: Float[Array, "*batch n n"],  # noqa: F722  # jaxtyping shape
) -> SpaceDMat[SpaceA, ContractA]:
    """Private producer for a fully indexed mathematical matrix."""
    return SpaceDMat(values=values, _token=_PRODUCER_TOKEN)


def _pairwise_estimate[
    DomainA: ObjectDomain,
    EstimandA,
    EstimatorA,
    OriginA: ConstructionOrigin,
](
    values: Float[Array, "*batch m n"],  # noqa: F722  # jaxtyping shape
) -> PairwiseEstimateMatrix[DomainA, EstimandA, EstimatorA, OriginA]:
    """Construct a branded pairwise estimate from an in-package producer.

    Args:
        values: The realized estimates the producer computed.

    Returns:
        The estimate branded with the parameters the call site requires.

    """
    return PairwiseEstimateMatrix(values=values, _token=_PRODUCER_TOKEN)


def unsafe_assume_dmat[DomainA: ObjectDomain, LawA: Law, OriginA: ConstructionOrigin](
    values: Float[Array, "*batch n n"],  # noqa: F722  # jaxtyping shape
) -> DMat[DomainA, LawA, OriginA]:
    """Assert a domain, law, and origin that nothing in this call verifies.

    The only sanctioned place to write an unsupported three-axis brand, and the
    name is the review trigger. Every use is a claim the *caller* owes evidence
    for: :func:`checked_dmat` is the door when the array is merely persisted or
    external, and a private producer door is the door when jcor built it.

    Args:
        values: The square array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return DMat(values=values, _token=_PRODUCER_TOKEN)


def unsafe_assume_space_dmat[SpaceA: SpaceDescriptor, ContractA: MatrixDescriptor](
    values: Float[Array, "*batch n n"],  # noqa: F722  # jaxtyping shape
) -> SpaceDMat[SpaceA, ContractA]:
    """Assert a space and a full matrix contract without checking either.

    Still six semantic axes, five of them inside ``ContractA``; the door is no
    weaker than before, only shorter to write.
    """
    return SpaceDMat(values=values, _token=_PRODUCER_TOKEN)


def unsafe_assume_pairwise_estimate[
    DomainA: ObjectDomain,
    EstimandA,
    EstimatorA,
    OriginA: ConstructionOrigin,
](
    values: Float[Array, "*batch m n"],  # noqa: F722  # jaxtyping shape
) -> PairwiseEstimateMatrix[DomainA, EstimandA, EstimatorA, OriginA]:
    """Assert an estimand, estimator, and origin that nothing here verifies.

    Args:
        values: The rectangular array of realized estimates to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return PairwiseEstimateMatrix(values=values, _token=_PRODUCER_TOKEN)


def checked_dmat(
    values: Float[Array, "n n"],  # noqa: F722  # jaxtyping shape
    *,
    tolerance: float = 0.0,
) -> DMat[ObjectDomain, ObservedSquareLaw, ConstructionOrigin]:
    """Admit a persisted or external square array through observable checks only.

    Verifies squareness, finiteness, nonnegativity, a zero diagonal, and
    symmetry — every axiom that is decidable by looking at the numbers, and
    nothing else. The return brand is therefore the widest one in the lattice:

    * ``Domain=ObjectDomain`` — the door sees numbers, not indexed objects, so
      it cannot say the rows are unit-sphere values or empirical measures;
    * ``Origin=ConstructionOrigin`` — provenance is not recoverable from
      values, so the result can never satisfy an ``EnergyFunctional[...]``
      requirement and can never be metrized by :func:`metrize_energy`;
    * ``Law=ObservedSquareLaw`` — no separation (undecidable without objects and
      a domain-aware equality) and no negative type (observed negative
      definiteness on one finite sample is not population strong negative type).

    Eager host door: it calls ``bool()`` on array comparisons and therefore
    raises ``TracerBoolConversionError`` inside ``jit``. Call it at the
    artifact boundary, before the transform.

    Args:
        values: The square array read from a store or a third-party producer.
        tolerance: Absolute tolerance for the zero-diagonal and symmetry checks.

    Returns:
        The array branded at the widest domain, law, and origin.

    Raises:
        MatrixShapeError: The array is not 2-D or not square.
        NonfiniteMatrixError: The array contains ``NaN`` or an infinity.
        NegativeEntryError: Some entry is negative.
        NonzeroDiagonalError: Some diagonal entry exceeds ``tolerance``.
        AsymmetricMatrixError: ``|d - d.T|`` exceeds ``tolerance`` somewhere.

    Examples:
        >>> checked_dmat(jnp.zeros((3, 3))).values.shape
        (3, 3)

    """
    if values.ndim != 2 or values.shape[0] != values.shape[1]:  # noqa: PLR2004
        message = f"expected a square 2-D matrix, got shape {values.shape}"
        raise MatrixShapeError(message)
    if not bool(jnp.all(jnp.isfinite(values))):
        message = "matrix contains NaN or infinite entries"
        raise NonfiniteMatrixError(message)
    if not bool(jnp.all(values >= 0.0)):
        message = "dissimilarity matrix contains a negative entry"
        raise NegativeEntryError(message)
    if not bool(jnp.all(jnp.abs(jnp.diagonal(values)) <= tolerance)):
        message = f"diagonal is not zero within tolerance {tolerance}"
        raise NonzeroDiagonalError(message)
    if not bool(jnp.all(jnp.abs(values - values.T) <= tolerance)):
        message = f"matrix is not symmetric within tolerance {tolerance}"
        raise AsymmetricMatrixError(message)
    return DMat(values=values, _token=_PRODUCER_TOKEN)


def checked_pairwise_estimate(
    values: Float[Array, "m n"],  # noqa: F722  # jaxtyping shape
) -> PairwiseEstimateMatrix[ObjectDomain, object, object, ConstructionOrigin]:
    """Admit a persisted or external block of realized estimates.

    Checks only what an estimate permits: rectangularity and finiteness.
    Nonnegativity, a zero diagonal, and symmetry are **not** checked and must
    not be, because a signed U-statistic estimate legitimately violates all
    three. The estimand and estimator come back as ``object``, the top of both
    axes, so the result cannot stand in for any specific estimand or scheme.

    Eager host door; see :func:`checked_dmat`.

    Args:
        values: The rectangular array read from a store or third party.

    Returns:
        The array branded at the widest estimand, estimator, and origin.

    Raises:
        MatrixShapeError: The array is not 2-D.
        NonfiniteMatrixError: The array contains ``NaN`` or an infinity.

    Examples:
        >>> checked_pairwise_estimate(jnp.zeros((2, 3))).values.shape
        (2, 3)

    """
    if values.ndim != 2:  # noqa: PLR2004
        message = f"expected a 2-D matrix of estimates, got shape {values.shape}"
        raise MatrixShapeError(message)
    if not bool(jnp.all(jnp.isfinite(values))):
        message = "estimate matrix contains NaN or infinite entries"
        raise NonfiniteMatrixError(message)
    return PairwiseEstimateMatrix(values=values, _token=_PRODUCER_TOKEN)


# Flags, never alias names (see `requires`): the observable, finite-sample half
# of the contract. It is deliberately NOT the whole precondition. Rooting is
# valid only when the *ground* is of strong negative type and the required moment
# is finite — population hypotheses about the law behind the sample, unobservable
# in the matrix and uncheckable at runtime. Those are carried statically by the
# `ground` and `moment` witnesses this signature demands. Do not add them here:
# `Axioms` records what a numeric matrix can be *shown* to satisfy, and inventing
# a runtime check for a population property is the exact error t55.3 exists to
# prevent (node §Requirements 7, §Type-exposed correctness defects 3).
#
# `ZERO_DIAGONAL`, **not** `IDENTITY`. `Axioms.IDENTITY` is
# `ZERO_DIAGONAL | SEPARATION` (`core/axioms.py:123`), and separation is exactly
# what an energy functional does *not* carry on its own — it is what strong
# negative type of the ground supplies, which is the whole subject of node defect
# 6. The parameter is `PremetricLaw` (nonnegative + zero-diagonal,
# `core/axioms.py:256`), so demanding `IDENTITY` would assert a property neither
# the type nor the numbers provide; `ObservedSquareLaw` above declines separation
# for the same reason. The former `discrepancy.metrize.sqrt_energy` shortcut
# could not express these hypotheses and was removed during T55.5 rather than
# retained as an unsound parallel door.
@requires(Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY)
def metrize_energy[
    DomainA: ObjectDomain,
    GroundA: ObjectDomain,
    MomentA: MomentOrder,
    SchemeA: NonnegativeEnergyScheme,
](
    matrix: DMat[DomainA, PremetricLaw, EnergyFunctional[SchemeA]],
    ground: DomainLaw[GroundA, StrongNegativeTypeLaw],
    moment: FiniteMoment[MomentA],
) -> DMat[DomainA, HilbertianMetricLaw, RootEnergyMetric[SchemeA]]:
    """Root an energy functional into a measure metric, keeping its scheme.

    The signature *is* the content. Metrization is unavailable to an arbitrary
    nonnegative matrix, however lawful: the input must carry the exact
    ``EnergyFunctional[...]`` origin, so anything from :func:`checked_dmat`, a
    transport construction, or a direct ground evaluation is a static error even
    when its declared law is stronger. The two hypotheses that justify the step
    travel as arguments rather than as prose — a ground of **strong** negative
    type (ordinary negative type is not enough; node defect 6 records the
    ``l1`` counterexample) and the applicable finite moment.

    The scheme is preserved, not erased: rooting a V-statistic plug-in yields
    ``RootEnergyMetric[VStatisticScheme]``, which never satisfies a requirement
    for ``RootEnergyMetric[PopulationScheme]``. And because
    :class:`UStatisticScheme` cannot appear inside :class:`EnergyFunctional` at
    all, a signed U-statistic estimate has no path into this function.

    Args:
        matrix: The energy functional matrix, at a nonnegative scheme.
        ground: The declared strong-negative-type law of the ground metric.
        moment: The population finite-moment condition in force.

    Returns:
        The rooted matrix, a metric of negative type, at the same scheme.

    Examples:
        >>> from jcor.core.axioms import PremetricLaw
        >>> from jcor.core.domains import (
        ...     DomainLaw,
        ...     EmpiricalDistribution,
        ...     FiniteMoment,
        ...     FirstMomentOrder,
        ...     RawVector,
        ... )
        >>> functional: DMat[
        ...     EmpiricalDistribution, PremetricLaw, EnergyFunctional[VStatisticScheme]
        ... ] = unsafe_assume_dmat(
        ...     jnp.full((2, 2), 4.0).at[jnp.diag_indices(2)].set(0.0)
        ... )
        >>> rooted = metrize_energy(
        ...     functional,
        ...     DomainLaw[RawVector, StrongNegativeTypeLaw](),
        ...     FiniteMoment[FirstMomentOrder](),
        ... )
        >>> float(rooted.values[0, 1])
        2.0

    """
    del ground, moment
    return DMat(values=jnp.sqrt(matrix.values), _token=_PRODUCER_TOKEN)

"""Object domains, feature maps, separation evidence, and construction origins.

Position
--------
rank 0 · importable by every stage

What this module is for
-----------------------
:mod:`jcor.core.axioms` says *which axioms* a dissimilarity satisfies.
It cannot say **on what** they hold. That second question is not decoration:
the same formula ``d(x, y) = ‖N(x) - N(y)‖`` with ``N(x) = x / ‖x‖`` is

* a genuine **metric** on checked unit-sphere values,
* only a **pseudometric** on raw finite-nonzero vectors — every positive
  multiple of ``x`` collapses onto ``x``,
* a **metric** again on the quotient by *positive* rays ``x ~ c·x``, ``c > 0``,
* and **not** a metric on real projective space, because ``x`` and ``-x`` stay
  at distance two after normalization.

Four different carriers, one formula, four different separation facts. This
module makes those four distinct *types* so a consumer cannot silently read a
sphere-level fact as a raw-vector fact. The same discipline extends to
documents (a representation distance separates documents only through an
injective encoder), to measures (an empirical sample is not a population law),
and to pushforwards (``N#P = N#Q`` does not give ``P = Q``).

Nothing here is a runtime leaf
------------------------------
Domain, origin, and moment-order markers are **plain empty classes**, exactly
like the compatibility chain in :mod:`jcor.core.axioms`. They are used as
phantom type parameters and are never instantiated into an array tree. The
evidence markers are zero-field frozen dataclasses, created only at eager
checked doors (``sample/evidence.py``) and held as *static* metadata; they
carry no arrays and must not become pytree leaves.

Why the axes carry two variances
--------------------------------
One subtype order does triple duty here, because the domain lattice is ordered
by "is a subset of, with an at-least-as-fine equality":

* a **carrier** of values is covariant (:data:`DomainT`) — unit-sphere values
  *are* finite-nonzero vectors, and they *are* representatives of positive-ray
  classes, so a more specific carrier flows into a general requirement;
* a **consumer** of values is contravariant (:data:`DomainInT`) — a law that is
  only defined where ``‖x‖ ≠ 0`` must **not** be usable where a law over an
  unconstrained raw array is required (:class:`DomainLaw`), and separation of a
  coarse equality restricts to a finer carrier but never the reverse
  (:class:`Separation`).

Both readings agree because the lattice order is simultaneously "smaller
carrier" and "finer equality". Where an implication would be unsound in *both*
directions the domains are deliberately left as unrelated siblings:
:class:`Document`/:class:`DocumentRepresentation` (needs an injectivity
witness), :class:`EmpiricalDistribution`/:class:`PopulationLaw` (a finite
sample proves nothing about a population), and :class:`ProjectiveClass`
(antipodal identification is never implied by directional geometry).

What the types do *not* prove
-----------------------------
The lattice enforces the **implication** structure, not provenance. The
evidence and separation markers are ordinary constructible values, so a caller
can still write ``Separation[Document]()`` by hand. That is the same contract
:mod:`jcor.core.axioms` already has for law descriptors: a declaration justified
by the caller, checked by the eager door that is supposed to mint it, never a
proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, final

from jcor.core.typing import LawT

__all__ = [
    "AlmostSureNonzeroSupport",
    "AlphaMomentOrder",
    "AuthorityT",
    "AuthorizedSpaceLaw",
    "CarrierStructure",
    "CheckedEvidence",
    "ClaimAuthority",
    "ComputedEvidence",
    "ConstructionOrigin",
    "CoordinateEquality",
    "DeclaredAuthority",
    "DirectOrigin",
    "Document",
    "DocumentIdentity",
    "DocumentRepresentation",
    "DomainInT",
    "DomainLaw",
    "DomainT",
    "EmpiricalDistribution",
    "EmpiricalMeasureEquality",
    "EmpiricalMeasureSpace",
    "EmpiricalWitnessAuthority",
    "EnergyOrigin",
    "EqualityRelation",
    "EvidenceOrigin",
    "EvidenceOriginT",
    "FeatureMap",
    "FiniteMoment",
    "FiniteValues",
    "FirstMomentOrder",
    "GovernedReferenceAuthority",
    "InjectiveFeatureMap",
    "L1Norm",
    "L1UnitNormSupport",
    "L1UnitSphereSpace",
    "L1UnitSphereValue",
    "L2Norm",
    "L2UnitSphereSpace",
    "L2UnitSphereValue",
    "LInfNorm",
    "LInfUnitNormSupport",
    "LInfUnitSphereSpace",
    "LInfUnitSphereValue",
    "LeanCertifiedAuthority",
    "LpUnitSphereValue",
    "MathematicalSpace",
    "MeasureDomain",
    "MeasureSpaceStructure",
    "MmdOrigin",
    "MomentOrder",
    "MomentT",
    "NonnegativeValues",
    "NonzeroNormalizationDomain",
    "NonzeroPooledMean",
    "NonzeroSubsetStructure",
    "NonzeroVector",
    "NonzeroVectorCarrierSpace",
    "NormKind",
    "NormT",
    "NormalizedRows",
    "ObjectDomain",
    "OrientedRayClass",
    "OriginT",
    "PointDomain",
    "PointEquality",
    "PooledPrototype",
    "PopulationLaw",
    "PopulationMeasureEquality",
    "PopulationMeasureSpace",
    "PositiveRayEquality",
    "PositiveRaySpace",
    "PoweredGroundOrigin",
    "ProjectiveClass",
    "ProjectiveEquality",
    "ProjectiveSpace",
    "PushforwardLaw",
    "PushforwardMeasureSpace",
    "QuotientSpaceStructure",
    "RawVector",
    "RawVectorSpace",
    "RepresentationEquality",
    "RepresentationSpaceStructure",
    "Separation",
    "SpaceDescriptor",
    "SpaceEqualityT",
    "SpaceInT",
    "SpaceLaw",
    "SpaceStructureT",
    "SpaceT",
    "SumOneRows",
    "TargetDomainT",
    "TransportOrigin",
    "UnitSphereStructure",
    "UnitSphereValue",
    "UnsafeAssumption",
    "ValueEvidence",
    "VectorSpaceStructure",
    "pullback_separation",
]


# --- Space vocabulary. Representation, equality, and structure are separate axes. ---


class SpaceDescriptor:
    """Root marker for a fully specified mathematical carrier space."""


class EqualityRelation:
    """Root marker for the equality used by a separation declaration."""


class PointEquality(EqualityRelation):
    """Ordinary equality of mathematical points in the declared carrier."""


class CoordinateEquality(PointEquality):
    """Coordinatewise equality of stored finite vectors."""


class PositiveRayEquality(EqualityRelation):
    """Equality modulo multiplication by a strictly positive scalar."""


class ProjectiveEquality(EqualityRelation):
    """Equality modulo multiplication by any nonzero scalar, including ``-1``."""


class DocumentIdentity(EqualityRelation):
    """Identity of source documents before representation."""


class RepresentationEquality(EqualityRelation):
    """Equality of encoded representations, possibly shared by distinct sources."""


class EmpiricalMeasureEquality(EqualityRelation):
    """Equality of finite empirical probability measures."""


class PopulationMeasureEquality(EqualityRelation):
    """Equality of population probability measures."""


class CarrierStructure:
    """Root marker for algebraic or geometric structure available on a carrier."""


class VectorSpaceStructure(CarrierStructure):
    """Carrier is closed under vector addition and scalar multiplication."""


class NonzeroSubsetStructure(CarrierStructure):
    """The punctured subset of a vector space; not itself a vector space."""


class UnitSphereStructure(CarrierStructure):
    """A norm-one level set; not closed under general vector operations."""


class QuotientSpaceStructure(CarrierStructure):
    """Elements are equivalence classes rather than stored vector representatives."""


class RepresentationSpaceStructure(CarrierStructure):
    """Elements are equivalence classes induced by a representation map."""


class MeasureSpaceStructure(CarrierStructure):
    """Elements are probability measures on another declared carrier."""


# --- Claim authority: why a law is trusted, never how values were checked. ---


class ClaimAuthority:
    """Root marker for the authority supporting a mathematical law claim."""


class LeanCertifiedAuthority(ClaimAuthority):
    """The exact scoped claim is checked by a governed Lean theorem."""


class GovernedReferenceAuthority(ClaimAuthority):
    """The claim is supported by a governed source with matched hypotheses."""


class EmpiricalWitnessAuthority(ClaimAuthority):
    """A finite property battery witnesses examples but is not a proof."""


class DeclaredAuthority(ClaimAuthority):
    """The caller declares the claim and owns its external justification."""


class NormKind:
    """Root marker for the norm used by normalized-carrier evidence."""


class L1Norm(NormKind):
    """The taxicab norm ``sum(abs(x))``."""


class L2Norm(NormKind):
    """The Euclidean norm induced by the ambient inner product."""


class LInfNorm(NormKind):
    """The maximum-coordinate norm ``max(abs(x))``."""


# --- Object domains. Inheritance = smaller carrier with a finer equality. ---


class ObjectDomain:
    """Root of the object-domain axis; asserts nothing about any carrier."""


class PointDomain(ObjectDomain):
    """Objects compared pointwise by a ground dissimilarity ``d: X × X → ℝ``."""


class MeasureDomain(ObjectDomain):
    """Objects that are probability measures, compared by a discrepancy."""


class RawVector(PointDomain):
    """Finite real vectors with ordinary vector equality; zero rows admitted.

    The widest point carrier. Normalizing strategies are **not** total here —
    ``N(0)`` is undefined — which is why :class:`DomainLaw` is contravariant in
    its domain: a law declared over :class:`NonzeroVector` cannot be supplied
    where a law over ``RawVector`` is required.
    """


class NonzeroVector(RawVector):
    """Finite vectors additionally checked to have a nonzero norm.

    Subtype of :class:`RawVector`: the same equality on a smaller carrier, so
    a raw-domain separation fact restricts here but not conversely.
    """


class PooledPrototype(NonzeroVector):
    """A pooled mean vector checked to be nonzero.

    A prototype is a finite nonzero vector, hence a :class:`NonzeroVector`. It
    is deliberately **not** related to :class:`UnitSphereValue`: pooling unit
    rows does not produce a unit vector, and an antipodal cloud pools to zero
    (see :class:`NonzeroPooledMean`).
    """


class OrientedRayClass(PointDomain):
    """The quotient of nonzero vectors by ``x ~ c·x`` for ``c > 0``.

    A *sibling* of :class:`RawVector`, not a subtype: a ray class is an
    equivalence class, not a vector, and its equality is strictly coarser. The
    normalization pullback ``‖N(x) - N(y)‖`` is a genuine metric here, and that
    fact must never widen back to raw-vector equality.
    """


NormT = TypeVar("NormT", bound=NormKind, covariant=True)  # noqa: PLC0105


class LpUnitSphereValue(
    NonzeroVector,
    OrientedRayClass,
    Generic[NormT],  # noqa: UP046  # PEP 695 loses declared covariance
):
    """A value checked to satisfy ``‖x‖_p = 1`` for the phantom ``NormT``.

    Subtype of **both** parents, and both widenings are sound:

    * every unit vector is a finite nonzero vector, with identical equality;
    * the unit sphere is a section of the positive-ray quotient — ``x ↦ [x]``
      is a bijection onto :class:`OrientedRayClass` preserving chord distance —
      so a checked unit value is usable wherever a ray class is required.

    Only :class:`L2UnitSphereValue` licenses the inner-product chord identities.
    L1 and LInf unit values remain positive-ray representatives but must not
    enter an L2-specific consumer.
    """


class L1UnitSphereValue(LpUnitSphereValue[L1Norm]):
    """A coordinate vector checked on the L1 unit sphere."""


class L2UnitSphereValue(LpUnitSphereValue[L2Norm]):
    """A coordinate vector checked on the inner-product L2 unit sphere."""


class LInfUnitSphereValue(LpUnitSphereValue[LInfNorm]):
    """A coordinate vector checked on the L-infinity unit sphere."""


class UnitSphereValue(L2UnitSphereValue):
    """Compatibility spelling for :class:`L2UnitSphereValue`.

    Existing jcor declarations used ``UnitSphereValue`` when only L2
    normalization existed. Keeping this nominal subclass makes that assumption
    explicit without weakening current call sites to an arbitrary Lp sphere.
    """


class ProjectiveClass(PointDomain):
    """Real projective identification ``x ~ c·x`` for every ``c ≠ 0``.

    An unrelated sibling with no producer in jcor, kept so the fourth carrier
    of the chord story is nameable rather than accidentally conflated with
    :class:`OrientedRayClass`. The chord pullback is **not** a metric here:
    antipodes ``x`` and ``-x`` are identified yet stay at distance two.
    """


class Document(PointDomain):
    """Source objects encoded into vectors, with document identity as equality.

    Unrelated to :class:`DocumentRepresentation` in both directions. Widening
    representation-level separation to document-level separation is exactly the
    unsound step; it is only available through :func:`pullback_separation` with
    an :class:`InjectiveFeatureMap` witness.
    """


class DocumentRepresentation(PointDomain):
    """The quotient of documents by equality of their encoded representation.

    A representation distance is a genuine metric on these classes and only a
    pseudometric on :class:`Document` unless the encoder is injective.
    """


class EmpiricalDistribution(MeasureDomain):
    """A finitely-supported empirical measure over observed points.

    A sibling of :class:`PopulationLaw`, never a subtype. Finite support, finite
    observed values, and an observed nonnegative statistic prove nothing about
    population support, population moments, strong negative type, or the
    consistency of a plug-in estimator.
    """


class PopulationLaw(MeasureDomain):
    """A population probability law over an object domain."""


class PushforwardLaw(PopulationLaw):
    """The image ``F#P`` of a population law under a feature map.

    A pushforward *is* a population law, so the covariant carrier widening is
    sound. The contravariant :class:`Separation` axis then gives exactly the
    correct implication: separating source laws restricts to separating their
    pushforwards, while ``N#P ≠ N#Q`` never widens back to ``P ≠ Q`` — two laws
    differing only in their radial distribution share a pushforward.
    """


# --- Construction origin. How a value was built, not what it satisfies. ---


class ConstructionOrigin:
    """Root of the construction-origin axis; the generic, uninformative origin."""


class DirectOrigin(ConstructionOrigin):
    """Direct pointwise evaluation of a ground dissimilarity on a carrier."""


class NormalizationPullbackOrigin(DirectOrigin):
    """Direct evaluation composed with ``N(x) = x / ‖x‖``.

    Still a direct pointwise ground evaluation, hence a :class:`DirectOrigin`;
    the extra brand records that the values describe the *pullback* of a
    sphere-level distance and that the estimand for measures is ``N#P``.
    """


class PoweredGroundOrigin(DirectOrigin):
    """Direct ground evaluation raised to an exponent, ``d^α``."""


class EnergyOrigin(ConstructionOrigin):
    """An energy-functional construction over measures, not a ground evaluation."""


class MmdOrigin(ConstructionOrigin):
    """A maximum-mean-discrepancy construction from a kernel."""


class TransportOrigin(ConstructionOrigin):
    """An optimal-transport construction, exact or sliced."""


# --- Moment orders for the population finite-moment condition. ---


class MomentOrder:
    """Root of the moment-order axis used by :class:`FiniteMoment`."""


class FirstMomentOrder(MomentOrder):
    """``E‖X‖ < ∞`` — the condition the unit-exponent energy distance needs."""


class AlphaMomentOrder(MomentOrder):
    """``E‖X‖^α < ∞`` for the ground exponent ``α`` actually in use.

    A sibling of :class:`FirstMomentOrder`: for ``α`` chosen elsewhere in the
    open range used by energy statistics, neither condition implies the other
    without knowing ``α``, so no implication is declared here.
    """


# --- Axis parameters. Variance is load-bearing; see the module docstring. ---

#: **Covariant** object domain for carriers of values. A more specific domain
#: flows into a general requirement: a unit-sphere carrier satisfies a
#: nonzero-vector or oriented-ray requirement, while a raw-vector carrier does
#: not satisfy a unit-sphere requirement.
#:
#: The explicit ``TypeVar`` is required for the same reason as
#: :data:`jcor.core.typing.Ax`: these are phantom parameters, so PEP 695 has
#: nothing to infer variance from and settles on invariant, which would reject
#: the sound widening as well as the unsound one.
DomainT = TypeVar("DomainT", bound=ObjectDomain, covariant=True)  # noqa: PLC0105

#: **Contravariant** object domain for *consumers* of values — laws, separation
#: facts, and feature-map sources. A declaration over a wider domain satisfies a
#: narrower requirement (a law total on raw vectors is total on nonzero ones);
#: the reverse is the unsound direction and is a pyrefly error.
DomainInT = TypeVar("DomainInT", bound=ObjectDomain, contravariant=True)  # noqa: PLC0105

#: **Covariant** target domain of a feature map, in result position.
TargetDomainT = TypeVar("TargetDomainT", bound=ObjectDomain, covariant=True)  # noqa: PLC0105

#: **Covariant** construction origin. A specific origin satisfies a generic
#: requirement (``EnergyOrigin`` where ``ConstructionOrigin`` is asked for,
#: ``NormalizationPullbackOrigin`` where ``DirectOrigin`` is asked for); a
#: generic origin never satisfies a specific one.
OriginT = TypeVar("OriginT", bound=ConstructionOrigin, covariant=True)  # noqa: PLC0105

#: Covariant authority supporting an :class:`AuthorizedSpaceLaw`.
AuthorityT = TypeVar("AuthorityT", bound=ClaimAuthority, covariant=True)  # noqa: PLC0105

#: Covariant origin describing how value evidence was obtained.
EvidenceOriginT = TypeVar(  # noqa: PLC0105
    "EvidenceOriginT",
    bound="EvidenceOrigin",
    covariant=True,
)

#: **Covariant** moment order carried by :class:`FiniteMoment`.
MomentT = TypeVar("MomentT", bound=MomentOrder, covariant=True)  # noqa: PLC0105

#: Covariant equality and structure axes for :class:`MathematicalSpace`.
SpaceEqualityT = TypeVar(  # noqa: PLC0105
    "SpaceEqualityT",
    bound=EqualityRelation,
    covariant=True,
)
SpaceStructureT = TypeVar(  # noqa: PLC0105
    "SpaceStructureT",
    bound=CarrierStructure,
    covariant=True,
)
SpaceT = TypeVar("SpaceT", bound=SpaceDescriptor, covariant=True)  # noqa: PLC0105
SpaceInT = TypeVar(  # noqa: PLC0105
    "SpaceInT",
    bound=SpaceDescriptor,
    contravariant=True,
)


# --- Universal contracts. Phantom parameters, zero fields, no array leaves. ---


@final
@dataclass(frozen=True, slots=True)
class Separation(Generic[DomainInT]):  # noqa: UP046  # PEP 695 loses variance
    """Evidence that a dissimilarity's zero set is exactly ``DomainInT``-equality.

    Contravariant, because separation of a coarser equality is the *stronger*
    fact: ``Separation[OrientedRayClass]`` is usable where
    ``Separation[UnitSphereValue]`` is required, and ``Separation[RawVector]``
    is usable where ``Separation[NonzeroVector]`` is required, while neither
    reverse flow type-checks.

    The unsound widenings this blocks, each a distinct pyrefly error:
    sphere or ray separation read as raw-vector separation, representation
    separation read as document separation (use :func:`pullback_separation`),
    and pushforward separation read as source-law separation.

    Examples:
        >>> ray_level: Separation[OrientedRayClass] = Separation()
        >>> sphere_level: Separation[UnitSphereValue] = ray_level

    """


@dataclass(frozen=True, slots=True)
class MathematicalSpace(
    SpaceDescriptor,
    Generic[DomainT, SpaceEqualityT, SpaceStructureT],  # noqa: UP046
):
    """Zero-field descriptor tying carrier, equality, and structure together.

    The parameters are phantom and covariant. Algorithms should request the
    weakest structure they actually use: a unit-sphere descriptor cannot enter
    a vector-space consumer merely because both are represented by arrays.
    """


type RawVectorSpace = MathematicalSpace[
    RawVector,
    CoordinateEquality,
    VectorSpaceStructure,
]
type NonzeroVectorCarrierSpace = MathematicalSpace[
    NonzeroVector,
    CoordinateEquality,
    NonzeroSubsetStructure,
]
type L1UnitSphereSpace = MathematicalSpace[
    L1UnitSphereValue,
    CoordinateEquality,
    UnitSphereStructure,
]
type L2UnitSphereSpace = MathematicalSpace[
    L2UnitSphereValue,
    CoordinateEquality,
    UnitSphereStructure,
]
type LInfUnitSphereSpace = MathematicalSpace[
    LInfUnitSphereValue,
    CoordinateEquality,
    UnitSphereStructure,
]
type PositiveRaySpace = MathematicalSpace[
    OrientedRayClass,
    PositiveRayEquality,
    QuotientSpaceStructure,
]
type ProjectiveSpace = MathematicalSpace[
    ProjectiveClass,
    ProjectiveEquality,
    QuotientSpaceStructure,
]
type EmpiricalMeasureSpace = MathematicalSpace[
    EmpiricalDistribution,
    EmpiricalMeasureEquality,
    MeasureSpaceStructure,
]
type PopulationMeasureSpace = MathematicalSpace[
    PopulationLaw,
    PopulationMeasureEquality,
    MeasureSpaceStructure,
]
type PushforwardMeasureSpace = MathematicalSpace[
    PushforwardLaw,
    PopulationMeasureEquality,
    MeasureSpaceStructure,
]


@final
@dataclass(frozen=True, slots=True)
class SpaceLaw(Generic[SpaceInT, LawT]):  # noqa: UP046  # declared variance
    """A structural law attached to one complete mathematical space.

    Unlike :class:`DomainLaw`, this contract carries the carrier's equality and
    available structure as part of ``SpaceInT``. It is contravariant in the
    space and covariant in the law, matching a consumer that accepts a law total
    on a wider space wherever a restriction is requested.
    """


@final
@dataclass(frozen=True, slots=True)
class AuthorizedSpaceLaw(
    Generic[SpaceInT, LawT, AuthorityT]  # noqa: UP046  # declared variance
):
    """A space law whose proof/reference/declaration authority stays visible.

    Authority is deliberately independent of value-evidence origin: checking
    every row of one carrier cannot turn an empirical witness into a theorem,
    and a Lean theorem cannot certify that an unchecked runtime array satisfies
    its hypotheses.
    """


@dataclass(frozen=True, slots=True)
class FeatureMap(Generic[DomainInT, TargetDomainT]):  # noqa: UP046  # see variance
    """A declared measurable map from a source domain to a target domain.

    Contravariant in the source and covariant in the target, so a map declared
    on raw vectors is usable where a map on nonzero vectors is wanted, and a map
    into unit-sphere values is usable where a map into oriented rays is wanted.

    It carries **no injectivity claim**. Pulling a target-domain separation fact
    back to the source needs :class:`InjectiveFeatureMap`.
    """


@final
@dataclass(frozen=True, slots=True)
class InjectiveFeatureMap(
    FeatureMap[DomainInT, TargetDomainT],
    Generic[DomainInT, TargetDomainT],  # noqa: UP046  # PEP 695 loses variance
):
    """A feature map declared injective, hence able to transport separation.

    The witness required by :func:`pullback_separation`. Sole legitimate route
    from representation-level or sphere-level separation back to equality of the
    source objects.
    """


@final
@dataclass(frozen=True, slots=True)
class DomainLaw(Generic[DomainInT, LawT]):  # noqa: UP046  # see variance
    """A structural law together with the domain on which it is declared total.

    Contravariant in the domain and covariant in the law. This is what stops a
    normalizing strategy from advertising itself over an unconstrained raw
    array: ``DomainLaw[NonzeroVector, PseudometricLaw]`` is **not** assignable
    to ``DomainLaw[RawVector, PseudometricLaw]``, because ``N`` is undefined on
    a zero row and the declaration would be a claim of totality the strategy
    cannot honour.
    """


# --- Value-level evidence. Minted at eager checked doors, never inferred. ---


class ValueEvidence:
    """Root of the value-evidence axis; carries no claim of its own.

    Subclasses are checked facts about *values*, established at an eager door
    (``sample/evidence.py``) and thereafter static. The inheritance edges below
    are the only implications; every other pair is deliberately unrelated.
    """


class EvidenceOrigin:
    """Root marker for how a value-level fact entered a checked carrier."""


@final
@dataclass(frozen=True, slots=True)
class ComputedEvidence(EvidenceOrigin):
    """The producing operation establishes the fact by construction."""


@final
@dataclass(frozen=True, slots=True)
class CheckedEvidence(EvidenceOrigin):
    """An eager boundary inspected concrete values against an explicit policy."""


@final
@dataclass(frozen=True, slots=True)
class UnsafeAssumption(EvidenceOrigin):
    """A caller asserted the fact without computation or eager validation."""


@dataclass(frozen=True, slots=True)
class FiniteValues(ValueEvidence):
    """Every active value is finite — no ``NaN``, no ``±inf``.

    A finite-sample fact. It says nothing about population support, population
    moments, or consistency; :class:`FiniteMoment` is a separate, unreachable
    sibling for exactly that reason.
    """


@final
@dataclass(frozen=True, slots=True)
class NonnegativeValues(ValueEvidence):
    """Every active scalar is nonnegative; no sum or norm fact is implied."""


@final
@dataclass(frozen=True, slots=True)
class SumOneRows(ValueEvidence):
    """Every active row sums to one; nonnegativity is a separate fact."""


@dataclass(frozen=True, slots=True)
class NonzeroNormalizationDomain(FiniteValues):
    """Every active row is finite with a nonzero norm, so ``N(x)`` is defined."""


@dataclass(frozen=True, slots=True)
class NormalizedRows(NonzeroNormalizationDomain, Generic[NormT]):  # noqa: UP046
    """Every active row is unit under the phantom norm kind ``NormT``."""


@final
@dataclass(frozen=True, slots=True)
class L1UnitNormSupport(NormalizedRows[L1Norm]):
    """Every active row is checked to have L1 norm one."""


@final
@dataclass(frozen=True, slots=True)
class UnitNormSupport(NormalizedRows[L2Norm]):
    """Every active row is checked to satisfy ``‖x‖₂ = 1``.

    Sound implications: unit rows are nonzero rows and are finite.

    **The implication that must not exist**: unit rows do *not* give a nonzero
    pooled mean. An antipodal cloud ``{x, -x}`` has unit rows and averages to
    exactly zero, so :class:`NonzeroPooledMean` is an unrelated sibling and the
    only way to obtain it is an eager door that actually measures the pooled
    norm. Both classes are ``@final`` so the implication cannot be reintroduced
    downstream by a subclass inheriting from the two of them.
    """


@final
@dataclass(frozen=True, slots=True)
class LInfUnitNormSupport(NormalizedRows[LInfNorm]):
    """Every active row is checked to have L-infinity norm one."""


@final
@dataclass(frozen=True, slots=True)
class NonzeroPooledMean(FiniteValues):
    """The pooled mean of the active rows is checked to have a nonzero norm.

    Deliberately **not** reachable from :class:`UnitNormSupport`; see there.
    """


@final
@dataclass(frozen=True, slots=True)
class AlmostSureNonzeroSupport(ValueEvidence):
    """A population law is supported almost surely off the origin.

    The policy a normalization pushforward ``N#P`` needs. A population claim,
    so it is unreachable from :class:`NonzeroNormalizationDomain`, which only
    inspects finitely many observed rows.
    """


@final
@dataclass(frozen=True, slots=True)
class FiniteMoment(ValueEvidence, Generic[MomentT]):  # noqa: UP046  # see variance
    """A population finite-moment condition at a declared order.

    Not a subtype of :class:`FiniteValues`: observing finitely many finite
    values is not evidence that the population moment exists, and the energy
    functional's identification results are conditional on the moment, not on
    the sample.
    """


def pullback_separation[SourceT: ObjectDomain, TargetT: ObjectDomain](
    separation: Separation[TargetT],
    encoder: InjectiveFeatureMap[SourceT, TargetT],
) -> Separation[SourceT]:
    """Transport a target-domain separation fact back along an injective map.

    If ``d`` separates ``TargetT``-equality and ``F`` is injective, then
    ``d(F(a), F(b)) = 0`` gives ``F(a) = F(b)`` and hence ``a = b``. Injectivity
    is the entire content of the step: with an ordinary :class:`FeatureMap` the
    pullback is only a pseudometric on the source, which is why this signature
    accepts nothing weaker.

    Args:
        separation: Separation established on the target domain.
        encoder: The declared **injective** feature map ``SourceT → TargetT``.

    Returns:
        The transported separation fact on the source domain.

    Examples:
        >>> representation = Separation[DocumentRepresentation]()
        >>> encoder = InjectiveFeatureMap[Document, DocumentRepresentation]()
        >>> pullback_separation(representation, encoder)
        Separation()

    """
    del separation, encoder
    return Separation()

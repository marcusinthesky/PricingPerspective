"""Typed mathematical transforms and static-preserving JAX doors.

The map is part of the theorem
------------------------------
A law is never transported by formula alone.  Its source carrier, target
carrier, equality, totality requirement, and construction origin determine
what survives composition.  In particular, L2 normalization is total only on
nonzero vectors and is not injective for coordinate equality.  Pulling a
sphere metric back through it therefore yields a pseudometric on raw nonzero
vectors, while negative type is preserved.

The public functions below make the permitted proof steps explicit.  There is
deliberately no generic ``pullback`` returning its input law: such a function
would allow a separating target metric to cross a noninjective map unchanged.
Only an :class:`InjectiveTransform` can use :func:`restrict_law`, and quotient
metrization additionally requires an explicit fibre/equality witness.

JAX's runtime pytrees preserve jcor's branded carriers, but the upstream
``jit`` and ``vmap`` annotations currently erase generic return parameters for
Pyrefly. These narrow adapters restore the ordinary callable contract: the
transformed callable has the same parameter and return types as its input.

The promise is intentionally limited. Shape batching is still expressed by
jaxtyping/runtime tests; these functions preserve semantic Python types and do
not claim that an arbitrary callable is transformable.

Both adapters are caller-facing, and take no static-argument parameters. jcor
compiles at the leaf with ``@partial(jax.jit, static_argnames=…)``, which they
cannot express, so having no call site under ``src/`` is correct rather than an
oversight. :mod:`tests.core.test_transforms` pins what they buy: substituting
raw ``jax.jit``/``jax.vmap`` into its static fixture drops Pyrefly from seven
detected bad assignments to three and silences the sibling-brand comparison
altogether.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast, final

import jax

from jcor.core.domains import (
    ConstructionOrigin,
    L2UnitSphereSpace,
    NonzeroVectorCarrierSpace,
    NormalizationPullbackOrigin,
    OriginT,
    SpaceDescriptor,
    SpaceLaw,
)

__all__ = [
    "EmpiricalPushforwardMap",
    "IdentityMap",
    "InjectiveTransform",
    "L2Normalization",
    "L2NormalizationContract",
    "MapKind",
    "NonzeroInputRequired",
    "PartialDomainRequirement",
    "PopulationPushforwardMap",
    "PositiveRayQuotient",
    "ProjectiveQuotient",
    "QuotientMetricWitness",
    "QuotientTransform",
    "RestrictionMap",
    "TotalOnDeclaredSource",
    "TransformContract",
    "TransformDescriptor",
    "pullback_negative_type",
    "pullback_nonnegative",
    "pullback_pseudometric",
    "pullback_symmetric",
    "pullback_triangle",
    "pullback_zero_diagonal",
    "quotient_metric",
    "restrict_law",
    "typed_jit",
    "typed_vmap",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from jcor.core.axioms import (
        Law,
        MetricLaw,
        NegativeTypeLaw,
        Nonnegative,
        PseudometricLaw,
        Symmetric,
        TriangleInequality,
        ZeroDiagonal,
    )


class MapKind:
    """Root marker for a mathematical map's identity."""


class IdentityMap(MapKind):
    """The identity map on one declared mathematical space."""


class RestrictionMap(MapKind):
    """Inclusion of a restricted carrier into a wider carrier."""


class L2Normalization(MapKind):
    """The map ``x ↦ x / ‖x‖₂`` on finite nonzero vectors."""


class PositiveRayQuotient(MapKind):
    """Quotient map identifying strictly positive scalar multiples."""


class ProjectiveQuotient(MapKind):
    """Quotient map identifying all nonzero scalar multiples."""


class EmpiricalPushforwardMap(MapKind):
    """Image of a finite empirical measure under a declared point map."""


class PopulationPushforwardMap(MapKind):
    """Image of a population probability law under a declared point map."""


class PartialDomainRequirement:
    """Root marker for the totality condition recorded by a transform."""


class TotalOnDeclaredSource(PartialDomainRequirement):
    """The map is total on every value of its declared source space."""


class NonzeroInputRequired(PartialDomainRequirement):
    """The map requires a finite input whose L2 norm is nonzero."""


SourceSpaceInT = TypeVar(  # noqa: PLC0105
    "SourceSpaceInT",
    bound=SpaceDescriptor,
    contravariant=True,
)
TargetSpaceT = TypeVar(  # noqa: PLC0105
    "TargetSpaceT",
    bound=SpaceDescriptor,
    covariant=True,
)
MapT = TypeVar("MapT", bound=MapKind, covariant=True)  # noqa: PLC0105
RequirementT = TypeVar(  # noqa: PLC0105
    "RequirementT",
    bound=PartialDomainRequirement,
    covariant=True,
)
QuotientSourceT = TypeVar(
    "QuotientSourceT",
    bound=SpaceDescriptor,
)
QuotientTargetT = TypeVar(
    "QuotientTargetT",
    bound=SpaceDescriptor,
)
QuotientMapT = TypeVar("QuotientMapT", bound=MapKind)
QuotientRequirementT = TypeVar(
    "QuotientRequirementT",
    bound=PartialDomainRequirement,
)
QuotientOriginT = TypeVar(
    "QuotientOriginT",
    bound=ConstructionOrigin,
)


@dataclass(frozen=True, slots=True)
class TransformDescriptor:
    """Non-generic root for a complete static transform descriptor."""


class TransformContract(
    TransformDescriptor,
    Generic[SourceSpaceInT, TargetSpaceT, MapT, RequirementT, OriginT],  # noqa: UP046
):
    """Zero-field descriptor of a source-to-target mathematical map.

    All five parameters are static facts.  In particular, the requirement is
    not inferred from an array and the origin is construction provenance, not
    claim authority.  Eager doors must establish any value evidence needed to
    enter the declared source carrier.
    """


@final
@dataclass(frozen=True, slots=True)
class InjectiveTransform(
    TransformContract[SourceSpaceInT, TargetSpaceT, MapT, RequirementT, OriginT],
    Generic[SourceSpaceInT, TargetSpaceT, MapT, RequirementT, OriginT],  # noqa: UP046
):
    """A transform injective for the source and target equalities."""


@final
@dataclass(frozen=True, slots=True)
class QuotientTransform(
    TransformContract[
        QuotientSourceT,
        QuotientTargetT,
        QuotientMapT,
        QuotientRequirementT,
        QuotientOriginT,
    ],
    Generic[  # noqa: UP046
        QuotientSourceT,
        QuotientTargetT,
        QuotientMapT,
        QuotientRequirementT,
        QuotientOriginT,
    ],
):
    """Invariant quotient descriptor, preventing cross-quotient witnesses."""


type L2NormalizationContract = TransformContract[
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    L2Normalization,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
]


@final
@dataclass(frozen=True, slots=True)
class QuotientMetricWitness(
    Generic[QuotientSourceT, QuotientTargetT, QuotientMapT]  # noqa: UP046
):
    """Witness that zero-distance fibres are exactly the target equality.

    This is intentionally constructible only as a declaration: Python types do
    not prove the theorem.  Its presence nevertheless prevents a quotient map
    from silently upgrading a pseudometric to a metric.
    """


def pullback_nonnegative[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, Nonnegative],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, Nonnegative]:
    """Transport nonnegativity through any declared transform."""
    del law, transform
    return SpaceLaw()


def pullback_zero_diagonal[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, ZeroDiagonal],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, ZeroDiagonal]:
    """Transport a zero diagonal through any declared transform."""
    del law, transform
    return SpaceLaw()


def pullback_symmetric[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, Symmetric],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, Symmetric]:
    """Transport symmetry through any declared transform."""
    del law, transform
    return SpaceLaw()


def pullback_triangle[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, TriangleInequality],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, TriangleInequality]:
    """Transport the triangle inequality through any declared transform."""
    del law, transform
    return SpaceLaw()


def pullback_pseudometric[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, PseudometricLaw],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, PseudometricLaw]:
    """Pull a pseudometric back, deliberately making no separation claim."""
    del law, transform
    return SpaceLaw()


def pullback_negative_type[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[TargetT, NegativeTypeLaw],
    transform: TransformContract[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, NegativeTypeLaw]:
    """Transport conditional negative definiteness through composition."""
    del law, transform
    return SpaceLaw()


def restrict_law[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
    PropertyT: Law,
](
    law: SpaceLaw[TargetT, PropertyT],
    transform: InjectiveTransform[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
) -> SpaceLaw[SourceT, PropertyT]:
    """Preserve every declared law along an equality-injective restriction."""
    del law, transform
    return SpaceLaw()


def quotient_metric[
    SourceT: SpaceDescriptor,
    TargetT: SpaceDescriptor,
    MapKindT: MapKind,
    RequirementKindT: PartialDomainRequirement,
    ConstructionT: ConstructionOrigin,
](
    law: SpaceLaw[SourceT, PseudometricLaw],
    transform: QuotientTransform[
        SourceT, TargetT, MapKindT, RequirementKindT, ConstructionT
    ],
    witness: QuotientMetricWitness[SourceT, TargetT, MapKindT],
) -> SpaceLaw[TargetT, MetricLaw]:
    """Descend a pseudometric to a metric only with an exact-fibre witness."""
    del law, transform, witness
    return SpaceLaw()


def typed_jit[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Compile ``function`` while preserving its full static signature."""
    return cast("Callable[P, R]", jax.jit(function))


def typed_vmap[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Vectorize ``function`` while preserving its semantic return type."""
    return cast("Callable[P, R]", jax.vmap(function))

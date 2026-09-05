"""Typed, JAX-static point-kernel strategy contracts.

Names are diagnostic only. Numerical consumers dispatch by calling a declared
strategy, while the strategy's kernel law and carrier metadata propagate
structurally through MMD producers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

import jax

from jcor.core.domains import (
    ClaimAuthority,
    DeclaredAuthority,
    SpaceDescriptor,
    ValueEvidence,
)
from jcor.core.transforms import TransformDescriptor

__all__ = [
    "CharacteristicKernelLaw",
    "CharacteristicKernelProperties",
    "PositiveSemidefiniteKernelLaw",
    "PositiveSemidefiniteKernelProperties",
    "SimilarityFunction",
    "SimilarityStrategy",
    "declare_similarity",
]

type SimilarityFunction = Callable[
    [jax.Array, jax.Array],
    jax.Array,
]


@runtime_checkable
class PositiveSemidefiniteKernelLaw(Protocol):
    """Capability that every finite Gram matrix is positive semidefinite."""

    @property
    def positive_semidefinite(self) -> Literal[True]:
        """Declare the universal PSD kernel property."""
        ...


@runtime_checkable
class CharacteristicKernelLaw(PositiveSemidefiniteKernelLaw, Protocol):
    """PSD kernel whose mean embedding separates probability measures."""

    @property
    def characteristic(self) -> Literal[True]:
        """Declare injectivity of the population kernel mean embedding."""
        ...


@dataclass(frozen=True, slots=True)
class PositiveSemidefiniteKernelProperties:
    """Runtime descriptor for an unconditionally PSD point kernel."""

    positive_semidefinite: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class CharacteristicKernelProperties(PositiveSemidefiniteKernelProperties):
    """Runtime descriptor for a characteristic PSD point kernel."""

    characteristic: Literal[True] = field(default=True, init=False)


KernelLawT_co = TypeVar(
    "KernelLawT_co",
    bound=PositiveSemidefiniteKernelLaw,
    covariant=True,
)
KernelSpaceT_co = TypeVar(
    "KernelSpaceT_co",
    bound=SpaceDescriptor,
    covariant=True,
    default=SpaceDescriptor,
)
KernelTransformT_co = TypeVar(
    "KernelTransformT_co",
    bound=TransformDescriptor,
    covariant=True,
    default=TransformDescriptor,
)
KernelAuthorityT_co = TypeVar(
    "KernelAuthorityT_co",
    bound=ClaimAuthority,
    covariant=True,
    default=ClaimAuthority,
)
KernelEvidenceT_co = TypeVar(
    "KernelEvidenceT_co",
    bound=ValueEvidence,
    covariant=True,
    default=ValueEvidence,
)


@runtime_checkable
class SimilarityStrategy(
    Protocol[
        KernelLawT_co,
        KernelSpaceT_co,
        KernelTransformT_co,
        KernelAuthorityT_co,
        KernelEvidenceT_co,
    ]
):
    """A pointwise Gram producer with its exact kernel and carrier contract."""

    @property
    def name(self) -> str:
        """Stable diagnostic name, never a dispatch key."""
        ...

    @property
    def law(self) -> KernelLawT_co:
        """PSD or characteristic kernel law carried by this strategy."""
        ...

    @property
    def space(self) -> KernelSpaceT_co:
        """Carrier on which the kernel law is declared."""
        ...

    @property
    def transform(self) -> KernelTransformT_co:
        """Direct or pullback construction of the kernel."""
        ...

    @property
    def authority(self) -> KernelAuthorityT_co:
        """Authority supporting the universal kernel claim."""
        ...

    @property
    def required_evidence(self) -> type[KernelEvidenceT_co]:
        """Weakest checked value evidence required by the strategy."""
        ...

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        *,
        reference: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate a Gram block; ordinary kernels ignore ``reference``."""
        ...


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class _DeclaredSimilarity(
    Generic[  # noqa: UP046  # explicit TypeVars retain covariance and defaults
        KernelLawT_co,
        KernelSpaceT_co,
        KernelTransformT_co,
        KernelAuthorityT_co,
        KernelEvidenceT_co,
    ]
):
    _function: SimilarityFunction
    _law: KernelLawT_co
    _name: str
    _space: KernelSpaceT_co
    _transform: KernelTransformT_co
    _authority: KernelAuthorityT_co
    _required_evidence: type[KernelEvidenceT_co]

    @property
    def name(self) -> str:
        return self._name

    @property
    def law(self) -> KernelLawT_co:
        return self._law

    @property
    def space(self) -> KernelSpaceT_co:
        return self._space

    @property
    def transform(self) -> KernelTransformT_co:
        return self._transform

    @property
    def authority(self) -> KernelAuthorityT_co:
        return self._authority

    @property
    def required_evidence(self) -> type[KernelEvidenceT_co]:
        return self._required_evidence

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        *,
        reference: jax.Array | None = None,
    ) -> jax.Array:
        del reference
        return self._function(x, y)


_UNSPECIFIED_SPACE = SpaceDescriptor()
_UNSPECIFIED_TRANSFORM = TransformDescriptor()
_DECLARED_AUTHORITY = DeclaredAuthority()


def declare_similarity[
    KernelLawKind: PositiveSemidefiniteKernelLaw,
    KernelSpaceKind: SpaceDescriptor,
    KernelTransformKind: TransformDescriptor,
    KernelAuthorityKind: ClaimAuthority,
    KernelEvidenceKind: ValueEvidence,
](
    function: SimilarityFunction,
    *,
    law: KernelLawKind,
    name: str | None = None,
    space: KernelSpaceKind = _UNSPECIFIED_SPACE,
    transform: KernelTransformKind = _UNSPECIFIED_TRANSFORM,
    authority: KernelAuthorityKind = _DECLARED_AUTHORITY,
    required_evidence: type[KernelEvidenceKind],
) -> SimilarityStrategy[
    KernelLawKind,
    KernelSpaceKind,
    KernelTransformKind,
    KernelAuthorityKind,
    KernelEvidenceKind,
]:
    """Declare a third-party PSD kernel without erasing its carrier metadata."""
    if not callable(function):
        message = "a declared similarity requires a callable function"
        raise TypeError(message)
    if not isinstance(law, PositiveSemidefiniteKernelLaw):
        message = "law must declare positive_semidefinite=True"
        raise TypeError(message)
    resolved_name = name if name is not None else getattr(function, "__name__", "")
    if not isinstance(resolved_name, str) or not resolved_name:
        message = "a declared similarity requires a non-empty name"
        raise ValueError(message)
    declared_evidence: type[KernelEvidenceKind] = required_evidence
    if not isinstance(required_evidence, type) or not issubclass(
        required_evidence, ValueEvidence
    ):
        message = "required_evidence must be a ValueEvidence marker type"
        raise TypeError(message)
    try:
        hash(function)
        hash(law)
    except TypeError as error:
        message = "declared similarity function and law must be hashable"
        raise TypeError(message) from error
    declared: _DeclaredSimilarity[
        KernelLawKind,
        KernelSpaceKind,
        KernelTransformKind,
        KernelAuthorityKind,
        KernelEvidenceKind,
    ] = _DeclaredSimilarity[
        KernelLawKind,
        KernelSpaceKind,
        KernelTransformKind,
        KernelAuthorityKind,
        KernelEvidenceKind,
    ](
        function,
        law,
        resolved_name,
        space,
        transform,
        authority,
        declared_evidence,
    )
    return declared

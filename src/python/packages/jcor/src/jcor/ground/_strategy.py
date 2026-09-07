"""Typed, JAX-static ground-distance strategy contracts.

This private implementation sits below :mod:`jcor.ground.metrics`: it knows the
ground-distance call shape, but it knows no built-in names or implementations.
Built-ins and third-party declarations therefore share one structural protocol,
while serialized selection remains isolated in :mod:`jcor.ground.config`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

import jax

from jcor.core.axioms import Axioms, Law, LawProperties
from jcor.core.domains import (
    ClaimAuthority,
    DeclaredAuthority,
    SpaceDescriptor,
    ValueEvidence,
)
from jcor.core.transforms import TransformDescriptor
from jcor.core.typing import Array, Float, LawT

__all__ = [
    "GroundDistance",
    "GroundDistanceFunction",
    "GroundLawDeclaration",
    "declare_ground_distance",
]

type GroundDistanceFunction = Callable[
    [Float[Array, "n d"], Float[Array, "m d"]],
    Float[Array, "n m"],
]
"""Callable shape for a vectorized point-to-point ground distance."""


GroundSpaceT = TypeVar(  # noqa: PLC0105
    "GroundSpaceT",
    bound=SpaceDescriptor,
    covariant=True,
    default=SpaceDescriptor,
)
GroundTransformT = TypeVar(  # noqa: PLC0105
    "GroundTransformT",
    bound=TransformDescriptor,
    covariant=True,
    default=TransformDescriptor,
)
GroundAuthorityT = TypeVar(  # noqa: PLC0105
    "GroundAuthorityT",
    bound=ClaimAuthority,
    covariant=True,
    default=ClaimAuthority,
)
GroundEvidenceT = TypeVar(  # noqa: PLC0105  # public semantic axis name
    "GroundEvidenceT",
    bound=ValueEvidence,
    covariant=True,
    default=ValueEvidence,
)

_UNSPECIFIED_SPACE = SpaceDescriptor()
_UNSPECIFIED_TRANSFORM = TransformDescriptor()
_DECLARED_AUTHORITY = DeclaredAuthority()


@runtime_checkable
class GroundLawDeclaration(Protocol[LawT]):
    """Signature-independent name and exact mathematical ground law."""

    @property
    def name(self) -> str:
        """Stable diagnostic/configuration name; never used for dispatch."""
        ...

    @property
    def law(self) -> LawT:
        """Mathematical law declared for this ground distance."""
        ...

    @property
    def requires_nonzero_rows(self) -> bool:
        """Whether the callable's declared value domain excludes zero rows."""
        ...


@runtime_checkable
class GroundDistance(
    GroundLawDeclaration[LawT],
    Protocol[
        LawT,
        GroundSpaceT,
        GroundTransformT,
        GroundAuthorityT,
        GroundEvidenceT,
    ],
):
    """A JAX point-cloud distance carrying its exact law."""

    @property
    def backend(self) -> Literal["jax"]:
        """Execution backend used by this strategy."""
        ...

    @property
    def space(self) -> GroundSpaceT:
        """Complete carrier/equality/structure on which ``law`` is declared."""
        ...

    @property
    def transform(self) -> GroundTransformT:
        """Direct, restriction, pullback, or quotient construction descriptor."""
        ...

    @property
    def authority(self) -> GroundAuthorityT:
        """Authority supporting the universal law declaration."""
        ...

    @property
    def required_evidence(self) -> type[GroundEvidenceT]:
        """Weakest checked value evidence that admits the callable's inputs."""
        ...

    def __call__(
        self,
        x: Float[Array, "n d"],
        y: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        """Evaluate the vectorized ground distance."""
        ...


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class _DeclaredGroundDistance(
    Generic[  # noqa: UP046  # explicit TypeVars retain covariance and defaults
        LawT,
        GroundSpaceT,
        GroundTransformT,
        GroundAuthorityT,
        GroundEvidenceT,
    ]
):
    """Immutable implementation returned by :func:`declare_ground_distance`."""

    _function: GroundDistanceFunction
    _law: LawT
    _name: str
    _requires_nonzero_rows: bool
    _space: GroundSpaceT
    _transform: GroundTransformT
    _authority: GroundAuthorityT
    _required_evidence: type[GroundEvidenceT]

    @property
    def backend(self) -> Literal["jax"]:
        """Execution backend used by this strategy."""
        return "jax"

    @property
    def law(self) -> LawT:
        """Mathematical law declared for this callable."""
        return self._law

    @property
    def space(self) -> GroundSpaceT:
        """Declared mathematical input space."""
        return self._space

    @property
    def transform(self) -> GroundTransformT:
        """Construction map responsible for the declared source law."""
        return self._transform

    @property
    def authority(self) -> GroundAuthorityT:
        """Authority supporting the law claim."""
        return self._authority

    @property
    def required_evidence(self) -> type[GroundEvidenceT]:
        """Weakest checked value evidence required at an eager carrier door."""
        return self._required_evidence

    @property
    def name(self) -> str:
        """Stable diagnostic/configuration name."""
        return self._name

    @property
    def requires_nonzero_rows(self) -> bool:
        """Whether the callable's declared value domain excludes zero rows."""
        return self._requires_nonzero_rows

    def __call__(
        self,
        x: Float[Array, "n d"],
        y: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        """Evaluate the declared vectorized distance."""
        return self._function(x, y)


def declare_ground_distance[
    GroundLawT: Law,
    GroundSpaceKind: SpaceDescriptor,
    GroundTransformKind: TransformDescriptor,
    GroundAuthorityKind: ClaimAuthority,
    GroundEvidenceKind: ValueEvidence,
](
    function: GroundDistanceFunction,
    *,
    law: GroundLawT,
    name: str | None = None,
    requires_nonzero_rows: bool = False,
    space: GroundSpaceKind = _UNSPECIFIED_SPACE,
    transform: GroundTransformKind = _UNSPECIFIED_TRANSFORM,
    authority: GroundAuthorityKind = _DECLARED_AUTHORITY,
    required_evidence: type[GroundEvidenceKind],
) -> GroundDistance[
    GroundLawT,
    GroundSpaceKind,
    GroundTransformKind,
    GroundAuthorityKind,
    GroundEvidenceKind,
]:
    """Declare a third-party ground callable without erasing its law or signature.

    Args:
        function: Vectorized ``(n, d) × (m, d) -> (n, m)`` distance callable.
        law: Immutable structural law descriptor justified by the caller.
        name: Stable diagnostic name; defaults to the callable's ``__name__``.
        requires_nonzero_rows: Whether normalization excludes zero-valued rows.
        space: Carrier, equality, and structure on which ``law`` holds.
        transform: Construction map, including its source/target and partiality.
        authority: Why the universal law declaration is trusted.
        required_evidence: Weakest value evidence accepted by checked consumers.

    Returns:
        A hashable, leafless JAX-static strategy retaining ``GroundLawT``.

    Raises:
        ValueError: If no non-empty strategy name can be determined.

    """
    if not callable(function):
        message = "a declared ground distance requires a callable function"
        raise TypeError(message)
    if (
        not isinstance(law, Law)
        or not isinstance(law.axioms, Axioms)
        or not isinstance(law.properties, LawProperties)
    ):
        message = "law must be a runtime Law descriptor with typed flag fields"
        raise TypeError(message)
    resolved_name = name if name is not None else getattr(function, "__name__", "")
    if not isinstance(resolved_name, str) or not resolved_name:
        message = "a declared ground distance requires a non-empty name"
        raise ValueError(message)
    if not isinstance(requires_nonzero_rows, bool):
        message = "requires_nonzero_rows must be a bool"
        raise TypeError(message)
    declared_evidence: type[GroundEvidenceKind] = required_evidence
    if not isinstance(required_evidence, type) or not issubclass(
        required_evidence, ValueEvidence
    ):
        message = "required_evidence must be a ValueEvidence marker type"
        raise TypeError(message)
    try:
        hash(function)
        hash(law)
    except TypeError as error:
        message = "declared ground function and law must be hashable for JAX static use"
        raise TypeError(message) from error
    declared: _DeclaredGroundDistance[
        GroundLawT,
        GroundSpaceKind,
        GroundTransformKind,
        GroundAuthorityKind,
        GroundEvidenceKind,
    ] = _DeclaredGroundDistance[
        GroundLawT,
        GroundSpaceKind,
        GroundTransformKind,
        GroundAuthorityKind,
        GroundEvidenceKind,
    ](
        function,
        law,
        resolved_name,
        requires_nonzero_rows,
        space,
        transform,
        authority,
        declared_evidence,
    )
    return declared

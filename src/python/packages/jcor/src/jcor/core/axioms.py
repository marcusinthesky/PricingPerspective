"""Structural law capabilities and executable dissimilarity declarations.

Position
--------
rank 0 · importable by every stage

The stage taxonomy names objects (``ground`` compares *points*, ``discrepancy``
compares *measures*), never axioms: Wasserstein *is* a metric, energy distance
over a separating ground metric is a semimetric whose square root is a metric,
and ``cosine_distance`` is weaker still. Axioms are therefore a **declared
property** carried here, not a directory.

Two complementary planes
------------------------
The read-only protocols are the static plane. Their small, orthogonal
capabilities compose structurally, so a third party can declare a law without
inheriting a jcor nominal class. :class:`MetricLaw` therefore widens to
:class:`PseudometricLaw`, while :class:`StrongNegativeTypeLaw` widens to
:class:`NegativeTypeLaw` independently of every distance axiom.

The :class:`Axioms` and :class:`LawProperties` flags are the executable plane.
They drive property batteries and diagnostics; types remain declarations, not
proofs that a universal mathematical law holds. Preconditions declare flags,
never alias names: write ``requires=Axioms.TRIANGLE`` rather than
``requires=SEMIMETRIC``.

Convention
----------
Names follow Deza & Deza, *Encyclopedia of Distances*, §1.1 ("Basic
definitions"). That choice matters because **"semimetric" is contested**: Deza &
Deza use it for a metric minus the triangle inequality, while much of the
analysis literature uses "semimetric" for what Deza & Deza call a
*pseudo-metric* (triangle holds, but ``d(x, y) = 0`` is permitted for
``x != y``). Both readings appear in the sources this package implements, hence
the flags-not-names rule above.

Under the Deza & Deza convention, on a set ``X`` with ``d: X × X → ℝ``:

===============  ====================================================
alias            axioms
===============  ====================================================
``DIVERGENCE``   ``NONNEGATIVE | ZERO_DIAGONAL | SEPARATION``
``SEMIMETRIC``   ``DIVERGENCE | SYMMETRY``
``QUASIMETRIC``  ``DIVERGENCE | TRIANGLE``
``PSEUDOMETRIC`` ``NONNEGATIVE | ZERO_DIAGONAL | SYMMETRY | TRIANGLE``
``METRIC``       ``PSEUDOMETRIC | SEPARATION``
===============  ====================================================

Compatibility markers
---------------------
The nominal :class:`Premetric` → :class:`Semimetric` → :class:`Metric` →
:class:`NegativeType` chain remains temporarily because shipped ``DMat``
producers and consumers still name it. It is not the new law model: off-chain
facts continue to brand down while migration proceeds. T55.5 removes this
compatibility chain after typed strategies and containers use the structural
protocols end to end.
"""

from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DIVERGENCE",
    "MARKER_AXIOMS",
    "MARKER_PROPERTIES",
    "METRIC",
    "NEGATIVE_TYPE",
    "PSEUDOMETRIC",
    "QUASIMETRIC",
    "REQUIRES_ATTR",
    "SEMIMETRIC",
    "STRONG_NEGATIVE_TYPE",
    "Axioms",
    "DivergenceLaw",
    "Law",
    "LawProperties",
    "Metric",
    "MetricLaw",
    "NegativeType",
    "NegativeTypeLaw",
    "Nonnegative",
    "Premetric",
    "PremetricLaw",
    "PseudometricLaw",
    "QuasimetricLaw",
    "Semimetric",
    "SemimetricLaw",
    "Separating",
    "StrongNegativeTypeLaw",
    "Symmetric",
    "TriangleInequality",
    "ZeroDiagonal",
    "missing_axioms",
    "missing_law_properties",
    "required_axioms",
    "requires",
]


class Axioms(Flag):
    """Distance axioms, composable with ``|``.

    Attributes:
        NONNEGATIVE: ``d(x, y) >= 0`` for all ``x, y``.
        ZERO_DIAGONAL: ``d(x, x) == 0`` for every ``x``.
        SEPARATION: ``d(x, y) == 0`` implies ``x == y``.
        IDENTITY: Compatibility composite ``ZERO_DIAGONAL | SEPARATION``.
        SYMMETRY: ``d(x, y) == d(y, x)``.
        TRIANGLE: ``d(x, z) <= d(x, y) + d(y, z)``.

    """

    NONNEGATIVE = auto()
    ZERO_DIAGONAL = auto()
    SEPARATION = auto()
    SYMMETRY = auto()
    TRIANGLE = auto()
    IDENTITY = ZERO_DIAGONAL | SEPARATION


class LawProperties(Flag):
    """Global properties orthogonal to the pointwise distance axioms.

    ``STRONG_NEGATIVE_TYPE`` is stored as its own bit so diagnostics can name
    it precisely. :func:`missing_law_properties` closes the implication
    ``strong negative type => negative type`` before comparing declarations.
    Use the module-level :data:`STRONG_NEGATIVE_TYPE` composite when declaring
    a strong-negative-type law.
    """

    NEGATIVE_TYPE = auto()
    STRONG_NEGATIVE_TYPE = auto()


#: A conditionally negative-definite dissimilarity.
NEGATIVE_TYPE: Final = LawProperties.NEGATIVE_TYPE

#: Strong negative type implies ordinary negative type.
STRONG_NEGATIVE_TYPE: Final = (
    LawProperties.NEGATIVE_TYPE | LawProperties.STRONG_NEGATIVE_TYPE
)


# --- Aliases. Prose and test declarations only — never a `requires=` value. ---

#: Nonnegative and zero iff equal; neither symmetric nor sub-additive.
#: Kullback-Leibler and the Bregman family live here.
DIVERGENCE: Final = Axioms.NONNEGATIVE | Axioms.IDENTITY

#: A metric minus the triangle inequality (Deza & Deza §1.1). ``energy_distance``
#: over a *separating* ground metric (e.g. ``metric="euclidean"``) is a semimetric
#: in this sense. Over ``metric="angular"`` — its default — it is not: the ground
#: metric cannot see scale, so ``IDENTITY`` fails too.
SEMIMETRIC: Final = Axioms.NONNEGATIVE | Axioms.IDENTITY | Axioms.SYMMETRY

#: A metric minus symmetry — a directed distance.
QUASIMETRIC: Final = Axioms.NONNEGATIVE | Axioms.IDENTITY | Axioms.TRIANGLE

#: A metric minus the identity of indiscernibles: distinct points may sit at
#: distance zero. (Much of the analysis literature calls this a "semimetric";
#: this package does not — see the module docstring.)
PSEUDOMETRIC: Final = (
    Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY | Axioms.TRIANGLE
)

#: A separating pseudometric.
METRIC: Final = PSEUDOMETRIC | Axioms.SEPARATION


# --- Structural capabilities. Read-only properties compose covariantly. ---


@runtime_checkable
class Law(Protocol):
    """Base protocol for an immutable runtime law descriptor."""

    @property
    def axioms(self) -> Axioms:
        """Pointwise axioms declared by the descriptor."""
        ...

    @property
    def properties(self) -> LawProperties:
        """Orthogonal global properties declared by the descriptor."""
        ...


class Nonnegative(Law, Protocol):
    """Capability declaring ``d(x, y) >= 0``."""

    @property
    def nonnegative(self) -> Literal[True]:
        """The nonnegativity witness marker."""
        ...


class ZeroDiagonal(Law, Protocol):
    """Capability declaring ``d(x, x) == 0``."""

    @property
    def zero_diagonal(self) -> Literal[True]:
        """The zero-diagonal witness marker."""
        ...


class Separating(Law, Protocol):
    """Capability declaring that zero dissimilarity separates points."""

    @property
    def separates_points(self) -> Literal[True]:
        """The separation witness marker."""
        ...


class Symmetric(Law, Protocol):
    """Capability declaring ``d(x, y) == d(y, x)``."""

    @property
    def symmetric(self) -> Literal[True]:
        """The symmetry witness marker."""
        ...


class TriangleInequality(Law, Protocol):
    """Capability declaring the triangle inequality."""

    @property
    def triangle_inequality(self) -> Literal[True]:
        """The triangle-inequality witness marker."""
        ...


class NegativeTypeLaw(Law, Protocol):
    """Capability declaring conditional negative definiteness."""

    @property
    def negative_type(self) -> Literal[True]:
        """The negative-type witness marker."""
        ...


class StrongNegativeTypeLaw(NegativeTypeLaw, Protocol):
    """Capability declaring strong negative type, hence negative type."""

    @property
    def strong_negative_type(self) -> Literal[True]:
        """The strong-negative-type witness marker."""
        ...


class PremetricLaw(Nonnegative, ZeroDiagonal, Protocol):
    """Nonnegative dissimilarity with a zero diagonal."""


class DivergenceLaw(PremetricLaw, Separating, Protocol):
    """A separating premetric; symmetry and triangle are not required."""


class SemimetricLaw(DivergenceLaw, Symmetric, Protocol):
    """A Deza--Deza semimetric: a metric without triangle inequality."""


class QuasimetricLaw(DivergenceLaw, TriangleInequality, Protocol):
    """A directed metric: symmetry is not required."""


class PseudometricLaw(PremetricLaw, Symmetric, TriangleInequality, Protocol):
    """A metric whose zero-distance relation need not separate points."""


class MetricLaw(
    PremetricLaw,
    Separating,
    Symmetric,
    TriangleInequality,
    Protocol,
):
    """A nonnegative, reflexive, separating, symmetric, subadditive law."""


# --- Static markers. Inheritance = more axioms = narrower type. ---


class Premetric:
    """Legacy weakest brand: only ``d >= 0`` is statically bridged.

    The old chain treated zero diagonal as an unstated package-wide assumption.
    New code uses :class:`PremetricLaw`, where that capability is explicit;
    T55.5 removes this deliberately weaker compatibility marker.
    """


class Semimetric(Premetric):
    """:data:`SEMIMETRIC` — adds symmetry and the identity of indiscernibles."""


class Metric(Semimetric):
    """:data:`METRIC` — adds the triangle inequality."""


class NegativeType(Metric):
    """A metric of negative type (Deza & Deza §15.2).

    ``Σ_i Σ_j c_i c_j d(x_i, x_j) <= 0`` whenever ``Σ_i c_i == 0``; equivalently
    the double-centred Gram matrix ``-½ J D J`` is positive semi-definite, which
    is exactly the condition classical MDS / PCoA and the energy statistics need.
    The narrowest brand in the chain.
    """


#: Axioms implied by each temporary static marker. Negative type is carried in
#: :data:`MARKER_PROPERTIES`, not folded into this pointwise flag set.
MARKER_AXIOMS: Final[dict[type[Premetric], Axioms]] = {
    Premetric: Axioms.NONNEGATIVE,
    Semimetric: SEMIMETRIC,
    Metric: METRIC,
    NegativeType: METRIC,
}

#: Orthogonal properties implied by each temporary static marker.
MARKER_PROPERTIES: Final[dict[type[Premetric], LawProperties]] = {
    Premetric: LawProperties(0),
    Semimetric: LawProperties(0),
    Metric: LawProperties(0),
    NegativeType: NEGATIVE_TYPE,
}


# --- Preconditions. A declaration, not a runtime guard. ---

#: Attribute :func:`requires` writes on the decorated callable. Named here so a
#: battery can read it without importing the decorator's implementation.
REQUIRES_ATTR: Final = "__jcor_requires__"


def requires[F](axioms: Axioms) -> Callable[[F], F]:
    """Declare the axioms a callable's dissimilarity input must satisfy.

    Returns the function **unchanged**, having recorded ``axioms`` under
    :data:`REQUIRES_ATTR`. It is deliberately not a runtime guard:

    * ``Ax`` on :class:`jcor.core.typing.DMat` is *phantom*, so at runtime there
      is nothing on the argument to check against — the brand exists only for
      pyrefly.
    * Checking the axioms directly is ``O(n²)`` (symmetry, identity) to
      ``O(n³)`` (triangle, over triples) per call, on the hot path of a
      ``jit``-compiled numerical package, and the values may be tracers.

    Enforcement is therefore static and test-time, matching where the axioms are
    already established: :mod:`jcor.testing.axioms` proves what each producer
    satisfies, ``DECLARED_AXIOMS`` records it, and a battery checks supplier ⊇
    consumer for every declaration this decorator leaves behind.

    Pass **flags, never alias names** — ``Axioms.TRIANGLE``, not ``SEMIMETRIC``.
    The rule cannot be enforced here, because ``SEMIMETRIC`` *is*
    ``NONNEGATIVE | IDENTITY | SYMMETRY`` once evaluated and the two are
    indistinguishable; the battery reads the call site with ``ast`` instead, the
    same technique ``test_kernels_api.py`` uses to check overload co-location.

    Args:
        axioms: Flags the input must be known to satisfy.

    Returns:
        A decorator that records ``axioms`` and returns its argument.

    Examples:
        >>> @requires(Axioms.NONNEGATIVE | Axioms.SYMMETRY)
        ... def embed(d: object) -> object:
        ...     return d
        >>> required_axioms(embed) is Axioms.NONNEGATIVE | Axioms.SYMMETRY
        True

    """

    def declare(function: F) -> F:
        setattr(function, REQUIRES_ATTR, axioms)
        return function

    return declare


def required_axioms(function: object) -> Axioms | None:
    """Read back the precondition :func:`requires` recorded on a callable.

    Args:
        function: Any object; typically a callable that may carry a declaration.

    Returns:
        The declared flags, or ``None`` if the callable declares no
        precondition.

    """
    declared = getattr(function, REQUIRES_ATTR, None)
    return declared if isinstance(declared, Axioms) else None


def missing_axioms(supplied: Axioms, required: Axioms) -> Axioms:
    """Return the flags ``required`` asks for that ``supplied`` does not carry.

    The empty flag is the pass condition, so a battery reads as
    ``assert not missing_axioms(...)`` and the failure message names exactly
    which axioms are short.

    Args:
        supplied: Flags the producer is known to guarantee — its
            ``DECLARED_AXIOMS`` entry.
        required: Flags the consumer declared via :func:`requires`.

    Returns:
        The shortfall, ``required & ~supplied``.

    Examples:
        >>> missing_axioms(PSEUDOMETRIC, METRIC) is Axioms.SEPARATION
        True
        >>> not missing_axioms(METRIC, SEMIMETRIC)
        True

    """
    return required & ~supplied


def missing_law_properties(
    supplied: LawProperties,
    required: LawProperties,
) -> LawProperties:
    """Return the orthogonal law properties missing from ``supplied``.

    Strong negative type implies negative type even if a caller constructs the
    atomic enum member directly instead of using the module-level composite.
    The implication is closed on both sides before computing the shortfall.

    Args:
        supplied: Properties guaranteed by a producer.
        required: Properties required by a consumer.

    Returns:
        The implication-aware property shortfall.

    Examples:
        >>> not missing_law_properties(STRONG_NEGATIVE_TYPE, NEGATIVE_TYPE)
        True
        >>> missing_law_properties(
        ...     NEGATIVE_TYPE, STRONG_NEGATIVE_TYPE
        ... ) is LawProperties.STRONG_NEGATIVE_TYPE
        True

    """

    def close(properties: LawProperties) -> LawProperties:
        if LawProperties.STRONG_NEGATIVE_TYPE in properties:
            return properties | LawProperties.NEGATIVE_TYPE
        return properties

    return close(required) & ~close(supplied)

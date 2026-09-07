"""Typed ground/exponent contracts for energy statistics and DISCO.

The distribution-sensitive interval is not one undifferentiated theorem.
Snowflaking a negative-type ground (``0 < alpha < 1``), preserving it
(``alpha = 1``), and raising it beyond one (``1 < alpha < 2``) require
different capabilities. ``alpha = 2`` is the separate mean/ANOVA branch.
Numerical estimators consume a geometry built here; they never infer these
claims from an arbitrary float and string inside a compiled kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, Protocol, overload, runtime_checkable

import jax

from jcor.core.axioms import (
    METRIC,
    NEGATIVE_TYPE,
    SEMIMETRIC,
    STRONG_NEGATIVE_TYPE,
    Axioms,
    Law,
    LawProperties,
    MetricLaw,
    NegativeTypeLaw,
    Separating,
    StrongNegativeTypeLaw,
)
from jcor.ground._strategy import GroundDistance  # noqa: TC001  # runtime contract

__all__ = [
    "ANOVA_EXPONENT",
    "DEFAULT_DISTRIBUTION_EXPONENT",
    "UNIT_EXPONENT",
    "AnovaEnergyLaw",
    "AnovaExponent",
    "AnovaGroundLaw",
    "DistributionEnergyGroundLaw",
    "DistributionEnergyLaw",
    "DistributionExponent",
    "EnergyExponent",
    "EnergyGeometry",
    "FullFractionalPowerGroundLaw",
    "MeanEnergyLaw",
    "SeparatingDistributionEnergyGroundLaw",
    "SeparatingDistributionEnergyLaw",
    "SeparatingStrongNegativeTypeGroundLaw",
    "SeparatingSubunitPowerGroundLaw",
    "StrongFullFractionalPowerGroundLaw",
    "SubunitExponent",
    "SubunitPowerGroundLaw",
    "SuperunitExponent",
    "UnitExponent",
    "UnitPowerMeanGroundLaw",
    "energy_geometry",
    "parse_distribution_exponent",
    "parse_energy_exponent",
    "unsafe_energy_geometry",
]

_UNIT_EXPONENT = 1.0
_ANOVA_EXPONENT = 2.0


class DistributionEnergyGroundLaw(NegativeTypeLaw, Protocol):
    """Minimal ground capability for distribution energy at ``alpha = 1``."""


class UnitPowerMeanGroundLaw(DistributionEnergyGroundLaw, Protocol):
    """Unit power reduces energy to a mean-embedding discrepancy."""

    @property
    def unit_power_mean_only(self) -> Literal[True]:
        """At ``alpha = 1`` only the represented first moment is identified."""
        ...


class SeparatingDistributionEnergyGroundLaw(
    DistributionEnergyGroundLaw,
    Separating,
    Protocol,
):
    """Negative-type ground that separates points on its original domain."""


class SeparatingStrongNegativeTypeGroundLaw(
    StrongNegativeTypeLaw,
    Separating,
    Protocol,
):
    """Internally consistent strong-negative-type ground on the original domain."""


class SubunitPowerGroundLaw(DistributionEnergyGroundLaw, Protocol):
    """Snowflakes are strong on a separable represented-space quotient."""

    @property
    def subunit_power_strong_on_quotient(self) -> Literal[True]:
        """Powers in ``0 < alpha < 1`` are strong on the quotient."""
        ...


class SeparatingSubunitPowerGroundLaw(
    SubunitPowerGroundLaw,
    Separating,
    Protocol,
):
    """Subunit-power theorem capability with an injective representation."""


class FullFractionalPowerGroundLaw(DistributionEnergyGroundLaw, Protocol):
    """Ground whose powers through ``1 < alpha < 2`` remain negative type.

    This capability alone does not claim that the powered ground is strong.
    """

    @property
    def full_fractional_power_negative_type(self) -> Literal[True]:
        """Every power in ``1 < alpha < 2`` remains negative type."""
        ...


class StrongFullFractionalPowerGroundLaw(
    FullFractionalPowerGroundLaw,
    Separating,
    Protocol,
):
    """Fractional powers are strong on the original point domain."""

    @property
    def full_fractional_power_strong(self) -> Literal[True]:
        """Every power in ``1 < alpha < 2`` is strong on the original domain."""
        ...


class AnovaGroundLaw(MetricLaw, Protocol):
    """Ground law whose squared distance realizes classical ANOVA geometry."""

    @property
    def squared_distance_anova(self) -> Literal[True]:
        """Squaring this ground distance realizes inner-product ANOVA."""
        ...


@dataclass(frozen=True, slots=True)
class DistributionEnergyLaw:
    """Possibly nonseparating symmetric negative-type energy functional."""

    axioms: Axioms = field(
        default=Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY,
        init=False,
    )
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class SeparatingDistributionEnergyLaw(DistributionEnergyLaw):
    """Separating energy law justified by the selected theorem branch."""

    axioms: Axioms = field(default=SEMIMETRIC, init=False)
    separates_points: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class MeanEnergyLaw(DistributionEnergyLaw):
    """Mean-embedding-only energy functional, not a distribution distance."""


@dataclass(frozen=True, slots=True)
class AnovaEnergyLaw(MeanEnergyLaw):
    """Mean-sensitive squared-distance law for the ``alpha = 2`` branch."""


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class SubunitExponent:
    """Validated snowflake exponent ``0 < alpha < 1``."""

    value: float

    def __post_init__(self) -> None:
        """Validate the finite open subunit interval."""
        if not math.isfinite(self.value) or not 0.0 < self.value < _UNIT_EXPONENT:
            message = f"subunit exponent must be in the range (0, 1), got {self.value}"
            raise ValueError(message)


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class UnitExponent:
    """Validated singleton exponent ``alpha = 1``."""

    value: float = field(default=_UNIT_EXPONENT, init=False)


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class SuperunitExponent:
    """Validated full-fractional-power exponent ``1 < alpha < 2``."""

    value: float

    def __post_init__(self) -> None:
        """Validate the finite open superunit interval."""
        if (
            not math.isfinite(self.value)
            or not _UNIT_EXPONENT < self.value < _ANOVA_EXPONENT
        ):
            message = (
                f"superunit exponent must be in the range (1, 2), got {self.value}"
            )
            raise ValueError(message)


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class AnovaExponent:
    """Validated singleton exponent for the classical ``alpha = 2`` branch."""

    value: float = field(default=_ANOVA_EXPONENT, init=False)


type DistributionExponent = SubunitExponent | UnitExponent | SuperunitExponent
"""Validated distribution-sensitive exponent regime."""

type EnergyExponent = DistributionExponent | AnovaExponent
"""Validated distribution-sensitive or ANOVA exponent regime."""

UNIT_EXPONENT = UnitExponent()
"""Unit-power distribution-energy exponent singleton."""

DEFAULT_DISTRIBUTION_EXPONENT = UNIT_EXPONENT
"""Default distribution-sensitive energy exponent."""

ANOVA_EXPONENT = AnovaExponent()
"""Classical mean/ANOVA exponent singleton."""


@runtime_checkable
class EnergyGeometry[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](Protocol):
    """Read-only compatible ground, exponent regime, and resulting law."""

    @property
    def ground(self) -> GroundDistance[GroundLawT]:
        """Declared ground strategy."""
        ...

    @property
    def exponent(self) -> ExponentT:
        """Validated exponent regime."""
        ...

    @property
    def law(self) -> EnergyLawT:
        """Result law implied by the compatible construction."""
        ...


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class _DeclaredEnergyGeometry[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
]:
    """Private immutable implementation returned only by the two factories."""

    _ground: GroundDistance[GroundLawT]
    _exponent: ExponentT
    _law: EnergyLawT

    @property
    def ground(self) -> GroundDistance[GroundLawT]:
        return self._ground

    @property
    def exponent(self) -> ExponentT:
        return self._exponent

    @property
    def law(self) -> EnergyLawT:
        return self._law


@overload
def energy_geometry[
    SeparatingSubunitGroundLawT: SeparatingSubunitPowerGroundLaw,
](
    ground: GroundDistance[SeparatingSubunitGroundLawT],
    exponent: SubunitExponent,
) -> EnergyGeometry[
    SeparatingSubunitGroundLawT,
    SubunitExponent,
    SeparatingDistributionEnergyLaw,
]: ...


@overload
def energy_geometry[NegativeGroundLawT: DistributionEnergyGroundLaw](
    ground: GroundDistance[NegativeGroundLawT],
    exponent: SubunitExponent,
) -> EnergyGeometry[
    NegativeGroundLawT,
    SubunitExponent,
    DistributionEnergyLaw,
]: ...


@overload
def energy_geometry[MeanGroundLawT: UnitPowerMeanGroundLaw](
    ground: GroundDistance[MeanGroundLawT],
    exponent: UnitExponent,
) -> EnergyGeometry[
    MeanGroundLawT,
    UnitExponent,
    MeanEnergyLaw,
]: ...


@overload
def energy_geometry[
    StrongGroundLawT: SeparatingStrongNegativeTypeGroundLaw,
](
    ground: GroundDistance[StrongGroundLawT],
    exponent: UnitExponent,
) -> EnergyGeometry[
    StrongGroundLawT,
    UnitExponent,
    SeparatingDistributionEnergyLaw,
]: ...


@overload
def energy_geometry[NegativeGroundLawT: DistributionEnergyGroundLaw](
    ground: GroundDistance[NegativeGroundLawT],
    exponent: UnitExponent,
) -> EnergyGeometry[
    NegativeGroundLawT,
    UnitExponent,
    DistributionEnergyLaw,
]: ...


@overload
def energy_geometry[
    StrongFullPowerGroundLawT: StrongFullFractionalPowerGroundLaw,
](
    ground: GroundDistance[StrongFullPowerGroundLawT],
    exponent: SuperunitExponent,
) -> EnergyGeometry[
    StrongFullPowerGroundLawT,
    SuperunitExponent,
    SeparatingDistributionEnergyLaw,
]: ...


@overload
def energy_geometry[FullPowerGroundLawT: FullFractionalPowerGroundLaw](
    ground: GroundDistance[FullPowerGroundLawT],
    exponent: SuperunitExponent,
) -> EnergyGeometry[
    FullPowerGroundLawT,
    SuperunitExponent,
    DistributionEnergyLaw,
]: ...


@overload
def energy_geometry[AnovaGroundLawT: AnovaGroundLaw](
    ground: GroundDistance[AnovaGroundLawT],
    exponent: AnovaExponent,
) -> EnergyGeometry[AnovaGroundLawT, AnovaExponent, AnovaEnergyLaw]: ...


def energy_geometry(
    ground: GroundDistance[Law],
    exponent: EnergyExponent,
) -> EnergyGeometry[Law, EnergyExponent, Law]:
    """Construct a supported energy geometry without an unchecked assertion."""
    if ground.backend != "jax":
        message = "energy geometry requires a JAX ground-distance strategy"
        raise TypeError(message)
    result_law = _energy_result_law(ground.law, ground.name, exponent)
    try:
        hash(ground)
        hash(exponent)
        hash(result_law)
    except TypeError as error:
        message = "energy geometry requires hashable static components"
        raise TypeError(message) from error
    return _DeclaredEnergyGeometry(
        ground,
        exponent,
        result_law,
    )


def _has_axioms(law: Law, required: Axioms, *markers: str) -> bool:
    """Return whether runtime flags and their structural markers agree."""
    return (law.axioms & required) == required and all(
        getattr(law, marker, False) is True for marker in markers
    )


def _has_properties(law: Law, required: LawProperties, *markers: str) -> bool:
    """Return whether runtime properties and their structural markers agree."""
    return (law.properties & required) == required and all(
        getattr(law, marker, False) is True for marker in markers
    )


def _require_negative_type(law: Law, ground_name: str) -> None:
    """Require an internally consistent ordinary-negative-type declaration."""
    if not _has_properties(law, NEGATIVE_TYPE, "negative_type"):
        message = f"{ground_name!r} does not declare negative type"
        raise TypeError(message)


def _subunit_result_law(law: Law, ground_name: str) -> DistributionEnergyLaw:
    """Resolve the snowflake theorem branch."""
    _require_negative_type(law, ground_name)
    separates = _has_axioms(law, Axioms.SEPARATION, "separates_points")
    if separates and getattr(law, "subunit_power_strong_on_quotient", False) is True:
        return SeparatingDistributionEnergyLaw()
    return DistributionEnergyLaw()


def _unit_result_law(law: Law, ground_name: str) -> DistributionEnergyLaw:
    """Resolve the preserved-ground theorem branch."""
    _require_negative_type(law, ground_name)
    if getattr(law, "unit_power_mean_only", False) is True:
        return MeanEnergyLaw()
    separates = _has_axioms(law, Axioms.SEPARATION, "separates_points")
    if separates and _has_properties(
        law,
        STRONG_NEGATIVE_TYPE,
        "strong_negative_type",
    ):
        return SeparatingDistributionEnergyLaw()
    return DistributionEnergyLaw()


def _superunit_result_law(law: Law, ground_name: str) -> DistributionEnergyLaw:
    """Resolve the full-fractional-power theorem branch."""
    _require_negative_type(law, ground_name)
    if getattr(law, "full_fractional_power_negative_type", False) is not True:
        message = (
            f"{ground_name!r} does not declare full fractional-power "
            "compatibility for exponents in (1, 2)"
        )
        raise TypeError(message)
    separates = _has_axioms(law, Axioms.SEPARATION, "separates_points")
    if separates and getattr(law, "full_fractional_power_strong", False) is True:
        return SeparatingDistributionEnergyLaw()
    return DistributionEnergyLaw()


def _anova_result_law(law: Law, ground_name: str) -> AnovaEnergyLaw:
    """Resolve the exact squared-distance ANOVA branch."""
    if (
        not _has_axioms(
            law,
            METRIC,
            "nonnegative",
            "zero_diagonal",
            "separates_points",
            "symmetric",
            "triangle_inequality",
        )
        or getattr(law, "squared_distance_anova", False) is not True
    ):
        message = f"{ground_name!r} does not declare squared-distance ANOVA geometry"
        raise TypeError(message)
    return AnovaEnergyLaw()


def _energy_result_law(
    law: Law,
    ground_name: str,
    exponent: EnergyExponent,
) -> Law:
    """Validate one theorem branch and return its runtime result descriptor."""
    if not isinstance(law, Law):
        message = f"{ground_name!r} has no runtime Law descriptor"
        raise TypeError(message)
    if isinstance(exponent, SubunitExponent):
        return _subunit_result_law(law, ground_name)
    if isinstance(exponent, UnitExponent):
        return _unit_result_law(law, ground_name)
    if isinstance(exponent, SuperunitExponent):
        return _superunit_result_law(law, ground_name)
    if isinstance(exponent, AnovaExponent):
        return _anova_result_law(law, ground_name)
    message = "exponent must be a validated energy-exponent regime"
    raise TypeError(message)


def unsafe_energy_geometry[
    UnsafeGroundLawT: Law,
    UnsafeExponentT: EnergyExponent,
    UnsafeEnergyLawT: Law,
](
    ground: GroundDistance[UnsafeGroundLawT],
    exponent: UnsafeExponentT,
    *,
    assume_law: UnsafeEnergyLawT,
) -> EnergyGeometry[UnsafeGroundLawT, UnsafeExponentT, UnsafeEnergyLawT]:
    """Construct geometry from an explicit unchecked result-law assertion.

    Only the implication from ``ground`` and ``exponent`` to ``assume_law`` is
    unchecked. The ground must still be a declared JAX strategy, the exponent
    must still be a validated nominal regime, and every component must remain
    a hashable runtime contract. Any conditional value-domain requirements on
    ``ground`` (for example nonzero rows) remain in force.
    """
    if ground.backend != "jax":
        message = "unsafe energy geometry requires a JAX ground-distance strategy"
        raise TypeError(message)
    if not isinstance(
        exponent,
        (SubunitExponent, UnitExponent, SuperunitExponent, AnovaExponent),
    ):
        message = "unsafe energy geometry requires a validated exponent regime"
        raise TypeError(message)
    if (
        not isinstance(assume_law, Law)
        or not isinstance(assume_law.axioms, Axioms)
        or not isinstance(assume_law.properties, LawProperties)
    ):
        message = "assume_law must be a runtime Law descriptor"
        raise TypeError(message)
    try:
        hash(ground)
        hash(exponent)
        hash(assume_law)
    except TypeError as error:
        message = "unsafe energy geometry requires hashable static components"
        raise TypeError(message) from error
    return _DeclaredEnergyGeometry(ground, exponent, assume_law)


def parse_distribution_exponent(value: object) -> DistributionExponent:
    """Validate a serialized exponent into a precise distribution regime."""
    parsed = parse_energy_exponent(value)
    if isinstance(parsed, AnovaExponent):
        message = f"exponent must be in the range (0, 2), got {value}"
        # ``2`` has a valid numeric type but is outside this parser's domain.
        raise ValueError(message)  # noqa: TRY004
    return parsed


def parse_energy_exponent(value: object) -> EnergyExponent:
    """Validate a serialized numeric exponent into its theorem regime."""
    if isinstance(value, bool) or not isinstance(value, Real):
        message = f"exponent must be in the range (0, 2], got {value!r}"
        raise TypeError(message)
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= _ANOVA_EXPONENT:
        message = f"exponent must be in the range (0, 2], got {value}"
        raise ValueError(message)
    if parsed < _UNIT_EXPONENT:
        return SubunitExponent(parsed)
    if parsed == _UNIT_EXPONENT:
        return UNIT_EXPONENT
    if parsed < _ANOVA_EXPONENT:
        return SuperunitExponent(parsed)
    return ANOVA_EXPONENT

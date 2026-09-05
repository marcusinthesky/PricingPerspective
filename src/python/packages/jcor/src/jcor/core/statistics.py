"""Estimand, estimator, design, hypothesis, and calibration axes for scalars.

Position
--------
rank 0 · importable by every stage

What this module is for
-----------------------
:mod:`jcor.core.axioms` says which axioms a *dissimilarity* satisfies and
:mod:`jcor.core.domains` says on *what* they hold. Neither says anything about
a **number**: what population quantity it targets, by which estimator scheme it
was computed, under which grouped design, against which null, and by which
calibration its p-value was obtained. ``TestResult[S]`` records only the
statistic's array type, so an energy V-statistic, a signed diagonal-debiased
U-statistic, and a DISCO F-ratio are the same type today.

This module supplies those axes. Two failure modes are avoided by
construction:

**No universal container.** ``RealizedEstimate``, ``RealizedStatistic``,
:class:`GroupDesign`, :class:`AdditiveDispersionDecomposition`, and
:class:`CalibratedTest` are separate carriers that *compose*. A calibrated
result **holds** its typed statistic; it does not flatten the statistic's axes
into its own field list, and it carries no null-sample field, because an
asymptotic calibration has no null sample and a ``None`` there would be the
universal container reappearing. Resampled nulls are a sibling carrier,
:class:`ResampledNull`, composed alongside by ``inference``.

**No metric law on a scalar.** Neither realized carrier accepts a ``LawT``,
``Separation``, or ``DMat`` parameter, and there is no way to add one: a
statistic is a number with provenance. It is not a distance, it does not
separate anything, and the fact that its inputs came from a metric is a fact
about the inputs. Pyrefly rejects
``RealizedStatistic[float, StatisticProvenance[Metric, ...]]`` with
``bad-specialization``, not by convention but by the bound on
:data:`StatisticT` — and rejects a law marker supplied as the descriptor itself
by the bound on :data:`StatisticProvenanceT`. Bundling the axes into descriptors
narrowed that surface rather than widening it.

V-statistics and U-statistics are different types
-------------------------------------------------
:class:`VStatisticEstimator` and :class:`UStatisticEstimator` are nominal
*siblings* under :class:`EmpiricalEstimator`, so neither is usable where the
other is required. That is the whole point of the axis. The energy V-statistic
plug-in is nonnegative under negative type of the ground; the diagonal-debiased
U-statistic can be **signed**, and a consumer that averages, square-roots, or
logs one having type-checked against the other is silently wrong. Neither
inherits the population functional's separation law merely because its inputs
are finite: that is why :class:`PopulationValue` is a third sibling rather than
a supertype of either.

Nothing semantic is a runtime leaf
----------------------------------
The axis markers are plain empty classes, exactly as in
:mod:`jcor.core.domains`. Provenance objects are zero-field frozen dataclasses
held as **static** pytree fields, so they travel in the treedef's aux data and
can never enter a compiled array tree — ``tree_leaves`` of a realized carrier is
exactly its numeric value.

Known gaps, recorded rather than papered over
---------------------------------------------
1. **Provenance is unenforced**, exactly as in :mod:`jcor.core.domains`.
   ``EstimateProvenance[TotalDispersion, VStatisticEstimator, EnergyOrigin]()``
   is freely constructible; the axes govern *implication* and *composition*,
   never origin. Closing this needs the private-constructor / ``unsafe_assume_*``
   door convention reserved for ``core/matrices.py``.
2. **V-statistic nonnegativity is conditional.** Typed energy geometry and DISCO
   doors now carry the negative-type premise; the estimator marker alone still
   makes no unconditional sign claim.
3. :class:`GroupDesign` is a carrier, not a public constructor. The eager
   :func:`jcor.association.decomposition.group_design` door validates contiguity,
   nonempty groups, and agreement with the static group count before minting it.
4. :class:`ExactCalibration` has no producer in jcor — the resampling paths are
   Monte-Carlo add-one and the parametric paths are asymptotic. It is named so
   the three-way axis is complete; do not read it as coverage.
5. Every property here is **type-declared**, for t55.5's ledger. Nothing in this
   module is Lean-certified, governed-reference, or an empirical witness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar, final

import jax

from jcor.core.domains import ConstructionOrigin, OriginT, SpaceDescriptor
from jcor.core.transforms import TransformDescriptor
from jcor.core.typing import Array, Int  # noqa: TC001  # runtime fields; see typing.py

__all__ = [
    "AdditiveDispersionDecomposition",
    "AsymptoticCalibration",
    "BetweenDispersion",
    "CalibratedTest",
    "Calibration",
    "CalibrationDescriptor",
    "CalibrationProvenance",
    "CalibrationProvenanceT",
    "CalibrationT",
    "DecompositionOrigin",
    "Design",
    "DesignT",
    "DiscoConstruction",
    "DiscrepancyEstimand",
    "DispersionEstimand",
    "DispersionIndex",
    "EmpiricalEstimator",
    "EnergyConstruction",
    "EnergyTwoSampleStatistic",
    "Estimand",
    "EstimandT",
    "EstimateDescriptor",
    "EstimateProvenance",
    "EstimateProvenanceT",
    "EstimatorScheme",
    "EstimatorT",
    "ExactCalibration",
    "GroupDesign",
    "GroupedDesign",
    "Homogeneity",
    "Hypothesis",
    "HypothesisT",
    "Independence",
    "MathematicalConstruction",
    "MmdConstruction",
    "NoGroupEffect",
    "PopulationValue",
    "ProcedureOriginT",
    "RatioStatistic",
    "RealizedEstimate",
    "RealizedStatistic",
    "ResampledCalibration",
    "ResampledNull",
    "StatisticDescriptor",
    "StatisticKind",
    "StatisticProvenance",
    "StatisticProvenanceT",
    "StatisticT",
    "TotalDispersion",
    "TransformedEstimand",
    "UStatisticEstimator",
    "UngroupedDesign",
    "VStatisticEstimator",
    "ValueT",
    "VarianceRatioStatistic",
    "WithinDispersion",
]


# --- Estimand. *What* population quantity a number targets. ---


class Estimand:
    """Root of the estimand axis; asserts nothing about what is targeted."""


class DiscrepancyEstimand(Estimand):
    """A functional of two probability laws, such as an energy or MMD functional.

    The concrete energy/MMD/transport estimands — and in particular the
    distinction between the nonnegative functional ``S(P, Q)`` and its
    square-root measure metric — belong to ``discrepancy``, which owns those
    constructions. This root exists so ``core`` can state the composition rules
    without importing a stage above it.
    """


class DispersionEstimand(Estimand):
    """A dispersion functional of one law together with a grouping.

    The three components below are the DISCO decomposition's population
    targets. ``association`` owns the algorithm; the axis lives here so its
    typed decomposition is expressible without a rank-4 import.
    """


class MathematicalConstruction:
    """Root marker naming the statistical construction of an estimand."""


class EnergyConstruction(MathematicalConstruction):
    """Energy functional or energy statistic construction."""


class MmdConstruction(MathematicalConstruction):
    """Maximum mean discrepancy construction."""


class DiscoConstruction(MathematicalConstruction):
    """DISCO dispersion-decomposition construction."""


ConstructionT = TypeVar(  # noqa: PLC0105
    "ConstructionT",
    bound=MathematicalConstruction,
    covariant=True,
)
EstimandSourceSpaceT = TypeVar(  # noqa: PLC0105
    "EstimandSourceSpaceT",
    bound=SpaceDescriptor,
    covariant=True,
)
EstimandTargetSpaceT = TypeVar(  # noqa: PLC0105
    "EstimandTargetSpaceT",
    bound=SpaceDescriptor,
    covariant=True,
)
EstimandTransformT = TypeVar(  # noqa: PLC0105
    "EstimandTransformT",
    bound=TransformDescriptor,
    covariant=True,
)


class TransformedEstimand(
    Estimand,
    Generic[  # noqa: UP046  # explicit TypeVars retain covariance
        ConstructionT,
        EstimandSourceSpaceT,
        EstimandTargetSpaceT,
        EstimandTransformT,
    ],
):
    """A statistic targeting the pushforward/quotient estimand of a transform.

    The source is retained, but no inheritance edge widens this into the same
    construction on source-law equality. Energy, MMD, and DISCO therefore can
    share transport machinery without becoming interchangeable estimands.
    """


@final
class TotalDispersion(DispersionEstimand):
    """Total dispersion ``T``, the pooled energy dispersion of every observation."""


@final
class WithinDispersion(DispersionEstimand):
    """Within-group dispersion ``S_W``, summed over groups."""


@final
class BetweenDispersion(DispersionEstimand):
    """Between-group dispersion ``S_B = T - S_W``.

    All three components are ``@final`` nominal siblings, for the reason
    :class:`jcor.core.domains.UnitNormSupport` is final: siblings are only
    siblings while nobody writes a class inheriting two of them to "unify" a
    decomposition. ``@final`` turns that shortcut into a static error in the
    file that attempts it, so ``AdditiveDispersionDecomposition`` cannot be
    assembled by passing the within estimate where the total is required.
    """


# --- Estimator scheme. *How* a number was computed from data, or not. ---


class EstimatorScheme:
    """Root of the estimator-scheme axis; the uninformative scheme."""


class PopulationValue(EstimatorScheme):
    """The exact value of the estimand under a declared population law.

    A sibling of :class:`EmpiricalEstimator`, never a supertype of it. A finite
    sample does not deliver a population value, and a population identity — the
    separation of ``sqrt(S(P, Q))`` under strong negative type, for instance —
    does not transfer to a plug-in merely because its inputs are finite.
    """


class EmpiricalEstimator(EstimatorScheme):
    """A statistic computed from a finite sample."""


class VStatisticEstimator(EmpiricalEstimator):
    """A V-statistic plug-in: the zero diagonal is kept and the sum divided by ``n²``.

    Nonnegative under negative type of the ground — but that nonnegativity is
    **not** typed here; see the module docstring's known gap 2.
    """


class UStatisticEstimator(EmpiricalEstimator):
    """A diagonal-debiased U-statistic: the sum is divided by ``n(n - 1)``.

    A *sibling* of :class:`VStatisticEstimator`, and the reason this axis
    exists. The debiased estimate can be **negative** even where the population
    functional is nonnegative, so a consumer that takes a square root or a log
    of one, having type-checked against the other, produces ``nan`` with no
    static symptom.
    """


# --- Design. The grouping or pairing structure the number was computed over. ---


class Design:
    """Root of the design axis; asserts no structure over the observations."""


class UngroupedDesign(Design):
    """One undifferentiated sample, or two samples compared as wholes."""


class GroupedDesign(Design):
    """Observations partitioned into ``K`` labelled groups."""


# --- Hypothesis. The null a calibrated test is calibrated against. ---


class Hypothesis:
    """Root of the hypothesis axis; names no null."""


class Homogeneity(Hypothesis):
    """``H₀: P₁ = ⋯ = P_K`` — equality of the compared laws."""


@final
class NoGroupEffect(Homogeneity):
    """``H₀``: the group labels carry no distributional information.

    A subtype of :class:`Homogeneity`, and the widening is sound: exchangeability
    of the labels under a grouped design *is* equality of the group laws, so a
    DISCO null satisfies a homogeneity requirement while the reverse — reading a
    two-sample homogeneity null as a statement about ``K`` labelled groups —
    does not type-check.
    """


@final
class Independence(Hypothesis):
    """``H₀: P_{XY} = P_X ⊗ P_Y`` — an unrelated sibling of homogeneity."""


# --- Calibration. How the p-value was obtained. ---


class Calibration:
    """Root of the calibration axis; states nothing about the p-value's validity."""


@final
class ExactCalibration(Calibration):
    """Enumeration of the full null orbit; see known gap 4."""


@final
class ResampledCalibration(Calibration):
    """A Monte-Carlo resampling null, add-one corrected.

    ``inference`` owns the draw. The add-one correction is what keeps the
    p-value valid at finite resample counts; the marker records the family, not
    the correction, so a producer that omits it is not caught here.
    """


@final
class AsymptoticCalibration(Calibration):
    """A limiting-null reference distribution.

    An unrelated sibling of the two resampling families: a large-sample
    approximation is not a finite-sample guarantee, and neither direction of
    substitution is sound.
    """


# --- Statistic kind. What sort of number a realized statistic is. ---


class StatisticKind:
    """Root of the statistic-kind axis.

    The bound that makes "no metric law on a scalar" a pyrefly error rather
    than a convention: ``Metric``, ``Separation``, and every other law marker
    fails this bound, so :class:`RealizedStatistic` cannot be parameterized by
    one.
    """


class RatioStatistic(StatisticKind):
    """A ratio of two dispersion or variance quantities."""


class EnergyTwoSampleStatistic(StatisticKind):
    """Sample-size-scaled energy V-statistic for a two-sample design."""


@final
class DispersionIndex(RatioStatistic):
    """A share of total dispersion, such as the DISCO index ``R²_E = S_B / T``."""


@final
class VarianceRatioStatistic(RatioStatistic):
    """A degrees-of-freedom-scaled ratio, such as the DISCO F-ratio."""


# --- Construction origin extension. Reuses the `core/domains.py` axis. ---


class DecompositionOrigin(ConstructionOrigin):
    """A value obtained by decomposing a dispersion total over a design.

    Sits on :class:`jcor.core.domains.ConstructionOrigin` rather than starting a
    parallel axis, so one origin parameter carries energy, MMD, transport,
    direct, and decomposition provenance through both matrix and scalar
    containers.
    """


# --- Axis parameters. Variance names the flow it blocks; phantom otherwise. ---

#: **Covariant** numeric value type, in read-only field position — the same role
#: ``S`` plays in :class:`jcor.core.results.TestResult`. Deliberately
#: **unbounded**: jaxtyping's parameterized hints are runtime-generated classes
#: and beartype rejects ``Float[Array, ""]`` as violating a nominal ``Array``
#: bound while importing a specialized container (``core/results.py:53-58``).
#: Producing APIs pin the concrete array hint instead.
ValueT = TypeVar("ValueT", covariant=True)  # noqa: PLC0105

#: **Covariant** estimand. Blocks: a generic ``Estimand`` estimate supplied where
#: a ``TotalDispersion`` estimate is required, and — through nominal siblinghood
#: — a within-dispersion estimate supplied where the total is required.
EstimandT = TypeVar("EstimandT", bound=Estimand, covariant=True)  # noqa: PLC0105

#: **Covariant** estimator scheme. Blocks: a signed U-statistic supplied where a
#: V-statistic is required (and conversely), and an unspecified-scheme estimate
#: supplied where an empirical one is required.
EstimatorT = TypeVar("EstimatorT", bound=EstimatorScheme, covariant=True)  # noqa: PLC0105

#: **Covariant** design. Blocks: an ungrouped result supplied where a grouped
#: design is required, which is the substitution a DISCO consumer must not make.
DesignT = TypeVar("DesignT", bound=Design, covariant=True)  # noqa: PLC0105

#: **Covariant** hypothesis. Blocks: a homogeneity test supplied where an
#: independence test is required, and a generic ``Hypothesis`` result supplied
#: where a named null is required.
HypothesisT = TypeVar("HypothesisT", bound=Hypothesis, covariant=True)  # noqa: PLC0105

#: **Covariant** calibration. Blocks: an asymptotic p-value supplied where an
#: exact or resampled one is required — a large-sample approximation standing in
#: for a finite-sample guarantee.
CalibrationT = TypeVar("CalibrationT", bound=Calibration, covariant=True)  # noqa: PLC0105

#: **Covariant** statistic kind. Blocks: an F-ratio supplied where a dispersion
#: index is required, and — through the bound — any attempt to parameterize a
#: scalar statistic by a structural law.
StatisticT = TypeVar("StatisticT", bound=StatisticKind, covariant=True)  # noqa: PLC0105

#: **Covariant** origin of the *testing procedure*, distinct from the origin of
#: the statistic it calibrates: a permutation test over an energy statistic has
#: an energy statistic origin and a resampling procedure origin.
ProcedureOriginT = TypeVar(  # noqa: PLC0105
    "ProcedureOriginT", bound=ConstructionOrigin, covariant=True
)


# --- Provenance. Zero-field, hashable, static; never a pytree leaf. ---
#
# These are the module's **descriptors**: they carry the axes so the carriers
# below can carry one token each. A carrier holds at most two phantom
# parameters — its value type and a descriptor — and a new axis is added inside
# a descriptor rather than onto the carrier. Same rule as
# `jcor.core.matrices.MatrixContract` and `jcor.core.transforms.TransformContract`;
# recorded in `AGENTS.md` and exercised by the public typing and contract tests.
# Each descriptor
# gets a non-generic root so the carrier parameter has something to bind to.


@dataclass(frozen=True, slots=True)
class EstimateDescriptor:
    """Non-generic root for a complete static estimate provenance."""


@dataclass(frozen=True, slots=True)
class StatisticDescriptor:
    """Non-generic root for a complete static statistic provenance."""


@dataclass(frozen=True, slots=True)
class CalibrationDescriptor:
    """Non-generic root for a complete static calibration provenance."""


@final
@dataclass(frozen=True, slots=True)
class EstimateProvenance(
    EstimateDescriptor,
    Generic[EstimandT, EstimatorT, OriginT],  # noqa: UP046  # PEP 695 loses variance
):
    """What an estimate targets, by which scheme, from which construction.

    Zero fields by design: the content is entirely in the type parameters, so
    the object is hashable and can only ever ride as ``static`` pytree metadata.
    Held as a mandatory field of :class:`RealizedEstimate`, which is what makes
    provenance non-optional and pins the phantom parameters to a real field.
    """


@final
@dataclass(frozen=True, slots=True)
class StatisticProvenance(
    StatisticDescriptor,
    Generic[StatisticT, OriginT],  # noqa: UP046  # see variance
):
    """What kind of number a realized statistic is, and how it was constructed.

    Carries **no** estimand: a statistic need not estimate anything. A DISCO
    F-ratio is a ratio of dispersion estimates with a null distribution, not a
    plug-in for a population F. That asymmetry is the reason
    :class:`RealizedStatistic` is a separate carrier from
    :class:`RealizedEstimate` rather than one container with an optional field.
    """


@final
@dataclass(frozen=True, slots=True)
class CalibrationProvenance(
    CalibrationDescriptor,
    Generic[DesignT, HypothesisT, CalibrationT, ProcedureOriginT],  # noqa: UP046
):
    """The design, null, calibration family, and procedure origin of a test."""


#: **Covariant** descriptor parameters — the second slot of each carrier below.
#: Covariance is what preserves every widening the axes already declared: a
#: ``StatisticProvenance[DispersionIndex, DecompositionOrigin]`` flows into a
#: ``StatisticProvenance[RatioStatistic, ConstructionOrigin]`` requirement
#: exactly as it did when those two axes sat directly on the carrier.
EstimateProvenanceT = TypeVar(  # noqa: PLC0105
    "EstimateProvenanceT",
    bound=EstimateDescriptor,
    covariant=True,
)
StatisticProvenanceT = TypeVar(  # noqa: PLC0105
    "StatisticProvenanceT",
    bound=StatisticDescriptor,
    covariant=True,
)
CalibrationProvenanceT = TypeVar(  # noqa: PLC0105
    "CalibrationProvenanceT",
    bound=CalibrationDescriptor,
    covariant=True,
)


# --- Realized carriers. Arrays are leaves; provenance is static metadata. ---


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RealizedEstimate(
    Generic[ValueT, EstimateProvenanceT],  # noqa: UP046  # see variance
):
    """A number computed as an estimate of a named estimand.

    Two parameters carrying four axes: the value type, and an
    :class:`EstimateProvenance` naming the estimand, the scheme and the
    construction. The descriptor is not merely a bundle — it is the field, so
    the parameter is pinned to a real value rather than left phantom.

    Attributes:
        value: The estimate itself — a scalar array in the single-run case, or
            ``(*batch,)`` once a transform has batched it. The sole pytree leaf.
        provenance: Static estimand/scheme/origin record. A **static** pytree
            field, so it lands in the treedef's aux data and never reaches a
            compiled array tree.

    Examples:
        >>> type PooledDispersion = EstimateProvenance[
        ...     TotalDispersion, VStatisticEstimator, DecompositionOrigin
        ... ]
        >>> e: RealizedEstimate[float, PooledDispersion] = RealizedEstimate(
        ...     value=1.0, provenance=EstimateProvenance()
        ... )
        >>> jax.tree_util.tree_leaves(e)
        [1.0]

    """

    value: ValueT
    provenance: EstimateProvenanceT = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RealizedStatistic(Generic[ValueT, StatisticProvenanceT]):  # noqa: UP046
    """A number used as a test statistic, with its construction provenance.

    Not a distance. This carrier has no law parameter, no ``Separation`` field,
    and no route to either: :class:`StatisticProvenance` parameterized with a
    law marker fails :data:`StatisticT`'s bound, a law marker supplied directly
    as the descriptor fails :data:`StatisticProvenanceT`'s, and passing a
    :class:`jcor.core.typing.DMat` where one is required is a
    ``bad-argument-type``. Bundling the axes narrowed that surface rather than
    widening it.

    Attributes:
        value: The statistic, the sole pytree leaf.
        provenance: Static kind/origin record.

    """

    value: ValueT
    provenance: StatisticProvenanceT = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupDesign:
    """Group labels for a grouped design.

    Attributes:
        codes: Contiguous integer group code per observation, shape ``(n,)`` —
            or ``(*batch, n)`` once a transform has batched it.
        n_groups: Number of distinct groups. A **static** pytree field: it is a
            shape, so it must not become a traced leaf. Not validated against
            ``codes``; see the module docstring's known gap 3.

    """

    codes: Int[Array, "*batch n"]  # noqa: F722  # jaxtyping shape, not a forward ref
    n_groups: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResampledNull(Generic[ValueT, CalibrationT]):  # noqa: UP046  # see variance
    """Statistics drawn under the null, for a resampling calibration.

    A **sibling** of :class:`CalibratedTest`, not a field of it. An asymptotic
    calibration has no null sample, so folding this into the test container
    would force a ``None`` — and a container whose fields are optional for half
    its inhabitants is the universal container this module exists to avoid.

    Attributes:
        draws: Null statistics, shape ``(resamples,)`` — or
            ``(*batch, resamples)`` once a transform has batched it.
        n_resamples: Number of draws. A **static** pytree field.

    """

    draws: ValueT
    n_resamples: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CalibratedTest(
    Generic[  # noqa: UP046  # PEP 695 loses variance on phantom parameters
        ValueT,
        StatisticProvenanceT,
        CalibrationProvenanceT,
    ],
):
    """A typed statistic together with the calibration that produced its p-value.

    Composition, not flattening: the statistic keeps its own carrier and its own
    descriptor, and this container adds only what calibration contributes.
    ``StatisticProvenanceT`` appears here solely to be forwarded to the held
    :class:`RealizedStatistic`, which is why the statistic's provenance survives
    assignment into a generic consumer.

    Three parameters carrying seven axes. This is the carrier the parameter
    budget was written for: at the flattened spelling a single field annotation
    ran to seven lines in both ``inference/disco.py`` and
    ``inference/permutation.py``, and every axis added to *either* descriptor
    lengthened both.

    Attributes:
        statistic: The typed statistic that was calibrated.
        pvalue: The calibrated p-value. Not validated — a host comparison of a
            tracer raises ``TracerBoolConversionError``; see
            :mod:`jcor.core.results`.
        provenance: Static design/null/calibration/procedure record.

    """

    statistic: RealizedStatistic[ValueT, StatisticProvenanceT]
    pvalue: ValueT
    provenance: CalibrationProvenanceT = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AdditiveDispersionDecomposition(
    Generic[ValueT, EstimatorT, OriginT],  # noqa: UP046  # see variance
):
    """``T = S_W + S_B`` over a grouped design, with each part typed by its estimand.

    The generic carrier the ``association``-owned ``DiscoDecomposition``
    composes: it adds the derived
    ``RealizedStatistic[..., DispersionIndex, ...]`` and
    ``RealizedStatistic[..., VarianceRatioStatistic, ...]`` and pins the
    estimator scheme to :class:`VStatisticEstimator` — the cited DISCO
    ``g_alpha(A, A)`` keeps its zero diagonal and divides by ``n_g²``. The
    algorithm is not implemented here.

    Three components with three ``@final`` sibling estimands is what makes
    "cannot be constructed from arbitrary scalars" a static fact: each field
    demands a :class:`RealizedEstimate` at its own estimand, so a bare array,
    and a within estimate supplied as the total, are both pyrefly errors.

    Attributes:
        total: Estimate of ``T``.
        within: Estimate of ``S_W``.
        between: Estimate of ``S_B``.
        design: The grouping the decomposition is taken over.

    """

    total: RealizedEstimate[
        ValueT, EstimateProvenance[TotalDispersion, EstimatorT, OriginT]
    ]
    within: RealizedEstimate[
        ValueT, EstimateProvenance[WithinDispersion, EstimatorT, OriginT]
    ]
    between: RealizedEstimate[
        ValueT, EstimateProvenance[BetweenDispersion, EstimatorT, OriginT]
    ]
    design: GroupDesign

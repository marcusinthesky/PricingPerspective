"""Typed experiment and geometry registry at the pipeline I/O boundary.

The YAML document remains the canonical resolved configuration.  This module is
its runtime boundary: it validates references and cross-family semantics once,
then hands numerical code immutable model objects rather than stringly typed
parameter dictionaries.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,  # noqa: TC002
    ValidationError,
    field_validator,
    model_validator,
)

Normalization = Literal["raw", "unit_rows"]
ProviderKind = Literal["openrouter_parquet", "local_parquet", "encoder_vintage"]
ProviderStatus = Literal["canonical", "exploratory", "deprecated"]
PointMetric = Literal["euclidean", "cosine", "angular"]
PointDomain = Literal["euclidean", "unit_sphere"]
StatisticalFamily = Literal["energy", "mmd", "wasserstein"]
StatisticalValueSemantics = Literal[
    "statistical_distance", "squared_statistical_distance"
]
EstimatorName = Literal[
    "v_statistic",
    "u_statistic",
    "balanced_wasserstein_1",
    "balanced_wasserstein_2",
]
#: Transport order lives in the estimator, not in ``exponent``. ``exponent`` is
#: the energy ground exponent alpha in (0, 2); a Wasserstein geometry uses the
#: chord ground distance to the first power and raises the *transport* cost to
#: the order, so both members below declare ``exponent: 1.0``.
BALANCED_WASSERSTEIN_ESTIMATORS: frozenset[str] = frozenset(
    {"balanced_wasserstein_1", "balanced_wasserstein_2"}
)
ProblemName = Literal["target_projection", "source_barycentre"]
FeasibleSetName = Literal[
    "simplex_nonnegative",
    "affine_signed",
    "free_signed",
    "prescribed_source_weights",
]
TargetPolicy = Literal["leave_one_out", "explicit"]
PRESCRIBED_WEIGHT_TOLERANCE = 1e-10
TWO_SOURCE_COUNT = 2
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DVC_SAFE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TypedAnalysisError(ValueError):
    """Raised when the typed analysis registry is malformed."""


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _safe_registry_key(identity: str) -> str:
    """Return the DVC dot-path-safe spelling of one external identity."""
    return identity.replace(".", "_")


class ProviderConfig(BaseModel):
    """A model-backed or local embedding provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    kind: ProviderKind
    model_id: str = Field(min_length=1)
    vintage: str | None = Field(default=None, min_length=1)
    dimensions: PositiveInt
    normalization: Normalization
    source_hash: str | None = Field(default=None, min_length=1)
    status: ProviderStatus = "canonical"
    caveat: str | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> ProviderConfig:
        """Validate provider-specific vintage fields."""
        if self.kind == "encoder_vintage" and self.vintage is None:
            _invalid("encoder_vintage providers require vintage")
        if self.kind != "encoder_vintage" and self.vintage is not None:
            _invalid("vintage is only valid for encoder_vintage providers")
        if (
            self.source_hash is not None
            and SHA256_PATTERN.fullmatch(self.source_hash) is None
        ):
            _invalid("provider source_hash must be a lowercase SHA-256 digest")
        return self


class OpenRouterProvider(ProviderConfig):
    """Provider variant for OpenRouter-backed Parquet artifacts."""

    kind: Literal["openrouter_parquet"] = "openrouter_parquet"


class LocalParquetProvider(ProviderConfig):
    """Provider variant for local Parquet artifacts."""

    kind: Literal["local_parquet"] = "local_parquet"


class EncoderVintageProvider(ProviderConfig):
    """Provider variant for vintage-specific encoder artifacts."""

    kind: Literal["encoder_vintage"] = "encoder_vintage"


ProviderEntry = Annotated[
    OpenRouterProvider | LocalParquetProvider | EncoderVintageProvider,
    Field(discriminator="kind"),
]


class EmbeddingProvider(ProviderConfig):
    """Named provider entry with a stable registry identity."""


class RepresentationSpec(BaseModel):
    """Representation transform applied before a point-distance computation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    transform: Literal["identity", "l2_normalize_rows"] = "identity"
    normalization: Normalization
    dimension: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_normalization(self) -> RepresentationSpec:
        """Validate transform and normalization agreement."""
        if self.transform == "l2_normalize_rows" and self.normalization != "unit_rows":
            _invalid("l2_normalize_rows requires unit_rows normalization")
        if self.dimension is not None and self.transform != "l2_normalize_rows":
            _invalid("dimension-specific representations require l2_normalize_rows")
        return self


class PointDistanceSpec(BaseModel):
    """Point-level geometry used as the ground distance for distributions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distance_id: str = Field(min_length=1)
    kind: Literal["point"] = "point"
    metric: PointMetric
    domain: PointDomain

    @model_validator(mode="after")
    def validate_domain(self) -> PointDistanceSpec:
        """Validate metric domain compatibility."""
        if self.metric in {"cosine", "angular"} and self.domain != "unit_sphere":
            _invalid(f"{self.metric} requires unit_sphere domain")
        return self


class SampleWindowSpec(BaseModel):
    """Observation-date window restricting which provider rows enter a cloud.

    A window is a *sample-design* fact, not a geometry fact: it changes which
    observations are averaged, never the functional averaging them.  It earns a
    registry entry rather than an inline pair of dates because two stages that
    claim the same window must be unable to disagree about it — Paper 1's
    selection cutoff and Paper 3's 2020 point-in-time vintage are the same
    window, and the shared component they read is only reusable if that is
    stated once.

    Bounds are inclusive on both ends and either may be omitted for an open
    side.  ``start`` is normally the corpus start, in which case it is
    numerically a no-op; declaring it anyway keeps the artifact honest about
    what it covers rather than implicitly inheriting whatever the corpus
    happens to begin with.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: str = Field(min_length=1)
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> SampleWindowSpec:
        """Reject an unbounded or inverted window."""
        if self.start is None and self.end is None:
            _invalid(f"window {self.window_id!r} must bound at least one side")
        if self.start is not None and self.end is not None and self.start > self.end:
            _invalid(f"window {self.window_id!r} starts after it ends")
        return self


class StatisticalDistanceSpec(BaseModel):
    """Distribution-level functional with explicit estimator semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distance_id: str = Field(min_length=1)
    kind: Literal["statistical"] = "statistical"
    family: StatisticalFamily
    ground_distance: str = Field(min_length=1)
    exponent: float = Field(gt=0.0, lt=2.0)
    estimator: EstimatorName
    normalization: Literal["unhalved", "halved", "rooted"] = "unhalved"
    kernel: Literal["linear", "energy", "rbf"] | None = None
    bandwidth: float | None = Field(default=None, gt=0.0)
    bandwidth_rule: Literal["fixed", "median_positive_pairwise"] = "fixed"
    sample_design: Literal[
        "full_sample", "calibration", "point_in_time", "return_aligned"
    ] = "full_sample"
    sample_size: PositiveInt | None = None
    sample_window: str | None = Field(default=None, min_length=1)

    @property
    def value_semantics(self) -> StatisticalValueSemantics:
        """Name the persisted scale without exposing family-specific formulas."""
        if self.normalization == "rooted":
            return "statistical_distance"
        return "squared_statistical_distance"

    @model_validator(mode="after")
    def validate_family(self) -> StatisticalDistanceSpec:  # noqa: C901
        """Validate estimator, kernel, and normalization semantics."""
        if self.family == "energy":
            if self.estimator not in {"v_statistic", "u_statistic"}:
                _invalid("energy distance requires v_statistic or u_statistic")
            if self.normalization != "unhalved":
                _invalid("energy distance currently emits unhalved functionals")
        if self.family == "mmd":
            if self.kernel is None:
                _invalid("mmd distance requires kernel")
            if self.normalization not in {"halved", "unhalved", "rooted"}:
                _invalid("mmd distance has an unsupported normalization")
        if self.family == "wasserstein":
            if self.estimator not in BALANCED_WASSERSTEIN_ESTIMATORS:
                _invalid("wasserstein requires a balanced_wasserstein_N estimator")
            if self.normalization != "rooted":
                _invalid("wasserstein distance currently emits rooted values")
        if (
            self.kernel == "rbf"
            and self.bandwidth is None
            and self.bandwidth_rule == "fixed"
        ):
            _invalid("fixed-bandwidth RBF MMD requires positive bandwidth")
        if (
            self.kernel == "rbf"
            and self.bandwidth is not None
            and self.bandwidth_rule != "fixed"
        ):
            _invalid("derived-bandwidth RBF MMD must not also provide bandwidth")
        if self.kernel is not None and self.family != "mmd":
            _invalid("kernel is only valid for mmd distances")
        return self

    @model_validator(mode="after")
    def validate_sample_design(self) -> StatisticalDistanceSpec:
        """Bind the declared sample design to the presence of a window.

        Kept separate from :meth:`validate_family` because it constrains the
        sample, not the functional.  The coupling is total in both directions:
        a ``full_sample`` geometry that also names a window is claiming two
        incompatible samples, and any restricted design without a window is a
        restriction nothing can reproduce.
        """
        if self.sample_design == "full_sample":
            if self.sample_window is not None:
                _invalid(
                    f"{self.distance_id!r} is full_sample but names sample_window "
                    f"{self.sample_window!r}"
                )
        elif self.sample_window is None:
            _invalid(
                f"{self.distance_id!r} declares sample_design "
                f"{self.sample_design!r} without a sample_window"
            )
        return self


DistanceEntry = Annotated[
    PointDistanceSpec | StatisticalDistanceSpec,
    Field(discriminator="kind"),
]


class FeasibleSetSpec(BaseModel):
    """Constraint family for a barycentre or target projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feasible_set_id: str = Field(min_length=1)
    kind: FeasibleSetName
    lower_bound: float | None = Field(default=None, ge=0.0)
    source_weights: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def validate_constraints(self) -> FeasibleSetSpec:
        """Validate feasible-set-specific bounds and weights."""
        if self.kind == "simplex_nonnegative" and self.lower_bound not in {None, 0.0}:
            _invalid("ordinary simplex lower_bound must be zero or omitted")
        if self.kind == "prescribed_source_weights":
            if not self.source_weights:
                _invalid("prescribed_source_weights requires source_weights")
            if any(weight < 0 for weight in self.source_weights):
                _invalid("prescribed source weights must be nonnegative")
            if abs(sum(self.source_weights) - 1.0) > PRESCRIBED_WEIGHT_TOLERANCE:
                _invalid("prescribed source weights must sum to one")
        elif self.source_weights is not None:
            _invalid("source_weights only applies to prescribed sources")
        return self


class SimplexFeasibleSet(FeasibleSetSpec):
    """Nonnegative simplex constraint variant."""

    kind: Literal["simplex_nonnegative"] = "simplex_nonnegative"


class AffineSignedFeasibleSet(FeasibleSetSpec):
    """Affine signed constraint variant."""

    kind: Literal["affine_signed"] = "affine_signed"


class FreeSignedFeasibleSet(FeasibleSetSpec):
    """Free signed constraint variant."""

    kind: Literal["free_signed"] = "free_signed"


class PrescribedSourceWeights(FeasibleSetSpec):
    """Prescribed source-measure weights constraint variant."""

    kind: Literal["prescribed_source_weights"] = "prescribed_source_weights"


FeasibleSetEntry = Annotated[
    SimplexFeasibleSet
    | AffineSignedFeasibleSet
    | FreeSignedFeasibleSet
    | PrescribedSourceWeights,
    Field(discriminator="kind"),
]


class BarycentreArm(BaseModel):
    """Typed geometry, feasible set, solver, and target policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1)
    problem: ProblemName
    geometry: str = Field(min_length=1)
    feasible_set: str = Field(min_length=1)
    solver: Literal[
        "energy_qp",
        "mmd_qp",
        "wasserstein_ot",
        "wasserstein_free_support",
        "wasserstein_w1_target",
        "wasserstein_w2_target",
        "pgd",
    ]
    target_policy: TargetPolicy
    tol: float | None = Field(default=None, gt=0.0)
    maxiter: PositiveInt | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_problem(self) -> BarycentreArm:  # noqa: C901
        """Validate source-barycentre solver, policy, and solver-budget fields."""
        if self.problem == "source_barycentre" and self.target_policy != "explicit":
            _invalid("source_barycentre requires explicit target policy")
        if self.problem == "source_barycentre" and self.solver not in {
            "wasserstein_ot",
            "wasserstein_free_support",
        }:
            _invalid(
                "source_barycentre must use wasserstein_ot or wasserstein_free_support"
            )
        if self.problem == "source_barycentre" and self.solver in {
            "wasserstein_w1_target",
            "wasserstein_w2_target",
        }:
            _invalid("Wasserstein target solvers cannot declare source_barycentre")
        if self.problem == "target_projection" and self.solver in {
            "wasserstein_ot",
            "wasserstein_free_support",
        }:
            _invalid("source Wasserstein solvers cannot declare target_projection")
        if self.problem == "target_projection" and self.solver in {
            "wasserstein_w1_target",
            "wasserstein_w2_target",
        }:
            if self.target_policy != "leave_one_out":
                _invalid("Wasserstein target projections require leave_one_out policy")
            if self.feasible_set != "simplex_nonnegative":
                _invalid("Wasserstein target projections require simplex_nonnegative")
        if self.solver == "wasserstein_ot":
            # The two-source solver is a closed form: an optimal assignment plus
            # displacement interpolation. There is nothing to iterate, so a
            # declared budget would be inert and is rejected rather than ignored.
            if self.tol is not None or self.maxiter is not None:
                _invalid("wasserstein_ot arms take no iterative solver budget")
        elif self.tol is None or self.maxiter is None:
            # wasserstein_free_support falls here deliberately: past two sources
            # the barycentre is an alternating minimization to a local optimum,
            # so its budget must be declared like any other iterative arm.
            _invalid(f"{self.solver} arms require an explicit tol and maxiter")
        return self


def iteration_budget(arm: BarycentreArm) -> tuple[float, int]:
    """Return an iterative solver's budget; closed-form arms have none.

    Keep consumer helpers outside registry classes: Semflow can then exclude
    them from commands that only load and validate the shared configuration.
    """
    if arm.tol is None or arm.maxiter is None:
        _invalid(f"{arm.solver} has no iterative solver budget")
    return arm.tol, arm.maxiter


class TypedAnalysisRegistry(BaseModel):
    """Complete validated registry for typed analysis stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: dict[str, ProviderEntry]
    representations: dict[str, RepresentationSpec]
    distances: dict[str, DistanceEntry]
    feasible_sets: dict[str, FeasibleSetEntry]
    barycentres: dict[str, BarycentreArm]
    # Defaulted, and deliberately absent from require_nonempty below: a registry
    # whose geometries are all full-sample declares no windows, and every
    # existing test fixture predates this section.
    windows: dict[str, SampleWindowSpec] = {}

    @field_validator(
        "providers", "representations", "distances", "feasible_sets", "barycentres"
    )
    @classmethod
    def require_nonempty(cls, value: dict[str, object]) -> dict[str, object]:
        """Reject empty typed-analysis registry sections."""
        if not value:
            _invalid("typed-analysis registry sections must not be empty")
        return value

    @model_validator(mode="after")
    def validate_references(  # noqa: C901, PLR0912, PLR0915
        self,
    ) -> TypedAnalysisRegistry:
        """Validate cross-section IDs and geometry/solver compatibility."""
        provider_ids = {provider.provider_id for provider in self.providers.values()}
        provider_by_id = {
            provider.provider_id: provider for provider in self.providers.values()
        }
        point_ids = {
            key
            for key, value in self.distances.items()
            if isinstance(value, PointDistanceSpec)
        }
        statistical_ids = {
            key
            for key, value in self.distances.items()
            if isinstance(value, StatisticalDistanceSpec)
        }
        for key, provider in self.providers.items():
            if key != _safe_registry_key(provider.provider_id):
                _invalid(f"provider key {key!r} disagrees with provider_id")
        for key, representation in self.representations.items():
            if key != _safe_registry_key(representation.representation_id):
                _invalid(f"representation key {key!r} disagrees with representation_id")
            if representation.provider_id not in provider_ids:
                _invalid(f"representation {key!r} references unknown provider")
            provider = provider_by_id[representation.provider_id]
            if (
                representation.dimension is not None
                and representation.dimension > provider.dimensions
            ):
                _invalid(
                    f"representation {key!r} dimension exceeds provider native width"
                )
            if (
                representation.transform == "identity"
                and representation.normalization != provider.normalization
            ):
                _invalid(
                    f"identity representation {key!r} must preserve provider "
                    "normalization"
                )
        for key, window in self.windows.items():
            if key != window.window_id:
                _invalid(f"window key {key!r} disagrees with window_id")
        for key, distance in self.distances.items():
            if key != distance.distance_id:
                _invalid(f"distance key {key!r} disagrees with distance_id")
            if not isinstance(distance, StatisticalDistanceSpec):
                continue
            if distance.ground_distance not in point_ids:
                _invalid(
                    f"statistical distance {key!r} references non-point ground distance"
                )
            if (
                distance.sample_window is not None
                and distance.sample_window not in self.windows
            ):
                _invalid(
                    f"statistical distance {key!r} references unknown sample window "
                    f"{distance.sample_window!r}"
                )
        for key, arm in self.barycentres.items():
            if key != arm.arm_id:
                _invalid(f"barycentre key {key!r} disagrees with arm_id")
            if arm.geometry not in statistical_ids:
                _invalid(f"barycentre {key!r} references unknown statistical geometry")
            if arm.feasible_set not in self.feasible_sets:
                _invalid(f"barycentre {key!r} references unknown feasible set")
            geometry = self.distances[arm.geometry]
            if not isinstance(geometry, StatisticalDistanceSpec):
                _invalid(f"barycentre {key!r} has non-statistical geometry")
            if geometry.family == "energy" and arm.solver not in {"energy_qp", "pgd"}:
                _invalid(f"energy barycentre {key!r} has incompatible solver")
            if geometry.family == "mmd" and arm.solver != "mmd_qp":
                _invalid(f"mmd barycentre {key!r} has incompatible solver")
            if geometry.family == "mmd" and (
                arm.problem != "target_projection"
                or arm.target_policy != "leave_one_out"
                or self.feasible_sets[arm.feasible_set].kind != "simplex_nonnegative"
            ):
                _invalid(
                    f"mmd barycentre {key!r} requires a leave-one-out simplex "
                    "target projection"
                )
            if geometry.family == "wasserstein":
                if arm.problem == "source_barycentre" and arm.solver not in {
                    "wasserstein_ot",
                    "wasserstein_free_support",
                }:
                    _invalid(
                        f"wasserstein source barycentre {key!r} has incompatible solver"
                    )
                if arm.problem == "target_projection" and arm.solver not in {
                    "wasserstein_w1_target",
                    "wasserstein_w2_target",
                }:
                    _invalid(
                        f"wasserstein target projection {key!r} has incompatible solver"
                    )
                if arm.solver == "wasserstein_w1_target" and geometry.estimator != (
                    "balanced_wasserstein_1"
                ):
                    _invalid(f"wasserstein W1 target {key!r} has non-W1 geometry")
                if arm.solver == "wasserstein_w2_target" and geometry.estimator != (
                    "balanced_wasserstein_2"
                ):
                    _invalid(f"wasserstein W2 target {key!r} has non-W2 geometry")
            # The closed-form solver is exact only for two sources. Guard the
            # arity here, at registry load, so a third prescribed weight cannot
            # reach the kernel and be silently rejected at stage runtime.
            if arm.solver == "wasserstein_ot":
                prescribed = self.feasible_sets[arm.feasible_set].source_weights
                if prescribed is not None and len(prescribed) != TWO_SOURCE_COUNT:
                    _invalid(
                        f"wasserstein barycentre {key!r} uses the exact two-source "
                        "solver but prescribes "
                        f"{len(prescribed)} source weights; declare "
                        "wasserstein_free_support instead"
                    )
        for key, feasible_set in self.feasible_sets.items():
            if key != feasible_set.feasible_set_id:
                _invalid(f"feasible-set key {key!r} disagrees with feasible_set_id")
        return self

    def ground_distance_for(
        self, geometry: StatisticalDistanceSpec
    ) -> PointDistanceSpec:
        """Resolve the point ground distance a statistical geometry declares.

        The numerical kernels must never re-derive a metric from an ID string:
        the ground metric and domain are registry facts, and substituting one
        for another silently changes the geometry an artifact claims to carry.
        """
        ground = self.distances.get(geometry.ground_distance)
        if not isinstance(ground, PointDistanceSpec):
            _invalid(
                f"{geometry.distance_id!r} references non-point ground distance "
                f"{geometry.ground_distance!r}"
            )
        return ground

    def window_for(self, geometry: StatisticalDistanceSpec) -> SampleWindowSpec | None:
        """Resolve the observation window a statistical geometry declares.

        Returns ``None`` for a full-sample geometry.  Mirrors
        :meth:`ground_distance_for`: a numerical kernel must never infer a
        window from an ID string, because substituting one silently changes
        which observations an artifact claims to have averaged.
        """
        if geometry.sample_window is None:
            return None
        window = self.windows.get(geometry.sample_window)
        if window is None:
            _invalid(
                f"{geometry.distance_id!r} references unknown sample window "
                f"{geometry.sample_window!r}"
            )
        return window

    def barycentre_for(
        self, arm_id: str
    ) -> tuple[BarycentreArm, StatisticalDistanceSpec]:
        """Resolve one barycentre arm and its registered statistical geometry."""
        arm = self.barycentres.get(arm_id)
        if arm is None:
            _invalid(f"unknown barycentre arm {arm_id!r}")
        geometry = self.distances.get(arm.geometry)
        if not isinstance(geometry, StatisticalDistanceSpec):
            _invalid(f"barycentre {arm_id!r} references a non-statistical geometry")
        return arm, geometry


class TypedAnalysisConfig(TypedAnalysisRegistry):
    """Top-level typed-analysis configuration parsed from ``params.yaml``."""


class ScientificSelectors(BaseModel):
    """Canonical scientific choices that DVC must track as parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_model_key: str = Field(pattern=DVC_SAFE_KEY_PATTERN.pattern)
    baseline_model: str = Field(min_length=1)
    typed_distance_provider_key: str = Field(pattern=DVC_SAFE_KEY_PATTERN.pattern)
    typed_distance_provider: str = Field(min_length=1)
    typed_distance_representation_key: str = Field(pattern=DVC_SAFE_KEY_PATTERN.pattern)
    typed_distance_representation: str = Field(min_length=1)
    typed_distance_id: str = Field(min_length=1)


class _RosterCell(BaseModel):
    """Provider and representation identity shared by every DVC roster cell.

    Keys are DVC dot-path-safe registry keys; the IDs are the external
    identities that name artifact folders. Both are authored so ``dvc.yaml`` can
    interpolate either without string rewriting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_key: str = Field(pattern=DVC_SAFE_KEY_PATTERN.pattern)
    provider_id: str = Field(min_length=1)
    representation_key: str = Field(pattern=DVC_SAFE_KEY_PATTERN.pattern)
    representation_id: str = Field(min_length=1)


class TypedGeometryComponentCell(_RosterCell):
    """One full-sample raw energy or MMD geometry component cell."""

    geometry_id: str = Field(min_length=1)


class TypedDistanceCell(_RosterCell):
    """One full-sample statistical-distance matrix cell."""

    distance_id: str = Field(min_length=1)


class TypedWindowedDistanceCell(TypedDistanceCell):
    """One final statistical-distance cell with an exact observation window."""

    window_key: str = Field(pattern=DVC_SAFE_KEY_PATTERN.pattern)


class BarycentreCell(_RosterCell):
    """One barycentre cell: a registry arm at one provider representation."""

    arm_id: str = Field(min_length=1)
    geometry_id: str = Field(min_length=1)


class DvcScientificConfig(BaseModel):
    """Cross-registry contract for the scientific portions of ``params.yaml``.

    Each roster lists exactly the cells the DVC graph materializes. Rosters are
    validated cell by cell against the typed registry; they are not required to
    cover every registered provider or representation, because the registry is
    the declared API and the rosters are what is consumed.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    typed_analysis: TypedAnalysisConfig
    baseline_embedding_models: tuple[str, ...]
    embedding_models: tuple[str, ...]
    symbols: tuple[str, ...]
    market_symbol_exclusions: tuple[str, ...]
    market_symbols: tuple[str, ...]
    scientific_selectors: ScientificSelectors
    typed_geometry_component_specs: dict[str, TypedGeometryComponentCell]
    typed_distance_specs: dict[str, TypedDistanceCell]
    typed_distance_windowed_specs: dict[str, TypedWindowedDistanceCell] = {}
    barycentre_specs: dict[str, BarycentreCell]

    @model_validator(mode="after")
    def validate_rosters(self) -> DvcScientificConfig:
        """Validate roster identities, references, and selectors."""
        self._validate_model_and_market_rosters()
        components = self._validate_component_roster()
        distances = self._validate_distance_roster(components)
        self._validate_windowed_distance_roster()
        self._validate_barycentre_roster(components)
        self._validate_selectors(distances)
        return self

    def _validate_model_and_market_rosters(self) -> None:
        """Check model partitions and the exact excluded-symbol projection."""
        baseline = set(self.baseline_embedding_models)
        models = set(self.embedding_models)
        if len(baseline) != len(self.baseline_embedding_models):
            _invalid("baseline_embedding_models contains duplicates")
        if len(models) != len(self.embedding_models):
            _invalid("embedding_models contains duplicates")
        if baseline - models:
            _invalid("baseline_embedding_models must be a subset of embedding_models")
        selectors = self.scientific_selectors
        if selectors.baseline_model not in baseline:
            _invalid("scientific baseline_model is not in baseline_embedding_models")
        if selectors.baseline_model_key != _safe_registry_key(selectors.baseline_model):
            _invalid("scientific baseline_model_key disagrees with baseline_model")
        expected_market = tuple(
            symbol
            for symbol in self.symbols
            if symbol not in set(self.market_symbol_exclusions)
        )
        if self.market_symbols != expected_market:
            _invalid("market_symbols must equal symbols minus market_symbol_exclusions")
        if len(set(self.symbols)) != len(self.symbols):
            _invalid("symbols contains duplicates")
        if not set(self.market_symbol_exclusions) <= set(self.symbols):
            _invalid("market_symbol_exclusions must be drawn from symbols")

    def _validate_cell(self, roster: str, key: str, cell: _RosterCell) -> None:
        """Resolve one cell's provider and representation against the registry."""
        if DVC_SAFE_KEY_PATTERN.fullmatch(key) is None:
            _invalid(f"{roster} cell key {key!r} is not DVC-safe")
        provider = self.typed_analysis.providers.get(cell.provider_key)
        if provider is None or provider.provider_id != cell.provider_id:
            _invalid(f"{roster} cell {key!r} has invalid provider")
        representation = self.typed_analysis.representations.get(
            cell.representation_key
        )
        if (
            representation is None
            or representation.representation_id != cell.representation_id
            or representation.provider_id != cell.provider_id
        ):
            _invalid(f"{roster} cell {key!r} has invalid representation")

    def _statistical(
        self, roster: str, key: str, distance_id: str
    ) -> StatisticalDistanceSpec:
        """Resolve one cell's statistical geometry."""
        distance = self.typed_analysis.distances.get(distance_id)
        if not isinstance(distance, StatisticalDistanceSpec):
            _invalid(f"{roster} cell {key!r} has invalid geometry")
        return distance

    def _validate_component_roster(self) -> set[tuple[str, str, str]]:
        """Check full-sample energy/MMD component cells; return their identities."""
        roster = "typed_geometry_component_specs"
        identities: set[tuple[str, str, str]] = set()
        for key, cell in self.typed_geometry_component_specs.items():
            self._validate_cell(roster, key, cell)
            geometry = self._statistical(roster, key, cell.geometry_id)
            if geometry.family == "wasserstein":
                _invalid(f"{roster} cell {key!r} has no reusable Wasserstein component")
            if geometry.sample_window is not None:
                _invalid(f"{roster} cell {key!r} must be a full-sample geometry")
            identities.add((cell.provider_id, cell.representation_id, cell.geometry_id))
        if len(identities) != len(self.typed_geometry_component_specs):
            _invalid(f"{roster} contains duplicate identities")
        return identities

    def _validate_distance_roster(
        self, components: set[tuple[str, str, str]]
    ) -> set[tuple[str, str, str]]:
        """Check full-sample distance cells; return their identities.

        An energy or MMD matrix is derived from its cached component, so its
        component cell must be on the component roster.
        """
        roster = "typed_distance_specs"
        identities: set[tuple[str, str, str]] = set()
        for key, cell in self.typed_distance_specs.items():
            self._validate_cell(roster, key, cell)
            distance = self._statistical(roster, key, cell.distance_id)
            if distance.sample_window is not None:
                _invalid(
                    f"{roster} cell {key!r} is windowed; windowed cells belong to "
                    "typed_distance_windowed_specs"
                )
            identity = (cell.provider_id, cell.representation_id, cell.distance_id)
            if distance.family in {"energy", "mmd"} and identity not in components:
                _invalid(f"{roster} cell {key!r} has no geometry component cell")
            identities.add(identity)
        if len(identities) != len(self.typed_distance_specs):
            _invalid(f"{roster} contains duplicate identities")
        return identities

    def _validate_windowed_distance_roster(self) -> None:
        """Check point-in-time W2 final-distance cells."""
        selectors = self.scientific_selectors
        actual_distances = {
            self._validate_windowed_distance_cell(key, cell)
            for key, cell in self.typed_distance_windowed_specs.items()
        }
        if len(actual_distances) != len(self.typed_distance_windowed_specs):
            _invalid("windowed typed distance roster contains duplicate identities")
        pit_distance_ids = {
            key
            for key, distance in self.typed_analysis.distances.items()
            if isinstance(distance, StatisticalDistanceSpec)
            and distance.sample_design == "point_in_time"
            and distance.family == "wasserstein"
        }
        provider_representations = {
            (provider_id, representation_id)
            for provider_id, representation_id, _distance_id in actual_distances
        }
        provider_representations.add(
            (selectors.baseline_model, selectors.typed_distance_representation)
        )
        expected_distances = {
            (provider_id, representation_id, distance_id)
            for provider_id, representation_id in provider_representations
            for distance_id in pit_distance_ids
        }
        if actual_distances != expected_distances:
            _invalid("windowed typed distance roster differs from PIT contract")

    def _validate_windowed_distance_cell(
        self, key: str, cell: TypedWindowedDistanceCell
    ) -> tuple[str, str, str]:
        """Resolve one point-in-time W2 cell to its external identity."""
        roster = "windowed distance"
        self._validate_cell(roster, key, cell)
        distance = self._statistical(roster, key, cell.distance_id)
        if distance.family != "wasserstein":
            _invalid(f"windowed distance cell {key!r} is not Wasserstein")
        if distance.sample_window != cell.window_key:
            _invalid(f"windowed distance cell {key!r} disagrees with distance window")
        window = self.typed_analysis.windows[cell.window_key]
        year_match = re.search(r"_pit_(\d{4})$", cell.distance_id)
        if (
            year_match is None
            or window.end is None
            or window.end.year != int(year_match.group(1))
        ):
            _invalid(
                f"windowed distance cell {key!r} distance year disagrees with window"
            )
        return (cell.provider_id, cell.representation_id, cell.distance_id)

    def _validate_barycentre_roster(
        self, components: set[tuple[str, str, str]]
    ) -> None:
        """Check barycentre cells against their registry arms."""
        roster = "barycentre_specs"
        identities: set[tuple[str, str, str]] = set()
        for key, cell in self.barycentre_specs.items():
            self._validate_cell(roster, key, cell)
            arm = self.typed_analysis.barycentres.get(cell.arm_id)
            if arm is None:
                _invalid(f"{roster} cell {key!r} names no registry arm")
            if arm.geometry != cell.geometry_id:
                _invalid(f"{roster} cell {key!r} disagrees with registry arm")
            geometry = self._statistical(roster, key, cell.geometry_id)
            identity = (cell.provider_id, cell.representation_id, cell.geometry_id)
            if geometry.family in {"energy", "mmd"} and identity not in components:
                _invalid(f"{roster} cell {key!r} has no geometry component cell")
            identities.add((cell.provider_id, cell.representation_id, cell.arm_id))
        if len(identities) != len(self.barycentre_specs):
            _invalid(f"{roster} contains duplicate identities")

    def _validate_selectors(self, distances: set[tuple[str, str, str]]) -> None:
        """Resolve the canonical selector key/ID pairs."""
        selectors = self.scientific_selectors
        selected_provider = self.typed_analysis.providers.get(
            selectors.typed_distance_provider_key
        )
        if selected_provider is None or selected_provider.provider_id != (
            selectors.typed_distance_provider
        ):
            _invalid("canonical typed distance provider key/ID does not resolve")
        if selectors.typed_distance_provider != selectors.baseline_model:
            _invalid("canonical typed distance provider must equal the baseline model")
        representation = self.typed_analysis.representations.get(
            selectors.typed_distance_representation_key
        )
        if representation is None or representation.representation_id != (
            selectors.typed_distance_representation
        ):
            _invalid("canonical typed distance representation is unknown")
        if representation.provider_id != selectors.typed_distance_provider:
            _invalid("canonical typed distance representation has the wrong provider")
        canonical = (
            selectors.typed_distance_provider,
            selectors.typed_distance_representation,
            selectors.typed_distance_id,
        )
        if canonical not in distances:
            _invalid("canonical typed distance cell is not in typed_distance_specs")


def load_dvc_scientific_config(path: Path) -> DvcScientificConfig:
    """Parse and cross-validate the scientific portions of ``params.yaml``."""
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    try:
        return DvcScientificConfig.model_validate(raw)
    except (TypeError, ValidationError) as exc:
        message = f"{path}: scientific DVC configuration validation failed: {exc}"
        raise TypedAnalysisError(message) from exc


def load_typed_analysis(path: Path) -> TypedAnalysisConfig:
    """Parse the typed-analysis subtree from the canonical YAML document."""
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    try:
        return TypedAnalysisConfig.model_validate(raw["typed_analysis"])
    except (KeyError, TypeError) as exc:
        message = f"{path}: params.yaml is missing typed_analysis"
        raise TypedAnalysisError(message) from exc
    except ValidationError as exc:
        message = f"{path}: typed_analysis validation failed: {exc}"
        raise TypedAnalysisError(message) from exc


def canonical_typed_analysis(config: TypedAnalysisConfig) -> str:
    """Serialize the resolved typed registry independently of YAML key order."""
    return json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _scoped_typed_analysis(
    config: TypedAnalysisConfig,
    *,
    provider_ids: Iterable[str] | None,
    representation_ids: Iterable[str] | None,
    distance_ids: Iterable[str] | None,
    feasible_set_ids: Iterable[str] | None,
    arm_ids: Iterable[str] | None,
) -> dict[str, object]:
    """Select the resolved registry closure that one artifact actually uses."""
    resolved = config.model_dump(mode="json")
    if all(
        selection is None
        for selection in (
            provider_ids,
            representation_ids,
            distance_ids,
            feasible_set_ids,
            arm_ids,
        )
    ):
        return resolved

    providers = set(provider_ids or ())
    representations = set(representation_ids or ())
    distances = set(distance_ids or ())
    feasible_sets = set(feasible_set_ids or ())
    barycentres = set(arm_ids or ())

    for representation_id in representations:
        representation_key = next(
            key
            for key, value in resolved["representations"].items()  # type: ignore[union-attr]
            if key == representation_id
            or value["representation_id"] == representation_id  # type: ignore[index]
        )
        providers.add(
            resolved["representations"][representation_key]["provider_id"]  # type: ignore[index]
        )
    for arm_id in barycentres:
        arm = resolved["barycentres"][arm_id]  # type: ignore[index]
        distances.add(arm["geometry"])
        feasible_sets.add(arm["feasible_set"])
    for distance_id in tuple(distances):
        ground_distance = resolved["distances"][distance_id].get("ground_distance")  # type: ignore[index]
        if ground_distance is not None:
            distances.add(ground_distance)
    # Last, so it sees geometries reached directly and via a barycentre arm. A
    # window that fell out of the closure would leave two components with
    # different samples sharing one parameter identity.
    windows: set[str] = set()
    for distance_id in distances:
        entry = resolved["distances"].get(distance_id)  # type: ignore[union-attr]
        window_id = entry.get("sample_window") if isinstance(entry, dict) else None
        if window_id is not None:
            windows.add(cast("str", window_id))

    def pick(section: str, keys: set[str], identity_field: str) -> dict[str, object]:
        values = resolved[section]
        return {
            key: value
            for key, value in values.items()  # type: ignore[union-attr]
            if key in keys or value[identity_field] in keys  # type: ignore[index]
        }

    return {
        "providers": pick("providers", providers, "provider_id"),
        "representations": pick(
            "representations", representations, "representation_id"
        ),
        "distances": pick("distances", distances, "distance_id"),
        "feasible_sets": pick("feasible_sets", feasible_sets, "feasible_set_id"),
        "barycentres": pick("barycentres", barycentres, "arm_id"),
        "windows": pick("windows", windows, "window_id"),
    }


def typed_analysis_parameter_identity(
    path: Path,
    *,
    provider_ids: Iterable[str] | None = None,
    representation_ids: Iterable[str] | None = None,
    distance_ids: Iterable[str] | None = None,
    feasible_set_ids: Iterable[str] | None = None,
    arm_ids: Iterable[str] | None = None,
) -> str:
    """Return a stable SHA-256 identity for a full or scoped registry closure."""
    config = load_typed_analysis(path)
    scoped = _scoped_typed_analysis(
        config,
        provider_ids=provider_ids,
        representation_ids=representation_ids,
        distance_ids=distance_ids,
        feasible_set_ids=feasible_set_ids,
        arm_ids=arm_ids,
    )
    payload = json.dumps(
        scoped,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BarycentreArm",
    "BarycentreCell",
    "DvcScientificConfig",
    "EmbeddingProvider",
    "FeasibleSetSpec",
    "PointDistanceSpec",
    "RepresentationSpec",
    "SampleWindowSpec",
    "StatisticalDistanceSpec",
    "TypedAnalysisConfig",
    "TypedAnalysisError",
    "TypedAnalysisRegistry",
    "TypedDistanceCell",
    "TypedGeometryComponentCell",
    "TypedWindowedDistanceCell",
    "canonical_typed_analysis",
    "iteration_budget",
    "load_dvc_scientific_config",
    "load_typed_analysis",
    "typed_analysis_parameter_identity",
]

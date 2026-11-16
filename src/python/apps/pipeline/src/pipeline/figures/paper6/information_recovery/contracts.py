"""Typed saved-result projections for the information-recovery publication."""

from pydantic import BaseModel


class RecoveryCell(BaseModel):
    """A saved recovery summary; missing values remain missing in exhibits."""

    bias: float | None
    bias_mcse: float | None
    normalized_rmse: float | None
    rmse_mcse: float | None
    undefined: int
    recovered: bool


class Diagnostics(BaseModel):
    """Premises checked before publication of a canonical operating path."""

    maximum_solver_bound: float
    minimum_capacity: float
    minimum_baseline_price: float
    active_domain_violations: int


class DimensionCell(BaseModel):
    """Bounded generic-economy check, without a two-opportunity sign claim."""

    dimension: int
    budgets: list[int]
    recovery: list[RecoveryCell]
    failures: list[dict[str, str | int]]


class Campaign(BaseModel):
    """Publication consumes the final durable campaign summary only."""

    schema_version: int
    status: str
    histories: int
    redraws: int
    budgets: list[int]
    reporting_amplitudes: list[float]
    regimes: list[str]
    recovery: list[list[RecoveryCell]]
    finite_affine_departure: float | None
    reporting_floor: list[float | None]
    reporting_only_floor: list[float | None]
    smallest_recovered_budget: list[int | None]
    failures: list[dict[str, str | int]]
    diagnostics: Diagnostics
    appendix: list[DimensionCell]


class Contribution(BaseModel):
    """One frozen annual moment contribution."""

    label: str
    contribution: float


class Deletion(BaseModel):
    """A coefficient after deleting one role or annual transition."""

    beta: float


class Share(BaseModel):
    """One frozen denominator influence share."""

    share: float


class Denominator(BaseModel):
    """Saved denominator concentration, without an effective-sample claim."""

    shares: list[Share]


class EmpiricalCell(BaseModel):
    """Frozen T81 quantities needed for display and reconciliation."""

    beta: float
    annual_theta_contributions: list[Contribution]
    annual_second_moment_contributions: list[Contribution]
    target_denominator: Denominator
    reference_denominator: Denominator
    leave_one_firm_out: list[Deletion]
    leave_one_year_out: list[Deletion]
    article_measurement_percentile95: list[float]


class Empirical(BaseModel):
    """T81 is immutable evidence, rather than a new specification search."""

    firms: int
    directional_coordinates_per_cell: int
    draws: int
    cells: dict[str, EmpiricalCell]


class Inventory(BaseModel):
    """Frozen article-count and representation metadata."""

    matched_articles: int
    expanded_articles: int
    width: int


class FixedGMM(BaseModel):
    """A fixed-weight descriptive annual compromise."""

    beta: float
    effective_year_weights: list[float]


class GMMCell(BaseModel):
    """Annual GMM changes the target under heterogeneous annual slopes."""

    original_pooled_beta: float
    original_pooled_effective_year_weights: list[float]
    fixed_identity_common_slope: FixedGMM


class GMM(BaseModel):
    """Saved T82 numerical comparison to T81."""

    cells: dict[str, GMMCell]


class FrontierCell(BaseModel):
    """A hypothetical deterministic error frontier, not an error estimate."""

    normalized_alignment: float
    equal_relative_budget_sufficient_threshold: float


class Frontier(BaseModel):
    """Saved T82 error-bound projection."""

    cells: dict[str, FrontierCell]


class AnnualSkill(BaseModel):
    """A held-out annual squared-error comparison."""

    year: int
    heldout_skill: float


class Profile(BaseModel):
    """Held-out profile fit; no population sampling statement."""

    heldout_skill: float
    per_year: list[AnnualSkill]


class Locality(BaseModel):
    """Finite-path departures under the frozen affine-rank calibration."""

    cumulative_defined_count: int
    at_or_beyond_nearest_distinct_origin_reference_count: int


class ObservationCell(BaseModel):
    """Return-blind T83 evidence kept separate from financial fitting."""

    primary_centered_profile: Profile
    locality: Locality


class Observation(BaseModel):
    """Frozen T83 observation audit."""

    cells: dict[str, ObservationCell]

"""Typed saved directional diagnostics and their publication projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from pipeline.figures.paper6._publication import number, table

if TYPE_CHECKING:
    from pathlib import Path


class SignCell(BaseModel):
    """Saved historical literal-positive frequency, including undefined draws."""

    positive: int
    total: int
    frequency: float
    mcse: float
    undefined: int


class DirectionalDiagnostics(BaseModel):
    """Legacy fixed-history summaries; no rejection or population claim."""

    cells: list[list[SignCell]]
    reporting_only_positive_histories: int


class ValidationCell(BaseModel):
    """One design-selected saved summary; no scenarios are chosen here."""

    publication_label: str
    target_label: str
    correct_sign_rate: float | None
    correct_sign_mcse: float | None
    correct_sign_denominator: int | None
    positive_rate: float
    positive_mcse: float
    positive_denominator: int
    rmse: float | None
    undefined: int
    total: int
    failed_premises: int
    domain_departures: int


class PerformanceRate(BaseModel):
    """An already estimated stratified performance rate."""

    rate: float
    mcse: float
    denominator: int


class ScenarioSummary(BaseModel):
    """Named scalar inputs from the full saved scenario summary."""

    scenario_id: str
    regime: str
    budget: int
    target: str
    correct_sign_rate: float | None
    correct_sign_mcse: float | None
    correct_sign_denominator: int | None
    wrong_sign: PerformanceRate | None
    positive_rate: float
    positive_mcse: float
    positive_denominator: int
    rmse: float | None
    normalized_rmse: float | None
    mean_oracle_ratio: float | None
    normalized_rmse_mcse: float | None
    normalization_reason: str | None
    sign_ineligibility_reason: str | None
    undefined: int
    total: int
    failed_premises: int
    domain_departures: int

    def scalar_values(self) -> dict[str, float | int | str | None]:
        """Project saved statistics, converting rates to percentage points."""
        return {
            "target": self.target,
            "correct-pct": _percent(self.correct_sign_rate),
            "correct-se-pct": _percent(self.correct_sign_mcse),
            "correct-denominator": self.correct_sign_denominator,
            "wrong-pct": None
            if self.wrong_sign is None
            else _percent(self.wrong_sign.rate),
            "wrong-se-pct": None
            if self.wrong_sign is None
            else _percent(self.wrong_sign.mcse),
            "wrong-denominator": None
            if self.wrong_sign is None
            else self.wrong_sign.denominator,
            "positive-pct": _percent(self.positive_rate),
            "positive-se-pct": _percent(self.positive_mcse),
            "positive-denominator": self.positive_denominator,
            "rmse": self.rmse,
            "mean-target": self.mean_oracle_ratio,
            "normalized-rmse": self.normalized_rmse,
            "normalized-rmse-se": self.normalized_rmse_mcse,
            "normalization-reason": self.normalization_reason,
            "sign-reason": self.sign_ineligibility_reason,
            "undefined": self.undefined,
            "total": self.total,
            "failed-premises": self.failed_premises,
            "domain-departures": self.domain_departures,
        }


class InferenceRate(BaseModel):
    """A saved coverage or rejection rate under the iid fixture."""

    rate: float
    mcse: float
    attempts: int
    count: int


class FixtureScenario(BaseModel):
    """Saved inferential diagnostics at one explicitly known fixture target."""

    target: float
    average_candidate_se: float | None
    empirical_dispersion: float | None
    candidate_se_to_dispersion: float | None
    target_containment: InferenceRate
    positive_bound: InferenceRate

    def scalar_values(self) -> dict[str, float | int | str | None]:
        """Project coverage and rejection separately from sign recovery."""
        return {
            "target": self.target,
            "coverage-pct": _percent(self.target_containment.rate),
            "coverage-se-pct": _percent(self.target_containment.mcse),
            "coverage-count": self.target_containment.count,
            "coverage-attempts": self.target_containment.attempts,
            "rejection-pct": _percent(self.positive_bound.rate),
            "rejection-se-pct": _percent(self.positive_bound.mcse),
            "rejection-count": self.positive_bound.count,
            "rejection-attempts": self.positive_bound.attempts,
            "mean-se": self.average_candidate_se,
            "empirical-sd": self.empirical_dispersion,
            "se-ratio": self.candidate_se_to_dispersion,
        }


class IidFixture(BaseModel):
    """The saved, separate iid-history experiment."""

    firms: int
    years: int
    replications_per_target: int
    scenarios: list[FixtureScenario]


class CorporateInference(BaseModel):
    """Saved empirical eligibility, preserving unavailable inferential values."""

    eligible: bool
    lower_bound: float | None
    p_value: float | None
    reason: str


class InferenceStatus(BaseModel):
    """Read the analysis decision rather than deriving empirical eligibility."""

    one_sided_z: float
    iid_fixture: IidFixture
    empirical_inference: CorporateInference

    def scalar_values(self) -> dict[str, float | int | str | None]:
        """Bind fixed null/alternative diagnostics from the saved iid audit."""
        fixture = self.iid_fixture
        scenarios = {scenario.target: scenario for scenario in fixture.scenarios}
        if len(scenarios) != len(fixture.scenarios):
            message = "duplicate saved iid fixture target"
            raise ValueError(message)
        if not {0.0, 0.5}.issubset(scenarios):
            message = "saved iid fixture requires the declared zero and 0.5 targets"
            raise ValueError(message)
        null = scenarios[0.0].scalar_values()
        values: dict[str, float | int | str | None] = {
            "inference-replications": fixture.replications_per_target,
            "inference-firms": fixture.firms,
            "inference-years": fixture.years,
            "inference-z": self.one_sided_z,
            "inference-coverage-pct": null["coverage-pct"],
            "inference-coverage-se-pct": null["coverage-se-pct"],
            "inference-mean-se": null["mean-se"],
            "inference-empirical-sd": null["empirical-sd"],
            "inference-se-ratio": null["se-ratio"],
            "inference-corporate-eligible": self.empirical_inference.eligible,
            "inference-corporate-status": (
                "eligible" if self.empirical_inference.eligible else "unavailable"
            ),
            "inference-corporate-reason": self.empirical_inference.reason,
            "inference-corporate-lower-bound": self.empirical_inference.lower_bound,
            "inference-corporate-p-value": self.empirical_inference.p_value,
        }
        for label, target in (("null", 0.0), ("alternative", 0.5)):
            values.update(
                {
                    f"inference-{label}-{key}": value
                    for key, value in scenarios[target].scalar_values().items()
                }
            )
        return values


def _percent(value: float | None) -> float | None:
    return None if value is None else 100 * value


def _named_scenarios() -> tuple[tuple[str, tuple[str, str, int, str]], ...]:
    """Name the requested saved identities; never select using their outcomes."""
    return (
        *(
            (f"{regime}-{budget}", ("A", regime, budget, "beta0_h"))
            for regime in ("common", "changing")
            for budget in (1, 4, 16)
        ),
        *(
            (f"stationary-{budget}", ("B", "common", budget, "beta_lambda_h"))
            for budget in (1, 4, 16)
        ),
        ("reporting-only", ("C", "reporting_0.005", 1, "beta_lambda_h")),
        ("half", ("D_half", "common", 1, "beta0_h")),
        ("double", ("D_double", "common", 1, "beta0_h")),
    )


class ValidationResults(BaseModel):
    """The new analysis owns metrics and the declared publication selection."""

    schema_version: Literal[1]
    histories: int
    redraws: int
    scenarios: list[ScenarioSummary]
    inference_status: InferenceStatus
    legacy_directional: DirectionalDiagnostics
    publication_scenarios: list[ValidationCell] = Field(min_length=1)

    def publication_values(self) -> dict[str, float | int | str | None]:
        """Convert saved rates to percentage-point display units."""
        values: dict[str, float | int | str | None] = {
            "validation-histories": self.histories,
            "validation-redraws": self.redraws,
        }
        scenarios = {
            (cell.scenario_id, cell.regime, cell.budget, cell.target): cell
            for cell in self.scenarios
        }
        if len(scenarios) != len(self.scenarios):
            message = "duplicate saved validation scenario identity"
            raise ValueError(message)
        for name, identity in _named_scenarios():
            if identity not in scenarios:
                message = f"missing saved validation scenario for {name}"
                raise ValueError(message)
            values.update(
                {
                    f"validation-{name}-{key}": value
                    for key, value in scenarios[identity].scalar_values().items()
                }
            )
        values.update(self.inference_status.scalar_values())
        for index, cell in enumerate(self.publication_scenarios):
            prefix = f"validation-{index}"
            values.update(
                {
                    f"{prefix}-label": cell.publication_label,
                    f"{prefix}-target": cell.target_label,
                    f"{prefix}-correct-pct": (
                        None
                        if cell.correct_sign_rate is None
                        else 100 * cell.correct_sign_rate
                    ),
                    f"{prefix}-correct-se-pct": (
                        None
                        if cell.correct_sign_mcse is None
                        else 100 * cell.correct_sign_mcse
                    ),
                    f"{prefix}-positive-pct": 100 * cell.positive_rate,
                    f"{prefix}-positive-se-pct": 100 * cell.positive_mcse,
                    f"{prefix}-correct-denominator": cell.correct_sign_denominator,
                    f"{prefix}-positive-denominator": cell.positive_denominator,
                    f"{prefix}-rmse": cell.rmse,
                    f"{prefix}-undefined": cell.undefined,
                    f"{prefix}-total": cell.total,
                    f"{prefix}-failed-premises": cell.failed_premises,
                    f"{prefix}-domain-departures": cell.domain_departures,
                }
            )
        return values

    def publication_table(self) -> str:
        """Display the five predeclared mechanism and measurement-control rows."""
        identities = (
            ("Common reporting", 1, "common-1"),
            ("Common reporting", 4, "common-4"),
            ("Common reporting", 16, "common-16"),
            ("Stationary reporting", 1, "stationary-1"),
            ("Reporting only", 1, "reporting-only"),
        )
        values = self.publication_values()
        common_attempts = all(
            values[f"validation-{name}-positive-denominator"]
            == values["validation-common-1-positive-denominator"]
            == values[f"validation-{name}-total"]
            for _, _, name in identities
        )
        rows = []
        exceptions = []
        for label, budget, name in identities:
            prefix = f"validation-{name}"
            rows.append(
                " & ".join(
                    [
                        label,
                        "$" + str(budget) + r"\times$",
                        number(f"{prefix}-mean-target", 3),
                        number(f"{prefix}-positive-pct", 1)
                        + "("
                        + number(f"{prefix}-positive-se-pct", 2)
                        + ")"
                        + (
                            ""
                            if common_attempts
                            else "/" + number(f"{prefix}-positive-denominator", 0)
                        ),
                        number(f"{prefix}-rmse", 3),
                    ]
                )
                + r" \\"
            )
            if any(
                values[f"{prefix}-{status}"] != 0
                for status in ("undefined", "failed-premises", "domain-departures")
            ):
                exceptions.append(
                    label
                    + " $"
                    + str(budget)
                    + r"\times$: "
                    + "/".join(
                        number(f"{prefix}-{status}", 0)
                        for status in (
                            "undefined",
                            "failed-premises",
                            "domain-departures",
                        )
                    )
                )
        caption = (
            "Mechanism and measurement controls. Article-count multipliers scale "
            "observed coverage. Mean target is the arithmetic mean of history-specific "
            "population coefficients. Positive reports numerically resolved "
            "positive estimates as percentages (stratified Monte Carlo SE in "
            "percentage points). Absolute RMSE uses "
            "each history's matching target in coefficient units. Stationary "
            "reporting is the zero-alignment measurement control; reporting only "
            "holds capacities fixed and uses reporting amplitude 0.005. "
            r"Table~\ref{tab:directional-validation-details} provides full "
            "diagnostics and definitions."
        )
        if common_attempts:
            caption += (
                " Each percentage uses "
                + number("validation-common-1-positive-denominator", 0)
                + " attempted redraws."
            )
        else:
            caption += " Each percentage is followed by its attempted denominator."
        if exceptions:
            caption += (
                " Exceptions (undefined attempts/failed-premise attempts/"
                "domain-departing histories): " + "; ".join(exceptions) + "."
            )
        return table(
            caption,
            "tab:directional-validation-controls",
            r"@{}l@{\enspace}r@{\enspace}r@{\enspace}r@{\enspace}r@{}",
            r"Design & \shortstack{Article-count\\multiplier} & Mean target & "
            r"\shortstack{Positive estimates\\(\%)} & Absolute RMSE",
            rows,
        )

    def details_table(self) -> str:
        """Retain all thirteen original directional diagnostics in the appendix."""
        common_attempts = all(
            cell.total == self.histories * self.redraws
            and cell.positive_denominator == cell.total
            and cell.correct_sign_denominator in (None, cell.total)
            for cell in self.publication_scenarios
        )
        show_status = any(
            cell.undefined or cell.failed_premises or cell.domain_departures
            for cell in self.publication_scenarios
        )
        rows = []
        for index, cell in enumerate(self.publication_scenarios):
            prefix = f"validation-{index}"
            correct = (
                r"\ppmissing"
                if cell.correct_sign_denominator is None
                else number(f"{prefix}-correct-pct", 1)
                + "("
                + number(f"{prefix}-correct-se-pct", 2)
                + ")"
                + (
                    ""
                    if common_attempts
                    else "/" + number(f"{prefix}-correct-denominator", 0)
                )
            )
            positive = (
                number(f"{prefix}-positive-pct", 1)
                + "("
                + number(f"{prefix}-positive-se-pct", 2)
                + ")"
                + (
                    ""
                    if common_attempts
                    else "/" + number(f"{prefix}-positive-denominator", 0)
                )
            )
            columns = [
                rf"\ppvalue{{ir-{prefix}-label}}",
                rf"\ppvalue{{ir-{prefix}-target}}",
                correct,
                positive,
                number(f"{prefix}-rmse", 3),
            ]
            if show_status:
                columns.append(
                    number(f"{prefix}-undefined", 0)
                    + "/"
                    + number(f"{prefix}-failed-premises", 0)
                    + "/"
                    + number(f"{prefix}-domain-departures", 0)
                )
            rows.append(" & ".join(columns) + r" \\")
        caption = (
            "Directional measurement under fixed economic histories. Targets "
            "distinguish common-reporting operating coefficients from reported-law "
            "coefficients, including the zero operating target in reporting-only "
            "cells. Rates are percentages (stratified Monte Carlo SE in percentage "
            "points), not test power. RMSE is in coefficient units against each "
            "history's matching target. "
            r"Appendix~\ref{app:directional-diagnostics} defines resolved nonzero "
            "oracles, eligible rate denominators and undefined-attempt accounting. "
            "The main table's mean target is an arithmetic mean of history "
            "coefficients, not a ratio of expected moments."
        )
        if all(
            cell.total == self.histories * self.redraws
            and cell.positive_denominator == cell.total
            for cell in self.publication_scenarios
        ):
            caption += (
                r" Each row retains \ppnum[0]{ir-validation-histories} histories "
                r"and \ppnum[0]{ir-validation-redraws} redraws per history."
            )
        if common_attempts:
            caption += " All attempts enter the rate denominators."
        else:
            caption += " Each rate is followed by its eligible denominator."
        if show_status:
            caption += (
                " U/P/D counts undefined estimates, failed-premise attempts "
                "and domain-departing histories."
            )
        else:
            caption += " No undefined estimates, failed premises or domain departures."
        caption += " Correct sign is unavailable for zero or unresolved targets."
        return table(
            caption,
            "tab:directional-validation-details",
            r"@{}l@{\enspace}l@{\enspace}r@{\enspace}r@{\enspace}r"
            + (r"@{\enspace}r@{}" if show_status else "@{}"),
            "Cell & Target & Correct & Positive & RMSE"
            + (" & U/P/D" if show_status else ""),
            rows,
        )


def load_directional(path: Path) -> ValidationResults:
    """Read saved metrics; checkpoints and numerical algorithms belong upstream."""
    return ValidationResults.model_validate_json(path.read_bytes())

"""Publish saved nested article-redraw calibration summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from pipeline.figures.paper6._publication import number, table, write_numbers

if TYPE_CHECKING:
    from pathlib import Path


class CalibrationRate(BaseModel):
    """Saved unconditional performance and conditional availability diagnostics."""

    equal_history_rate: float | None
    mcse: float | None
    count: int
    attempts: int
    available_attempts: int
    unavailable_attempts: int
    available_rate: float | None
    unavailable_histories: list[int]
    unavailable_reason: str | None

    def values(self) -> dict[str, float | int | str | None]:
        """Apply display units without recomputing a performance rate."""
        return {
            "pct": None
            if self.equal_history_rate is None
            else 100 * self.equal_history_rate,
            "se-pct": None if self.mcse is None else 100 * self.mcse,
            "success-count": self.count,
            "attempts": self.attempts,
            "available": self.available_attempts,
            "unavailable": self.unavailable_attempts,
            "available-only-pct": None
            if self.available_rate is None
            else 100 * self.available_rate,
            "unavailable-histories": ", ".join(map(str, self.unavailable_histories)),
            "unavailable-reason": self.unavailable_reason,
        }


class CalibrationScenario(BaseModel):
    """One of the three predeclared observed-coverage controls."""

    scenario_id: str
    budget: int
    support: str
    target: str
    outer_attempts: int
    available_outer_attempts: int
    sampling_target: str
    lower_bound_method: str
    quantile_method: str
    status: str
    reason: str | None
    positive_q05_interpretation: str
    inner_undefined: int
    outer_failed_execution: int
    failed_premises: int
    central_containment: CalibrationRate
    q05_containment: CalibrationRate
    positive_q05: CalibrationRate


class SignalScale(BaseModel):
    """Saved operating-response scales in fractions of total activity capacity."""

    status: str
    units: str | None = None
    first_response_abs_share_change_min: float | None = None
    first_response_abs_share_change_max: float | None = None
    maximum_abs_share_change_from_origin: float | None = None
    news_source_tick: int | None = None
    first_response_tick: int | None = None
    unavailable_histories: list[int] = Field(default_factory=list)
    reason: str | None = None

    def values(self) -> dict[str, float | int | str | None]:
        """Expose saved scales in fractions and percentage points."""
        values: dict[str, float | int | str | None] = {
            "status": self.status,
            "units": self.units,
            "news-source-tick": self.news_source_tick,
            "first-response-tick": self.first_response_tick,
            "unavailable-histories": ", ".join(map(str, self.unavailable_histories)),
            "reason": self.reason,
        }
        for name, value in (
            ("first-response-min", self.first_response_abs_share_change_min),
            ("first-response-max", self.first_response_abs_share_change_max),
            ("maximum-origin-change", self.maximum_abs_share_change_from_origin),
        ):
            values[name] = value
            values[f"{name}-pct"] = None if value is None else 100 * value
        return values


class CalibrationResults(BaseModel):
    """Publication reads scalar summaries, never inner or outer replicate arrays."""

    schema_version: Literal[1]
    histories: int
    outer_replications: int
    inner_redraws: int
    signal_scale: SignalScale
    scenarios: list[CalibrationScenario]


_ROWS = (
    ("A", "mechanism", "Common reporting"),
    ("B", "stationary", "Stationary reporting"),
    ("C", "reporting", "Reporting only"),
)
_RATES = (
    ("central-coverage", "central_containment"),
    ("q05-coverage", "q05_containment"),
    ("positive-q05", "positive_q05"),
)


def _rate_cell(prefix: str, *, common_counts: bool) -> str:
    displayed = number(f"{prefix}-pct", 1) + "(" + number(f"{prefix}-se-pct", 2) + ")"
    if not common_counts:
        displayed += (
            "; "
            + number(f"{prefix}-available", 0)
            + "/"
            + number(f"{prefix}-attempts", 0)
        )
    return displayed


def _count_note(*, common_counts: bool) -> str:
    if common_counts:
        return (
            " Each percentage uses "
            + number("redraw-mechanism-central-coverage-attempts", 0)
            + " attempted outer panels, with "
            + number("redraw-mechanism-central-coverage-available", 0)
            + " available bounds."
        )
    return " Each percentage is followed by available/attempted outer panels."


def _availability_note(rates: list[CalibrationRate]) -> str:
    notes = []
    if any(rate.unavailable_attempts for rate in rates):
        notes.append(
            "Unavailable bounds remain non-events in the all-attempt rates "
            "when the declared target is available."
        )
    if any(rate.equal_history_rate is None for rate in rates):
        notes.append(
            "Containment with a missing history target is unavailable; "
            "the exceptions give its reason."
        )
    return " " + " ".join(notes) if notes else ""


def _project(
    result: CalibrationResults,
) -> tuple[dict[str, float | int | str | None], str]:
    scenarios = {row.scenario_id: row for row in result.scenarios}
    if len(scenarios) != len(result.scenarios) or set(scenarios) != {"A", "B", "C"}:
        message = "article calibration requires exactly the saved A/B/C controls"
        raise ValueError(message)
    rates = [
        rate
        for cell in result.scenarios
        for rate in (
            cell.central_containment,
            cell.q05_containment,
            cell.positive_q05,
        )
    ]
    common_counts = all(
        (rate.attempts, rate.available_attempts)
        == (rates[0].attempts, rates[0].available_attempts)
        and rate.equal_history_rate is not None
        and rate.mcse is not None
        and rate.unavailable_reason is None
        for rate in rates
    )
    values: dict[str, float | int | str | None] = {
        "redraw-histories": result.histories,
        "redraw-outer-replications": result.outer_replications,
        "redraw-inner-redraws": result.inner_redraws,
        **{
            f"redraw-signal-{key}": value
            for key, value in result.signal_scale.values().items()
        },
    }
    rows = []
    exceptions = []
    if result.signal_scale.status != "complete":
        exceptions.append(
            r"signal scale: \ppvalue{ir-redraw-signal-status}, "
            r"\ppvalue{ir-redraw-signal-reason}; histories "
            r"\ppvalue{ir-redraw-signal-unavailable-histories}"
        )
    complete_counts = True
    for scenario_id, name, label in _ROWS:
        cell = scenarios[scenario_id]
        if (cell.budget, cell.support, cell.target) != (1, "expanded", "beta_lambda_h"):
            message = (
                "article calibration differs from its frozen observed-support design"
            )
            raise ValueError(message)
        prefix = f"redraw-{name}"
        values.update(
            {
                f"{prefix}-attempts": cell.outer_attempts,
                f"{prefix}-available": cell.available_outer_attempts,
                f"{prefix}-sampling-target": cell.sampling_target,
                f"{prefix}-method": cell.lower_bound_method,
                f"{prefix}-quantile-method": cell.quantile_method,
                f"{prefix}-status": cell.status,
                f"{prefix}-reason": cell.reason,
                f"{prefix}-positive-interpretation": cell.positive_q05_interpretation,
                f"{prefix}-inner-undefined": cell.inner_undefined,
                f"{prefix}-failed-execution": cell.outer_failed_execution,
                f"{prefix}-failed-premises": cell.failed_premises,
            }
        )
        measures = {
            "central_containment": cell.central_containment,
            "q05_containment": cell.q05_containment,
            "positive_q05": cell.positive_q05,
        }
        displayed = []
        for suffix, attribute in _RATES:
            rate = measures[attribute]
            rate_prefix = f"{prefix}-{suffix}"
            values.update(
                {f"{rate_prefix}-{key}": value for key, value in rate.values().items()}
            )
            displayed.append(_rate_cell(rate_prefix, common_counts=common_counts))
            complete_counts &= (
                rate.attempts == result.histories * result.outer_replications
            )
            if rate.unavailable_reason is not None:
                exceptions.append(
                    label
                    + " "
                    + suffix
                    + r": \ppvalue{ir-"
                    + rate_prefix
                    + r"-unavailable-reason}; histories \ppvalue{ir-"
                    + rate_prefix
                    + "-unavailable-histories}"
                )
        rows.append(" & ".join([label, *displayed]) + r" \\")
        if cell.inner_undefined or cell.outer_failed_execution or cell.failed_premises:
            exceptions.append(
                label
                + ": "
                + "/".join(
                    number(f"{prefix}-{key}", 0)
                    for key in (
                        "inner-undefined",
                        "failed-execution",
                        "failed-premises",
                    )
                )
            )
    caption = (
        "Conditional article-redraw performance under simulated reporting laws "
        "at observed expanded support. "
        "Central coverage uses the 2.5th--97.5th percentile interval; q.05 "
        "coverage uses the lower fifth percentile. Each cell gives the "
        "equal-history percentage (stratified Monte Carlo SE in percentage "
        "points). "
        "Coverage concerns the matching history-specific reported-law target. "
        "Positive q.05 is a zero-null rejection frequency only for the stationary "
        "control; the other rows have positive reported-law targets."
    )
    caption += _count_note(common_counts=common_counts)
    caption += _availability_note(rates)
    if complete_counts:
        caption += (
            r" The design retains \ppnum[0]{ir-redraw-histories} histories, "
            r"\ppnum[0]{ir-redraw-outer-replications} outer panels per history "
            r"and \ppnum[0]{ir-redraw-inner-redraws} inner redraws per panel."
        )
    if exceptions:
        caption += (
            " Exceptions (unavailable targets or undefined inner draws/failed "
            "outer execution/failed premises): " + "; ".join(exceptions) + "."
        )
    return values, table(
        caption,
        "tab:article-redraw-validation",
        r"@{}l@{\enspace}r@{\enspace}r@{\enspace}r@{}",
        "Design & Central coverage & q.05 coverage & Positive q.05",
        rows,
    )


def render(results: Path, project: Path) -> None:
    """Project saved calibration summaries into their exclusive exhibit files."""
    result = CalibrationResults.model_validate_json(results.read_bytes())
    values, exhibit = _project(result)
    generated = project / "src/generated"
    write_numbers(generated / "article_redraw_validation_numbers.tex", values)
    (generated / "article_redraw_validation_table.tex").write_text(
        exhibit, encoding="utf-8"
    )

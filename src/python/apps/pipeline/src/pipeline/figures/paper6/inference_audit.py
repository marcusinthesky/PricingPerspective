"""Publish the saved N=16/N=100 iid-history inference audit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from pipeline.figures.paper6._publication import number, table, write_numbers
from pipeline.figures.paper6.information_recovery.directional import (
    CorporateInference,  # noqa: TC001 - Pydantic resolves this model at runtime.
    FixtureScenario,
)

if TYPE_CHECKING:
    from pathlib import Path

_YEARS = 4


class AuditScenario(FixtureScenario):
    """The saved candidate diagnostics, including unavailable bounds."""

    attempts: int
    undefined_estimates: int
    unavailable_candidate_se: int
    degenerate_or_negative_candidate_se: int
    usable_bounds: int
    one_sided_z: float


class AuditFixture(BaseModel):
    """One saved iid-history fixture size."""

    firms: int
    years: int
    replications_per_target: int
    scenarios: list[AuditScenario]


class AuditResults(BaseModel):
    """Saved N=100 output with its immutable N=16 comparison."""

    schema_version: Literal[1]
    iid_fixture: AuditFixture
    baseline_iid_fixture: AuditFixture
    empirical_inference: CorporateInference


def _project(result: AuditResults) -> tuple[dict[str, float | int | str | None], str]:
    values: dict[str, float | int | str | None] = {}
    rows = []
    exceptions = []
    cells = [
        cell
        for fixture in (result.baseline_iid_fixture, result.iid_fixture)
        for cell in fixture.scenarios
    ]
    common_counts = bool(cells) and all(
        (cell.attempts, cell.usable_bounds)
        == (cells[0].attempts, cells[0].usable_bounds)
        and cell.target_containment.attempts
        == cell.positive_bound.attempts
        == cell.attempts
        and cell.average_candidate_se is not None
        and cell.empirical_dispersion is not None
        and cell.candidate_se_to_dispersion is not None
        for cell in cells
    )
    for firms, fixture, prefix in (
        (16, result.baseline_iid_fixture, "inference-audit-n16"),
        (100, result.iid_fixture, "inference-n100"),
    ):
        if fixture.firms != firms or fixture.years != _YEARS:
            message = "iid audit publication requires the declared N=16/N=100 designs"
            raise ValueError(message)
        scenarios = {cell.target: cell for cell in fixture.scenarios}
        if len(scenarios) != len(fixture.scenarios) or set(scenarios) != {0.0, 0.5}:
            message = "iid audit publication requires exactly the zero/0.5 targets"
            raise ValueError(message)
        values.update(
            {
                f"{prefix}-firms": fixture.firms,
                f"{prefix}-years": fixture.years,
                f"{prefix}-replications": fixture.replications_per_target,
                f"{prefix}-z": scenarios[0.0].one_sided_z,
            }
        )
        for label, target in (("null", 0.0), ("alternative", 0.5)):
            cell = scenarios[target]
            cell_prefix = f"{prefix}-{label}"
            saved = cell.scalar_values()
            values.update(
                {f"{cell_prefix}-{key}": value for key, value in saved.items()}
            )
            values.update(
                {
                    f"{cell_prefix}-attempts": cell.attempts,
                    f"{cell_prefix}-available": cell.usable_bounds,
                    f"{cell_prefix}-undefined": cell.undefined_estimates,
                    f"{cell_prefix}-unavailable-se": cell.unavailable_candidate_se,
                    f"{cell_prefix}-degenerate-se": (
                        cell.degenerate_or_negative_candidate_se
                    ),
                    f"{cell_prefix}-z": cell.one_sided_z,
                }
            )
            displayed_rates = [
                number(f"{cell_prefix}-{rate}-pct", 1)
                + "("
                + number(f"{cell_prefix}-{rate}-se-pct", 2)
                + ")"
                + (
                    ""
                    if common_counts
                    else "/" + number(f"{cell_prefix}-{rate}-attempts", 0)
                )
                for rate in ("coverage", "rejection")
            ]
            displayed = [
                number(f"{prefix}-firms", 0),
                number(f"{cell_prefix}-target", 1),
                *displayed_rates,
                number(f"{cell_prefix}-se-ratio", 3),
            ]
            if not common_counts:
                displayed.append(
                    number(f"{cell_prefix}-available", 0)
                    + "/"
                    + number(f"{cell_prefix}-attempts", 0)
                )
            rows.append(" & ".join(displayed) + r" \\")
            if (
                cell.undefined_estimates
                or cell.unavailable_candidate_se
                or cell.degenerate_or_negative_candidate_se
            ):
                exceptions.append(
                    f"N={firms}, target={target}: "
                    + "/".join(
                        number(f"{cell_prefix}-{key}", 0)
                        for key in ("undefined", "unavailable-se", "degenerate-se")
                    )
                )
        # These main-text aliases refer to the saved null-target diagnostics.
        for suffix in (
            "coverage-pct",
            "coverage-se-pct",
            "mean-se",
            "empirical-sd",
            "se-ratio",
        ):
            values[f"{prefix}-{suffix}"] = values[f"{prefix}-null-{suffix}"]
    eligibility = result.empirical_inference
    values.update(
        {
            "inference-n100-corporate-eligible": eligibility.eligible,
            "inference-n100-corporate-status": (
                "eligible" if eligibility.eligible else "unavailable"
            ),
            "inference-n100-corporate-reason": eligibility.reason,
            "inference-n100-corporate-lower-bound": eligibility.lower_bound,
            "inference-n100-corporate-p-value": eligibility.p_value,
        }
    )
    caption = (
        "One-sided lower-bound performance under independent firm-history "
        "sampling. Nominal 95\\% lower bounds "
        "use the one-sided normal critical value. Coverage and rejection of "
        r"$H_0:\beta\leq0$ are percentages (Monte Carlo SE in percentage "
        "points). SE/SD compares mean candidate standard error with empirical "
        "dispersion; unavailable bounds remain in the performance denominators. "
        "Both panel sizes retain four dependent years and each firm's target "
        "and reference roles."
    )
    if common_counts:
        caption += (
            " Each row uses "
            + number("inference-audit-n16-null-attempts", 0)
            + " attempted replications, with "
            + number("inference-audit-n16-null-available", 0)
            + " usable bounds."
        )
    else:
        caption += (
            " Each percentage is followed by its attempted denominator. "
            "Avail. gives usable bounds / attempted replications."
        )
    if exceptions:
        caption += (
            " Exceptions (undefined estimates/unavailable SE/degenerate SE): "
            + "; ".join(exceptions)
            + "."
        )
    return values, table(
        caption,
        "tab:iid-inference-audit",
        (
            r"@{}r@{\enspace}r@{\enspace}r@{\enspace}r@{\enspace}r@{}"
            if common_counts
            else r"@{}r@{\enspace}r@{\enspace}r@{\enspace}r@{\enspace}r@{\enspace}r@{}"
        ),
        r"Firms & Target & Coverage & Reject $H_0$ & SE/SD"
        + ("" if common_counts else " & Avail."),
        rows,
    )


def render(results: Path, project: Path) -> None:
    """Project saved iid-history diagnostics into exclusive audit exhibit files."""
    result = AuditResults.model_validate_json(results.read_bytes())
    values, exhibit = _project(result)
    generated = project / "src/generated"
    write_numbers(generated / "inference_audit_numbers.tex", values)
    (generated / "inference_audit_table.tex").write_text(exhibit, encoding="utf-8")

"""Publish saved simulation evidence independently of corporate and audit inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hotelling.dynamics.recovery import DIMENSIONS, REPORTING_AMPLITUDES

from pipeline._kernels.paper6.recovery import BUDGETS, REDRAWS
from pipeline.figures.paper6._publication import number as _number
from pipeline.figures.paper6._publication import table as _table
from pipeline.figures.paper6._publication import write_numbers
from pipeline.figures.paper6.information_recovery.contracts import Campaign
from pipeline.figures.paper6.information_recovery.directional import (
    load_directional,
)
from pipeline.figures.paper6.information_recovery.figures import render
from pipeline.stages.papers.paper6.recovery.campaign import HISTORIES, audit_campaign

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.figures.paper6.information_recovery.directional import (
        DirectionalDiagnostics,
    )


def _campaign_values(campaign: Campaign) -> dict[str, float | int | None]:
    values: dict[str, float | int | None] = {
        "finite-affine": campaign.finite_affine_departure,
        "floor-005": campaign.reporting_floor[3],
        "solve-target": 1e-10,
        "failures": len(campaign.failures),
        "maximum-solver-bound": campaign.diagnostics.maximum_solver_bound,
        "minimum-capacity": campaign.diagnostics.minimum_capacity,
        "minimum-baseline-price": campaign.diagnostics.minimum_baseline_price,
        "active-domain-violations": campaign.diagnostics.active_domain_violations,
    }
    for regime, budget in zip(
        campaign.regimes, campaign.smallest_recovered_budget, strict=True
    ):
        values[f"{regime}-smallest-recovered-budget"] = budget
    for regime, cells in zip(campaign.regimes, campaign.recovery, strict=True):
        for budget, cell in zip(campaign.budgets, cells, strict=True):
            prefix = f"{regime}-{budget}"
            values[f"{prefix}-bias"] = cell.bias
            values[f"{prefix}-bias-se"] = cell.bias_mcse
            values[f"{prefix}-rmse"] = cell.normalized_rmse
            values[f"{prefix}-rmse-se"] = cell.rmse_mcse
            values[f"{prefix}-undefined"] = cell.undefined
    for dimension in campaign.appendix:
        for budget, cell in zip(dimension.budgets, dimension.recovery, strict=True):
            prefix = f"dimension-{dimension.dimension}-{budget}"
            values[f"{prefix}-rmse"] = cell.normalized_rmse
            values[f"{prefix}-se"] = cell.rmse_mcse
            values[f"{prefix}-undefined"] = cell.undefined
    return values


def _directional_values(
    diagnostics: DirectionalDiagnostics, campaign: Campaign
) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "reporting-only-positive-histories": (
            diagnostics.reporting_only_positive_histories
        )
    }
    for regime, cells in zip(campaign.regimes, diagnostics.cells, strict=True):
        for budget, cell in zip(campaign.budgets, cells, strict=True):
            prefix = f"{regime}-{budget}"
            values[f"{prefix}-positive-pct"] = 100 * cell.frequency
            values[f"{prefix}-positive-se-pct"] = 100 * cell.mcse
            values[f"{prefix}-positive-count"] = cell.positive
            values[f"{prefix}-sign-total"] = cell.total
            values[f"{prefix}-sign-undefined"] = cell.undefined
    return values


def _directional_table(campaign: Campaign) -> str:
    rows = [
        " & ".join(
            [
                regime.capitalize(),
                str(budget),
                _number(f"{regime}-{budget}-positive-pct", 1),
                _number(f"{regime}-{budget}-positive-se-pct", 2),
                _number(f"{regime}-{budget}-sign-undefined", 0),
            ]
        )
        + r" \\"
        for regime in campaign.regimes
        for budget in campaign.budgets
    ]
    return _table(
        "Historical positive-estimate frequencies from finite article panels, "
        r"using the original literal $\widehat\beta>0$ convention. All sixteen "
        "common-reporting population coefficients are positive. Frequencies "
        "average 128 redraws within each history and weight histories equally. "
        "SE is the stratified Monte Carlo standard error in percentage points. "
        "Zero and undefined estimates do not count as positive; undefined draws "
        "remain in the denominator.",
        "tab:directional-recovery",
        "llrrr",
        r"Reporting & \shortstack{Article-count\\multiplier} & "
        r"Positive (\%) & SE (pp) & Undefined",
        rows,
    )


def _recovery_table(campaign: Campaign) -> str:
    rows = []
    for regime in campaign.regimes:
        for budget in campaign.budgets:
            prefix = f"{regime}-{budget}"
            rows.append(
                " & ".join(
                    [
                        regime.capitalize(),
                        str(budget),
                        _number(f"{prefix}-bias", 5),
                        _number(f"{prefix}-bias-se", 5),
                        _number(f"{prefix}-rmse", 3),
                        _number(f"{prefix}-rmse-se", 3),
                        _number(f"{prefix}-undefined", 0),
                    ]
                )
                + r" \\"
            )
    return _table(
        "Recovery diagnostics against the finite-capacity common-reporting oracle. "
        "Bias and its Monte Carlo standard error (SE) are in coefficient "
        "units; RMSE and its SE are "
        "normalized by oracle RMS. Undefined estimates remain counted. "
        "Histories receive equal weight.",
        "tab:recovery-diagnostics",
        "llrrrrr",
        r"Reporting & \shortstack{Article-count\\multiplier} & "
        "Bias & SE & RMSE & SE & Undefined",
        rows,
    )


def _dimensions_table(campaign: Campaign) -> str:
    show_undefined = any(
        cell.undefined for dimension in campaign.appendix for cell in dimension.recovery
    )
    show_status = any(dimension.failures for dimension in campaign.appendix)
    rows = []
    for dimension in campaign.appendix:
        for budget in dimension.budgets:
            prefix = f"dimension-{dimension.dimension}-{budget}"
            columns = [
                str(dimension.dimension),
                str(budget),
                _number(f"{prefix}-rmse", 3),
                _number(f"{prefix}-se", 3),
            ]
            if show_undefined:
                columns.append(_number(f"{prefix}-undefined", 0))
            if show_status:
                columns.append("Failed" if dimension.failures else "Complete")
            rows.append(" & ".join(columns) + r" \\")
    caption = (
        "Bounded dimension check under common reporting. All cells hold "
        "100 firms, eight activities, "
        "64 demand nodes and adjustment weight one fixed. RMSE is "
        "normalized by oracle RMS, with stratified Monte Carlo SE. "
    )
    if not show_status and not show_undefined:
        caption += (
            f"All {len(rows)} configurations completed with no undefined estimates."
        )
    elif not show_status:
        caption += f"All {len(rows)} configurations completed."
    elif not show_undefined:
        caption += "No undefined estimates."
    return _table(
        caption,
        "tab:recovery-dimensions",
        "llrr" + ("r" if show_undefined else "") + ("l" if show_status else ""),
        r"Dimension & \shortstack{Article-count\\multiplier} & RMSE & SE"
        + (" & Undefined" if show_undefined else "")
        + (" & Status" if show_status else ""),
        rows,
    )


def _validate_design(campaign: Campaign) -> None:
    expected = (
        campaign.schema_version == 1,
        campaign.histories == HISTORIES,
        campaign.redraws == REDRAWS,
        campaign.budgets == list(BUDGETS),
        campaign.reporting_amplitudes == list(REPORTING_AMPLITUDES),
        campaign.regimes == ["common", "changing"],
        len(campaign.recovery) == len(campaign.regimes),
        all(len(cells) == len(BUDGETS) for cells in campaign.recovery),
        len(campaign.reporting_floor) == len(REPORTING_AMPLITUDES),
        len(campaign.reporting_only_floor) == len(REPORTING_AMPLITUDES),
        [cell.dimension for cell in campaign.appendix] == list(DIMENSIONS),
        all(cell.budgets == [BUDGETS[0], BUDGETS[-1]] for cell in campaign.appendix),
    )
    if not all(expected):
        message = "campaign summary differs from the frozen publication design/order"
        raise ValueError(message)


def publish(summary: Path, project: Path, directional_summary: Path) -> None:
    """Project saved simulation summaries; no estimates are computed here."""
    audit = audit_campaign(summary.parent)
    if audit["status"] != "complete" or audit["integrity"] is not True:
        message = "publication requires a complete campaign with verified saved bytes"
        raise ValueError(message)
    campaign = Campaign.model_validate_json(summary.read_bytes())
    _validate_design(campaign)
    directional = load_directional(directional_summary)
    if any(
        cell.total != campaign.histories * campaign.redraws
        for row in directional.legacy_directional.cells
        for cell in row
    ):
        message = "directional checkpoint count differs from the frozen design"
        raise ValueError(message)
    values: dict[str, float | int | str | None] = dict(_campaign_values(campaign))
    values.update(_directional_values(directional.legacy_directional, campaign))
    values.update(directional.publication_values())
    generated = project / "src/generated"
    write_numbers(generated / "information_recovery_numbers.tex", values)
    for name, content in (
        ("directional", _directional_table(campaign)),
        ("validation", directional.publication_table()),
        ("validation_details", directional.details_table()),
        ("diagnostics", _recovery_table(campaign)),
        ("dimensions", _dimensions_table(campaign)),
    ):
        (generated / f"information_recovery_{name}_table.tex").write_text(
            content, encoding="utf-8"
        )
    render(campaign, project / "src/images/information_recovery.pgf")

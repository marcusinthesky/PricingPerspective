"""Publish the archived observation and locality audit independently."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.figures.paper6._publication import (
    CELLS,
    PRIMARY,
    number,
    table,
    write_numbers,
)
from pipeline.figures.paper6.information_recovery.contracts import (
    GMM,
    Frontier,
    Observation,
)

if TYPE_CHECKING:
    from pathlib import Path


def audit_values(
    gmm: GMM, frontier: Frontier, observation: Observation
) -> dict[str, float | int | None]:
    """Keep the saved secondary findings, including unfavorable locality."""
    for record in (gmm, frontier, observation):
        if set(record.cells) != {key for key, _ in CELLS}:
            message = "secondary audit requires all four retained cells"
            raise ValueError(message)
    primary = observation.cells[PRIMARY]
    annual = gmm.cells[PRIMARY].fixed_identity_common_slope
    count = primary.locality.at_or_beyond_nearest_distinct_origin_reference_count
    defined = primary.locality.cumulative_defined_count
    values: dict[str, float | int | None] = {
        "observation-primary-heldout-pct": (
            primary.primary_centered_profile.heldout_skill * 100
        ),
        "observation-locality-count": count,
        "observation-locality-defined": defined,
        "observation-locality-pct": None if defined == 0 else 100 * count / defined,
        "observation-gmm-beta": annual.beta,
        "observation-gmm-2020-weight-pct": annual.effective_year_weights[1] * 100,
        "observation-equal-budget-pct": (
            frontier.cells[PRIMARY].equal_relative_budget_sufficient_threshold * 100
        ),
    }
    for row in primary.primary_centered_profile.per_year:
        values[f"observation-heldout-{row.year}"] = row.heldout_skill
    for key, _ in CELLS:
        values[f"observation-{key}-heldout-pct"] = (
            observation.cells[key].primary_centered_profile.heldout_skill * 100
        )
    return values


def observation_table() -> str:
    """Keep article-fold scores in their separate observation-model exhibit."""
    rows = [
        f"{label} & {number(f'observation-{key}-heldout-pct', 2)}" + r" \\"
        for key, label in CELLS
    ]
    return table(
        "Historical held-out observation-profile audit. Skill is the percentage "
        "reduction in centered profile squared error relative to predicting zero "
        "change; negative values are worse than zero. The article-fold audit "
        "uses geometry alone and is a secondary observation-model diagnostic. "
        "Its held-out score does not validate the financial coefficient.",
        "tab:observation-fit",
        "lr",
        r"Cell & Skill (\%)",
        rows,
    )


def publish(evidence: Path, project: Path) -> None:
    """Read only archived T82/T83 records, without any primary-table input."""
    values = audit_values(
        GMM.model_validate_json((evidence / "t82/gmm-results.json").read_bytes()),
        Frontier.model_validate_json(
            (evidence / "t82/frontier-results.json").read_bytes()
        ),
        Observation.model_validate_json((evidence / "t83/results.json").read_bytes()),
    )
    generated = project / "src/generated"
    write_numbers(generated / "information_audit_numbers.tex", values)
    (generated / "information_recovery_observation_table.tex").write_text(
        observation_table(), encoding="utf-8"
    )

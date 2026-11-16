"""Publish the complete-sample corporate projection from its saved analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from pipeline.figures.paper6._publication import (
    CELLS,
    PRIMARY,
    number,
    table,
    write_numbers,
)
from pipeline.figures.paper6.information_recovery import contracts

if TYPE_CHECKING:
    from pathlib import Path


class AnnualBeta(BaseModel):
    """An annual coefficient already computed by the analysis stage."""

    label: str
    beta: float


class Inference(BaseModel):
    """An unavailable inferential result remains unavailable in publication."""

    status: str = "unavailable"
    reason: str = Field(
        default="No calibrated coefficient inference was supplied.", min_length=1
    )
    lower95: float | None = None

    @model_validator(mode="after")
    def unavailable_has_no_bound(self) -> Inference:
        """Reject a numeric uncertainty bound attached to unavailable inference."""
        if self.status == "unavailable" and self.lower95 is not None:
            message = "unavailable inference cannot carry a numeric lower bound"
            raise ValueError(message)
        return self


class CorporateCell(contracts.EmpiricalCell):
    """Saved corporate values; publication performs no estimator calculation."""

    article_sensitivity_q025: float
    article_sensitivity_q05: float
    article_sensitivity_q975: float
    normalized_alignment: float
    within_projection_fit: float
    annual_beta: list[AnnualBeta]
    year_denominator: contracts.Denominator
    inference: Inference = Field(default_factory=Inference)


class CorporateAnalysis(BaseModel):
    """Independent publication input, with no campaign or T82/T83 dependency."""

    schema_version: Literal[1]
    firms: int
    directional_coordinates_per_cell: int
    draws: int
    inventory: contracts.Inventory
    cells: dict[str, CorporateCell]


def corporate_values(result: CorporateAnalysis) -> dict[str, float | int | str | None]:
    """Select saved values and apply display-unit conversions only."""
    if set(result.cells) != {key for key, _ in CELLS}:
        message = "corporate publication requires all four retained cells"
        raise ValueError(message)
    primary = result.cells[PRIMARY]
    values: dict[str, float | int | str | None] = {
        "firms": result.firms,
        "coordinates": result.directional_coordinates_per_cell,
        "redraws": result.draws,
        "native-dimension": result.inventory.width,
        "expanded-articles": result.inventory.expanded_articles,
        "matched-articles": result.inventory.matched_articles,
        "primary-beta": primary.beta,
        "financial-example": 0.01,
        "primary-percent-effect": primary.beta * 0.01,
        "primary-low": primary.article_sensitivity_q025,
        "primary-high": primary.article_sensitivity_q975,
        "primary-q05": primary.article_sensitivity_q05,
        "year-deletion-low": min(row.beta for row in primary.leave_one_year_out),
        "year-deletion-high": max(row.beta for row in primary.leave_one_year_out),
        "role-deletion-low": min(row.beta for row in primary.leave_one_firm_out),
        "role-deletion-high": max(row.beta for row in primary.leave_one_firm_out),
        "reference-share-pct": 100
        * max(row.share for row in primary.reference_denominator.shares),
        "target-share-pct": 100
        * max(row.share for row in primary.target_denominator.shares),
        "primary-2020-weight-pct": primary.year_denominator.shares[1].share * 100,
        "normalized-alignment": primary.normalized_alignment,
        "primary-inference-status": primary.inference.status,
        "primary-inference-reason": primary.inference.reason,
        "primary-inference-lower95": primary.inference.lower95,
    }
    values.update({f"annual-{row.label}-beta": row.beta for row in primary.annual_beta})
    for key, cell in result.cells.items():
        values.update(
            {
                f"{key}-beta": cell.beta,
                f"{key}-low": cell.article_sensitivity_q025,
                f"{key}-high": cell.article_sensitivity_q975,
                f"{key}-q05": cell.article_sensitivity_q05,
                f"{key}-alignment": cell.normalized_alignment,
                f"{key}-within-fit": cell.within_projection_fit,
            }
        )
    return values


def empirical_table() -> str:
    """Keep coefficient sensitivity separate from population inference."""
    rows = [
        " & ".join(
            [
                label,
                number(f"{key}-beta"),
                number(f"{key}-low"),
                number(f"{key}-q05"),
                number(f"{key}-high"),
                number(f"{key}-alignment", 4),
            ]
        )
        + r" \\"
        for key, label in CELLS
    ]
    return table(
        "Complete-sample historical directional association. Native denotes "
        "2,560 normalized coordinates. Percentiles describe conditional "
        "article-resampling sensitivity under the scheme "
        r"in Section~\ref{sec:data}. Alignment is the normalized pooled "
        "association; its square is descriptive within-projection fit. "
        "All measurement specifications use 100 firms, fixed 2018 reference "
        "profiles and four transitions ending in 2019--2022.",
        "tab:directional-evidence",
        "lrrrrr",
        r"Specification & $\widehat\beta$ & \shortstack{2.5th\\percentile} & "
        r"\shortstack{5th\\percentile} & \shortstack{97.5th\\percentile} & Alignment",
        rows,
    )


def publish(results: Path, project: Path) -> None:
    """Render the primary table without loading unrelated scientific evidence."""
    result = CorporateAnalysis.model_validate_json(results.read_bytes())
    values = corporate_values(result)
    generated = project / "src/generated"
    write_numbers(generated / "corporate_analysis_numbers.tex", values)
    (generated / "information_recovery_empirical_table.tex").write_text(
        empirical_table(), encoding="utf-8"
    )

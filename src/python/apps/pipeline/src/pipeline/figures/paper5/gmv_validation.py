"""Render the appendix-only Paper 5 covariance allocation diagnostic."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Never, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class Paper5GMVValidationError(ValueError):
    """Raised when the governed artifact cannot support the exhibit."""


def _invalid(message: str) -> Never:
    """Raise the renderer's contract error."""
    raise Paper5GMVValidationError(message)


_EXPECTED_ANALYSIS_ID = "paper3-spatial-portfolio-comparison-v4"
_EXPECTED_SCHEMA_VERSION = "paper3_spatial_portfolio_comparison.v4"
_EXPECTED_PANEL = "oos_primary"
_EXPECTED_REFERENCE_DESIGN = "uniform_cap_15"
_EXPECTED_CAP = 0.15
_EXPECTED_CANDIDATES = (
    "factor_diagonal_ledoit_wolf",
    "spatial_diagonal_w1_ledoit_wolf",
    "spatial_diagonal_w2_ledoit_wolf",
    "sample_gmv",
    "ledoit_wolf_gmv",
    "equal_weight",
)
_LABELS = {
    "factor_diagonal_ledoit_wolf": "Factor only + LW residual covariance",
    "spatial_diagonal_w1_ledoit_wolf": "Target-anchored W1",
    "spatial_diagonal_w2_ledoit_wolf": "Target-anchored W2",
    "sample_gmv": "Sample GMV",
    "ledoit_wolf_gmv": "Direct Ledoit--Wolf GMV",
    "equal_weight": "Equal weight",
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Require a mapping from the JSON scalar results."""
    if not isinstance(value, dict):
        _invalid(f"{label} must be a mapping")
    return cast("Mapping[str, object]", value)


def _finite(row: Mapping[str, object], key: str) -> float:
    """Read one finite numeric result field."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        _invalid(f"GMV result field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _invalid(f"GMV result field {key!r} must be finite")
    return result


def _validate_summary(results_json: Path) -> None:
    """Validate the scalar-results contract consumed by the exhibit."""
    raw_summary = json.loads(results_json.read_text(encoding="utf-8"))
    summary = _mapping(raw_summary, "Paper 3 portfolio summary")
    if summary.get("analysis_id") != _EXPECTED_ANALYSIS_ID:
        _invalid("Paper 3 portfolio summary has an unknown analysis ID")
    if summary.get("schema_version") != _EXPECTED_SCHEMA_VERSION:
        _invalid("Paper 3 portfolio summary has an unknown schema version")
    sample = _mapping(summary.get("sample"), "Paper 3 portfolio sample")
    if (
        sample.get("risk_unit") != "excess_return"
        or sample.get("factor_model") != "ff5_mom"
    ):
        _invalid("GMV validation requires FF5-plus-momentum excess returns")
    protocol = _mapping(summary.get("protocol"), "Paper 3 portfolio protocol")
    if protocol.get("maximum_weight") != _EXPECTED_CAP:
        _invalid("GMV exhibit requires the governed 15% cap")
    if protocol.get("residual_shrinkage_target") != "constant_correlation":
        _invalid("GMV exhibit requires the governed constant-correlation shrinkage")


def _validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Select the declared total-risk rows and return the matched variance."""
    required_columns = {
        "panel",
        "candidate",
        "reference_design",
        "candidate_status",
        "portfolio_role",
        "variance",
        "annualized_volatility",
        "effective_number_assets",
        "maximum_weight",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        _invalid(f"GMV result parquet is missing columns: {', '.join(sorted(missing))}")
    selected = frame.loc[
        (frame["panel"] == _EXPECTED_PANEL)
        & (frame["reference_design"] == _EXPECTED_REFERENCE_DESIGN)
        & frame["candidate"].isin(_EXPECTED_CANDIDATES)
    ].copy()
    if set(selected["candidate"]) != set(_EXPECTED_CANDIDATES):
        _invalid("GMV result parquet lacks the declared candidate set")
    if len(selected) != len(_EXPECTED_CANDIDATES):
        _invalid("GMV result parquet contains duplicate candidate rows")
    if not (selected["candidate_status"] == "computed").all():
        _invalid("GMV exhibit cannot render an infeasible candidate")
    if not (selected["portfolio_role"] == "total_risk_primary").all():
        _invalid("GMV exhibit cannot include residual-only diagnostics")
    if not (selected["maximum_weight"] <= _EXPECTED_CAP + 1e-8).all():
        _invalid("GMV exhibit contains a weight above the governed cap")

    factor_variance = float(
        selected.loc[
            selected["candidate"] == "factor_diagonal_ledoit_wolf", "variance"
        ].iloc[0]
    )
    if not math.isfinite(factor_variance) or factor_variance <= 0.0:
        _invalid("matched factor-only variance must be positive")
    return selected, factor_variance


def _validated_rows(
    results_path: Path, results_json: Path
) -> list[dict[str, float | str]]:
    """Validate and reduce the governed GMV-validation result artifact."""
    _validate_summary(results_json)
    selected, factor_variance = _validate_frame(pd.read_parquet(results_path))

    rows: list[dict[str, float | str]] = []
    for candidate in _EXPECTED_CANDIDATES:
        source = cast(
            "dict[str, object]",
            selected.loc[selected["candidate"] == candidate].iloc[0].to_dict(),
        )
        variance = _finite(source, "variance")
        rows.append(
            {
                "candidate": candidate,
                "label": _LABELS[candidate],
                "variance_1e4": 1e4 * variance,
                "annualized_volatility_pct": 100.0
                * _finite(source, "annualized_volatility"),
                "relative_to_factor_pct": 100.0 * variance / factor_variance,
                "effective_number_assets": _finite(source, "effective_number_assets"),
                "maximum_weight_pct": 100.0 * _finite(source, "maximum_weight"),
            }
        )
    return rows


def render_paper5_gmv_validation_table(
    results_path: Path,
    results_json: Path,
    output_path: Path,
) -> None:
    """Render the fixed-weight covariance allocation diagnostic."""
    rows = _validated_rows(results_path, results_json)
    generated_comment = (
        "% AUTO-GENERATED by pipeline render-paper5-gmv-validation-table -- "
        "do not edit by hand."
    )
    header = (
        r"    Covariance input & Realized variance & Ann. vol. & "
        r"Rel. to factor only & Eff. $N$ & Max. wt. \\"
    )
    caption = (
        r"  \caption{Covariance-sensitive portfolio diagnostic. Candidate weights "
        r"are formed from factor-complete 2018--2022 excess returns and frozen "
        r"at the 2022 formation vintage. They are long-only, fully invested, "
        r"and subject to a 15\% single-name cap; realized risk is evaluated on "
        r"constant-weight excess-return mixtures over 2023--2026. The "
        r"factor-only, W1, and W2 rows use diagonal innovation variances with "
        r"common constant-correlation residual shrinkage. Realized variance is "
        r"scaled by $10^4$; lower values are better.}"
    )
    rendered_rows = [
        " & ".join(
            [
                str(row["label"]),
                f"{float(row['variance_1e4']):.4f}",
                f"{float(row['annualized_volatility_pct']):.2f}\\%",
                f"{float(row['relative_to_factor_pct']):.1f}\\%",
                f"{float(row['effective_number_assets']):.2f}",
                f"{float(row['maximum_weight_pct']):.1f}\\%",
            ]
        )
        + r" \\"
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                generated_comment,
                r"\begin{table}[H]",
                r"  \centering",
                r"  \scriptsize",
                r"  \setlength{\tabcolsep}{3.0pt}",
                r"  \begin{tabular}{lrrrrr}",
                r"    \toprule",
                header,
                r"    \midrule",
                *rendered_rows,
                r"    \bottomrule",
                r"  \end{tabular}",
                caption,
                r"  \label{tab:p5-gmv-validation}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = [
    "Paper5GMVValidationError",
    "render_paper5_gmv_validation_table",
]

"""Render Paper 5's factor-conditioned field and peer interpretation tables."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path


def _number(row: dict[str, object], key: str) -> float:
    """Read a finite numeric field from a generated economic-value row."""
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"economic-value table field {key!r} must be numeric"
        raise TypeError(message)
    return float(value)


def _optional_number(row: dict[str, object], key: str) -> str:
    """Render a numeric field or an em dash for non-spatial benchmarks."""
    value = row.get(key)
    if value is None:
        return "--"
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"economic-value table field {key!r} must be numeric or null"
        raise TypeError(message)
    precision = 1 if key == "loglik" else 3
    return f"{float(value):.{precision}f}"


def _yes_no(value: object) -> str:
    """Render a boolean diagnostic without implying a statistical test."""
    if not isinstance(value, bool):
        message = "economic-value table boolean field is malformed"
        raise TypeError(message)
    return "Yes" if value else "No"


def _load(path: Path) -> dict[str, object]:
    """Load one economic-value result object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = "Paper 5 economic-value results must be a JSON object"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def render_paper5_economic_value_tables(
    results_path: Path,
    comparator_output_path: Path,
    peer_output_path: Path,
) -> None:
    """Render generated comparator and descriptive peer-panel fragments."""
    payload = _load(results_path)
    raw_models = payload.get("model_rows")
    raw_peers = payload.get("peer_panel_rows")
    if not isinstance(raw_models, list) or not isinstance(raw_peers, list):
        message = "Paper 5 economic-value results lack generated table rows"
        raise TypeError(message)
    models = cast("list[dict[str, object]]", raw_models)
    peers = cast("list[dict[str, object]]", raw_peers)
    comparator_rows: list[str] = []
    for row in models:
        model = str(row["model"])
        label = str(row["label"])
        if model == "factor_only":
            interval = "--"
        else:
            interval = (
                f"[{_number(row, 'held_out_gain_ci_lower'):.4f}, "
                f"{_number(row, 'held_out_gain_ci_upper'):.4f}]"
            )
        comparator_rows.append(
            " & ".join(
                [
                    label,
                    _optional_number(row, "rho_hat"),
                    _optional_number(row, "loglik"),
                    f"{_number(row, 'held_out_score'):.6f}",
                    f"{_number(row, 'held_out_score_gain_vs_factor_only'):.6f}",
                    interval,
                ]
            )
            + r" \\"
        )
    comparator_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparator_output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.0pt}",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                (
                    "Dependence specification & $\\hat\\rho$ & QLL & "
                    "Held-out score & Gain "
                    "& 95\\% CI "
                    r"\\"
                ),
                r"\midrule",
                *comparator_rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Factor-conditioned peer-field and covariance "
                    r"benchmarks. "
                    r"All rows use the primary FF5-plus-momentum controls, the same "
                    r"firms and dates; fields use firm information dated through 2022 "
                    r"and are held fixed throughout the 2023--2026 evaluation, and "
                    r"covariance inputs are calibrated on 2018--2022 returns. "
                    r"The held-out score is the mean Gaussian "
                    r"quasi-log score "
                    r"per firm-day on 2023--2026; gain is relative to the factor-only "
                    r"fit. The direct Ledoit--Wolf row estimates a conventional "
                    r"factor-residual covariance from the calibration sample and "
                    r"therefore has no spatial coefficient or QLL. Intervals are "
                    r"paired stationary-block bootstrap intervals that refit every "
                    r"comparator on each calibration draw, matching the resampling "
                    r"architecture of \Cref{tab:spatial-value}.}"
                ),
                r"\label{tab:p5-economic-field-comparison}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    peer_table_rows: list[str] = []
    previous_target: str | None = None
    for row in peers:
        target = str(row["target"])
        if previous_target is not None and target != previous_target:
            peer_table_rows.append(r"\addlinespace[1.5pt]")
        previous_target = target
        role_key = str(row["target_role"])
        role = {"declared_amd": "declared AMD"}.get(
            role_key, role_key.replace("_", " ")
        )
        peer_table_rows.append(
            " & ".join(
                [
                    target,
                    role,
                    str(row["candidate"]),
                    f"{_number(row, 'weight'):.4f}",
                    str(int(_number(row, "distance_rank"))),
                    _yes_no(row["same_sector"]),
                    _yes_no(row["same_industry"]),
                    _yes_no(row["co_mention"]),
                ]
            )
            + r" \\"
        )
    peer_output_path.parent.mkdir(parents=True, exist_ok=True)
    peer_output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.5pt}",
                r"\begin{tabular}{ll l r r c c c}",
                r"\toprule",
                (
                    "Target & Selection role & Peer & $W^\\flat_{ij}$ & W2 rank & "
                    "Same sector & Same industry & Co-mention "
                    r"\\"
                ),
                r"\midrule",
                *peer_table_rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Descriptive W2 peer interpretation panel. Targets are "
                    r"AMD and deterministic minimum, median, and maximum effective "
                    r"source-count firms, with lexical tie-breaking. Each row lists "
                    r"one of the target's five largest actual reconstruction weights; "
                    r"the distance rank is the target's pairwise W2 rank. Sector, "
                    r"industry, and co-mention columns report metadata associated "
                    r"with each candidate.}"
                ),
                r"\label{tab:p5-peer-interpretation}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = ["render_paper5_economic_value_tables"]

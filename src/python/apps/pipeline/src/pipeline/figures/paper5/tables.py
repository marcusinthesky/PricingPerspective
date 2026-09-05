"""Generated Paper 5 representation-sensitivity tables."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

_OPERATOR_LABELS = {
    "w_flat": r"Barycentric $W^\flat$",
    "w_h": r"RBF--Wasserstein $W^h$",
    "w_co_mentions": r"Persistent news $W^{\mathrm{news}}$",
    "equal_support": r"Equal active support",
}
_CONTRAST_PRECISION = 5


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Paper 5 representation table requires numeric {key!r}"
        raise TypeError(message)
    return float(value)


def _result_row(label: str, dimension: str, matrix_id: str, result: object) -> str:
    if not isinstance(result, dict):
        message = f"Paper 5 table result for {matrix_id} must be a mapping"
        raise TypeError(message)
    row = cast("dict[str, object]", result)
    rho = _number(row, "rho_hat")
    lower = _number(row, "rho_ci_lower")
    upper = _number(row, "rho_ci_upper")
    adjustment = rho / (1.0 - rho)
    return (
        " & ".join(
            [
                label,
                dimension,
                _OPERATOR_LABELS[matrix_id],
                f"{rho:.3f} [{lower:.3f}, {upper:.3f}]",
                f"{adjustment:.2f}",
                f"{_number(row, 'loglik'):.1f}",
            ]
        )
        + r" \\"
    )


def _cell_label(cell: dict[str, object]) -> str:
    label = str(cell["label"])
    if str(cell.get("provider_id", "")).startswith("ettax-"):
        return label
    dimension = cell.get("effective_dimension")
    native = cell.get("native_dimension")
    if isinstance(dimension, int) and isinstance(native, int):
        return f"{label} full" if dimension == native else f"{label}@{dimension}"
    return label


def _contrast_rows(payload: dict[str, object]) -> list[str]:
    analysis = payload.get("contrast_analysis", {})
    if not isinstance(analysis, dict) or not any(
        isinstance(analysis.get(metric), list) for metric in ("direct_rho", "gap")
    ):
        return []
    labels = {
        str(cast("dict[str, object]", row["cell"]).get("cell_id", "")): _cell_label(
            cast("dict[str, object]", row["cell"])
        )
        for row in cast("list[dict[str, object]]", payload["cells"])
    }
    rows: list[str] = []
    for metric, panel_heading in (
        (
            "direct_rho",
            (
                r"\multicolumn{7}{l}{\textit{Panel A: direct "
                r"barycentric-$\rho$ contrasts}} \\"
            ),
        ),
        (
            "gap",
            (
                r"\multicolumn{7}{l}{\textit{Panel B: "
                r"barycentric-minus-RBF gap contrasts}} \\"
            ),
        ),
    ):
        raw_rows = analysis.get(metric, [])
        if not isinstance(raw_rows, list):
            continue
        if rows:
            rows.append(r"\midrule")
        rows.append(panel_heading)
        previous_family: object | None = None
        for raw in cast("list[dict[str, object]]", raw_rows):
            family = raw.get("family")
            if previous_family is not None and family != previous_family:
                rows.append(r"\addlinespace[1.5pt]")
            previous_family = family
            candidate = labels.get(
                str(raw.get("candidate_cell")), str(raw.get("candidate_cell"))
            )
            reference = labels.get(
                str(raw.get("reference_cell")), str(raw.get("reference_cell"))
            )
            label = f"{candidate} $-$ {reference}"
            if raw.get("status") != "computed":
                rows.append(
                    f"{label} & \\ppmissing & \\ppmissing & \\ppmissing & "
                    f"\\ppmissing & \\ppmissing & \\ppmissing \\\\"
                )
                continue
            equivalent = "Yes" if raw.get("equivalent") is True else "No"
            bound = raw.get("equivalence_bound")
            rows.append(
                " & ".join(
                    [
                        label,
                        f"{_number(raw, 'estimate'):.{_CONTRAST_PRECISION}f}",
                        (
                            f"[{_number(raw, 'ci_low'):.{_CONTRAST_PRECISION}f}, "
                            f"{_number(raw, 'ci_high'):.{_CONTRAST_PRECISION}f}]"
                        ),
                        f"{_number(raw, 'pvalue'):.3f}",
                        f"{_number(raw, 'holm_pvalue'):.3f}",
                        f"{float(bound):.{_CONTRAST_PRECISION}f}"
                        if isinstance(bound, int | float)
                        else "--",
                        equivalent if family == "vintage" else "--",
                    ]
                )
                + r" \\",
            )
    return rows


def render_paper5_representation_ablation_table(
    summary_path: Path,
    output_path: Path,
) -> None:
    """Render the pooled barycentric/RBF roster and paired contrast tables."""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        message = "Paper 5 representation merge must contain a cells list"
        raise TypeError(message)
    rows: list[str] = []
    for raw_cell in cast("list[dict[str, object]]", payload["cells"]):
        cell = cast("dict[str, object]", raw_cell["cell"])
        results = cast("dict[str, object]", raw_cell["results"])
        result_periods = cast("dict[str, object]", results["results"])
        pooled = cast("dict[str, object]", result_periods["pooled_2023_2026"])
        label = _cell_label(cell)
        if bool(cell["canonical"]):
            label += " (primary)"
        for index, matrix_id in enumerate(("w_flat", "w_h")):
            rows.append(
                _result_row(
                    label if index == 0 else "",
                    str(cell["effective_dimension"]) if index == 0 else "",
                    matrix_id,
                    pooled[matrix_id],
                )
            )
        rows.append(r"\addlinespace[1.5pt]")
    fixed = cast("dict[str, object]", payload["fixed_comparators"])
    contrast_rows = _contrast_rows(payload)
    period = payload.get("period")
    schedule = "the common joint-date schedule"
    if isinstance(period, dict) and isinstance(period.get("trading_days"), int):
        schedule = f"the common {period['trading_days']}-date schedule"
    rows.extend(
        [
            r"\midrule",
            _result_row(
                "Embedding-free", "--", "w_co_mentions", fixed["w_co_mentions"]
            ),
            _result_row(
                "Primary support", "--", "equal_support", fixed["equal_support"]
            ),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.3pt}",
                r"\begin{tabular}{lllrrr}",
                r"\toprule",
                (
                    r"Encoder & Dim. & Interaction matrix & $\hat\rho$ [95\% CI] "
                    r"& $\hat\lambda$ & Log-likelihood \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Pooled 2023--2026 representation sensitivity of "
                    r"the single-field spatial horse race across seven base "
                    r"representations and three matched EttaX encoder vintages. "
                    r"Each encoder row compares "
                    r"the barycentric interaction field $W^\flat$, obtained by "
                    r"target-anchored Wasserstein-barycentric reconstruction, with the "
                    r"median-bandwidth RBF transform of its matching squared-W$_2$ "
                    r"matrix. Intervals use the same 2,000 joint-date "
                    rf"stationary bootstrap on {schedule}. "
                    r"$\hat\lambda=\hat\rho/(1-\hat\rho)$ is the implied adjustment "
                    r"index; log-likelihood is descriptive. Persistent news "
                    r"co-mentions are "
                    r"embedding-free; equal active support is shown once for the "
                    r"primary barycentric interaction field. These are paired "
                    r"conditional-fit "
                    r"sensitivities, not a representation-selection test.}"
                ),
                r"\label{tab:p5-representation-ablation}",
                r"\end{table}",
                r"\medskip",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.0pt}",
                r"\begin{tabularx}{\linewidth}{Xrrrrrr}",
                r"\toprule",
                (
                    r"Contrast ($c-r$) & Estimate & 95\% CI & Raw $p$ & "
                    r"Holm $p$ & Bound & Joint equiv. \\"
                ),
                r"\midrule",
                *contrast_rows,
                r"\bottomrule",
                r"\end{tabularx}",
                (
                    r"\caption{Paired common-schedule QMLE contrasts. Panel A "
                    r"estimates "
                    r"$\rho^\flat_c-\rho^\flat_r$; Panel B estimates "
                    r"$(\rho^\flat_c-\rho^h_c)-(\rho^\flat_r-\rho^h_r)$. Raw "
                    r"$p$-values are two-sided add-one sign-tail probabilities, and "
                    r"Holm adjustment is applied separately within each metric's seven "
                    r"base and three vintage contrasts. "
                    r"Estimates and intervals retain five decimal places so endpoints "
                    r"near zero remain visible. "
                    r"Vintage equivalence requires both intervals to lie strictly "
                    r"inside "
                    r"their respective BGE-large-minus-Qwen3-8B@1024 "
                    r"benchmark-calibrated, post-specified bounds; the decision shown "
                    r"is joint. No contrast is causal or an encoder-superiority claim.}"
                ),
                r"\label{tab:p5-representation-contrasts}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = ["render_paper5_representation_ablation_table"]

"""Generated Paper 3 robustness tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from pathlib import Path

_DESIGN_ORDER = (
    "uniform_cap_12p5",
    "uniform_cap_15",
    "effective_n_matched",
    "concentrated_cap_15",
)

_MAIN_TEXT_CELL_ORDER = (
    "qwen8b_full",
    "qwen8b_1024",
    "qwen4b_1024",
    "bge_large_full",
    "qwen8b_64",
    "ettax_v0",
    "ettax_v1",
    "ettax_v3",
)


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Paper 3 representation table requires numeric {key!r}"
        raise TypeError(message)
    return float(value)


def _percentile(designs: object, label: str, candidate: str) -> str:
    if not isinstance(designs, list):
        message = "Paper 3 representation table requires reference designs"
        raise TypeError(message)
    matches = [
        design
        for design in designs
        if isinstance(design, dict) and design.get("label") == label
    ]
    if len(matches) != 1:
        message = f"Paper 3 representation table requires design {label!r}"
        raise ValueError(message)
    percentile = cast("dict[str, object]", matches[0]).get("percentile")
    if not isinstance(percentile, dict):
        message = f"Paper 3 design {label!r} has no percentile mapping"
        raise TypeError(message)
    value = percentile.get(candidate)
    return "--" if value is None else f"{100.0 * float(value):.2f}"


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
    if not isinstance(analysis, dict) or not isinstance(
        analysis.get("contrasts"), list
    ):
        return []
    labels = {
        str(cast("dict[str, object]", row["cell"])["cell_id"]): _cell_label(
            cast("dict[str, object]", row["cell"])
        )
        for row in cast("list[dict[str, object]]", payload["cells"])
    }
    rows: list[str] = []
    previous_family: object | None = None
    for raw in cast("list[dict[str, object]]", analysis["contrasts"]):
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
            rows.append(f"{label} {' & \\\\ppmissing' * 6} \\\\")
            continue
        bound = raw.get("equivalence_bound")
        equivalent = "Yes" if raw.get("equivalent") is True else "No"
        rows.append(
            " & ".join(
                [
                    label,
                    f"{_number(raw, 'estimate'):.3f}",
                    f"[{_number(raw, 'ci_low'):.3f}, {_number(raw, 'ci_high'):.3f}]",
                    f"{_number(raw, 'pvalue'):.3f}",
                    f"{_number(raw, 'holm_pvalue'):.3f}",
                    f"{float(bound):.3f}" if isinstance(bound, int | float) else "--",
                    equivalent if family == "vintage" else "--",
                ]
            )
            + r" \\",
        )
    return rows


def _main_text_rows(cells: list[dict[str, object]]) -> list[str]:
    """Select one compact set of rows spanning the four measurement margins."""
    indexed = {
        str(cast("dict[str, object]", row["cell"])["cell_id"]): row for row in cells
    }
    missing = [cell_id for cell_id in _MAIN_TEXT_CELL_ORDER if cell_id not in indexed]
    if missing:
        message = (
            "Paper 3 main-text representation table is missing cells: "
            + ", ".join(missing)
        )
        raise ValueError(message)

    rows: list[str] = []
    for cell_id in _MAIN_TEXT_CELL_ORDER:
        if cell_id == "ettax_v0":
            rows.append(r"\addlinespace[1.5pt]")
        row = indexed[cell_id]
        cell = cast("dict[str, object]", row["cell"])
        benchmark = cast("dict[str, object]", row["news_only_benchmark"])
        portfolios = cast("dict[str, object]", benchmark["portfolios"])
        news = cast("dict[str, object]", portfolios["news_only"])
        reference = cast("dict[str, object]", benchmark["reference_population"])
        label = _cell_label(cell)
        if bool(cell["canonical"]):
            label += r" (primary)"
        rows.append(
            " & ".join(
                [
                    label,
                    str(cell["effective_dimension"]),
                    _percentile(
                        reference["designs"], "effective_n_matched", "news_only"
                    ),
                    f"{100.0 * _number(news, 'gmv_relative_variance'):.1f}",
                ]
            )
            + r" \\"
        )
    return rows


def render_paper3_representation_ablation_table(
    summary_path: Path,
    output_path: Path,
    main_output_path: Path | None = None,
) -> None:
    """Render appendix detail and, when requested, a compact main-text table."""
    payload = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        message = "Paper 3 representation merge must contain a cells list"
        raise TypeError(message)
    cells = cast("list[dict[str, object]]", payload["cells"])
    rows: list[str] = []
    for row in cells:
        cell = cast("dict[str, object]", row["cell"])
        benchmark = cast("dict[str, object]", row["news_only_benchmark"])
        portfolios = cast("dict[str, object]", benchmark["portfolios"])
        news = cast("dict[str, object]", portfolios["news_only"])
        reference = cast("dict[str, object]", benchmark["reference_population"])
        values = [
            _percentile(reference["designs"], label, "news_only")
            for label in _DESIGN_ORDER
        ]
        label = _cell_label(cell)
        if bool(cell["canonical"]):
            label += r" (primary)"
        rows.append(
            " & ".join(
                [
                    label,
                    str(cell["effective_dimension"]),
                    *values,
                    f"{100.0 * _number(news, 'gmv_relative_variance'):.1f}",
                ]
            )
            + r" \\"
        )

    canonical = next(
        row
        for row in cells
        if bool(cast("dict[str, object]", row["cell"])["canonical"])
    )
    benchmark = cast("dict[str, object]", canonical["news_only_benchmark"])
    portfolios = cast("dict[str, object]", benchmark["portfolios"])
    equal = cast("dict[str, object]", portfolios["equal_weight"])
    reference = cast("dict[str, object]", benchmark["reference_population"])
    rows.extend(
        [
            r"\midrule",
            " & ".join(
                [
                    "Equal risk weights",
                    "--",
                    *[
                        _percentile(reference["designs"], label, "equal_weight")
                        for label in _DESIGN_ORDER
                    ],
                    f"{100.0 * _number(equal, 'gmv_relative_variance'):.1f}",
                ]
            )
            + r" \\",
        ]
    )
    contrast_rows = _contrast_rows(payload)
    universe = payload.get("universe")
    sample_scope = "the common standardized-return panel"
    if isinstance(universe, dict):
        n_tickers = universe.get("n_tickers")
        n_observations = universe.get("n_return_observations")
        if isinstance(n_tickers, int) and isinstance(n_observations, int):
            sample_scope = (
                f"the common {n_tickers}-firm, {n_observations:,}-date "
                "standardized-return panel"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.2pt}",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                (
                    r"Encoder & Dim. & Uniform 12.5\% & Uniform 15\% & "
                    r"Fixed $\alpha=0.8$ & Concentrated 15\% & Rel. GMV \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Representation sensitivity of the news-only allocation "
                    r"across seven base representations and three matched "
                    f"EttaX encoder vintages on {sample_scope}. The four middle "
                    r"columns report the "
                    r"percentage of "
                    r"fixed-law allocations with standardized variance no greater "
                    r"than the candidate's; lower is better. Relative variance indexes "
                    r"the long-only sample GMV to 100, so values above 100 measure "
                    r"percentage excess variance and lower is better. A dash marks a "
                    r"candidate that "
                    r"violates the law's unchanged cap. The $\alpha=0.8$ law is a "
                    r"fixed moderately concentrated comparison; its realized mean "
                    r"effective $N$ is reported rather than assumed to match the "
                    r"primary allocation. Returns "
                    r"evaluate every fixed news-only allocation in sample; "
                    r"they do not construct it.}"
                ),
                r"\label{tab:p3-representation-ablation}",
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
                    r"Holm $p$ & Bound & Equiv. \\"
                ),
                r"\midrule",
                *contrast_rows,
                r"\bottomrule",
                r"\end{tabularx}",
                (
                    r"\caption{Paired stationary-return-bootstrap contrasts for the "
                    r"ten-cell representation ladder. The seven base contrasts are "
                    r"descriptive candidate-minus-reference relative-GMV index-point "
                    r"differences, so a negative estimate favours the candidate. Raw "
                    r"$p$-values are two-sided add-one sign-tail probabilities; Holm "
                    r"adjustment is applied separately to the seven base and three "
                    r"vintage contrasts. The "
                    r"EttaX rows additionally use the BGE-large-minus-Qwen3-8B@1024 "
                    r"interval as a benchmark-calibrated, post-specified robustness "
                    r"threshold. Equivalence requires strict containment of the full "
                    r"interval; this is not a causal or architecture-only comparison.}"
                ),
                r"\label{tab:p3-representation-contrasts}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if main_output_path is None:
        return
    main_output_path.parent.mkdir(parents=True, exist_ok=True)
    main_output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                (
                    r"\caption{Selected representation sensitivity of the news-only "
                    r"allocation. The percentile column reports the percentage of "
                    r"allocations under the fixed $\alpha=0.8$ reference law whose "
                    r"standardized "
                    r"variance is no greater than the candidate's. Relative variance "
                    r"indexes the long-only sample GMV to 100. Lower is better in both "
                    r"columns. The complete ladder and paired bootstrap contrasts "
                    r"appear in \Cref{sec:p3-representation-ablation}.}"
                ),
                r"\label{tab:p3-representation-main}",
                r"\setlength{\tabcolsep}{7.0pt}",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                (
                    r"Encoder & Dim. & Fixed-$\alpha=0.8$ percentile "
                    r"& Rel. GMV \\"
                ),
                r"\midrule",
                *_main_text_rows(cells),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = ["render_paper3_representation_ablation_table"]

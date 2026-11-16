"""Paper 1 representation-sensitivity table fragments.

``render-paper1-representation-sensitivity`` writes the encoder/width/distance
and article-sampling fragments from the saved
``paper1-representation-sensitivity`` results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pipeline.figures._latex import escape_latex
from pipeline.figures.paper1._format import fmt
from pipeline.figures.shared_tables import _render_longtable

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Paper1RepresentationSensitivityRenderPaths:
    """Saved results consumed and fragments produced by the renderer."""

    results_file: Path
    output_dir: Path


def render_paper1_representation_sensitivity(
    paths: Paper1RepresentationSensitivityRenderPaths,
) -> None:
    """Render the representation and sampling sensitivity fragments."""
    representation = json.loads(paths.results_file.read_text(encoding="utf-8"))
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "w2_representation_sensitivity.tex").write_text(
        render_w2_representation_sensitivity(representation) + "\n"
    )
    (output_dir / "w2_sampling_diagnostics.tex").write_text(
        render_w2_sampling_diagnostics(representation) + "\n"
    )


_DISTANCE_LABELS = {
    "wasserstein_w2": r"$W_2$",
    "wasserstein_w1": r"$W_1$",
    "energy_v": "Energy",
}


_MODEL_LABELS = {
    "bge-large-en-v1.5": "BGE-large-v1.5",
    "bge-m3": "BGE-m3",
    **{f"chronobert-{year}": f"ChronoBERT {year}" for year in range(2017, 2025)},
    "qwen3-embedding-4b": "Qwen3-4B",
    "qwen3-embedding-8b": "Qwen3-8B",
}


_NATIVE_DIMENSIONS = {
    "bge-large-en-v1.5": 1024,
    "bge-m3": 1024,
    "qwen3-embedding-4b": 2560,
    "qwen3-embedding-8b": 4096,
}


def _model_label(value: object) -> str:
    model = str(value)
    return _MODEL_LABELS.get(model, escape_latex(model))


def _contrast_cell_label(cell: dict[str, object]) -> str:
    """Make repeated provider labels distinguish full and truncated rows."""
    label = _model_label(cell.get("model", ""))
    model = str(cell.get("model", ""))
    dimension = cell.get("embedding_dimension")
    native = _NATIVE_DIMENSIONS.get(model)
    if isinstance(dimension, int) and native is not None:
        return f"{label} full" if dimension == native else f"{label}@{dimension}"
    return label


def _sensitivity_rows(
    cells: list[dict[str, object]], group: str, *, size_field: str
) -> list[str]:
    rows: list[str] = []
    for cell in cells:
        if cell.get("group") != group:
            continue
        if cell.get("status") != "computed":
            rows.append(
                " & ".join(
                    [
                        _model_label(cell.get("model", "")),
                        str(cell.get(size_field, "")),
                        _DISTANCE_LABELS.get(
                            str(cell.get("distance", "")),
                            escape_latex(str(cell.get("distance", ""))),
                        ),
                        r"\ppmissing",
                        r"\ppmissing",
                        r"\ppmissing",
                    ]
                )
                + r" \\"
            )
            continue
        effect_ci = (
            f"[{fmt(cell.get('effect_one_sd_ci_low'), 3)}, "
            f"{fmt(cell.get('effect_one_sd_ci_high'), 3)}]"
        )
        rows.append(
            " & ".join(
                [
                    _model_label(cell["model"]),
                    str(cell[size_field]),
                    _DISTANCE_LABELS.get(
                        str(cell["distance"]), escape_latex(str(cell["distance"]))
                    ),
                    fmt(cell.get("geometry_spearman_vs_canonical_w2"), 3),
                    f"{fmt(cell.get('effect_one_sd'), 3)} {effect_ci}",
                    fmt(cell.get("partial_r2"), 3),
                ]
            )
            + r" \\"
        )
    return rows


def _contrast_rows(summary: dict[str, object], family: str) -> list[str]:
    analysis = cast("dict[str, object]", summary.get("encoder_vintage_analysis", {}))
    contrasts = cast(
        "list[dict[str, object]]", summary.get("representation_contrasts", [])
    )
    if not contrasts:
        contrasts = cast("list[dict[str, object]]", analysis.get("contrasts", []))
    labels = {
        str(cell.get("representation_cell")): _contrast_cell_label(cell)
        for cell in cast("list[dict[str, object]]", summary["cells"])
    }
    rows: list[str] = []
    for contrast in contrasts:
        if contrast.get("family", family) != family:
            continue
        candidate = labels.get(
            str(contrast.get("candidate_cell")),
            _model_label(
                contrast.get("later_provider", contrast.get("candidate_cell", ""))
            ),
        )
        reference = labels.get(
            str(contrast.get("reference_cell")),
            _model_label(
                contrast.get("earlier_provider", contrast.get("reference_cell", ""))
            ),
        )
        label = f"{candidate} $-$ {reference}"
        if contrast.get("status") != "computed":
            missing = 6 if family == "vintage" else 4
            rows.append(f"{label} {' & \\\\ppmissing' * missing} \\\\")
            continue
        interval = (
            f"[{fmt(contrast.get('ci_low'), 3)}, {fmt(contrast.get('ci_high'), 3)}]"
        )
        decision = "Yes" if contrast.get("equivalent") is True else "No"
        values = [
            label,
            fmt(contrast.get("estimate"), 3),
            interval,
            fmt(contrast.get("pvalue"), 3),
            fmt(contrast.get("holm_pvalue"), 3),
        ]
        if family == "vintage":
            values.extend([fmt(contrast.get("equivalence_bound"), 3), decision])
        rows.append(" & ".join(values) + r" \\")
    return rows


def render_w2_representation_sensitivity(summary: dict[str, object]) -> str:
    """Render encoder/width and statistical-distance comparisons."""
    cells = cast("list[dict[str, object]]", summary["cells"])
    sample_cell = next((cell for cell in cells if cell.get("status") == "computed"), {})
    n_firms = sample_cell.get("n_firms")
    n_dyads = sample_cell.get("n_dyads")
    sample_scope = (
        f"the common {n_firms}-firm, {n_dyads:,}-dyad panel"
        if isinstance(n_firms, int) and isinstance(n_dyads, int)
        else "the common dyad panel"
    )
    vintage_analysis = cast(
        "dict[str, object]", summary.get("encoder_vintage_analysis", {})
    )
    vintage_rows = _sensitivity_rows(
        cells, "encoder_vintage", size_field="embedding_dimension"
    )
    base_contrast_rows = _contrast_rows(summary, "base")
    contrast_rows = _contrast_rows(summary, "vintage")
    lines = [
        "% AUTO-GENERATED -- do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrcr}",
        r"\toprule",
        (
            r"Encoder & Dim. & Distance & Rank $\rho$ & "
            r"$\Delta D$/1 SD [95\% CI] & Partial $R^2$ \\"
        ),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel A: encoder and embedding dimension}} \\",
        *_sensitivity_rows(
            cells, "encoder_dimension", size_field="embedding_dimension"
        ),
        r"\midrule",
        (
            r"\multicolumn{6}{l}{\textit{Panel B: distance family, canonical "
            r"encoder and dimension}} \\"
        ),
        *_sensitivity_rows(cells, "distance_family", size_field="embedding_dimension"),
    ]
    if vintage_rows:
        lines.extend(
            [
                r"\midrule",
                (
                    r"\multicolumn{6}{l}{\textit{Panel C: published annual "
                    r"ChronoBERT sensitivity}} \\"
                ),
                *vintage_rows,
            ]
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    lines.extend(
        [
            (
                r"\caption{Representation sensitivity under the primary estimator. "
                r"Every row uses symmetric additive firm effects and the same 1,999 "
                r"node-bootstrap schedule. Rank agreement is Spearman correlation "
                r"with the canonical Qwen3-Embedding-8B, 128-article rooted-$W_2$ "
                rf"geometry on {sample_scope}. The full-width Qwen rows change "
                r"capacity and "
                r"native width jointly; the 8B prefixes isolate Matryoshka truncation; "
                r"BGE-large changes architecture, training pipeline, and provider. "
                r"The ChronoBERT panel uses published annual cumulative-training "
                r"checkpoints with a fixed final-layer token-mean inference policy; "
                r"it is not a matched-compute causal leakage test. "
                r"Standardized effects use each row's dyadic standard deviation. "
                r"$W_1$ and energy are lower-cost descriptive comparators.}"
            ),
            r"\label{tab:w2-representation-sensitivity}",
            r"\end{table}",
        ]
    )
    for family, rows, label in (
        ("base", base_contrast_rows, "tab:w2-base-contrasts"),
        ("vintage", contrast_rows, "tab:w2-vintage-contrasts"),
    ):
        if not rows:
            continue
        vintage = family == "vintage"
        confidence = float(str(vintage_analysis.get("confidence_level", 0.95)))
        header = (
            rf"Contrast ($c-r$) & Estimate & {100.0 * confidence:.0f}\% CI & "
            r"$p$ & $p_H$"
        )
        if vintage:
            header += r" & Bound & Eq."
        header += r" \\"
        caption = (
            f"{'ChronoBERT annual vintage' if vintage else 'Base representation'} "
            r"paired contrasts under the primary estimator. Estimates are "
            r"candidate-minus-reference differences in $\Delta D$ per one dyadic "
            r"standard deviation, using identical draws from the 1,999 "
            r"node-bootstrap schedule. $p$ denotes the two-sided add-one sign-tail "
            r"probability; $p_H$ denotes Holm adjustment separately within the seven "
            r"base and 28 vintage contrasts. Intervals are marginal, not simultaneous."
        )
        if vintage:
            caption += (
                r" Eq. requires strict containment of the full interval in the "
                r"fixed inherited benchmark-calibrated robustness bound shown; "
                r"this is not a new ChronoBERT preregistration or a matched-compute "
                r"causal comparison."
            )
        lines.append(
            _render_longtable(
                rows,
                columns=(
                    r"@{}p{0.35\linewidth}" + ("rrrrrr" if vintage else "rrrr") + "@{}"
                ),
                header=header,
                caption=caption,
                label=label,
                font_size="footnotesize",
                tabcolsep="3.5pt",
            )
        )
    return "\n".join(lines)


def render_w2_sampling_diagnostics(summary: dict[str, object]) -> str:
    """Render the canonical encoder's article-count sensitivity."""
    cells = cast("list[dict[str, object]]", summary["cells"])
    return "\n".join(
        [
            "% AUTO-GENERATED -- do not edit by hand.",
            r"\begin{table}[H]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lrrrcr}",
            r"\toprule",
            (
                r"Encoder & Articles & Distance & Rank $\rho$ & "
                r"$\Delta D$/1 SD [95\% CI] & Partial $R^2$ \\"
            ),
            r"\midrule",
            *_sensitivity_rows(cells, "sample_size", size_field="articles"),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Article-sampling sensitivity for Qwen3-Embedding-8B. "
                r"Prefixes are deterministic, rank agreement is measured against "
                r"the 128-article $W_2$ matrix, and all association intervals "
                r"reuse the "
                r"common node schedule. The rows diagnose finite article sampling; "
                r"they do not quantify embedding-estimation uncertainty.}"
            ),
            r"\label{tab:w2-sampling-diagnostics}",
            r"\end{table}",
        ]
    )

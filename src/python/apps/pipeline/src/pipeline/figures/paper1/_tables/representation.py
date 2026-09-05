"""Primary W2 and representation-sensitivity table fragments."""

from __future__ import annotations

from typing import cast

from pipeline.figures.paper1._tables.format import _escape, _fmt

_DISTANCE_LABELS = {
    "wasserstein_w2": r"$W_2$",
    "wasserstein_w1": r"$W_1$",
    "energy_v": "Energy",
}

_MODEL_LABELS = {
    "bge-large-en-v1.5": "BGE-large-v1.5",
    "bge-m3": "BGE-m3",
    "ettax-v0": "EttaX V0",
    "ettax-v1": "EttaX V1",
    "ettax-v3": "EttaX V3",
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
    return _MODEL_LABELS.get(model, _escape(model))


def _contrast_cell_label(cell: dict[str, object]) -> str:
    """Make repeated provider labels distinguish full and truncated rows."""
    label = _model_label(cell.get("model", ""))
    model = str(cell.get("model", ""))
    dimension = cell.get("embedding_dimension")
    native = _NATIVE_DIMENSIONS.get(model)
    if isinstance(dimension, int) and native is not None:
        return f"{label} full" if dimension == native else f"{label}@{dimension}"
    return label


def _primary_row(
    label: str, arm: dict[str, object], n_firms: object, n_dyads: object
) -> str:
    coefficient = arm.get("coefficient", arm.get("coef"))
    effect = arm.get("effect_one_sd")
    interval = f"[{_fmt(arm.get('ci_low'), 3)}, {_fmt(arm.get('ci_high'), 3)}]"
    return (
        " & ".join(
            [
                label,
                str(n_firms),
                str(n_dyads),
                _fmt(coefficient, 3),
                interval,
                _fmt(arm.get("pvalue"), 3),
                _fmt(effect, 3),
                _fmt(arm.get("within_r2"), 3),
            ]
        )
        + r" \\"
    )


def render_w2_primary_results(dyadic: dict[str, object], oos: dict[str, object]) -> str:
    """Render the in-window and timing-separated primary estimates."""
    primary = cast("dict[str, object]", dyadic["w2_primary_regression"])
    panel = cast("dict[str, object]", oos.get("panel", {}))
    oos_arm = cast("dict[str, object]", oos.get("w2_arm", {}))
    rows = [
        _primary_row(
            "2018--2022 returns",
            primary,
            dyadic["n_firms"],
            dyadic["n_dyads"],
        )
    ]
    if oos_arm:
        rows.append(
            _primary_row(
                "2023--2026 returns",
                oos_arm,
                panel["n_firms"],
                panel["n_dyads"],
            )
        )
    return "\n".join(
        [
            "% AUTO-GENERATED -- do not edit by hand.",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\begin{tabularx}{\linewidth}{Xrrrrrrr}",
            r"\toprule",
            (
                r"Return window & Firms & Dyads & $\hat\beta_{W_2}$ & 95\% CI & "
                r"$p$ & $\Delta D$/1 SD & Partial $R^2$ \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            (
                r"\caption{Primary symmetric-firm-effect association. The text-side "
                r"rooted $W_2$ matrix is fixed across rows. The first row relates the "
                r"2018--2022 return chord distance to that geometry; the second uses "
                r"the later 2023--2026 return panel. Intervals and two-sided sign-tail "
                r"probabilities use the multinomial node bootstrap. $\Delta D$/1 SD "
                r"standardizes by the full-sample dyadic standard deviation of $W_2$; "
                r"partial $R^2$ is relative to symmetric additive firm effects. These "
                r"are conditional associations, not causal estimates or tests of the "
                r"covariance envelope.}"
            ),
            r"\label{tab:w2-primary-results}",
            r"\end{table}",
        ]
    )


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
                            _escape(str(cell.get("distance", ""))),
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
            f"[{_fmt(cell.get('effect_one_sd_ci_low'), 3)}, "
            f"{_fmt(cell.get('effect_one_sd_ci_high'), 3)}]"
        )
        rows.append(
            " & ".join(
                [
                    _model_label(cell["model"]),
                    str(cell[size_field]),
                    _DISTANCE_LABELS.get(
                        str(cell["distance"]), _escape(str(cell["distance"]))
                    ),
                    _fmt(cell.get("geometry_spearman_vs_canonical_w2"), 3),
                    f"{_fmt(cell.get('effect_one_sd'), 3)} {effect_ci}",
                    _fmt(cell.get("partial_r2"), 3),
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
            f"[{_fmt(contrast.get('ci_low'), 3)}, {_fmt(contrast.get('ci_high'), 3)}]"
        )
        decision = "Yes" if contrast.get("equivalent") is True else "No"
        values = [
            label,
            _fmt(contrast.get("estimate"), 3),
            interval,
            _fmt(contrast.get("pvalue"), 3),
            _fmt(contrast.get("holm_pvalue"), 3),
        ]
        if family == "vintage":
            values.extend([_fmt(contrast.get("equivalence_bound"), 3), decision])
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
                    r"\multicolumn{6}{l}{\textit{Panel C: matched encoder-vintage "
                    r"negative control}} \\"
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
    if contrast_rows:
        confidence = cast("float", vintage_analysis["confidence_level"])
        lines.extend(
            [
                r"\medskip",
                r"\begin{tabularx}{\linewidth}{Xrrrrrr}",
                r"\toprule",
                (
                    rf"Vintage contrast ($c-r$) & Estimate & "
                    rf"{100.0 * float(confidence):.0f}\% CI & Raw $p$ & Holm $p$ & "
                    rf"Bound & Equiv. \\"
                ),
                r"\midrule",
                *contrast_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
    if base_contrast_rows:
        lines.extend(
            [
                r"\medskip",
                r"\begin{tabularx}{\linewidth}{Xrrrrr}",
                r"\toprule",
                (
                    r"Base contrast ($c-r$) & Estimate & 95\% CI & "
                    r"Raw $p$ & Holm $p$ \\"
                ),
                r"\midrule",
                *base_contrast_rows,
                r"\bottomrule",
                r"\end{tabularx}",
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
                r"The EttaX panel holds architecture, recipe, compute, and sampling "
                r"fixed while changing only the encoder's Wikipedia snapshot; its "
                r"paired contrasts reuse identical bootstrap draws and declare "
                r"equivalence only when the full interval lies inside the configured "
                r"symmetric bound. "
                r"Standardized effects use each row's dyadic "
                r"standard deviation. Raw $p$-values are two-sided add-one sign-tail "
                r"probabilities; Holm adjustment is applied separately to the seven "
                r"base and three vintage contrasts. Contrast estimates are "
                r"candidate-minus-reference differences in $\Delta D$ per one dyadic "
                r"standard deviation ($c-r$); the vintage equivalence bound is shown "
                r"in its contrast panel. $W_1$ and energy are lower-cost "
                r"descriptive "
                r"comparators; the table does not test representation superiority.}"
            ),
            r"\label{tab:w2-representation-sensitivity}",
            r"\end{table}",
        ]
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

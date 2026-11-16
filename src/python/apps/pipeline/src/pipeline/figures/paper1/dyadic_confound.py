"""Paper 1 dyadic confound exhibits: result tables and number bindings.

``render-paper1-dyadic-confound`` writes the primary-result, ladder,
classification-control, design-diagnostic, and random-exposure table fragments
from the saved ``paper1-dyadic-confound`` and ``paper1-dyadic-oos`` results.
:func:`append_dyadic_confound_records` contributes the dyadic block of
``numbers.tex`` and fixes its emission order; record order is the order of the
generated LaTeX file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from pipeline.figures._latex import escape_latex
from pipeline.figures.paper1._format import EM_DASH, fmt, fmt_int, opt_float
from pipeline.io.values import Blank, Comment, PublicationValue
from pipeline.stages.papers.paper1._dyadic.random_exposure import (
    SCENARIO_LIPSCHITZ,
    SCENARIO_SLACK,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.io.values import Record


@dataclass(frozen=True, slots=True)
class Paper1DyadicConfoundRenderPaths:
    """Saved results consumed and fragments produced by the dyadic renderer."""

    dyadic_results_file: Path
    oos_dyadic_results_file: Path
    design_diagnostics_file: Path
    output_dir: Path


def render_paper1_dyadic_confound(paths: Paper1DyadicConfoundRenderPaths) -> None:
    """Render the dyadic result fragments from the saved stage results."""
    dyadic = json.loads(paths.dyadic_results_file.read_text(encoding="utf-8"))
    oos_dyadic = json.loads(paths.oos_dyadic_results_file.read_text(encoding="utf-8"))
    design_diagnostics = pd.read_parquet(paths.design_diagnostics_file)
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "w2_primary_results.tex").write_text(
        render_w2_primary_results(dyadic, oos_dyadic) + "\n"
    )
    (output_dir / "dyadic_ladder.tex").write_text(render_dyadic_ladder(dyadic))
    (output_dir / "classification_controls.tex").write_text(
        render_classification_controls(dyadic)
    )
    (output_dir / "dyadic_design_diagnostics.tex").write_text(
        render_dyadic_design_diagnostics(design_diagnostics, dyadic)
    )
    (output_dir / "random_exposure_diagnostic.tex").write_text(
        render_random_exposure_diagnostic(dyadic)
    )


def _primary_row(
    label: str, arm: dict[str, object], n_firms: object, n_dyads: object
) -> str:
    coefficient = arm.get("coefficient", arm.get("coef"))
    effect = arm.get("effect_one_sd")
    interval = f"[{fmt(arm.get('ci_low'), 3)}, {fmt(arm.get('ci_high'), 3)}]"
    return (
        " & ".join(
            [
                label,
                str(n_firms),
                str(n_dyads),
                fmt(coefficient, 3),
                interval,
                fmt(arm.get("pvalue"), 3),
                fmt(effect, 3),
                fmt(arm.get("within_r2"), 3),
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
                r"partial $R^2$ is relative to symmetric additive firm effects.}"
            ),
            r"\label{tab:w2-primary-results}",
            r"\end{table}",
        ]
    )


_DISPERSION_SPEC_INDEX = 5


_DYADIC_ADJUSTMENT_LABELS = {
    1: "W2 distance only",
    2: "+ same-sector indicator",
    3: "pooled controls",
    4: "+ article-count controls",
    6: "W2 metric + symmetric firm effects",
}


def render_dyadic_ladder(summary: dict[str, object]) -> str:
    """Format the governed W2 specification ladder and its primary estimate."""
    ladder = cast("list[dict[str, object]]", summary.get("model_ladder", []))
    rows_by_spec: dict[int, dict[str, object]] = {}
    for row in ladder:
        spec_index = int(cast("str | bytes | int | float", row["spec_index"]))
        if spec_index in rows_by_spec:
            message = f"duplicate dyadic ladder specification {spec_index}"
            raise ValueError(message)
        rows_by_spec[spec_index] = row

    expected_specs = set(range(1, 7))
    actual_specs = set(rows_by_spec)
    if actual_specs != expected_specs:
        missing = sorted(expected_specs - actual_specs)
        extra = sorted(actual_specs - expected_specs)
        message = (
            "dyadic ladder must contain specifications 1--6; "
            f"missing={missing}, extra={extra}"
        )
        raise ValueError(message)

    panels = (
        (
            r"\multicolumn{8}{l}{\textit{Panel A: staged pooled adjustments}} \\",
            (1, 2, 3, 4, 5),
        ),
        (
            (
                r"\multicolumn{8}{l}{\textit{Panel B: canonical "
                r"endpoint-adjusted specification}} \\"
            ),
            (6,),
        ),
    )
    lines = [
        "% AUTO-GENERATED by pipeline render-paper1-dyadic-confound.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\linewidth}{r>{\raggedright\arraybackslash}Xrrrrrr}",
        r"\toprule",
        (
            r"Spec. & Adjustment & $\hat\beta_{W_2}$ & $\Delta D$/1 SD & 95\% CI & "
            r"$p$ & $R^2_{\mathrm{full}}$ & $R^2_{\mathrm{partial}}$ \\"
        ),
        r"\midrule",
    ]
    for panel_index, (heading, spec_indices) in enumerate(panels):
        if panel_index:
            lines.append(r"\midrule")
        lines.append(heading)
        for spec_index in spec_indices:
            row = rows_by_spec[spec_index]
            adjustment = (
                r"+ within-firm dispersion ($B_i+B_j$, $|B_i-B_j|$)"
                if spec_index == _DISPERSION_SPEC_INDEX
                else escape_latex(
                    _DYADIC_ADJUSTMENT_LABELS.get(spec_index, str(row["label"]))
                )
            )
            ci = (
                f"[{fmt(row.get('ci_low_w2_distance'), 3)}, "
                f"{fmt(row.get('ci_high_w2_distance'), 3)}]"
            )
            lines.append(
                " & ".join(
                    [
                        str(spec_index),
                        adjustment,
                        fmt(row.get("coef_w2_distance"), 3),
                        fmt(row.get("effect_one_sd_w2_distance"), 3),
                        ci,
                        fmt(row.get("pvalue_w2_distance"), 3),
                        fmt(row.get("overall_r2"), 3),
                        # Conditional fit exists only where a restricted
                        # fixed-effect model does; pooled rungs carry None and
                        # render as an em dash rather than inviting a
                        # pooled-versus-FE comparison.
                        fmt(row.get("within_r2"), 3),
                    ]
                )
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            (
                r"\caption{Dyadic association ladder. Every interval and two-sided "
                r"sign-tail probability uses the same deterministic multinomial "
                r"node-count array across rungs. Panel A separates the W2 "
                r"metric's staged pooled adjustments from canonical "
                r"specification 6, which absorbs symmetric additive firm effects. "
                r"Specification 5 conditions on the within-firm dispersion "
                r"components of article geometry and is a decomposition "
                r"diagnostic. "
                r"$\Delta D$/1 SD "
                r"is the fitted change in return chord distance for a one-standard-"
                r"deviation change in rooted W2 distance. "
                r"$R^2_{\mathrm{full}}$ is the full-model coefficient of "
                r"determination; the pooled and fixed-effect blocks absorb different "
                r"variation and are compared within block. "
                r"$R^2_{\mathrm{partial}} = 1 - "
                r"\mathrm{SSE}_{\mathrm{FE}+W_2}/\mathrm{SSE}_{\mathrm{FE}}$ is the "
                r"conditional fit against the restricted firm-effect model and is "
                r"undefined (\textemdash) for the pooled rungs, which have no such "
                r"restricted model.}"
            ),
            r"\label{tab:dyadic-ladder}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _validated_primary_spec(
    summary: dict[str, object],
) -> tuple[int, dict[str, object]]:
    """Validate and return the governed primary W2 specification."""
    primary_spec_index = summary["primary_spec_index"]
    if isinstance(primary_spec_index, bool) or not isinstance(primary_spec_index, int):
        message = "dyadic summary primary_spec_index must be an integer"
        raise TypeError(message)
    model_ladder = summary["model_ladder"]
    if not isinstance(model_ladder, list):
        message = "dyadic summary model_ladder must be a list"
        raise TypeError(message)
    primary = next(
        (
            row
            for row in model_ladder
            if isinstance(row, dict)
            and int(row.get("spec_index", -1)) == primary_spec_index
        ),
        None,
    )
    if primary is None:
        message = f"primary W2 specification {primary_spec_index} is absent"
        raise ValueError(message)
    if primary.get("firm_effects") is not True:
        message = "primary W2 specification must use symmetric firm effects"
        raise ValueError(message)
    if primary.get("regressors") != ["w2_distance"]:
        message = "primary W2 specification has unexpected regressors"
        raise ValueError(message)
    if summary.get("se_type") != "multinomial node bootstrap (dyad weight w_i*w_j)":
        message = "dyadic design table requires multinomial node-bootstrap inference"
        raise ValueError(message)
    if summary.get("w2_scale") != "rooted balanced quadratic Wasserstein distance":
        message = "dyadic design table requires the governed rooted W2 input"
        raise ValueError(message)
    return primary_spec_index, primary


def render_dyadic_design_diagnostics(
    frame: pd.DataFrame, summary: dict[str, object]
) -> str:
    """Appendix design contract and pre-fit audit for the dyadic ladder."""
    primary_spec_index, primary = _validated_primary_spec(summary)

    bootstrap_draws = int(
        cast("str | bytes | int | float", primary["bootstrap_requested_draws"])
    )
    rows = []
    for _, row in frame.sort_values("spec_index").iterrows():
        dropped_value = row["dropped_columns"]
        if isinstance(dropped_value, (list, tuple)):
            dropped = ", ".join(str(value) for value in dropped_value) or EM_DASH
        else:
            dropped = EM_DASH
        rows.append(
            " & ".join(
                [
                    fmt_int(row["spec_index"]),
                    f"{fmt_int(row['design_rank'])}/{fmt_int(row['design_columns'])}",
                    fmt(row["scaled_condition_number"], 2),
                    fmt(row["focal_vif"], 2),
                    fmt(row["focal_fwl_sd_ratio"], 3),
                    escape_latex(dropped),
                ]
            )
            + r" \\"
        )
    first = frame.iloc[0]
    return "\n".join(
        [
            "% AUTO-GENERATED by pipeline render-paper1-dyadic-confound.",
            r"\begin{table}[H]",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabularx}{\linewidth}{@{}lX@{}}",
            r"\toprule",
            r"\multicolumn{2}{@{}l}{\textit{Panel A: design contract}} \\",
            r"\midrule",
            (
                r"Sample timing & Pooled news: \ppvalue{sample-year-start}--"
                r"\ppvalue{sample-year-end}; complete-case returns: "
                r"\ppvalue{return-panel-n-observations} common dates, "
                r"\ppvalue{return-panel-date-start}--\ppvalue{return-panel-date-end}. "
                r"News and returns are pooled over the same window. \\"
            ),
            (
                rf"Observational units & {int(frame.iloc[0]['n_nodes'])} firms and "
                rf"{int(frame.iloc[0]['n_dyads'])} unordered dyads $i<j$; one "
                r"observation "
                r"per pair and no self-dyads. \\"
            ),
            (
                r"Metrics & $W_{ij}$ is rooted balanced quadratic Wasserstein "
                r"distance between row-normalized article clouds under Euclidean "
                r"chord ground cost; "
                r"$D_{ij}=\sqrt{2(1-\hat\rho_{ij})}$ from complete-case daily "
                r"returns. \\"
            ),
            (
                rf"Primary design & Specification {primary_spec_index}: "
                r"$D_{ij}=\mu+a_i+a_j+\beta_{W_2} W_{ij}+u_{ij}$, with "
                r"symmetric additive firm effects and one firm indicator "
                r"omitted. \\"
            ),
            (
                rf"Resampling & {bootstrap_draws} multinomial node-bootstrap draws; "
                r"firms are resampled and dyads receive endpoint-multiplicity weight "
                r"$w_iw_j$, "
                r"without artificial self-dyads. \\"
            ),
            (
                r"Estimand & Same-window contemporaneous association conditional on "
                r"realized $W$ and $D$ and additive firm margins. \\"
            ),
            r"\bottomrule",
            r"\end{tabularx}",
            r"\medskip",
            r"\begin{tabularx}{\linewidth}{rrrrrX}",
            r"\toprule",
            (
                r"\multicolumn{6}{l}{\textit{Panel B: pre-fit conditioning by "
                r"ladder rung}} \\"
            ),
            r"\midrule",
            (
                r"Spec. & Rank/cols. & Scaled $\kappa$ & VIF$_{W_2}$ & FWL SD ratio & "
                r"Dropped \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            (
                r"\caption{Dyadic design and pre-fit audit. Panel A fixes the analysis "
                r"sample, metrics, specification, resampling unit, and interpretation "
                r"boundary. Panel B reports conditioning over the complete "
                rf"graph on {int(first['n_nodes'])} firms: "
                rf"{int(first['n_dyads'])} dyads, "
                rf"degree {int(first['degree_min'])} for every firm, and "
                rf"{float(first['endpoint_overlap_share']):.3f} of unordered "
                r"dyad pairs "
                r"share an endpoint. $\kappa$ is computed after centring and unit-norm "
                r"scaling non-intercept columns. VIF$_{W_2}$ and the FWL "
                r"residual-SD ratio "
                r"measure how much W2-distance variation remains after the other "
                r"regressors and summarize numerical conditioning.}"
            ),
            r"\label{tab:dyadic-design-diagnostics}",
            r"\end{table}",
            "",
        ]
    )


def _interval(row: dict[str, object], name: str) -> str:
    """Format an observed share with its node-bootstrap percentile interval."""
    return (
        f"{fmt(row.get(f'{name}_share'), 3)} "
        f"[{fmt(row.get(f'{name}_ci_low'), 3)}, "
        f"{fmt(row.get(f'{name}_ci_high'), 3)}]"
    )


def render_random_exposure_diagnostic(summary: dict[str, object]) -> str:
    """Render the maintained-scale W2 envelope scenarios."""
    n_dyads = summary.get("n_dyads")
    if isinstance(n_dyads, bool) or not isinstance(n_dyads, int) or n_dyads < 1:
        message = "random-exposure summary must declare a positive integer n_dyads"
        raise ValueError(message)
    diagnostic = cast(
        "dict[str, object]", summary.get("random_exposure_diagnostic", {})
    )
    scenarios = cast("list[dict[str, object]]", diagnostic.get("scenarios", []))
    lines = [
        "% AUTO-GENERATED by pipeline render-paper1-dyadic-confound.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\linewidth}{rr>{\raggedleft\arraybackslash}X>{\raggedleft\arraybackslash}X>{\raggedleft\arraybackslash}X}",
        r"\toprule",
        r"$L$ & $\tau_i=\tau_j$ & Nonnegative & Unresolved & Violation \\",
        r"\midrule",
    ]
    if diagnostic.get("status") != "computed" or not scenarios:
        lines.append(r"\multicolumn{5}{c}{Unavailable: " + EM_DASH + r"} \\")
    else:
        lines.extend(
            " & ".join(
                [
                    fmt(row.get("L"), 2),
                    fmt(row.get("tau"), 2),
                    _interval(row, "coverage"),
                    _interval(row, "unresolved"),
                    _interval(row, "violation"),
                ]
            )
            + r" \\"
            for row in scenarios
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            (
                r"\caption{Fixed-W2 total-return compatibility diagnostic. "
                r"Intervals bracket $H_{ij}=\Delta_{ij}+q_{ij}$, where "
                r"$q_{ij}$ is the residual-variance sum in standardized returns "
                r"and $c^{\mathrm{actual}}_{ij}=2-2\rho_{ij}$. "
                r"Nonnegative denotes a nonnegative lower endpoint; violation "
                r"denotes a negative upper endpoint; unresolved denotes an interval "
                r"spanning zero. A violation flags plug-in inconsistency "
                r"with the scenario. "
                rf"Entries are shares of the {n_dyads:,} dyads; "
                r"brackets are percentile "
                r"intervals from the common multinomial node bootstrap. "
                r"Article distance uses 128-article measures and "
                r"unit-chord ground cost. "
                r"$L$ and $\tau$ specify transmission distortion and firm-level slack.}"
            ),
            r"\label{tab:random-exposure-w2}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_classification_controls(summary: dict[str, object]) -> str:
    """Render generated estimates with explicit nuisance baselines and timing."""
    block = cast("dict[str, object]", summary["classification_robustness"])
    labels = {
        "sector": "FE + same sector",
        "industry": "FE + same sector + same industry",
        "within-industry": "Within industry, FE",
        "etnic-roster": "ETNIC firms: FE + same sector",
        "etnic": r"\quad + ETNIC similarity",
    }
    rows = []
    for row in cast("list[dict[str, object]]", block["rows"]):
        if row["key"] == "etnic-roster":
            # The ETNIC rows are a sub-roster; the rule marks the change of sample.
            rows.append(r"\midrule")
        interval = f"[{fmt(row['ci_low'], 3)}, {fmt(row['ci_high'], 3)}]"
        rows.append(
            " & ".join(
                [
                    labels[str(row["key"])],
                    str(row["n_firms"]),
                    str(row["n_dyads"]),
                    fmt(row["coefficient"], 3),
                    interval,
                    fmt(row["effect_one_sd"], 3),
                    fmt(row["delta_correlation"], 3),
                    fmt(row["partial_r2"], 3),
                ]
            )
            + r" \\"
        )
    lines = [
        "% AUTO-GENERATED -- do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}Xrrrrrrr}",
        r"\toprule",
        (
            r"Controls/sample & Firms & Dyads & $\hat\beta_{W_2}$ & 95\% CI & "
            r"$\Delta D$/1 SD & $\Delta\rho$ & Partial $R^2$ \\"
        ),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        (
            r"\caption{Classification controls in the primary symmetric-firm-effects "
            r"design, 2018--2022 returns. FE denotes symmetric additive firm effects. "
            r"The first two rows add pair-level matching indicators to the full panel; "
            r"the third retains only pairs sharing an industry. Labels come from the "
            r"undated sample-roster snapshot, so these are retrospective controls. "
            r"Intervals are 95\% multinomial node-bootstrap percentile intervals. "
            r"Full-panel rows reuse the main schedule; the within-industry row "
            r"resamples its incident firms, retaining only within-industry edges. "
            r"$\Delta D$/1 SD uses each row's distance standard deviation; "
            r"$\Delta\rho$ translates that effect at the row's median chord distance "
            r"using $\rho=1-D^2/2$ and is a fitted contrast. Partial $R^2$ is "
            r"$1-\mathrm{SSE}_{\mathrm{full}}/\mathrm{SSE}_{\mathrm{restricted}}$, "
            r"where the restricted model removes only $W_2$, retaining all controls. "
            r"The within-industry row changes the sample and has a distinct baseline. "
            r"The last two rows restrict the panel to firms covered by the "
            r"Hoberg--Phillips ETNIC network and share one resampling schedule; "
            r"ETNIC is the mean 10-K business-description similarity over the "
            r"fiscal years of the return window.}"
        ),
        r"\label{tab:classification-controls}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def append_dyadic_confound_records(
    records: list[Record], dyadic_confound: dict[str, Any] | None
) -> None:
    """Append the saved dyadic estimates and diagnostics as publication values."""
    if not dyadic_confound:
        return
    _append_dyadic_primary_records(records, dyadic_confound)
    _append_random_exposure_records(records, dyadic_confound)
    _append_dyadic_margin_records(records, dyadic_confound)
    _append_dyadic_postfit_records(records, dyadic_confound)
    _append_fwl_records(records, dyadic_confound)
    _append_correlation_scale_records(records, dyadic_confound)
    _append_classification_records(records, dyadic_confound)


def _append_classification_records(
    records: list[Record], summary: dict[str, Any]
) -> None:
    """Bind estimates and sample sizes for the classification-adjusted FE fits."""
    block = summary.get("classification_robustness")
    if block is None:
        return
    records.append(Comment("--- Classification controls within the FE design ---"))
    for row in block["rows"]:
        prefix = f"classification-{row['key']}"
        records.extend(
            PublicationValue(f"{prefix}-{field.replace('_', '-')}", float(row[field]))
            for field in (
                "coefficient",
                "se",
                "ci_low",
                "ci_high",
                "pvalue",
                "effect_one_sd",
                "delta_correlation",
                "anchor_correlation",
                "partial_r2",
            )
        )
        records.extend(
            PublicationValue(f"{prefix}-{field.replace('_', '-')}", int(row[field]))
            for field in ("n_firms", "n_dyads", "bootstrap_valid_draws")
        )
        control = row.get("etnic_similarity")
        if control is not None:
            records.extend(
                PublicationValue(
                    f"{prefix}-control-{field.replace('_', '-')}", float(control[field])
                )
                for field in (
                    "coefficient",
                    "se",
                    "ci_low",
                    "ci_high",
                    "effect_one_sd",
                    "partial_r2",
                    "correlation_with_w2",
                )
            )
    records.append(
        PublicationValue(
            "classification-n-industries", int(block["classification"]["n_industries"])
        )
    )
    etnic = block.get("etnic")
    if etnic is not None:
        correlation = etnic["validation_matrix_correlation_w2"]
        records.extend(
            [
                PublicationValue("etnic-n-firms", int(etnic["n_firms"])),
                PublicationValue("etnic-year-start", int(min(etnic["years"]))),
                PublicationValue("etnic-year-end", int(max(etnic["years"]))),
                PublicationValue(
                    "etnic-matrix-correlation-w2", float(correlation["pearson"])
                ),
                PublicationValue(
                    "etnic-matrix-correlation-w2-spearman",
                    float(correlation["spearman"]),
                ),
                PublicationValue(
                    "etnic-matrix-correlation-w2-partial-sector",
                    float(correlation["partial_sector"]),
                ),
                PublicationValue(
                    "etnic-matrix-correlation-pvalue-max",
                    float(correlation["pvalue_max"]),
                ),
                PublicationValue(
                    "etnic-matrix-correlation-permutations",
                    int(correlation["permutations"]),
                ),
            ]
        )


def _append_dyadic_primary_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind the primary specification's coefficient, interval, and fit."""
    primary = cast("dict[str, Any]", dyadic_confound["w2_primary_regression"])
    records += [
        Blank(),
        Comment(
            "--- Dyadic confound-aware robustness check "
            "(data/papers/paper1/dyadic_confound/results.json) ---"
        ),
        PublicationValue("n-firms", int(dyadic_confound.get("n_firms", 0))),
        PublicationValue("n-pairs", int(dyadic_confound.get("n_dyads", 0))),
        PublicationValue(
            "confound-w2-coef",
            float(primary["coefficient"]),
        ),
        PublicationValue("confound-w2-se", float(primary["se"])),
        PublicationValue(
            "confound-w2-p",
            float(primary["pvalue"]),
        ),
        PublicationValue("confound-n-dyads", int(dyadic_confound.get("n_dyads", 0))),
        PublicationValue(
            "confound-w2-ci-low",
            opt_float(primary.get("ci_low")),
        ),
        PublicationValue(
            "confound-w2-ci-high",
            opt_float(primary.get("ci_high")),
        ),
        PublicationValue(
            "confound-overall-r2", opt_float(dyadic_confound.get("overall_r2"))
        ),
        PublicationValue(
            "confound-within-r2", opt_float(dyadic_confound.get("within_r2"))
        ),
        PublicationValue(
            "confound-w2-effect-one-sd",
            opt_float(primary.get("effect_one_sd")),
        ),
        PublicationValue(
            "confound-bootstrap-valid",
            int(primary.get("bootstrap_valid_draws", 0)),
        ),
    ]


def _append_random_exposure_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind the complete optional-W2 catalog, using MISSING when unavailable."""
    diagnostic = dyadic_confound.get("random_exposure_diagnostic", {}) or {}
    w2_fit = dyadic_confound.get("w2_primary_regression", {}) or {}
    scenarios: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", diagnostic.get("scenarios", []))
        if isinstance(diagnostic, dict)
        else []
    )
    scenario_by_key = {
        (float(row.get("L")), float(row.get("tau"))): row
        for row in scenarios
        if isinstance(row, dict)
        and row.get("L") is not None
        and row.get("tau") is not None
    }
    records += [
        Blank(),
        Comment(
            "--- Fixed-W2 random-exposure envelope diagnostic "
            "(data/papers/paper1/dyadic_confound/results.json) ---"
        ),
        PublicationValue(
            "confound-w2-within-r2",
            opt_float(w2_fit.get("within_r2") if isinstance(w2_fit, dict) else None),
        ),
    ]
    for lipschitz in SCENARIO_LIPSCHITZ:
        for slack in SCENARIO_SLACK:
            row = scenario_by_key.get((lipschitz, slack), {})
            key_prefix = f"confound-w2-l{lipschitz:g}-tau{slack:g}".replace(".", "p")
            records.extend(
                [
                    PublicationValue(
                        f"{key_prefix}-coverage",
                        opt_float(row.get("coverage_share")),
                    ),
                    PublicationValue(
                        f"{key_prefix}-violation",
                        opt_float(row.get("violation_share")),
                    ),
                    PublicationValue(
                        f"{key_prefix}-unresolved",
                        opt_float(row.get("unresolved_share")),
                    ),
                ]
            )


def _append_dyadic_margin_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind the firm-margin variance-share decomposition."""
    margin_diagnostics = dyadic_confound.get("firm_margin_diagnostics", {}) or {}
    if margin_diagnostics:
        records += [
            PublicationValue(
                "confound-firm-share-w2",
                opt_float(margin_diagnostics.get("w2_metric_variance_share")),
            ),
            PublicationValue(
                "confound-firm-share-return",
                opt_float(margin_diagnostics.get("return_distance_variance_share")),
            ),
            PublicationValue(
                "confound-within-correlation",
                opt_float(margin_diagnostics.get("within_residual_correlation")),
            ),
            PublicationValue(
                "confound-between-correlation",
                opt_float(margin_diagnostics.get("between_fitted_correlation")),
            ),
        ]


def _append_dyadic_postfit_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind influence and residual-scale diagnostics for the canonical rung."""
    influence = dyadic_confound.get("influence_diagnostics", {}) or {}
    residual_scale = dyadic_confound.get("residual_scale_diagnostic", {}) or {}
    if influence or residual_scale:
        records += [
            PublicationValue(
                "confound-lofo-max-se",
                opt_float(influence.get("max_abs_lofo_delta_in_bootstrap_se")),
            ),
            PublicationValue(
                "confound-lofo-sign-flips",
                int(influence.get("lofo_sign_flip_count", 0)),
            ),
            PublicationValue(
                "confound-max-leverage-dyad",
                str(influence.get("max_leverage_dyad", "unavailable")),
            ),
            PublicationValue(
                "confound-max-leverage",
                opt_float(influence.get("max_leverage")),
            ),
            PublicationValue(
                "confound-leverage-threshold",
                opt_float(influence.get("leverage_flag_threshold_2p_over_n")),
            ),
            PublicationValue(
                "confound-max-studentized-residual",
                opt_float(influence.get("most_extreme_studentized_residual")),
            ),
            PublicationValue(
                "confound-residual-scale-stat",
                opt_float(residual_scale.get("statistic")),
            ),
            PublicationValue(
                "confound-residual-scale-p",
                opt_float(residual_scale.get("permutation_pvalue")),
            ),
        ]


def _append_fwl_records(records: list[Record], dyadic_confound: dict[str, Any]) -> None:
    """Bind the FWL slope check and the formula-defined partial R-squared."""
    fwl = dyadic_confound.get("fwl_diagnostics", {}) or {}
    partial = dyadic_confound.get("partial_r2_diagnostics", {}) or {}
    if fwl:
        records += [
            Blank(),
            Comment(
                "--- True FWL display: both sides residualized on the primary "
                "point design's intercept + symmetric firm effects ---"
            ),
            PublicationValue("confound-fwl-slope", opt_float(fwl.get("slope"))),
            PublicationValue(
                "confound-fwl-slope-abs-error",
                opt_float(fwl.get("abs_slope_reproduction_error")),
            ),
            PublicationValue(
                "confound-fwl-nuisance-columns",
                int(fwl.get("nuisance_columns", 0)),
            ),
            PublicationValue(
                "confound-fwl-omitted-firm",
                str(fwl.get("omitted_firm_indicator", "unavailable")),
            ),
        ]
    if partial:
        records += [
            PublicationValue(
                "confound-partial-r2", opt_float(partial.get("partial_r2"))
            ),
            PublicationValue(
                "confound-sse-firm-effects",
                opt_float(partial.get("sse_firm_effects_only")),
            ),
            PublicationValue(
                "confound-sse-firm-effects-w2",
                opt_float(partial.get("sse_firm_effects_plus_focal")),
            ),
        ]


def _append_correlation_scale_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind return-chord quantiles and their correlation-scale translation.

    Deterministic arithmetic on the already-governed coefficient: ``rho = 1 -
    D^2/2``. Signed correlations are emitted at each anchor, before and after a
    one-standard-deviation energy increase, so no consumer has to infer a sign.
    """
    translation = dyadic_confound.get("correlation_scale_translation", {}) or {}
    if not translation:
        return
    quantiles = translation.get("return_chord_distance_quantiles", {}) or {}
    anchors = translation.get("anchors", {}) or {}
    records += [
        Blank(),
        Comment(
            "--- Return-chord quantiles and the deterministic correlation-scale "
            "re-expression rho = 1 - D^2/2 (t34.6.3.1); not a new estimand ---"
        ),
        PublicationValue(
            "chord-delta-one-sd",
            opt_float(translation.get("delta_chord_one_sd_w2")),
        ),
    ]
    for key, label in (("q25", "q25"), ("q50", "median"), ("q75", "q75")):
        anchor = anchors.get(key, {}) or {}
        records += [
            PublicationValue(f"chord-{label}", opt_float(quantiles.get(key))),
            PublicationValue(
                f"chord-rho-{label}", opt_float(anchor.get("correlation"))
            ),
            PublicationValue(
                f"chord-rho-after-{label}",
                opt_float(anchor.get("correlation_after_one_sd")),
            ),
            PublicationValue(
                f"chord-delta-rho-{label}",
                opt_float(anchor.get("delta_correlation")),
            ),
        ]

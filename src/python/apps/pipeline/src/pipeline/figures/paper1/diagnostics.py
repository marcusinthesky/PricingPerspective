"""Paper 1 referee-response diagnostic figures, drawn from saved results.

``paper1-diagnostics`` (:mod:`pipeline.stages.papers.paper1.diagnostics`)
computes the truncated-vs-full W2 agreement; this module only draws.

``matryoshka_distance_preservation.pgf`` plots truncated against full-width W2
distance per dyad at each reported width, with the rank correlation annotated.
``dyadic_postfit_diagnostics.pgf`` reports the FWL, functional-form, and
leave-one-firm-out diagnostics of the primary W2 specification from the
``paper1-dyadic-confound`` results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.figures._common import (
    _C,
    _save,
    categorical_axis,
    figsize,
    fit_to_width,
    thin_for_vector,
)
from pipeline.figures.paper1._format import opt_float
from pipeline.figures.paper1._typography import MIN_PT, paper1_style
from pipeline.io.values import Blank, Comment, PublicationValue

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.artist import Artist

    from pipeline.io.values import Record

# Equal-count bins used purely as descriptive overlays; no fit uses them.
_FWL_BIN_COUNT = 20


@dataclass(frozen=True, slots=True)
class Paper1DiagnosticsRenderPaths:
    """Saved results consumed and figures produced by the diagnostic renderer."""

    results_file: Path
    preservation_file: Path
    dyadic_results_file: Path
    w2_primary_fit_file: Path
    lofo_file: Path
    output_dir: Path


def _fig_distance_preservation(
    out: Path,
    frames: dict[int, pd.DataFrame],
    stats: dict[int, dict[str, float]],
) -> None:
    """Plot truncated versus full-width distance, one panel per width.

    Each panel carries the identity line and rank correlation against full
    width. The figure diagnoses geometric distortion, not return association.
    """
    paper1_style()
    widths = sorted(frames)
    fig, axes = plt.subplots(
        1, len(widths), figsize=figsize(1.0, 3.2 / 6.5), sharey=True
    )
    if len(widths) == 1:  # pragma: no cover - defensive
        axes = [axes]

    lo = min(float(f.to_numpy().min()) for f in frames.values())
    hi = max(float(f.to_numpy().max()) for f in frames.values())
    pad = 0.02 * (hi - lo)

    for ax, k in zip(axes, widths, strict=True):
        merged = frames[k]
        # Quadratic in the roster and three panels of it, which exhausted TeX's
        # main memory at 100 firms. Thinned deterministically; the panel is read
        # as a mass about the diagonal, not as individual dyads.
        (shown_full, shown_truncated), total_dyads = thin_for_vector(
            merged["full"].to_numpy(dtype=float),
            merged["truncated"].to_numpy(dtype=float),
        )
        ax.scatter(
            shown_full,
            shown_truncated,
            s=3,
            alpha=0.25,
            edgecolors="none",
            color=_C["blue"],
            label=(
                f"{len(shown_full):,} of {total_dyads:,} dyads"
                if len(shown_full) < total_dyads
                else f"{total_dyads:,} dyads"
            ),
        )
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="0.35",
            linestyle="dashed",
            linewidth=0.7,
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        title = rf"$k = {k}$   ($\rho_s = {stats[k]['spearman']:.2f}$)"
        ax.set_title(title, fontsize=MIN_PT)
        ax.set_xlabel("Full-width W2 distance", fontsize=MIN_PT)
        ax.tick_params(axis="both", which="major", labelsize=MIN_PT)

    axes[0].set_ylabel("Truncated W2 distance", fontsize=MIN_PT)
    fig.tight_layout()
    fit_to_width(fig, 1.0)
    _save(fig, out)


def primary_slope(dyadic_summary: dict[str, object]) -> float:
    """Resolve the primary W2 coefficient the FWL display must draw.

    Both the appendix multi-panel diagnostic and the main-text standalone panel
    draw the *same* reported coefficient, so they must not resolve it by two
    code paths that can drift apart.
    """
    fwl = cast("dict[str, object]", dyadic_summary.get("fwl_diagnostics", {}))
    primary = cast("dict[str, object]", dyadic_summary.get("w2_primary_regression", {}))
    return float(
        cast(
            "float | int | str",
            fwl.get("slope", primary.get("coefficient", np.nan)),
        )
    )


def _take_legend_entries(
    ax: plt.Axes, labels: tuple[str, ...]
) -> tuple[list[Artist], list[str]]:
    """Remove ``ax``'s legend and return its handles under short labels.

    Used by the three-panel context, where a 9 pt legend inside a third of the
    text block covers the data; the caller gathers the entries into one legend
    beneath the panels instead.
    """
    handles, _existing = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    return list(handles[: len(labels)]), list(labels[: len(handles)])


def panel_fwl(
    ax: plt.Axes,
    primary_fit: pd.DataFrame,
    slope: float,
    n_bins: int = _FWL_BIN_COUNT,
    title: str = "Panel A: Frisch-Waugh-Lovell, both sides residualized",
) -> None:
    """Plot the true Frisch--Waugh--Lovell scatter of the primary coefficient.

    Both axes are residuals on the *same* nuisance projector (intercept plus
    symmetric additive firm effects) taken from the primary point design, so the
    plotted line's slope is the reported coefficient exactly.  The line is fitted
    on all dyads; the equal-count bin means are a descriptive overlay and are not
    what the slope is computed from.

    ``title`` is a parameter only so the standalone main-text figure can drop the
    "Panel A" prefix it does not have; the residualization statement stays in
    every caller's title because the display is only honest with it.
    """
    w2 = primary_fit["w2_fwl_residual"].to_numpy(dtype=float)
    outcome = primary_fit["outcome_fwl_residual"].to_numpy(dtype=float)
    # Dyad-scale and therefore quadratic in the roster; thinned so the cloud
    # stays within TeX's vector budget. The binned means drawn over it use the
    # FULL sample and are what the reader actually reads off this panel.
    (shown_w2, shown_outcome), total_dyads = thin_for_vector(w2, outcome)
    ax.scatter(
        shown_w2,
        shown_outcome,
        s=5,
        color=_C["blue"],
        alpha=0.28,
        linewidths=0.0,
        label=(
            f"{len(shown_w2):,} of {total_dyads:,} dyads"
            if len(shown_w2) < total_dyads
            else f"{total_dyads:,} dyads"
        ),
    )
    binned = (
        pd.DataFrame({"w2": w2, "outcome": outcome})
        .assign(_bin=pd.qcut(w2, n_bins, duplicates="drop"))
        .groupby("_bin", observed=True)
        .agg(w2=("w2", "mean"), outcome=("outcome", "mean"))
    )
    ax.scatter(
        binned["w2"],
        binned["outcome"],
        s=26,
        color="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
        label=f"{len(binned)} equal-count bin means (overlay)",
    )
    grid = np.linspace(float(w2.min()), float(w2.max()), 2)
    ax.plot(
        grid,
        slope * grid,
        color=_C["vermilion"],
        linewidth=1.4,
        zorder=4,
        label=rf"all-dyad fit, slope $\hat\beta_{{W_2}}$ = {slope:.3f}",
    )
    ax.axhline(0.0, color="0.35", linewidth=0.7)
    ax.axvline(0.0, color="0.35", linewidth=0.7)
    ax.set_xlabel("W2 distance, residualized on intercept + firm effects")
    # Wording is a contract (tests/papers/paper1/test_paper1_fwl_panel.py): the label
    # must say the residualization is the SAME as the x-axis, not merely that one
    # happened. Shrunk rather than shortened, which is what fixed the print-size clip.
    ax.set_ylabel("Return chord distance, same residualization", fontsize=MIN_PT)
    ax.set_title(title)
    ax.legend(fontsize=MIN_PT, loc="best")


def fig_fwl_primary(out: Path, primary_fit: pd.DataFrame, slope: float) -> None:
    """Emit the standalone main-text Frisch--Waugh--Lovell display.

    This is an *addition*, not a move: ``dyadic_postfit_diagnostics.pgf`` keeps
    its three-panel appendix form, and this figure re-renders the same panel
    from the same frame and the same reported slope so the two can never
    disagree. A separate asset avoids cropping one panel out of a three-panel
    diagnostic with brittle manuscript layout commands.

    The height preserves the established 3.25-inch panel geometry.
    """
    paper1_style()
    fig, ax = plt.subplots(figsize=figsize(0.92, 3.25 / 5.6))
    panel_fwl(
        ax,
        primary_fit,
        slope,
        title="Frisch-Waugh-Lovell: both sides residualized on firm effects",
    )
    fig.tight_layout()
    fit_to_width(fig, 0.92)
    _save(fig, out)


def _panel_quadratic(ax: plt.Axes, primary_fit: pd.DataFrame) -> None:
    """Plot the outcome-only-residualized quadratic sensitivity, not the FWL display.

    This is the pre-existing functional-form check: raw W2 distance on the
    horizontal axis against the firm-effect-residualized outcome.  It is a
    different object from Panel A -- the regressor here is *not* residualized --
    and is labelled so that it cannot be read as the FWL visualization of the
    primary coefficient.
    """
    bins = pd.qcut(primary_fit["w2_distance"], _FWL_BIN_COUNT, duplicates="drop")
    binned = (
        primary_fit.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(
            w2_distance=("w2_distance", "mean"),
            outcome_fe_residual=("outcome_fe_residual", "mean"),
            quadratic_fe_partial_fit=("quadratic_fe_partial_fit", "mean"),
        )
    )
    ax.scatter(
        binned["w2_distance"],
        binned["outcome_fe_residual"],
        s=24,
        color="white",
        edgecolors="black",
        linewidths=0.8,
        label=f"{len(binned)} equal-count bin means",
        zorder=3,
    )
    ax.plot(
        binned["w2_distance"],
        binned["quadratic_fe_partial_fit"],
        color="0.25",
        linestyle="--",
        linewidth=1.3,
        label="symmetric-FE quadratic sensitivity",
    )
    ax.axhline(0.0, color="0.35", linewidth=0.7)
    ax.set_xlabel("W2 distance (raw, not residualized)")
    ax.set_ylabel("Return chord distance, residualized")
    ax.set_title("Panel B: quadratic sensitivity, outcome only")
    ax.legend(fontsize=MIN_PT, loc="best")


def _fig_dyadic_postfit(
    out: Path,
    primary_fit: pd.DataFrame,
    lofo: pd.DataFrame,
    dyadic_summary: dict[str, object],
) -> None:
    """FWL, functional-form, and firm-influence diagnostics for specification 6."""
    paper1_style()
    # `constrained_layout` rather than `tight_layout`: the latter lays panels out
    # without reserving room for a two-line `suptitle`, and at three panels on a
    # 10-inch canvas the long panel titles and the middle panel's y-label ran into
    # their neighbours in the released PDF.  Panel titles are also shortened to
    # what the axes can hold at this width -- the detail they carried is in the
    # caption, which is where a reader looks for it.
    fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, 0.5), layout="constrained")
    slope = primary_slope(dyadic_summary)
    panel_fwl(
        axes[0],
        primary_fit,
        slope,
        title="Panel A: FWL",
    )
    axes[0].set_title("Panel A: FWL", fontsize=MIN_PT)
    # The standalone panels' legend labels ("20 equal-count bin means (overlay)",
    # "all-dyad fit, slope ...") are far wider than a third of the text block, and
    # even the short forms cover the data at 9 pt. Panels A and B therefore share
    # one legend row beneath the figure; the bin-mean marker is identical in both,
    # so it is listed once. The long forms stay on the single-panel figures.
    handles, labels = _take_legend_entries(
        axes[0], ("dyads", "bin means", rf"fit: $\hat\beta_{{W_2}}={slope:.3f}$")
    )
    axes[0].set_xlabel("W2 distance (residualized)", fontsize=MIN_PT)
    axes[0].set_ylabel("Return chord distance", fontsize=MIN_PT)
    _panel_quadratic(axes[1], primary_fit)
    axes[1].set_title("Panel B: quadratic", fontsize=MIN_PT)
    quadratic_handles, _labels = _take_legend_entries(
        axes[1], ("bin means", "FE quadratic")
    )
    handles.append(quadratic_handles[1])
    labels.append("FE quadratic (Panel B)")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(handles))
    axes[1].set_xlabel("W2 distance (raw)", fontsize=MIN_PT)
    axes[1].set_ylabel("Return chord distance", fontsize=MIN_PT)

    influential = (
        lofo.assign(_absolute_shift=np.abs(lofo["delta_in_bootstrap_se"]))
        .nlargest(10, "_absolute_shift")
        .sort_values("delta_in_bootstrap_se")
    )
    positions = np.arange(len(influential))
    colors = np.where(influential["sign_flip"], "0.15", "0.55")
    axes[2].barh(
        positions,
        influential["delta_in_bootstrap_se"],
        color=colors,
        alpha=0.85,
    )
    axes[2].axvline(0.0, color="0.35", linewidth=0.7)
    axes[2].axvline(1.0, color="0.6", linestyle="dashed", linewidth=0.6)
    axes[2].axvline(-1.0, color="0.6", linestyle="dashed", linewidth=0.6)
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels(influential["left_out_ticker"], fontsize=MIN_PT)
    categorical_axis(axes[2], "y")
    axes[2].set_xlabel(r"$\Delta\hat\beta_{W_2}$ / bootstrap SE", fontsize=MIN_PT)
    axes[2].set_title("Panel C: node-level shifts", fontsize=MIN_PT)
    quadratic = cast(
        "dict[str, object]", dyadic_summary.get("functional_form_sensitivity", {})
    )
    quadratic_coefficient = float(
        cast(
            "float | int | str",
            quadratic.get("quadratic_coefficient", np.nan),
        )
    )
    fig.suptitle(
        "Primary W2 specification (6)\n"
        r"descriptive quadratic coefficient $\hat\beta_{W_2^2}$ = "
        f"{quadratic_coefficient:.3f}",
        fontsize=9,
    )
    # No `tight_layout` here: the figure is laid out by `constrained_layout`,
    # which already reserves space for the two-line suptitle, and calling both
    # makes matplotlib discard the constrained solver.
    fit_to_width(fig, 1.0)
    _save(fig, out)


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def render_paper1_diagnostics(paths: Paper1DiagnosticsRenderPaths) -> None:
    """Draw both diagnostic figures from the saved diagnostic and dyadic results."""
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_json(paths.results_file)
    preservation = cast("dict[str, dict[str, float]]", results["distance_preservation"])
    stats = {int(width): values for width, values in preservation.items()}
    pairs = pd.read_parquet(paths.preservation_file)
    frames = {
        int(cast("int", width)): group[["truncated", "full"]].reset_index(drop=True)
        for width, group in pairs.groupby("width", sort=True)
    }
    _fig_distance_preservation(
        output_dir / "matryoshka_distance_preservation.pgf", frames, stats
    )

    dyadic_results = _load_json(paths.dyadic_results_file)
    primary_fit = pd.read_parquet(paths.w2_primary_fit_file)
    lofo = pd.read_parquet(paths.lofo_file)
    _fig_dyadic_postfit(
        output_dir / "dyadic_postfit_diagnostics.pgf",
        primary_fit,
        lofo,
        dyadic_results,
    )


def append_diagnostics_records(
    records: list[Record], diagnostics: dict[str, Any] | None
) -> None:
    """Bind the truncated-vs-full W2 rank agreement per Matryoshka width."""
    if diagnostics:
        preservation = diagnostics.get("distance_preservation", {}) or {}
        if preservation:
            records += [
                Blank(),
                Comment(
                    "--- Truncated-vs-full W2 agreement per "
                    "Matryoshka width (data/papers/paper1/diagnostics/results.json) ---"
                ),
            ]
            for width in sorted(preservation, key=int):
                stats = preservation[width] or {}
                records += [
                    PublicationValue(
                        f"matryoshka-preserve-{width}-spearman",
                        opt_float(stats.get("spearman")),
                    ),
                ]

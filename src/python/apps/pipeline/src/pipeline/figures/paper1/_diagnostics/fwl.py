"""Frisch-Waugh-Lovell residual panels for the primary dyadic specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.figures._common import (
    _C,
    _save,
    _style,
    figsize,
    fit_to_width,
    thin_for_vector,
)
from pipeline.figures.paper1._diagnostics.contracts import _FWL_BIN_COUNT

if TYPE_CHECKING:
    from pathlib import Path


def _primary_slope(dyadic_summary: dict[str, object]) -> float:
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


def _relabel_legend(ax: plt.Axes, labels: tuple[str, ...]) -> None:
    """Redraw ``ax``'s legend with short labels, keeping the existing handles.

    Used by the three-panel context, where the standalone panels' descriptive
    labels are wider than a third of the text block.
    """
    handles, _existing = ax.get_legend_handles_labels()
    ax.legend(
        handles[: len(labels)],
        labels[: len(handles)],
        fontsize=5.5,
        loc="lower right",
    )


def _panel_fwl(
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
    ax.set_ylabel("Return chord distance, same residualization", fontsize=8)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")


def _fig_fwl_primary(out: Path, primary_fit: pd.DataFrame, slope: float) -> None:
    """Emit the standalone main-text Frisch--Waugh--Lovell display.

    This is an *addition*, not a move: ``dyadic_postfit_diagnostics.pgf`` keeps
    its three-panel appendix form, and this figure re-renders the same panel
    from the same frame and the same reported slope so the two can never
    disagree. A separate asset avoids cropping one panel out of a three-panel
    diagnostic with brittle manuscript layout commands.

    The height preserves the established 3.25-inch panel geometry.
    """
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.92, 3.25 / 5.6))
    _panel_fwl(
        ax,
        primary_fit,
        slope,
        title="Frisch-Waugh-Lovell: both sides residualized on firm effects",
    )
    fig.tight_layout()
    fit_to_width(fig, 0.92)
    _save(fig, out)

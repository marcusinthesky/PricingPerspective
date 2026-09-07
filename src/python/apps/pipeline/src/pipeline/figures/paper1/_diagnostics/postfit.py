"""Post-fit dyadic diagnostics: quadratic-term panel plus the four-panel figure."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.figures._common import (
    _save,
    _style,
    categorical_axis,
    figsize,
    fit_to_width,
)
from pipeline.figures.paper1._diagnostics.contracts import _FWL_BIN_COUNT
from pipeline.figures.paper1._diagnostics.fwl import (
    _panel_fwl,
    _primary_slope,
    _relabel_legend,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    ax.legend(fontsize=7, loc="best")


def _fig_dyadic_postfit(
    out: Path,
    primary_fit: pd.DataFrame,
    lofo: pd.DataFrame,
    dyadic_summary: dict[str, object],
) -> None:
    """FWL, functional-form, and firm-influence diagnostics for specification 6."""
    _style()
    # `constrained_layout` rather than `tight_layout`: the latter lays panels out
    # without reserving room for a two-line `suptitle`, and at three panels on a
    # 10-inch canvas the long panel titles and the middle panel's y-label ran into
    # their neighbours in the released PDF.  Panel titles are also shortened to
    # what the axes can hold at this width -- the detail they carried is in the
    # caption, which is where a reader looks for it.
    fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, 0.42), layout="constrained")
    slope = _primary_slope(dyadic_summary)
    _panel_fwl(
        axes[0],
        primary_fit,
        slope,
        title="Panel A: FWL",
    )
    axes[0].set_title("Panel A: FWL", fontsize=8)
    # The standalone panels' legend labels ("20 equal-count bin means (overlay)",
    # "all-dyad fit, slope ...") are far wider than a third of the text block, so at
    # printed size they ran out of the axes and over the neighbouring y-labels. The
    # long forms stay on the single-panel figures, which have room for them.
    _relabel_legend(
        axes[0], ("dyads", "bin means", rf"fit: $\hat\beta_{{W_2}}={slope:.3f}$")
    )
    axes[0].set_xlabel("W2 distance (residualized)", fontsize=8)
    axes[0].set_ylabel("Return chord distance", fontsize=8)
    _panel_quadratic(axes[1], primary_fit)
    axes[1].set_title("Panel B: quadratic", fontsize=8)
    _relabel_legend(axes[1], ("bin means", "FE quadratic"))
    axes[1].set_xlabel("W2 distance (raw)", fontsize=8)
    axes[1].set_ylabel("Return chord distance", fontsize=8)

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
    axes[2].set_yticklabels(influential["left_out_ticker"], fontsize=7)
    categorical_axis(axes[2], "y")
    axes[2].set_xlabel(r"$\Delta\hat\beta_{W_2}$ / bootstrap SE", fontsize=8)
    axes[2].set_title("Panel C: node-level shifts", fontsize=8)
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

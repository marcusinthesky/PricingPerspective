"""One composite recovery exhibit, drawn only from a durable campaign summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.figures._common import _C, Grid, categorical_axis, figsize, figure

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.figures.paper6.information_recovery.contracts import Campaign


def render(campaign: Campaign, output: Path) -> None:
    """Separate finite-article error from the exact reporting-law floor."""
    with figure(output, figsize(0.95, 0.55), Grid(1, 2), frac=0.95) as (
        fig,
        axes,
    ):
        left, right = axes
        for regime, label, color, marker in (
            (0, "Common reporting", "blue", "o"),
            (1, "Changing reporting", "orange", "s"),
        ):
            cells = campaign.recovery[regime]
            floor = 0.0 if regime == 0 else campaign.reporting_floor[3]
            values = [cell.normalized_rmse for cell in cells]
            errors = [
                None if cell.rmse_mcse is None else 2 * cell.rmse_mcse for cell in cells
            ]
            left.errorbar(
                np.arange(3),
                values,
                yerr=errors,
                color=_C[color],
                marker=marker,
                markersize=3.5,
                capsize=2,
                linewidth=1,
                label=label,
            )
            left.plot([3], [floor], color=_C[color], marker=marker, markersize=3.5)
        left.axhline(0.25, color=_C["gray"], linestyle=":", linewidth=0.8)
        left.set_yscale("symlog", linthresh=0.25)
        left.set_ylim(bottom=0)
        left.set_yticks([0, 0.25, 1, 4, 16], ["0", "0.25", "1", "4", "16"])
        left.set_xticks(np.arange(4), ["1", "4", "16", r"$\infty$"])
        categorical_axis(left)
        left.set_xlabel("Article count multiplier")
        left.set_ylabel("Normalized recovery RMSE")
        left.set_title("(a) Information ladder")
        left.legend(loc="upper right", fontsize=7.5)
        amplitudes = np.asarray(campaign.reporting_amplitudes) * 1000
        for values, label, color, style in (
            (campaign.reporting_floor, "Operating + reporting", "orange", "-"),
            (campaign.reporting_only_floor, "Reporting only", "purple", "--"),
        ):
            right.plot(
                amplitudes,
                values,
                color=_C[color],
                marker="o",
                markersize=3,
                linestyle=style,
                linewidth=1,
                label=label,
            )
        right.set_xlabel(r"Log-intensity amplitude ($10^{-3}$)")
        right.set_ylabel("Normalized population distortion")
        right.set_title("(b) Infinite information")
        right.set_ylim(bottom=0)
        right.legend(loc="upper left", fontsize=7.5)
        fig.tight_layout(w_pad=1)

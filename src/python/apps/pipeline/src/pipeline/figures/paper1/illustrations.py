"""Paper 1 analytic schematic illustrations (non-data-driven)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pipeline.figures._common import _save, _style, figsize, fit_to_width

logger = logging.getLogger(__name__)


def _fig_continuous_bound_construction(out: Path) -> None:
    """Draw the connected three-step construction of the covariance bound."""
    _style()
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.6 / 7.2), facecolor="white")
    ax.set_axis_off()
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 5.2)

    # Box text is sized in points but the boxes are sized in data units, so
    # authoring at the printed width (5.906in, down from 7.2in) grows every label
    # ~22% relative to the box that has to hold it. Widths are therefore set from
    # measured text extents (scratch `measure.py` in t39) rather than kept
    # uniform: A's three-line header and C's second display formula are the two
    # binding constraints, and both now clear their box by >=6%.
    centers = (2.125, 7.25, 12.575)
    widths = (4.15, 4.60, 4.65)
    panel_y, panel_h = 0.48, 4.18
    for center, width in zip(centers, widths, strict=True):
        ax.add_patch(
            FancyBboxPatch(
                (center - width / 2, panel_y),
                width,
                panel_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="white",
                edgecolor="black",
                linewidth=0.7,
            )
        )

    arrow_style = {
        "arrowstyle": "-|>",
        "mutation_scale": 10,
        "linewidth": 0.9,
        "color": "black",
    }
    for left, left_width, right, right_width in zip(
        centers[:-1], widths[:-1], centers[1:], widths[1:], strict=True
    ):
        ax.add_patch(
            FancyArrowPatch(
                (left + left_width / 2 + 0.08, 2.57),
                (right - right_width / 2 - 0.08, 2.57),
                **arrow_style,
            )
        )

    x = np.linspace(-2.4, 2.4, 160)
    curve_i = np.exp(-0.5 * ((x + 0.62) / 0.67) ** 2)
    curve_j = 0.83 * np.exp(-0.5 * ((x - 0.54) / 0.92) ** 2)
    x_plot = centers[0] + 0.68 * x
    y_base = 1.62
    ax.plot(x_plot, y_base + 0.95 * curve_i, color="black", linewidth=1.0)
    ax.plot(
        x_plot,
        y_base + 0.95 * curve_j,
        color="black",
        linewidth=1.0,
        linestyle="--",
    )
    ax.plot(
        [centers[0] - 1.7, centers[0] + 1.7],
        [y_base, y_base],
        color="black",
        linewidth=0.5,
    )
    ax.text(
        centers[0],
        4.02,
        "A  Distribution-valued\ncharacteristics",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="semibold",
        linespacing=0.95,
    )
    ax.text(
        centers[0],
        3.27,
        r"$C_i,C_j\in\mathcal{P}_2(\Omega)$",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    ax.text(
        centers[0],
        1.06,
        r"compare with $W_2(C_i,C_j)$",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    ax.text(centers[0] - 1.5, 2.73, r"$C_i$", fontsize=8.0, ha="left")
    ax.text(centers[0] + 1.34, 2.48, r"$C_j$", fontsize=8.0, ha="right")

    ax.text(
        centers[1],
        4.25,
        "B  Stochastic transmission\nkernel",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="semibold",
    )
    ax.text(
        centers[1],
        3.36,
        r"$X_i\sim C_i,\quad B_i=T(X_i,U_i)$",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    ax.text(
        centers[1],
        2.43,
        r"$P_i=C_iK,\quad K(x,\cdot)=\mathcal{L}(T(x,U))$",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    ax.text(
        centers[1],
        1.79,
        r"$W_{2,\Gamma}(P_i,P_j)$",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    ax.text(
        centers[1],
        1.06,
        r"$\leq L\,W_2(C_i,C_j)+\tau_i+\tau_j$"
        "\n"
        "common kernel and scales\nare maintained, not estimated",
        ha="center",
        va="center",
        fontsize=8.0,
    )

    ax.text(
        centers[2],
        4.25,
        "C  Hilbert return and\ncovariance envelope",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="semibold",
    )
    ax.text(
        centers[2],
        3.42,
        r"$\widetilde r_t^i=\langle B_i,F_t\rangle_{\mathcal H}+e_t^i$"
        "\n"
        r"$Z_i=\Gamma^{1/2}B_i$",
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=1.25,
    )
    ax.text(centers[2], 2.62, r"$\Downarrow$", ha="center", va="center", fontsize=9)
    ax.text(
        centers[2],
        1.91,
        r"$\bar\kappa_{ij}=\frac{1}{2}(v_i+v_j"
        r"-W_{2,\Gamma}^2(P_i,P_j))$"
        "\n"
        r"$\kappa_\pi=\bar\kappa_{ij}-\frac{1}{2}\Delta_\pi$",
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=1.25,
    )
    ax.text(
        centers[2],
        1.06,
        r"$\Delta_\pi\geq0$: sharp $W_2$ envelope;"
        "\n"
        "energy is a comparator",
        ha="center",
        va="center",
        fontsize=8.0,
    )

    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.025)
    fit_to_width(fig, 1.0)
    _save(fig, out)


def render_paper1_illustrations(
    output_dir: Path = Path("src/latex/projects/01_continuous_bounds/src/images"),
) -> None:
    """Render Paper 1's analytic bound-construction schematic."""
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig_continuous_bound_construction(output_dir / "continuous_bound_construction.pgf")

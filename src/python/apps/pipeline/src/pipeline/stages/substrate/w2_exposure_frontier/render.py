"""Shared W2 exposure-frontier figure."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.figures._common import _C, _save, _style, figsize, fit_to_width

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_FRONTIER_SAMPLES = 100


@dataclass(frozen=True, slots=True)
class FrontierSeries:
    """The four arrays the frontier scatter needs, decoupled from the estimator.

    Taking these rather than the stage's ``FrontierResult`` keeps the dependency
    one-way (stages import figures, never the reverse).
    """

    weights: np.ndarray
    ratios: np.ndarray
    ell_hat: float
    L_hat: float


def render_frontier(series: FrontierSeries, out: Path) -> None:
    r"""Scatter :math:`\|\beta_i-\beta_j\|` against :math:`D_{ij}` with the two rays."""
    _style()
    x = series.weights
    y = series.ratios * series.weights
    breach = (series.ratios < series.ell_hat) | (series.ratios > series.L_hat)

    fig, ax = plt.subplots(figsize=figsize(0.8, 4.0 / 6.0))
    ax.scatter(x[~breach], y[~breach], s=6, alpha=0.5, color=_C["blue"], label="pairs")
    ax.scatter(x[breach], y[breach], s=8, alpha=0.7, color=_C["vermilion"])
    xline = np.linspace(0.0, float(x.max()), _FRONTIER_SAMPLES)
    (lower,) = ax.plot(xline, series.ell_hat * xline, "-", color=_C["black"], lw=1.0)
    (upper,) = ax.plot(xline, series.L_hat * xline, "--", color=_C["black"], lw=1.0)
    ax.set_xlabel(r"$D_{ij}$ (Hilbertian text distance)")
    ax.set_ylabel(r"$\|\beta_i-\beta_j\|$ (returns-based loading)")
    ax.set_title("H1 identification frontier (returns-based)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=_C["blue"], marker="o", linestyle="None"),
            plt.Line2D([0], [0], color=_C["vermilion"], marker="o", linestyle="None"),
            lower,
            upper,
        ],
        labels=["pairs", "breach", r"$\hat{\ell}\cdot D$", r"$\hat{L}\cdot D$"],
    )
    fig.tight_layout()
    fit_to_width(fig, 0.8)
    _save(fig, out)

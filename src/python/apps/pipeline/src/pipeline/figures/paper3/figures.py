"""Paper 3 publication figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.figures._common import (
    _C,
    _save,
    _style,
    categorical_axis,
    figsize,
    fit_to_width,
)

if TYPE_CHECKING:
    from pathlib import Path


def _fig_conceptual_pipeline(out: Path) -> None:
    """Render the governed observed-to-decision object flow."""
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.24))
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    labels = (
        r"$C_i$",
        r"$W_2(C_i,C_j)$",
        r"$\ell_{ij}$",
        r"$\mathcal{C}(q)$",
        "variance bound",
        r"$x^\star$",
    )
    colours = (
        _C["blue"],
        _C["blue"],
        _C["orange"],
        _C["orange"],
        _C["green"],
        _C["purple"],
    )
    positions = (0.35, 1.4, 2.45, 3.45, 4.55, 5.7)
    half_widths = (0.18, 0.45, 0.22, 0.30, 0.62, 0.20)
    for idx, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        ax.text(
            positions[idx],
            0.0,
            label,
            ha="center",
            va="center",
            color=colour,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": colour,
            },
        )
        if idx:
            maintained = label == r"$\ell_{ij}$"
            ax.annotate(
                "",
                xy=(positions[idx] - half_widths[idx] - 0.05, 0.0),
                xytext=(positions[idx - 1] + half_widths[idx - 1] + 0.05, 0.0),
                arrowprops={
                    "arrowstyle": "->",
                    "color": _C["gray"],
                    "linestyle": "--" if maintained else "-",
                },
            )
    fig.tight_layout()
    fit_to_width(fig, 0.95)
    _save(fig, out)


def _fig_news_variance_benchmark(out: Path, summary: dict[str, object]) -> None:
    """Plot standardized in-sample variance relative to the sample GMV."""
    benchmark = summary.get("news_only_benchmark")
    if not isinstance(benchmark, dict):
        message = "news-variance figure input has no news-only benchmark"
        raise TypeError(message)
    if benchmark.get("status") != "full_sample_descriptive":
        message = "news-variance figure input has an unsupported status"
        raise ValueError(message)
    portfolios = benchmark.get("portfolios")
    if not isinstance(portfolios, dict):
        message = "news-variance figure input has no portfolio diagnostics"
        raise TypeError(message)
    news = portfolios.get("news_only")
    equal = portfolios.get("equal_weight")
    sample_gmv = portfolios.get("sample_gmv")
    if not all(isinstance(item, dict) for item in (news, equal, sample_gmv)):
        message = "news-variance figure input has incomplete portfolio diagnostics"
        raise TypeError(message)
    news = cast("dict[str, object]", news)
    equal = cast("dict[str, object]", equal)
    sample_gmv = cast("dict[str, object]", sample_gmv)

    def summary_float(portfolio: dict[str, object], key: str) -> float:
        value = portfolio.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            message = f"news-variance figure field {key!r} must be numeric"
            raise TypeError(message)
        return float(value)

    labels = ("News-only", "Equal risk weights", "Sample GMV")
    values = np.asarray(
        [
            100.0 * summary_float(news, "gmv_relative_variance"),
            100.0 * summary_float(equal, "gmv_relative_variance"),
            100.0 * summary_float(sample_gmv, "gmv_relative_variance"),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        message = "news-variance figure values must be finite and positive"
        raise ValueError(message)
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.72, 3.6 / 5.4))
    x = np.arange(len(labels))
    ax.bar(
        x,
        values,
        color=(_C["blue"], _C["orange"], _C["green"]),
        edgecolor=_C["black"],
        linewidth=0.35,
    )
    ax.axhline(100.0, color=_C["gray"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    categorical_axis(ax)
    ax.set_ylabel("Standardized variance\n(relative to sample GMV)")
    ax.set_ylim(0.0, max(110.0, float(values.max()) * 1.18))
    ax.grid(visible=True, axis="y", linestyle=":", alpha=0.45)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fit_to_width(fig, 0.72)
    _save(fig, out)

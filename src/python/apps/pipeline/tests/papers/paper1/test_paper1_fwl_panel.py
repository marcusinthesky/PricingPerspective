"""Display-level contract for the primary Frisch--Waugh--Lovell panel.

The stage-side residuals were already correct before this task; the defect was
in the renderer, which binned the raw metric against an
outcome-only-residualized series while the panel was labelled a within-margin
diagnostic. These tests therefore assert on what the axes actually receive, not
on what the stage computes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# No `mpl.use("Agg")` here. This module used to force Agg, which meant it wrote a
# real `.pdf` while production wrote `.pgf` under the global PGF backend that
# `pipeline.figures._common` installs on import — so the one test that exercised
# `_save` end to end exercised a backend nothing ships. That is how the `.pdf`
# path stayed green in CI while every `.pdf` figure stage was broken. Importing
# `_common` (below, via the `diagnostics` facade) selects PGF, as production does.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pipeline.figures._common import _save
from pipeline.figures.paper1.diagnostics import (
    fig_fwl_primary,
    panel_fwl,
    primary_slope,
)

if TYPE_CHECKING:
    from pathlib import Path

SLOPE = 1.634146818997988
BIN_COUNT = 20


def _display_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    w2_residual = rng.normal(size=n)
    return pd.DataFrame(
        {
            # A deliberately different series: if the panel ever plots the raw
            # metric again, the residual assertions below fail.
            "w2_distance": rng.uniform(0.1, 0.9, n),
            "w2_fwl_residual": w2_residual,
            "outcome_fwl_residual": SLOPE * w2_residual + rng.normal(scale=0.2, size=n),
            "fwl_fitted": SLOPE * w2_residual,
        }
    )


def test_fwl_panel_plots_both_residualized_series_and_the_reported_slope() -> None:
    """Plot residual-vs-residual and draw a line whose slope is the estimate."""
    frame = _display_frame()
    fig, ax = plt.subplots()
    try:
        panel_fwl(ax, frame, SLOPE)
        scatter = np.asarray(ax.collections[0].get_offsets())
        # The fitted line is the first Line2D drawn; the zero rules follow it.
        line = ax.lines[0]
        line_x = np.asarray(line.get_xdata(), dtype=float)
        line_y = np.asarray(line.get_ydata(), dtype=float)
    finally:
        plt.close(fig)

    assert scatter[:, 0] == pytest.approx(frame["w2_fwl_residual"].to_numpy())
    assert scatter[:, 1] == pytest.approx(frame["outcome_fwl_residual"].to_numpy())
    drawn_slope = (line_y[-1] - line_y[0]) / (line_x[-1] - line_x[0])
    assert drawn_slope == pytest.approx(SLOPE, abs=1e-10)
    # The line is the all-dyad fit, so it spans the full residual support
    # rather than the range of any binned summary.
    assert float(line_x[0]) == pytest.approx(float(frame["w2_fwl_residual"].min()))
    assert float(line_x[-1]) == pytest.approx(float(frame["w2_fwl_residual"].max()))


def test_fwl_panel_bin_means_are_a_labelled_overlay_only() -> None:
    """Keep equal-count bins descriptive: they never define the fitted line."""
    frame = _display_frame()
    fig, ax = plt.subplots()
    try:
        panel_fwl(ax, frame, SLOPE)
        bin_offsets = np.asarray(ax.collections[1].get_offsets())
        legend = ax.get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        xlabel, ylabel, title = ax.get_xlabel(), ax.get_ylabel(), ax.get_title()
    finally:
        plt.close(fig)

    assert len(bin_offsets) <= BIN_COUNT
    assert any("overlay" in label for label in labels)
    assert any("all-dyad fit" in label for label in labels)
    # Axes must say what was residualized, on both sides.
    assert "residualized on intercept + firm effects" in xlabel
    assert "same residualization" in ylabel
    assert "both sides residualized" in title


def test_standalone_main_text_figure_is_one_faithful_fwl_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emit a single-panel asset that plots exactly what Panel A plots.

    The main-text promotion must not become a second, differently-computed
    display: same residualized series on both axes, same all-dyad fitted slope,
    bins still descriptive.
    """
    frame = _display_frame()
    captured: list[plt.Figure] = []

    def _capture(fig: plt.Figure, path: Path) -> None:
        captured.append(fig)
        _save(fig, path)

    # Patch the module defining the figure so capture uses the production saver.
    monkeypatch.setattr("pipeline.figures.paper1.diagnostics._save", _capture)
    out = tmp_path / "dyadic_fwl_primary.pgf"
    fig_fwl_primary(out, frame, SLOPE)

    assert out.exists()
    assert out.read_text().lstrip().startswith("%% Creator: Matplotlib")
    fig = captured[0]
    assert len(fig.axes) == 1, "the main-text asset is a standalone single panel"
    ax = fig.axes[0]
    scatter = np.asarray(ax.collections[0].get_offsets())
    assert scatter[:, 0] == pytest.approx(frame["w2_fwl_residual"].to_numpy())
    assert scatter[:, 1] == pytest.approx(frame["outcome_fwl_residual"].to_numpy())
    line = ax.lines[0]
    line_x = np.asarray(line.get_xdata(), dtype=float)
    line_y = np.asarray(line.get_ydata(), dtype=float)
    drawn_slope = (line_y[-1] - line_y[0]) / (line_x[-1] - line_x[0])
    assert drawn_slope == pytest.approx(SLOPE, abs=1e-10)
    assert float(line_x[0]) == pytest.approx(float(frame["w2_fwl_residual"].min()))
    assert float(line_x[-1]) == pytest.approx(float(frame["w2_fwl_residual"].max()))
    assert len(np.asarray(ax.collections[1].get_offsets())) <= BIN_COUNT
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("overlay" in label for label in labels)
    assert any("all-dyad fit" in label for label in labels)
    # The standalone title drops the "Panel A" prefix but must keep saying what
    # was residualized, on both sides.
    title = ax.get_title()
    assert "Panel A" not in title
    assert "both sides residualized" in title
    assert "residualized on intercept + firm effects" in ax.get_xlabel()
    assert "same residualization" in ax.get_ylabel()


def test_appendix_panel_title_is_unchanged_by_the_promotion() -> None:
    """Keep the three-panel appendix figure intact: this was an addition."""
    frame = _display_frame()
    fig, ax = plt.subplots()
    try:
        panel_fwl(ax, frame, SLOPE)
        title = ax.get_title()
    finally:
        plt.close(fig)
    assert title == "Panel A: Frisch-Waugh-Lovell, both sides residualized"


def test_both_displays_resolve_the_same_reported_slope() -> None:
    """One helper resolves the coefficient, so the two figures cannot drift."""
    assert primary_slope(
        {
            "fwl_diagnostics": {"slope": SLOPE},
            "w2_primary_regression": {"coefficient": 99.0},
        }
    ) == pytest.approx(SLOPE)
    # Fallback when the stage summary predates the fwl_diagnostics block.
    assert primary_slope(
        {"w2_primary_regression": {"coefficient": SLOPE}}
    ) == pytest.approx(SLOPE)

"""Saved Paper 1 metric MDS, principal coordinates and spectral decay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.transforms import Bbox

from pipeline.figures._common import (
    _C,
    _save,
    figsize,
    fit_to_width,
    thin_for_vector,
)
from pipeline.figures._latex import escape_latex
from pipeline.figures.paper1._typography import MIN_PT, paper1_style

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Annotation


# Three columns keep the 9 pt sector legend narrower than the axes it sits under;
# four made the legend the widest artist, so width fitting shrank the plot.
_LEGEND_MAX_COLUMNS = 3


@dataclass(frozen=True, slots=True)
class Paper1PcoaPaths:
    """Saved analysis inputs and the manuscript's image/table destinations."""

    analysis_dir: Path
    output_dir: Path
    spectrum_table: Path


@dataclass(frozen=True, slots=True)
class ScatterText:
    """Axis and legend text for an embedding scatter."""

    xlabel: str
    ylabel: str
    title: str
    legend_title: str


def _label_palette(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    """Deterministic colour map over unique labels (Okabe-Ito, then tab20)."""
    uniq = sorted(set(labels))
    base = list(_C.values())
    if len(uniq) <= len(base):
        cols = [mpl.colors.to_rgb(base[i]) for i in range(len(uniq))]
    else:
        cmap = plt.get_cmap("tab20")
        cols = [cmap(i % 20)[:3] for i in range(len(uniq))]
    return dict(zip(uniq, cols, strict=True))


def _fig_embedding_scatter(
    out: Path,
    coords: np.ndarray,
    tickers: list[str],
    labels: list[str],
    text: ScatterText,
) -> None:
    """Annotated 2D scatter coloured by a categorical label (sector/industry)."""
    paper1_style()
    palette = _label_palette(labels)
    # Printed at 0.96 of the text width, the fraction the manuscript includes it
    # at: every point carries a 9 pt ticker, which needs close to the full width.
    # Marker area, edge width, and the annotation offset are all point-valued, so
    # they shrink with the canvas rather than growing chunkier.
    fig, ax = plt.subplots(figsize=figsize(0.96, 6.0 / 7.0))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=[palette[lbl] for lbl in labels],
        s=64,
        alpha=0.85,
        edgecolors=_C["black"],
        linewidths=0.35,
        zorder=2,
    )
    ax.set_xlabel(text.xlabel)
    ax.set_ylabel(text.ylabel)
    ax.set_title(text.title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(visible=True, linestyle=":", alpha=0.4)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[lbl],
            markeredgecolor=_C["black"],
            markeredgewidth=0.35,
            markersize=6,
            label=lbl,
        )
        for lbl in sorted(palette)
    ]
    # Below the axes, not `loc="best"`: every point is annotated, so an in-axes
    # legend sits on top of ticker labels wherever it lands.
    ax.legend(
        handles=handles,
        title=text.legend_title,
        fontsize=MIN_PT,
        title_fontsize=MIN_PT,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(_LEGEND_MAX_COLUMNS, len(handles)),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.tight_layout()
    # Placement measures display space, so it runs on the final canvas: fitting
    # afterwards would rescale marker spacing under fixed-size text and reopen
    # the collisions it resolved. The labels stay inside the axes frame, so the
    # second fit is a check that the saved width did not move.
    fit_to_width(fig, 0.96)
    _place_point_labels(fig, ax, coords, tickers)
    fit_to_width(fig, 0.96)
    _save(fig, out)


# Candidate label positions around a marker, in offset points: the conventional
# upper-right first, then the other diagonals, then the four sides. Each clears
# the marker's own bounding box, so a label never sits on its own point.
_LABEL_OFFSETS: tuple[tuple[float, float, str, str], ...] = (
    (4.0, 4.0, "left", "bottom"),
    (4.0, -4.0, "left", "top"),
    (-4.0, 4.0, "right", "bottom"),
    (-4.0, -4.0, "right", "top"),
    (0.0, 5.0, "center", "bottom"),
    (0.0, -5.0, "center", "top"),
    (5.0, 0.0, "left", "center"),
    (-5.0, 0.0, "right", "center"),
)
# Half the diameter of an `s=64` scatter marker, in points.
_MARKER_RADIUS_PT = 4.0
# Clear space kept around each label, in points, so adjacent labels never abut.
_LABEL_PAD_PT = 1.5


def _overlap_area(a: Bbox, b: Bbox) -> float:
    """Area of the intersection of two display-space boxes."""
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(width, 0.0) * max(height, 0.0)


def _place_point_labels(
    fig: Figure, ax: Axes, coords: np.ndarray, labels: list[str]
) -> None:
    """Annotate every point, choosing each label's side to avoid collisions.

    At the 9 pt floor a fixed upper-right offset runs neighbouring tickers into
    one another. Labels are placed greedily, most crowded point first, at the
    first candidate side that overlaps no placed label, other marker, or the
    axes frame; failing that, at the side with the least overlap. Placement is
    deterministic, so a re-render reproduces the figure exactly. Agg measures
    text in a wider face than the document's Times, so the clearances are
    conservative in print.
    """
    points = ax.transData.transform(coords)
    radius = _MARKER_RADIUS_PT * fig.dpi / 72.0
    pad = _LABEL_PAD_PT * fig.dpi / 72.0
    markers = [
        Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
        for x, y in points
    ]
    frame = ax.get_window_extent()
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    crowding = (distances < 12.0 * radius).sum(axis=1)
    placed: list[Bbox] = []
    for i in sorted(range(len(labels)), key=lambda k: (-crowding[k], k)):
        best: tuple[float, Annotation, Bbox] | None = None
        for dx, dy, ha, va in _LABEL_OFFSETS:
            label = ax.annotate(
                labels[i],
                (coords[i, 0], coords[i, 1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=MIN_PT,
                alpha=0.9,
                ha=ha,
                va=va,
                zorder=3,
            )
            box = label.get_window_extent().padded(pad)
            cost = sum(_overlap_area(box, other) for other in placed)
            cost += sum(_overlap_area(box, marker) for marker in markers)
            cost += box.width * box.height - _overlap_area(box, frame)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, label, box)
            else:
                label.remove()
            if cost <= 0.0:
                break
        if best is not None:
            placed.append(best[2])


def _fig_projection_shepard(
    out: Path,
    raw_distances: np.ndarray,
    projected: np.ndarray,
    spearman: float | None,
    method: str,
) -> None:
    """Original versus saved projected distances for one two-axis view."""
    paper1_style()
    fig, ax = plt.subplots(figsize=figsize(0.62, 3.6 / 4.4))
    # One mark per dyad, so quadratic in the roster: 1,326 at 52 firms, 4,950 at
    # 100, which overran TeX's main memory as vector paths. A Shepard cloud is
    # read as a band about the identity rather than point by point, so a fixed
    # deterministic thinning preserves what the panel shows.
    (shown_raw, shown_projected), total_dyads = thin_for_vector(
        raw_distances, projected
    )
    ax.scatter(
        shown_raw,
        shown_projected,
        s=3.5,
        alpha=0.25,
        edgecolors="none",
        color=_C["blue"],
        label=(
            f"{len(shown_raw):,} of {total_dyads:,} dyads"
            if len(shown_raw) < total_dyads
            else f"{total_dyads:,} dyads"
        ),
    )
    ax.set_xlabel(r"Original exact $W_2$ distance")
    ax.set_ylabel(f"Distance in 2D {method} projection")
    statistic = "undefined" if spearman is None else f"{spearman:.3f}"
    ax.set_title(rf"{method} Shepard diagnostic ($\rho_S={statistic}$)")
    fig.tight_layout()
    fit_to_width(fig, 0.62)
    _save(fig, out)


_SCREE_COMPONENTS = 20
_LINE_STYLES = ("-", "--", ":", "-.")


def _check_spectrum(
    spectrum: pd.DataFrame, cells: list[dict[str, Any]], n: int
) -> None:
    expected = {cell["cell_id"] for cell in cells} | {"return_pca"}
    if len(expected) != len(cells) + 1 or set(spectrum["cell_id"]) != expected:
        message = (
            "PCoA spectra must contain every unique comparison cell and return PCA"
        )
        raise ValueError(message)
    for _, rows in spectrum.groupby("cell_id", sort=False):
        if rows["component"].sort_values().tolist() != list(range(1, n + 1)):
            message = "PCoA spectra must retain each component exactly once"
            raise ValueError(message)


def _tuple_label(cell: dict[str, Any]) -> str:
    return f"{cell['label']} / {int(cell['effective_dimension']):,}"


def _fig_scree(out: Path, cells: list[dict[str, Any]], spectrum: pd.DataFrame) -> None:
    """Overlay saved component/cumulative shares with one common PCA reference."""
    paper1_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize(1.0, 1.05))
    colours = [value for key, value in _C.items() if key != "black"]
    providers = list(dict.fromkeys(cell["provider_id"] for cell in cells))
    for cell in cells:
        provider = providers.index(cell["provider_id"])
        widths = sorted(
            {
                int(other["effective_dimension"])
                for other in cells
                if other["provider_id"] == cell["provider_id"]
            },
            reverse=True,
        )
        style = widths.index(int(cell["effective_dimension"])) + provider // len(
            colours
        )
        rows = spectrum.loc[spectrum["cell_id"] == cell["cell_id"]].sort_values(
            "component"
        )
        for ax, field, limit in (
            (axes[0], "positive_inertia_share", _SCREE_COMPONENTS),
            (axes[1], "cumulative_positive_inertia_share", len(rows)),
        ):
            shown = rows.iloc[:limit]
            ax.plot(
                shown["component"],
                shown[field],
                color=colours[provider % len(colours)],
                linestyle=_LINE_STYLES[style % len(_LINE_STYLES)],
                linewidth=1.2,
                label=_tuple_label(cell),
            )
    reference = spectrum.loc[spectrum["cell_id"] == "return_pca"].sort_values(
        "component"
    )
    for ax, field, limit in (
        (axes[0], "positive_inertia_share", _SCREE_COMPONENTS),
        (axes[1], "cumulative_positive_inertia_share", len(reference)),
    ):
        shown = reference.iloc[:limit]
        ax.plot(
            shown["component"],
            shown[field],
            color=_C["black"],
            linestyle="--",
            linewidth=1.8,
            label="Return PCA",
        )
        ax.set_xlim(1, max(2, int(shown["component"].max())))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(visible=True, axis="y", linestyle=":", alpha=0.3)
    peak = max(float(np.max(line.get_ydata())) for line in axes[0].lines)
    axes[0].set_ylim(0, 1.05 * peak if peak > 0 else 1)
    axes[1].set_ylim(0, 1)
    axes[0].set(xlabel="Leading component", ylabel="Component share")
    axes[1].set(xlabel="Number of leading components", ylabel="Cumulative share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=MIN_PT
    )
    # The shared legend belongs outside both data panels; retain the 9 pt floor.
    legend_rows = (len(labels) + 1) // 2
    bottom = 0.04 + legend_rows * MIN_PT / 72.0 * 1.6 / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fit_to_width(fig, 1.0)
    _save(fig, out)


def _spectrum_table(cells: list[dict[str, Any]]) -> str:
    """Keep geometry concentration distinct from the return-PCA comparator."""
    lines = [
        "% AUTO-GENERATED by render-paper1-pcoa; edit the producer, not this file.",
        r"\begin{table}[H]",
        r"\centering\small",
        (
            r"\caption{Spectral concentration of the Paper 1 news geometries. "
            r"The first two axes are a share of total positive inertia; negative "
            r"inertia is a share of absolute spectral mass.}"
        ),
        r"\label{tab:pcoa-spectrum}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model / vintage & Width & First two axes (\%) & Negative mass (\%) \\",
        r"\midrule",
    ]
    lines.extend(
        (
            f"{escape_latex(str(cell['label']))} & "
            f"{int(cell['effective_dimension']):,} & "
            f"{100 * float(cell['two_axis_share']):.1f} & "
            f"{100 * float(cell['negative_inertia_fraction']):.1f} " + r"\\"
        )
        for cell in cells
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_paper1_pcoa(paths: Paper1PcoaPaths) -> None:
    """Project the saved analysis into figures and a complete comparison table."""
    summary = json.loads(
        (paths.analysis_dir / "results.json").read_text(encoding="utf-8")
    )
    cells = cast("list[dict[str, Any]]", summary["cells"])
    spectrum = pd.read_parquet(paths.analysis_dir / "spectrum.parquet")
    _check_spectrum(spectrum, cells, int(summary["n_tickers"]))
    canonical = next(
        cell for cell in cells if cell["cell_id"] == summary["canonical_cell_id"]
    )
    tickers = [str(ticker) for ticker in summary["tickers"]]
    mds = summary["metric_mds"]
    mds_coordinates = pd.read_parquet(
        paths.analysis_dir / "metric_mds_coords_2d.parquet"
    ).set_index("ticker", verify_integrity=True)
    if mds["cell_id"] != canonical["cell_id"] or set(mds_coordinates.index) != set(
        tickers
    ):
        message = "metric MDS must retain the canonical cell and complete ticker cohort"
        raise ValueError(message)
    mds_coordinates = mds_coordinates.loc[tickers]
    mds_neighbours = pd.read_parquet(
        paths.analysis_dir / "metric_mds_neighbours.parquet"
    )
    mds_dyads = mds_neighbours.loc[
        mds_neighbours["anchor"] < mds_neighbours["neighbour"]
    ]
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _fig_embedding_scatter(
        paths.output_dir / "metric_mds_sector_plot.pgf",
        mds_coordinates[["mds_1", "mds_2"]].to_numpy(dtype=np.float64),
        tickers,
        mds_coordinates["sector"].astype(str).tolist(),
        ScatterText(
            xlabel="Metric MDS axis 1",
            ylabel="Metric MDS axis 2",
            title=r"Metric MDS of pairwise exact $W_2$ distances",
            legend_title="Sector",
        ),
    )
    _fig_projection_shepard(
        paths.output_dir / "metric_mds_shepard_diagnostic.pgf",
        mds_dyads["raw_distance"].to_numpy(dtype=np.float64),
        mds_dyads["projected_distance"].to_numpy(dtype=np.float64),
        float(mds["shepard_spearman"]) if mds["shepard_spearman"] is not None else None,
        "metric MDS",
    )
    _fig_scree(
        paths.output_dir / "pcoa_scree.pgf",
        [cell for cell in cells if cell["family"] == "representation"],
        spectrum,
    )
    _fig_scree(
        paths.output_dir / "pcoa_vintage_scree.pgf",
        [cell for cell in cells if cell["family"] == "vintage"],
        spectrum,
    )
    paths.spectrum_table.parent.mkdir(parents=True, exist_ok=True)
    paths.spectrum_table.write_text(_spectrum_table(cells), encoding="utf-8")

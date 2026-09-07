"""Paper 3 portfolio-anatomy figures and generated holdings table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pipeline.figures._common import (
    _C,
    _TINT,
    Grid,
    categorical_axis,
    figsize,
    figure,
)
from pipeline.figures.paper3.manifests import write_paper3_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

ANATOMY_EXHIBIT_NAMES = (
    "portfolio_anatomy.pgf",
    "portfolio_evolution.pgf",
)
_ANATOMY_FRACTION = 0.95
_TOP_PAIRS = 12
_TOP_HOLDINGS = 8
_DECLARED_CAP_PCT = 12.5
_TICKER_LABEL_STRIDE = 2
# Okabe-Ito, in full. Eight is the palette's ceiling and a real accessibility
# limit rather than an arbitrary list length: there is no ninth hue that stays
# distinguishable under deuteranopia and protanopia, so a larger sector
# partition must reuse one rather than invent one. `_sector_colours` cycles.
_SECTOR_COLOUR_NAMES = (
    "blue",
    "orange",
    "green",
    "vermilion",
    "purple",
    "sky",
    "gray",
    "black",
)


def _sector_order(firms: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    order = {sector: index for index, sector in enumerate(sectors["sector"])}
    ordered = firms.assign(
        _sector_order=firms["sector"].map(order),
    ).sort_values(
        ["_sector_order", "weight", "ticker"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    return ordered.drop(columns="_sector_order")


def _sector_colours(sectors: pd.DataFrame) -> dict[str, str]:
    """Assign a palette colour per sector, cycling once the palette is spent.

    The 100-firm universe partitions into nine sectors against an eight-colour
    palette, so one colour repeats. That is safe HERE and would not be in
    general: sectors are drawn as contiguous, individually labelled spans
    (:func:`_group_spans`), so the label carries the identity and the colour is
    a grouping cue. Sectors are ordered by portfolio weight, so a repeat falls
    between the largest and the smallest span rather than between neighbours.
    """
    labels = sectors["sector"].astype(str).tolist()
    palette = _SECTOR_COLOUR_NAMES
    return {
        label: _C[palette[index % len(palette)]] for index, label in enumerate(labels)
    }


def _group_spans(ordered: pd.DataFrame) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    for sector, group in ordered.groupby("sector", sort=False):
        stop = start + len(group)
        spans.append((str(sector), start, stop))
        start = stop
    return spans


def _short_sector_label(sector: str) -> str:
    return {
        "Communication Services": "Comm.",
        "Consumer Cyclical": "Cons. cyc.",
        "Consumer Defensive": "Cons. def.",
        "Financial Services": "Financial",
        "Healthcare": "Health",
        "Industrials": "Industr.",
        "Technology": "Tech.",
    }.get(sector, sector)


def _draw_sector_boundaries(ax: Axes, spans: list[tuple[str, int, int]]) -> None:
    for index, (sector, start, stop) in enumerate(spans):
        if start:
            ax.axvline(start - 0.5, color=_C["gray"], linewidth=0.35, alpha=0.55)
        ax.text(
            (start + stop - 1) / 2,
            1.01 + 0.05 * (index % 2),
            _short_sector_label(sector),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.0,
            rotation=35,
        )


def _ticker_label_positions(count: int) -> np.ndarray:
    """Return alternating ticker positions while retaining every plotted value."""
    if count < 1:
        message = "ticker label count must be positive"
        raise ValueError(message)
    return np.arange(count)[::_TICKER_LABEL_STRIDE]


def _fig_portfolio_anatomy(
    path: Path, firms: pd.DataFrame, pairs: pd.DataFrame, sectors: pd.DataFrame
) -> None:
    """Show the selected holdings, additive pair credits, and sector weights."""
    ordered = _sector_order(firms, sectors)
    colours = _sector_colours(sectors)
    spans = _group_spans(ordered)
    grid = Grid(nrows=3, ncols=1, height_ratios=(1.45, 1.0, 0.9))
    with figure(
        path,
        figsize(_ANATOMY_FRACTION, 1.16),
        grid,
        frac=_ANATOMY_FRACTION,
    ) as (fig, axes):
        fig.subplots_adjust(hspace=0.78)
        ax_weights, ax_pairs, ax_sectors = axes
        x = np.arange(len(ordered))
        ax_weights.bar(
            x,
            100.0 * ordered["weight"].to_numpy(),
            color=[colours[str(sector)] for sector in ordered["sector"]],
            width=0.82,
        )
        ax_weights.axhline(
            100.0 / len(ordered),
            color=_C["black"],
            linestyle="--",
            linewidth=0.8,
        )
        ax_weights.axhline(
            _DECLARED_CAP_PCT,
            color=_C["vermilion"],
            linestyle=":",
            linewidth=0.9,
        )
        label_positions = _ticker_label_positions(len(ordered))
        ax_weights.set_xticks(
            label_positions,
            ordered.iloc[label_positions]["ticker"],
            rotation=90,
            fontsize=5.2,
        )
        ax_weights.set_ylabel(r"Portfolio weight (\%)")
        ax_weights.set_ylim(0.0, _DECLARED_CAP_PCT * 1.08)
        ax_weights.set_title(
            "(a) Firm weights, grouped by sector", loc="left", pad=22.0
        )
        categorical_axis(ax_weights)
        _draw_sector_boundaries(ax_weights, spans)
        ax_weights.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=_C["black"],
                    linestyle="--",
                    linewidth=0.8,
                    label="Equal risk weight",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_C["vermilion"],
                    linestyle=":",
                    linewidth=0.9,
                    label=r"12.5\% reference-law cap",
                ),
            ],
            loc="upper right",
            fontsize=6.3,
        )

        top_pairs = pairs.head(_TOP_PAIRS).iloc[::-1]
        pair_labels = (
            top_pairs["ticker_i"].astype(str) + "--" + top_pairs["ticker_j"].astype(str)
        )
        pair_colours = np.where(
            top_pairs["same_sector"].to_numpy(), _C["orange"], _C["blue"]
        )
        ax_pairs.barh(
            np.arange(len(top_pairs)),
            100.0 * top_pairs["certificate_share"].to_numpy(),
            color=pair_colours,
        )
        ax_pairs.set_yticks(np.arange(len(top_pairs)), pair_labels, fontsize=6.0)
        ax_pairs.set_xlabel(r"Share of total certificate credit (\%)")
        ax_pairs.set_title("(b) Largest unordered pair contributions", loc="left")
        categorical_axis(ax_pairs, "y")
        ax_pairs.legend(
            handles=[
                Patch(facecolor=_C["blue"], label="Cross-sector pair"),
                Patch(facecolor=_C["orange"], label="Within-sector pair"),
            ],
            loc="lower right",
            fontsize=6.3,
        )

        sector_display = sectors.iloc[::-1]
        y = np.arange(len(sector_display))
        height = 0.36
        ax_sectors.barh(
            y + height / 2,
            100.0 * sector_display["portfolio_weight"].to_numpy(),
            height=height,
            color=_C["blue"],
            label="News-only",
        )
        ax_sectors.barh(
            y - height / 2,
            100.0 * sector_display["equal_weight_share"].to_numpy(),
            height=height,
            color=_C["orange"],
            label="Equal risk weights",
        )
        ax_sectors.set_yticks(y, sector_display["sector"], fontsize=6.2)
        ax_sectors.set_xlabel(r"Portfolio weight (\%)")
        ax_sectors.set_title("(c) Sector weights", loc="left")
        categorical_axis(ax_sectors, "y")
        ax_sectors.legend(loc="lower right", fontsize=6.3)
        fig.align_ylabels(axes)


def _metric_table(ax: Axes, summary: pd.DataFrame) -> None:
    years = [str(int(year)) for year in summary["year"]]
    turnover = [
        "--" if pd.isna(value) else f"{100.0 * float(value):.1f}"
        for value in summary["one_way_turnover"]
    ]
    maximum = [
        f"{100.0 * float(weight):.1f} {ticker}"
        for weight, ticker in zip(
            summary["maximum_weight"],
            summary["maximum_weight_ticker"],
            strict=True,
        )
    ]
    cell_text = [
        [f"{float(value):.2f}" for value in summary["effective_number_assets"]],
        maximum,
        turnover,
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=(
            "Effective names",
            r"Largest weight (\%; ticker)",
            r"Turnover (\%)",
        ),
        colLabels=years,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(value=False)
    table.set_fontsize(6.4)
    table.scale(1.0, 1.15)
    for cell in table.get_celld().values():
        cell.set_edgecolor(_C["gray"])
        cell.set_linewidth(0.35)
    ax.axis("off")
    ax.set_title("(b) Allocation diagnostics by cutoff", loc="left")


def _weight_scale(ax: Axes, cmap: LinearSegmentedColormap) -> None:
    """Draw a vector colour scale without a raster PGF sidecar."""
    steps = 20
    x_start = 1.18
    y_start = 0.08
    width = 0.025
    height = 0.84
    for index in range(steps):
        ax.add_patch(
            Rectangle(
                (x_start, y_start + index * height / steps),
                width,
                height / steps,
                transform=ax.transAxes,
                facecolor=cmap(index / (steps - 1)),
                edgecolor="none",
                clip_on=False,
            )
        )
    for share in (0.0, 0.5, 1.0):
        ax.text(
            x_start + width + 0.012,
            y_start + share * height,
            f"{share * _DECLARED_CAP_PCT:g}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            clip_on=False,
        )
    ax.text(
        x_start + width + 0.10,
        y_start + height / 2,
        r"Portfolio weight (\%)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.2,
        clip_on=False,
    )


def _fig_portfolio_evolution(
    path: Path,
    firms: pd.DataFrame,
    sectors: pd.DataFrame,
    vintage_weights: pd.DataFrame,
    vintage_summary: pd.DataFrame,
) -> None:
    """Show the allocation induced by each expanding information cutoff."""
    ordered = _sector_order(firms, sectors)
    spans = _group_spans(ordered)
    years = [int(year) for year in vintage_summary["year"]]
    pivot = vintage_weights.pivot_table(
        index="ticker", columns="year", values="weight", aggfunc="mean"
    )
    values = 100.0 * pivot.loc[ordered["ticker"], years].to_numpy()
    cmap = LinearSegmentedColormap.from_list(
        "paper3_weight", (_TINT["neutral"], _TINT["blue"], _C["blue"])
    )
    grid = Grid(nrows=2, ncols=1, height_ratios=(5.4, 1.0))
    with figure(
        path,
        figsize(_ANATOMY_FRACTION, 1.42),
        grid,
        frac=_ANATOMY_FRACTION,
    ) as (_fig, axes):
        ax_heatmap, ax_metrics = axes
        ax_heatmap.pcolormesh(
            np.arange(len(years) + 1),
            np.arange(len(ordered) + 1),
            values,
            cmap=cmap,
            vmin=0.0,
            vmax=_DECLARED_CAP_PCT,
            shading="flat",
        )
        ax_heatmap.set_xticks(np.arange(len(years)) + 0.5, years)
        ax_heatmap.xaxis.tick_top()
        label_positions = _ticker_label_positions(len(ordered))
        ax_heatmap.set_yticks(
            label_positions + 0.5,
            ordered.iloc[label_positions]["ticker"],
            fontsize=5.3,
        )
        ax_heatmap.invert_yaxis()
        ax_heatmap.set_title(
            "(a) Firm weights under expanding information cutoffs", loc="left"
        )
        categorical_axis(ax_heatmap, "both")
        for sector, start, stop in spans:
            if start:
                ax_heatmap.axhline(start, color=_C["gray"], linewidth=0.5)
            ax_heatmap.text(
                len(years) + 0.08,
                (start + stop) / 2,
                _short_sector_label(sector),
                ha="left",
                va="center",
                fontsize=5.2,
                clip_on=False,
            )
        _weight_scale(ax_heatmap, cmap)
        _metric_table(ax_metrics, vintage_summary)


def _escape_tex(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
    }
    return "".join(replacements.get(character, character) for character in text)


def _render_holdings_table(path: Path, firms: pd.DataFrame) -> None:
    top = firms.sort_values(["weight", "ticker"], ascending=[False, True]).head(
        _TOP_HOLDINGS
    )
    rows = [
        " & ".join(
            [
                _escape_tex(ticker),
                _escape_tex(name),
                _escape_tex(sector),
                f"{100.0 * float(weight):.2f}",
                _escape_tex(partner),
                f"{100.0 * float(pair_share):.2f}",
            ]
        )
        + r" \\"
        for ticker, name, sector, weight, partner, pair_share in zip(
            top["ticker"].astype(str).tolist(),
            top["name"].astype(str).tolist(),
            top["sector"].astype(str).tolist(),
            top["weight"].to_numpy(dtype=np.float64),
            top["strongest_partner"].astype(str).tolist(),
            top["strongest_pair_share"].to_numpy(dtype=np.float64),
            strict=True,
        )
    ]
    fragment = "\n".join(
        [
            "% AUTO-GENERATED by pipeline render-paper3-portfolio-anatomy",
            "% -- do not edit by hand.",
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabularx}{\linewidth}{lXlrrr}",
            r"\toprule",
            (
                r"Ticker & Firm & Sector & Weight (\%) & Partner & "
                r"Pair credit (\%)" + r" \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            (
                r"\caption{Largest positions in the canonical news-only "
                r"allocation. Partner identifies the holding that forms the largest "
                r"unordered additive certificate term with the row firm; pair credit "
                r"reports that term as a percentage of total certificate credit.}"
            ),
            r"\label{tab:p3-top-holdings}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def render_paper3_portfolio_anatomy(
    anatomy_dir: Path,
    output_dir: Path,
    table_path: Path,
    manifest_path: Path,
) -> None:
    """Render the two anatomy figures, holdings table, and exhibit manifest."""
    firms = pd.read_parquet(anatomy_dir / "firm_anatomy.parquet")
    pairs = pd.read_parquet(anatomy_dir / "pair_contributions.parquet")
    sectors = pd.read_parquet(anatomy_dir / "sector_summary.parquet")
    vintage_weights = pd.read_parquet(anatomy_dir / "vintage_weights.parquet")
    vintage_summary = pd.read_parquet(anatomy_dir / "vintage_summary.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig_portfolio_anatomy(output_dir / ANATOMY_EXHIBIT_NAMES[0], firms, pairs, sectors)
    _fig_portfolio_evolution(
        output_dir / ANATOMY_EXHIBIT_NAMES[1],
        firms,
        sectors,
        vintage_weights,
        vintage_summary,
    )
    _render_holdings_table(table_path, firms)
    outputs = [output_dir / name for name in ANATOMY_EXHIBIT_NAMES] + [table_path]
    upstreams = tuple(
        anatomy_dir / name
        for name in (
            "firm_anatomy.parquet",
            "pair_contributions.parquet",
            "sector_summary.parquet",
            "vintage_weights.parquet",
            "vintage_summary.parquet",
            "summary.yaml",
        )
    )
    write_paper3_manifest(
        manifest_path,
        stage_key="p3_portfolio_anatomy_exhibits",
        phase="exhibit",
        outputs=outputs,
        upstreams=upstreams,
    )


__all__ = ["ANATOMY_EXHIBIT_NAMES", "render_paper3_portfolio_anatomy"]

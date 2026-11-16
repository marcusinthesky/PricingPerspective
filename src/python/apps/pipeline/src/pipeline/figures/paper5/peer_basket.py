"""Named peer-basket exhibits for one Paper 5 target.

``peer_basket`` shows the five largest target-anchored W2 barycentric weights
with the classification of each peer and one bar carrying the residual mass.
``peer_definitions`` sets the same target's basket under the three scored peer
definitions side by side.  A peer that appears in two or more baskets keeps one
colour and hatch across both figures; every other named peer is neutral, and the
residual bar is outlined only, so the encoding survives grayscale print.  Both
exhibits read the governed ``p5_peer_definition_rows`` table and nothing else.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.transforms import blended_transform_factory

from pipeline.figures._common import _C, _TINT, Grid, categorical_axis, figsize, figure

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_DEFINITIONS = ("barycentric_w2", "rbf_w2_median", "co_mention")
_PANEL_TITLES = {
    "barycentric_w2": "(A) W2 barycentric\n(target-anchored)",
    "rbf_w2_median": "(B) Median-bandwidth\nW2 RBF",
    "co_mention": "(C) Persistent\nco-mentions",
}
_REQUIRED_COLUMNS = {
    "target",
    "definition",
    "rank",
    "candidate",
    "weight",
    "remaining_mass",
    "remaining_count",
    "candidate_industry",
    "target_industry",
}
# Okabe-Ito hues paired with distinct hatches: the hatch alone identifies a
# shared peer in grayscale, the hue does so on screen.
_SHARED_STYLES = (
    (_C["blue"], ""),
    (_C["orange"], "////"),
    (_C["green"], "\\\\\\\\"),
    (_C["vermilion"], "xxx"),
    (_C["purple"], "..."),
    (_C["sky"], "---"),
)
_BAR_HEIGHT = 0.68
_ANNOTATION_PAD = 0.012
_XLIM_HEADROOM = 1.22
_PANEL_HEADROOM = 1.45
_MIN_SHARED = 2
_TAG_WIDTH = 24


@dataclass(frozen=True, slots=True)
class PeerBasketFigurePaths:
    """Governed peer-definition rows, the featured target, and the image folder."""

    rows_path: Path
    output_dir: Path
    target: str = "AMD"
    definitions_target: str = "HON"


@dataclass(frozen=True, slots=True)
class _Style:
    face: str
    edge: str
    hatch: str


def _tex(text: str) -> str:
    """Escape the LaTeX specials that occur in universe classification labels."""
    return text.replace("&", r"\&")


def _target_rows(rows: pd.DataFrame, target: str) -> pd.DataFrame:
    """Validate and return one target's complete three-definition basket."""
    missing = _REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        message = f"peer-definition rows are missing columns {sorted(missing)}"
        raise ValueError(message)
    selected = rows.loc[rows["target"] == target]
    if selected.empty:
        message = f"peer-definition rows contain no basket for {target}"
        raise ValueError(message)
    present = set(selected["definition"])
    if present != set(_DEFINITIONS):
        message = (
            f"{target} basket lacks definitions {sorted(set(_DEFINITIONS) - present)}"
        )
        raise ValueError(message)
    return selected.sort_values(["definition", "rank"], kind="stable")


def _peer_styles(selected: pd.DataFrame) -> dict[str, _Style]:
    """Give each peer shared by two or more definitions one fixed style."""
    counts = selected.groupby("candidate")["definition"].nunique()
    ordered = (
        selected.assign(order=selected["definition"].map(_DEFINITIONS.index))
        .sort_values(["order", "rank"], kind="stable")["candidate"]
        .drop_duplicates()
        .tolist()
    )
    styles: dict[str, _Style] = {}
    shared = [peer for peer in ordered if counts[peer] >= _MIN_SHARED]
    for peer, (colour, hatch) in zip(shared, _SHARED_STYLES, strict=False):
        styles[peer] = _Style(colour, _C["black"] if hatch else colour, hatch)
    for peer in ordered:
        styles.setdefault(peer, _Style(_TINT["neutral"], _C["gray"], ""))
    return styles


def _draw_basket(
    ax: Axes,
    basket: pd.DataFrame,
    styles: dict[str, _Style],
    xmax: float,
    *,
    residual_bar: bool,
) -> list[float]:
    """Draw one ranked basket plus its residual row; return the y positions.

    With ``residual_bar`` the residual mass is a dashed outline bar on the same
    scale; otherwise it is stated as text so a panel may use its own scale.
    """
    weights = [*basket["weight"].astype(float).tolist()]
    remaining = float(basket["remaining_mass"].iloc[0])
    remaining_count = int(basket["remaining_count"].iloc[0])
    labels = [*basket["candidate"].astype(str).tolist()]
    positions = [float(index) for index in range(len(labels) + 1)]
    for position, peer, weight in zip(positions, labels, weights, strict=False):
        style = styles[peer]
        ax.barh(
            position,
            weight,
            height=_BAR_HEIGHT,
            color=style.face,
            edgecolor=style.edge,
            hatch=style.hatch,
            linewidth=0.6,
            zorder=2,
        )
    if not residual_bar:
        ax.text(
            _ANNOTATION_PAD * xmax,
            positions[-1],
            f"rest {remaining:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_C["gray"],
            fontstyle="italic",
        )
    else:
        _residual_bar(ax, positions[-1], remaining)
    annotated = [*weights, remaining] if residual_bar else weights
    for position, value in zip(positions, annotated, strict=False):
        ax.text(
            value + _ANNOTATION_PAD * xmax,
            position,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
        )
    ax.set_yticks(positions, [*labels, f"Other {remaining_count}"])
    ax.set_ylim(positions[-1] + 0.6, -0.6)
    ax.set_xlim(0.0, xmax)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    categorical_axis(ax, "y")
    return positions


def _residual_bar(ax: Axes, position: float, remaining: float) -> None:
    """Draw the residual peer mass as an outline-only bar."""
    ax.barh(
        position,
        remaining,
        height=_BAR_HEIGHT,
        color="white",
        edgecolor=_C["gray"],
        hatch="",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=2,
    )


def _render_peer_basket(
    selected: pd.DataFrame, target: str, output_file: Path, frac: float
) -> None:
    """Render the W2 basket with a classification column."""
    styles = _peer_styles(selected)
    basket = selected.loc[selected["definition"] == "barycentric_w2"]
    remaining = float(basket["remaining_mass"].iloc[0])
    xmax = max(float(basket["weight"].max()), remaining) * _XLIM_HEADROOM
    target_industry = _tex(str(basket["target_industry"].iloc[0]))
    with figure(output_file, figsize(frac, 0.52), frac=frac) as (_fig, ax):
        positions = _draw_basket(ax, basket, styles, xmax, residual_bar=True)
        column = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            1.03,
            -0.95,
            "Industry",
            transform=column,
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        tags = [
            _tex(textwrap.fill(str(industry), _TAG_WIDTH))
            for industry in basket["candidate_industry"].tolist()
        ]
        for position, tag in zip(positions, [*tags, "remaining peers"], strict=True):
            ax.text(
                1.03,
                position,
                tag,
                transform=column,
                ha="left",
                va="center",
                fontsize=7,
                linespacing=0.95,
                color=_C["black"] if tag == target_industry else _C["gray"],
            )
        ax.set_xlabel(
            rf"Barycentric weight $W^\flat_{{\mathrm{{{target}}},j}}$ "
            rf"(target industry: {target_industry})"
        )
        ax.set_ylabel("Peer firm $j$")


def _render_peer_definitions(
    selected: pd.DataFrame, target: str, output_file: Path, frac: float
) -> None:
    """Render the three definitions side by side on one weight scale."""
    styles = _peer_styles(selected)
    with figure(output_file, figsize(frac, 0.46), Grid(1, 3), frac=frac) as (
        fig,
        axes,
    ):
        for ax, definition in zip(axes, _DEFINITIONS, strict=True):
            basket = selected.loc[selected["definition"] == definition]
            xmax = float(basket["weight"].max()) * _PANEL_HEADROOM
            _draw_basket(ax, basket, styles, xmax, residual_bar=False)
            ax.set_title(_PANEL_TITLES[definition], fontsize=8.5)
            ax.locator_params(axis="x", nbins=3)
        axes[0].set_ylabel(f"Peer of {target}")
        fig.supxlabel(
            f"Weight on peer $j$ in the {target} row (own scale per panel)",
            fontsize=9,
        )
        fig.tight_layout(w_pad=1.6)


def render_paper5_peer_basket_figures(paths: PeerBasketFigurePaths) -> None:
    """Render both peer-basket exhibits at the shared authoring width."""
    rows = pd.read_parquet(paths.rows_path)
    basket_rows = _target_rows(rows, paths.target)
    definition_rows = _target_rows(rows, paths.definitions_target)
    _render_peer_basket(
        basket_rows, paths.target, paths.output_dir / "peer_basket.pgf", 1.0
    )
    _render_peer_definitions(
        definition_rows,
        paths.definitions_target,
        paths.output_dir / "peer_definitions.pgf",
        1.0,
    )
    logger.info(
        "Rendered Paper 5 peer basket for %s and definitions for %s",
        paths.target,
        paths.definitions_target,
    )


__all__ = ["PeerBasketFigurePaths", "render_paper5_peer_basket_figures"]

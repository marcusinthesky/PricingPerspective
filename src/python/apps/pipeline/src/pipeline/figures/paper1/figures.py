"""Paper 1 publication figures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import MDS

from pipeline.figures._common import (
    _C,
    _save,
    _style,
    figsize,
    fit_to_width,
    thin_for_vector,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

_LEGEND_MAX_COLUMNS = 4


@dataclass(frozen=True, slots=True)
class Paper1FigurePaths:
    """Artifact paths consumed by the Paper 1 data-figure leaf."""

    distance_artifact_dir: Path
    universe_csv: Path
    output_dir: Path


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
    _style()
    palette = _label_palette(labels)
    # Printed full-width: 47 annotated points cannot carry 6.5pt tickers at the
    # 0.8 fraction this used to be included at. Marker area, edge width, and the
    # annotation offset are all point-valued, so they shrink with the canvas —
    # otherwise the drawing gets chunkier as the text finally comes out right.
    fig, ax = plt.subplots(figsize=figsize(1.0, 6.0 / 7.0))
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
    for i, ticker in enumerate(tickers):
        ax.annotate(
            ticker,
            (coords[i, 0], coords[i, 1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            alpha=0.9,
            ha="left",
            va="bottom",
            zorder=3,
        )
    ax.set_xlabel(text.xlabel)
    ax.set_ylabel(text.ylabel)
    ax.set_title(text.title)
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
        fontsize=7,
        title_fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(_LEGEND_MAX_COLUMNS, len(handles)),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.tight_layout()
    fit_to_width(fig, 1.0)
    _save(fig, out)


def _fig_mds_shepard(
    out: Path, raw_distances: np.ndarray, coords_2d: np.ndarray, spearman: float
) -> None:
    """Original versus projected distance for the descriptive 2D MDS view."""
    _style()
    differences = coords_2d[:, None, :] - coords_2d[None, :, :]
    projected = np.sqrt(np.sum(differences * differences, axis=2))
    upper = np.triu_indices(raw_distances.shape[0], k=1)
    fig, ax = plt.subplots(figsize=figsize(0.62, 3.6 / 4.4))
    # One mark per dyad, so quadratic in the roster: 1,326 at 52 firms, 4,950 at
    # 100, which overran TeX's main memory as vector paths. A Shepard cloud is
    # read as a band about the identity rather than point by point, so a fixed
    # deterministic thinning preserves what the panel shows.
    (shown_raw, shown_projected), total_dyads = thin_for_vector(
        raw_distances[upper], projected[upper]
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
    ax.set_ylabel("Distance in 2D MDS projection")
    ax.set_title(rf"MDS Shepard diagnostic ($\rho_S={spearman:.3f}$)")
    fig.tight_layout()
    fit_to_width(fig, 0.62)
    _save(fig, out)


def render_paper1_figures(paths: Paper1FigurePaths) -> None:
    """Render Paper 1 data-figures from the latest reproduced data.

    Regenerates the sector MDS and its Shepard diagnostic. Number bindings are
    written separately by ``render_paper1_numbers``.
    """
    _style()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # --- align the primary W2 geometry with universe metadata ---
    w2_frame, w2_summary = read_typed_distance_artifact(
        paths.distance_artifact_dir,
        expected_identity={
            "provider_id": "qwen3-embedding-8b",
            "representation_id": "qwen3-embedding-8b-unit",
            "distance_id": "wasserstein_w2",
        },
    )
    item_ids = w2_summary.get("item_ids")
    if not isinstance(item_ids, list):
        message = "typed distance summary item_ids must be a list"
        raise TypeError(message)
    w2_tickers = [str(ticker) for ticker in item_ids]
    w2_distances = (
        w2_frame["value"]
        .to_numpy(dtype=np.float64)
        .reshape(len(w2_tickers), len(w2_tickers))
    )
    universe = pd.read_csv(paths.universe_csv)
    sectors = dict(zip(universe["Symbol"], universe["Sector"], strict=False))

    common = sorted(set(w2_tickers) & set(sectors.keys()))
    w2_idx = [w2_tickers.index(t) for t in common]
    aligned_w2 = w2_distances[np.ix_(w2_idx, w2_idx)]

    sector_labels = [str(sectors.get(t, "Unknown")) for t in common]
    logger.info("Paper 1 figures over %d common tickers", len(common))

    # Match the governed dimensionality stage's metric-MDS specification so
    # the plotted Shepard statistic and generated number binding agree.
    mds = MDS(n_components=2)
    mds.set_params(
        metric_mds=True,
        metric="precomputed",
        init="random",
        random_state=42,
        n_init=100,
        max_iter=1000,
        normalized_stress=True,
    )
    mds_2d = mds.fit_transform(aligned_w2)
    upper = np.triu_indices_from(aligned_w2, k=1)
    projected = np.linalg.norm(mds_2d[:, None, :] - mds_2d[None, :, :], axis=2)
    mds_spearman = float(spearmanr(aligned_w2[upper], projected[upper]).statistic)

    # --- Fig 2: Metric MDS of the pairwise W2-distance matrix ---
    _fig_embedding_scatter(
        paths.output_dir / "mds_sector_plot.pgf",
        mds_2d,
        common,
        sector_labels,
        ScatterText(
            xlabel="MDS dimension 1",
            ylabel="MDS dimension 2",
            title=r"Metric MDS of pairwise exact $W_2$ distances",
            legend_title="Sector",
        ),
    )
    _fig_mds_shepard(
        paths.output_dir / "mds_shepard_diagnostic.pgf",
        aligned_w2,
        mds_2d,
        mds_spearman,
    )

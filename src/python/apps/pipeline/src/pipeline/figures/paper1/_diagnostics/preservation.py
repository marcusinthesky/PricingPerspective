"""Matryoshka distance-preservation panel and rank-agreement series."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def _load_pair_series(
    artifact_dir: Path,
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> pd.DataFrame:
    """Off-diagonal typed distances in a canonical dyad order."""
    frame, summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
        },
    )
    item_ids = summary.get("item_ids")
    if not isinstance(item_ids, list):
        message = "typed distance summary has no item roster"
        raise TypeError(message)
    values = frame["value"].to_numpy(dtype=float).reshape(len(item_ids), len(item_ids))
    rows = [
        {"a": str(item_ids[i]), "b": str(item_ids[j]), "distance": float(values[i, j])}
        for i in range(len(item_ids))
        for j in range(i + 1, len(item_ids))
    ]
    return pd.DataFrame(rows).sort_values(["a", "b"]).reset_index(drop=True)


def _distance_preservation(
    *,
    full_artifact_dir: Path,
    truncated_artifact_dirs: dict[int, Path],
    provider_id: str,
    full_representation_id: str,
    truncated_representation_ids: dict[int, str],
    distance_id: str,
    widths: tuple[int, ...],
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, float]]]:
    """Truncated-vs-full typed distances and their agreement statistics.

    The manuscript's truncation robustness claim is about the *distance
    matrix*: if truncating to the leading ``k`` coordinates left the pairwise
    geometry intact, all original-space pair rankings would also be intact. The
    diagnostic therefore measures how closely each truncated matrix tracks the
    full-width one.

    Returns the aligned per-width dyad frames and, per width, the Pearson and
    Spearman correlations against full width.
    """
    full = _load_pair_series(
        full_artifact_dir,
        provider_id=provider_id,
        representation_id=full_representation_id,
        distance_id=distance_id,
    )
    frames: dict[int, pd.DataFrame] = {}
    stats: dict[int, dict[str, float]] = {}
    for k in widths:
        trunc = _load_pair_series(
            truncated_artifact_dirs[k],
            provider_id=provider_id,
            representation_id=truncated_representation_ids[k],
            distance_id=distance_id,
        )
        if not (
            trunc["a"].equals(full["a"]) and trunc["b"].equals(full["b"])
        ):  # pragma: no cover - guards a silent misalignment
            message = (
                f"dyad ordering differs between dim_{k} and dim_full distance tests"
            )
            raise ValueError(message)
        merged = pd.DataFrame(
            {
                "truncated": trunc["distance"].to_numpy(),
                "full": full["distance"].to_numpy(),
            }
        )
        frames[k] = merged
        stats[k] = {
            "pearson": float(np.corrcoef(merged["truncated"], merged["full"])[0, 1]),
            "spearman": float(spearmanr(merged["truncated"], merged["full"]).statistic),
            "n_pairs": len(merged),
        }
        logger.info(
            "distance preservation k=%d: pearson=%.4f spearman=%.4f over %d dyads",
            k,
            stats[k]["pearson"],
            stats[k]["spearman"],
            stats[k]["n_pairs"],
        )
    return frames, stats


def _fig_distance_preservation(
    out: Path,
    frames: dict[int, pd.DataFrame],
    stats: dict[int, dict[str, float]],
) -> None:
    """Plot truncated versus full-width distance, one panel per width.

    Each panel carries the identity line and rank correlation against full
    width. The figure diagnoses geometric distortion, not return association.
    """
    _style()
    widths = sorted(frames)
    fig, axes = plt.subplots(
        1, len(widths), figsize=figsize(1.0, 3.2 / 6.5), sharey=True
    )
    if len(widths) == 1:  # pragma: no cover - defensive
        axes = [axes]

    lo = min(float(f.to_numpy().min()) for f in frames.values())
    hi = max(float(f.to_numpy().max()) for f in frames.values())
    pad = 0.02 * (hi - lo)

    for ax, k in zip(axes, widths, strict=True):
        merged = frames[k]
        # Quadratic in the roster and three panels of it, which exhausted TeX's
        # main memory at 100 firms. Thinned deterministically; the panel is read
        # as a mass about the diagonal, not as individual dyads.
        (shown_full, shown_truncated), total_dyads = thin_for_vector(
            merged["full"].to_numpy(dtype=float),
            merged["truncated"].to_numpy(dtype=float),
        )
        ax.scatter(
            shown_full,
            shown_truncated,
            s=3,
            alpha=0.25,
            edgecolors="none",
            color=_C["blue"],
            label=(
                f"{len(shown_full):,} of {total_dyads:,} dyads"
                if len(shown_full) < total_dyads
                else f"{total_dyads:,} dyads"
            ),
        )
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="0.35",
            linestyle="dashed",
            linewidth=0.7,
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        title = rf"$k = {k}$   ($\rho_s = {stats[k]['spearman']:.2f}$)"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Full-width W2 distance", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=7)

    axes[0].set_ylabel("Truncated W2 distance", fontsize=8)
    fig.tight_layout()
    fit_to_width(fig, 1.0)
    _save(fig, out)

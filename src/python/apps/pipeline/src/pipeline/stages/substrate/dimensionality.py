"""Governed metric-MDS projection-fidelity analysis for Paper 1."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd
import typer
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS

from pipeline.stages.substrate.mantel import _load_typed_distance_metric

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
MIN_MDS_TICKERS = 3


@dataclass(frozen=True)
class ProjectionLabels:
    """Sector and industry labels used by projection diagnostics."""

    sectors: dict[str, str]
    industries: dict[str, str]


def _load_covariance_tickers(parquet_path: Path) -> list[str]:
    """Return the priced-firm universe represented by the covariance artifact."""
    con = duckdb.connect(":memory:")
    rows = con.execute(
        """
        SELECT ticker_i AS ticker FROM read_parquet(?)
        UNION
        SELECT ticker_j AS ticker FROM read_parquet(?)
        ORDER BY ticker
        """,
        [str(parquet_path), str(parquet_path)],
    ).fetchall()
    con.close()
    return [str(row[0]) for row in rows]


def _compute_mds_2d(
    dist_matrix: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, float]:
    """Compute 2D metric MDS and retain normalized Stress-1."""
    # scikit-learn 1.9 renamed the metric-MDS arguments ahead of the type stubs
    # bundled with the current checker. Isolate that temporary mismatch here.
    mds_class: Any = MDS
    mds = mds_class(
        n_components=2,
        metric_mds=True,
        metric="precomputed",
        init="random",
        random_state=random_state,
        n_init=100,
        max_iter=1000,
        normalized_stress=True,
    )
    coords = mds.fit_transform(dist_matrix)
    return coords, float(mds.stress_)


def _projection_fidelity(
    tickers: list[str],
    raw_distances: np.ndarray,
    coords_2d: np.ndarray,
    labels: ProjectionLabels,
    top_k: int = 5,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """MDS Shepard and deterministic neighbour-preservation diagnostics."""
    differences = coords_2d[:, None, :] - coords_2d[None, :, :]
    projected = np.sqrt(np.sum(differences * differences, axis=2))
    upper = np.triu_indices(len(tickers), k=1)
    raw_values = raw_distances[upper]
    projected_values = projected[upper]
    ticker_key = np.asarray(tickers, dtype=object)
    neighbour_rows: list[dict[str, object]] = []
    recalls: list[float] = []
    exact_nearest = 0
    for anchor_idx, anchor in enumerate(tickers):
        candidate = np.arange(len(tickers))
        candidate = candidate[candidate != anchor_idx]
        raw_order = candidate[
            np.lexsort((ticker_key[candidate], raw_distances[anchor_idx, candidate]))
        ]
        projected_order = candidate[
            np.lexsort((ticker_key[candidate], projected[anchor_idx, candidate]))
        ]
        raw_rank = {int(index): rank + 1 for rank, index in enumerate(raw_order)}
        projected_rank = {
            int(index): rank + 1 for rank, index in enumerate(projected_order)
        }
        recalls.append(
            len(set(raw_order[:top_k].tolist()) & set(projected_order[:top_k].tolist()))
            / top_k
        )
        exact_nearest += int(raw_order[0] == projected_order[0])
        for neighbour_idx in raw_order:
            neighbour = tickers[int(neighbour_idx)]
            neighbour_rows.append(
                {
                    "anchor": anchor,
                    "neighbour": neighbour,
                    "raw_distance": float(raw_distances[anchor_idx, neighbour_idx]),
                    "raw_rank": int(raw_rank[int(neighbour_idx)]),
                    "projected_distance": float(projected[anchor_idx, neighbour_idx]),
                    "projected_rank": int(projected_rank[int(neighbour_idx)]),
                    "anchor_sector": str(labels.sectors.get(anchor, "Unknown")),
                    "neighbour_sector": str(labels.sectors.get(neighbour, "Unknown")),
                    "anchor_industry": str(labels.industries.get(anchor, "Unknown")),
                    "neighbour_industry": str(
                        labels.industries.get(neighbour, "Unknown")
                    ),
                }
            )
    diagnostics: dict[str, float | int] = {
        "shepard_pearson": float(pearsonr(raw_values, projected_values).statistic),
        "shepard_spearman": float(spearmanr(raw_values, projected_values).statistic),
        "top5_neighbour_recall_mean": float(np.mean(recalls)),
        "exact_nearest_neighbour_preserved": int(exact_nearest),
        "n_tickers": len(tickers),
    }
    return diagnostics, pd.DataFrame(neighbour_rows)


def run_dimensionality_analysis(
    distance_artifact_dir: Path,
    covariance_matrix: Path,
    universe_csv: Path,
    output_file: Path,
) -> None:
    """Materialize metric-MDS coordinates and projection-fidelity diagnostics."""
    typer.echo("Loading data...")
    distance_tickers, distance_dist, distance_id = _load_typed_distance_metric(
        distance_artifact_dir
    )
    cov_tickers = _load_covariance_tickers(covariance_matrix)
    universe = pd.read_csv(universe_csv)
    sectors = dict(zip(universe["Symbol"], universe["Sector"], strict=False))
    industries = dict(zip(universe["Symbol"], universe["Industry"], strict=False))
    common_tickers = sorted(
        set(distance_tickers) & set(cov_tickers) & set(sectors.keys())
    )
    typer.echo(
        f"  {distance_id} firms: {len(distance_tickers)}; priced/common firms: "
        f"{len(common_tickers)}"
    )
    if len(common_tickers) < MIN_MDS_TICKERS:
        typer.echo(f"Error: Need at least {MIN_MDS_TICKERS} common tickers", err=True)
        raise typer.Exit(code=1)

    distance_idx = [distance_tickers.index(ticker) for ticker in common_tickers]
    aligned_distance = distance_dist[np.ix_(distance_idx, distance_idx)]
    typer.echo("Computing 2D metric MDS and fidelity diagnostics...")
    distance_mds_2d, mds_stress_1 = _compute_mds_2d(aligned_distance)
    fidelity, neighbours = _projection_fidelity(
        common_tickers,
        aligned_distance,
        distance_mds_2d,
        ProjectionLabels(sectors=sectors, industries=industries),
    )
    typer.echo(
        f"  Stress-1={mds_stress_1:.4f}; Shepard Spearman="
        f"{float(fidelity['shepard_spearman']):.4f}; top-5 recall="
        f"{float(fidelity['top5_neighbour_recall_mean']):.4f}"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "metric": f"{distance_id}_mds_2d",
                "distance_id": distance_id,
                "n_tickers": len(common_tickers),
                "n_sectors": len({sectors[ticker] for ticker in common_tickers}),
                "distance_analysis_scale": distance_id,
                "mds_stress_1": mds_stress_1,
                "mds_shepard_pearson": float(fidelity["shepard_pearson"]),
                "mds_shepard_spearman": float(fidelity["shepard_spearman"]),
                "mds_top5_neighbour_recall_mean": float(
                    fidelity["top5_neighbour_recall_mean"]
                ),
                "mds_exact_nearest_neighbour_preserved": int(
                    fidelity["exact_nearest_neighbour_preserved"]
                ),
            }
        ]
    )
    summary.to_csv(output_file, index=False)

    coords_path = output_file.parent / "coords_2d.parquet"
    coords = pd.DataFrame(
        {
            "mds_1": distance_mds_2d[:, 0],
            "mds_2": distance_mds_2d[:, 1],
            "distance_id": [distance_id] * len(common_tickers),
            "distance_analysis_scale": [distance_id] * len(common_tickers),
            "mds_stress_1": [mds_stress_1] * len(common_tickers),
            "mds_shepard_pearson": [float(fidelity["shepard_pearson"])]
            * len(common_tickers),
            "mds_shepard_spearman": [float(fidelity["shepard_spearman"])]
            * len(common_tickers),
            "mds_top5_neighbour_recall_mean": [
                float(fidelity["top5_neighbour_recall_mean"])
            ]
            * len(common_tickers),
            "mds_exact_nearest_neighbour_preserved": [
                int(fidelity["exact_nearest_neighbour_preserved"])
            ]
            * len(common_tickers),
        },
        index=pd.Index(common_tickers, name="ticker"),
    )
    coords.to_parquet(coords_path)
    neighbours_path = output_file.parent / "neighbours.parquet"
    neighbours.to_parquet(neighbours_path, index=False)

    typer.echo(f"Saved: {output_file}")
    typer.echo(f"Saved: {coords_path}")
    typer.echo(f"Saved: {neighbours_path}")

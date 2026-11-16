"""Paper 1 PCoA spectra and metric-MDS fidelity on one priced firm roster.

All numerical work uses NumPy float64 on CPU. Eigenvalue sign/rank decisions
use machine epsilon times matrix order and spectral scale, rather than an
absolute cutoff that would change the answer when distance units change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from statadic.observation.geometry import ObservedDissimilarity
from statadic.ordination.fit import Ordination

from pipeline.io.params import load_params
from pipeline.precision import own_float64
from pipeline.stages.shared.representation_ablation import (
    load_extended_representation_ablation_specs,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.shared.representation_ablation import (
        RepresentationAblationSpec,
    )

_MATRIX_RANK = 2
_MIN_FIRMS = 3


@dataclass(frozen=True, slots=True)
class Paper1PcoaOptions:
    """Declared inputs and output directory for the Paper 1 projection leaf."""

    distance_root: Path
    covariance_matrix: Path
    universe_csv: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class MetricMdsOptions:
    """Explicit metric-MDS solver budget and deterministic initialization."""

    n_init: int = 100
    max_iter: int = 5000
    eps: float = 1e-6
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class ProjectionLabels:
    """Sector and industry labels used by projection diagnostics."""

    sectors: dict[str, str]
    industries: dict[str, str]


def _tolerance(values: np.ndarray) -> float:
    """Bound accumulated float64 roundoff relative to the numerical scale."""
    return float(np.finfo(values.dtype).eps * values.size * np.max(np.abs(values)))


def _spectrum(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    """Retain signed eigenvalues and normalize over every positive axis."""
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    return eigenvalues, _inertia(eigenvalues)


def _inertia(eigenvalues: np.ndarray) -> dict[str, int | float]:
    """Classify a descending signed spectrum against its scale-aware zero."""
    tolerance = _tolerance(eigenvalues)
    positive = eigenvalues > tolerance
    negative = eigenvalues < -tolerance
    positive_mass = float(eigenvalues[positive].sum())
    negative_mass = float(np.abs(eigenvalues[negative]).sum())
    absolute_mass = positive_mass + negative_mass
    shares = np.zeros_like(eigenvalues)
    if positive_mass > 0.0:
        shares[positive] = eigenvalues[positive] / positive_mass
    return {
        "n_positive_axes": int(positive.sum()),
        "n_negative_axes": int(negative.sum()),
        "negative_inertia_fraction": (
            negative_mass / absolute_mass if absolute_mass > 0.0 else 0.0
        ),
        "axis_1_share": float(shares[0]),
        "axis_2_share": float(shares[1]),
        "two_axis_share": float(shares[:2].sum()),
        "eigenvalue_tolerance": tolerance,
    }


def _spectrum_frame(cell_id: str, eigenvalues: np.ndarray) -> pd.DataFrame:
    """Emit full signed spectra, including numerical zero and negative axes."""
    positive = np.where(eigenvalues > _tolerance(eigenvalues), eigenvalues, 0.0)
    total = float(positive.sum())
    shares = positive / total if total > 0.0 else np.zeros_like(positive)
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "positive_inertia_share": shares,
            "cumulative_positive_inertia_share": np.cumsum(shares),
        }
    )


def _pcoa(
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Fit statadic PCoA, padding absent positive coordinate axes with zero.

    Stress-1 is descriptive, not minimized by PCoA: the square root of the
    upper-triangle squared distance residual sum divided by the original
    upper-triangle squared distance sum. An all-zero geometry has stress zero.
    """
    own_float64()
    distances = np.asarray(distances, dtype=np.float64)
    if (
        distances.ndim != _MATRIX_RANK
        or distances.shape[0] != distances.shape[1]
        or distances.shape[0] < _MIN_FIRMS
        or not np.isfinite(distances).all()
    ):
        message = "PCoA requires a finite square distance matrix with at least 3 firms"
        raise ValueError(message)
    tolerance = _tolerance(distances.reshape(-1))
    if (
        not np.allclose(distances, distances.T, rtol=0.0, atol=tolerance)
        or np.any(distances < -tolerance)
        or np.any(np.abs(np.diag(distances)) > tolerance)
    ):
        message = "PCoA distances must be symmetric, nonnegative and zero-diagonal"
        raise ValueError(message)
    distances = np.maximum((distances + distances.T) / 2.0, 0.0)
    np.fill_diagonal(distances, 0.0)
    if not np.isfinite(distances**2).all():
        message = "PCoA squared-distance centering overflowed"
        raise ValueError(message)
    # Classical scaling is statadic's; this stage keeps its own eps*n zero rule and
    # the signed spectrum, which statadic's strict negative-type door would reject.
    spectrum = Ordination.pcoa(
        dim=None,
        negative_inertia="positive-subspace",
        tolerance=float(np.finfo(np.float64).eps * len(distances)),
    ).estimate(ObservedDissimilarity.from_array(distances))
    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    inertia = _inertia(eigenvalues)
    rank = min(2, int(inertia["n_positive_axes"]))
    coordinates = np.zeros((len(distances), 2), dtype=np.float64)
    if rank:
        axes = np.asarray(spectrum.coordinates[:, :rank], dtype=np.float64)
        # Deterministic sign: the largest-magnitude entry of each axis is positive.
        pivot = np.argmax(np.abs(axes), axis=0)
        coordinates[:, :rank] = axes * np.sign(axes[pivot, np.arange(rank)])
    return (
        coordinates,
        eigenvalues,
        {
            **inertia,
            **_stress_diagnostics(distances, coordinates),
        },
    )


def _stress_diagnostics(
    distances: np.ndarray, coordinates: np.ndarray
) -> dict[str, float]:
    """Use original-distance normalization before and after one fitted dilation.

    The fitted scale minimizes squared distance error without changing point
    order, neighbours or the PCoA spectrum. Saved coordinates stay unscaled.
    """
    upper = np.triu_indices(len(distances), k=1)
    original = np.asarray(distances, dtype=np.float64)[upper]
    projected = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2
    )[upper]
    denominator = float(original @ original)
    projected_sum = float(projected @ projected)
    scale = float(original @ projected) / projected_sum if projected_sum > 0 else 1.0
    residual = original - projected
    fitted_residual = original - scale * projected
    return {
        "stress_1": (
            float(np.sqrt((residual @ residual) / denominator))
            if denominator > 0.0
            else 0.0
        ),
        "scale_fitted_stress_1": (
            float(np.sqrt((fitted_residual @ fitted_residual) / denominator))
            if denominator > 0.0
            else 0.0
        ),
        "optimal_distance_scale": scale,
    }


def _metric_mds(
    distances: np.ndarray, options: MetricMdsOptions
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Fit metric MDS, then measure stress with the same convention as PCoA."""
    matrix = np.asarray(distances, dtype=np.float64)
    if np.any(matrix):
        # These public 1.9 arguments are newer than the bundled MDS stubs.
        estimator = MDS(
            n_components=2,
            metric_mds=True,  # pyrefly: ignore[unexpected-keyword]
            metric="precomputed",  # pyrefly: ignore[bad-argument-type]
            init="random",  # pyrefly: ignore[unexpected-keyword]
            n_init=options.n_init,
            max_iter=options.max_iter,
            eps=options.eps,
            random_state=options.random_state,
            normalized_stress=False,
        )
        coordinates = estimator.fit_transform(matrix)
        iterations = int(estimator.n_iter_)
    else:
        coordinates = np.zeros((len(matrix), 2), dtype=np.float64)
        iterations = 0
    return coordinates, {
        "n_iter": iterations,
        "converged": iterations < options.max_iter,
        **_stress_diagnostics(matrix, coordinates),
    }


def _projection_fidelity(
    tickers: list[str],
    raw_distances: np.ndarray,
    coords_2d: np.ndarray,
    labels: ProjectionLabels,
    top_k: int = 5,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Compare all dyads and resolve equal-distance ranks by ticker name.

    Correlations are missing for a constant distance vector. Neighbour recall
    uses all available neighbours when there are fewer than five.
    """
    projected = np.linalg.norm(coords_2d[:, None, :] - coords_2d[None, :, :], axis=2)
    upper = np.triu_indices(len(tickers), k=1)
    raw_values, projected_values = raw_distances[upper], projected[upper]
    correlations_defined = np.ptp(raw_values) > 0 and np.ptp(projected_values) > 0
    ticker_key = np.asarray(tickers, dtype=object)
    rows: list[dict[str, object]] = []
    recalls: list[float] = []
    exact_nearest = 0
    k = min(top_k, len(tickers) - 1)
    for i, anchor in enumerate(tickers):
        candidates = np.delete(np.arange(len(tickers)), i)
        raw_order = candidates[
            np.lexsort((ticker_key[candidates], raw_distances[i, candidates]))
        ]
        projected_order = candidates[
            np.lexsort((ticker_key[candidates], projected[i, candidates]))
        ]
        projected_rank = {
            int(index): rank + 1 for rank, index in enumerate(projected_order)
        }
        recalls.append(len(set(raw_order[:k]) & set(projected_order[:k])) / k)
        exact_nearest += int(raw_order[0] == projected_order[0])
        for rank, j in enumerate(raw_order, start=1):
            neighbour = tickers[int(j)]
            rows.append(
                {
                    "anchor": anchor,
                    "neighbour": neighbour,
                    "raw_distance": float(raw_distances[i, j]),
                    "raw_rank": rank,
                    "projected_distance": float(projected[i, j]),
                    "projected_rank": projected_rank[int(j)],
                    "anchor_sector": labels.sectors[anchor],
                    "neighbour_sector": labels.sectors[neighbour],
                    "anchor_industry": labels.industries[anchor],
                    "neighbour_industry": labels.industries[neighbour],
                }
            )
    return {
        "shepard_pearson": (
            float(pearsonr(raw_values, projected_values).statistic)
            if correlations_defined
            else None
        ),
        "shepard_spearman": (
            float(spearmanr(raw_values, projected_values).statistic)
            if correlations_defined
            else None
        ),
        "top5_neighbour_recall_mean": float(np.mean(recalls)),
        "exact_nearest_neighbour_preserved": exact_nearest,
    }, pd.DataFrame(rows)


def _load_covariance(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the full priced covariance panel, including its diagonal."""
    frame = pd.read_parquet(path, columns=["ticker_i", "ticker_j", "covariance"])
    if frame.empty or frame.isna().any().any():
        message = "return covariance must contain finite complete pair records"
        raise ValueError(message)
    if frame.duplicated(["ticker_i", "ticker_j"]).any():
        message = "return covariance has duplicate firm pairs"
        raise ValueError(message)
    tickers = sorted(set(frame["ticker_i"]) | set(frame["ticker_j"]))
    # Duplicate pairs are rejected above; pivot must not average malformed input.
    matrix = frame.pivot(  # noqa: PD010
        index="ticker_i", columns="ticker_j", values="covariance"
    )
    covariance = matrix.reindex(index=tickers, columns=tickers).to_numpy(
        dtype=np.float64
    )
    if len(tickers) < _MIN_FIRMS or not np.isfinite(covariance).all():
        message = "return covariance must cover every pair of at least 3 priced firms"
        raise ValueError(message)
    tolerance = _tolerance(covariance.reshape(-1))
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=tolerance):
        message = "return covariance must be symmetric"
        raise ValueError(message)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, _metadata = _spectrum(covariance)
    if np.any(eigenvalues < -_tolerance(eigenvalues)):
        message = "return covariance has materially negative eigenvalues"
        raise ValueError(message)
    return tickers, covariance


def _load_distances(
    root: Path, spec: RepresentationAblationSpec, tickers: list[str]
) -> np.ndarray:
    """Read one validated rooted-W2 leaf without changing the priced roster."""
    frame, summary = read_typed_distance_artifact(
        root / spec.provider_id / spec.representation_id / "wasserstein_w2",
        expected_identity={
            "provider_id": spec.provider_id,
            "representation_id": spec.representation_id,
            "distance_id": "wasserstein_w2",
        },
    )
    item_ids = summary["item_ids"]
    if not isinstance(item_ids, list) or not all(
        isinstance(item, str) for item in item_ids
    ):
        message = f"{spec.cell_id}: typed distance item IDs must be strings"
        raise ValueError(message)
    metadata = summary["metadata"]
    if (
        not isinstance(metadata, dict)
        or metadata.get("value_semantics") != "statistical_distance"
    ):
        message = f"{spec.cell_id}: PCoA requires rooted statistical distances"
        raise ValueError(message)
    if not set(tickers) <= set(item_ids):
        message = f"{spec.cell_id}: distance artifact does not cover the priced roster"
        raise ValueError(message)
    positions = [item_ids.index(ticker) for ticker in tickers]
    matrix = (
        frame["value"].to_numpy(dtype=np.float64).reshape(len(item_ids), len(item_ids))
    )
    return matrix[np.ix_(positions, positions)]


def run_paper1_pcoa(options: Paper1PcoaOptions) -> None:
    """Write the PCoA ladders, canonical metric MDS and matched return PCA."""
    specs = load_extended_representation_ablation_specs(options.params_file)
    mds_options = MetricMdsOptions(
        **load_params(options.params_file).get("paper1", {}).get("mds", {})
    )
    tickers, covariance = _load_covariance(options.covariance_matrix)
    universe = pd.read_csv(options.universe_csv, dtype=str)
    if universe["Symbol"].isna().any() or universe["Symbol"].duplicated().any():
        message = "universe metadata requires unique nonmissing symbols"
        raise ValueError(message)
    metadata = universe.set_index("Symbol").reindex(tickers)
    if metadata[["Sector", "Industry"]].isna().any().any():
        message = (
            "universe metadata must cover sector and industry for every priced firm"
        )
        raise ValueError(message)
    labels = ProjectionLabels(
        metadata["Sector"].to_dict(), metadata["Industry"].to_dict()
    )
    cells: list[dict[str, object]] = []
    coordinates: list[pd.DataFrame] = []
    spectra: list[pd.DataFrame] = []
    canonical_neighbours = pd.DataFrame()
    canonical_distances = np.empty((0, 0))
    for spec in specs:
        distances = _load_distances(options.distance_root, spec, tickers)
        coords, eigenvalues, inertia = _pcoa(distances)
        fidelity, neighbours = _projection_fidelity(tickers, distances, coords, labels)
        cells.append(
            {
                "cell_id": spec.cell_id,
                "label": spec.label,
                "provider_id": spec.provider_id,
                "representation_id": spec.representation_id,
                "effective_dimension": spec.effective_dimension,
                "family": "vintage"
                if spec.contrast_role == "encoder_vintage"
                else "representation",
                **inertia,
                **fidelity,
            }
        )
        coordinates.append(
            pd.DataFrame(
                {
                    "cell_id": spec.cell_id,
                    "ticker": tickers,
                    "pcoa_1": coords[:, 0],
                    "pcoa_2": coords[:, 1],
                    "sector": [labels.sectors[ticker] for ticker in tickers],
                    "industry": [labels.industries[ticker] for ticker in tickers],
                }
            )
        )
        spectra.append(_spectrum_frame(spec.cell_id, eigenvalues))
        if spec.canonical:
            canonical_neighbours = neighbours
            canonical_distances = distances
    mds_coordinates, mds_stress = _metric_mds(canonical_distances, mds_options)
    mds_fidelity, mds_neighbours = _projection_fidelity(
        tickers, canonical_distances, mds_coordinates, labels
    )
    return_eigenvalues, return_inertia = _spectrum(covariance)
    spectra.append(_spectrum_frame("return_pca", return_eigenvalues))
    covariance_metadata = pq.read_schema(options.covariance_matrix).metadata or {}
    return_sample: dict[str, object] = {}
    for field, output_key in (
        (b"return_panel_rows", "n_dates"),
        (b"return_panel_start_date", "start_date"),
        (b"return_panel_end_date", "end_date"),
    ):
        if field in covariance_metadata:
            value = covariance_metadata[field].decode("utf-8")
            return_sample[output_key] = int(value) if output_key == "n_dates" else value
    results = {
        "schema_version": 1,
        "method": "PCoA",
        "n_tickers": len(tickers),
        "tickers": tickers,
        "canonical_cell_id": next(spec.cell_id for spec in specs if spec.canonical),
        "inertia_definition": (
            "positive eigenvalue / sum of all positive eigenvalues above the "
            "scale-aware numerical-zero threshold"
        ),
        "negative_inertia_definition": (
            "absolute negative eigenvalue mass / total absolute eigenvalue mass, "
            "excluding numerical zeros"
        ),
        "stress_1_definition": (
            "sqrt(sum_upper((D - D_2d)^2) / sum_upper(D^2)); zero for the "
            "all-zero geometry; same original-distance denominator for both methods"
        ),
        "scale_fitted_stress_1_definition": (
            "stress_1 after multiplying projected distances by "
            "optimal_distance_scale = dot(D, D_2d) / dot(D_2d, D_2d); "
            "scale is 1 for a zero projection; saved coordinates and PCoA inertia "
            "are unchanged"
        ),
        "cells": cells,
        "metric_mds": {
            "method": "metric MDS",
            "cell_id": next(spec.cell_id for spec in specs if spec.canonical),
            "n_tickers": len(tickers),
            "settings": asdict(mds_options),
            **mds_stress,
            **mds_fidelity,
        },
        "return_pca": {
            "cell_id": "return_pca",
            "label": "Return PCA",
            "method": "PCA",
            "definition": (
                "eigenvalues of the same-roster unstandardized daily-return sample "
                "covariance; dates are observations, firms are features"
            ),
            **return_sample,
            **return_inertia,
        },
    }
    serialized = json.dumps(results, indent=2, allow_nan=False) + "\n"
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / "results.json").write_text(serialized, encoding="utf-8")
    pd.concat(coordinates, ignore_index=True).to_parquet(
        options.output_dir / "coords_2d.parquet", index=False
    )
    pd.concat(spectra, ignore_index=True).to_parquet(
        options.output_dir / "spectrum.parquet", index=False
    )
    canonical_neighbours.to_parquet(
        options.output_dir / "neighbours.parquet", index=False
    )
    pd.DataFrame(
        {
            "ticker": tickers,
            "mds_1": mds_coordinates[:, 0],
            "mds_2": mds_coordinates[:, 1],
            "sector": [labels.sectors[ticker] for ticker in tickers],
            "industry": [labels.industries[ticker] for ticker in tickers],
        }
    ).to_parquet(options.output_dir / "metric_mds_coords_2d.parquet", index=False)
    mds_neighbours.to_parquet(
        options.output_dir / "metric_mds_neighbours.parquet", index=False
    )

"""PCoA spectral meaning, matched return PCA and fail-closed artifact inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pipeline.io.contracts import write_manifest
from pipeline.stages.papers.paper1.pcoa import (
    MetricMdsOptions,
    Paper1PcoaOptions,
    ProjectionLabels,
    _load_covariance,
    _metric_mds,
    _pcoa,
    _projection_fidelity,
    _spectrum,
    _stress_diagnostics,
    run_paper1_pcoa,
)
from pipeline.stages.shared.representation_ablation import (
    load_extended_representation_ablation_specs,
)
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from pipeline.stages.shared.representation_ablation import (
        RepresentationAblationSpec,
    )

_ROOT = next(
    parent for parent in Path(__file__).parents if (parent / "params.yaml").exists()
)
_N_FIRMS = 5
_CELL_COUNT = 15
_SPECTRUM_COUNT = 16
_DISPLAY_AXES = 2
_N_DATES = 31
_PLANAR_MDS_STRESS_LIMIT = 1e-4


def _distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def test_exact_coordinates_have_perfect_shepard_and_neighbour_preservation() -> None:
    """Exact coordinates preserve distances and nearest neighbours."""
    tickers = ["A", "B", "C", "D"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    labels = dict.fromkeys(tickers, "Sector")
    stats, neighbours = _projection_fidelity(
        tickers, _distances(coords), coords, ProjectionLabels(labels, labels), top_k=2
    )
    assert stats["shepard_pearson"] == pytest.approx(1.0)
    assert stats["shepard_spearman"] == pytest.approx(1.0)
    assert stats["top5_neighbour_recall_mean"] == pytest.approx(1.0)
    assert stats["exact_nearest_neighbour_preserved"] == len(tickers)
    assert len(neighbours) == len(tickers) * (len(tickers) - 1)


def test_tied_neighbour_ranks_break_by_ticker() -> None:
    """Ticker names deterministically break equal-distance rank ties."""
    tickers = ["A", "B", "C"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    labels = dict.fromkeys(tickers, "x")
    stats, neighbours = _projection_fidelity(
        tickers, _distances(coords), coords, ProjectionLabels(labels, labels)
    )
    assert stats["top5_neighbour_recall_mean"] == pytest.approx(1.0)
    anchor_a = neighbours.loc[neighbours["anchor"] == "A"].sort_values("raw_rank")
    assert anchor_a.iloc[0]["neighbour"] == "B"
    assert anchor_a.iloc[0]["projected_rank"] == 1


def test_stress_normalization_separates_dilation_from_shape_error() -> None:
    """Uniform shrinkage affects raw stress but vanishes after the fitted scale."""
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    stats = _stress_diagnostics(_distances(coordinates), coordinates / 3.0)
    assert stats["stress_1"] == pytest.approx(2.0 / 3.0)
    assert stats["optimal_distance_scale"] == pytest.approx(3.0)
    assert stats["scale_fitted_stress_1"] == pytest.approx(0.0, abs=1e-15)


def test_metric_mds_recovers_planar_geometry_with_original_stress() -> None:
    """The library fit preserves planar distances and uses the shared denominator."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    distances = _distances(points)
    coordinates, stats = _metric_mds(distances, MetricMdsOptions(n_init=4, eps=1e-9))
    upper = np.triu_indices(len(points), k=1)
    expected = np.linalg.norm(
        (distances - _distances(coordinates))[upper]
    ) / np.linalg.norm(distances[upper])
    assert stats["stress_1"] == pytest.approx(expected)
    assert stats["stress_1"] < _PLANAR_MDS_STRESS_LIMIT
    assert stats["scale_fitted_stress_1"] <= stats["stress_1"]
    assert stats["n_iter"] > 0
    assert stats["converged"]


def test_metric_mds_exposes_an_exhausted_iteration_budget() -> None:
    """A fit stopped by the budget must not claim the convergence criterion passed."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    _coordinates, stats = _metric_mds(
        _distances(points), MetricMdsOptions(n_init=1, max_iter=1)
    )
    assert stats["n_iter"] == 1
    assert not stats["converged"]


def test_metric_mds_handles_zero_geometry_without_nonfinite_statistics() -> None:
    """An all-zero geometry has zero stress and needs no iterative fit."""
    coordinates, stats = _metric_mds(np.zeros((4, 4)), MetricMdsOptions())
    assert np.count_nonzero(coordinates) == 0
    assert stats["stress_1"] == pytest.approx(0.0)
    assert stats["scale_fitted_stress_1"] == pytest.approx(0.0)
    assert stats["n_iter"] == 0
    assert stats["converged"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_pcoa_matches_euclidean_pca_and_uses_the_full_spectrum(
    dtype: type[np.floating],
    scale: float,
) -> None:
    """Three-dimensional data retain a third inertia axis after a 2D display."""
    points = np.random.default_rng(42).normal(size=(9, 3)) * [3.0, 2.0, 1.0]
    distances = (_distances(points) * scale).astype(dtype)
    coords, eigenvalues, stats = _pcoa(distances)
    reference = PCA(svd_solver="full").fit(points * scale)
    expected = reference.explained_variance_ * (len(points) - 1)
    # Float32 input quantization is immaterial to inertia shares and pairwise
    # projected distances; the host eigensystem and output stay float64.
    tolerance = float(20 * np.finfo(dtype).eps)
    np.testing.assert_allclose(eigenvalues[:3], expected, rtol=tolerance)
    np.testing.assert_allclose(
        _distances(coords),
        _distances(reference.transform(points * scale)[:, :2]),
        rtol=tolerance,
        atol=scale * tolerance,
    )
    assert coords.dtype == np.float64
    assert stats["two_axis_share"] == pytest.approx(
        reference.explained_variance_ratio_[:2].sum(), rel=tolerance
    )
    assert stats["two_axis_share"] < 1.0
    upper = np.triu_indices(len(points), k=1)
    expected_stress = np.linalg.norm(
        (distances - _distances(coords))[upper]
    ) / np.linalg.norm(distances[upper])
    assert stats["stress_1"] == pytest.approx(expected_stress)


def test_signed_inertia_reports_non_euclidean_geometry() -> None:
    """The four-cycle shortest-path metric has genuine negative PCoA inertia."""
    distances = np.array(
        [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]], dtype=float
    )
    coords, eigenvalues, stats = _pcoa(distances)
    np.testing.assert_allclose(eigenvalues, [2.0, 2.0, 0.0, -1.0], atol=1e-14)
    assert stats["n_positive_axes"] == _DISPLAY_AXES
    assert stats["n_negative_axes"] == 1
    assert stats["negative_inertia_fraction"] == pytest.approx(0.2)
    assert stats["axis_1_share"] == pytest.approx(0.5)
    assert np.isfinite(coords).all()


@pytest.mark.parametrize("rank", [0, 1])
def test_degenerate_positive_rank_is_zero_padded_without_nan(rank: int) -> None:
    """Absent coordinate axes remain zero; undefined correlations remain missing."""
    points = np.arange(4, dtype=float).reshape(-1, 1) * rank
    distances = _distances(points)
    coords, _eigenvalues, stats = _pcoa(distances)
    assert stats["n_positive_axes"] == rank
    assert coords[:, 1].tolist() == [0.0] * len(points)
    assert np.isfinite(coords).all()
    assert stats["stress_1"] == pytest.approx(0.0, abs=1e-14)
    if rank == 0:
        labels = dict.fromkeys(["A", "B", "C", "D"], "x")
        fidelity, _neighbours = _projection_fidelity(
            list(labels), distances, coords, ProjectionLabels(labels, labels)
        )
        assert fidelity["shepard_pearson"] is None
        assert fidelity["shepard_spearman"] is None
        assert stats["two_axis_share"] == pytest.approx(0.0)


@pytest.mark.parametrize("failure", ["asymmetric", "negative", "diagonal", "nonfinite"])
def test_pcoa_rejects_invalid_dissimilarities(failure: str) -> None:
    """Malformed inputs cannot become plausible projection coordinates."""
    distances = _distances(np.arange(4, dtype=float).reshape(-1, 1))
    if failure == "asymmetric":
        distances[0, 1] += 1.0
    elif failure == "negative":
        distances[0, 1] = distances[1, 0] = -1.0
    elif failure == "diagonal":
        distances[0, 0] = 1.0
    else:
        distances[0, 1] = np.nan
    with pytest.raises(ValueError, match="PCoA"):
        _pcoa(distances)


def _write_covariance(path: Path, tickers: list[str], matrix: np.ndarray) -> None:
    frame = pd.DataFrame(
        [
            {"ticker_i": left, "ticker_j": right, "covariance": matrix[i, j]}
            for i, left in enumerate(tickers)
            for j, right in enumerate(tickers)
        ]
    )
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"return_panel_rows": b"31",
            b"return_panel_start_date": b"2020-01-02",
            b"return_panel_end_date": b"2020-02-14",
        }
    )
    pq.write_table(table, path)


def test_return_covariance_spectrum_matches_dates_by_firms_pca(tmp_path: Path) -> None:
    """Covariance eigenvalues equal unstandardized centered-return PCA variances."""
    returns = (
        np.random.default_rng(10).normal(size=(31, _N_FIRMS))
        * np.arange(1, _N_FIRMS + 1)
        * 1e-3
    )
    expected = PCA(svd_solver="full").fit(returns)
    path = tmp_path / "covariance.parquet"
    tickers = ["E", "D", "C", "B", "A"]
    _write_covariance(path, tickers, np.cov(returns, rowvar=False))
    actual_tickers, covariance = _load_covariance(path)
    eigenvalues, stats = _spectrum(covariance)
    assert actual_tickers == sorted(tickers)
    np.testing.assert_allclose(eigenvalues, expected.explained_variance_, rtol=1e-12)
    assert stats["two_axis_share"] == pytest.approx(
        expected.explained_variance_ratio_[:2].sum()
    )


def test_covariance_roundoff_is_classified_relative_to_variance_scale(
    tmp_path: Path,
) -> None:
    """A roundoff-sized negative axis is numerical zero at a small return scale."""
    scale = 1e-6
    path = tmp_path / "covariance.parquet"
    matrix = np.diag([scale, scale / 2.0, -np.finfo(np.float64).eps * scale])
    _write_covariance(path, ["A", "B", "C"], matrix)
    _tickers, covariance = _load_covariance(path)
    eigenvalues, stats = _spectrum(covariance)
    assert eigenvalues[-1] < 0.0
    assert stats["n_positive_axes"] == _DISPLAY_AXES
    assert stats["n_negative_axes"] == 0
    assert stats["negative_inertia_fraction"] == pytest.approx(0.0)
    assert not np.signbit(stats["negative_inertia_fraction"])


@pytest.mark.parametrize(
    "failure", ["negative", "asymmetric", "missing", "duplicate", "nonfinite"]
)
def test_return_covariance_rejects_invalid_panels(tmp_path: Path, failure: str) -> None:
    """Invalid covariance, including a significant negative eigenvalue, fails."""
    path = tmp_path / "covariance.parquet"
    matrix = np.eye(3)
    if failure == "negative":
        matrix[0, 0] = -1e-4
    elif failure == "asymmetric":
        matrix[0, 1] = 0.5
    elif failure == "nonfinite":
        matrix[0, 1] = np.nan
    _write_covariance(path, ["A", "B", "C"], matrix)
    if failure in {"missing", "duplicate"}:
        frame = pd.read_parquet(path)
        frame = (
            frame.iloc[1:]
            if failure == "missing"
            else pd.concat([frame, frame.iloc[:1]])
        )
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="return covariance"):
        _load_covariance(path)


def _write_distance(
    root: Path, spec: RepresentationAblationSpec, tickers: list[str], matrix: np.ndarray
) -> None:
    directory = root / spec.provider_id / spec.representation_id / "wasserstein_w2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "distances.parquet"
    pd.DataFrame(
        [
            {"item_i": left, "item_j": right, "value": matrix[i, j]}
            for i, left in enumerate(tickers)
            for j, right in enumerate(tickers)
        ]
    ).to_parquet(path, index=False)
    identity = {
        "path": str(path),
        "kind": "parquet",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    metadata = {
        "provider_id": spec.provider_id,
        "representation_id": spec.representation_id,
        "distance_id": "wasserstein_w2",
        "value_semantics": "statistical_distance",
        "output_dtype": "float64",
        "shape": [len(tickers)] * 2,
    }
    summary = {
        "provider_id": spec.provider_id,
        "representation_id": spec.representation_id,
        "distance_id": "wasserstein_w2",
        "item_ids": tickers,
        "metadata": metadata,
        "output_inventory": [identity],
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    write_manifest(
        directory / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "test",
            "lane": "test",
            "phase": "test",
            "protocol_id": "test",
            "code_identity": {"kind": "sha256", "value": "a" * 64},
            "parameter_identity": {"kind": "sha256", "value": "b" * 64},
            "seed_policy": "fixed",
            "upstream_artifacts": [],
            "environment": {},
            "outputs": [identity],
            "risk_class": "low",
            "metadata": metadata,
        },
    )


def _inputs(tmp_path: Path) -> Paper1PcoaOptions:
    tickers = ["E", "D", "C", "B", "A"]
    covariance = tmp_path / "covariance.parquet"
    _write_covariance(
        covariance, tickers, np.diag(np.arange(1, _N_FIRMS + 1, dtype=float))
    )
    universe = tmp_path / "universe.csv"
    pd.DataFrame({"Symbol": tickers, "Sector": "S", "Industry": "I"}).to_csv(
        universe, index=False
    )
    points = np.random.default_rng(12).normal(size=(_N_FIRMS, 3))
    distance_root = tmp_path / "distances"
    for spec in load_extended_representation_ablation_specs(_ROOT / "params.yaml"):
        _write_distance(distance_root, spec, tickers, _distances(points))
    return Paper1PcoaOptions(
        distance_root, covariance, universe, _ROOT / "params.yaml", tmp_path / "output"
    )


def test_all_ladder_cells_share_the_priced_roster_and_full_spectra(
    tmp_path: Path,
) -> None:
    """Validated artifacts yield all fifteen cells and a matched PCA spectrum."""
    options = _inputs(tmp_path)
    run_paper1_pcoa(options)
    result = json.loads((options.output_dir / "results.json").read_text())
    specs = load_extended_representation_ablation_specs(options.params_file)
    assert result["tickers"] == ["A", "B", "C", "D", "E"]
    assert result["n_tickers"] == _N_FIRMS
    assert result["canonical_cell_id"] == "qwen8b_full"
    assert [cell["cell_id"] for cell in result["cells"]] == [
        spec.cell_id for spec in specs
    ]
    assert [cell["family"] for cell in result["cells"]] == ["representation"] * 7 + [
        "vintage"
    ] * 8
    assert result["return_pca"]["n_dates"] == _N_DATES
    assert result["return_pca"]["start_date"] == "2020-01-02"
    spectra = pd.read_parquet(options.output_dir / "spectrum.parquet")
    assert len(spectra) == _SPECTRUM_COUNT * _N_FIRMS
    assert set(spectra.groupby("cell_id").size()) == {_N_FIRMS}
    np.testing.assert_allclose(
        spectra.groupby("cell_id")["positive_inertia_share"].sum(), 1.0
    )
    coords = pd.read_parquet(options.output_dir / "coords_2d.parquet")
    assert len(coords) == _CELL_COUNT * _N_FIRMS
    assert (
        coords.groupby("cell_id")["ticker"].apply(list).tolist()
        == [result["tickers"]] * _CELL_COUNT
    )
    neighbours = pd.read_parquet(options.output_dir / "neighbours.parquet")
    assert len(neighbours) == _N_FIRMS * (_N_FIRMS - 1)
    mds_coordinates = pd.read_parquet(
        options.output_dir / "metric_mds_coords_2d.parquet"
    )
    assert mds_coordinates["ticker"].tolist() == result["tickers"]
    mds_neighbours = pd.read_parquet(
        options.output_dir / "metric_mds_neighbours.parquet"
    )
    np.testing.assert_array_equal(
        mds_neighbours[["anchor", "neighbour", "raw_distance"]],
        neighbours[["anchor", "neighbour", "raw_distance"]],
    )
    mds = result["metric_mds"]
    assert mds["cell_id"] == result["canonical_cell_id"]
    assert mds["n_tickers"] == result["n_tickers"]
    residual = mds_neighbours["raw_distance"] - mds_neighbours["projected_distance"]
    expected_stress = np.linalg.norm(residual) / np.linalg.norm(
        mds_neighbours["raw_distance"]
    )
    assert mds["stress_1"] == pytest.approx(expected_stress)


def test_declared_metric_mds_settings_are_loaded_and_saved(tmp_path: Path) -> None:
    """The materialization records the actual configured fit budget and seed."""
    options = _inputs(tmp_path)
    params = yaml.safe_load(options.params_file.read_text())
    settings = {"n_init": 3, "max_iter": 117, "eps": 1e-7, "random_state": 19}
    params.setdefault("paper1", {})["mds"] = settings
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params))
    run_paper1_pcoa(replace(options, params_file=params_file))
    result = json.loads((options.output_dir / "results.json").read_text())
    assert result["metric_mds"]["settings"] == settings
    assert result["metric_mds"]["n_iter"] <= settings["max_iter"]


@pytest.mark.parametrize("failure", ["missing", "roster", "identity", "metadata"])
def test_incomplete_or_mislabeled_cells_fail_before_outputs(
    tmp_path: Path, failure: str
) -> None:
    """A single cell cannot shrink or relabel the matched population."""
    options = _inputs(tmp_path)
    spec = load_extended_representation_ablation_specs(options.params_file)[-1]
    directory = (
        options.distance_root
        / spec.provider_id
        / spec.representation_id
        / "wasserstein_w2"
    )
    if failure == "missing":
        (directory / "distances.parquet").unlink()
    elif failure == "roster":
        _write_distance(
            options.distance_root,
            spec,
            ["A", "B", "C", "D"],
            _distances(np.arange(4, dtype=float).reshape(-1, 1)),
        )
    elif failure == "identity":
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["provider_id"] = "wrong-provider"
        path.write_text(json.dumps(summary))
    else:
        universe = pd.read_csv(options.universe_csv)
        universe.iloc[:-1].to_csv(options.universe_csv, index=False)
    with pytest.raises((FileNotFoundError, ValueError)):
        run_paper1_pcoa(options)
    assert not options.output_dir.exists()

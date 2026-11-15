"""Publication projections of saved PCoA coordinates, spectra and fidelity."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from pipeline.figures.paper1 import pcoa
from pipeline.figures.paper1.numbers import (
    Paper1NumberPaths,
    render_paper1_numbers,
)
from pipeline.figures.paper1.pcoa import Paper1PcoaPaths, render_paper1_pcoa

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_FIRMS = 24
_PANELS = 2
_SCREE_COMPONENTS = 20
_SCREE_PEAK_CAP = 0.6
_FONT_FLOOR = 9
_BASE = (
    ("qwen8b_full", "Qwen3 8B", "qwen3-embedding-8b", 4096),
    ("qwen4b_full", "Qwen3 4B", "qwen3-embedding-4b", 2560),
    ("qwen8b_1024", "Qwen3 8B", "qwen3-embedding-8b", 1024),
    ("qwen4b_1024", "Qwen3 4B", "qwen3-embedding-4b", 1024),
    ("qwen8b_256", "Qwen3 8B", "qwen3-embedding-8b", 256),
    ("qwen8b_64", "Qwen3 8B", "qwen3-embedding-8b", 64),
    ("bge_large_full", "BGE-large", "bge-large-en-v1.5", 1024),
)


def _saved_analysis(root: Path) -> Paper1PcoaPaths:
    analysis = root / "analysis"
    analysis.mkdir()
    tickers = [f"T{index:02d}" for index in range(_FIRMS)]
    roster = [
        *_BASE,
        *(
            (f"chronobert_{year}", f"ChronoBERT {year}", f"chronobert-{year}", 768)
            for year in range(2017, 2025)
        ),
    ]
    cells = []
    coordinates = []
    spectra = []
    for index, (cell_id, label, provider, width) in enumerate(roster):
        shares = np.zeros(_FIRMS)
        shares[:2] = [0.4 + index * 0.005, 0.25]
        tail = np.arange(_FIRMS - 2, 0, -1, dtype=float)
        shares[2:] = tail / tail.sum() * (1 - shares[:2].sum())
        negative = 0.02 + index * 0.001
        cells.append(
            {
                "cell_id": cell_id,
                "label": label,
                "provider_id": provider,
                "representation_id": f"{provider}-{width}-unit",
                "effective_dimension": width,
                "family": "representation" if index < len(_BASE) else "vintage",
                "stress_1": 0.234,
                "shepard_pearson": 0.8,
                "shepard_spearman": 0.75,
                "top5_neighbour_recall_mean": 0.6,
                "exact_nearest_neighbour_preserved": 9,
                "two_axis_share": float(shares[:2].sum()),
                "negative_inertia_fraction": negative,
            }
        )
        coordinates.extend(
            {
                "cell_id": cell_id,
                "ticker": ticker,
                "pcoa_1": float(position),
                "pcoa_2": float(position % 3),
                "sector": "Technology" if position % 2 else "Healthcare",
                "industry": "Example",
            }
            for position, ticker in enumerate(tickers)
        )
        spectra.extend(
            {
                "cell_id": cell_id,
                "component": component,
                "eigenvalue": float(share),
                "positive_inertia_share": float(share),
                "cumulative_positive_inertia_share": float(shares[:component].sum()),
            }
            for component, share in enumerate(shares, 1)
        )
    return_shares = np.r_[0.5, 0.2, np.full(_FIRMS - 2, 0.3 / (_FIRMS - 2))]
    spectra.extend(
        {
            "cell_id": "return_pca",
            "component": component,
            "eigenvalue": float(share),
            "positive_inertia_share": float(share),
            "cumulative_positive_inertia_share": float(return_shares[:component].sum()),
        }
        for component, share in enumerate(return_shares, 1)
    )
    summary = {
        "schema_version": 1,
        "method": "PCoA",
        "n_tickers": len(tickers),
        "tickers": tickers,
        "canonical_cell_id": "qwen8b_full",
        "cells": cells,
        "metric_mds": {
            "method": "metric MDS",
            "cell_id": "qwen8b_full",
            "stress_1": 0.21,
            "scale_fitted_stress_1": 0.20,
            "optimal_distance_scale": 1.01,
            "shepard_pearson": 0.85,
            "shepard_spearman": 0.81,
            "top5_neighbour_recall_mean": 0.65,
            "exact_nearest_neighbour_preserved": 10,
        },
        "return_pca": {"method": "PCA"},
    }
    (analysis / "results.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(coordinates).sample(frac=1, random_state=42).to_parquet(
        analysis / "coords_2d.parquet", index=False
    )
    pd.DataFrame(spectra).sample(frac=1, random_state=47).to_parquet(
        analysis / "spectrum.parquet", index=False
    )
    pd.DataFrame(
        {
            "ticker": tickers,
            "mds_1": np.arange(_FIRMS) * 0.7,
            "mds_2": np.arange(_FIRMS) % 4,
            "sector": ["Technology" if i % 2 else "Healthcare" for i in range(_FIRMS)],
            "industry": "Example",
        }
    ).sample(frac=1, random_state=51).to_parquet(
        analysis / "metric_mds_coords_2d.parquet", index=False
    )
    # Deliberately distinct from coordinate-derived distances: the renderer must
    # use the saved dyads rather than silently recomputing projection statistics.
    neighbours = pd.DataFrame(
        [
            {
                "anchor": anchor,
                "neighbour": neighbour,
                "raw_distance": float(abs(i - j) + 1),
                "projected_distance": float(abs(i - j) * 0.3 + 0.2),
            }
            for i, anchor in enumerate(tickers)
            for j, neighbour in enumerate(tickers)
            if i != j
        ]
    )
    neighbours.to_parquet(analysis / "neighbours.parquet", index=False)
    neighbours.assign(
        projected_distance=neighbours["projected_distance"] * 2.0
    ).to_parquet(analysis / "metric_mds_neighbours.parquet", index=False)
    return Paper1PcoaPaths(analysis, root / "images", root / "generated/spectrum.tex")


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, Figure]:
    captured = {}
    subplots = plt.subplots

    def create(
        nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float]
    ) -> tuple[Figure, Axes | np.ndarray]:
        fig, axes = subplots(nrows, ncols, figsize=figsize)
        FigureCanvasAgg(fig)
        return fig, axes

    def save(fig: Figure, path: Path) -> None:
        captured[path.name] = fig
        plt.close(fig)

    monkeypatch.setattr(pcoa.plt, "subplots", create)
    monkeypatch.setattr(pcoa, "fit_to_width", lambda *_args: None)
    monkeypatch.setattr(pcoa, "_save", save)
    return captured


def test_saved_coordinates_and_dyads_reach_the_display_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinates, correlations and Shepard points come from saved artifacts."""
    paths = _saved_analysis(tmp_path)
    figures = _capture_figures(monkeypatch)
    render_paper1_pcoa(paths)
    assert set(figures) == {
        "metric_mds_sector_plot.pgf",
        "metric_mds_shepard_diagnostic.pgf",
        "pcoa_scree.pgf",
        "pcoa_vintage_scree.pgf",
    }
    neighbours = pd.read_parquet(paths.analysis_dir / "metric_mds_neighbours.parquet")
    dyads = neighbours.loc[neighbours["anchor"] < neighbours["neighbour"]]
    mds_scatter = figures["metric_mds_sector_plot.pgf"].axes[0]
    expected_mds = np.column_stack((np.arange(_FIRMS) * 0.7, np.arange(_FIRMS) % 4))
    assert np.asarray(mds_scatter.collections[0].get_offsets()) == pytest.approx(
        expected_mds
    )
    assert mds_scatter.get_xlabel() == "Metric MDS axis 1"
    assert mds_scatter.get_aspect() == pytest.approx(1.0)
    mds_diagnostic = figures["metric_mds_shepard_diagnostic.pgf"].axes[0]
    assert np.asarray(mds_diagnostic.collections[0].get_offsets()) == pytest.approx(
        dyads[["raw_distance", "projected_distance"]].to_numpy()
    )
    assert r"\rho_S=0.810" in mds_diagnostic.get_title()
    table = paths.spectrum_table.read_text(encoding="utf-8")
    assert r"\label{tab:pcoa-spectrum}" in table
    assert table.count(" & ") == 3 * 16
    assert "4,096 & 65.0 & 2.0" in table


def test_every_tuple_has_a_curve_and_the_pca_reference_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both ladders use complete saved spectra and one shared return-PCA curve."""
    paths = _saved_analysis(tmp_path)
    figures = _capture_figures(monkeypatch)
    render_paper1_pcoa(paths)
    summary = json.loads((paths.analysis_dir / "results.json").read_text())
    spectra = pd.read_parquet(paths.analysis_dir / "spectrum.parquet")
    references = []
    for filename, family in (
        ("pcoa_scree.pgf", "representation"),
        ("pcoa_vintage_scree.pgf", "vintage"),
    ):
        figure = figures[filename]
        cells = [cell for cell in summary["cells"] if cell["family"] == family]
        assert len(figure.axes) == _PANELS
        assert all(len(ax.lines) == len(cells) + 1 for ax in figure.axes)
        for cell, first, cumulative in zip(
            cells, figure.axes[0].lines[:-1], figure.axes[1].lines[:-1], strict=True
        ):
            rows = spectra.loc[spectra["cell_id"] == cell["cell_id"]].sort_values(
                "component"
            )
            assert (
                first.get_label()
                == f"{cell['label']} / {cell['effective_dimension']:,}"
            )
            assert first.get_ydata() == pytest.approx(
                rows["positive_inertia_share"].iloc[:20]
            )
            assert cumulative.get_ydata() == pytest.approx(
                rows["cumulative_positive_inertia_share"]
            )
            assert np.asarray(first.get_xdata()).size == _SCREE_COMPONENTS
            assert np.asarray(cumulative.get_xdata()).size == _FIRMS
        reference = figure.axes[0].lines[-1]
        assert reference.get_label() == "Return PCA"
        references.append(np.asarray(reference.get_ydata()))
        assert figure.axes[0].get_ylim()[1] < _SCREE_PEAK_CAP
        assert figure.axes[1].get_ylim() == (0, 1)
        assert all(
            float(text.get_fontsize()) >= _FONT_FLOOR for text in figure.findobj(Text)
        )
    assert np.array_equal(references[0], references[1])


@pytest.mark.parametrize("failure", ["missing_cell", "duplicate_component"])
def test_incomplete_spectrum_inventory_fails_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """The renderer cannot silently omit a comparison tuple or repeat an axis."""
    paths = _saved_analysis(tmp_path)
    spectra = pd.read_parquet(paths.analysis_dir / "spectrum.parquet")
    if failure == "missing_cell":
        spectra = spectra.loc[spectra["cell_id"] != "chronobert_2024"]
    else:
        spectra = pd.concat([spectra, spectra.iloc[:1]], ignore_index=True)
    spectra.to_parquet(paths.analysis_dir / "spectrum.parquet", index=False)
    figures = _capture_figures(monkeypatch)
    with pytest.raises(ValueError, match="PCoA spectra must"):
        render_paper1_pcoa(paths)
    assert not figures
    assert not paths.spectrum_table.exists()


@pytest.mark.parametrize("failure", ["cohort", "cell"])
def test_metric_mds_must_match_the_saved_canonical_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """An unrelated MDS fit cannot be displayed beside the retained spectra."""
    paths = _saved_analysis(tmp_path)
    if failure == "cohort":
        filename = paths.analysis_dir / "metric_mds_coords_2d.parquet"
        coordinates = pd.read_parquet(filename)
        coordinates.iloc[:-1].to_parquet(filename, index=False)
    else:
        filename = paths.analysis_dir / "results.json"
        summary = json.loads(filename.read_text())
        summary["metric_mds"]["cell_id"] = "chronobert_2024"
        filename.write_text(json.dumps(summary))
    figures = _capture_figures(monkeypatch)
    with pytest.raises(ValueError, match="metric MDS must retain"):
        render_paper1_pcoa(paths)
    assert not figures


@pytest.mark.parametrize("missing_correlation", [False, True])
def test_numbers_load_the_canonical_pcoa_result_as_fraction_bindings(
    tmp_path: Path, *, missing_correlation: bool
) -> None:
    """Bindings select the named canonical tuple, preserving undefined statistics."""
    paths = _saved_analysis(tmp_path)
    summary_path = paths.analysis_dir / "results.json"
    summary = json.loads(summary_path.read_text())
    summary["cells"].reverse()
    canonical = next(
        cell for cell in summary["cells"] if cell["cell_id"] == "qwen8b_full"
    )
    if missing_correlation:
        canonical["shepard_spearman"] = None
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    absent = tmp_path / "absent"
    numbers_path = tmp_path / "numbers.tex"
    render_paper1_numbers(
        Paper1NumberPaths(
            numbers_path=numbers_path,
            params_path=absent,
            corpus_summary=absent,
            dyadic_confound_results=absent,
            w2_permanova_results=absent,
            diagnostics_results=absent,
            sample_descriptives_results=absent,
            pcoa_summary=summary_path,
            neighbour_outcome_split_results=absent,
            covariance_matrix=absent,
            oos_covariance_matrix=None,
            oos_dyadic_results=None,
        )
    )
    emitted = numbers_path.read_text()
    assert r"\ppDeclareValue{pcoa-two-axis-share}{0.650000}" in emitted
    assert r"\ppDeclareValue{pcoa-negative-inertia-share}{0.020000}" in emitted
    assert r"\ppDeclareValue{pcoa-exact-nearest-preserved}{9}" in emitted
    expected = r"\ppmissing" if missing_correlation else "0.750000"
    assert rf"\ppDeclareValue{{pcoa-shepard-spearman}}{{{expected}}}" in emitted
    assert r"\ppDeclareValue{metric-mds-stress-1}{0.210000}" in emitted
    assert r"\ppDeclareValue{metric-mds-top5-neighbour-recall}{0.650000}" in emitted
    assert r"\ppDeclareValue{metric-mds-exact-nearest-preserved}{10}" in emitted
    assert "{mds-" not in emitted

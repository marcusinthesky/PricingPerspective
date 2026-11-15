"""Tests for the Paper 1 nearest-neighbour outcome-split contract."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pipeline.figures.paper1.neighbour_outcome_split import (
    MANUSCRIPT_ANCHOR,
    _displayed_values,
    neighbour_outcome_split_value_records,
)
from pipeline.io.values import PublicationValue
from pipeline.stages.papers.paper1.neighbour_outcome_split import (
    MIN_SECTOR_PAIR_MEMBERS,
    SELECTION_GEOMETRY_ID,
    SELECTION_WINDOW_ID,
    TRAIN_END,
    TRAIN_START,
    _comparative_percentile,
    _EligibleUniverse,
    _nearest_same_sector_pairs,
    _outcome_metrics,
    _partition_sector_members,
    _resolve_geometry_spec,
    _select_sector_pairs,
    _SelectionGeometry,
)

W2_DISPLAY_PRECISION = 3
REPO_ROOT = Path(__file__).resolve().parents[5]
UNIVERSE_CSV = REPO_ROOT / "data/universe.csv"


def _geometry(item_ids: tuple[str, ...], metric: np.ndarray) -> _SelectionGeometry:
    return _SelectionGeometry(
        item_ids=item_ids,
        metric=metric,
        article_counts=dict.fromkeys(item_ids, 200),
        component_summary={},
    )


def _universe(sectors: dict[str, str]) -> _EligibleUniverse:
    return _EligibleUniverse(
        symbols=sorted(sectors),
        sectors=sectors,
        return_paths={},
        exclusions=[],
    )


def _pair_metrics(
    universe: _EligibleUniverse, geometry: _SelectionGeometry
) -> pd.DataFrame:
    position = {symbol: index for index, symbol in enumerate(geometry.item_ids)}
    rows: list[dict[str, object]] = []
    for symbol1, symbol2 in combinations(universe.symbols, 2):
        distance = float(geometry.metric[position[symbol1], position[symbol2]])
        rows.append(
            {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "sector1": universe.sectors[symbol1],
                "sector2": universe.sectors[symbol2],
                "w2_distance": distance,
                "rms_cumulative_log_gap": distance,
                "terminal_absolute_log_gap": distance,
                "n_preperiod1": geometry.article_counts[symbol1],
                "n_preperiod2": geometry.article_counts[symbol2],
            }
        )
    return pd.DataFrame(rows)


def test_outcome_metrics_and_comparative_percentile_are_deterministic() -> None:
    """Verify outcome gaps and comparative ranks are deterministic."""
    rms, terminal = _outcome_metrics(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 1.0])
    )
    assert rms == pytest.approx(math.sqrt(2.0 / 3.0))
    assert terminal == pytest.approx(1.0)

    pairs = pd.DataFrame(
        {
            "symbol1": ["AAA", "AAA", "BBB"],
            "symbol2": ["BBB", "CCC", "CCC"],
            "rms_cumulative_log_gap": [0.3, 0.1, 0.2],
        }
    )
    assert _comparative_percentile(pairs, "BBB", "CCC") == (2, 3, 50.0)
    singleton = pairs.iloc[[0]]
    assert _comparative_percentile(singleton, "AAA", "BBB") == (1, 1, None)


def test_nearest_same_sector_pairs_ignores_cross_sector_neighbours() -> None:
    """The nearest *same-sector* pair wins even when a closer neighbour exists."""
    items = ("AAA", "BBB", "CCC", "DDD")
    # AAA's globally nearest neighbour is CCC (0.1) but CCC is in another
    # sector; within sector "S" the nearest pair must be AAA-BBB.
    metric = np.array(
        [
            [0.0, 0.4, 0.1, 0.9],
            [0.4, 0.0, 0.8, 0.2],
            [0.1, 0.8, 0.0, 0.7],
            [0.9, 0.2, 0.7, 0.0],
        ]
    )
    sectors = {"AAA": "S", "BBB": "S", "CCC": "T", "DDD": "T"}
    nearest = _nearest_same_sector_pairs(_universe(sectors), _geometry(items, metric))
    assert nearest == {"S": ("AAA", "BBB"), "T": ("CCC", "DDD")}


def test_nearest_same_sector_pairs_breaks_ties_by_symbol_order() -> None:
    """Equal distances resolve to the preregistered (symbol1, symbol2) order."""
    items = ("AAA", "BBB", "CCC")
    metric = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    sectors = dict.fromkeys(items, "S")
    nearest = _nearest_same_sector_pairs(_universe(sectors), _geometry(items, metric))
    assert nearest == {"S": ("AAA", "BBB")}


def test_sector_selection_derives_pairable_and_singleton_sectors() -> None:
    """Selection follows the supplied taxonomy rather than a hard-coded roster."""
    items = ("AAA", "BBB", "CCC", "DDD", "EEE")
    metric = np.array(
        [
            [0.0, 0.1, 0.8, 0.9, 0.7],
            [0.1, 0.0, 0.6, 0.5, 0.8],
            [0.8, 0.6, 0.0, 0.2, 0.4],
            [0.9, 0.5, 0.2, 0.0, 0.3],
            [0.7, 0.8, 0.4, 0.3, 0.0],
        ]
    )
    universe = _universe(
        {"AAA": "Alpha", "BBB": "Alpha", "CCC": "Beta", "DDD": "Beta", "EEE": "Solo"}
    )
    geometry = _geometry(items, metric)

    selection = _select_sector_pairs(
        universe, geometry, _pair_metrics(universe, geometry)
    )

    assert [row["sector"] for row in selection.selected_pairs] == ["Alpha", "Beta"]
    assert selection.singleton_sectors == [{"sector": "Solo", "members": ["EEE"]}]


@pytest.mark.skipif(
    not UNIVERSE_CSV.is_file(),
    reason=f"{UNIVERSE_CSV} is DVC-tracked and absent from CI checkouts",
)
def test_current_universe_sector_taxonomy_is_governed() -> None:
    """The 100-firm roster has seven pairable sectors and two named singletons."""
    frame = pd.read_csv(UNIVERSE_CSV, dtype=str)
    universe = _universe(dict(zip(frame["Symbol"], frame["Sector"], strict=True)))
    sector_members = _partition_sector_members(universe)
    pairable = {
        sector
        for sector, members in sector_members.items()
        if len(members) >= MIN_SECTOR_PAIR_MEMBERS
    }
    singletons = {
        sector: members
        for sector, members in sector_members.items()
        if len(members) < MIN_SECTOR_PAIR_MEMBERS
    }

    assert pairable == {
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Healthcare",
        "Industrials",
        "Technology",
        "Utilities",
    }
    assert singletons == {
        "Energy": ["FANG"],
        "Financial Services": ["PYPL"],
    }


def test_selection_distance_is_the_shared_windowed_artifact() -> None:
    """The protocol dates and the registry window must not be able to drift."""
    geometry = _resolve_geometry_spec(REPO_ROOT / "params.yaml")
    assert geometry.distance_id == SELECTION_GEOMETRY_ID
    assert geometry.sample_design == "point_in_time"
    assert geometry.sample_window == SELECTION_WINDOW_ID
    # The point-in-time W2 distance is the governed selection artifact.
    assert geometry.family == "wasserstein"
    assert geometry.estimator == "balanced_wasserstein_2"
    assert geometry.ground_distance == "euclidean_chord"
    assert geometry.exponent == pytest.approx(1.0)


def test_paper1_and_paper3_read_one_window_not_two() -> None:
    """The 2020 PIT vintage IS Paper 1's selection window, not a copy of it."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    grid = {
        entry["vintage"]: entry["distance_id"]
        for entry in params["paper3"]["pit_vintage_grid"]
    }
    assert grid[TRAIN_END.date().isoformat()] == SELECTION_GEOMETRY_ID
    window = params["typed_analysis"]["windows"][SELECTION_WINDOW_ID]
    assert str(window["start"]) == TRAIN_START.date().isoformat()
    assert str(window["end"]) == TRAIN_END.date().isoformat()


def test_registered_values_match_renderer_format() -> None:
    """Keep registered values aligned with their displayed representation."""
    summary = {
        "n_eligible_firms": 100,
        "n_pair_metrics": 4950,
        "n_selected_pairs": 1,
        "n_evaluation_dates": 503,
        "minimum_preperiod_embeddings": 140,
        "selected_pairs": [
            {
                "sector": "Industrials",
                "w2_distance": 0.123456,
                "rms_cumulative_log_gap": 0.234567,
                "terminal_absolute_log_gap": 0.345678,
                "within_sector_rms_gap_percentile": None,
            }
        ],
    }
    records = [
        record
        for record in neighbour_outcome_split_value_records(summary)
        if isinstance(record, PublicationValue)
    ]
    bindings = {record.key: record for record in records}
    displayed = {
        row["binding_key"]: row["formatted_value"] for row in _displayed_values(summary)
    }
    assert "neighbour-holdout-industrials-percentile" not in displayed
    assert displayed["neighbour-holdout-industrials-w2"] == "0.123"
    assert (
        bindings["neighbour-holdout-industrials-w2"].precision == W2_DISPLAY_PRECISION
    )


def test_dvc_has_isolated_empirical_and_render_leaves() -> None:
    """Keep empirical computation and rendering in isolated DVC leaves."""
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    empirical = dvc["stages"]["p1_neighbour_outcome_split"]
    render = dvc["stages"]["p1_neighbour_outcome_split_render"]
    empirical_outs = {
        item if isinstance(item, str) else next(iter(item))
        for item in empirical["outs"]
    }
    render_outs = {
        item if isinstance(item, str) else next(iter(item)) for item in render["outs"]
    }
    assert "${p1}/neighbour_outcome_split/pair_paths.parquet" in empirical_outs
    # t80: unconsumed outs. Re-adding either needs a declared consumer first.
    assert "${p1}/neighbour_outcome_split/pair_metrics.parquet" not in empirical_outs
    assert "${p1}/neighbour_outcome_split/returns.parquet" not in empirical_outs
    assert (
        "${p1_tex}/generated/paper1_neighbour_outcome_split.interpretation.json"
    ) in render_outs
    assert "${p1_tex}/images/neighbour_holdout_paths.pgf" in render_outs


def test_split_selects_on_the_component_not_on_raw_embeddings() -> None:
    """The DAG edge must run through the shared component, not the embeddings."""
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    deps = dvc["stages"]["p1_neighbour_outcome_split"]["deps"]
    assert any("typed_distances_windowed" in dep for dep in deps)
    assert not any(dep.startswith("${shared}/embeddings") for dep in deps)


def test_live_contract_registers_manuscript_anchor() -> None:
    """Keep the live contract attached to its manuscript section.

    The exploratory main-text subsection owns the published interpretation.
    """
    assert MANUSCRIPT_ANCHOR == "sec:exploratory-geometry-outcomes"

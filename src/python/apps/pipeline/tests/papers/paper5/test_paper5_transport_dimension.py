"""Tests for Paper 5's normalized Matryoshka W1/W2 mechanism ladder."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml
from pipeline._kernels.typed_geometry import EmbeddingCloud, apply_representation
from pipeline.cli.commands.p5 import transport_dimension_ladder as adapter
from pipeline.figures.paper5.transport_dimension import (
    Paper5TransportDimensionTableError,
    render_paper5_transport_dimension_table,
)
from pipeline.io.typed_analysis import RepresentationSpec
from pipeline.stages.papers.paper5 import transport_dimension as stage
from pipeline.stages.papers.paper5.transport_dimension import (
    Paper5TransportDimensionConfig,
    Paper5TransportDimensionError,
    Paper5TransportDimensionPaths,
    _assignment_geometry,
    _registered_cell_ids,
)
from scipy.spatial.distance import cdist as scipy_cdist
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[7]
FIXTURE_ATOMS = 3


def test_cli_passes_explicit_embedding_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ladder reads embeddings from its given path, outside registry identity."""
    seen: list[
        tuple[Paper5TransportDimensionPaths, Paper5TransportDimensionConfig]
    ] = []
    monkeypatch.setattr(
        stage,
        "run_paper5_transport_dimension_ladder",
        lambda paths, config: seen.append((paths, config)),
    )
    for name in ("embeddings", "returns", "oos", "barycentres"):
        (tmp_path / name).mkdir()
    params = tmp_path / "params.yaml"
    params.write_text("{}\n", encoding="utf-8")
    config = Paper5TransportDimensionConfig()
    args = [
        "paper5-transport-dimension-ladder",
        "--embeddings-dir", str(tmp_path / "embeddings"),
        "--returns-dir", str(tmp_path / "returns"),
        "--oos-returns-dir", str(tmp_path / "oos"),
        "--barycentres-root", str(tmp_path / "barycentres"),
        "--params-file", str(params),
        "--output-dir", str(tmp_path / "output"),
    ]  # fmt: skip
    for field, value in asdict(config).items():
        args.extend((f"--{field.replace('_', '-')}", str(value)))

    result = CliRunner().invoke(adapter.commands, args)

    assert result.exit_code == 0, result.output
    assert seen == [
        (
            Paper5TransportDimensionPaths(
                embeddings_dir=tmp_path / "embeddings",
                returns_dir=tmp_path / "returns",
                oos_returns_dir=tmp_path / "oos",
                barycentres_root=tmp_path / "barycentres",
                params_file=params,
                output_dir=tmp_path / "output",
            ),
            config,
        )
    ]


def _cloud(item_id: str, values: list[list[float]]) -> EmbeddingCloud:
    """Build one unit-sphere fixture cloud."""
    return EmbeddingCloud(
        item_id=item_id,
        values=np.asarray(values, dtype=np.float64),
        provider_id="test-provider",
        representation_id="test-representation",
    )


def test_matryoshka_prefix_is_truncated_before_renormalization() -> None:
    """A shorter prefix is re-normalized instead of inheriting the full-row norm."""
    raw = np.asarray([[3.0, 4.0, 12.0], [4.0, -3.0, 12.0]], dtype=np.float64)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    representation = RepresentationSpec(
        representation_id="test-prefix-2-unit",
        provider_id="test-provider",
        transform="l2_normalize_rows",
        normalization="unit_rows",
        dimension=2,
    )

    transformed = apply_representation(raw, representation)
    expected = raw[:, :2] / np.linalg.norm(raw[:, :2], axis=1, keepdims=True)

    np.testing.assert_allclose(transformed, expected, atol=1.0e-15, rtol=0.0)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0)
    assert not np.allclose(transformed, raw[:, :2])


def test_assignment_geometry_enumerates_costs_and_detects_order_difference() -> None:
    """The census and overlap use the same chord matrix for d and d-squared."""
    left = _cloud(
        "left",
        [
            [-0.5886162668, -0.4564652797, -0.6672108654],
            [0.6823845914, -0.0777928624, -0.7268421699],
            [0.2172300301, 0.4400704271, -0.8712916465],
        ],
    )
    right = _cloud(
        "right",
        [
            [-0.2496224145, -0.9535948369, -0.1683613291],
            [-0.9994400770, 0.0160373107, -0.0293655789],
            [-0.2621519286, -0.9026680176, -0.3412723492],
        ],
    )
    expected_cost = scipy_cdist(left.values, right.values)

    pairs, summary = _assignment_geometry((left, right), cell_id="fixture", dimension=3)

    assert len(pairs) == 1
    assert summary["article_distance_count"] == expected_cost.size
    assert summary["assignment_atom_count"] == FIXTURE_ATOMS
    assert summary["assignment_overlap"] == pytest.approx(1.0 / FIXTURE_ATOMS)
    assert summary["chord_distance_mean"] == pytest.approx(expected_cost.mean())
    assert summary["chord_distance_population_sd"] == pytest.approx(
        expected_cost.std(ddof=0)
    )
    w1_cost = cast("float", pairs.loc[0, "w1_mean_matched_chord_cost"])
    w1_cross_cost = cast("float", pairs.loc[0, "w1_cost_under_w2_assignment"])
    w2_cost = cast("float", pairs.loc[0, "w2_mean_matched_squared_cost"])
    w2_cross_cost = cast("float", pairs.loc[0, "w2_cost_under_w1_assignment"])
    assert w1_cost <= w1_cross_cost
    assert w2_cost <= w2_cross_cost


def test_switched_assignment_tail_uses_one_scale_and_exact_switch_accounting() -> None:
    """The threshold is assignment-free and switched sums cover the reassignments."""
    left = _cloud(
        "left",
        [
            [-0.5886162668, -0.4564652797, -0.6672108654],
            [0.6823845914, -0.0777928624, -0.7268421699],
            [0.2172300301, 0.4400704271, -0.8712916465],
        ],
    )
    right = _cloud(
        "right",
        [
            [-0.2496224145, -0.9535948369, -0.1683613291],
            [-0.9994400770, 0.0160373107, -0.0293655789],
            [-0.2621519286, -0.9026680176, -0.3412723492],
        ],
    )
    cost = scipy_cdist(left.values, right.values)

    pairs, summary = _assignment_geometry((left, right), cell_id="fixture", dimension=3)

    switched = cast("int", pairs.loc[0, "switched_atom_count"])
    matched = cast("int", pairs.loc[0, "assignment_match_count"])
    assert switched == FIXTURE_ATOMS - matched
    assert summary["switched_atom_total"] == switched

    # The threshold comes from the complete cost matrix, not from either match set.
    assert cast("float", pairs.loc[0, "matched_tail_threshold"]) == pytest.approx(
        float(np.quantile(cost, 0.95))
    )

    # Both orders are scored on chord distance, so each maximum is attained by
    # some matched entry of the shared cost matrix.
    for key in ("w1_matched_chord_max", "w2_matched_chord_max"):
        assert cast("float", pairs.loc[0, key]) <= cost.max()
        assert cast("float", pairs.loc[0, key]) >= cost.min()

    # Pooled switched-atom mean is the switched sum over the switched count.
    switched_difference = (
        cast("float", pairs.loc[0, "w2_switched_chord_sum"])
        - cast("float", pairs.loc[0, "w1_switched_chord_sum"])
    ) / switched
    assert summary["switched_matched_chord_w2_minus_w1"] == pytest.approx(
        switched_difference
    )


def test_longest_match_shares_partition_the_firm_pairs() -> None:
    """Lower, higher, and tied longest matches exhaust the pair census exactly."""
    left = _cloud(
        "left",
        [
            [-0.5886162668, -0.4564652797, -0.6672108654],
            [0.6823845914, -0.0777928624, -0.7268421699],
            [0.2172300301, 0.4400704271, -0.8712916465],
        ],
    )
    right = _cloud(
        "right",
        [
            [-0.2496224145, -0.9535948369, -0.1683613291],
            [-0.9994400770, 0.0160373107, -0.0293655789],
            [-0.2621519286, -0.9026680176, -0.3412723492],
        ],
    )

    _pairs, summary = _assignment_geometry(
        (left, right), cell_id="fixture", dimension=3
    )

    lower = cast("float", summary["max_reduction_pair_share"])
    higher = cast("float", summary["max_increase_pair_share"])
    assert 0.0 <= lower + higher <= 1.0
    assert cast("float", summary["max_reduction_given_lower"]) <= 0.0
    assert cast("float", summary["max_increase_given_higher"]) >= 0.0


def test_switched_assignment_summary_rejects_a_fully_agreeing_census() -> None:
    """With no reassignment anywhere the diagnostic has no population to describe."""
    identical = _cloud("identical", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shifted = _cloud("shifted", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(Paper5TransportDimensionError, match="assigns any target atom"):
        _assignment_geometry((identical, shifted), cell_id="fixture", dimension=3)


def test_registered_ladder_rejects_any_distance_rescaling() -> None:
    """Scale manipulation is outside the authored mechanism experiment."""
    ladder: dict[str, object] = {
        "cell_ids": ["full", "d1024", "d256", "d64"],
        "expected_dimensions": [4096, 1024, 256, 64],
        "ground_metric": "euclidean_chord",
        "distance_rescaling": "none",
        "factor_model": "ff5_mom",
    }
    assert _registered_cell_ids(ladder, Paper5TransportDimensionConfig()) == [
        "full",
        "d1024",
        "d256",
        "d64",
    ]

    ladder["distance_rescaling"] = "median"
    with pytest.raises(Paper5TransportDimensionError, match="does not admit"):
        _registered_cell_ids(ladder, Paper5TransportDimensionConfig())


def test_dvc_graph_materializes_the_four_consumed_w1_dimensions() -> None:
    """The shared roster schedules the canonical W1 field and three prefixes."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = cast("dict[str, dict[str, object]]", dvc["stages"])

    w1_cells = [
        cell
        for cell in params["barycentre_specs"].values()
        if cell["arm_id"] == "wasserstein_w1_loo"
    ]
    assert [cell["representation_id"] for cell in w1_cells] == [
        "qwen3-embedding-8b-unit",
        "qwen3-embedding-8b-1024-unit",
        "qwen3-embedding-8b-256-unit",
        "qwen3-embedding-8b-64-unit",
    ]
    assert all(cell["provider_id"] == "qwen3-embedding-8b" for cell in w1_cells)
    ladder = params["paper5"]["transport_dimension_ladder"]
    assert ladder["cell_ids"] == [
        "qwen8b_full",
        "qwen8b_1024",
        "qwen8b_256",
        "qwen8b_64",
    ]
    assert ladder["distance_rescaling"] == "none"
    assert stages["barycentres"]["foreach"] == "${barycentre_specs}"
    analysis = stages["p5_transport_dimension_ladder"]
    assert "paper5-transport-dimension-ladder" in str(analysis["cmd"])
    assert "--distance-rescaling" not in str(analysis["cmd"])
    table = stages["p5_transport_dimension_ladder_render"]
    assert "render-paper5-transport-dimension-ladder" in str(table["cmd"])


def test_transport_dimension_table_reports_native_scale_stagewise_ladder(
    tmp_path: Path,
) -> None:
    """The publication table retains the geometry and claim-boundary disclosures."""
    cells = []
    for index, dimension in enumerate((4096, 1024, 256, 64)):
        cells.append(
            {
                "effective_dimension": dimension,
                "representation_transform": "truncate_then_l2_normalize_rows",
                "ground_metric": "euclidean_chord",
                "distance_rescaling": "none",
                "article_distance_count": 81_100_800,
                "firm_pair_count": 4_950,
                "assignment_atom_count": 633_600,
                "chord_distance_cv": 0.01 + index * 0.01,
                "assignment_overlap": 0.95 - index * 0.10,
                "off_diagonal_weight_correlation": 0.99 - index * 0.05,
                "spatial_lag_correlation": 0.999 - index * 0.01,
                "held_out_score_w2_minus_w1": -0.0001 + index * 0.00005,
            }
        )
    payload = {
        "analysis_id": "paper5-matryoshka-w1-w2-mechanism-v1",
        "design": {"distance_rescaling": "none"},
        "cells": cells,
    }
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "table.tex"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    render_paper5_transport_dimension_table(results_path, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "4,096 & 1.00\\% & 95.0\\%" in rendered
    assert "81,100,800 article-pair costs" in rendered
    # Ground-cost and sign-convention disclosures the caption must carry; the
    # "not a test of investment-cost convexity" boundary sentence was removed
    # editorially in 4133ac798 from both the renderer and the published table.
    assert "unscaled Euclidean chord distance" in rendered
    assert "633,600 matched target atoms" in rendered
    assert "positive values favour W2" in rendered


def test_transport_dimension_table_rejects_rescaled_cell(tmp_path: Path) -> None:
    """A renderer cannot silently publish a transformed ground cost."""
    empty_cells: list[dict[str, object]] = []
    payload = {
        "analysis_id": "paper5-matryoshka-w1-w2-mechanism-v1",
        "design": {"distance_rescaling": "median"},
        "cells": empty_cells,
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Paper5TransportDimensionTableError, match="forbids"):
        render_paper5_transport_dimension_table(results_path, tmp_path / "table.tex")

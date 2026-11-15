"""Tests for Paper 5's shared field-operator builders and peer-definition rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pipeline.stages.papers.paper5.baskets import equal_support_weights
from pipeline.stages.papers.paper5.field_operators import (
    FieldOperatorIdentity,
    FieldOperatorInputs,
    FieldOperatorPaths,
    co_mention_operator,
    equal_support_operator,
    load_field_operator_inputs,
    median_bandwidth,
    rbf_median_operator,
    validate_operator,
)
from pipeline.stages.papers.paper5.peer_definitions import (
    PeerDefinitionConfig,
    caption_facts,
    peer_definition_operators,
    peer_definition_rows,
)

_REPO = Path(__file__).resolve().parents[7]
_SHARED = _REPO / "data" / "shared"
_UNIT = "qwen3-embedding-8b/qwen3-embedding-8b-unit"
_PEER_ROWS = _REPO / "data/papers/paper5/economic_value/peer_rows.parquet"
_SPARSITY_CUTOFF = 0.35


def _pre_extraction_rbf(squared: np.ndarray) -> np.ndarray:
    """Verbatim copy of the in-stage median-bandwidth RBF before extraction."""
    off_diagonal = squared[~np.eye(len(squared), dtype=bool)]
    bandwidth = float(np.median(off_diagonal))
    kernel = np.exp(-squared / bandwidth)
    np.fill_diagonal(kernel, 0.0)
    return kernel / kernel.sum(axis=1, keepdims=True)


def _fixture() -> FieldOperatorInputs:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(6, 3))
    squared = np.square(points[:, None, :] - points[None, :, :]).sum(axis=2)
    w2 = rng.uniform(size=(6, 6))
    w2[w2 < _SPARSITY_CUTOFF] = 0.0
    np.fill_diagonal(w2, 0.0)
    w2 /= w2.sum(axis=1, keepdims=True)
    co = rng.integers(1, 5, size=(6, 6)).astype(np.float64)
    co = co + co.T
    np.fill_diagonal(co, 0.0)
    co /= co.sum(axis=1, keepdims=True)
    tickers = ["AMD", "AAA", "BBB", "CCC", "DDD", "EEE"]
    return FieldOperatorInputs(
        tickers=tickers,
        matrices={"w1": w2, "w2": w2},
        squared_distances=squared,
        co_mentions=co,
        industry=dict(zip(tickers, ["s", "s", "t", "t", "s", "u"], strict=True)),
        sector=dict(zip(tickers, ["X", "X", "X", "Y", "X", "Y"], strict=True)),
    )


def test_extracted_builders_reproduce_pre_extraction_matrices() -> None:
    """Shared builders are bit-identical to the former in-stage expressions."""
    inputs = _fixture()
    assert np.array_equal(
        rbf_median_operator(inputs.squared_distances),
        _pre_extraction_rbf(inputs.squared_distances),
    )
    assert np.array_equal(
        equal_support_operator(inputs.matrices["w2"], active_tolerance=1e-8),
        equal_support_weights(inputs.matrices["w2"], active_tolerance=1e-8),
    )
    assert np.array_equal(co_mention_operator(inputs.co_mentions), inputs.co_mentions)


def test_peer_rows_are_ranked_row_coefficients_with_residual_mass() -> None:
    """Each basket is the top-k of one scored operator row plus its residual."""
    inputs = _fixture()
    operators = peer_definition_operators(inputs)
    config = PeerDefinitionConfig(targets=("AMD", "CCC"), top_k=3)
    rows = peer_definition_rows(inputs, operators, config)
    for (target, definition), basket in rows.groupby(["target", "definition"]):
        row = operators[definition][inputs.tickers.index(target)]
        expected = np.sort(row[row > 0.0])[::-1][: config.top_k]
        assert_allclose(basket.sort_values("rank")["weight"], expected)
        assert_allclose(basket["weight"].sum() + basket["remaining_mass"].iloc[0], 1.0)
    facts = caption_facts(rows)
    assert set(facts["targets"]) == {"AMD", "CCC"}  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (_SHARED / "barycentres" / _UNIT / "wasserstein_w2_loo").is_dir()
    or not _PEER_ROWS.is_file(),
    reason="production Paper 5 artifacts are not materialized",
)
def test_real_artifacts_match_the_scored_economic_value_operators() -> None:
    """On production artifacts the W2 rows match the stage-written peer panel."""
    inputs = load_field_operator_inputs(
        FieldOperatorPaths(
            returns_dir=_SHARED / "returns",
            w1_barycentre_dir=_SHARED / "barycentres" / _UNIT / "wasserstein_w1_loo",
            w2_barycentre_dir=_SHARED / "barycentres" / _UNIT / "wasserstein_w2_loo",
            distance_artifact_dir=_SHARED
            / "typed_distances"
            / _UNIT
            / "wasserstein_w2",
            co_mentions_adjacency=_SHARED / "co_mentions" / "adjacency.parquet",
            universe_csv=_REPO / "data" / "universe.csv",
        ),
        FieldOperatorIdentity(),
    )
    peer_rows = pd.read_parquet(_PEER_ROWS)
    w2 = inputs.matrices["w2"]
    for record in peer_rows.to_dict(orient="records"):
        target = inputs.tickers.index(str(record["target"]))
        candidate = inputs.tickers.index(str(record["candidate"]))
        assert w2[target, candidate] == pytest.approx(record["weight"], abs=1e-15)
    rbf = rbf_median_operator(inputs.squared_distances)
    assert np.array_equal(rbf, _pre_extraction_rbf(inputs.squared_distances))
    assert median_bandwidth(inputs.squared_distances) > 0.0
    validate_operator(co_mention_operator(inputs.co_mentions), name="co-mentions")

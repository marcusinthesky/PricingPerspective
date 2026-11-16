"""Admission and saved-result contracts for the original-W2 sector diagnostic."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline.stages.papers.paper1 import w2_permanova as stage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def admitted_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[stage.Paper1W2PermanovaPaths, dict[str, object]]:
    """Provide scalar-mass W2 geometry with reversed classification-file order."""
    tickers = ["A", "B", "C", "D"]
    x = np.array([0.0, 1.0, 4.0, 5.0])
    distances = np.abs(x[:, None] - x[None, :])
    summary: dict[str, object] = {
        "item_ids": tickers,
        "metadata": {
            "estimator": "balanced_wasserstein_2",
            "normalization": "rooted",
            "code_identity": "upstream-test-identity",
        },
    }

    def read_distance(
        _output_dir: Path, *, expected_identity: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        assert expected_identity["distance_id"] == "wasserstein_w2"
        assert expected_identity["value_semantics"] == "statistical_distance"
        return pd.DataFrame({"value": distances.ravel()}), summary

    monkeypatch.setattr(stage, "read_typed_distance_artifact", read_distance)
    covariance = tmp_path / "cohort.parquet"
    # The scientific values are intentionally missing: only ticker identities
    # select the cohort for this diagnostic, never return information.
    pd.DataFrame({"ticker_i": tickers, "ticker_j": tickers}).to_parquet(covariance)
    universe = tmp_path / "universe.csv"
    pd.DataFrame({"Symbol": tickers[::-1], "Sector": ["Y", "Y", "X", "X"]}).to_csv(
        universe, index=False
    )
    paths = stage.Paper1W2PermanovaPaths(
        distance_artifact_dir=tmp_path / "typed",
        covariance_matrix=covariance,
        universe_csv=universe,
        output_dir=tmp_path / "results",
    )
    return paths, summary


def _run(paths: stage.Paper1W2PermanovaPaths) -> dict[str, object]:
    return stage.run_w2_permanova(
        paths,
        provider_id="qwen3-embedding-8b",
        representation_id="qwen3-embedding-8b-unit",
        n_permutations=19,
        random_state=7,
    )


def test_priced_cohort_and_permutation_audit_are_saved(
    admitted_inputs: tuple[stage.Paper1W2PermanovaPaths, dict[str, object]],
) -> None:
    """Preserve cohort identities and permit replay from saved permutation draws."""
    paths, _ = admitted_inputs
    result = _run(paths)
    assert result["tickers"] == ["A", "B", "C", "D"]
    permanova = result["permanova"]
    assert isinstance(permanova, dict)
    assert permanova["r2"] == pytest.approx(16 / 17)
    assert permanova["pseudo_f"] == pytest.approx(32)
    assert json.loads((paths.output_dir / "results.json").read_text()) == result
    with np.load(
        paths.output_dir / "permutation_audit.npz", allow_pickle=False
    ) as audit:
        assert audit["tickers"].tolist() == result["tickers"]
        assert audit["sector_labels"].tolist() == ["X", "X", "Y", "Y"]
        assert np.isfinite(audit["permanova_null_r2"]).all()


@pytest.mark.parametrize(
    ("field", "value"), [("estimator", "sinkhorn"), ("normalization", "squared")]
)
def test_rejects_a_different_transport_or_distance_power(
    admitted_inputs: tuple[stage.Paper1W2PermanovaPaths, dict[str, object]],
    field: str,
    value: str,
) -> None:
    """Enforce fields the general typed reader does not independently admit."""
    paths, summary = admitted_inputs
    metadata = summary["metadata"]
    assert isinstance(metadata, dict)
    metadata[field] = value
    with pytest.raises(ValueError, match="rooted balanced-W2"):
        _run(paths)


def test_incomplete_sector_metadata_fails_without_shrinking_the_sample(
    admitted_inputs: tuple[stage.Paper1W2PermanovaPaths, dict[str, object]],
) -> None:
    """Reject missing classifications and invalidate a stale completion result."""
    paths, _ = admitted_inputs
    pd.DataFrame({"Symbol": ["A", "B", "C"], "Sector": ["X", "X", "Y"]}).to_csv(
        paths.universe_csv, index=False
    )
    paths.output_dir.mkdir()
    completion = paths.output_dir / "results.json"
    completion.write_text('{"stale": true}')
    with pytest.raises(ValueError, match="every priced firm"):
        _run(paths)
    assert not completion.exists()


def test_missing_geometry_fails_without_shrinking_the_sample(
    admitted_inputs: tuple[stage.Paper1W2PermanovaPaths, dict[str, object]],
) -> None:
    """Require geometry for every explicitly selected priced firm."""
    paths, summary = admitted_inputs
    summary["item_ids"] = ["A", "B", "C"]
    with pytest.raises(ValueError, match="full priced cohort"):
        _run(paths)

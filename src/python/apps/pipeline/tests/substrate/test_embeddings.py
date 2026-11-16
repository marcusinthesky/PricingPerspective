"""Tests for embedding-model coverage contracts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pipeline.stages.substrate.embeddings import (
    _resolve_coverage_bounds,
    validate_embedding,
    validate_public_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def test_model_coverage_contract_overrides_global_sample_size() -> None:
    """A frozen model panel can use a row contract distinct from the source sample."""
    params: Mapping[str, object] = {
        "n_samples": 445,
        "embedding": {
            "models": {
                "qwen3-embedding-8b": {
                    "coverage": {"target_rows": 128, "minimum_rows": 128}
                }
            }
        },
    }

    assert _resolve_coverage_bounds(params, "qwen3-embedding-8b") == (128, 128)


def test_coverage_contract_falls_back_to_global_sample_size() -> None:
    """Models without an override retain the bounded global-sample contract."""
    params: Mapping[str, object] = {
        "n_samples": 445,
        "embedding": {"models": {"other": {}}},
    }

    assert _resolve_coverage_bounds(params, "other") == (445, 400)


def test_coverage_contract_rejects_inverted_bounds() -> None:
    """A malformed model contract fails before any parquet is scanned."""
    params: Mapping[str, object] = {
        "n_samples": 445,
        "embedding": {
            "models": {
                "broken": {"coverage": {"target_rows": 100, "minimum_rows": 101}}
            }
        },
    }

    with pytest.raises(ValueError, match="invalid coverage bounds"):
        _resolve_coverage_bounds(params, "broken")


def _input(path: Path, *, with_text: bool = False) -> dict[str, object]:
    columns = {
        "url_hash": ["stable-id"],
        "created_date": ["2020-01-01"],
        "embedding": [[0.25, 0.75]],
    }
    if with_text:
        columns["body"] = ["An excluded article field"]
    table = pa.table(columns)
    pq.write_table(table, path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": 1,
        "columns": table.column_names,
    }


def test_numeric_shard_retains_exact_vectors(tmp_path: Path) -> None:
    """Approved metadata and vector values survive validation unchanged."""
    path = tmp_path / "vectors.parquet"
    expected = _input(path)
    validate_embedding(path, expected)
    assert pq.read_table(path)["embedding"].to_pylist() == [[0.25, 0.75]]


def test_article_column_is_rejected_even_with_valid_checksum(tmp_path: Path) -> None:
    """A matching file hash cannot authorize an excluded article column."""
    path = tmp_path / "vectors.parquet"
    expected = _input(path, with_text=True)
    with pytest.raises(ValueError, match="undeclared"):
        validate_embedding(path, expected)


def test_modified_shard_is_rejected(tmp_path: Path) -> None:
    """A substituted input cannot pass under its original published checksum."""
    path = tmp_path / "vectors.parquet"
    expected = _input(path)
    expected["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        validate_embedding(path, expected)


def test_changed_cohort_is_rejected_even_with_valid_checksum(tmp_path: Path) -> None:
    """The manifest's observation count is part of the fixed sample contract."""
    path = tmp_path / "vectors.parquet"
    expected = _input(path)
    expected["rows"] = 2
    with pytest.raises(ValueError, match="cohort"):
        validate_embedding(path, expected)


def test_manifest_validates_every_declared_shard(tmp_path: Path) -> None:
    """A valid first shard cannot conceal a changed later embedding input."""
    first = _input(tmp_path / "first.parquet")
    second = _input(tmp_path / "second.parquet")
    second["sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "embeddings": {"first.parquet": first, "second.parquet": second},
            }
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        validate_public_inputs(manifest, tmp_path)


def test_manifest_cannot_escape_input_root(tmp_path: Path) -> None:
    """Manifest paths stay within the declared public input namespace."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "embeddings": {"../escape": {}}}))
    with pytest.raises(ValueError, match="unsafe"):
        validate_public_inputs(manifest, tmp_path)

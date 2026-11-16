"""Paper 1 Wasserstein comparator integration-contract tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline.stages.papers.paper1.wasserstein_comparator import (
    _MODEL_DIRS,
    TransportMatrices,
    _load_prefix,
    _pair_metrics,
    _validate_common_hashes,
)

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_HASH_HEX_LENGTH = 64
EXPECTED_PAIR_COUNT = 3


def test_url_hash_prefix_is_lexicographic_and_deterministic(tmp_path: Path) -> None:
    """Load a deterministic lexicographic prefix of URL hashes."""
    path = tmp_path / "AAA.parquet"
    frame = pd.DataFrame(
        {
            "url_hash": ["d", "a", "c", "b"],
            "embedding": [
                np.array([1.0, 0.0]),
                np.array([0.0, 2.0]),
                np.array([1.0, 1.0]),
                np.array([-2.0, 0.0]),
            ],
        }
    )
    frame.to_parquet(path, index=False)
    first_hashes, first_rows = _load_prefix(path, 3)
    second_hashes, second_rows = _load_prefix(path, 3)
    assert first_hashes == ("a", "b", "c")
    assert first_hashes == second_hashes
    assert np.array_equal(first_rows, second_rows)
    assert np.linalg.norm(first_rows, axis=1) == pytest.approx(np.ones(3))


def test_all_four_encoder_prefixes_require_coverage_and_hash_equality() -> None:
    """Require equal ticker coverage and hashes across all encoders."""
    shared: dict[str, tuple[str, ...]] = {
        "AAA": ("h1", "h2"),
        "BBB": ("h3", "h4"),
    }
    aligned: dict[str, dict[str, tuple[str, ...]]] = {
        model: dict(shared) for model in _MODEL_DIRS
    }
    digest, reference = _validate_common_hashes(aligned, 2)
    assert len(digest) == EXPECTED_HASH_HEX_LENGTH
    assert reference == shared

    mismatched: dict[str, dict[str, tuple[str, ...]]] = {
        model: dict(shared) for model in _MODEL_DIRS
    }
    mismatched["bge-m3"]["AAA"] = ("h1", "different")
    with pytest.raises(ValueError, match="url_hash prefix mismatch"):
        _validate_common_hashes(mismatched, 2)

    missing: dict[str, dict[str, tuple[str, ...]]] = {
        model: dict(shared) for model in _MODEL_DIRS
    }
    missing["qwen3-embedding-8b"].pop("BBB")
    with pytest.raises(ValueError, match="ticker coverage differs"):
        _validate_common_hashes(missing, 2)


def test_matched_pair_matrices_persist_signed_negative_u_without_sqrt() -> None:
    """Persist signed U-energy values without applying a square root."""
    base = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    samples = {
        "AAA": base,
        "BBB": base.copy(),
        "CCC": np.roll(base, 1, axis=1),
    }
    return_dist = np.array(
        [[0.0, 0.3, 0.8], [0.3, 0.0, 0.7], [0.8, 0.7, 0.0]],
        dtype=np.float64,
    )
    frame = _pair_metrics(samples, sorted(samples), return_dist, "test", 4)
    assert len(frame) == EXPECTED_PAIR_COUNT
    assert "centroid_chord" not in frame
    assert np.isfinite(frame["energy_v"]).all()
    assert (frame["energy_u_functional"] < 0.0).any()
    assert "energy_u_sqrt" not in frame


def test_pair_metrics_uses_supplied_typed_wasserstein_matrix() -> None:
    """The comparator consumes a persisted W1 matrix when one is supplied."""
    rng = np.random.default_rng(19)
    tickers = ["AAA", "BBB", "CCC"]
    samples = {ticker: rng.normal(size=(4, 3)) for ticker in tickers}
    return_dist = np.eye(len(tickers), dtype=np.float64)
    typed_matrix = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    frame = _pair_metrics(
        samples,
        tickers,
        return_dist,
        "test",
        4,
        transport=TransportMatrices(
            order_one=typed_matrix,
            order_two=typed_matrix,
        ),
    )

    assert np.allclose(frame["wasserstein_w1"], 0.0)


def test_duplicate_or_zero_norm_prefixes_are_rejected(tmp_path: Path) -> None:
    """Reject duplicate hashes and zero-norm embeddings in prefixes."""
    duplicate = tmp_path / "duplicate.parquet"
    pd.DataFrame(
        {
            "url_hash": ["a", "a"],
            "embedding": [np.ones(2), np.ones(2)],
        }
    ).to_parquet(duplicate, index=False)
    with pytest.raises(ValueError, match="unique"):
        _load_prefix(duplicate, 2)

    zero = tmp_path / "zero.parquet"
    pd.DataFrame(
        {
            "url_hash": ["a", "b"],
            "embedding": [np.ones(2), np.zeros(2)],
        }
    ).to_parquet(zero, index=False)
    with pytest.raises(ValueError, match="zero-norm"):
        _load_prefix(zero, 2)

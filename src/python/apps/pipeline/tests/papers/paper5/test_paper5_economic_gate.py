"""Contracts for Paper 5's post-corpus bootstrap and comparator baskets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pipeline.stages.papers.paper5.baskets import (
    equal_support_weights,
    industry_weights,
    random_matched_sparsity_weights,
)
from pipeline.stages.papers.paper5.bootstrap import stationary_bootstrap_indices
from pipeline.stages.papers.paper5.periods import POST_CORPUS_END_DATE

REPO_ROOT = Path(__file__).resolve().parents[7]


def test_post_corpus_cutoff_matches_governed_return_panel() -> None:
    """Paper 5's folds cannot extend beyond the complete-case OOS contract."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    oos_panel = next(panel for panel in params["return_panels"] if panel["id"] == "oos")
    assert oos_panel["expected_end_date"] == POST_CORPUS_END_DATE


def _example_weights() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.8, 0.2, 0.0],
            [0.4, 0.0, 0.0, 0.6],
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0],
        ],
        dtype=np.float64,
    )


def test_equal_support_changes_magnitudes_without_changing_support() -> None:
    """The decisive comparator isolates row magnitudes from the text support."""
    weights = _example_weights()
    comparator = equal_support_weights(weights, active_tolerance=1e-8)

    np.testing.assert_array_equal(comparator > 0.0, weights > 0.0)
    np.testing.assert_allclose(comparator.sum(axis=1), 1.0)
    np.testing.assert_allclose(comparator[0], [0.0, 0.5, 0.5, 0.0])


def test_matched_random_baskets_are_seeded_and_preserve_row_sparsity() -> None:
    """Random supports and magnitudes are fixed once and reusable across folds."""
    weights = _example_weights()
    first = random_matched_sparsity_weights(
        weights,
        draws=9,
        seed=17,
        active_tolerance=1e-8,
    )
    second = random_matched_sparsity_weights(
        weights,
        draws=9,
        seed=17,
        active_tolerance=1e-8,
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_array_equal(
        np.sum(first > 0.0, axis=2),
        np.broadcast_to(np.sum(weights > 0.0, axis=1), (9, 4)),
    )
    np.testing.assert_allclose(first.sum(axis=2), 1.0)
    np.testing.assert_allclose(np.diagonal(first, axis1=1, axis2=2), 0.0)


def test_industry_basket_has_a_deterministic_singleton_fallback() -> None:
    """A singleton industry and sector receives the non-gating universe basket."""
    tickers = ["A", "B", "C"]
    weights = industry_weights(
        tickers,
        industry_of={"A": "one", "B": "two", "C": "three"},
        sector_of={"A": "x", "B": "y", "C": "z"},
    )

    np.testing.assert_allclose(
        weights,
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    )


def test_stationary_bootstrap_is_reproducible() -> None:
    """Joint-date resampling is fixed by the registered seed."""
    first = stationary_bootstrap_indices(
        80,
        draws=199,
        expected_block_length=7,
        seed=9,
    )
    second = stationary_bootstrap_indices(
        80,
        draws=199,
        expected_block_length=7,
        seed=9,
    )
    np.testing.assert_array_equal(first, second)

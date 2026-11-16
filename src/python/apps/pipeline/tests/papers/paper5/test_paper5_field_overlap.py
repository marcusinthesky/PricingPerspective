"""Tests for Paper 5's interaction-field overlap diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from pipeline.stages.papers.paper5.field_overlap import (
    FieldOverlapError,
    induced_field_similarity,
    matched_random_support_null,
    support_overlap,
    weight_similarity,
)

Float = NDArray[np.float64]

_TOLERANCE = 1e-8
_SEED = 20260823
_IMPERFECT_CORRELATION = 0.99


@pytest.fixture(name="disjoint_pair")
def fixture_disjoint_pair() -> tuple[Float, Float]:
    """Two admissible fields whose supports are deliberately half-shared."""
    focal = np.array(
        [
            [0.0, 0.6, 0.4, 0.0],
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 0.7, 0.0],
        ],
        dtype=np.float64,
    )
    comparator = np.array(
        [
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    return focal, comparator


def test_identical_fields_agree_on_every_diagnostic(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """A field compared with itself reaches perfect agreement everywhere."""
    focal, _ = disjoint_pair
    overlap = support_overlap(focal, focal, tolerance=_TOLERANCE)
    similarity = weight_similarity(focal, focal, tolerance=_TOLERANCE)

    np.testing.assert_allclose(overlap.jaccard, 1.0)
    np.testing.assert_allclose(overlap.directed_share, 1.0)
    np.testing.assert_allclose(overlap.rank_matched_share, 1.0)
    np.testing.assert_allclose(similarity.union_cosine, 1.0)
    assert similarity.offdiagonal_pearson == pytest.approx(1.0)


def test_support_overlap_matches_hand_computed_rows(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """Row-level Jaccard and directed share reproduce the hand-checked values."""
    focal, comparator = disjoint_pair
    overlap = support_overlap(focal, comparator, tolerance=_TOLERANCE)

    # Row 0: focal {1, 2}, comparator {1, 3}; intersection {1}, union {1, 2, 3}.
    assert overlap.jaccard[0] == pytest.approx(1.0 / 3.0)
    assert overlap.directed_share[0] == pytest.approx(0.5)
    # Row 2: focal {1}, comparator {1, 3}; the single focal peer is shared.
    assert overlap.directed_share[2] == pytest.approx(1.0)
    np.testing.assert_array_equal(overlap.focal_support_sizes, [2, 2, 1, 2])
    np.testing.assert_array_equal(overlap.comparator_support_sizes, [2, 2, 2, 2])


def test_rank_matched_share_compares_only_the_strongest_shared_peers(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """Size matching uses the smaller row and never pads with inactive peers."""
    focal, comparator = disjoint_pair
    overlap = support_overlap(focal, comparator, tolerance=_TOLERANCE)

    # k = min(focal, comparator) row sizes = [2, 2, 1, 2]. Rows 0 and 1 share one
    # of their two strongest peers; rows 2 and 3 agree on all of theirs.
    np.testing.assert_allclose(overlap.rank_matched_share, [0.5, 0.5, 1.0, 1.0])
    assert np.all(
        (overlap.rank_matched_share >= 0.0) & (overlap.rank_matched_share <= 1.0)
    )


def test_rank_matched_share_ignores_padding_beyond_the_active_set(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """A sparser comparator cannot gain agreement from its inactive entries."""
    focal, _ = disjoint_pair
    sparse = np.zeros_like(focal)
    sparse[:, 0] = 1.0
    sparse[0, 0], sparse[0, 1] = 0.0, 1.0

    overlap = support_overlap(focal, sparse, tolerance=_TOLERANCE)

    # Every comparator row names exactly one peer, so k = 1 throughout and the
    # share can only reflect that single genuinely selected peer.
    np.testing.assert_array_equal(overlap.comparator_support_sizes, [1, 1, 1, 1])
    assert np.all(np.isin(overlap.rank_matched_share, [0.0, 1.0]))


def test_common_edge_similarity_ignores_unshared_peers(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """Common-edge counts record exactly the peers both constructions chose."""
    focal, comparator = disjoint_pair
    similarity = weight_similarity(focal, comparator, tolerance=_TOLERANCE)

    # Rows 0-2 share one peer each; row 3 shares both of its peers.
    np.testing.assert_array_equal(similarity.common_edge_counts, [1, 1, 1, 2])
    # A single shared edge, or a constant comparator row, has no rank correlation.
    assert np.all(np.isnan(similarity.common_edge_spearman))


def test_induced_field_similarity_is_perfect_for_a_shared_operator(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """A field compared with itself induces an identical spatial lag."""
    focal, _ = disjoint_pair
    rng = np.random.default_rng(_SEED)
    returns = rng.normal(scale=0.01, size=(50, 4))

    similarity = induced_field_similarity(returns, focal, focal)

    assert similarity.pooled_pearson == pytest.approx(1.0)
    assert similarity.pooled_r_squared == pytest.approx(1.0)
    np.testing.assert_allclose(similarity.per_asset_pearson, 1.0, rtol=1e-10)


def test_induced_field_similarity_separates_distinct_operators(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """Two genuinely different fields induce imperfectly correlated lags."""
    focal, comparator = disjoint_pair
    rng = np.random.default_rng(_SEED)
    returns = rng.normal(scale=0.01, size=(400, 4))

    similarity = induced_field_similarity(returns, focal, comparator)

    assert -1.0 <= similarity.pooled_pearson < _IMPERFECT_CORRELATION
    assert similarity.per_asset_pearson.shape == (4,)


def test_matched_random_null_calibrates_support_agreement(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """The null draws share the focal sparsity and stay inside the unit interval."""
    focal, comparator = disjoint_pair

    null = matched_random_support_null(
        focal, comparator, draws=16, seed=_SEED, tolerance=_TOLERANCE
    )

    assert null.shape == (16,)
    assert np.all((null >= 0.0) & (null <= 1.0))


def test_overlap_rejects_a_field_that_is_not_an_interaction_matrix(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """A comparator with a nonzero diagonal violates the peer-average contract."""
    focal, comparator = disjoint_pair
    broken = comparator.copy()
    broken[0, 0] = 0.5
    broken[0, 1] = 0.0

    with pytest.raises(ValueError, match="zero diagonal"):
        support_overlap(focal, broken, tolerance=_TOLERANCE)


def test_overlap_rejects_a_shape_mismatch(
    disjoint_pair: tuple[Float, Float],
) -> None:
    """Fields of different sizes cannot be compared row by row."""
    focal, _ = disjoint_pair
    with pytest.raises(FieldOverlapError, match="same shape"):
        support_overlap(focal, np.eye(3, dtype=np.float64), tolerance=_TOLERANCE)

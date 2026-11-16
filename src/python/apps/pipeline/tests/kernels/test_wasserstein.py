"""Contract tests for the exact balanced empirical W1 kernel."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
from statadic.geometry.discrepancy.transport.exact import (
    balanced_wasserstein1_from_cost,
    chord_cost_matrix,
)
from statadic.geometry.discrepancy.transport.exact import (
    exact_balanced_wasserstein1 as _packed_wasserstein1,
)
from statadic.measure.clouds import PackedClouds


def exact_balanced_wasserstein1(left: np.ndarray, right: np.ndarray) -> float:
    """Adapt sample arrays to Statadic's uniform single-cloud boundary."""
    return _packed_wasserstein1(
        PackedClouds.from_points(left),
        PackedClouds.from_points(right),
    )


def _brute_force(cost: np.ndarray) -> float:
    return min(
        float(np.mean([cost[i, j] for i, j in enumerate(order)]))
        for order in permutations(range(len(cost)))
    )


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Apply the pipeline-owned normalization used by production loaders."""
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _mean_roundoff(cost: np.ndarray) -> float:
    """Bound different reduction orders in the caller's transport precision."""
    return 2 * len(cost) * np.finfo(cost.dtype).eps * max(1.0, float(cost.max()))


def _chord_roundoff(cost: np.ndarray, dimension: int) -> float:
    """Square root amplifies dot-product roundoff near coincident unit rows.

    The standard gamma bound accounts for the dot product and the subtraction
    in 2 - 2 dot(x, y). This is a ground-cost bound, not assignment error.
    """
    accumulated = (dimension + 2) * np.finfo(cost.dtype).eps
    return float(np.sqrt(2 * accumulated / (1 - accumulated)))


def test_exact_assignment_matches_brute_force() -> None:
    """Match exhaustive assignment on a small deterministic instance."""
    rng = np.random.default_rng(2)
    left, right = (_unit_rows(rng.normal(size=(4, 5))) for _ in range(2))
    cost = chord_cost_matrix(left, right)
    # `chord_cost_matrix` returns a `jax.Array` as of t65.4; the brute-force
    # oracle below is deliberately NumPy, so this is the host boundary.
    assert balanced_wasserstein1_from_cost(cost) == pytest.approx(
        _brute_force(np.asarray(cost)), abs=_mean_roundoff(np.asarray(cost))
    )


def test_identity_symmetry_and_triangle_inequality() -> None:
    """Satisfy the defining metric identities on normalized samples."""
    rng = np.random.default_rng(3)
    a, b, c = (_unit_rows(rng.normal(size=(5, 7))) for _ in range(3))
    d_ab = exact_balanced_wasserstein1(a, b)
    d_bc = exact_balanced_wasserstein1(b, c)
    d_ac = exact_balanced_wasserstein1(a, c)
    cost = np.asarray(chord_cost_matrix(a, a))
    assert exact_balanced_wasserstein1(a, a) == pytest.approx(
        0.0, abs=_chord_roundoff(cost, a.shape[1])
    )
    assert d_ab == pytest.approx(
        exact_balanced_wasserstein1(b, a), abs=_mean_roundoff(cost)
    )
    assert d_ac <= d_ab + d_bc + 1e-12


def test_chord_roundoff_against_direct_euclidean_reference() -> None:
    """Bound cancellation at identity without accepting collapsed distances."""
    rng = np.random.default_rng(13)
    rows = _unit_rows(rng.normal(size=(6, 17)))
    cost = np.asarray(chord_cost_matrix(rows, rows))
    reference = np.linalg.norm(rows[:, None] - rows[None, :], axis=-1)
    assert np.max(np.abs(cost - reference)) <= _chord_roundoff(cost, rows.shape[1])
    # Distinct rows remain separated beyond the cancellation error bound.
    assert np.all(
        cost[~np.eye(len(rows), dtype=bool)] > _chord_roundoff(cost, rows.shape[1])
    )


def test_row_permutations_do_not_change_empirical_w1() -> None:
    """Remain invariant to independent observation ordering."""
    rng = np.random.default_rng(7)
    left, right = (_unit_rows(rng.normal(size=(8, 4))) for _ in range(2))
    expected = exact_balanced_wasserstein1(left, right)
    assert exact_balanced_wasserstein1(
        left[::-1], right[[3, 0, 7, 1, 6, 2, 5, 4]]
    ) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (np.ones((2, 3)), np.ones((3, 3)), "equal-cardinality"),
        (np.ones((2, 3)), np.ones((2, 4)), "equal feature width"),
        (np.array([[0.0, 0.0], [1.0, 0.0]]), np.ones((2, 2)), "zero-norm"),
        (
            np.array([[np.nan, 0.0], [1.0, 0.0]]),
            np.ones((2, 2)),
            "Every active support coordinate must be finite",
        ),
        (np.ones((2, 2)), np.eye(2), "already be L2-normalized"),
    ],
)
def test_invalid_samples_are_rejected(
    left: np.ndarray, right: np.ndarray, message: str
) -> None:
    """Reject incompatible, degenerate, or non-finite sample matrices."""
    with pytest.raises(ValueError, match=message):
        exact_balanced_wasserstein1(left, right)


def test_invalid_cost_matrices_are_rejected() -> None:
    """Reject non-square and negative transport-cost matrices."""
    with pytest.raises(ValueError, match="square"):
        balanced_wasserstein1_from_cost(np.ones((2, 3)))
    with pytest.raises(ValueError, match="negative"):
        balanced_wasserstein1_from_cost(np.array([[0.0, -1.0], [1.0, 0.0]]))

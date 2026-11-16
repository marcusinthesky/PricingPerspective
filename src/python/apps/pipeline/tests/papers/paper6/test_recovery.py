"""Nested measurement, shared dates and deterministic history redraws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hotelling.orthant import SOLVE_TARGET
from pipeline._kernels.paper6.recovery import BUDGETS, article_redraws, measure_history

if TYPE_CHECKING:
    import pytest
    from numpy.typing import NDArray


def test_nested_budgets_share_dates_across_regimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One date increment drives both financial tails and the two article laws."""
    captured: list[NDArray[np.int64]] = []

    def capture(
        pooled: NDArray[np.float64],
        dates: NDArray[np.int64],
        news: NDArray[np.float64],
    ) -> float:
        assert pooled.shape == (5, 4, 4)
        assert news.shape == (60, 4)
        np.testing.assert_allclose(pooled.sum(axis=-1), 1)
        captured.append(dates.copy())
        return float(dates.sum())

    monkeypatch.setattr("pipeline._kernels.paper6.recovery.estimate_panel", capture)
    counts = np.ones((5, 12, 4), dtype=np.int64)
    laws = np.full((2, 5, 12, 4, 4), 0.25)
    result = article_redraws(laws, counts, np.zeros((60, 4)), 3, redraws=1)
    for index, budget in enumerate(BUDGETS):
        np.testing.assert_array_equal(captured[2 * index], captured[2 * index + 1])
        np.testing.assert_array_equal(
            captured[2 * index].sum(axis=1), budget * counts.sum(axis=1)
        )
        if index:
            assert np.all(captured[2 * index] >= captured[2 * (index - 1)])
    assert result.shape == (2, 3, 1)


def test_history_redraws_are_repeatable_without_shared_executor_state() -> None:
    """Resuming a history reproduces scientific statistics, including NaN outcomes."""
    counts = np.ones((5, 12, 4), dtype=np.int64)
    laws = np.full((2, 5, 12, 4, 4), 0.25)
    news = np.zeros((60, 4))
    news[18, :] = np.array([-1.0, -0.2, 0.3, 1.0])
    first = article_redraws(laws, counts, news, 7, redraws=2)
    second = article_redraws(laws, counts, news, 7, redraws=2)
    np.testing.assert_array_equal(first, second)


def test_population_oracles_separate_reporting_from_finite_capacity() -> None:
    """The zero-amplitude law recovers beta0 and the fixed-capacity control is zero."""
    result = measure_history(0, np.ones((5, 12, 100), dtype=np.int64), redraws=2)
    np.testing.assert_allclose(result.beta0, result.population_beta[0], rtol=0, atol=0)
    np.testing.assert_allclose(result.reporting_only_beta[0], 0, atol=1e-10)
    np.testing.assert_allclose(result.affine_beta, result.beta0, rtol=1e-3)
    assert result.minimum_capacity > 0
    assert result.maximum_solver_bound <= SOLVE_TARGET
    assert result.estimates.shape == (2, 3, 2)

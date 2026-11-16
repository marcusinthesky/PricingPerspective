"""Contract tests for frozen Paper 3 certificate validation."""

from __future__ import annotations

import numpy as np
import pytest
from pipeline._kernels.paper3.validation import (
    CalibrationCell,
    capital_to_budget_weights,
    optimize_certificate,
    residual_bound,
    select_calibration_cell,
    select_stationary_block_length,
    stationary_block_indices,
)


def test_inverse_volatility_capital_is_equal_budget() -> None:
    """Inverse-volatility capital weights imply equal budget shares."""
    volatility = np.array([0.1, 0.2, 0.4])
    capital = (1.0 / volatility) / np.sum(1.0 / volatility)
    assert capital_to_budget_weights(capital, volatility) == pytest.approx(
        np.full(3, 1.0 / 3.0)
    )


def test_calibration_selection_ignores_holdout_and_breaks_ties() -> None:
    """Selection uses calibration arrays and deterministic tie breaks."""
    safe = np.full(120, 0.1)
    cells = (
        CalibrationCell(1.1, 0.025, safe, np.full(120, 0.2)),
        CalibrationCell(1.0, 0.05, safe, np.full(120, 0.2)),
    )
    assert select_calibration_cell(cells, confidence=0.95, coverage_target=0.9) == (
        1.0,
        0.05,
    )


def test_calibration_zero_credit_fallback() -> None:
    """No passing cell freezes the perfect-dependence benchmark."""
    failing = CalibrationCell(1.0, 0.0, -np.ones(20), np.ones(20))
    assert select_calibration_cell(
        (failing,), confidence=0.95, coverage_target=0.9
    ) == (float("inf"), float("inf"))


def test_stationary_blocks_are_deterministic_and_in_range() -> None:
    """Stationary blocks are seeded and use the short-series fallback."""
    first = stationary_block_indices(7, 20, 5, 42)
    second = stationary_block_indices(7, 20, 5, 42)
    assert np.array_equal(first, second)
    assert first.shape == (5, 7)
    assert first.min() >= 0
    assert first.max() < first.shape[1]


def test_block_length_has_deterministic_short_series_fallback() -> None:
    """Short article series use the documented bounded cube-root fallback."""
    expected = 2
    assert select_stationary_block_length(np.arange(7.0)) == expected


def test_qp_feasibility_and_fail_closed_kernel_gate() -> None:
    """The optimizer respects the cap and rejects a failed kernel gate."""
    squared_distance = np.ones((12, 12)) - np.eye(12)
    weights, credit = optimize_certificate(squared_distance, 0.1)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.1 + 1e-9
    assert credit > 0.0
    failed = squared_distance.copy()
    failed[0, 1] = failed[1, 0] = 10.0
    with pytest.raises(ValueError, match="positive definite"):
        optimize_certificate(failed, 0.1)


def test_residual_budget_is_monotone() -> None:
    """Larger residual budgets cannot tighten the variance bound."""
    assert residual_bound(0.7, 0.05) >= residual_bound(0.7, 0.025)

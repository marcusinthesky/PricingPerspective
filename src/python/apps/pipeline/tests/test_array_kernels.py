"""Mathematical boundaries of the shared pipeline array kernels."""

from __future__ import annotations

import numpy as np
import pytest
from pipeline._kernels.arrays import (
    InvalidNormalizationRowsError,
    l2_normalize_rows,
    l2_normalize_rows_then_mean,
)


def test_l2_normalize_rows_mints_unit_rows_without_changing_dtype() -> None:
    """The successful door establishes finite L2-unit rows."""
    values = np.array([[3.0, 4.0], [-5.0, 12.0]], dtype=np.float32)

    normalized = l2_normalize_rows(values)

    assert normalized.dtype == values.dtype
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


@pytest.mark.parametrize(
    "values",
    [
        np.empty((0, 2), dtype=np.float64),
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[np.nan, 1.0]], dtype=np.float64),
        np.array([[np.inf, 1.0]], dtype=np.float64),
    ],
)
def test_l2_normalize_rows_rejects_undefined_directions(values: np.ndarray) -> None:
    """No shared pipeline caller can silently pass through a false unit row."""
    with pytest.raises(InvalidNormalizationRowsError):
        l2_normalize_rows(values)


def test_prototype_transform_records_row_then_pool_then_normalize_order() -> None:
    """The pooled feature gives equal weight to directions, not magnitudes."""
    values = np.array([[100.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    rows, prototype = l2_normalize_rows_then_mean(values)

    expected = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(rows, np.eye(2))
    np.testing.assert_allclose(prototype, expected)
    assert not np.allclose(prototype, l2_normalize_rows(values.mean(axis=0)[None])[0])


def test_prototype_transform_rejects_a_zero_pooled_mean() -> None:
    """Unit rows do not imply that their empirical mean has a direction."""
    antipodes = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(InvalidNormalizationRowsError):
        l2_normalize_rows_then_mean(antipodes)

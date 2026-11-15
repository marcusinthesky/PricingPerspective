"""Numerical and degenerate-sample contracts for bootstrap skewness."""

from __future__ import annotations

import numpy as np
import pytest
from pipeline.stages.papers.paper1._dyadic.estimation import _sample_skewness


@pytest.mark.parametrize("constant", [0.0, 0.37, -2.1])
def test_constant_draws_have_no_skewness(constant: float) -> None:
    """Rounding in a constant sample must not create a shape diagnostic."""
    for dtype in (np.float32, np.float64):
        assert np.isnan(_sample_skewness(np.full(50, constant, dtype=dtype)))


@pytest.mark.parametrize("draws", [[], [1.0, 2.0], [np.nan, np.inf, 1.0, 2.0]])
def test_fewer_than_three_finite_draws_are_unavailable(draws: list[float]) -> None:
    """The count threshold applies after excluding nonfinite draws."""
    assert np.isnan(_sample_skewness(np.asarray(draws)))


def test_skewness_uses_population_moments_and_finite_draws() -> None:
    """The asymmetric three-point population has g1 = 1/sqrt(2)."""
    draws = np.array([0.0, 0.0, 1.0, np.nan, np.inf])
    assert _sample_skewness(draws) == pytest.approx(1 / np.sqrt(2))


def test_near_constant_draws_do_not_report_roundoff_as_skewness() -> None:
    """SciPy marks variation below numerical resolution unavailable."""
    draws = np.full(50, 0.37)
    draws[-1] = np.nextafter(draws[-1], np.inf)
    with pytest.warns(RuntimeWarning, match="Precision loss"):
        assert np.isnan(_sample_skewness(draws))

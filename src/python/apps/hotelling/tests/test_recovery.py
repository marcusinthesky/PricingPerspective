"""Fixed recovery design, primitive timing and observation-counterfactual checks."""

from __future__ import annotations

import numpy as np
import pytest
from hotelling.dynamics.recovery import (
    DIMENSIONS,
    NEWS_TICKS,
    histories,
    reporting_laws,
    solve_dimensional_history,
    solve_history,
)

HISTORY_COUNT = 16
TARGET = 1e-10


def test_locality_shocks_preserve_history_order_and_legacy_default() -> None:
    """Bounded strength changes scale news without changing dates or signs."""
    base, explicit = histories(), histories(shock_multiplier=1)
    half, double = histories(shock_multiplier=0.5), histories(shock_multiplier=2)
    for original, same, low, high in zip(base, explicit, half, double, strict=True):
        assert original.signs == same.signs == low.signs == high.signs
        np.testing.assert_array_equal(original.news, same.news)
        np.testing.assert_array_equal(low.news * 2, original.news)
        np.testing.assert_array_equal(high.news, original.news * 2)
        np.testing.assert_allclose(
            high.levels - 1, 2 * (original.levels - 1), atol=1e-15
        )
    with pytest.raises(ValueError, match="finite and positive"):
        histories(shock_multiplier=0)


def test_all_news_histories_are_immutable_positive_and_strictly_source_dated() -> None:
    """Every sign history appears once and a source shock changes next-date levels."""
    paths = histories()
    assert len(paths) == HISTORY_COUNT
    assert len({path.signs for path in paths}) == HISTORY_COUNT
    for path in paths:
        with pytest.raises(TypeError, match="immutable"):
            path.news[0] = 0
        with pytest.raises(TypeError, match="immutable"):
            path.levels[0, 0] = 0
        assert np.all(path.levels > 0)
        np.testing.assert_array_equal(np.flatnonzero(path.news), NEWS_TICKS)
        np.testing.assert_array_equal(path.levels[:19], np.ones((19, 2)))
        np.testing.assert_allclose(np.diff(path.levels[:, 0]), path.news)


def test_common_reporting_recovers_shares_and_reporting_only_moves_fixed_capacity() -> (
    None
):
    """Changing intensity alone changes observations; a common intensity cancels."""
    capacity = np.broadcast_to(np.array([[0.2, 0.8], [0.6, 0.4]]), (3, 2, 2))
    cumulative = np.array([0.0, 0.001, 0.002])
    common = reporting_laws(capacity, cumulative, 0)
    np.testing.assert_allclose(common[..., ::2] * 2, capacity)
    changing = reporting_laws(capacity, cumulative, 0.005)
    assert np.max(np.abs(changing[-1] - changing[0])) > 0
    np.testing.assert_allclose(changing.sum(axis=-1), 1)


def test_canonical_path_uses_monthly_discount_and_positive_permanent_news_prices() -> (
    None
):
    """One completed canonical path validates stage certificates and tangent domain."""
    path = solve_history(histories()[0])
    assert np.max(path.operating.bounds) <= TARGET
    assert path.minimum_affine_capacity > 0
    assert np.min(path.operating.capacity) > 0
    np.testing.assert_allclose(
        path.operating.economy.config.discount**12, 0.95, rtol=0, atol=1e-14
    )
    assert np.isfinite(path.financial_news).all()
    np.testing.assert_array_equal(path.financial_news[:18], 0)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_bounded_dimension_changes_only_opportunity_geometry(dimension: int) -> None:
    """Eight activities and 64 nodes remain fixed across four numeric extensions."""
    path = solve_dimensional_history(histories()[0], dimension)
    config = path.operating.economy.config
    assert (config.firms, config.activities, config.nodes) == (100, 8, 64)
    assert config.dimension == dimension
    assert config.kappa == 1
    assert path.active_domain_violations == 0
    assert path.minimum_baseline_price > 0
    assert path.minimum_opportunity > 0
    assert path.operating.bounds.max() <= TARGET
    laws = reporting_laws(path.operating.capacity, histories()[0].cumulative, 0)
    assert laws.shape == (61, 100, 16)
    np.testing.assert_allclose(laws.sum(axis=-1), 1, rtol=1e-14, atol=1e-14)

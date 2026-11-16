"""Independent transport, reporting, pooling and strict-date witnesses."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
from hotelling.dynamics.observation import (
    component_support,
    population_law,
    reporting_intensity,
    reporting_weights,
)
from hotelling.dynamics.transport import axis_point_w2, axis_w2, weighted_w2

if TYPE_CHECKING:
    from numpy.typing import NDArray

LP_LIMIT = 1e-9


def test_weighted_transport_honors_expired_deadline() -> None:
    """Transport cannot start an LP after the global campaign budget expires."""
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    weights = np.array([0.5, 0.5])
    with pytest.raises(TimeoutError, match="campaign time limit"):
        weighted_w2(points, weights, points, weights, deadline=perf_counter() - 1)


def test_reporting_scale_and_capacity_reporting_equivalence() -> None:
    """Law weights identify lambda*q; neither separate factor is identified."""
    q = np.array([[1.0, 2, 3], [4, 1, 2]])
    intensity = np.array([[2.0, 1, 0.5], [1, 0.25, 2]])
    scale = np.array([[0.3, 7, 2], [5, 2, 1]])
    law = reporting_weights(q, intensity)
    np.testing.assert_allclose(reporting_weights(q * scale, intensity / scale), law)
    np.testing.assert_allclose(reporting_weights(q, intensity * 5), law)
    np.testing.assert_allclose(reporting_weights(q * 3, intensity), law)
    np.testing.assert_allclose(law.sum(axis=-1), 1)


@pytest.mark.parametrize(
    ("q", "intensity"),
    [
        (np.zeros((2, 2)), np.ones((2, 2))),
        (np.ones((2, 2)), np.zeros((2, 2))),
        (np.array([[1.0, -1]]), np.ones((1, 2))),
        (np.ones((1, 2)), np.array([[np.nan, 1]])),
    ],
)
def test_undefined_reporting_laws_raise(
    q: NDArray[np.float64], intensity: NDArray[np.float64]
) -> None:
    """Undefined firm laws remain explicit failures rather than roster deletions."""
    with pytest.raises(ValueError, match=r"nonnegative|undefined"):
        reporting_weights(q, intensity)


def test_reporting_only_can_move_observations_at_fixed_capacity() -> None:
    """Changing reporting produces movement with no economic capacity response."""
    q = np.ones((25, 4, 3))
    intensity = reporting_intensity(
        q.shape,
        "reporting_only",
        jax.random.fold_in(jax.random.key(12, impl="threefry2x32"), 300),
    )
    laws = np.asarray(reporting_weights(q, intensity))
    assert float(axis_w2(laws[0], laws[-1]).sum()) > 0
    np.testing.assert_array_equal(q, 1)
    common = np.asarray(
        reporting_weights(
            q,
            reporting_intensity(
                q.shape,
                "common_fixed",
                jax.random.fold_in(jax.random.key(12, impl="threefry2x32"), 300),
            ),
        )
    )
    np.testing.assert_allclose(axis_w2(common[0], common[-1]), 0)


@pytest.mark.parametrize(("activities", "dimension"), [(2, 2), (3, 5), (8, 11)])
def test_axis_identity_against_independent_lp(activities: int, dimension: int) -> None:
    """A general sparse LP independently verifies the exact higher-dimensional law."""
    rng = np.random.default_rng(9)
    left, right = rng.dirichlet(np.ones(activities), size=2)
    support = np.asarray(component_support(activities, dimension))
    np.testing.assert_allclose(
        support[::2] @ support[::2].T, np.eye(activities), atol=1e-14
    )
    result = weighted_w2(
        support,
        np.asarray(population_law(left)),
        support,
        np.asarray(population_law(right)),
    )
    assert result.cost == pytest.approx(float(axis_w2(left, right)), abs=1e-12)
    assert result.marginal_error < LP_LIMIT
    assert result.dual_gap < LP_LIMIT
    assert result.error_bound < LP_LIMIT


def test_transport_depends_on_all_coordinates() -> None:
    """Equal first-coordinate projections can hide positive actual transport."""
    left, right = np.array([[0.0, -1]]), np.array([[0.0, 1]])
    result = weighted_w2(left, np.ones(1), right, np.ones(1))
    assert result.cost == pytest.approx(4)
    projected = weighted_w2(left[:, :1], np.ones(1), right[:, :1], np.ones(1))
    assert projected.cost == pytest.approx(0)


@pytest.mark.parametrize("activities", [1, 2, 3, 8])
def test_arbitrary_axis_point_mass_formula_against_independent_lp(
    activities: int,
) -> None:
    """The point-law formula covers sign imbalance and binding Hall deficits."""
    rng = np.random.default_rng(736)
    support = np.asarray(component_support(activities))
    for left, right in rng.dirichlet(np.ones(2 * activities), size=(8, 2)):
        lp = weighted_w2(support, left, support, right)
        assert float(axis_point_w2(left, right)) == pytest.approx(lp.cost, abs=1e-12)
    left, right = np.zeros(2 * activities), np.zeros(2 * activities)
    left[0], right[1] = 1, 1
    assert float(axis_point_w2(left, right)) == pytest.approx(4)
    assert float(axis_point_w2(left, left)) == pytest.approx(0)
    if activities > 1:
        right[1], right[2] = 0, 1
        assert float(axis_point_w2(left, right)) == pytest.approx(2)


def test_transport_known_weighted_objective_and_probability_validation() -> None:
    """Moving .3 mass by distance three has squared cost 2.7."""
    support = np.array([[0.0], [3.0]])
    result = weighted_w2(support, np.array([0.7, 0.3]), support, np.array([0.4, 0.6]))
    assert result.cost == pytest.approx(2.7)
    assert abs(result.cost - 2.7) <= result.error_bound
    with pytest.raises(ValueError, match="probability"):
        weighted_w2(support, np.array([0.7, 0.4]), support, np.array([0.4, 0.6]))


def test_crossed_two_component_path_uses_finite_distance() -> None:
    """A crossing changes the absolute-distance branch and invalidates one tangent."""
    origin, later, reference = (np.array([p, 1 - p]) for p in (0.2, 0.8, 0.6))
    finite = float(axis_w2(origin, reference) - axis_w2(later, reference))
    assert finite == pytest.approx(0.4)
    assert finite != pytest.approx(2 * (later[0] - origin[0]))
    support = np.asarray(component_support(2))
    before = weighted_w2(
        support,
        np.asarray(population_law(origin)),
        support,
        np.asarray(population_law(reference)),
    )
    after = weighted_w2(
        support,
        np.asarray(population_law(later)),
        support,
        np.asarray(population_law(reference)),
    )
    assert before.cost - after.cost == pytest.approx(finite)

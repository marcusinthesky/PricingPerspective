"""Independent derivative, feasibility and timing checks for joint dynamics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter

import jax.numpy as jnp
import numpy as np
import pytest
from hotelling.dynamics.config import DynamicsConfig
from hotelling.dynamics.engine import (
    Game,
    advance,
    build_canonical_economy,
    build_economy,
    canonical_capacity,
    forcing,
    generate_innovations,
    initial_state,
    innovation_levels,
    operating_cash_flow,
    operator_jvp,
)
from hotelling.dynamics.geometry import sobol_points, squared_distance
from hotelling.dynamics.mesa import run_mesa_path

DIFFERENCE_STEP = 1e-6
DERIVATIVE_TOLERANCE = 2e-8
NONZERO_MOVEMENT = 1e-6


@pytest.fixture
def config() -> DynamicsConfig:
    """Small multidimensional design with post-control observations."""
    return DynamicsConfig(firms=6, activities=4, nodes=16, dimension=3, periods=15)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("firms", True),
        ("dimension", 0),
        ("article_dimension", 0),
        ("solve_target", float("nan")),
        ("outside", 0),
        ("rho", 1),
        ("discount", 1),
        ("kappa", -1),
        ("seed", -1),
        ("shock_kind", "unknown"),
    ],
)
def test_invalid_design_rejected(name: str, value: object) -> None:
    """An invalid scientific control cannot become a repaired run."""
    with pytest.raises(ValueError, match=name if name != "rho" else "rho"):
        DynamicsConfig.from_dict({name: value})


def test_config_roundtrip_and_immutability(config: DynamicsConfig) -> None:
    """Full defaults survive JSON-like roundtrips and fields remain immutable."""
    assert DynamicsConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown dynamics"):
        DynamicsConfig.from_dict({"typo": 2})
    with pytest.raises(FrozenInstanceError):
        config.__setattr__("firms", 5)


def test_geometry_nested_and_all_coordinates_used() -> None:
    """Changing activity count preserves sites; every coordinate affects distance."""
    short = sobol_points(4, 8, seed=4, stream=0)
    long = sobol_points(16, 8, seed=4, stream=0)
    np.testing.assert_array_equal(short, long[:4])
    np.testing.assert_array_equal(long, sobol_points(16, 8, seed=4, stream=0))
    assert not np.array_equal(long, sobol_points(16, 8, seed=4, stream=1))
    origin = np.zeros((1, 8))
    shifts = np.eye(8)
    np.testing.assert_allclose(squared_distance(origin, shifts), np.full((1, 8), 1 / 8))


def test_operator_and_shock_derivatives_match_differences(
    config: DynamicsConfig,
) -> None:
    """JVP and forcing differentiate the exact finite-shock live-rival operator."""
    economy = build_economy(config)
    q = np.full(economy.cost.shape, 0.2)
    direction = np.random.default_rng(7).normal(size=q.shape)
    game = Game(economy, np.zeros(2))
    numeric = (
        game.operator(q + DIFFERENCE_STEP * direction)
        - game.operator(q - DIFFERENCE_STEP * direction)
    ) / (2 * DIFFERENCE_STEP)
    np.testing.assert_allclose(
        operator_jvp(economy, q, direction), numeric, atol=DERIVATIVE_TOLERANCE
    )
    impulse = forcing(economy, q)
    for factor in range(2):
        step = DIFFERENCE_STEP * np.eye(2)[factor]
        numeric = -(
            Game(economy, step).operator(q) - Game(economy, -step).operator(q)
        ) / (2 * DIFFERENCE_STEP)
        np.testing.assert_allclose(
            impulse[:, :, factor], numeric, atol=DERIVATIVE_TOLERANCE
        )


def test_cost_curvature_bounds_the_economic_jacobian(config: DynamicsConfig) -> None:
    """The numerical modulus is conservative for the full live-rival field."""
    economy = build_economy(config)
    q = np.full(economy.cost.shape, 0.2)
    identity = np.eye(q.size)
    matrix = np.column_stack(
        [
            operator_jvp(economy, q, column.reshape(q.shape)).ravel()
            for column in identity
        ]
    )
    minimum = np.linalg.eigvalsh((matrix + matrix.T) / 2).min()
    assert minimum >= economy.modulus
    hessian = np.array(economy.cost_hessian, copy=True)
    hessian[0, 1] += 0.01
    with pytest.raises(ValueError, match="symmetric"):
        replace(economy, cost_hessian=jnp.asarray(hessian))
    with pytest.raises(ValueError, match="modulus"):
        replace(economy, modulus=jnp.asarray(10.0))


def test_canonical_fixture_matches_lean_primitives() -> None:
    """Canonical costs, identity kernel and equilibrium match the formal witness."""
    economy = build_canonical_economy(firms=6)
    expected = canonical_capacity(6)
    game = Game(economy, np.zeros(2))
    np.testing.assert_allclose(game.operator(expected), 0, atol=1e-16)
    np.testing.assert_allclose(initial_state(economy), expected, atol=1e-9)
    np.testing.assert_allclose(expected.sum(axis=1), 1)
    np.testing.assert_allclose(expected.sum(axis=0), 3)
    assert np.all(economy.cost > 0)
    np.testing.assert_array_equal(economy.kernel, np.eye(2))


def test_certified_stages_and_payout_accounting(config: DynamicsConfig) -> None:
    """Simultaneous stages preserve feasibility and use the actual cost functional."""
    economy = build_economy(config)
    q = initial_state(economy)
    control = advance(economy, q, np.zeros(2))
    np.testing.assert_allclose(control.capacity, q, atol=1e-9)
    result = advance(economy, q, np.array([0.02, -0.01]))
    assert result.relative_bound <= config.solve_target
    assert result.capacity.dtype == np.float64
    assert np.isfinite(result.capacity).all()
    assert np.all(result.capacity >= 0)
    game = Game(economy, np.array([0.02, -0.01]))
    linear = np.sum(game.cost * result.capacity, axis=1)
    quadratic = 0.5 * np.einsum(
        "ia,ab,ib->i", result.capacity, economy.cost_hessian, result.capacity
    )
    np.testing.assert_allclose(
        game.flow(result.capacity), game.revenues(result.capacity) - linear - quadratic
    )
    np.testing.assert_allclose(game.revenue.sum(), economy.revenue.sum())


def test_path_reproducible_and_no_future_shock_leakage(config: DynamicsConfig) -> None:
    """A later innovation cannot change any earlier capacity or payout."""
    innovations = generate_innovations(config)
    assert np.all(innovations[:12] == 0)
    first = run_mesa_path(
        build_economy(config),
        innovation_levels(build_economy(config), innovations),
        innovations,
    )
    second = run_mesa_path(
        build_economy(config),
        innovation_levels(build_economy(config), innovations),
        innovations,
    )
    np.testing.assert_array_equal(first.capacity, second.capacity)
    modified = np.array(innovations, copy=True)
    modified[13] += 0.02
    later = run_mesa_path(
        build_economy(config),
        innovation_levels(build_economy(config), modified),
        modified,
    )
    np.testing.assert_array_equal(first.capacity[:14], later.capacity[:14])
    assert np.linalg.norm(first.capacity[14] - later.capacity[14]) > NONZERO_MOVEMENT
    np.testing.assert_allclose(
        first.shock_states[1:], config.rho * first.shock_states[:-1] + innovations
    )
    np.testing.assert_allclose(
        first.cash_flow[0], operating_cash_flow(first.economy, first.capacity[0])
    )
    penalty = 0.5 * config.kappa * np.sum(np.diff(first.capacity, axis=0) ** 2, axis=2)
    np.testing.assert_allclose(first.net_cash_flow[1:], first.cash_flow[1:] - penalty)


def test_budgets_and_invalid_state_fail_explicitly(config: DynamicsConfig) -> None:
    """No trajectory can silently continue with expired budgets or NaN inputs."""
    economy = build_economy(config)
    with pytest.raises(TimeoutError, match="time limit"):
        initial_state(economy, deadline=perf_counter() - 1)
    with pytest.raises(ArithmeticError, match="iteration limit"):
        initial_state(build_economy(replace(config, max_iterations=1)))
    with pytest.raises(ValueError, match="finite"):
        run_mesa_path(economy, np.full((config.periods + 1, config.nodes), np.nan))
    with pytest.raises(ValueError, match="nonnegative"):
        advance(economy, -np.ones_like(economy.cost), np.zeros(2))


@pytest.mark.parametrize("dimension", [1, 2, 4, 8, 16])
def test_hundred_firm_dimension_coverage(dimension: int) -> None:
    """Each mandatory dimension has a feasible certified hundred-firm trajectory."""
    config = DynamicsConfig(dimension=dimension, periods=2)
    economy = build_economy(config)
    innovations = np.array([[0.01, -0.01], [0.01, 0.01]])
    result = run_mesa_path(
        economy, innovation_levels(economy, innovations), innovations
    )
    assert result.capacity.shape == (3, 100, 16)
    assert np.isfinite(result.capacity).all()
    assert np.all(result.capacity >= 0)
    assert np.max(result.bounds) <= config.solve_target

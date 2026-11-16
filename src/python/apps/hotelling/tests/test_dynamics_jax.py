"""Incoming NumPy witnesses and actual JAX composition boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hotelling.dynamics.config import DynamicsConfig
from hotelling.dynamics.engine import (
    Game,
    advance,
    build_canonical_economy,
    build_economy,
    forcing,
    generate_innovations,
    initial_state,
    operator_jvp,
)
from hotelling.dynamics.observation import population_law, reporting_weights

if TYPE_CHECKING:
    from statadic.core.typing import Array

WITNESS = Path(__file__).parent / "fixtures" / "dynamics_numpy_before.npz"
STEP = 1e-6


@pytest.mark.parametrize("family", ["generic", "canonical"])
def test_captured_incoming_numpy_equations_and_certified_stages(family: str) -> None:
    """Fixed inputs isolate the backend migration from the intentional RNG change.

    The fixture was captured from incoming NumPy sources before their migration:
    generic seed 8 with six firms, four activities, sixteen nodes and dimension
    three, plus the six-firm canonical economy. No current JAX run generates it.
    """
    economy = (
        build_economy(
            DynamicsConfig(
                firms=6, activities=4, nodes=16, dimension=3, periods=15, seed=8
            )
        )
        if family == "generic"
        else build_canonical_economy(firms=6, periods=3)
    )
    with np.load(WITNESS) as saved:
        for name in (
            "sites",
            "nodes",
            "kernel",
            "cost",
            "cost_hessian",
            "revenue",
            "shock_revenue",
            "shock_cost",
        ):
            np.testing.assert_allclose(
                getattr(economy, name),
                saved[f"{family}_{name}"],
                atol=2e-15,
                rtol=2e-15,
            )
        q = jnp.asarray(saved[f"{family}_baseline"])
        direction = jnp.asarray(saved[f"{family}_direction"])
        game = Game(economy, jnp.array([0.005, -0.01]))
        actual = {
            "operator": game.operator(q),
            "flow": game.flow(q),
            "actual_revenue": game.revenue,
            "actual_cost": game.cost,
            "jvp": operator_jvp(economy, q, direction),
            "forcing": forcing(economy, q),
        }
        for name, value in actual.items():
            np.testing.assert_allclose(
                value, saved[f"{family}_{name}"], atol=2e-14, rtol=2e-14
            )
        np.testing.assert_allclose(initial_state(economy), q, atol=1e-9, rtol=0)
        moved = advance(economy, q, game.shock_state)
        np.testing.assert_allclose(
            moved.capacity, saved[f"{family}_moved"], atol=1e-9, rtol=0
        )
        assert moved.relative_bound <= economy.config.solve_target
        reported = reporting_weights(moved.capacity, 1.0)
        np.testing.assert_allclose(
            reported, saved[f"{family}_reported"], atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            population_law(reported), saved[f"{family}_population"], atol=1e-9, rtol=0
        )


def test_dynamic_game_lowering_batching_and_objective_derivatives() -> None:
    """Economic arrays remain dynamic through lowering and smooth objectives."""
    economy = build_canonical_economy(6, periods=1)
    q = jnp.asarray(
        np.linspace(0.1, 0.9, economy.cost.size).reshape(economy.cost.shape)
    )
    game = Game(economy, jnp.array([0.005, -0.01]))
    operator = jax.jit(lambda current, capacity: current.operator(capacity))
    compiled = operator.lower(game, q).compile()
    np.testing.assert_allclose(compiled(game, q), game.operator(q), atol=1e-15)
    profiles = jnp.stack((q, q * 1.1))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(operator, in_axes=(None, 0)))(game, profiles),
        jnp.stack((game.operator(q), game.operator(q * 1.1))),
        atol=1e-15,
    )

    def objective(transform: Array) -> Array:
        capacity = q @ transform
        return jnp.sum(game.operator(capacity) ** 2) + jnp.sum(
            reporting_weights(capacity, 1.0) ** 2
        )

    transform = jnp.eye(2)
    direction = jnp.array([[0.1, -0.2], [0.3, 0.05]])
    gradient = jax.jit(jax.grad(objective))(transform)
    numeric = (
        objective(transform + STEP * direction)
        - objective(transform - STEP * direction)
    ) / (2 * STEP)
    np.testing.assert_allclose(
        jnp.sum(gradient * direction), numeric, atol=2e-9, rtol=0
    )


@pytest.mark.parametrize("kind", ["mixed", "opportunity", "cost"])
def test_typed_threefry_innovation_schedule_retains_control_and_masks(
    kind: str,
) -> None:
    """The published stream schedule, burn-in and scientific masks are explicit."""
    config = DynamicsConfig(periods=20, seed=19, shock_kind=kind)
    key = jax.random.fold_in(jax.random.key(19, impl="threefry2x32"), 200)
    expected = config.shock_size * (2 * jax.random.randint(key, (20, 2), 0, 2) - 1)
    expected = expected.astype(jnp.float64).at[:12].set(0)
    if kind == "opportunity":
        expected = expected.at[:, 1].set(0)
    elif kind == "cost":
        expected = expected.at[:, 0].set(0)
    np.testing.assert_array_equal(generate_innovations(config), expected)

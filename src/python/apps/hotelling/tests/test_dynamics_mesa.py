"""Mesa lifecycle, shared agent rows and simultaneous transition checks."""

from __future__ import annotations

import numpy as np
import pytest
from hotelling.dynamics.config import DynamicsConfig
from hotelling.dynamics.engine import (
    Game,
    advance,
    build_canonical_economy,
    build_economy,
    innovation_levels,
)
from hotelling.dynamics.mesa import Firm, JointModel, run_mesa_path
from mesa.agentset import AbstractAgentSet


def test_registered_rows_follow_joint_commits_and_mesa_clock() -> None:
    """Every agent sees one joint commit; Mesa owns source/article time alignment."""
    config = DynamicsConfig(firms=4, activities=3, nodes=8, dimension=3, periods=3)
    innovations = np.array([[0.01, 0.0], [-0.01, 0.02], [0.0, 0.01]])
    economy = build_economy(config)
    levels = innovation_levels(economy, innovations)
    model = JointModel(economy, levels, innovations)
    assert isinstance(model.agents, AbstractAgentSet)
    assert len(model.agents_by_type[Firm]) == config.firms
    assert model.steps == model.time == 0
    agents = tuple(model.agents)
    assert [firm.index for firm in agents] == list(range(config.firms))
    old = model.capacity.copy()
    expected = advance(model.economy, old, innovations[0])
    with pytest.raises(RuntimeError, match="partial"):
        model.result()
    model.step()
    assert model.steps == model.time == 1
    np.testing.assert_array_equal(model.capacity, expected.capacity)
    np.testing.assert_array_equal(model.capacity_path[0], old)
    assert tuple(model.agents) == agents
    for firm in agents:
        np.testing.assert_array_equal(firm.capacity, expected.capacity[firm.index])
        with pytest.raises(TypeError, match="immutable"):
            firm.capacity[0] = 0
    model.run_model()
    assert model.steps == model.time == config.periods
    assert not model.running
    np.testing.assert_array_equal(model.result().capacity[-1], model.capacity)


def test_scalar_collection_and_reproducible_scientific_outputs() -> None:
    """Scalar DataCollector histories complement tensor artifacts without copies."""
    config = DynamicsConfig(firms=4, activities=3, nodes=8, periods=3)
    innovations = np.array([[0.01, 0.01], [0.0, -0.01], [-0.01, 0.0]])
    economy = build_economy(config)
    levels = innovation_levels(economy, innovations)
    model = JointModel(economy, levels, innovations)
    model.run_model()
    expected = run_mesa_path(economy, levels, innovations)
    actual = model.result()
    for field in (
        "capacity",
        "shock_states",
        "cash_flow",
        "net_cash_flow",
        "bounds",
        "iterations",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    diagnostics = model.datacollector.get_model_vars_dataframe()
    assert len(diagnostics) == config.periods + 1
    np.testing.assert_array_equal(diagnostics["relative_bound"][1:], actual.bounds)
    np.testing.assert_array_equal(
        diagnostics["solver_iterations"][1:], actual.iterations
    )
    np.testing.assert_array_equal(
        diagnostics["minimum_capacity"], actual.capacity.min(axis=(1, 2))
    )
    assert model.datacollector.agent_reporters == {}
    assert not np.shares_memory(actual.capacity, model.capacity_path)


def test_explicit_additive_levels_are_frozen_and_solve_the_actual_stage() -> None:
    """A supplied permanent opportunity change reaches the simultaneous game."""
    economy = build_canonical_economy(6, kappa=0.01, periods=3)
    levels = np.array([[1.0, 1.0], [1.001, 0.999], [1.001, 0.999], [1.001, 0.999]])
    model = JointModel(economy, levels)
    levels[1:] = 2
    assert not model.opportunity_levels.flags.writeable
    previous = model.capacity.copy()
    expected = advance(
        economy, previous, np.zeros(2), opportunity_levels=np.array([1.001, 0.999])
    )
    model.step()
    np.testing.assert_array_equal(model.capacity, expected.capacity)
    np.testing.assert_array_equal(model.capacity_path[0], previous)
    np.testing.assert_allclose(
        model.cash_flow[1],
        Game(economy, np.zeros(2), opportunity_levels=np.array([1.001, 0.999])).flow(
            model.capacity
        ),
    )
    model.run_model()
    assert np.max(model.result().bounds) <= economy.config.solve_target


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_explicit_levels_fail_before_any_stage(bad: float) -> None:
    """Zero, negative and nonfinite opportunities cannot enter a path."""
    economy = build_canonical_economy(6, periods=1)
    levels = np.array([[1.0, 1.0], [bad, 1.0]])
    with pytest.raises(ValueError, match="finite positive"):
        JointModel(economy, levels)

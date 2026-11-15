"""Mesa lifecycle for simultaneous, live-rival ESCAPE operating games.

AgentSet owns firm identity. A single immutable capacity matrix owns economic
state; no agent is activated independently of the joint equilibrium solve.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.experimental.scenarios.scenario import Scenario
from statadic.core.typing import Array, ArrayLike, Float  # noqa: TC002

from hotelling.dynamics.engine import (
    RNG_SCHEDULE,
    Game,
    PathResult,
    advance,
    initial_state,
    operating_cash_flow,
)

if TYPE_CHECKING:
    from hotelling.dynamics.config import DynamicsConfig
    from hotelling.dynamics.engine import Economy


class Firm(Agent["JointModel"]):
    """Stable row identity in the model's firm-by-activity capacity matrix."""

    def __init__(self, model: JointModel, index: int) -> None:
        """Register a firm with Mesa and bind its permanent matrix row."""
        super().__init__(model)
        self._index = index

    @property
    def index(self) -> int:
        """Return the immutable firm row bound at registration."""
        return self._index

    @property
    def capacity(self) -> Float[Array, " activity"]:
        """Read the current joint commit without copying or permitting mutation."""
        return self.model.capacity[self.index]


class JointModel(Model[Firm, Scenario["JointModel"]]):
    """All registered firms commit once per Mesa step, after the joint solve."""

    def __init__(
        self,
        economy: Economy,
        opportunity_levels: ArrayLike,
        innovations: ArrayLike | None = None,
        deadline: float | None = None,
    ) -> None:
        """Separate Mesa's stream from the immutable economic innovation stream."""
        config = economy.config
        super().__init__(rng=np.random.SeedSequence([config.seed, 400]))
        self.started = perf_counter()
        self.economy = economy
        self.opportunity_levels = np.array(
            opportunity_levels, dtype=np.float64, copy=True
        )
        if (
            self.opportunity_levels.shape != (config.periods + 1, config.nodes)
            or not np.isfinite(self.opportunity_levels).all()
            or np.any(self.opportunity_levels <= 0)
            or not np.array_equal(self.opportunity_levels[0], economy.revenue)
        ):
            msg = "levels require finite positive [T+1,nodes] with baseline at zero"
            raise ValueError(msg)
        self.opportunity_levels.setflags(write=False)
        self.innovations = (
            np.zeros((config.periods, 2))
            if innovations is None
            else np.array(innovations, dtype=np.float64, copy=True)
        )
        if (
            self.innovations.shape != (config.periods, 2)
            or not np.isfinite(self.innovations).all()
        ):
            msg = "innovations must be a finite (periods, 2) array"
            raise ValueError(msg)
        self.innovations.setflags(write=False)
        self.deadline = deadline
        self._initialize_path(config)
        Firm.create_agents(self, config.firms, index=list(range(config.firms)))
        self.datacollector = DataCollector(
            model_reporters={
                "minimum_capacity": "minimum_capacity",
                "relative_bound": "relative_bound",
                "solver_iterations": "solver_iterations",
                "active_set_changes": "active_set_changes",
            }
        )
        self.datacollector.collect(self)

    def _initialize_path(self, config: DynamicsConfig) -> None:
        """Allocate tensor artifacts once; scalar reporters retain no agent history."""
        shape = (config.periods + 1, config.firms, config.activities)
        self.capacity_path = np.empty(shape)
        self.shock_states = np.zeros((config.periods + 1, 2))
        self.cash_flow = np.empty((config.periods + 1, config.firms))
        self.net_cash_flow = np.empty_like(self.cash_flow)
        self.bounds = np.empty(config.periods)
        self.iterations = np.empty(config.periods, dtype=np.int64)
        self.active_set_changes = 0
        self.relative_bound: float | None = None
        self.solver_iterations: int | None = None
        self.capacity_path[0] = initial_state(self.economy, deadline=self.deadline)
        self.capacity = jnp.asarray(self.capacity_path[0])
        self.minimum_capacity = float(self.capacity.min())
        self.cash_flow[0] = operating_cash_flow(self.economy, self.capacity)
        self.net_cash_flow[0] = self.cash_flow[0]

    @override
    def step(self) -> None:
        """Solve against live rivals and publish the full next matrix together.

        Mesa increments ``steps`` and ``time`` before this method runs. Therefore
        step one consumes source innovation zero and writes article date one.
        """
        config = self.economy.config
        source = self.steps - 1
        if source >= config.periods:
            msg = "the predeclared innovation sequence is exhausted"
            raise RuntimeError(msg)
        old = self.capacity
        state = config.rho * self.shock_states[source] + self.innovations[source]
        result = advance(
            self.economy,
            old,
            state,
            opportunity_levels=self.opportunity_levels[self.steps],
            deadline=self.deadline,
        )
        self.capacity_path[self.steps] = result.capacity
        self.shock_states[self.steps] = state
        self.cash_flow[self.steps] = Game(
            self.economy, state, opportunity_levels=self.opportunity_levels[self.steps]
        ).flow(result.capacity)
        self.net_cash_flow[self.steps] = self.cash_flow[self.steps] - (
            0.5 * config.kappa * np.sum((result.capacity - old) ** 2, axis=1)
        )
        self.bounds[source] = result.relative_bound
        self.iterations[source] = result.iterations
        self.capacity = result.capacity
        self.minimum_capacity = float(self.capacity.min())
        self.relative_bound = result.relative_bound
        self.solver_iterations = result.iterations
        self.active_set_changes += result.active_set_changed
        self.running = self.steps < config.periods
        self.datacollector.collect(self)

    def result(self) -> PathResult:
        """Return a completed tensor path independent of subsequent model changes."""
        if self.running:
            msg = "a partial Mesa path cannot be presented as a completed run"
            raise RuntimeError(msg)
        return PathResult(
            economy=self.economy,
            innovations=jnp.asarray(self.innovations),
            shock_states=jnp.asarray(self.shock_states),
            capacity=jnp.asarray(self.capacity_path),
            cash_flow=jnp.asarray(self.cash_flow),
            net_cash_flow=jnp.asarray(self.net_cash_flow),
            bounds=jnp.asarray(self.bounds),
            iterations=jnp.asarray(self.iterations),
            opportunity_levels=jnp.asarray(self.opportunity_levels),
            diagnostics={
                "ordinary_rng_schedule": RNG_SCHEDULE,
                "elapsed_seconds": perf_counter() - self.started,
                "minimum_capacity": float(self.capacity_path.min()),
                "maximum_relative_bound": float(self.bounds.max()),
                "active_set_changes": self.active_set_changes,
                "cost_curvature_lower_bound": float(self.economy.modulus),
            },
        )


def run_mesa_path(
    economy: Economy,
    opportunity_levels: ArrayLike,
    innovations: ArrayLike | None = None,
    deadline: float | None = None,
) -> PathResult:
    """Run the single production lifecycle through Mesa's stopping condition."""
    model = JointModel(economy, opportunity_levels, innovations, deadline)
    model.run_model()
    return model.result()

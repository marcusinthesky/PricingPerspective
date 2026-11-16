"""Simultaneous rational-share operating dynamics on an arbitrary finite grid.

Certificates evaluate analytical strong-monotonicity bounds in float64; they
are numerical diagnostics rather than interval or formal arithmetic proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Self

import jax
import jax.numpy as jnp
from statadic.core.typing import (  # noqa: TC002  # runtime shape contracts
    Array,
    ArrayLike,
    Float,
    Int,
)

from hotelling.dynamics.config import DynamicsConfig
from hotelling.dynamics.geometry import (
    centered_direction,
    interaction,
    sobol_points,
    squared_distance,
)
from hotelling.orthant import solve_certified
from hotelling.precision import require_float64

Diagnostics = dict[str, float | int | bool | str]
BURN_IN = 12
GEOMETRY_STREAM = 100
INNOVATION_STREAM = 200
RNG_SCHEDULE = "hotelling-dynamics-threefry-fold-in-v1"
CANONICAL_MIN_FIRMS = 4


def _readonly(value: Array) -> Array:
    return jnp.asarray(value, dtype=jnp.float64)


def _traced(*values: object) -> bool:
    return any(
        isinstance(value, jax.core.Tracer)
        for value in jax.tree_util.tree_leaves(values)
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Economy:
    """Frozen primitives, independent of orchestration.

    Capacity and cost axes are ``(firm, activity)``; sites and nodes are
    ``(activity, dimension)`` and ``(node, dimension)``. The interaction kernel
    maps activity to node, and two shock factors precede the relevant axes.
    Every array boundary is validated here before the numerical solver runs.
    """

    config: DynamicsConfig
    sites: Float[Array, "*batch activity dimension"]
    nodes: Float[Array, "*batch node dimension"]
    kernel: Float[Array, "*batch activity node"]
    cost: Float[Array, "*batch firm activity"]
    cost_hessian: Float[Array, "*batch activity activity"]
    revenue: Float[Array, "*batch node"]
    modulus: Float[Array, "*batch"]
    shock_revenue: Float[Array, "*batch factor node"]
    shock_cost: Float[Array, "*batch factor firm activity"]

    def __init__(
        self,
        config: DynamicsConfig,
        sites: ArrayLike,
        nodes: ArrayLike,
        kernel: ArrayLike,
        cost: ArrayLike,
        cost_hessian: ArrayLike,
        revenue: ArrayLike,
        modulus: ArrayLike,
        shock_revenue: ArrayLike,
        shock_cost: ArrayLike,
    ) -> None:
        """Coerce external buffers once while retaining precise array fields."""
        require_float64()
        object.__setattr__(self, "config", config)
        for name, value in (
            ("sites", sites),
            ("nodes", nodes),
            ("kernel", kernel),
            ("cost", cost),
            ("cost_hessian", cost_hessian),
            ("revenue", revenue),
            ("modulus", modulus),
            ("shock_revenue", shock_revenue),
            ("shock_cost", shock_cost),
        ):
            object.__setattr__(self, name, _readonly(jnp.asarray(value)))
        self.__post_init__()

    def tree_flatten(self) -> tuple[tuple[Array, ...], DynamicsConfig]:
        """Keep numerical leaves dynamic and configuration outside the graph."""
        return (
            self.sites,
            self.nodes,
            self.kernel,
            self.cost,
            self.cost_hessian,
            self.revenue,
            self.modulus,
            self.shock_revenue,
            self.shock_cost,
        ), self.config

    @classmethod
    def tree_unflatten(cls, config: DynamicsConfig, leaves: tuple[Array, ...]) -> Self:
        """Rebuild JAX placeholders without rerunning the eager validation door."""
        result = object.__new__(cls)
        object.__setattr__(result, "config", config)
        for name, value in zip(
            (
                "sites",
                "nodes",
                "kernel",
                "cost",
                "cost_hessian",
                "revenue",
                "modulus",
                "shock_revenue",
                "shock_cost",
            ),
            leaves,
            strict=True,
        ):
            object.__setattr__(result, name, value)
        return result

    def __post_init__(self) -> None:
        """Validate shapes, nonnegativity, cost symmetry and the claimed modulus."""
        c = self.config
        shapes = {
            "sites": (c.activities, c.dimension),
            "nodes": (c.nodes, c.dimension),
            "kernel": (c.activities, c.nodes),
            "cost": (c.firms, c.activities),
            "cost_hessian": (c.activities, c.activities),
            "revenue": (c.nodes,),
            "shock_revenue": (2, c.nodes),
            "shock_cost": (2, c.firms, c.activities),
        }
        require_float64()
        batch_shape = self.cost.shape[:-2]
        for name, shape in shapes.items():
            value = _readonly(getattr(self, name))
            if value.shape != (*batch_shape, *shape):
                msg = f"{name} must have shape {(*batch_shape, *shape)}"
                raise ValueError(msg)
            object.__setattr__(self, name, value)
        modulus = jnp.asarray(self.modulus, dtype=jnp.float64)
        if modulus.shape != batch_shape:
            msg = "modulus must match the economy batch axes"
            raise ValueError(msg)
        object.__setattr__(self, "modulus", modulus)
        if _traced(self):
            return
        if not all(bool(jnp.isfinite(getattr(self, name)).all()) for name in shapes):
            msg = "economy arrays must be finite"
            raise ValueError(msg)
        if (
            jnp.any(self.kernel < 0)
            or jnp.any(self.cost < 0)
            or jnp.any(self.revenue <= 0)
        ):
            msg = "nonnegative kernel/cost and strictly positive revenue required"
            raise ValueError(msg)
        scale = jnp.maximum(
            1.0, jnp.linalg.norm(self.cost_hessian, ord=2, axis=(-2, -1))
        )
        slack = 64 * jnp.finfo(jnp.float64).eps * c.activities * scale
        asymmetric = jnp.max(
            jnp.abs(self.cost_hessian - jnp.swapaxes(self.cost_hessian, -1, -2)),
            axis=(-2, -1),
        )
        if jnp.any(asymmetric > slack):
            msg = "cost Hessian must be symmetric before using a modulus"
            raise ValueError(msg)
        minimum = jnp.linalg.eigvalsh(self.cost_hessian)[..., 0]
        if jnp.any(
            ~jnp.isfinite(modulus) | (modulus <= 0) | (modulus > minimum + slack)
        ):
            msg = "modulus must be positive and bounded by cost curvature"
            raise ValueError(msg)


def build_economy(config: DynamicsConfig) -> Economy:
    """Build reproducible nested sites, nodes and permanent firm capabilities."""
    require_float64()
    sites = sobol_points(
        config.activities, config.dimension, config.seed, GEOMETRY_STREAM
    )
    nodes = sobol_points(
        config.nodes, config.dimension, config.seed, GEOMETRY_STREAM + 1
    )
    centers = sobol_points(
        config.firms, config.dimension, config.seed, GEOMETRY_STREAM + 2
    )
    separation = squared_distance(centers, sites)
    hessian = config.own_curvature * jnp.eye(
        config.activities
    ) + config.congestion * interaction(sites, sites, config.tau)
    scale = max(1.0, float(jnp.linalg.norm(hessian, ord=2)))
    roundoff = 64 * jnp.finfo(jnp.float64).eps * config.activities * scale
    modulus = float(jnp.linalg.eigvalsh(hessian).min()) - roundoff
    revenue_shock = jnp.zeros((2, config.nodes))
    revenue_shock = revenue_shock.at[0].set(centered_direction(nodes.mean(axis=1)))
    cost_shock = jnp.zeros((2, config.firms, config.activities))
    cost_shock = cost_shock.at[1].set(centered_direction(separation))
    return Economy(
        config=config,
        sites=sites,
        nodes=nodes,
        kernel=interaction(sites, nodes, config.tau),
        cost=0.1 + config.advantage * separation,
        cost_hessian=hessian,
        revenue=jnp.full(config.nodes, config.firms / config.nodes),
        modulus=modulus,
        shock_revenue=revenue_shock,
        shock_cost=cost_shock,
    )


def canonical_capacity(firms: int) -> Float[Array, "firm activity"]:
    """Construct distinct interior shares averaging one half."""
    if type(firms) is not int or firms < CANONICAL_MIN_FIRMS:
        msg = "canonical all-firm fixture requires at least four firms"
        raise ValueError(msg)
    require_float64()
    share = jnp.arange(1, firms + 1, dtype=jnp.float64) / (firms + 1)
    return jnp.column_stack((share, 1 - share))


def build_canonical_economy(
    firms: int = 6, *, kappa: float = 1.0, periods: int = 60, seed: int = 0
) -> Economy:
    """Build the symmetric two-activity numerical baseline."""
    q = canonical_capacity(firms)
    denominator = 1 + firms / 2
    nu = 1 / (4 * denominator**2)
    config = DynamicsConfig(
        firms=firms,
        activities=2,
        nodes=2,
        dimension=1,
        outside=1,
        own_curvature=nu,
        congestion=0,
        kappa=kappa,
        periods=periods,
        seed=seed,
    )
    cost_shock = jnp.zeros((2, firms, 2))
    cost_shock = cost_shock.at[1].set(centered_direction(q))
    return Economy(
        config=config,
        sites=jnp.array([[0.0], [1.0]]),
        nodes=jnp.array([[0.0], [1.0]]),
        kernel=jnp.eye(2),
        cost=(denominator - q) / denominator**2 - nu * q,
        cost_hessian=nu * jnp.eye(2),
        revenue=jnp.ones(2),
        modulus=nu,
        shock_revenue=jnp.array([[1.0, -1.0], [0.0, 0.0]]),
        shock_cost=cost_shock,
    )


def _profile(economy: Economy, q: ArrayLike) -> Float[Array, "firm activity"]:
    q = jnp.asarray(q, dtype=jnp.float64)
    if q.shape != economy.cost.shape or (
        not _traced(q) and (not jnp.isfinite(q).all() or jnp.any(q < 0))
    ):
        msg = "capacity must be a finite nonnegative joint profile"
        raise ValueError(msg)
    return jnp.asarray(q, dtype=jnp.float64)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Game:
    """Live-rival stage with a capacity anchor and two economic shock factors."""

    economy: Economy
    shock_state: Float[Array, "*batch factor"]
    anchor: Float[Array, "*batch firm activity"] | None = None
    kappa: Float[Array, "*batch"] | float = 0.0
    opportunity_levels: Float[Array, "*batch node"] | None = None

    def __init__(
        self,
        economy: Economy,
        shock_state: ArrayLike,
        anchor: ArrayLike | None = None,
        kappa: ArrayLike = 0.0,
        opportunity_levels: ArrayLike | None = None,
    ) -> None:
        """Prepare external state without hiding numerical leaves from JAX."""
        require_float64()
        object.__setattr__(self, "economy", economy)
        object.__setattr__(self, "shock_state", _readonly(jnp.asarray(shock_state)))
        object.__setattr__(
            self, "anchor", None if anchor is None else _readonly(jnp.asarray(anchor))
        )
        object.__setattr__(self, "kappa", jnp.asarray(kappa, dtype=jnp.float64))
        object.__setattr__(
            self,
            "opportunity_levels",
            None
            if opportunity_levels is None
            else _readonly(jnp.asarray(opportunity_levels)),
        )
        self.__post_init__()

    def tree_flatten(
        self,
    ) -> tuple[tuple[Economy, Array, Array | None, Array | float, Array | None], None]:
        """Expose actual economic state to compiled operators."""
        return (
            self.economy,
            self.shock_state,
            self.anchor,
            self.kappa,
            self.opportunity_levels,
        ), None

    @classmethod
    def tree_unflatten(
        cls,
        metadata: None,
        leaves: tuple[Economy, Array, Array | None, Array | float, Array | None],
    ) -> Self:
        """Preserve validated structure during tracing and compiler inspection."""
        del metadata
        result = object.__new__(cls)
        for name, value in zip(
            ("economy", "shock_state", "anchor", "kappa", "opportunity_levels"),
            leaves,
            strict=True,
        ):
            object.__setattr__(result, name, value)
        return result

    def __post_init__(self) -> None:
        """Freeze the supplied state and validate the optional adjustment anchor."""
        state = _readonly(self.shock_state)
        if state.shape != (*self.economy.cost.shape[:-2], 2) or (
            not _traced(state) and not jnp.isfinite(state).all()
        ):
            msg = "shock state must be a finite two-vector"
            raise ValueError(msg)
        object.__setattr__(self, "shock_state", state)
        if self.opportunity_levels is not None:
            levels = _readonly(self.opportunity_levels)
            if levels.shape != self.economy.revenue.shape or (
                not _traced(levels)
                and (not jnp.isfinite(levels).all() or jnp.any(levels <= 0))
            ):
                msg = "opportunity levels must be finite positive node values"
                raise ValueError(msg)
            object.__setattr__(self, "opportunity_levels", levels)
        if not _traced(self.kappa) and (
            jnp.any(~jnp.isfinite(self.kappa)) or jnp.any(self.kappa < 0)
        ):
            msg = "kappa must be finite and nonnegative"
            raise ValueError(msg)
        if self.anchor is not None:
            object.__setattr__(
                self, "anchor", _readonly(_profile(self.economy, self.anchor))
            )
        elif not _traced(self.kappa) and jnp.any(self.kappa > 0):
            msg = "a positive adjustment weight requires an anchor"
            raise ValueError(msg)

    @property
    def revenue(self) -> Array:
        """Use supplied positive levels, or the benchmark's normalized tilt law."""
        if self.opportunity_levels is not None:
            return self.opportunity_levels
        tilt = self.shock_state @ self.economy.shock_revenue
        weight = self.economy.revenue * jnp.exp(tilt - jnp.max(tilt))
        value = weight * (self.economy.revenue.sum() / weight.sum())
        if not _traced(value) and (
            not jnp.isfinite(value).all() or jnp.any(value <= 0)
        ):
            msg = "shock leaves the numerically positive opportunity domain"
            raise ValueError(msg)
        return _readonly(value)

    @property
    def cost(self) -> Array:
        """Permanent capabilities multiplied by the finite cost-shock field."""
        value = self.economy.cost * jnp.exp(
            jnp.einsum("k,kia->ia", self.shock_state, self.economy.shock_cost)
        )
        if not _traced(value) and not jnp.isfinite(value).all():
            msg = "shock leaves the numerically finite cost domain"
            raise ValueError(msg)
        return _readonly(value)

    def operator(self, q: ArrayLike) -> Float[Array, "firm activity"]:
        """Negative own operating-profit gradient plus anchored adjustment."""
        q = jnp.asarray(q, dtype=jnp.float64)
        e = self.economy
        appeal = q @ e.kernel
        denominator = e.config.outside + appeal.sum(axis=0)
        rivals = denominator - appeal
        value = (
            self.cost
            + q @ e.cost_hessian
            - (self.revenue * rivals / denominator**2) @ e.kernel.T
        )
        if self.anchor is not None:
            value += self.kappa * (q - self.anchor)
        return value

    def revenues(self, q: ArrayLike) -> Float[Array, " firm"]:
        """Compute captured value with the outside share left unallocated."""
        q = jnp.asarray(q, dtype=jnp.float64)
        appeal = q @ self.economy.kernel
        return (
            appeal / (self.economy.config.outside + appeal.sum(axis=0))
        ) @ self.revenue

    def flow(self, q: ArrayLike) -> Float[Array, " firm"]:
        """Operating payout excludes the transitory adjustment penalty."""
        q = jnp.asarray(q, dtype=jnp.float64)
        value = (
            self.revenues(q)
            - jnp.sum(self.cost * q, axis=1)
            - 0.5 * jnp.einsum("ia,ab,ib->i", q, self.economy.cost_hessian, q)
        )
        if not _traced(value) and not jnp.isfinite(value).all():
            msg = "nonfinite operating cash flow"
            raise ArithmeticError(msg)
        return value


def operating_cash_flow(
    economy: Economy, q: ArrayLike, shock_state: ArrayLike | None = None
) -> Float[Array, " firm"]:
    """Evaluate the primary payout from validated latent capacities."""
    state = jnp.zeros(2) if shock_state is None else shock_state
    return Game(economy, state).flow(_profile(economy, q))


def operator_jvp(
    economy: Economy, q: ArrayLike, v: ArrayLike
) -> Float[Array, "firm activity"]:
    """Baseline economic Jacobian product, without a dense firm-by-firm matrix."""
    q, v = jnp.asarray(q), jnp.asarray(v)
    appeal = q @ economy.kernel
    change = v @ economy.kernel
    denominator = economy.config.outside + appeal.sum(axis=0)
    total_change = change.sum(axis=0)
    derivative = economy.revenue * (
        change / denominator**2
        + (denominator - 2 * appeal) * total_change / denominator**3
    )
    return v @ economy.cost_hessian + derivative @ economy.kernel.T


def forcing(economy: Economy, q: ArrayLike) -> Float[Array, "firm activity factor"]:
    """Minus the baseline shock derivative of the actual economic operator."""
    q = jnp.asarray(q)
    appeal = q @ economy.kernel
    denominator = economy.config.outside + appeal.sum(axis=0)
    centered = (
        economy.shock_revenue
        - (economy.shock_revenue @ economy.revenue / economy.revenue.sum())[:, None]
    )
    revenue_derivative = centered * economy.revenue
    return jnp.einsum(
        "kl,il,al->iak",
        revenue_derivative,
        (denominator - appeal) / denominator**2,
        economy.kernel,
    ) - jnp.moveaxis(economy.cost * economy.shock_cost, 0, -1)


@dataclass(frozen=True)
class StageResult:
    """One simultaneously committed solve and its numerical certificate."""

    capacity: Float[Array, "firm activity"]
    relative_bound: float
    iterations: int
    active_set_changed: bool


def _deadline(economy: Economy, deadline: float | None) -> float:
    own = perf_counter() + economy.config.solve_seconds
    return own if deadline is None else min(own, deadline)


def initial_state(
    economy: Economy, *, deadline: float | None = None
) -> Float[Array, "firm activity"]:
    """Compute the unshocked stationary joint profile with a verified bound."""
    start = jnp.full(economy.cost.shape, 1 / economy.config.activities)
    q, _, _ = solve_certified(
        Game(economy, jnp.zeros(2)),
        start,
        economy.modulus,
        target=economy.config.solve_target,
        limit=economy.config.max_iterations,
        deadline=_deadline(economy, deadline),
    )
    return _profile(economy, q)


def advance(
    economy: Economy,
    q: ArrayLike,
    shock_state: ArrayLike,
    *,
    opportunity_levels: ArrayLike | None = None,
    deadline: float | None = None,
) -> StageResult:
    """Solve an anchored simultaneous game with live rivals and a fixed old state."""
    old = _profile(economy, q)
    game = Game(
        economy,
        shock_state,
        anchor=old,
        kappa=economy.config.kappa,
        opportunity_levels=opportunity_levels,
    )
    modulus = economy.modulus + economy.config.kappa
    moved, bound, iterations = solve_certified(
        game,
        old,
        modulus,
        target=economy.config.solve_target,
        limit=economy.config.max_iterations,
        deadline=_deadline(economy, deadline),
    )
    _profile(economy, moved)
    relative = bound / max(1.0, float(jnp.linalg.norm(moved)))
    return StageResult(
        capacity=moved,
        relative_bound=relative,
        iterations=iterations,
        active_set_changed=bool(jnp.any((old > 0) != (moved > 0))),
    )


def generate_innovations(config: DynamicsConfig) -> Float[Array, "period factor"]:
    """Independent bounded innovations; the first year is a stationary control."""
    require_float64()
    key = jax.random.fold_in(
        jax.random.key(config.seed, impl="threefry2x32"), INNOVATION_STREAM
    )
    innovations = config.shock_size * (
        2 * jax.random.randint(key, (config.periods, 2), 0, 2) - 1
    )
    innovations = innovations.astype(jnp.float64).at[:BURN_IN].set(0)
    if config.shock_kind == "opportunity":
        innovations = innovations.at[:, 1].set(0)
    elif config.shock_kind == "cost":
        innovations = innovations.at[:, 0].set(0)
    return innovations


def innovation_levels(
    economy: Economy, innovations: ArrayLike
) -> Float[Array, "date node"]:
    """Translate the AR(1) benchmark's primitive law into explicit dated levels.

    Permanent additive news supplies its levels directly and does not use this
    benchmark transformation. This function advances no operating state.
    """
    innovations = jnp.asarray(innovations, dtype=jnp.float64)
    if innovations.shape != (economy.config.periods, 2):
        msg = "innovations must have shape (periods, 2)"
        raise ValueError(msg)

    def step(state: Array, innovation: Array) -> tuple[Array, Array]:
        state = economy.config.rho * state + innovation
        return state, Game(economy, state).revenue

    _, levels = jax.lax.scan(step, jnp.zeros(2), jnp.asarray(innovations))
    return jnp.concatenate((economy.revenue[None], levels), axis=0)


@dataclass(frozen=True)
class PathResult:
    """Dated opportunity levels determine simultaneous capacity, cash and net payout.

    Capacity axes are ``(date, firm, activity)``; payouts are ``(date, firm)``.
    The initial date has no source innovation or anchored-stage certificate.
    """

    economy: Economy
    innovations: Float[Array, "period factor"]
    shock_states: Float[Array, "date factor"]
    capacity: Float[Array, "date firm activity"]
    cash_flow: Float[Array, "date firm"]
    net_cash_flow: Float[Array, "date firm"]
    bounds: Float[Array, " period"]
    iterations: Int[Array, " period"]
    opportunity_levels: Float[Array, "date node"]
    diagnostics: Diagnostics = field(default_factory=dict)

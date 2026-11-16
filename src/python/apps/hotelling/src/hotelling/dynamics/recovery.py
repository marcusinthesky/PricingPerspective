"""Frozen economic histories for the directional-estimator recovery experiment.

Only operating, valuation and observation primitives live here. Pipeline owns
dated pooling and the estimator. These finite nonlinear paths are not equity
prices: financial news uses the separate permanent-response tangent resolvent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from statadic.core.typing import Array, ArrayLike, Float  # noqa: TC002

from hotelling.dynamics.config import DynamicsConfig
from hotelling.dynamics.engine import (
    build_canonical_economy,
    build_economy,
    canonical_capacity,
    initial_state,
)
from hotelling.dynamics.mesa import run_mesa_path
from hotelling.dynamics.valuation import derive_model, permanent_tilt_readout
from hotelling.precision import require_float64

if TYPE_CHECKING:
    from hotelling.dynamics.engine import PathResult

CAPACITY_AXES = 3
ACTIVITIES = 2

type FloatArray = Float[Array, "*axes"]
FIRMS = 100
PERIODS = 60
NEWS_TICKS = (18, 30, 42, 54)
NEWS_SIZE = 0.001
MAX_CUMULATIVE_NEWS = 0.004
ANNUAL_DISCOUNT = 0.95
REPORTING_AMPLITUDES = (0.0, 0.001, 0.0025, 0.005, 0.01)
DIMENSIONS = (2, 4, 8, 16)


@dataclass(frozen=True)
class RecoveryHistory:
    """One fully enumerated source-time news history and its article-date levels."""

    signs: tuple[int, ...]
    news: Float[Array, " period"]
    levels: Float[Array, "date node"]
    cumulative: Float[Array, " date"]


def histories(*, shock_multiplier: float = 1.0) -> tuple[RecoveryHistory, ...]:
    """Enumerate all sixteen histories; no random selection or excluded path."""
    require_float64()
    if not np.isfinite(shock_multiplier) or shock_multiplier <= 0:
        msg = "shock multiplier must be finite and positive"
        raise ValueError(msg)
    result: list[RecoveryHistory] = []
    for signs in product((-1, 1), repeat=len(NEWS_TICKS)):
        news = np.zeros(PERIODS)
        news[list(NEWS_TICKS)] = NEWS_SIZE * shock_multiplier * np.asarray(signs)
        cumulative = np.concatenate(([0.0], np.cumsum(news)))
        levels = 1 + cumulative[:, None] * np.array([1.0, -1.0])
        for array in (news, cumulative, levels):
            array.setflags(write=False)
        result.append(
            RecoveryHistory(
                signs, jnp.asarray(news), jnp.asarray(levels), jnp.asarray(cumulative)
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class RecoveryEconomy:
    """Same-history operating shares, affine-share benchmark and financial news."""

    operating: PathResult
    affine_shares: Float[Array, "date firm activity"]
    financial_news: Float[Array, "period firm"]
    minimum_affine_capacity: float
    minimum_baseline_price: float
    active_domain_violations: int
    minimum_opportunity: float


def solve_history(
    history: RecoveryHistory, *, deadline: float | None = None
) -> RecoveryEconomy:
    """Use nodeD anchoring and a monthly infinite permanent valuation resolvent."""
    denominator = 1 + FIRMS / 2
    node_d = 1 / (4 * denominator**2) + 1 / denominator**2
    economy = build_canonical_economy(FIRMS, kappa=node_d, periods=PERIODS)
    economy = replace(
        economy,
        config=replace(economy.config, discount=ANNUAL_DISCOUNT ** (1 / 12)),
    )
    baseline = canonical_capacity(FIRMS)
    model = derive_model(
        economy,
        baseline,
        opportunity_directions=jnp.array([[1.0, -1.0], [0.0, 0.0]]),
        deadline=deadline,
    )
    readout = permanent_tilt_readout(model, 0)
    operating = run_mesa_path(economy, history.levels, deadline=deadline)
    deviation = jnp.zeros_like(baseline)
    affine = jnp.zeros_like(operating.capacity)
    affine = affine.at[0].set(baseline)
    minimum_capacity = float(baseline.min())
    for tick in range(1, PERIODS + 1):
        deviation, _ = model.step(
            deviation, jnp.stack((history.cumulative[tick], jnp.asarray(0.0)))
        )
        affine = affine.at[tick].set(
            baseline + deviation - baseline * deviation.sum(axis=1, keepdims=True)
        )
        minimum_capacity = min(minimum_capacity, float(np.min(baseline + deviation)))
    if minimum_capacity <= 0 or np.any(affine <= 0):
        msg = "canonical history leaves the positive affine-response domain"
        raise ArithmeticError(msg)
    return RecoveryEconomy(
        operating,
        affine,
        history.news[:, None] * readout.return_loading[None, :],
        minimum_capacity,
        float(model.baseline_prices.min()),
        int(np.count_nonzero((operating.capacity > 0) != model.active[None, ...])),
        float(history.levels.min()),
    )


def solve_dimensional_history(
    history: RecoveryHistory, dimension: int, *, deadline: float | None = None
) -> RecoveryEconomy:
    """Bound geometry dimension, holding firms, activities, nodes and kappa fixed.

    This numerical extension has no inherited two-opportunity sign claim. The
    additive primitive direction is the generic economy's bounded opportunity
    direction; its units are unchanged in the permanent valuation derivative.
    """
    if dimension not in DIMENSIONS:
        msg = "appendix dimensions are fixed at 2, 4, 8 and 16"
        raise ValueError(msg)
    economy = build_economy(
        DynamicsConfig(
            firms=FIRMS,
            activities=8,
            nodes=64,
            dimension=dimension,
            kappa=1,
            periods=PERIODS,
            discount=ANNUAL_DISCOUNT ** (1 / 12),
        )
    )
    baseline = initial_state(economy, deadline=deadline)
    directions = jnp.stack((economy.shock_revenue[0], jnp.zeros(64)))
    model = derive_model(
        economy, baseline, opportunity_directions=directions, deadline=deadline
    )
    levels = economy.revenue + history.cumulative[:, None] * directions[0]
    operating = run_mesa_path(economy, levels, deadline=deadline)
    deviation = jnp.zeros_like(baseline)
    affine = jnp.zeros_like(operating.capacity)
    totals = baseline.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        msg = "appendix baseline has an empty firm"
        raise ArithmeticError(msg)
    shares = baseline / totals
    affine = affine.at[0].set(shares)
    minimum = float(baseline.min())
    violations = int(np.count_nonzero((operating.capacity > 0) != model.active))
    for tick in range(1, PERIODS + 1):
        deviation, _ = model.step(
            deviation, jnp.stack((history.cumulative[tick], jnp.asarray(0.0)))
        )
        q = baseline + deviation
        minimum = min(minimum, float(q.min()))
        violations += int(np.count_nonzero((q > 0) != model.active))
        affine = affine.at[tick].set(
            shares
            + (deviation - shares * deviation.sum(axis=1, keepdims=True)) / totals
        )
    if np.any(affine < 0):
        msg = "appendix history leaves the nonnegative affine-share domain"
        raise ArithmeticError(msg)
    readout = permanent_tilt_readout(model, 0)
    return RecoveryEconomy(
        operating,
        affine,
        history.news[:, None] * readout.return_loading,
        minimum,
        float(model.baseline_prices.min()),
        violations,
        float(levels.min()),
    )


def reporting_laws(
    capacities: ArrayLike, cumulative: ArrayLike, amplitude: float
) -> FloatArray:
    """Symmetric unit-axis atom laws with intensity exp(+-amplitude*u).

    Atoms are ordered (+axis0,-axis0,+axis1,-axis1). The reporting-only
    counterfactual calls this function with baseline capacities at every date.
    """
    capacities, cumulative = jnp.asarray(capacities), jnp.asarray(cumulative)
    traced = isinstance(capacities, jax.core.Tracer) or isinstance(
        cumulative, jax.core.Tracer
    )
    if (
        capacities.ndim != CAPACITY_AXES
        or capacities.shape[2] < 1
        or cumulative.shape != (len(capacities),)
        or (
            not traced
            and (
                not jnp.isfinite(capacities).all()
                or jnp.any(capacities < 0)
                or (capacities.shape[2] == ACTIVITIES and jnp.any(capacities <= 0))
                or jnp.any(capacities.sum(axis=-1) <= 0)
                or not jnp.isfinite(cumulative).all()
                or not jnp.isfinite(amplitude)
                or amplitude < 0
                or (amplitude > 0 and capacities.shape[2] != ACTIVITIES)
            )
        )
    ):
        msg = "reporting requires positive two-activity capacities and finite news"
        raise ValueError(msg)
    if amplitude == 0 and capacities.shape[2] != ACTIVITIES:
        share = capacities / capacities.sum(axis=-1, keepdims=True)
        return jnp.repeat(share / 2, 2, axis=-1)
    log_intensity = (
        amplitude
        * cumulative[:, None, None]
        / MAX_CUMULATIVE_NEWS
        * jnp.array([1.0, -1.0])
    )
    log_weight = jnp.log(capacities) + log_intensity
    weight = jnp.exp(log_weight - log_weight.max(axis=-1, keepdims=True))
    share = weight / weight.sum(axis=-1, keepdims=True)
    return jnp.repeat(share / 2, 2, axis=-1)

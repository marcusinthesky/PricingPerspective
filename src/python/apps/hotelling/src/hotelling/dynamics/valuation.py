"""Local operating and valuation readouts from the actual capacity game.

The derivatives are restricted to the baseline active face. Financial news uses
a conditional mean-zero AR(1) law and constant discounting. The readout gives
first-order ex-dividend values from the local response and news law.
All solves use the campaign's float64 and ``1e-10`` residual contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from statadic.core.typing import Array, Bool, Float

from hotelling.dynamics.engine import (
    Game,
    forcing,
    operating_cash_flow,
    operator_jvp,
)
from hotelling.precision import require_float64

if TYPE_CHECKING:
    from statadic.core.typing import ArrayLike

    from hotelling.dynamics.engine import Economy

type FloatArray = Float[Array, "..."]
type ActiveArray = Bool[Array, "*batch firm activity"]
type Diagnostics = dict[str, float | int | bool]

SOLVE_TARGET = 1e-10
MAX_ITERATIONS = 200
MATRIX_RANK = 2
SHOCK_COUNT = 2


def _check_deadline(deadline: float | None, *, campaign: bool = False) -> None:
    if deadline is not None and perf_counter() >= deadline:
        scope = (
            "local valuation exceeded the campaign"
            if campaign
            else "local linear solve exceeded its"
        )
        msg = f"{scope} time limit"
        raise TimeoutError(msg)


def _restricted_matvec(
    economy: Economy,
    baseline: Float[Array, "firm activity"],
    active: ActiveArray,
    shift: Float[Array, ""] | float,
    vector: Float[Array, "*batch coordinate"],
) -> Float[Array, "*batch coordinate"]:
    direction = vector.reshape(baseline.shape) * active
    result = operator_jvp(economy, baseline, direction) + shift * direction
    return jnp.where(active, result, vector.reshape(direction.shape)).ravel()


def _spectral_precondition(
    baseline: Float[Array, "firm activity"],
    eigenvalues: Float[Array, "firm activity"],
    eigenvectors: Float[Array, "firm activity activity"],
    vector: Float[Array, "*batch coordinate"],
) -> Float[Array, "*batch coordinate"]:
    shaped = vector.reshape(baseline.shape)
    rotated = jnp.einsum("iab,ia->ib", eigenvectors, shaped)
    return jnp.einsum("iab,ib->ia", eigenvectors, rotated / eigenvalues).ravel()


@partial(jax.jit, static_argnames=("width",))
def _gmres_block(
    economy: Economy,
    baseline: Float[Array, "firm activity"],
    active: ActiveArray,
    shift: Float[Array, ""] | float,
    eigenvalues: Float[Array, "firm activity"],
    eigenvectors: Float[Array, "firm activity activity"],
    rhs: Float[Array, "*batch coordinate"],
    initial: Float[Array, "*batch coordinate"],
    *,
    width: int,
) -> tuple[Float[Array, "*batch coordinate"], Float[Array, ""]]:
    """Run one maintained GMRES restart and measure the original residual."""

    def matvec(vector: FloatArray) -> FloatArray:
        return _restricted_matvec(economy, baseline, active, shift, vector)

    def precondition(vector: FloatArray) -> FloatArray:
        return _spectral_precondition(baseline, eigenvalues, eigenvectors, vector)

    solved, _info = gmres(
        matvec,
        rhs,
        x0=initial,
        M=precondition,
        tol=SOLVE_TARGET / 10,
        atol=0.0,
        restart=width,
        maxiter=1,
        solve_method="incremental",
    )
    relative = jnp.linalg.norm(matvec(solved) - rhs) / jnp.linalg.norm(rhs)
    return solved, relative


def payout_derivative(
    economy: Economy, baseline: ArrayLike, direction: ArrayLike
) -> FloatArray:
    """Differentiate every firm's operating payout, including rival responses."""
    baseline, direction = jnp.asarray(baseline), jnp.asarray(direction)
    appeal = baseline @ economy.kernel
    total = economy.config.outside + appeal.sum(axis=0)
    change = direction @ economy.kernel
    share_change = change / total - appeal * change.sum(axis=0) / total**2
    cost_gradient = economy.cost + baseline @ economy.cost_hessian
    return share_change @ economy.revenue - jnp.sum(cost_gradient * direction, axis=1)


def payout_shock_derivative(economy: Economy, baseline: ArrayLike) -> FloatArray:
    """Differentiate normalized opportunity values and primitive firm costs."""
    baseline = jnp.asarray(baseline)
    appeal = baseline @ economy.kernel
    total = economy.config.outside + appeal.sum(axis=0)
    mean_tilt = economy.shock_revenue @ economy.revenue / economy.revenue.sum()
    revenue_change = economy.revenue * (economy.shock_revenue - mean_tilt[:, None])
    revenue_loading = (appeal / total) @ revenue_change.T
    cost_loading = jnp.einsum("ia,kia->ik", baseline * economy.cost, economy.shock_cost)
    return revenue_loading - cost_loading


@dataclass(frozen=True)
class _RestrictedSystem:
    """A matrix-free active-face operator and own-block spectral preconditioner."""

    economy: Economy
    baseline: Float[Array, "firm activity"]
    active: ActiveArray
    shift: float
    eigenvalues: Float[Array, "firm activity"]
    eigenvectors: Float[Array, "firm activity activity"]
    deadline: float | None = None

    def matvec(self, vector: FloatArray) -> FloatArray:
        """Apply the restricted Jacobian without constructing a joint matrix."""
        return _restricted_matvec(
            self.economy, self.baseline, self.active, self.shift, vector
        )

    def precondition(self, vector: FloatArray) -> FloatArray:
        """Solve own blocks through their positive spectral decomposition."""
        return _spectral_precondition(
            self.baseline, self.eigenvalues, self.eigenvectors, vector
        )

    def solve(
        self, rhs: ArrayLike, *, deadline: float | None = None
    ) -> tuple[FloatArray, float]:
        """Certify GMRES by its unpreconditioned relative residual.

        Deadlines are cooperative: a synchronized restart of at most 100
        Krylov steps (including first compilation) is the interruption unit.
        Reserved Krylov widths never exceed the existing 200-step budget.
        GMRES's ``info`` does not certify convergence. This host-certified
        adapter makes no derivative guarantee for the solved equilibrium.
        """
        started = perf_counter()
        campaign_stop = self.deadline
        if deadline is not None:
            campaign_stop = (
                deadline if campaign_stop is None else min(deadline, campaign_stop)
            )
        _check_deadline(campaign_stop, campaign=True)
        stop = started + self.economy.config.solve_seconds
        if campaign_stop is not None:
            stop = min(stop, campaign_stop)
        _check_deadline(stop)
        rhs = jnp.asarray(rhs)
        flat = (rhs * self.active).ravel()
        # Max-absolute scaling preserves representable decaying impulses whose
        # squared Euclidean norms would underflow before Krylov iteration.
        scale = float(jnp.max(jnp.abs(flat)))
        _check_deadline(stop)
        if scale == 0:
            zero = jax.block_until_ready(jnp.zeros_like(rhs))
            _check_deadline(stop)
            return zero, 0.0
        scaled_rhs = flat / scale
        solved = jnp.zeros_like(flat)
        remaining = min(MAX_ITERATIONS, self.economy.config.max_iterations)
        relative = float("inf")
        while remaining > 0:
            _check_deadline(stop)
            width = min(100, flat.size, remaining)
            solved, measured = _gmres_block(
                self.economy,
                self.baseline,
                self.active,
                self.shift,
                self.eigenvalues,
                self.eigenvectors,
                scaled_rhs,
                solved,
                width=width,
            )
            solved, measured = jax.block_until_ready((solved, measured))
            _check_deadline(stop)
            relative = float(measured)
            if jnp.isfinite(relative) and relative <= SOLVE_TARGET:
                result = jax.block_until_ready((scale * solved).reshape(rhs.shape))
                _check_deadline(stop)
                return result, relative
            if not jnp.isfinite(relative):
                break
            remaining -= width
        msg = (
            f"local linear solve failed: residual={relative}, "
            "reserved Krylov budget exhausted"
        )
        raise ArithmeticError(msg)


def _system(
    economy: Economy,
    baseline: FloatArray,
    active: ActiveArray,
    shift: float,
    deadline: float | None = None,
) -> _RestrictedSystem:
    _check_deadline(deadline, campaign=True)
    appeal = baseline @ economy.kernel
    total = economy.config.outside + appeal.sum(axis=0)
    own_weight = 2 * economy.revenue * (total - appeal) / total**3
    blocks = jnp.einsum(
        "al,il,bl->iab", economy.kernel, own_weight, economy.kernel, optimize=True
    )
    blocks += economy.cost_hessian + shift * jnp.eye(baseline.shape[1])
    blocks *= active[:, :, None] * active[:, None, :]
    indices = jnp.arange(baseline.shape[1])
    blocks = blocks.at[:, indices, indices].add(~active)
    values, vectors = jax.block_until_ready(jnp.linalg.eigh(blocks))
    _check_deadline(deadline, campaign=True)
    if not jnp.isfinite(values).all() or jnp.any(values <= 0):
        msg = "local preconditioner requires positive own curvature"
        raise ArithmeticError(msg)
    return _RestrictedSystem(
        economy, baseline, active, shift, values, vectors, deadline
    )


def _jacobian_bound(economy: Economy, baseline: FloatArray) -> float:
    """Bound the joint Jacobian norm using the local kernel factorization."""
    appeal = baseline @ economy.kernel
    total = economy.config.outside + appeal.sum(axis=0)
    cross = economy.revenue * (total - 2 * appeal) / total**3
    own_remainder = float(jnp.max(economy.revenue / total**2))
    weight = own_remainder + baseline.shape[0] * float(jnp.max(jnp.abs(cross)))
    return float(jnp.linalg.norm(economy.cost_hessian, 2)) + weight * float(
        jnp.linalg.norm(economy.kernel, 2) ** 2
    )


@dataclass(frozen=True)
class LocalModel:
    """Restricted tangent dynamics and conditional financial innovation loadings."""

    economy: Economy
    baseline: Float[Array, "firm activity"]
    active: ActiveArray
    forcing: Float[Array, "firm activity factor"]
    response: Float[Array, "firm activity factor"]
    discounted_capacity: Float[Array, "firm activity factor"]
    payout_direct: Float[Array, "firm factor"]
    payout_loading: Float[Array, "firm factor"]
    total_payoff_loading: Float[Array, "firm factor"]
    price_loading: Float[Array, "firm factor"]
    return_loading: Float[Array, "firm factor"]
    baseline_flow: Float[Array, "*batch firm"]
    baseline_prices: Float[Array, "*batch firm"]
    diagnostics: Diagnostics
    _stage_system: _RestrictedSystem

    @property
    def deadline(self) -> float | None:
        """Return the campaign deadline shared by this model's local solves."""
        return self._stage_system.deadline

    def step(
        self,
        previous: ArrayLike,
        shock_state: ArrayLike,
        *,
        deadline: float | None = None,
    ) -> tuple[FloatArray, float]:
        """Solve the anchored tangent stage, returning its relative residual."""
        _check_deadline(deadline, campaign=True)
        _check_deadline(self._stage_system.deadline, campaign=True)
        previous, shock_state = jnp.asarray(previous), jnp.asarray(shock_state)
        drive = jnp.einsum("iak,k->ia", self.forcing, shock_state)
        return self._stage_system.solve(
            self.economy.config.kappa * previous + drive, deadline=deadline
        )

    def discounted_permanent_capacity(self, factor: int) -> FloatArray:
        """Sum the delayed permanent response, preserving this model's deadline."""
        config = self.economy.config
        resolvent = _system(
            self.economy,
            self.baseline,
            self.active,
            (1 - config.discount) * config.kappa,
            self._stage_system.deadline,
        )
        summed, _ = resolvent.solve(
            config.discount / (1 - config.discount) * self.forcing[..., factor]
        )
        return summed


def _primitive_derivatives(
    economy: Economy, q: FloatArray, opportunity_directions: ArrayLike | None
) -> tuple[FloatArray, FloatArray]:
    """Keep additive node units distinct from normalized tilt derivatives."""
    primitive = forcing(economy, q)
    direct = payout_shock_derivative(economy, q)
    if opportunity_directions is not None:
        directions = jnp.asarray(opportunity_directions, dtype=jnp.float64)
        if (
            directions.shape != (SHOCK_COUNT, economy.config.nodes)
            or not jnp.isfinite(directions).all()
        ):
            msg = "additive opportunity directions require finite [2,nodes]"
            raise ValueError(msg)
        appeal = q @ economy.kernel
        denominator = economy.config.outside + appeal.sum(axis=0)
        primitive = jnp.einsum(
            "kl,il,al->iak",
            directions,
            (denominator - appeal) / denominator**2,
            economy.kernel,
        )
        direct = (appeal / denominator) @ directions.T
    return primitive, direct


def derive_model(
    economy: Economy,
    baseline: ArrayLike,
    *,
    opportunity_directions: ArrayLike | None = None,
    deadline: float | None = None,
) -> LocalModel:
    """Derive active-face responses and the equivalent discounted resolvent.

    With ``L=J+kappa I`` and ``B=L^-1 G``, the impulse response obeys
    ``V[0]=B`` and ``V[h+1]=L^-1(kappa V[h]+G rho**(h+1))``. Its discounted
    sum therefore solves ``(J+(1-discount)kappa I) S=G/(1-discount*rho)``.
    Coercivity bounds imply discounted impulse summability without a dense
    augmented-state matrix. The resulting decay rate is not an operator-norm
    bound for that matrix: its forcing block can have arbitrarily large norm.
    This AR(1), multifactor readout is a numerical extension, not a checked
    instance of a theorem requiring ``discount * norm(augmented_operator) < 1``.
    """
    if deadline is not None and perf_counter() >= deadline:
        msg = "local valuation exceeded the campaign time limit"
        raise TimeoutError(msg)
    require_float64()
    config = economy.config
    q = jnp.asarray(baseline, dtype=jnp.float64)
    if q.shape != economy.cost.shape or not jnp.isfinite(q).all() or jnp.any(q < 0):
        msg = "local baseline must be a finite feasible capacity profile"
        raise ValueError(msg)
    if not 0 < config.discount < 1 or abs(config.rho) >= 1:
        msg = "local valuation requires 0 < discount < 1 and abs(rho) < 1"
        raise ValueError(msg)
    if economy.modulus <= 0 or config.kappa < 0:
        msg = "local valuation requires positive coercivity and nonnegative anchoring"
        raise ValueError(msg)
    active = q > 0
    field = Game(economy, jnp.zeros(2)).operator(q)
    baseline_residual = float(
        jnp.linalg.norm(jnp.where(active, field, jnp.minimum(field, 0)))
        / (economy.modulus * max(1.0, float(jnp.linalg.norm(q))))
    )
    if baseline_residual > 2 * SOLVE_TARGET:
        msg = f"uncertified local baseline: relative distance bound={baseline_residual}"
        raise ArithmeticError(msg)
    if jnp.any(~active) and jnp.any(field[~active] <= SOLVE_TARGET * economy.modulus):
        msg = "local derivative requires strict complementarity on inactive capacities"
        raise ArithmeticError(msg)
    stage = _system(economy, q, active, config.kappa, deadline)
    resolvent_shift = (1 - config.discount) * config.kappa
    resolvent = _system(economy, q, active, resolvent_shift, deadline)
    primitive, direct = _primitive_derivatives(economy, q, opportunity_directions)
    responses = []
    discounted_responses = []
    residuals: list[float] = []
    for shock in range(primitive.shape[-1]):
        response, error = stage.solve(primitive[..., shock])
        responses.append(response)
        residuals.append(error)
        discounted_response, error = resolvent.solve(
            primitive[..., shock] / (1 - config.discount * config.rho)
        )
        discounted_responses.append(discounted_response)
        residuals.append(error)
    response = jnp.stack(responses, axis=-1)
    discounted = jnp.stack(discounted_responses, axis=-1)
    current = direct + jnp.column_stack(
        [
            payout_derivative(economy, q, response[..., shock])
            for shock in range(primitive.shape[-1])
        ]
    )
    total = direct / (1 - config.discount * config.rho) + jnp.column_stack(
        [
            payout_derivative(economy, q, discounted[..., shock])
            for shock in range(primitive.shape[-1])
        ]
    )
    flow = operating_cash_flow(economy, q)
    prices = config.discount * flow / (1 - config.discount)
    if not jnp.isfinite(prices).all() or jnp.any(prices <= 0):
        msg = (
            "local return normalization requires strictly positive "
            "baseline ex-dividend prices"
        )
        raise ArithmeticError(msg)
    adjustment_bound = float(config.kappa / (economy.modulus + config.kappa))
    decay_bound = float(config.discount * max(abs(config.rho), adjustment_bound))
    diagnostics: Diagnostics = {
        "discounted_impulse_summable": decay_bound < 1,
        "discounted_decay_rate_bound": decay_bound,
        "augmented_operator_norm_certified": False,
        "adjustment_contraction_bound": adjustment_bound,
        "resolvent_inverse_norm_bound": float(1 / (economy.modulus + resolvent_shift)),
        "resolvent_condition_bound": float(
            (_jacobian_bound(economy, q) + resolvent_shift)
            / (economy.modulus + resolvent_shift)
        ),
        "max_linear_residual": max(residuals),
        "baseline_relative_distance_bound": baseline_residual,
        "minimum_baseline_price": float(prices.min()),
        "active_capacities": int(active.sum()),
    }
    return LocalModel(
        economy=economy,
        baseline=q,
        active=active,
        forcing=primitive,
        response=response,
        discounted_capacity=discounted,
        payout_direct=direct,
        payout_loading=current,
        total_payoff_loading=total,
        price_loading=total - current,
        return_loading=total / prices[:, None],
        baseline_flow=flow,
        baseline_prices=prices,
        diagnostics=diagnostics,
        _stage_system=stage,
    )


@dataclass(frozen=True)
class PermanentTiltReadout:
    """Restricted permanent forcing with Lean's pre-adjustment horizon zero.

    Capacity responses have axes ``(horizon, firm, activity)`` and payout
    loadings ``(horizon, firm)``. Total payoff and return loadings include the
    infinite discounted payout tangent starting at horizon zero. This benchmark
    requires a fixed active face and constant discounting; it does not replace
    the campaign's AR(1) shock law.
    """

    capacity_response: Float[Array, "horizon firm activity"]
    payout_loading: Float[Array, "horizon firm"]
    total_payoff_loading: Float[Array, "*batch firm"]
    return_loading: Float[Array, "*batch firm"]


def permanent_tilt_readout(
    model: LocalModel, horizon: int, factor: int = 0
) -> PermanentTiltReadout:
    """Price a permanent primitive tilt using the actual joint response and payout.

    At h=0 capacities have not moved and payout contains only direct primitive
    news. Subsequent stages use the same anchored tangent solver. If S denotes
    the discounted sum of these responses, then
    ``(J+(1-d)kappa I)S=d*G/(1-d)``. This differs from a contemporaneously
    adjusted AR(1) impulse, including when persistence tends to one.
    """
    _check_deadline(model.deadline, campaign=True)
    if horizon < 0 or not 0 <= factor < model.forcing.shape[-1]:
        msg = "nonnegative horizon and existing primitive factor required"
        raise ValueError(msg)
    config = model.economy.config
    capacity = jnp.zeros((horizon + 1, *model.baseline.shape))
    shock = jnp.eye(model.forcing.shape[-1])[factor]
    for h in range(horizon):
        next_capacity, _ = model.step(capacity[h], shock)
        capacity = capacity.at[h + 1].set(next_capacity)
    payout = jnp.stack(
        [
            model.payout_direct[:, factor]
            + payout_derivative(model.economy, model.baseline, direction)
            for direction in capacity
        ]
    )
    summed = model.discounted_permanent_capacity(factor)
    total = model.payout_direct[:, factor] / (1 - config.discount) + payout_derivative(
        model.economy, model.baseline, summed
    )
    return PermanentTiltReadout(capacity, payout, total, total / model.baseline_prices)


@dataclass(frozen=True)
class LocalPath:
    """A dated tangent path with source-time, innovation-only financial news."""

    capacity: Float[Array, "date firm activity"]
    shock_state: Float[Array, "date factor"]
    cash_flow: Float[Array, "date firm"]
    payout_news: Float[Array, "period firm"]
    total_payoff_news: Float[Array, "period firm"]
    price_news: Float[Array, "period firm"]
    return_news: Float[Array, "period firm"]
    baseline_prices: Float[Array, "*batch firm"]
    diagnostics: Diagnostics


def local_path(
    model: LocalModel, innovations: ArrayLike, *, deadline: float | None = None
) -> LocalPath:
    """Advance states from source ``t`` to ``t+1`` using innovation ``t`` only.

    Domain exits are retained and counted. Financial readouts are evaluated from
    fixed conditional innovation loadings, never from future realized payouts.
    """
    _check_deadline(deadline, campaign=True)
    _check_deadline(model.deadline, campaign=True)
    shocks = jnp.asarray(innovations, dtype=jnp.float64)
    if (
        shocks.ndim != MATRIX_RANK
        or shocks.shape[1] != SHOCK_COUNT
        or not jnp.isfinite(shocks).all()
    ):
        msg = "innovations must be a finite (periods, 2) array"
        raise ValueError(msg)
    economy = model.economy
    config = economy.config
    capacity = jnp.empty((len(shocks) + 1, *model.baseline.shape))
    states = jnp.zeros((len(shocks) + 1, 2))
    flow = jnp.empty((len(shocks) + 1, model.baseline.shape[0]))
    capacity = capacity.at[0].set(model.baseline)
    flow = flow.at[0].set(model.baseline_flow)
    deviation = jnp.zeros_like(model.baseline)
    baseline_field = Game(economy, jnp.zeros(2)).operator(model.baseline)
    max_residual = float(model.diagnostics["max_linear_residual"])
    domain_violations = 0
    minimum_slack = float(jnp.min(jnp.where(model.active, jnp.inf, baseline_field)))
    for source, innovation in enumerate(shocks):
        if deadline is not None and perf_counter() >= deadline:
            msg = "local valuation exceeded the campaign time limit"
            raise TimeoutError(msg)
        states = states.at[source + 1].set(config.rho * states[source] + innovation)
        drive = jnp.einsum("iak,k->ia", model.forcing, states[source + 1])
        deviation, residual = model.step(
            deviation, states[source + 1], deadline=deadline
        )
        max_residual = max(max_residual, residual)
        capacity = capacity.at[source + 1].set(model.baseline + deviation)
        flow = flow.at[source + 1].set(
            model.baseline_flow
            + payout_derivative(economy, model.baseline, deviation)
            + model.payout_direct @ states[source + 1]
        )
        slack = (
            baseline_field + operator_jvp(economy, model.baseline, deviation) - drive
        )
        minimum_slack = min(
            minimum_slack, float(jnp.min(jnp.where(model.active, jnp.inf, slack)))
        )
        domain_violations += int(
            jnp.any(capacity[source + 1] < 0) or jnp.any(slack[~model.active] < 0)
        )
    diagnostics = dict(model.diagnostics)
    diagnostics.update(
        {
            "max_linear_residual": max_residual,
            "active_domain_violations": domain_violations,
            "minimum_capacity": float(capacity.min()),
            "has_inactive_capacities": bool(jnp.any(~model.active)),
            "minimum_inactive_slack": minimum_slack
            if jnp.isfinite(minimum_slack)
            else 0.0,
        }
    )
    return LocalPath(
        capacity=capacity,
        shock_state=states,
        cash_flow=flow,
        payout_news=shocks @ model.payout_loading.T,
        total_payoff_news=shocks @ model.total_payoff_loading.T,
        price_news=shocks @ model.price_loading.T,
        return_news=shocks @ model.return_loading.T,
        baseline_prices=model.baseline_prices.copy(),
        diagnostics=diagnostics,
    )

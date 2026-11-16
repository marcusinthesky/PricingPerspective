"""Independent derivative, timing and resolvent checks for local financial news."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hotelling.dynamics import valuation
from hotelling.dynamics.config import DynamicsConfig
from hotelling.dynamics.engine import (
    Game,
    advance,
    build_canonical_economy,
    build_economy,
    canonical_capacity,
    forcing,
    initial_state,
    operating_cash_flow,
    operator_jvp,
)
from hotelling.dynamics.valuation import (
    derive_model,
    local_path,
    payout_derivative,
    payout_shock_derivative,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hotelling.dynamics.valuation import LocalModel

FINITE_DIFFERENCE_STEP = 1e-6
FINITE_DIFFERENCE_TOLERANCE = 1e-8
LINEAR_SOLVE_TOLERANCE = 1e-10
CANONICAL_FIRMS = 6
CAMPAIGN_FIRMS = 100


@pytest.fixture(scope="module")
def model() -> LocalModel:
    """Use the exact interior all-firm Lean fixture, without estimating its root."""
    economy = build_canonical_economy(CANONICAL_FIRMS, periods=5)
    return derive_model(economy, canonical_capacity(CANONICAL_FIRMS))


def test_payout_and_forcing_are_actual_game_derivatives(model: LocalModel) -> None:
    """Differentiate nonlinear payouts and fields with an independent central step."""
    economy = model.economy
    baseline = model.baseline
    direction = np.linspace(-0.4, 0.7, baseline.size).reshape(baseline.shape)
    step = FINITE_DIFFERENCE_STEP
    numeric_capacity = (
        operating_cash_flow(economy, baseline + step * direction)
        - operating_cash_flow(economy, baseline - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        payout_derivative(economy, baseline, direction),
        numeric_capacity,
        atol=FINITE_DIFFERENCE_TOLERANCE,
        rtol=FINITE_DIFFERENCE_TOLERANCE,
    )
    _, automatic = jax.jvp(
        Game(economy, jnp.zeros(2)).flow,
        (baseline,),
        (jnp.asarray(direction),),
    )
    np.testing.assert_allclose(
        jax.jit(payout_derivative)(economy, baseline, direction),
        automatic,
        atol=1e-14,
        rtol=1e-14,
    )
    payout_loadings = payout_shock_derivative(economy, baseline)
    primitive_loadings = forcing(economy, baseline)
    for factor in range(2):
        shock = np.eye(2)[factor] * step
        positive = Game(economy, shock)
        negative = Game(economy, -shock)
        numeric_flow = (positive.flow(baseline) - negative.flow(baseline)) / (2 * step)
        numeric_forcing = -(
            positive.operator(baseline) - negative.operator(baseline)
        ) / (2 * step)
        np.testing.assert_allclose(
            payout_loadings[:, factor],
            numeric_flow,
            atol=FINITE_DIFFERENCE_TOLERANCE,
            rtol=FINITE_DIFFERENCE_TOLERANCE,
        )
        np.testing.assert_allclose(
            primitive_loadings[..., factor],
            numeric_forcing,
            atol=FINITE_DIFFERENCE_TOLERANCE,
            rtol=FINITE_DIFFERENCE_TOLERANCE,
        )


def test_additive_opportunity_derivatives_preserve_primitive_units() -> None:
    """Non-unit baseline levels distinguish additive news from proportional tilts."""
    economy = build_economy(DynamicsConfig(firms=4, activities=3, nodes=8, periods=2))
    q = initial_state(economy)
    directions = np.stack([np.linspace(-0.3, 0.4, 8), np.zeros(8)])
    model = derive_model(economy, q, opportunity_directions=directions)
    step = FINITE_DIFFERENCE_STEP
    positive = Game(
        economy, jnp.zeros(2), opportunity_levels=economy.revenue + step * directions[0]
    )
    negative = Game(
        economy, jnp.zeros(2), opportunity_levels=economy.revenue - step * directions[0]
    )
    np.testing.assert_allclose(
        model.forcing[..., 0],
        -(positive.operator(q) - negative.operator(q)) / (2 * step),
        atol=FINITE_DIFFERENCE_TOLERANCE,
    )
    np.testing.assert_allclose(
        model.payout_direct[:, 0],
        (positive.flow(q) - negative.flow(q)) / (2 * step),
        atol=FINITE_DIFFERENCE_TOLERANCE,
    )


def test_zero_news_and_lean_ex_dividend_normalization(model: LocalModel) -> None:
    """No innovations leave capacities, payouts and all financial news unchanged."""
    path = local_path(model, np.zeros((5, 2)))
    np.testing.assert_array_equal(
        path.capacity, np.broadcast_to(model.baseline, path.capacity.shape)
    )
    np.testing.assert_array_equal(
        path.cash_flow, np.broadcast_to(model.baseline_flow, path.cash_flow.shape)
    )
    for news in (
        path.payout_news,
        path.price_news,
        path.total_payoff_news,
        path.return_news,
    ):
        np.testing.assert_array_equal(news, 0)
    discount = model.economy.config.discount
    np.testing.assert_allclose(
        model.baseline_prices, discount * model.baseline_flow / (1 - discount)
    )
    assert path.diagnostics["active_domain_violations"] == 0


def test_future_innovations_cannot_change_current_news(model: LocalModel) -> None:
    """Mutating future innovations preserves all source-time readouts and states."""
    innovations = np.array([[0.01, -0.02], [0.03, 0.01], [0.02, -0.01], [0.0, 0.02]])
    changed = innovations.copy()
    cutoff = 2
    changed[cutoff:] *= -7
    first = local_path(model, innovations)
    second = local_path(model, changed)
    np.testing.assert_array_equal(
        first.capacity[: cutoff + 1], second.capacity[: cutoff + 1]
    )
    np.testing.assert_array_equal(
        first.shock_state[: cutoff + 1], second.shock_state[: cutoff + 1]
    )
    for left, right in (
        (first.payout_news, second.payout_news),
        (first.price_news, second.price_news),
        (first.total_payoff_news, second.total_payoff_news),
        (first.return_news, second.return_news),
    ):
        np.testing.assert_array_equal(left[:cutoff], right[:cutoff])
    np.testing.assert_allclose(
        first.price_news + first.payout_news,
        first.total_payoff_news,
        atol=1e-17,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        first.return_news, first.total_payoff_news / first.baseline_prices
    )


@pytest.mark.parametrize("kappa", [0.0, 1.0, 4.0])
def test_resolvent_matches_independent_discounted_impulse_sum(kappa: float) -> None:
    """Explicit forecast-only impulses independently verify the resolvent formula."""
    economy = build_canonical_economy(CANONICAL_FIRMS, kappa=kappa)
    model = derive_model(economy, canonical_capacity(CANONICAL_FIRMS))
    horizon = 600
    weights = economy.config.discount ** np.arange(horizon)
    for factor in range(2):
        innovations = np.zeros((horizon, 2))
        innovations[0, factor] = 1e-3
        path = local_path(model, innovations)
        capacity_sum = np.einsum(
            "t,tia->ia", weights, (path.capacity[1:] - model.baseline) / 1e-3
        )
        np.testing.assert_allclose(
            capacity_sum, model.discounted_capacity[..., factor], atol=2e-10, rtol=1e-8
        )
        payout_impulse = (path.cash_flow[1:] - model.baseline_flow) / 1e-3
        summed = weights @ payout_impulse
        np.testing.assert_allclose(
            summed,
            model.total_payoff_loading[:, factor],
            atol=2e-10,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            summed - payout_impulse[0],
            model.price_loading[:, factor],
            atol=2e-10,
            rtol=1e-8,
        )
    assert model.diagnostics["discounted_impulse_summable"]
    assert model.diagnostics["discounted_decay_rate_bound"] < 1
    assert model.diagnostics["augmented_operator_norm_certified"] is False
    assert model.diagnostics["max_linear_residual"] <= LINEAR_SOLVE_TOLERANCE


def test_tangent_error_shrinks_against_paired_nonlinear_path(model: LocalModel) -> None:
    """Halving finite shocks reduces the normalized operating tangent error."""
    innovations = np.array([[1.0, -0.5], [0.2, 0.8], [-0.5, 0.1]])
    errors = []
    payout_errors = []
    for size in (0.002, 0.0005):
        tangent = local_path(model, size * innovations)
        nonlinear = np.empty_like(tangent.capacity)
        nonlinear_flow = np.empty_like(tangent.cash_flow)
        nonlinear[0] = model.baseline
        nonlinear_flow[0] = model.baseline_flow
        for source in range(len(innovations)):
            nonlinear[source + 1] = advance(
                model.economy, nonlinear[source], tangent.shock_state[source + 1]
            ).capacity
            nonlinear_flow[source + 1] = operating_cash_flow(
                model.economy, nonlinear[source + 1], tangent.shock_state[source + 1]
            )
        errors.append(
            np.linalg.norm(nonlinear - tangent.capacity)
            / np.linalg.norm(tangent.capacity - model.baseline)
        )
        payout_errors.append(
            np.linalg.norm(nonlinear_flow - tangent.cash_flow)
            / np.linalg.norm(tangent.cash_flow - model.baseline_flow)
        )
    assert errors[1] < errors[0] / 2
    assert payout_errors[1] < payout_errors[0] / 2


def test_discounted_decay_does_not_certify_augmented_operator_norm(
    model: LocalModel,
) -> None:
    """Amplifying primitive forcing preserves decay but enlarges the gap block."""
    economy = replace(model.economy, shock_revenue=1000 * model.economy.shock_revenue)
    amplified = derive_model(economy, model.baseline)
    size = model.baseline.size
    identity = np.eye(size)
    jacobian = np.column_stack(
        [
            operator_jvp(
                economy, model.baseline, direction.reshape(model.baseline.shape)
            ).ravel()
            for direction in identity
        ]
    )
    config = economy.config
    static_response = np.linalg.solve(jacobian, amplified.forcing.reshape(size, -1))
    adjustment = config.kappa * np.linalg.solve(
        jacobian + config.kappa * identity, identity
    )
    # Lean augM(s,h)=(rho*s,A*h+A*chi*(1-rho)*s). A unit source input
    # with zero gap supplies a lower bound in the product's maximum norm.
    gap_block = adjustment @ static_response * (1 - config.rho)
    norm_lower_bound = np.linalg.norm(gap_block[:, 0])
    assert config.discount * norm_lower_bound > 1
    assert amplified.diagnostics["discounted_decay_rate_bound"] < 1
    assert amplified.diagnostics["discounted_impulse_summable"] is True
    assert amplified.diagnostics["augmented_operator_norm_certified"] is False


def test_tiny_impulse_solve_avoids_squared_norm_underflow(model: LocalModel) -> None:
    """A representable impulse remains solvable when its squared norm underflows."""
    size = 1e-250
    response, residual = model.step(
        np.zeros_like(model.baseline), np.array([size, 0.0])
    )
    np.testing.assert_allclose(
        response / size, model.response[..., 0], atol=1e-10, rtol=1e-10
    )
    assert residual <= LINEAR_SOLVE_TOLERANCE


def test_local_linear_solver_enforces_time_limit(model: LocalModel) -> None:
    """An expired local solve budget fails explicitly rather than hanging a run."""
    economy = replace(
        model.economy, config=replace(model.economy.config, solve_seconds=1e-30)
    )
    with pytest.raises(TimeoutError, match="time limit"):
        derive_model(economy, model.baseline)


def test_campaign_deadline_bounds_valuation(model: LocalModel) -> None:
    """A global campaign deadline applies even to zero-innovation local paths."""
    expired = perf_counter() - 1
    with pytest.raises(TimeoutError, match="campaign time limit"):
        derive_model(model.economy, model.baseline, deadline=expired)
    with pytest.raises(TimeoutError, match="campaign time limit"):
        local_path(model, np.zeros((2, 2)), deadline=expired)


def test_local_domain_exits_are_retained_without_clipping(model: LocalModel) -> None:
    """A large tangent disturbance retains negative capacities and flags the exit."""
    direction = model.response[..., 0]
    index = np.unravel_index(np.argmax(np.abs(direction)), direction.shape)
    shock = -2 * model.baseline[index] / direction[index]
    path = local_path(model, np.array([[shock, 0.0]]))
    assert path.capacity.min() < 0
    assert path.diagnostics["active_domain_violations"] == 1


def test_nonpositive_price_normalization_is_explicit(model: LocalModel) -> None:
    """A shutdown equilibrium has zero payout and cannot normalize a local return."""
    shutdown = replace(model.economy, cost=jnp.full_like(model.baseline, 100.0))
    with pytest.raises(ArithmeticError, match="strictly positive baseline"):
        derive_model(shutdown, np.zeros_like(model.baseline))


def test_higher_dimensional_hundred_firm_loadings() -> None:
    """A campaign-sized firm panel uses restricted matrix-free financial loadings."""
    config = DynamicsConfig(
        firms=CAMPAIGN_FIRMS, activities=8, nodes=32, dimension=4, periods=2
    )
    economy = build_economy(config)
    model = derive_model(economy, initial_state(economy))
    path = local_path(model, np.array([[0.001, -0.001], [0.0, 0.001]]))
    assert path.capacity.shape == (3, CAMPAIGN_FIRMS, config.activities)
    assert path.return_news.shape == (config.periods, CAMPAIGN_FIRMS)
    assert np.isfinite(path.return_news).all()
    assert path.diagnostics["max_linear_residual"] <= LINEAR_SOLVE_TOLERANCE
    assert path.baseline_prices.min() > 0


def test_valuation_matches_incoming_numpy_witness(model: LocalModel) -> None:
    """Retain incoming financial loadings independently of the new solve backend."""
    # Captured from the incoming NumPy/SciPy implementation before migration,
    # using canonical_capacity(6), kappa=1, rho=.5, and discount=.95.
    expected = np.array(
        [
            [-0.12436271880264554, -0.374248984230464],
            [-0.09520180542823255, -0.19701445640100568],
            [-0.03681136476558384, -0.06586090580720602],
            [0.03681136476558189, -0.06586090580720613],
            [0.09520180542823078, -0.1970144564010056],
            [0.12436271880264423, -0.37424898423046404],
        ]
    )
    np.testing.assert_allclose(model.return_loading, expected, atol=1e-10, rtol=1e-10)
    path = local_path(
        model, np.array([[0.01, -0.02], [0.03, 0.01], [0.02, -0.01], [0.0, 0.02]])
    )
    np.testing.assert_allclose(
        path.capacity[-1],
        [
            [0.15200817353691962, 0.8479516913116512],
            [0.2957746568708264, 0.7044248556330132],
            [0.43945637841631346, 0.5608132581659554],
            [0.5830533381733809, 0.417116898910478],
            [0.7265655361420286, 0.2733357778665808],
            [0.8699929723222565, 0.1294698950342641],
        ],
        atol=1e-10,
        rtol=1e-10,
    )


def test_local_responses_match_independent_dense_solve(model: LocalModel) -> None:
    """Dense NumPy elimination checks the matrix-free restricted linear systems."""
    baseline = np.asarray(model.baseline)
    game = Game(model.economy, jnp.zeros(2))
    step = FINITE_DIFFERENCE_STEP
    columns = []
    for coordinate in np.eye(baseline.size):
        direction = coordinate.reshape(baseline.shape)
        columns.append(
            np.asarray(
                game.operator(baseline + step * direction)
                - game.operator(baseline - step * direction)
            ).ravel()
            / (2 * step)
        )
    jacobian = np.column_stack(columns)
    config = model.economy.config
    rhs = np.asarray(model.forcing).reshape(baseline.size, -1)
    for shift, loading, divisor in (
        (config.kappa, model.response, 1.0),
        (
            (1 - config.discount) * config.kappa,
            model.discounted_capacity,
            1 - config.discount * config.rho,
        ),
    ):
        expected = np.linalg.solve(
            jacobian + shift * np.eye(baseline.size), rhs / divisor
        ).reshape(loading.shape)
        np.testing.assert_allclose(
            loading,
            expected,
            atol=FINITE_DIFFERENCE_TOLERANCE,
            rtol=FINITE_DIFFERENCE_TOLERANCE,
        )


def test_payout_jvp_uses_dynamic_economic_arrays(model: LocalModel) -> None:
    """A compiled derivative varies numeric economy leaves without static capture."""
    baseline = jnp.asarray(model.baseline)
    direction = jnp.linspace(-0.4, 0.7, baseline.size).reshape(baseline.shape)
    evaluate = jax.jit(payout_shock_derivative)
    changed = replace(model.economy, revenue=1.7 * model.economy.revenue)
    for economy in (model.economy, changed):
        actual, tangent = jax.jvp(
            lambda profile, economy=economy: evaluate(economy, profile),
            (baseline,),
            (direction,),
        )
        step = FINITE_DIFFERENCE_STEP
        expected = (
            payout_shock_derivative(economy, baseline + step * direction)
            - payout_shock_derivative(economy, baseline - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(
            actual,
            payout_shock_derivative(economy, baseline),
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            tangent,
            expected,
            atol=FINITE_DIFFERENCE_TOLERANCE,
            rtol=FINITE_DIFFERENCE_TOLERANCE,
        )
    assert not np.allclose(
        evaluate(model.economy, baseline), evaluate(changed, baseline)
    )


def test_zero_rhs_obeys_deadlines_before_computation(model: LocalModel) -> None:
    """An empty or zero-news path cannot bypass an expired numerical budget."""
    expired = perf_counter() - 1
    with pytest.raises(TimeoutError, match="campaign time limit"):
        model.step(np.zeros_like(model.baseline), np.zeros(2), deadline=expired)
    with pytest.raises(TimeoutError, match="campaign time limit"):
        local_path(model, np.empty((0, 2)), deadline=expired)
    economy = replace(
        model.economy,
        config=replace(model.economy.config, solve_seconds=1e-30),
        cost=jnp.full_like(model.baseline, 100.0),
    )
    with pytest.raises(TimeoutError, match="time limit"):
        derive_model(economy, np.zeros_like(model.baseline))


def test_local_solver_rejects_false_backend_success(
    model: LocalModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A success-shaped backend result fails the independently measured residual."""

    def false_success(
        _operator: Callable[[jax.Array], jax.Array],
        rhs: jax.Array,
        **_kwargs: object,
    ) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros_like(rhs), jnp.asarray(0)

    monkeypatch.setattr(valuation, "gmres", false_success)
    jax.clear_caches()
    try:
        with pytest.raises(ArithmeticError, match=r"residual=1\.0"):
            model.step(np.zeros_like(model.baseline), np.array([1.0, 0.0]))
    finally:
        jax.clear_caches()


def test_local_solver_respects_small_iteration_budget(model: LocalModel) -> None:
    """A single Krylov direction cannot be reported as a converged response."""
    economy = replace(
        model.economy, config=replace(model.economy.config, max_iterations=1)
    )
    with pytest.raises(ArithmeticError, match="reserved Krylov budget exhausted"):
        derive_model(economy, model.baseline)


def test_local_derivative_requires_a_strictly_complementary_face() -> None:
    """Inactive capacities stay zero; a weakly active baseline has no local claim."""
    economy = build_canonical_economy(CANONICAL_FIRMS)
    baseline = canonical_capacity(CANONICAL_FIRMS).at[0, 0].set(0)
    initial_field = Game(economy, jnp.zeros(2)).operator(baseline)
    slack = jnp.zeros_like(baseline).at[0, 0].set(0.1)
    cost = economy.cost - initial_field
    interior_face = replace(economy, cost=cost + slack)
    model = derive_model(interior_face, baseline)
    np.testing.assert_array_equal(model.response[0, 0], 0)
    np.testing.assert_array_equal(model.discounted_capacity[0, 0], 0)
    with pytest.raises(ArithmeticError, match="strict complementarity"):
        derive_model(replace(economy, cost=cost), baseline)

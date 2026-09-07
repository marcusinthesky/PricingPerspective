"""One estimation window of the look-ahead-safe robust backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from pipeline.stages.papers.paper3.empirical import (
    _bootstrap_sharpe_ci,
)
from pipeline.stages.papers.paper5._energy_robust.contracts import (
    STRATEGIES,
    RobustPortfolioComputationError,
)
from pipeline.stages.papers.paper5._energy_robust.covariance import (
    _cov_and_weights,
    _portfolio_diagnostics,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import (
        Float,
        RobustCovarianceInputs,
    )


@dataclass(frozen=True)
class _WindowSimulation:
    """Portfolio paths and diagnostics produced for one estimation window."""

    series: dict[str, list[float]]
    turnover: dict[str, list[float]]
    condition: dict[str, list[float]]
    effective_n: dict[str, list[float]]
    hhi: dict[str, list[float]]
    certified_caps: dict[str, list[float]]
    realized_variance: dict[str, list[float]]
    rebalance_interval: int
    radius: float


def _simulate_backtest_window(
    returns: Float, window: int, inputs: RobustCovarianceInputs
) -> _WindowSimulation:
    """Simulate every strategy for one look-ahead-safe rolling window."""
    radius = inputs.r_by_window.get(window, 0.0)
    rebalance_interval = max(1, window // 4)
    series: dict[str, list[float]] = {method: [] for method in STRATEGIES}
    turnover: dict[str, list[float]] = {method: [] for method in STRATEGIES}
    condition: dict[str, list[float]] = {method: [] for method in STRATEGIES}
    effective_n: dict[str, list[float]] = {method: [] for method in STRATEGIES}
    hhi: dict[str, list[float]] = {method: [] for method in STRATEGIES}
    realized_variance: dict[str, list[float]] = {method: [] for method in STRATEGIES}
    certified_caps: dict[str, list[float]] = {
        "robust_corner": [],
        "robust_corner_cert": [],
        "cert_stale_vol": [],
        "robust_ridge": [],
    }
    held_weights: dict[str, Float | None] = dict.fromkeys(STRATEGIES)
    for time_index in range(window, returns.shape[0]):
        next_return = returns[time_index]
        if (time_index - window) % rebalance_interval == 0:
            built = _cov_and_weights(
                returns[time_index - window : time_index],
                radius,
                inputs,
            )
            weights = cast("dict[str, Float]", built["weights"])
            covariances = cast("dict[str, Float]", built["covs"])
            for method in STRATEGIES:
                weights_for_method = weights[method]
                diagnostics = _portfolio_diagnostics(
                    covariances[method], weights_for_method
                )
                previous = held_weights[method]
                turnover[method].append(
                    0.0
                    if previous is None
                    else float(np.sum(np.abs(weights_for_method - previous)))
                )
                condition[method].append(diagnostics["cond"])
                effective_n[method].append(diagnostics["eff_n"])
                hhi[method].append(diagnostics["hhi"])
                held_weights[method] = weights_for_method
            for method, cap_key in (
                ("robust_corner", "cap_corner"),
                ("robust_corner_cert", "cap_corner_cert"),
                ("cert_stale_vol", "cap_stale"),
                ("robust_ridge", "cap_ridge"),
            ):
                certified_caps[method].append(
                    cast("dict[str, float]", built[cap_key])["certified_cap"]
                )
        for method in STRATEGIES:
            weights_for_method = held_weights[method]
            if weights_for_method is None:
                message = f"strategy {method} has no held weights after rebalance"
                raise RobustPortfolioComputationError(message)
            portfolio_return = float(next_return @ weights_for_method)
            series[method].append(portfolio_return)
            realized_variance[method].append(portfolio_return**2)
            grown_weights = weights_for_method * (1.0 + next_return)
            held_weights[method] = grown_weights / grown_weights.sum()
    return _WindowSimulation(
        series,
        turnover,
        condition,
        effective_n,
        hhi,
        certified_caps,
        realized_variance,
        rebalance_interval,
        radius,
    )


def _summarize_backtest_window(
    window: int,
    simulation: _WindowSimulation,
    cost_bps: tuple[float, ...],
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    """Summarize simulated strategy paths, costs, and uncertainty intervals."""
    rows: list[dict[str, object]] = []
    annualization = np.sqrt(252.0)
    for method in STRATEGIES:
        pnl = np.asarray(simulation.series[method])
        turnover = np.asarray(simulation.turnover[method])
        cost_basis = np.zeros_like(pnl)
        rebalance_days = range(0, pnl.size, simulation.rebalance_interval)
        for index, day in enumerate(rebalance_days):
            if index < turnover.size:
                cost_basis[day] = turnover[index]
        realized_variance = float(np.mean(simulation.realized_variance[method]))
        row: dict[str, object] = {
            "window": window,
            "method": method,
            "realized_vol": float(np.std(pnl, ddof=1)) * annualization,
            "turnover": float(np.mean(turnover)) if turnover.size else 0.0,
            "cond": float(np.median(simulation.condition[method])),
            "eff_n": float(np.median(simulation.effective_n[method])),
            "hhi": float(np.median(simulation.hhi[method])),
            "n_periods": int(pnl.size),
            "realized_variance": realized_variance,
            "r_t": simulation.radius,
        }
        if method in simulation.certified_caps:
            caps = simulation.certified_caps[method]
            median_cap = float(np.median(caps)) if caps else float("nan")
            row["certified_cap_median"] = median_cap
            row["cap_covers_realized_variance"] = median_cap >= realized_variance
        for cost in cost_bps:
            net_returns = pnl - (cost / 1e4) * cost_basis
            standard_deviation = float(np.std(net_returns, ddof=1))
            sharpe = (
                float(np.mean(net_returns)) / standard_deviation * annualization
                if standard_deviation > 0
                else 0.0
            )
            lower, upper = _bootstrap_sharpe_ci(net_returns, n_boot, seed)
            suffix = f"{int(cost)}bps"
            row[f"sharpe_net_{suffix}"] = sharpe
            row[f"sharpe_ci_lo_{suffix}"] = lower * annualization
            row[f"sharpe_ci_hi_{suffix}"] = upper * annualization
        rows.append(row)
    return rows

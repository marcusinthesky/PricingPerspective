"""Step 3b: per-cell summarisation and the look-ahead-safe backtest sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper3._empirical.backtest_cells import (
    _BACKTEST_METHODS,
    SUB_PERIODS,
    _BacktestCell,
    _first_pit_evaluation_index,
    _simulate_backtest_cell,
)
from pipeline.stages.papers.paper3._empirical.contracts import (
    BacktestConfig,
    Paper3EmpiricalError,
)
from pipeline.stages.papers.paper3._empirical.sharpe import _bootstrap_sharpe_ci

if TYPE_CHECKING:
    from pipeline.stages.papers.paper3._empirical.backtest_cells import (
        _BacktestSimulation,
    )
    from pipeline.stages.papers.paper3._empirical.contracts import (
        BacktestInputs,
    )


def _summarize_backtest_cell(
    simulation: _BacktestSimulation,
    cell: _BacktestCell,
    config: BacktestConfig,
) -> list[dict[str, object]]:
    """Summarize full-sample and calendar-subperiod performance for one cell."""
    rows: list[dict[str, object]] = []
    annualization = np.sqrt(252.0)
    median_kappa = float(np.median(simulation.kappas)) if simulation.kappas else 0.0
    shrinkage = np.asarray(simulation.shrinkage)
    lambda_summary = {
        "lambda_sample_mean": (
            float(np.mean(shrinkage)) if shrinkage.size else float("nan")
        ),
        "lambda_sample_median": (
            float(np.median(shrinkage)) if shrinkage.size else float("nan")
        ),
        "lambda_sample_min": (
            float(np.min(shrinkage)) if shrinkage.size else float("nan")
        ),
        "lambda_sample_max": (
            float(np.max(shrinkage)) if shrinkage.size else float("nan")
        ),
    }
    constraint = "long_only" if cell.long_only else "unconstrained"
    pnl_dates = (
        None
        if cell.inputs.dates is None
        else cell.inputs.dates[cell.evaluation_start :]
    )
    evaluation_bounds = {
        "evaluation_start": (
            None if pnl_dates is None or pnl_dates.empty else str(pnl_dates[0].date())
        ),
        "evaluation_end": (
            None if pnl_dates is None or pnl_dates.empty else str(pnl_dates[-1].date())
        ),
    }
    for method in _BACKTEST_METHODS:
        pnl = np.asarray(simulation.series[method])
        turnover = np.asarray(simulation.turnover[method])
        costs = np.zeros_like(pnl)
        for index, day in enumerate(range(0, pnl.size, simulation.rebalance_interval)):
            if index < turnover.size:
                costs[day] = turnover[index]
        mean_turnover = float(np.mean(turnover)) if turnover.size else 0.0
        row: dict[str, object] = {
            "window": cell.window,
            "method": method,
            "constraint": constraint,
            "source": cell.source,
            "period": "full",
            "realized_vol": float(np.std(pnl, ddof=1)) * annualization,
            "turnover": mean_turnover,
            "cond": float(np.median(simulation.condition[method])),
            "eff_n": float(np.median(simulation.effective_n[method])),
            "hhi": float(np.median(simulation.hhi[method])),
            "n_periods": int(pnl.size),
            "kappa": median_kappa,
            **evaluation_bounds,
        }
        if method == "sigma_shrink":
            row.update(lambda_summary)
        for cost in config.cost_bps:
            net = pnl - (cost / 1e4) * costs
            standard_deviation = float(np.std(net, ddof=1))
            sharpe = (
                float(np.mean(net)) / standard_deviation * annualization
                if standard_deviation > 0
                else 0.0
            )
            lower, upper = _bootstrap_sharpe_ci(net, config.n_boot, config.seed)
            suffix = f"{int(cost)}bps"
            row[f"sharpe_net_{suffix}"] = sharpe
            row[f"sharpe_ci_lo_{suffix}"] = lower * annualization
            row[f"sharpe_ci_hi_{suffix}"] = upper * annualization
        rows.append(row)
        if pnl_dates is not None:
            pnl_date_values = np.asarray(pnl_dates, dtype="datetime64[ns]")
            for label, start, end in SUB_PERIODS:
                mask = (pnl_date_values >= np.datetime64(start)) & (
                    pnl_date_values <= np.datetime64(end)
                )
                if not mask.any():
                    continue
                net = pnl[mask] - (10.0 / 1e4) * costs[mask]
                standard_deviation = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
                subperiod_row = {
                    "window": cell.window,
                    "method": method,
                    "constraint": constraint,
                    "source": cell.source,
                    "period": label,
                    "realized_vol": (
                        float(np.std(net, ddof=1)) * annualization
                        if net.size > 1
                        else 0.0
                    ),
                    "turnover": mean_turnover,
                    "cond": float(np.median(simulation.condition[method])),
                    "eff_n": float(np.median(simulation.effective_n[method])),
                    "hhi": float(np.median(simulation.hhi[method])),
                    "n_periods": int(net.size),
                    "kappa": median_kappa,
                    "evaluation_start": str(pnl_dates[mask][0].date()),
                    "evaluation_end": str(pnl_dates[mask][-1].date()),
                    "sharpe_net_10bps": (
                        float(np.mean(net)) / standard_deviation * annualization
                        if standard_deviation > 0
                        else 0.0
                    ),
                }
                if method == "sigma_shrink":
                    subperiod_row.update(lambda_summary)
                rows.append(subperiod_row)
    return rows


def backtest(
    inputs: BacktestInputs,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Look-ahead-safe expanding-origin rolling OOS min-variance backtest.

    For each window ``T`` and method, at each rebalance date the covariance is
    estimated on the strictly-prior ``T`` observations and the resulting
    weights are held for the *next* period's realized return — no look-ahead.
    In particular κ (for ``Σ_dist`` / ``Σ_shrink``) is recalibrated on each
    training window, so no future (test-period) returns enter the estimator.
    When PIT matrices are supplied, observations on or before the first
    vintage are removed from both the PIT and fixed-matrix arms, so their
    reported sample dates match.  Strict prior-vintage lookup is required
    because each vintage includes articles dated on its cutoff day; same-day
    text therefore cannot price the same day's realized return.  A missing,
    malformed, or nonfinite declared vintage raises
    :class:`Paper3EmpiricalError`; the fixed full-corpus matrix is never used
    as a PIT fallback.
    Reports net-of-transaction-cost Sharpe (turnover-scaled, at each cost),
    realized volatility, turnover, condition number, effective number of
    assets, HHI, for both long-only and unconstrained; block-bootstrap Sharpe
    CIs use circular block bootstrap with Politis–White length under the
    circular/moving-block constant ``D_MB=(4/3)g²(0)``
    (:func:`jcor.inference.block_length.optimal_block_length_numpy` with
    ``scheme="circular"``).

    Args:
        inputs: Return panel, squared distances, and optional point-in-time data.
        config: Window, transaction-cost, and bootstrap controls.

    Returns:
        Long-format ``DataFrame`` with one row per (window, method,
        constraint, source) and the metric columns, including the common
        ``evaluation_start`` and ``evaluation_end`` dates.

    """
    config = BacktestConfig() if config is None else config
    rows: list[dict[str, object]] = []
    has_pit_matrices = inputs.pit_matrices is not None
    has_pit_dates = inputs.pit_dates is not None
    has_partial_pit_inputs = has_pit_matrices != has_pit_dates
    if has_partial_pit_inputs:
        message = "pit_matrices and pit_dates must be supplied together"
        raise Paper3EmpiricalError(message)
    use_pit = has_pit_matrices and has_pit_dates
    if use_pit and inputs.dates is None:
        message = "return dates are required for a PIT comparison"
        raise Paper3EmpiricalError(message)
    first_pit_index = _first_pit_evaluation_index(inputs) if use_pit else None
    for window in config.windows:
        evaluation_start = (
            window if first_pit_index is None else max(window, first_pit_index)
        )
        for long_only in (True, False):
            for source, baseline in (
                [("baseline", inputs.squared_distances), ("pit", None)]
                if use_pit
                else [("baseline", inputs.squared_distances)]
            ):
                cell = _BacktestCell(
                    inputs,
                    window,
                    evaluation_start,
                    long_only,
                    source,
                    baseline,
                )
                simulation = _simulate_backtest_cell(cell)
                rows.extend(
                    _summarize_backtest_cell(
                        simulation,
                        cell,
                        config,
                    )
                )
    return pd.DataFrame(rows)

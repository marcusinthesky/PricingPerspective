"""The window sweep that drives the robust backtest."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from pipeline.stages.papers.paper5._energy_robust.backtest_window import (
    _simulate_backtest_window,
    _summarize_backtest_window,
)
from pipeline.stages.papers.paper5._energy_robust.contracts import (
    RobustBacktestConfig,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import (
        RobustBacktestInputs,
    )

logger = logging.getLogger(__name__)


def backtest(
    inputs: RobustBacktestInputs,
    config: RobustBacktestConfig | None = None,
) -> pd.DataFrame:
    """Look-ahead-safe expanding-origin rolling OOS backtest (long-only only).

    Mirrors :func:`pipeline.paper3_empirical.backtest`'s rebalance/drift/cost
    logic exactly (see that docstring), extended to the 10
    :data:`STRATEGIES` and reporting the certified-cap diagnostics
    (:func:`certified_caps`) for the four robust strategies on each rebalance.

    Args:
        inputs: Evaluation return panel and calibration-fixed covariance inputs.
            The return panel has shape ``(T, n)`` (evaluation epoch only -- callers
            pass the 2022 panel; training windows draw only from within it,
            preserving no-look-ahead. NOTE: because ``windows`` includes T=250
            and the 2022 panel has ~250 trading days, T=250 effectively uses
            most of the calibration-adjacent history; see RUN_MANIFEST for
            the exact per-window sample count actually realized OOS).
        config: Window, transaction-cost, bootstrap, and seed controls.

    Returns:
        Long-format DataFrame, one row per (window, method) plus certified-cap
        diagnostics for the robust methods.

    """
    config = RobustBacktestConfig() if config is None else config
    n_observations = inputs.returns.shape[0]
    rows: list[dict[str, object]] = []
    for window in config.windows:
        if window >= n_observations:
            logger.warning(
                "backtest: window %d >= evaluation length %d; skipping",
                window,
                n_observations,
            )
            continue
        simulation = _simulate_backtest_window(
            inputs.returns, window, inputs.covariance
        )
        rows.extend(
            _summarize_backtest_window(
                window, simulation, config.cost_bps, config.n_boot, config.seed
            )
        )
    return pd.DataFrame(rows)

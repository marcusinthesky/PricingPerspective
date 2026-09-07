"""Step 3a: PIT matrices, the backtest cell grid, and one-cell simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper3._empirical.contracts import (
    Paper3EmpiricalError,
)
from pipeline.stages.papers.paper3._empirical.optimize import (
    _cov_estimators,
    _mv_weights,
    _portfolio_diagnostics,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.papers.paper3._empirical.contracts import (
        BacktestInputs,
        Float,
    )


# Sub-period boundaries (inclusive) used for the sub-period Sharpe slicing;
# mirrors params.yaml `paper3.sub_periods` (kept in sync manually — this is a
# small, human-legible schedule and does not warrant a params round-trip).
SUB_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
)


def _load_pit_matrices(
    output_dir: Path,
) -> tuple[dict[str, Float] | None, pd.DatetimeIndex | None]:
    """Load point-in-time squared-W2 matrices from the Paper-3 PIT stage.

    Looks for ``data/papers/paper3/pit_w2/matrices.npz`` (sibling of
    ``output_dir``, i.e. ``output_dir.parent / "pit_w2"``). Returns
    ``(None, None)`` when the stage has not been run — the PIT consumption
    path is opt-in and never blocks the baseline backtest.

    Args:
        output_dir: Paper-3 empirical output directory.

    Returns:
        ``(vintage -> D2 matrix, vintage_dates)`` or ``(None, None)``.

    """
    pit_dir = output_dir.parent / "pit_w2"
    npz_path = pit_dir / "matrices.npz"
    if not npz_path.exists():
        return None, None
    npz = np.load(npz_path, allow_pickle=False)
    vintages = sorted(npz.files)
    # The PIT stage already performs the governed rooted-W2 -> W2² conversion.
    # Keep the matrix unchanged at this consumption layer.
    matrices = {v: np.asarray(npz[v], dtype=np.float64) for v in vintages}
    return matrices, pd.DatetimeIndex([pd.Timestamp(v) for v in vintages])


def _nearest_vintage(
    as_of: pd.Timestamp, pit_dates: pd.DatetimeIndex
) -> pd.Timestamp | None:
    """Most recent PIT vintage strictly before ``as_of`` (no same-day text)."""
    eligible = pit_dates[pit_dates < as_of]
    if len(eligible) == 0:
        return None
    return eligible.max()


_BACKTEST_METHODS = (
    "sample",
    "ledoit_wolf",
    "sigma_dist",
    "sigma_shrink",
    "one_over_n",
)


@dataclass(frozen=True)
class _BacktestSimulation:
    """Portfolio paths and diagnostics for one window/constraint/source cell."""

    series: dict[str, list[float]]
    turnover: dict[str, list[float]]
    condition: dict[str, list[float]]
    effective_n: dict[str, list[float]]
    hhi: dict[str, list[float]]
    kappas: list[float]
    shrinkage: list[float]
    rebalance_interval: int


@dataclass(frozen=True)
class _BacktestCell:
    """One window, constraint, and distance-source simulation context."""

    inputs: BacktestInputs
    window: int
    evaluation_start: int
    long_only: bool
    source: str
    baseline: Float | None


def _validate_pit_vintages(
    pit_dates: pd.DatetimeIndex,
    pit_matrices: dict[str, Float],
    expected_shape: tuple[int, ...],
) -> None:
    """Fail closed when a declared PIT vintage is absent or unusable."""
    vintage_keys = [str(pd.Timestamp(vintage).date()) for vintage in pit_dates]
    if len(vintage_keys) != len(set(vintage_keys)):
        message = "PIT vintage dates must be unique"
        raise Paper3EmpiricalError(message)
    for key in vintage_keys:
        matrix = pit_matrices.get(key)
        if matrix is None:
            message = f"PIT vintage {key} has no distance matrix"
            raise Paper3EmpiricalError(message)
        if matrix.shape != expected_shape:
            message = (
                f"PIT vintage {key} has shape {matrix.shape}; expected {expected_shape}"
            )
            raise Paper3EmpiricalError(message)
        if not np.isfinite(matrix).all():
            message = f"PIT vintage {key} contains nonfinite distances"
            raise Paper3EmpiricalError(message)


def _first_pit_evaluation_index(inputs: BacktestInputs) -> int:
    """Validate PIT inputs and return the first usable return-row index.

    A PIT comparison is only meaningful when both sources are evaluated over
    the same dates.  The first vintage includes articles timestamped on its
    cutoff date, so observations on or before that vintage are excluded from
    *both* the PIT and fixed-matrix paths.  Invalid vintages fail closed instead
    of borrowing the full-corpus matrix.
    """
    if inputs.dates is None or inputs.pit_dates is None or inputs.pit_matrices is None:
        message = "PIT comparison requires dates, pit_dates, and pit_matrices"
        raise Paper3EmpiricalError(message)
    if len(inputs.dates) != inputs.returns.shape[0]:
        message = "return dates must have one entry per return observation"
        raise Paper3EmpiricalError(message)
    if not inputs.dates.is_monotonic_increasing:
        message = "return dates must be monotonic increasing"
        raise Paper3EmpiricalError(message)
    if inputs.pit_dates.empty:
        message = "PIT comparison requires at least one vintage"
        raise Paper3EmpiricalError(message)

    _validate_pit_vintages(
        inputs.pit_dates,
        inputs.pit_matrices,
        inputs.squared_distances.shape,
    )

    first_vintage = inputs.pit_dates.min()
    eligible = np.flatnonzero(inputs.dates > first_vintage)
    if eligible.size == 0:
        message = "no return observation is strictly after the first PIT vintage"
        raise Paper3EmpiricalError(message)
    return int(eligible[0])


def _rebalance_distance_matrix(
    cell: _BacktestCell,
    time_index: int,
) -> Float:
    """Select the latest declared PIT matrix, failing closed if it is invalid."""
    if cell.source == "baseline":
        if cell.baseline is None:
            message = "baseline distance matrix is unavailable"
            raise Paper3EmpiricalError(message)
        return cell.baseline
    inputs = cell.inputs
    if inputs.dates is None or inputs.pit_dates is None or inputs.pit_matrices is None:
        message = "PIT source selected without complete PIT data"
        raise Paper3EmpiricalError(message)
    vintage = _nearest_vintage(inputs.dates[time_index], inputs.pit_dates)
    if vintage is None:
        message = f"no PIT vintage is available at {inputs.dates[time_index].date()}"
        raise Paper3EmpiricalError(message)
    key = str(vintage.date())
    pit_matrix = inputs.pit_matrices.get(key)
    if pit_matrix is None:
        message = f"PIT vintage {key} has no distance matrix"
        raise Paper3EmpiricalError(message)
    if not np.isfinite(pit_matrix).all():
        message = f"PIT vintage {key} contains nonfinite distances"
        raise Paper3EmpiricalError(message)
    return pit_matrix


def _simulate_backtest_cell(cell: _BacktestCell) -> _BacktestSimulation:
    """Simulate one rolling backtest cell with drifting held weights."""
    inputs = cell.inputs
    rebalance_interval = max(1, cell.window // 4)
    series = {method: [] for method in _BACKTEST_METHODS}
    turnover = {method: [] for method in _BACKTEST_METHODS}
    condition = {method: [] for method in _BACKTEST_METHODS}
    effective_n = {method: [] for method in _BACKTEST_METHODS}
    hhi = {method: [] for method in _BACKTEST_METHODS}
    held_weights: dict[str, Float | None] = dict.fromkeys(_BACKTEST_METHODS)
    kappas: list[float] = []
    shrinkage: list[float] = []
    n_assets = inputs.returns.shape[1]
    for time_index in range(cell.evaluation_start, inputs.returns.shape[0]):
        next_return = inputs.returns[time_index]
        if (time_index - cell.evaluation_start) % rebalance_interval == 0:
            train = inputs.returns[time_index - cell.window : time_index]
            distance_matrix = _rebalance_distance_matrix(cell, time_index)
            estimators = _cov_estimators(train, distance_matrix)
            kappas.append(float(estimators["_kappa"]))
            shrinkage.append(float(estimators["_lambda"]))
            for method in _BACKTEST_METHODS:
                covariance = cast(
                    "Float",
                    estimators["sample" if method == "one_over_n" else method],
                )
                weights = (
                    np.full(n_assets, 1.0 / n_assets)
                    if method == "one_over_n"
                    else _mv_weights(covariance, long_only=cell.long_only)
                )
                diagnostics = _portfolio_diagnostics(
                    covariance, weights, rank_aware=method in {"sample", "one_over_n"}
                )
                previous = held_weights[method]
                turnover[method].append(
                    0.0
                    if previous is None
                    else float(np.sum(np.abs(weights - previous)))
                )
                condition[method].append(diagnostics["cond"])
                effective_n[method].append(diagnostics["eff_n"])
                hhi[method].append(diagnostics["hhi"])
                held_weights[method] = weights
        for method in _BACKTEST_METHODS:
            weights = held_weights[method]
            if weights is None:
                message = f"strategy {method} has no weights after rebalance"
                raise Paper3EmpiricalError(message)
            portfolio_return = float(next_return @ weights)
            series[method].append(portfolio_return)
            grown_weights = weights * (1.0 + next_return)
            held_weights[method] = grown_weights / grown_weights.sum()
    return _BacktestSimulation(
        series,
        turnover,
        condition,
        effective_n,
        hhi,
        kappas,
        shrinkage,
        rebalance_interval,
    )

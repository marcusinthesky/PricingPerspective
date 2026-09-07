"""Paper 5 sequential horizon-and-universe bridge for frozen W-flat operators.

The bridge starts from the historical 52-firm long-window result, shortens that
same universe to the governed current endpoint, and then expands the universe
to the current 100-firm specification at that endpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5._gate.bootstrap import stationary_bootstrap_indices
from pipeline.stages.papers.paper5._gate.driver import _load_gate_operator
from pipeline.stages.papers.paper5.post_corpus import _bootstrap_rho
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_LOWER_QUANTILE = 0.025
_UPPER_QUANTILE = 0.975
_PANEL_DIMENSIONS = 2
_REFERENCE_SHORT_CELL = "reference_52_short"
_REFERENCE_LONG_CELL = "reference_52_long"
_CURRENT_SHORT_CELL = "current_100_short"
_W_VARIANT = "w_flat"


@dataclass(frozen=True)
class Paper5UniverseHorizonBridgePaths:
    """Input and output boundaries for the universe-by-horizon bridge."""

    returns_dir: Path
    current_oos_returns_dir: Path
    current_barycentre_dir: Path
    current_results_json: Path
    reference_oos_returns_dir: Path
    reference_barycentre_dir: Path
    reference_results_json: Path
    output_dir: Path


@dataclass(frozen=True)
class Paper5UniverseHorizonBridgeConfig:
    """Pinned windows, universe sizes, operator identity, and bootstrap controls."""

    start: str = "2023-01-01"
    short_end: str = "2026-04-06"
    long_end: str = "2026-07-15"
    expected_reference_universe: int = 52
    expected_current_universe: int = 100
    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    shared_return_tolerance: float = 0.0
    result_tolerance: float = 1e-12
    reference_revision: str = "7381033e76244c68835bc232a593f9086384f339"
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    barycentre_arm_id: str = "wasserstein_w2_loo"
    geometry_id: str = "wasserstein_w2"

    def validate(self) -> None:
        """Reject an invalid bridge design before reading artifacts."""
        start = pd.Timestamp(self.start)
        short_end = pd.Timestamp(self.short_end)
        long_end = pd.Timestamp(self.long_end)
        if not start < short_end < long_end:
            message = "bridge dates must satisfy start < short_end < long_end"
            raise ValueError(message)
        if self.expected_reference_universe < 1:
            message = "reference universe size must be positive"
            raise ValueError(message)
        if self.expected_current_universe <= self.expected_reference_universe:
            message = "current universe must be larger than the reference universe"
            raise ValueError(message)
        if self.n_bootstrap < 1 or self.block_length < 1:
            message = "bootstrap draws and block length must be positive"
            raise ValueError(message)
        if (
            not np.isfinite(self.shared_return_tolerance)
            or self.shared_return_tolerance < 0.0
        ):
            message = "return comparison tolerance must be finite and nonnegative"
            raise ValueError(message)
        if not np.isfinite(self.result_tolerance) or self.result_tolerance < 0.0:
            message = "estimate verification tolerance must be finite and nonnegative"
            raise ValueError(message)
        if re.fullmatch(r"[0-9a-f]{40}", self.reference_revision) is None:
            message = "reference revision must be a lowercase 40-hex Git commit"
            raise ValueError(message)
        identities = (
            self.provider_id,
            self.representation_id,
            self.barycentre_arm_id,
            self.geometry_id,
        )
        if not all(identity.strip() for identity in identities):
            message = "operator identity fields must be nonempty"
            raise ValueError(message)


@dataclass(frozen=True)
class _PointFit:
    """Minimal point-QMLE record consumed by the bridge."""

    rho_hat: float
    loglik: float
    converged: bool


@dataclass(frozen=True)
class _GovernedRecord:
    """Validated governed result and its point-estimate reconciliation."""

    rho_hat: float
    rho_ci_lower: float
    rho_ci_upper: float
    absolute_difference: float


def _plain(value: object) -> object:
    """Convert NumPy containers and scalars to deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _mapping(value: object, context: str) -> dict[str, object]:
    """Narrow one decoded JSON value to an object with string keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{context} must be a JSON object with string keys"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _number(mapping: dict[str, object], key: str, context: str) -> float:
    """Read one finite numeric field while rejecting booleans."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"{context} requires numeric field {key!r}"
        raise TypeError(message)
    result = float(value)
    if not np.isfinite(result):
        message = f"{context} field {key!r} must be finite"
        raise ValueError(message)
    return result


def _read_json_object(path: Path, context: str) -> dict[str, object]:
    """Load one required governed JSON object."""
    if not path.is_file():
        message = f"missing {context}: {path}"
        raise FileNotFoundError(message)
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, context)


def _rho_to_lambda(rho: float) -> float:
    """Map stable spatial feedback to its implied adjustment intensity."""
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        message = (
            f"rho must be finite and strictly between minus one and one, got {rho}"
        )
        raise ValueError(message)
    return float(rho / (1.0 - rho))


def _rho_draws_to_lambda(rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the monotone intensity transform to finite bootstrap draws."""
    if not np.all(np.isfinite(rho)) or np.any(rho <= -1.0) or np.any(rho >= 1.0):
        message = "bootstrap rho draws must be finite and strictly inside (-1, 1)"
        raise ValueError(message)
    return rho / (1.0 - rho)


def _window_metadata(
    dates: NDArray[np.datetime64],
    returns: NDArray[np.float64],
) -> dict[str, object]:
    """Summarize a nonempty aligned panel window."""
    if len(dates) == 0 or len(returns) == 0:
        message = "bridge return panels must contain at least one common date"
        raise ValueError(message)
    if len(dates) != len(returns):
        message = "bridge return dates and rows have different lengths"
        raise ValueError(message)
    if returns.ndim != _PANEL_DIMENSIONS or not np.all(np.isfinite(returns)):
        message = "bridge returns must be a finite two-dimensional panel"
        raise ValueError(message)
    return {
        "window_start": str(pd.Timestamp(dates.min()).date()),
        "window_end": str(pd.Timestamp(dates.max()).date()),
        "trading_days": len(returns),
    }


def _verify_common_return_panel(
    current_dates: NDArray[np.datetime64],
    current_returns: NDArray[np.float64],
    reference_dates: NDArray[np.datetime64],
    reference_returns: NDArray[np.float64],
    *,
    tolerance: float,
) -> dict[str, object]:
    """Prove that the common 52-firm short-window observations agree."""
    current_window = _window_metadata(current_dates, current_returns)
    reference_window = _window_metadata(reference_dates, reference_returns)
    if current_dates.shape != reference_dates.shape or not np.array_equal(
        current_dates, reference_dates
    ):
        message = "current and reference 52-firm short panels have different dates"
        raise ValueError(message)
    if current_returns.shape != reference_returns.shape:
        message = "current and reference 52-firm short panels have different shapes"
        raise ValueError(message)
    absolute_difference = np.abs(current_returns - reference_returns)
    max_abs_difference = float(np.max(absolute_difference))
    if max_abs_difference > tolerance:
        message = (
            "current and reference 52-firm short returns differ by "
            f"{max_abs_difference:.3e}, above tolerance {tolerance:.3e}"
        )
        raise ValueError(message)
    if current_window != reference_window:
        message = "common 52-firm short panels have inconsistent window metadata"
        raise ValueError(message)
    return {
        "status": "identical_within_tolerance",
        "tolerance": tolerance,
        "n_dates": current_window["trading_days"],
        "window_start": current_window["window_start"],
        "window_end": current_window["window_end"],
        "max_abs_difference": max_abs_difference,
    }


def _validate_universes(
    reference_tickers: list[str],
    reference_weights: NDArray[np.float64],
    current_tickers: list[str],
    current_weights: NDArray[np.float64],
    config: Paper5UniverseHorizonBridgeConfig,
) -> None:
    """Validate exact counts, uniqueness, subset nesting, and matrix dimensions."""
    if len(reference_tickers) != config.expected_reference_universe:
        message = (
            f"reference operator has {len(reference_tickers)} firms; expected "
            f"{config.expected_reference_universe}"
        )
        raise ValueError(message)
    if len(current_tickers) != config.expected_current_universe:
        message = (
            f"current operator has {len(current_tickers)} firms; expected "
            f"{config.expected_current_universe}"
        )
        raise ValueError(message)
    if len(set(reference_tickers)) != len(reference_tickers):
        message = "reference operator ticker order contains duplicates"
        raise ValueError(message)
    if len(set(current_tickers)) != len(current_tickers):
        message = "current operator ticker order contains duplicates"
        raise ValueError(message)
    missing = sorted(set(reference_tickers) - set(current_tickers))
    if missing:
        message = f"reference universe is not a subset of current universe: {missing}"
        raise ValueError(message)
    expected_reference_shape = (
        config.expected_reference_universe,
        config.expected_reference_universe,
    )
    expected_current_shape = (
        config.expected_current_universe,
        config.expected_current_universe,
    )
    if reference_weights.shape != expected_reference_shape:
        message = f"reference W-flat matrix has shape {reference_weights.shape}"
        raise ValueError(message)
    if current_weights.shape != expected_current_shape:
        message = f"current W-flat matrix has shape {current_weights.shape}"
        raise ValueError(message)


def _fit_point_qmle(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    cell_id: str,
) -> _PointFit:
    """Fit and validate one point-only universe-by-horizon QMLE cell."""
    fit = est.sar_qmle(returns, weights, compute_se=False)
    rho_hat = _number(cast("dict[str, object]", fit), "rho_hat", cell_id)
    loglik = _number(cast("dict[str, object]", fit), "loglik", cell_id)
    converged = fit.get("converged")
    if not isinstance(converged, bool):
        message = f"{cell_id} QMLE convergence flag must be boolean"
        raise TypeError(message)
    if not converged:
        message = f"{cell_id} point QMLE did not converge"
        raise ValueError(message)
    _rho_to_lambda(rho_hat)
    return _PointFit(rho_hat=rho_hat, loglik=loglik, converged=converged)


def _validate_governed_record(
    path: Path,
    config: Paper5UniverseHorizonBridgeConfig,
    fit: _PointFit,
    window: dict[str, object],
    *,
    context: str,
) -> _GovernedRecord:
    """Reconcile a recomputed point QMLE with its governed pooled result."""
    payload = _read_json_object(path, context)
    identity = _mapping(payload.get("identity"), f"{context} identity")
    expected_identity = {
        "provider_id": config.provider_id,
        "representation_id": config.representation_id,
        "barycentre_arm_id": config.barycentre_arm_id,
        "geometry_id": config.geometry_id,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            message = (
                f"{context} operator identity {key!r} is {identity.get(key)!r}; "
                f"expected {expected!r}"
            )
            raise ValueError(message)
    primary_period = payload.get("primary_period")
    if not isinstance(primary_period, str):
        message = f"{context} requires a string primary_period"
        raise TypeError(message)
    results = _mapping(payload.get("results"), f"{context} results")
    period = _mapping(results.get(primary_period), f"{context} primary period")
    record = _mapping(period.get(_W_VARIANT), f"{context} {_W_VARIANT} result")
    governed_rho = _number(record, "rho_hat", context)
    lower = _number(record, "rho_ci_lower", context)
    upper = _number(record, "rho_ci_upper", context)
    if lower > upper:
        message = f"{context} has a reversed rho confidence interval"
        raise ValueError(message)
    _rho_to_lambda(lower)
    _rho_to_lambda(upper)
    difference = abs(fit.rho_hat - governed_rho)
    if difference > config.result_tolerance:
        message = (
            f"{context} recomputed rho differs by {difference:.3e}, above tolerance "
            f"{config.result_tolerance:.3e}"
        )
        raise ValueError(message)
    for key in ("start", "end"):
        expected_key = f"window_{key}"
        if record.get(key) != window[expected_key]:
            message = (
                f"{context} governed {key} {record.get(key)!r} does not match "
                f"recomputed {window[expected_key]!r}"
            )
            raise ValueError(message)
    trading_days = record.get("trading_days")
    if isinstance(trading_days, bool) or not isinstance(trading_days, int):
        message = f"{context} trading_days must be an integer"
        raise TypeError(message)
    if trading_days != window["trading_days"]:
        message = (
            f"{context} governed trading_days {trading_days} does not match "
            f"recomputed {window['trading_days']}"
        )
        raise ValueError(message)
    return _GovernedRecord(
        rho_hat=governed_rho,
        rho_ci_lower=lower,
        rho_ci_upper=upper,
        absolute_difference=difference,
    )


def _observed_cell(
    *,
    cell_id: str,
    universe_size: int,
    horizon: str,
    source: str,
    ci_source: str,
    window: dict[str, object],
    fit: _PointFit,
    rho_ci_lower: float,
    rho_ci_upper: float,
    governed_difference: float | None = None,
) -> dict[str, object]:
    """Build one observed cell with the monotone rho-to-lambda interval map."""
    if rho_ci_lower > rho_ci_upper:
        message = f"{cell_id} has a reversed rho confidence interval"
        raise ValueError(message)
    cell: dict[str, object] = {
        "cell_id": cell_id,
        "universe_size": universe_size,
        "horizon": horizon,
        "status": "observed",
        "source": source,
        "ci_source": ci_source,
        **window,
        "rho_hat": fit.rho_hat,
        "rho_ci_lower": rho_ci_lower,
        "rho_ci_upper": rho_ci_upper,
        "lambda_hat": _rho_to_lambda(fit.rho_hat),
        "lambda_ci_lower": _rho_to_lambda(rho_ci_lower),
        "lambda_ci_upper": _rho_to_lambda(rho_ci_upper),
        "loglik": fit.loglik,
        "converged": fit.converged,
    }
    if governed_difference is not None:
        cell["governed_rho_absolute_difference"] = governed_difference
    return cell


def _descriptive_contrasts(
    cells: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Return the two identified one-axis contrasts without inferential claims."""

    def difference(
        contrast_id: str,
        minuend_cell: str,
        subtrahend_cell: str,
    ) -> dict[str, object]:
        minuend = cells[minuend_cell]
        subtrahend = cells[subtrahend_cell]
        return {
            "contrast_id": contrast_id,
            "status": "descriptive_only",
            "minuend_cell": minuend_cell,
            "subtrahend_cell": subtrahend_cell,
            "rho_difference": _number(minuend, "rho_hat", minuend_cell)
            - _number(subtrahend, "rho_hat", subtrahend_cell),
            "lambda_difference": _number(minuend, "lambda_hat", minuend_cell)
            - _number(subtrahend, "lambda_hat", subtrahend_cell),
            "inference": "none",
        }

    return {
        "horizon_long_minus_short_at_52": difference(
            "horizon_long_minus_short_at_52",
            _REFERENCE_LONG_CELL,
            _REFERENCE_SHORT_CELL,
        ),
        "universe_100_minus_52_at_short": difference(
            "universe_100_minus_52_at_short",
            _CURRENT_SHORT_CELL,
            _REFERENCE_SHORT_CELL,
        ),
    }


def run_paper5_universe_horizon_bridge(
    paths: Paper5UniverseHorizonBridgePaths,
    config: Paper5UniverseHorizonBridgeConfig,
) -> dict[str, object]:
    """Estimate the three observed bridge cells and persist deterministic outputs."""
    config.validate()
    current_tickers, current_weights, _current_summary = _load_gate_operator(
        paths.current_barycentre_dir,
        config,
    )
    reference_tickers, reference_weights, _reference_summary = _load_gate_operator(
        paths.reference_barycentre_dir,
        config,
    )
    _validate_universes(
        reference_tickers,
        reference_weights,
        current_tickers,
        current_weights,
        config,
    )

    current_common_dates, current_common_returns = load_aligned_return_panel(
        paths.returns_dir,
        reference_tickers,
        config.start,
        config.short_end,
        oos_returns_dir=paths.current_oos_returns_dir,
    )
    reference_short_dates, reference_short_returns = load_aligned_return_panel(
        paths.returns_dir,
        reference_tickers,
        config.start,
        config.short_end,
        oos_returns_dir=paths.reference_oos_returns_dir,
    )
    reference_long_dates, reference_long_returns = load_aligned_return_panel(
        paths.returns_dir,
        reference_tickers,
        config.start,
        config.long_end,
        oos_returns_dir=paths.reference_oos_returns_dir,
    )
    current_short_dates, current_short_returns = load_aligned_return_panel(
        paths.returns_dir,
        current_tickers,
        config.start,
        config.short_end,
        oos_returns_dir=paths.current_oos_returns_dir,
    )
    common_return_check = _verify_common_return_panel(
        current_common_dates,
        current_common_returns,
        reference_short_dates,
        reference_short_returns,
        tolerance=config.shared_return_tolerance,
    )

    reference_short_window = _window_metadata(
        reference_short_dates,
        reference_short_returns,
    )
    reference_long_window = _window_metadata(
        reference_long_dates,
        reference_long_returns,
    )
    current_short_window = _window_metadata(
        current_short_dates,
        current_short_returns,
    )
    reference_short_fit = _fit_point_qmle(
        reference_short_returns,
        reference_weights,
        cell_id=_REFERENCE_SHORT_CELL,
    )
    reference_long_fit = _fit_point_qmle(
        reference_long_returns,
        reference_weights,
        cell_id=_REFERENCE_LONG_CELL,
    )
    current_short_fit = _fit_point_qmle(
        current_short_returns,
        current_weights,
        cell_id=_CURRENT_SHORT_CELL,
    )
    reference_governed = _validate_governed_record(
        paths.reference_results_json,
        config,
        reference_long_fit,
        reference_long_window,
        context="reference 52-firm long-horizon result",
    )
    current_governed = _validate_governed_record(
        paths.current_results_json,
        config,
        current_short_fit,
        current_short_window,
        context="current 100-firm short-horizon result",
    )

    indices = stationary_bootstrap_indices(
        len(reference_short_returns),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed,
    )
    rho_draws = _bootstrap_rho(
        reference_short_returns,
        reference_weights,
        indices,
    )
    if rho_draws.shape != (config.n_bootstrap,):
        message = (
            f"52-firm short bootstrap returned shape {rho_draws.shape}; expected "
            f"({config.n_bootstrap},)"
        )
        raise ValueError(message)
    lambda_draws = _rho_draws_to_lambda(rho_draws)
    short_lower = float(np.quantile(rho_draws, _LOWER_QUANTILE))
    short_upper = float(np.quantile(rho_draws, _UPPER_QUANTILE))

    observed_cells = {
        _REFERENCE_LONG_CELL: _observed_cell(
            cell_id=_REFERENCE_LONG_CELL,
            universe_size=config.expected_reference_universe,
            horizon="long",
            source="governed_reference_verified",
            ci_source="supplied_governed_result",
            window=reference_long_window,
            fit=reference_long_fit,
            rho_ci_lower=reference_governed.rho_ci_lower,
            rho_ci_upper=reference_governed.rho_ci_upper,
            governed_difference=reference_governed.absolute_difference,
        ),
        _REFERENCE_SHORT_CELL: _observed_cell(
            cell_id=_REFERENCE_SHORT_CELL,
            universe_size=config.expected_reference_universe,
            horizon="short",
            source="new_bridge_estimate",
            ci_source="new_stationary_bootstrap",
            window=reference_short_window,
            fit=reference_short_fit,
            rho_ci_lower=short_lower,
            rho_ci_upper=short_upper,
        ),
        _CURRENT_SHORT_CELL: _observed_cell(
            cell_id=_CURRENT_SHORT_CELL,
            universe_size=config.expected_current_universe,
            horizon="short",
            source="governed_current_verified",
            ci_source="supplied_governed_result",
            window=current_short_window,
            fit=current_short_fit,
            rho_ci_lower=current_governed.rho_ci_lower,
            rho_ci_upper=current_governed.rho_ci_upper,
            governed_difference=current_governed.absolute_difference,
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "design": {
            "start": config.start,
            "short_end": config.short_end,
            "long_end": config.long_end,
            "expected_reference_universe": config.expected_reference_universe,
            "expected_current_universe": config.expected_current_universe,
            "reference_revision": config.reference_revision,
            "specification_order": [
                _REFERENCE_LONG_CELL,
                _REFERENCE_SHORT_CELL,
                _CURRENT_SHORT_CELL,
            ],
        },
        "operator_identity": {
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "barycentre_arm_id": config.barycentre_arm_id,
            "geometry_id": config.geometry_id,
        },
        "universes": {
            "reference": {
                "n_firms": len(reference_tickers),
                "tickers": reference_tickers,
            },
            "current": {
                "n_firms": len(current_tickers),
                "tickers": current_tickers,
                "contains_reference": True,
            },
        },
        "common_return_check": common_return_check,
        "inference": {
            "newly_bootstrapped_cell": _REFERENCE_SHORT_CELL,
            "method": "joint-date stationary bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "seed": config.seed,
            "governed_result_tolerance": config.result_tolerance,
            "governed_ci_cells": [_REFERENCE_LONG_CELL, _CURRENT_SHORT_CELL],
        },
        "cells": observed_cells,
        "contrasts": _descriptive_contrasts(observed_cells),
        "scope": {
            "interpretation": "descriptive_noncausal",
            "reference_long_role": "historical_comparison",
        },
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "summary.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "cell_id": [_REFERENCE_SHORT_CELL] * config.n_bootstrap,
            "draw_id": np.arange(config.n_bootstrap, dtype=np.int64),
            "rho_draw": rho_draws,
            "lambda_draw": lambda_draws,
            "solve_ok": np.isfinite(rho_draws),
        }
    ).to_parquet(paths.output_dir / "bootstrap_draws.parquet", index=False)
    return payload


__all__ = [
    "Paper5UniverseHorizonBridgeConfig",
    "Paper5UniverseHorizonBridgePaths",
    "run_paper5_universe_horizon_bridge",
]

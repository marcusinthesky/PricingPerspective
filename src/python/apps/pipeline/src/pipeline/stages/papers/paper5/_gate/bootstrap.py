"""Dependent-date and randomization inference for the Paper 5 gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from pipeline.stages.papers.paper5._gate.metrics import covariance_row_tv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def stationary_bootstrap_indices(
    observations: int,
    *,
    draws: int,
    expected_block_length: int,
    seed: int,
) -> NDArray[np.int64]:
    """Draw Politis--Romano stationary-bootstrap date indices."""
    rng = np.random.default_rng(seed)
    indices = np.empty((draws, observations), dtype=np.int64)
    restart_probability = 1.0 / expected_block_length
    indices[:, 0] = rng.integers(0, observations, size=draws)
    restarts = rng.random((draws, observations - 1)) < restart_probability
    restart_values = rng.integers(0, observations, size=(draws, observations - 1))
    for time_index in range(1, observations):
        continued = (indices[:, time_index - 1] + 1) % observations
        indices[:, time_index] = np.where(
            restarts[:, time_index - 1],
            restart_values[:, time_index - 1],
            continued,
        )
    return indices


def paired_rmse_bootstrap_pvalue(
    focal_date_mse: NDArray[np.float64],
    comparator_date_mse: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[float, NDArray[np.float64]]:
    """Return a centered one-sided bootstrap p-value for lower focal RMSE."""
    observed = float(
        np.sqrt(np.mean(focal_date_mse)) - np.sqrt(np.mean(comparator_date_mse))
    )
    focal_boot = np.sqrt(np.mean(focal_date_mse[indices], axis=1))
    comparator_boot = np.sqrt(np.mean(comparator_date_mse[indices], axis=1))
    centered = focal_boot - comparator_boot - observed
    p_value = float((1 + np.count_nonzero(centered <= observed)) / (len(centered) + 1))
    return p_value, focal_boot - comparator_boot


def covariance_bootstrap_pvalue(
    returns: NDArray[np.float64],
    focal_weights: NDArray[np.float64],
    comparator_weights: NDArray[np.float64],
    indices: NDArray[np.int64],
    *,
    active_tolerance: float,
) -> tuple[float, NDArray[np.float64]]:
    """Recompute correlation-row TV under each joint-date bootstrap draw."""
    observed_focal, _ = covariance_row_tv(
        returns,
        focal_weights,
        active_tolerance=active_tolerance,
    )
    observed_comparator, _ = covariance_row_tv(
        returns,
        comparator_weights,
        active_tolerance=active_tolerance,
    )
    observed = observed_focal - observed_comparator
    differences = np.empty(indices.shape[0], dtype=np.float64)
    for draw, date_indices in enumerate(indices):
        sampled = returns[date_indices]
        focal, _ = covariance_row_tv(
            sampled,
            focal_weights,
            active_tolerance=active_tolerance,
        )
        comparator, _ = covariance_row_tv(
            sampled,
            comparator_weights,
            active_tolerance=active_tolerance,
        )
        differences[draw] = focal - comparator
    centered = differences - observed
    p_value = float((1 + np.count_nonzero(centered <= observed)) / (len(centered) + 1))
    return p_value, differences


def diebold_mariano_agreement(
    focal_date_mse: NDArray[np.float64],
    comparator_date_mse: NDArray[np.float64],
    *,
    max_lag: int,
) -> dict[str, float | bool]:
    """Compute the registered one-sided HAC Diebold--Mariano agreement check."""
    differential = focal_date_mse - comparator_date_mse
    observations = len(differential)
    mean = float(np.mean(differential))
    centered = differential - mean
    long_run_variance = float(centered @ centered / observations)
    usable_lag = min(max_lag, observations - 1)
    for lag in range(1, usable_lag + 1):
        weight = 1.0 - lag / (usable_lag + 1)
        autocovariance = float(centered[lag:] @ centered[:-lag] / observations)
        long_run_variance += 2.0 * weight * autocovariance
    long_run_variance = max(long_run_variance, np.finfo(np.float64).eps)
    statistic = mean / np.sqrt(long_run_variance / observations)
    p_value = float(norm.cdf(statistic))
    return {
        "loss_differential_mean": mean,
        "statistic": float(statistic),
        "p_value": p_value,
        "favours_focal": bool(mean < 0.0),
    }


def randomization_pvalue(focal: float, random_values: NDArray[np.float64]) -> float:
    """Return the lower-tail randomization p-value for an error metric."""
    return float(
        (1 + np.count_nonzero(random_values <= focal)) / (len(random_values) + 1)
    )


__all__ = [
    "covariance_bootstrap_pvalue",
    "diebold_mariano_agreement",
    "paired_rmse_bootstrap_pvalue",
    "randomization_pvalue",
    "stationary_bootstrap_indices",
]

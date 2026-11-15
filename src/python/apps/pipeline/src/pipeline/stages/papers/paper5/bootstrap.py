"""Dependent-date bootstrap resampling for Paper 5 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

CALIBRATION_SEED_OFFSET = 0
EVALUATION_FIT_SEED_OFFSET = 1
EVALUATION_SCORE_SEED_OFFSET = 2


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


def independent_stationary_bootstrap_indices(
    observations: int,
    n_assets: int,
    *,
    draws: int,
    expected_block_length: int,
    seed: int,
) -> NDArray[np.int64]:
    """Draw independent stationary-bootstrap paths for every asset.

    The returned array has shape ``(draws, n_assets, observations)``. A
    separate seeded generator per asset prevents null replicates from sharing
    cyclic path assignments.
    """
    if observations < 1 or n_assets < 1:
        message = "observations and n_assets must be positive"
        raise ValueError(message)
    if draws < 1 or expected_block_length < 1:
        message = "draws and expected_block_length must be positive"
        raise ValueError(message)
    paths = [
        stationary_bootstrap_indices(
            observations,
            draws=draws,
            expected_block_length=expected_block_length,
            seed=seed + asset,
        )
        for asset in range(n_assets)
    ]
    return np.stack(paths, axis=1)


@dataclass(frozen=True, slots=True)
class PairedRefitSchedule:
    """The three date schedules a paired out-of-period refit bootstrap needs.

    Every Paper 5 held-out comparison resamples the same way: one schedule for
    the calibration panel the scored parameters are fitted on, one for the
    evaluation panel any in-period coefficient is refitted on, and one for the
    evaluation dates the fitted model is scored over. A single seed convention
    across stages is what lets one gain carry one interval wherever it appears.
    """

    calibration: NDArray[np.int64]
    evaluation_fit: NDArray[np.int64]
    evaluation_score: NDArray[np.int64]
    draws: int
    expected_block_length: int
    seed: int


def paired_refit_schedule(
    *,
    calibration_observations: int,
    evaluation_observations: int,
    draws: int,
    expected_block_length: int,
    seed: int,
) -> PairedRefitSchedule:
    """Draw the canonical calibration, evaluation-fit, and evaluation-score paths."""
    if calibration_observations < 1 or evaluation_observations < 1:
        message = "calibration and evaluation observations must be positive"
        raise ValueError(message)
    if draws < 1 or expected_block_length < 1:
        message = "draws and expected_block_length must be positive"
        raise ValueError(message)
    return PairedRefitSchedule(
        calibration=stationary_bootstrap_indices(
            calibration_observations,
            draws=draws,
            expected_block_length=expected_block_length,
            seed=seed + CALIBRATION_SEED_OFFSET,
        ),
        evaluation_fit=stationary_bootstrap_indices(
            evaluation_observations,
            draws=draws,
            expected_block_length=expected_block_length,
            seed=seed + EVALUATION_FIT_SEED_OFFSET,
        ),
        evaluation_score=stationary_bootstrap_indices(
            evaluation_observations,
            draws=draws,
            expected_block_length=expected_block_length,
            seed=seed + EVALUATION_SCORE_SEED_OFFSET,
        ),
        draws=draws,
        expected_block_length=expected_block_length,
        seed=seed,
    )


def paired_refit_provenance(schedule: PairedRefitSchedule) -> dict[str, object]:
    """Describe the resampling architecture for an artifact payload."""
    return {
        "draws": schedule.draws,
        "expected_block_length": schedule.expected_block_length,
        "seed": schedule.seed,
        "refit_parameters": True,
        "architecture": "paired_refit_stationary_block",
        "seed_offsets": {
            "calibration": CALIBRATION_SEED_OFFSET,
            "evaluation_fit": EVALUATION_FIT_SEED_OFFSET,
            "evaluation_score": EVALUATION_SCORE_SEED_OFFSET,
        },
    }


__all__ = [
    "CALIBRATION_SEED_OFFSET",
    "EVALUATION_FIT_SEED_OFFSET",
    "EVALUATION_SCORE_SEED_OFFSET",
    "PairedRefitSchedule",
    "independent_stationary_bootstrap_indices",
    "paired_refit_provenance",
    "paired_refit_schedule",
    "stationary_bootstrap_indices",
]

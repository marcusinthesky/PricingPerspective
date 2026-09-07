"""Paper 5 energy-robustness contracts: strategy roster and dataclasses.

These are the types ``pipeline.stages.papers.paper5.energy_robust`` re-exports and that
every other ``_energy_robust`` module depends on; keeping them in the DAG root is what
makes the package acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path

Float = NDArray[np.float64]

STRATEGIES = (
    "sample",
    "ledoit_wolf",
    "sigma_dist",
    "robust_corner",
    "higham_corner",
    "robust_corner_cert",
    "cert_stale_vol",
    "robust_ridge",
    "robust_select",
    "equal_weight",
)
MAX_MATERIALITY_RATIO = 2.0


class RobustPortfolioError(ValueError):
    """Report invalid robust-portfolio inputs or strategy configuration."""


class RobustPortfolioComputationError(RuntimeError):
    """Report a failed robust-portfolio numerical solve."""


@dataclass(frozen=True)
class RobustOptimizationRequest:
    """Method selection and method-specific covariance inputs."""

    method: str
    sigma_hat: Float | None = None
    sigma_hi: Float | None = None
    sigma_lo: Float | None = None
    radius: float | None = None
    cap: float = 1.0
    n_steps: int = 1500


@dataclass(frozen=True)
class RobustCovarianceInputs:
    """Calibration-fixed ambiguity inputs shared across backtest windows."""

    squared_distances: Float
    eps_stat_matrix: Float
    eps_gmm_d2: Float
    envelope: dict[str, float]
    r_by_window: dict[int, float]
    calibration_kappa: float
    calibration_sigma_sq: Float


@dataclass(frozen=True)
class RobustBacktestInputs:
    """Evaluation returns and calibration-fixed covariance inputs."""

    returns: Float
    covariance: RobustCovarianceInputs


@dataclass(frozen=True)
class RobustBacktestConfig:
    """Window, cost, bootstrap, and seed controls for the robust backtest."""

    windows: tuple[int, ...] = (10, 20, 60, 250)
    cost_bps: tuple[float, ...] = (10.0, 50.0)
    n_boot: int = 500
    seed: int = 0


@dataclass(frozen=True)
class Paper5EnergyRobustnessPaths:
    """Input and output artifact boundaries for Paper 5 energy-robustness empirics."""

    returns_dir: Path
    eval_returns_dir: Path
    energy_tests_path: Path
    embeddings_dir: Path
    frontier_metrics_path: Path
    output_dir: Path


@dataclass(frozen=True)
class Paper5EnergyRobustnessConfig:
    """Epoch, backtest, bootstrap, and seed controls for Paper 5 empirics."""

    calib_start: str = "2018-01-01"
    calib_end: str = "2022-12-31"
    eval_start: str = "2023-01-01"
    eval_end: str = "2026-07-16"
    windows: tuple[int, ...] = (10, 20, 60, 250)
    n_boot_stat: int = 300
    n_boot_ridge: int = 200
    n_boot_sharpe: int = 500
    seed: int = 0
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    # Tracks the p5_energy_robustness stage selector; see the CLI option note.
    distance_id: str = "wasserstein_w2"

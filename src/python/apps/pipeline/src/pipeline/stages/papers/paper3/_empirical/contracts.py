"""Paper 3 empirical contracts: path/config dataclasses and shared constants.

These are the types ``pipeline.stages.papers.paper3.empirical`` re-exports and
that every other ``_empirical`` module depends on; keeping them in the DAG root
is what makes the package acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

Float = NDArray[np.float64]
SIGNIFICANCE_LEVEL = 0.05
MIN_BOOTSTRAP_OBSERVATIONS = 3


class Paper3EmpiricalError(ValueError):
    """Report invalid empirical inputs or an internal protocol violation."""


@dataclass(frozen=True)
class Paper3EmpiricalPaths:
    """Artifact boundaries for one Paper 3 empirical run."""

    returns_dir: Path
    distance_artifact_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class Paper3EmpiricalConfig:
    """Numerical controls shared by every step of an empirical run."""

    windows: tuple[int, ...] = (10, 20, 60, 250)
    n_clusters: int = 15
    n_boot: int = 500
    n_mc: int = 3000
    seed: int = 0
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    distance_id: str = "wasserstein_w2"


@dataclass(frozen=True)
class IdentityTestInputs:
    """Panel and calibrated envelope-saturation parameters for inference."""

    returns: Float
    squared_distances: Float
    kappa_hat: float
    kappa_se: float


@dataclass(frozen=True)
class IdentityTestConfig:
    """Simulation controls shared by the saturation and coverage tests."""

    n_clusters: int = 15
    n_mc: int = 3000
    seed: int = 0


@dataclass(frozen=True)
class BacktestInputs:
    """Panel and optional PIT matrices, eligible only after their cutoff date."""

    returns: Float
    squared_distances: Float
    dates: pd.DatetimeIndex | None = None
    pit_matrices: dict[str, Float] | None = None
    pit_dates: pd.DatetimeIndex | None = None


@dataclass(frozen=True)
class BacktestConfig:
    """Window, transaction-cost, and resampling controls for a backtest."""

    windows: tuple[int, ...] = (10, 20, 60, 250)
    cost_bps: tuple[float, ...] = (10.0, 50.0)
    n_boot: int = 500
    seed: int = 0

"""Configuration, artifact boundaries, and shared records for Paper 5 empirics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipeline.stages.papers.paper5._gate.contracts import POST_CORPUS_END_DATE

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

_IN_SAMPLE_FOLDS: dict[str, tuple[str, str]] = {
    "eval2020": ("2020-01-01", "2020-12-31"),
    "eval2021": ("2021-01-01", "2021-12-31"),
    "eval2022": ("2022-01-01", "2022-12-31"),
}
_OOS_FOLDS: dict[str, tuple[str, str]] = {
    "eval2023": ("2023-01-01", "2023-12-31"),
    "eval2024": ("2024-01-01", "2024-12-31"),
    "eval2025": ("2025-01-01", "2025-12-31"),
    "eval2026": ("2026-01-01", POST_CORPUS_END_DATE),
}
_CALIB_START = "2018-01-01"
_RHO_SE_FALLBACK = 0.05

EXPECTED_TICKERS = 100
ACTIVE_WEIGHT_TOLERANCE = 1e-8


class Paper5EmpiricalError(ValueError):
    """Report a violated Paper 5 empirical-data contract."""


@dataclass(frozen=True)
class Paper5EmpiricalPaths:
    """Input and output artifact boundaries for Paper 5 empirics."""

    returns_dir: Path
    distance_artifact_dir: Path
    embeddings_dir: Path
    universe_csv: Path
    output_dir: Path
    shared_barycentre_dir: Path
    wasserstein_w1_barycentre_dir: Path
    co_mentions_adjacency: Path
    oos_returns_dir: Path | None = None


@dataclass(frozen=True)
class Paper5EmpiricalConfig:
    """Bootstrap, neighborhood, and seed controls for Paper 5 empirics."""

    n_boot_stat: int = 300
    knn_k: int = 5
    seed: int = 0
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    distance_id: str = "wasserstein_w2"
    barycentre_arm_id: str = "wasserstein_w2_loo"
    wasserstein_w1_arm_id: str = "wasserstein_w1_loo"


@dataclass(frozen=True)
class _RhoFitInputs:
    """Fold, artifact, universe, and spatial-weight inputs for SAR fitting."""

    folds: dict[str, tuple[str, str]]
    paths: Paper5EmpiricalPaths
    tickers: list[str]
    frozen_w_variants: dict[str, np.ndarray]
    knn_k: int


@dataclass(frozen=True)
class _H3EvaluationInputs:
    """Shared fitted state and comparison benchmarks for H3 evaluation."""

    fold_data: dict[str, dict[str, np.ndarray]]
    rho_df: pd.DataFrame
    w_variants: dict[str, np.ndarray]
    flat_bound: dict[str, float | bool | str]
    delta_w_h: dict[str, float]
    coverage: tuple[float, float]
    lw_band: float


__all__ = [
    "ACTIVE_WEIGHT_TOLERANCE",
    "EXPECTED_TICKERS",
    "Paper5EmpiricalConfig",
    "Paper5EmpiricalError",
    "Paper5EmpiricalPaths",
]

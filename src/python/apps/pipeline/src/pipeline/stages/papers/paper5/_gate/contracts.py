"""Configuration and artifact contracts for the Paper 5 economic gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from pathlib import Path

POST_CORPUS_START_DATE = "2023-01-01"
POST_CORPUS_END_DATE = "2026-04-06"

GATE_FOLDS: dict[str, tuple[str, str]] = {
    "eval2023": (POST_CORPUS_START_DATE, "2023-12-31"),
    "eval2024": ("2024-01-01", "2024-12-31"),
    "eval2025": ("2025-01-01", "2025-12-31"),
    "eval2026": ("2026-01-01", POST_CORPUS_END_DATE),
}
CALIBRATION_START = "2018-01-01"
REGISTERED_PASSING_FOLDS = 3


class Paper5GateError(ValueError):
    """Report a violation of the preregistered gate contract."""


def raise_gate_error(message: str) -> NoReturn:
    """Raise the gate's domain error without embedding literals at call sites."""
    raise Paper5GateError(message)


@dataclass(frozen=True)
class Paper5GatePaths:
    """Input and output boundaries for the economic gate."""

    returns_dir: Path
    oos_returns_dir: Path
    shared_barycentre_dir: Path
    universe_csv: Path
    output_dir: Path


@dataclass(frozen=True)
class Paper5GateConfig:
    """Pre-registered resampling, comparator, and decision controls."""

    n_bootstrap: int = 2000
    n_random: int = 1000
    block_length: int = 21
    knn_k: int = 5
    seed: int = 0
    alpha: float = 0.05
    minimum_rmse_reduction: float = 0.05
    minimum_passing_folds: int = 3
    active_weight_tolerance: float = 1e-8
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    barycentre_arm_id: str = "wasserstein_w2_loo"
    geometry_id: str = "wasserstein_w2"

    def validate(self) -> None:
        """Reject a runtime configuration that changes the registered design."""
        if self.n_bootstrap < 1 or self.n_random < 1:
            raise_gate_error("bootstrap and random draw counts must be positive")
        if self.block_length < 1 or self.knn_k < 1:
            raise_gate_error("block length and kNN size must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise_gate_error("alpha must lie strictly between zero and one")
        if not 0.0 < self.minimum_rmse_reduction < 1.0:
            raise_gate_error("minimum RMSE reduction must lie in (0, 1)")
        if self.minimum_passing_folds != REGISTERED_PASSING_FOLDS:
            raise_gate_error("the registered random-floor rule is exactly 3 of 4")


__all__ = [
    "GATE_FOLDS",
    "POST_CORPUS_END_DATE",
    "POST_CORPUS_START_DATE",
    "Paper5GateConfig",
    "Paper5GateError",
    "Paper5GatePaths",
    "raise_gate_error",
]

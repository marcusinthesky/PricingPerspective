"""Immutable input contracts for the shared W2 exposure frontier."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_WOLAK_ELL0 = 0.0


class H1Config(BaseModel):
    """Base for immutable H1 stage configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class H1InputPaths(H1Config):
    """Artifacts consumed by every H1 returns-frontier stage."""

    returns_dir: Path
    distance_artifact_dir: Path

    @field_validator("returns_dir", "distance_artifact_dir")
    @classmethod
    def _normalize_input_path(cls, value: Path) -> Path:
        """Normalize path-like inputs at the artifact boundary."""
        return Path(value)


class H1ArtifactPaths(H1InputPaths):
    """H1 input artifacts plus the destination for a stage report."""

    output_dir: Path

    @field_validator("output_dir")
    @classmethod
    def _normalize_output_path(cls, value: Path) -> Path:
        """Normalize the report destination at the artifact boundary."""
        return Path(value)


class H1FrontierConfig(H1Config):
    """Settings shared by the H1 pilot and OOS frontier calibration."""

    train_start: str = "2018-01-01"
    train_end: str = "2021-12-31"
    ks: tuple[int, ...] = (3, 5, 10)
    k_target_explained_variance: float = 0.80
    tau_lo: float = 0.05
    tau_hi: float = 0.95
    n_boot: int = 1000
    n_clusters: int = 15
    n_mc: int = 3000
    n_dirichlet: int = 8
    wolak_ell0: float = DEFAULT_WOLAK_ELL0
    seed: int = 42


class H1PilotConfig(H1FrontierConfig):
    """Decision thresholds for the in-sample H1 frontier pilot."""

    cap_kill_threshold: float = 0.05
    cap_marginal_threshold: float = 0.10
    wolak_alpha: float = 0.05


class H1OOSCoverageConfig(H1FrontierConfig):
    """Test-window settings for the H1 out-of-sample coverage stage."""

    test_start: str = "2022-01-01"
    test_end: str = "2022-12-31"
    coverage_target: float = 0.90


class FrontierBootstrapConfig(H1Config):
    """Controls for one returns-panel frontier bootstrap."""

    k: int
    tau_lo: float = 0.05
    n_boot: int = 1000
    n_clusters: int = 15
    n_mc: int = 3000
    seed: int = 42
    wolak_ell0: float = DEFAULT_WOLAK_ELL0
    block_length: int | None = None


class HedgingCoverageConfig(H1Config):
    """Fixed design controls for hedging-error coverage evaluation."""

    n_dirichlet: int
    seed: int


class PseudoIPOConfig(H1Config):
    """Leave-one-out pseudo-IPO frontier settings."""

    train_start: str = "2018-01-01"
    train_end: str = "2021-12-31"
    k: int = 5
    tau_lo: float = 0.05
    tau_hi: float = 0.95


class ReturnsLoadingsConfig(H1Config):
    """Whitened-PCA loadings report settings."""

    train_start: str = "2018-01-01"
    train_end: str = "2021-12-31"
    ks: tuple[int, ...] = (3, 5, 10)
    k_target_explained_variance: float = 0.80


DEFAULT_H1_PILOT_CONFIG = H1PilotConfig()
DEFAULT_H1_OOS_COVERAGE_CONFIG = H1OOSCoverageConfig()
DEFAULT_PSEUDO_IPO_CONFIG = PseudoIPOConfig()
DEFAULT_RETURNS_LOADINGS_CONFIG = ReturnsLoadingsConfig()


__all__ = [
    "DEFAULT_H1_OOS_COVERAGE_CONFIG",
    "DEFAULT_H1_PILOT_CONFIG",
    "DEFAULT_PSEUDO_IPO_CONFIG",
    "DEFAULT_RETURNS_LOADINGS_CONFIG",
    "DEFAULT_WOLAK_ELL0",
    "FrontierBootstrapConfig",
    "H1ArtifactPaths",
    "H1Config",
    "H1FrontierConfig",
    "H1InputPaths",
    "H1OOSCoverageConfig",
    "H1PilotConfig",
    "HedgingCoverageConfig",
    "PseudoIPOConfig",
    "ReturnsLoadingsConfig",
]

"""Shared W2 exposure-frontier calibration used by downstream papers."""

from pipeline.stages.substrate.w2_exposure_frontier.contracts import (
    DEFAULT_H1_PILOT_CONFIG,
    H1ArtifactPaths,
    H1PilotConfig,
)
from pipeline.stages.substrate.w2_exposure_frontier.driver import run_h1_pilot

__all__ = [
    "DEFAULT_H1_PILOT_CONFIG",
    "H1ArtifactPaths",
    "H1PilotConfig",
    "run_h1_pilot",
]

"""Stable facade for Paper 5's preregistered economic-validation gate."""

from pipeline.stages.papers.paper5._gate.contracts import (
    GATE_FOLDS,
    Paper5GateConfig,
    Paper5GateError,
    Paper5GatePaths,
)
from pipeline.stages.papers.paper5._gate.driver import (
    classify_gate,
    run_paper5_economic_gate,
)

__all__ = [
    "GATE_FOLDS",
    "Paper5GateConfig",
    "Paper5GateError",
    "Paper5GatePaths",
    "classify_gate",
    "run_paper5_economic_gate",
]

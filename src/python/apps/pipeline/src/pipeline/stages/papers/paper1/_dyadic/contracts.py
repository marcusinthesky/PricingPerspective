"""Paper-specific contracts and specification constants for the dyadic stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from jcor.model.dyadic import DyadicSample

DEFAULT_BOOTSTRAP_ITERS = 1_999
DEFAULT_BOOTSTRAP_SEED = 42
HISTORICAL_SPEC_INDEX = 3
PRIMARY_SPEC_INDEX = 6

# The squared-distance candidate is a Paper-1 interpretation decision, not a
# reusable statistical-method contract, so it stays at the pipeline boundary.
SQUARED_DISTANCE_ARBITRATION: dict[str, object] = {
    "candidate": "return chord distance squared on energy distance squared "
    "with symmetric additive firm effects",
    "decision": "rejected",
    "decided": "2026-07-28",
    "reason": (
        "squaring does not preserve the additive firm margin: the endpoint "
        "structure implied for a squared regressor is (a_i + a_j)^2, whose "
        "cross term the symmetric-FE design cannot absorb, so the coefficient "
        "is not a within-margin quantity; with firm exposure scales and "
        "systematic variance shares unobserved it also carries no bounded "
        "interpretation"
    ),
    "retained_alternative": (
        "prespecified centred energy-distance-squared rung under symmetric "
        "firm effects, reported as functional_form_sensitivity"
    ),
}


@dataclass(frozen=True)
class DyadicConfoundConfig:
    """Artifact paths and bootstrap controls for the Paper 1 ladder."""

    distance_artifact_dir: Path
    energy_comparator_artifact_dir: Path
    covariance_matrix: Path
    universe_csv: Path
    market_data_dir: Path
    returns_dir: Path
    output_file: Path
    embeddings_dir: Path = Path("data/shared/embeddings/qwen3-embedding-8b")
    bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class PreparedDyadicInputs:
    """Aligned dyadic panel and optional within-firm dispersion diagnostics."""

    tickers: list[str]
    columns: dict[str, np.ndarray]
    sample: DyadicSample[str]
    dispersions: dict[str, float] | None
    dispersion_note: str | None
    u_sensitivity: dict[str, object] | None


@dataclass(frozen=True)
class LadderFits:
    """Model-ladder summaries and selected full fit payloads."""

    model_ladder: list[dict[str, object]]
    bootstrap_frames: list[pd.DataFrame]
    historical: dict[str, object]
    primary: dict[str, object]


@dataclass(frozen=True)
class DyadicArtifacts:
    """Summary and auditable frames emitted by the dyadic stage."""

    output_file: Path
    result: dict[str, object]
    tickers: list[str]
    node_counts: np.ndarray
    bootstrap_frames: list[pd.DataFrame]
    design_diagnostics: pd.DataFrame
    primary_fit: pd.DataFrame
    leave_one_out: pd.DataFrame
    residual_scale_null: pd.DataFrame
    random_exposure_diagnostic: pd.DataFrame

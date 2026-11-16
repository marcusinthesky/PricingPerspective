"""Adapt typed energy artifacts to Paper 1 pair records."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact


def load_energy_functional_pairs(
    artifact_dir: Path,
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> pd.DataFrame:
    """Adapt the typed full matrix to legacy paper-local pair column names."""
    frame, _summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
        },
    )
    return frame.rename(
        columns={"item_i": "symbol1", "item_j": "symbol2", "value": "energy_distance"}
    )

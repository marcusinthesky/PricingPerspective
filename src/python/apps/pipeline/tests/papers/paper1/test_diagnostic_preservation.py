"""Typed-artifact contract for the Paper 1 preservation diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pipeline.stages.papers.paper1 import diagnostics as preservation

if TYPE_CHECKING:
    import pytest


def test_distance_preservation_loads_explicit_typed_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each width carries an explicit path and expected representation identity."""
    calls: list[tuple[Path, str, str, str]] = []
    values = {
        "full": [0.0, 1.0, 4.0, 1.0, 0.0, 9.0, 4.0, 9.0, 0.0],
        "dim64": [0.0, 2.0, 3.0, 2.0, 0.0, 8.0, 3.0, 8.0, 0.0],
        "dim256": [0.0, 1.2, 3.8, 1.2, 0.0, 8.7, 3.8, 8.7, 0.0],
    }

    def fake_load(
        artifact_dir: Path,
        *,
        expected_identity: dict[str, str],
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append(
            (
                artifact_dir,
                expected_identity["provider_id"],
                expected_identity["representation_id"],
                expected_identity["distance_id"],
            )
        )
        frame = pd.DataFrame(
            {
                "item_i": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "item_j": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "value": values[artifact_dir.name],
            }
        )
        return frame, {"item_ids": ["A", "B", "C"]}

    monkeypatch.setattr(preservation, "read_typed_distance_artifact", fake_load)
    frames, stats = preservation.distance_preservation(
        full_artifact_dir=Path("full"),
        truncated_artifact_dirs={64: Path("dim64"), 256: Path("dim256")},
        provider_id="qwen3-embedding-8b",
        full_representation_id="qwen3-embedding-8b-unit",
        truncated_representation_ids={
            64: "qwen3-embedding-8b-64-unit",
            256: "qwen3-embedding-8b-256-unit",
        },
        distance_id="wasserstein_w2",
        widths=(64, 256),
    )

    assert calls == [
        (
            Path("full"),
            "qwen3-embedding-8b",
            "qwen3-embedding-8b-unit",
            "wasserstein_w2",
        ),
        (
            Path("dim64"),
            "qwen3-embedding-8b",
            "qwen3-embedding-8b-64-unit",
            "wasserstein_w2",
        ),
        (
            Path("dim256"),
            "qwen3-embedding-8b",
            "qwen3-embedding-8b-256-unit",
            "wasserstein_w2",
        ),
    ]
    assert set(frames) == {64, 256}
    expected_pair_count = 3
    assert stats[64]["n_pairs"] == expected_pair_count
    assert stats[256]["n_pairs"] == expected_pair_count

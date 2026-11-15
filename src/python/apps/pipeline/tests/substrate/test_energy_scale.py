"""Scale-contract guards for the immutable energy functional."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from pipeline.stages.substrate import energy_shared
from pipeline.stages.substrate.energy_metric import (
    _energy_metric_from_functional,
)
from pipeline.stages.substrate.energy_shared import (
    load_energy_functional_pairs,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_metric_squared_recovers_functional() -> None:
    """Squaring the projected metric recovers the energy functional."""
    functional = np.array([[0.0, 0.04], [0.04, 0.0]])
    metric = _energy_metric_from_functional(functional)
    assert np.allclose(metric**2, functional)


def test_loader_requires_explicit_metric_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The functional loader does not silently apply the metric projection."""
    frame = pd.DataFrame(
        {
            "item_i": ["A", "A", "B", "B"],
            "item_j": ["A", "B", "A", "B"],
            "value": [0.0, 0.25, 0.25, 0.0],
        }
    )
    monkeypatch.setattr(
        energy_shared,
        "read_typed_distance_artifact",
        lambda *_args, **_kwargs: (frame, {"item_ids": ["A", "B"]}),
    )
    path = tmp_path / "typed-distance"

    identity = {
        "provider_id": "qwen3-embedding-8b",
        "representation_id": "qwen3-embedding-8b-unit",
        "distance_id": "energy_v",
    }
    pairs = load_energy_functional_pairs(path, **identity)
    functional = pairs["energy_distance"].to_numpy()
    metric = _energy_metric_from_functional(functional)

    assert functional[1] == pytest.approx(0.25)
    assert metric[1] == pytest.approx(0.5)
    assert np.allclose(metric**2, functional)


def test_loader_requires_explicit_typed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared loader maps typed matrix rows to the energy contract."""
    captured: dict[str, object] = {}

    def fake_reader(*_args: object, **kwargs: object) -> tuple[object, object]:
        captured.update(kwargs)
        message = "distance artifact does not match expected provider_id"
        raise ValueError(message)

    monkeypatch.setattr(energy_shared, "read_typed_distance_artifact", fake_reader)
    path = tmp_path / "typed-distance"

    with pytest.raises(ValueError, match="expected provider_id"):
        load_energy_functional_pairs(
            path,
            provider_id="qwen3-embedding-8b",
            representation_id="qwen3-embedding-8b-unit",
            distance_id="energy_v",
        )
    assert captured["expected_identity"] == {
        "provider_id": "qwen3-embedding-8b",
        "representation_id": "qwen3-embedding-8b-unit",
        "distance_id": "energy_v",
    }


@pytest.mark.parametrize(
    ("provider_id", "representation_id"),
    [
        ("qwen3-embedding-8b", "qwen3-embedding-8b-unit"),
        ("bge-large-en-v1.5", "bge-large-en-v1.5-unit"),
    ],
)
def test_alternative_provider_identity_is_forwarded_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    representation_id: str,
) -> None:
    """Qwen8 and BGE arms are accepted when callers state full identity."""
    captured: dict[str, object] = {}
    frame = pd.DataFrame({"item_i": ["A"], "item_j": ["A"], "value": [0.0]})

    def fake_reader(*_args: object, **kwargs: object) -> tuple[object, object]:
        captured.update(cast("dict[str, object]", kwargs["expected_identity"]))
        return frame, {"item_ids": ["A"]}

    monkeypatch.setattr(energy_shared, "read_typed_distance_artifact", fake_reader)
    pairs = load_energy_functional_pairs(
        tmp_path / "opaque-location",
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id="energy_v",
    )

    assert captured == {
        "provider_id": provider_id,
        "representation_id": representation_id,
        "distance_id": "energy_v",
    }
    assert pairs.to_dict("records") == [
        {"symbol1": "A", "symbol2": "A", "energy_distance": 0.0}
    ]


def test_pair_adapter_renames_only_at_consumer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typed columns acquire paper-local names only after validated loading."""
    frame = pd.DataFrame({"item_i": ["A"], "item_j": ["B"], "value": [0.25]})
    monkeypatch.setattr(
        energy_shared,
        "read_typed_distance_artifact",
        lambda *_args, **_kwargs: (frame, {"item_ids": ["A", "B"]}),
    )

    pairs = load_energy_functional_pairs(
        tmp_path / "typed-distance",
        provider_id="bge-large-en-v1.5",
        representation_id="bge-large-en-v1.5-unit",
        distance_id="energy_v",
    )

    assert pairs.to_dict("records") == [
        {"symbol1": "A", "symbol2": "B", "energy_distance": 0.25}
    ]


def test_materially_negative_functional_is_rejected() -> None:
    """Negative functionals outside numerical tolerance are rejected."""
    with pytest.raises(ValueError, match="materially negative"):
        _energy_metric_from_functional(np.array([0.0, -0.01]))

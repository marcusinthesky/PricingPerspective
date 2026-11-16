"""Scoped parameter provenance for Paper 3's representation cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml
from pipeline.stages.papers.paper3 import representation_ablation as stage

if TYPE_CHECKING:
    import pytest

_ROOT = Path(__file__).resolve().parents[7]


def test_parameter_identity_excludes_unrelated_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell records its own roster and bootstrap settings without whole-file drift."""
    params = yaml.safe_load((_ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["paper3"]["certificate_validation"]["return_block_replications"] = 2
    params_file = tmp_path / "params.yaml"
    returns_dir = tmp_path / "returns"
    distance_dir = tmp_path / "distances"
    returns_dir.mkdir()
    distance_dir.mkdir()
    monkeypatch.setattr(
        stage,
        "load_priced_distance_universe",
        lambda *_args: (["AAA", "BBB"], np.eye(2), np.eye(2)),
    )
    monkeypatch.setattr(
        stage,
        "load_aligned_return_panel",
        lambda *_args: (
            np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
            np.ones((3, 2)),
        ),
    )
    monkeypatch.setattr(
        stage,
        "_news_only_benchmark",
        lambda *_args, **_kwargs: (pd.DataFrame({"weight": [0.5, 0.5]}), {}),
    )
    monkeypatch.setattr(stage, "optimal_block_length_numpy", lambda _values: 1.0)
    monkeypatch.setattr(
        stage,
        "_return_bootstrap_draws",
        lambda _returns, _weights, indices: (np.ones(len(indices)), len(indices)),
    )

    def run(name: str) -> dict[str, object]:
        params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
        output_dir = tmp_path / name
        stage.run_paper3_representation_ablation(
            stage.Paper3RepresentationAblationPaths(
                params_file=params_file,
                returns_dir=returns_dir,
                distance_artifact_dir=distance_dir,
                output_dir=output_dir,
            ),
            "qwen8b_full",
        )
        provenance = json.loads(
            (output_dir / "provenance.manifest.json").read_text(encoding="utf-8")
        )
        assert "params_file" not in provenance["inputs"]
        return provenance["parameter_identity"]

    baseline = run("baseline")
    params["unused_parameter"] = {"anything": 99}
    assert run("unrelated") == baseline
    params["paper3"]["certificate_validation"]["seed"] += 1
    assert run("used") != baseline

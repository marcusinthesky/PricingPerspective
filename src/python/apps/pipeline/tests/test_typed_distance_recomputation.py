"""Changed scientific identities must recompute an existing distance artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pipeline._kernels.typed_geometry import make_cloud
from pipeline.io.typed_analysis import load_typed_analysis
from pipeline.stages.shared import typed_artifacts

PARAMS_FILE = Path(__file__).resolve().parents[5] / "params.yaml"
PROVIDER_ID = "qwen3-embedding-8b"
REPRESENTATION_ID = f"{PROVIDER_ID}-unit"


@pytest.mark.integration
@pytest.mark.parametrize("change", ["parameter", "code"])
def test_existing_distance_is_recomputed_after_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """A valid old artifact cannot hide changed parameters or semantic code."""
    representation = load_typed_analysis(PARAMS_FILE).representations[REPRESENTATION_ID]
    clouds = tuple(
        make_cloud(str(index), values, PROVIDER_ID, representation)
        for index, values in enumerate(
            (np.eye(2), np.asarray([[1.0, 0.0], [1.0, 1.0]]))
        )
    )
    monkeypatch.setattr(
        typed_artifacts, "load_provider_clouds", lambda *_args, **_kwargs: clouds
    )
    monkeypatch.setattr(
        typed_artifacts, "entrypoint_code_identity", lambda _command: "a" * 64
    )
    artifact_dir = tmp_path / "distance"

    def compute(params_file: Path) -> dict[str, object]:
        return typed_artifacts.run_typed_distance(
            provider_id=PROVIDER_ID,
            representation_id=REPRESENTATION_ID,
            distance_id="energy_v",
            output_dir=artifact_dir,
            embeddings_dir=tmp_path / "embeddings",
            params_file=params_file,
        )

    compute(PARAMS_FILE)
    original = json.loads((artifact_dir / "provenance.manifest.json").read_text())
    params_file = tmp_path / "params.yaml"
    params = yaml.safe_load(PARAMS_FILE.read_text())
    if change == "parameter":
        params["typed_analysis"]["distances"]["energy_v"]["exponent"] = 0.75
    else:
        monkeypatch.setattr(
            typed_artifacts, "entrypoint_code_identity", lambda _command: "c" * 64
        )
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
    compute(params_file)
    updated = json.loads((artifact_dir / "provenance.manifest.json").read_text())
    identity = "parameter_identity" if change == "parameter" else "code_identity"
    assert updated[identity] != original[identity]
    typed_artifacts.read_typed_distance_artifact(
        artifact_dir,
        expected_identity={"provider_id": PROVIDER_ID},
    )

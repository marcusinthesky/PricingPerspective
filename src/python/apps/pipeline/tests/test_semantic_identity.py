"""Regression checks for semantic numerical identities and their invalidation scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pipeline._kernels.typed_geometry import make_cloud
from pipeline.io.semantic_identity import (
    entrypoint_code_identity,
    standalone_entrypoint_code_identity,
)
from pipeline.io.typed_analysis import load_typed_analysis
from pipeline.stages.shared import typed_artifacts, typed_geometry_components
from pydantic import ValidationError
from semflow.analysis import ModuleIndex, fingerprint_entrypoint
from semflow.config import load_config
from semflow.discovery import discover_command

REPO_ROOT = Path(__file__).resolve().parents[5]
SHARED_COMMANDS = ("typed-geometry-component", "typed-distance", "typed-barycentre")
COMMANDS = SHARED_COMMANDS


@pytest.mark.integration
def test_numerical_artifacts_publish_their_own_entrypoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tiny real computations publish the same code identity as their DVC dependency."""
    monkeypatch.chdir(REPO_ROOT)
    provider_id = "qwen3-embedding-8b"
    representation_id = "qwen3-embedding-8b-unit"
    representation = load_typed_analysis(REPO_ROOT / "params.yaml").representations[
        representation_id
    ]
    clouds = tuple(
        make_cloud(
            str(index),
            np.asarray(values),
            provider_id,
            representation,
            source_hash=str(index) * 64,
        )
        for index, values in enumerate(
            (
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            )
        )
    )
    monkeypatch.setattr(
        typed_geometry_components,
        "load_provider_clouds",
        lambda *_args, **_kwargs: clouds,
    )
    component_dir = tmp_path / "component"
    distance_dir = tmp_path / "distance"
    barycentre_dir = tmp_path / "barycentre"
    typed_geometry_components.run_typed_geometry_component(
        provider_id=provider_id,
        representation_id=representation_id,
        geometry_id="energy_v",
        embeddings_dir=tmp_path / "embeddings",
        output_dir=component_dir,
        params_file=REPO_ROOT / "params.yaml",
    )
    monkeypatch.setattr(
        typed_artifacts,
        "load_provider_clouds",
        lambda *_args, **_kwargs: pytest.fail(
            "A restored geometry component must not read provider embeddings"
        ),
    )
    typed_artifacts.run_typed_distance(
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id="energy_v",
        output_dir=distance_dir,
        component_dir=component_dir,
        params_file=REPO_ROOT / "params.yaml",
    )
    monkeypatch.setattr(
        typed_artifacts, "load_provider_clouds", lambda *_args, **_kwargs: clouds
    )
    typed_artifacts.run_typed_barycentre(
        provider_id=provider_id,
        representation_id=representation_id,
        arm_id="energy_simplex",
        embeddings_dir=tmp_path / "embeddings",
        output_dir=barycentre_dir,
        geometry_component_dir=component_dir,
        params_file=REPO_ROOT / "params.yaml",
    )
    for command, directory in zip(
        SHARED_COMMANDS, (component_dir, distance_dir, barycentre_dir), strict=True
    ):
        manifest = json.loads((directory / "provenance.manifest.json").read_text())
        assert manifest["code_identity"] == {
            "kind": "sha256",
            "value": entrypoint_code_identity(command),
        }


def test_identity_uses_the_declared_entrypoint_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact identities match DVC's digest, independently of source-file bytes."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".semflow/hashes/entrypoints/pipeline/typed-distance.json"
    path.parent.mkdir(parents=True)
    record = {
        "algorithm": "sha256",
        "entrypoint": "pipeline typed-distance",
        "digest": "sha256:" + "a" * 64,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    assert entrypoint_code_identity("typed-distance") == "a" * 64

    record["digest"] = "sha256:" + "b" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    assert entrypoint_code_identity("typed-distance") == "b" * 64

    record["entrypoint"] = "pipeline typed-barycentre"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="entrypoint mismatch"):
        entrypoint_code_identity("typed-distance")

    record["digest"] = "invalid"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValidationError):
        entrypoint_code_identity("typed-distance")

    path.unlink()
    with pytest.raises(FileNotFoundError):
        entrypoint_code_identity("typed-distance")


@pytest.mark.integration
def test_standalone_identity_matches_the_dvc_semantic_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct callback analysis and generated DVC metadata use identical digests."""
    monkeypatch.chdir(REPO_ROOT)
    assert standalone_entrypoint_code_identity("typed-distance") == (
        entrypoint_code_identity("typed-distance")
    )


@pytest.mark.integration
def test_legacy_standalone_identity_does_not_need_a_generated_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retained runner remains usable after its DVC producer is retired."""
    monkeypatch.chdir(REPO_ROOT)
    command = "paper6-information-recovery"
    with pytest.raises(FileNotFoundError):
        entrypoint_code_identity(command)
    identity = standalone_entrypoint_code_identity(command)
    assert len(bytes.fromhex(identity)) == hashlib.sha256().digest_size


@pytest.mark.integration
@pytest.mark.slow
def test_real_entrypoints_exclude_unused_helpers_and_preserve_runtime_schema(
    tmp_path: Path,
) -> None:
    """Prose and unused helpers stay clean; a used helper or schema edit invalidates."""
    config = load_config(REPO_ROOT / "semflow.toml")
    roots = tuple(
        tmp_path / root.relative_to(REPO_ROOT) for root in config.source_roots
    )
    for original, copied in zip(config.source_roots, roots, strict=True):
        for path in original.rglob("*.py"):
            target = copied / path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    targets = {
        command: discover_command("pipeline", (command,), overrides={})
        for command in COMMANDS
    }

    def fingerprints() -> dict[str, str]:
        """Analyze copied source with the repository's actual Semflow policy."""
        index = ModuleIndex(project_root=tmp_path, source_roots=roots)
        return {
            command: fingerprint_entrypoint(
                target, index=index, policy=config.policy
            ).digest
            for command, target in targets.items()
        }

    baseline = fingerprints()
    registry = roots[0] / "pipeline/io/typed_analysis.py"
    source = registry.read_text(encoding="utf-8")
    registry.write_text(
        source.replace(
            "Return an iterative solver's budget; closed-form arms have none.",
            "Revised explanation of the declared solver budget.",
        )
        + '\n\ndef unused_editorial_helper() -> str:\n    return "prose"\n',
        encoding="utf-8",
    )
    identity = roots[0] / "pipeline/io/semantic_identity.py"
    identity.write_text(
        identity.read_text(encoding="utf-8").replace(
            "def entrypoint_code_identity(command: str) -> str:",
            "def entrypoint_code_identity(command: object) -> object:",
        )
        + "\n# Manuscript wording and ordinary function annotations are nonsemantic.\n",
        encoding="utf-8",
    )
    chapters = tmp_path / "src/latex/projects/06_information_motion/src/chapters"
    chapters.mkdir(parents=True)
    (chapters / "introduction.tex").write_text("Revised manuscript prose.\n")
    assert fingerprints() == baseline

    artifacts = roots[0] / "pipeline/stages/shared/typed_artifacts.py"
    artifacts.write_text(
        artifacts.read_text(encoding="utf-8").replace(
            '"bytes": len(content),', '"bytes": len(content) + 1,'
        ),
        encoding="utf-8",
    )
    used_helper_change = fingerprints()
    assert (
        used_helper_change["typed-geometry-component"]
        == baseline["typed-geometry-component"]
    )
    assert used_helper_change["typed-barycentre"] == baseline["typed-barycentre"]
    assert used_helper_change["typed-distance"] != baseline["typed-distance"]

    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            "maxiter: PositiveInt | None = None", "maxiter: int | None = None"
        ),
        encoding="utf-8",
    )
    schema_change = fingerprints()
    for command in COMMANDS:
        assert schema_change[command] != used_helper_change[command]

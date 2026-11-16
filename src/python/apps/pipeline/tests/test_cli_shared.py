"""Typer boundary for the shared typed-artifact commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline.cli.commands.shared import commands
from pipeline.stages.shared import typed_artifacts, typed_geometry_components
from typer.testing import CliRunner

# Keep error text plain even when GitHub Actions forces Rich terminal styling.
runner = CliRunner(env={"TERM": "dumb"})
CLI_USAGE_ERROR = 2
REPO_ROOT = Path(__file__).resolve().parents[5]


def _cell(tmp_path: Path) -> list[str]:
    return [
        "typed-distance",
        "--provider-id",
        "qwen3-embedding-8b",
        "--representation-id",
        "qwen3-embedding-8b-unit",
        "--distance-id",
        "energy_v",
        "--embeddings-dir",
        str(tmp_path / "embeddings"),
        "--output-dir",
        str(tmp_path / "out"),
        "--params-file",
        str(tmp_path / "params.yaml"),
    ]


def test_typed_distance_passes_the_component_dir_to_the_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An energy cell derives from its cached component, named explicitly."""
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        typed_artifacts, "run_typed_distance", lambda **kwargs: observed.append(kwargs)
    )
    component = tmp_path / "component"
    cell = _cell(tmp_path)
    result = runner.invoke(
        commands,
        [*cell[:7], *cell[9:], "--geometry-component-dir", str(component)],
    )
    assert result.exit_code == 0, result.output
    assert observed == [
        {
            "provider_id": "qwen3-embedding-8b",
            "representation_id": "qwen3-embedding-8b-unit",
            "distance_id": "energy_v",
            "embeddings_dir": None,
            "output_dir": tmp_path / "out",
            "component_dir": component,
            "params_file": tmp_path / "params.yaml",
        }
    ]


def test_typed_distance_requires_params_path_and_has_no_roster_mode(
    tmp_path: Path,
) -> None:
    """dvc.yaml passes every path; the provider roster mode is gone."""
    cell = _cell(tmp_path)
    missing_params = runner.invoke(commands, cell[:-2])
    assert missing_params.exit_code == CLI_USAGE_ERROR
    roster = runner.invoke(commands, [*cell, "--roster-key", "qwen3-embedding-8b"])
    assert roster.exit_code == CLI_USAGE_ERROR
    assert "No such option" in roster.output


@pytest.mark.parametrize(
    ("distance_id", "component_options"),
    [("energy_v", []), ("wasserstein_w2", ["--geometry-component-dir", "unused"])],
)
def test_typed_distance_requires_embeddings_for_its_direct_branch(
    distance_id: str,
    component_options: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A component argument cannot replace raw inputs for Wasserstein geometry."""
    monkeypatch.setattr(
        typed_artifacts, "entrypoint_code_identity", lambda _cmd: "a" * 64
    )
    cell = _cell(tmp_path)
    cell[6] = distance_id
    cell[-1] = str(REPO_ROOT / "params.yaml")
    result = runner.invoke(commands, [*cell[:7], *cell[9:], *component_options])

    assert result.exit_code == CLI_USAGE_ERROR, result.output
    assert "requires --embeddings-dir" in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("command", "identity_option", "identity"),
    [
        ("typed-geometry-component", "--geometry-id", "energy_v"),
        ("typed-barycentre", "--arm-id", "energy_simplex"),
    ],
)
def test_typed_artifact_commands_require_and_forward_embeddings_dir(
    command: str,
    identity_option: str,
    identity: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every producer receives the runtime source path from its caller."""
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        typed_geometry_components,
        "run_typed_geometry_component",
        lambda **kwargs: observed.append(kwargs),
    )
    monkeypatch.setattr(
        typed_artifacts,
        "run_typed_barycentre",
        lambda **kwargs: observed.append(kwargs),
    )
    arguments = [
        command,
        "--provider-id",
        "qwen3-embedding-8b",
        "--representation-id",
        "qwen3-embedding-8b-unit",
        identity_option,
        identity,
        "--output-dir",
        str(tmp_path / "out"),
        "--params-file",
        str(tmp_path / "params.yaml"),
    ]

    missing = runner.invoke(commands, arguments)
    assert missing.exit_code == CLI_USAGE_ERROR
    assert "--embeddings-dir" in missing.output
    assert not observed

    embeddings_dir = tmp_path / "embeddings"
    result = runner.invoke(
        commands, [*arguments, "--embeddings-dir", str(embeddings_dir)]
    )
    assert result.exit_code == 0, result.output
    assert observed[0]["embeddings_dir"] == embeddings_dir

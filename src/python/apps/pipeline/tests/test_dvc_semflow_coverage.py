"""Every Python DVC stage declares its complete local source dependencies.

Numerical stages use semantic entrypoint fingerprints. The two input-boundary
control stages directly hash all retained local source trees and the locked
workspace environment. Neither form permits an untracked source edit.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

import pytest
import yaml

FINGERPRINT_PREFIX = ".semflow/hashes/entrypoints/"
# dvc.yaml `vars:` entries that expand to an installed console script. A stage
# whose cmd starts with one of these runs first-party Python and must therefore
# declare a fingerprint. `py_semflow` is deliberately absent: semflow_refresh is
# the control stage that regenerates the fingerprints and cannot depend on them.
APPLICATION_TOKENS = ("${py_pipeline}",)

BYTE_SOURCE_STAGES = frozenset({"co_mentions", "public_inputs_check"})
BYTE_SOURCE_DEPS = frozenset(
    [
        "src/python/apps/pipeline/src",
        "src/python/apps/hotelling/src",
        "src/python/packages/dotell/src",
        "src/python/pyproject.toml",
        "src/python/apps/pipeline/pyproject.toml",
        "src/python/apps/hotelling/pyproject.toml",
        "src/python/packages/dotell/pyproject.toml",
        "src/python/uv.lock",
    ]
)


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a directory containing `dvc.yaml` is found."""
    for candidate in (start, *start.parents):
        if (candidate / "dvc.yaml").is_file():
            return candidate
    message = f"no dvc.yaml found walking up from {start}"
    raise FileNotFoundError(message)


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
DVC_YAML_PATH = REPO_ROOT / "dvc.yaml"


def _uninstrumented_stages(dvc_config: dict[str, object]) -> list[str]:
    """Return Python stages without a fingerprint or complete direct source coverage.

    `foreach`/`matrix` families keep their body under `do:`; a plain stage is its
    own body.
    """
    stages = cast("dict[str, dict[str, object]]", dvc_config["stages"])
    missing: list[str] = []
    for name, stage in stages.items():
        body = cast("dict[str, object]", stage.get("do", stage))
        command = str(body.get("cmd", ""))
        if not any(token in command for token in APPLICATION_TOKENS):
            continue
        declared = cast("list[object]", body.get("deps") or [])
        dependencies = [str(item) for item in declared]
        has_fingerprint = any(
            item.startswith(FINGERPRINT_PREFIX) for item in dependencies
        )
        has_complete_bytes = name in BYTE_SOURCE_STAGES and BYTE_SOURCE_DEPS.issubset(
            dependencies
        )
        if not has_fingerprint and not has_complete_bytes:
            missing.append(name)
    return missing


def test_every_python_stage_declares_complete_source_dependencies() -> None:
    """No stage runs first-party Python without a source-change dependency."""
    with DVC_YAML_PATH.open() as handle:
        dvc_config = yaml.safe_load(handle)

    assert _uninstrumented_stages(dvc_config) == []


def test_declared_fingerprints_exist_on_disk() -> None:
    """A declared fingerprint path resolves, so no stage depends on a dead JSON."""
    with DVC_YAML_PATH.open() as handle:
        dvc_config = yaml.safe_load(handle)
    stages = cast("dict[str, dict[str, object]]", dvc_config["stages"])

    dangling: list[str] = []
    for name, stage in stages.items():
        body = cast("dict[str, object]", stage.get("do", stage))
        for item in cast("list[object]", body.get("deps") or []):
            path = str(item)
            if path.startswith(FINGERPRINT_PREFIX) and not (REPO_ROOT / path).is_file():
                dangling.append(f"{name} -> {path}")

    assert dangling == []


def test_guard_raises_when_a_python_stage_loses_its_fingerprint() -> None:
    """The guard logic actually guards: it must flag an uninstrumented stage."""
    snippet = yaml.safe_load(
        """
        stages:
          instrumented:
            cmd: ${py_pipeline} do-something
            deps:
            - .semflow/hashes/entrypoints/pipeline/do-something.json
            - data/shared/returns
          uninstrumented:
            cmd: ${py_pipeline} do-something-else
            deps:
            - data/shared/returns
          not_python:
            cmd: bash scripts/thing.sh
            deps:
            - data/shared/returns
        """
    )

    assert _uninstrumented_stages(snippet) == ["uninstrumented"]


@pytest.mark.parametrize("token", APPLICATION_TOKENS)
def test_every_application_token_is_used_by_dvc_yaml(token: str) -> None:
    """Keep the token roster honest: a stale token would silently skip stages."""
    assert token in DVC_YAML_PATH.read_text(encoding="utf-8")


def test_only_complete_source_control_stages_opt_out_of_semantic_hashes() -> None:
    """The opt-out set stays limited to the generator and two input controls."""
    configuration = tomllib.loads((REPO_ROOT / "semflow.toml").read_text())
    assert set(configuration["tracking"]["control-stages"]) == {
        "semflow_refresh",
        *BYTE_SOURCE_STAGES,
    }


@pytest.mark.parametrize("stage_name", sorted(BYTE_SOURCE_STAGES))
@pytest.mark.parametrize("dependency", sorted(BYTE_SOURCE_DEPS))
def test_control_guard_rejects_each_missing_source_dependency(
    stage_name: str, dependency: str
) -> None:
    """Every local source tree, manifest and lock is required independently."""
    snippet: dict[str, object] = {
        "stages": {
            stage_name: {
                "cmd": "${py_pipeline} control",
                "deps": sorted(BYTE_SOURCE_DEPS - {dependency}),
            }
        }
    }
    assert _uninstrumented_stages(snippet) == [stage_name]


@pytest.mark.parametrize("stage_name", sorted(BYTE_SOURCE_STAGES))
def test_control_guard_accepts_complete_direct_source_coverage(stage_name: str) -> None:
    """Complete byte coverage is a valid conservative source-change boundary."""
    snippet: dict[str, object] = {
        "stages": {
            stage_name: {
                "cmd": "${py_pipeline} control",
                "deps": sorted(BYTE_SOURCE_DEPS),
            }
        }
    }
    assert _uninstrumented_stages(snippet) == []

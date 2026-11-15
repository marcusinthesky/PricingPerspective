"""Every Python-command DVC stage carries a semflow entrypoint fingerprint.

`semflow check` only proves that the fingerprints which *exist* are current. It
cannot prove that a stage has one: `semflow.dvc._build_patch` skips any stage
with neither a raw Python dependency nor an entrypoint dependency, so a stage
authored with only data dependencies is silently invisible to instrumentation
and a source edit never invalidates it. `p5_network_certificate`,
`p5_post_corpus_qmle`, `p5_two_field_qmle` and `p5_wflat_gate` all sat in that
blind spot. This test closes it from the other side, making the t47 acceptance
criterion ("parse every Python-command DVC stage and fail if coverage drops")
non-vacuous.

Parses `dvc.yaml` with PyYAML directly rather than shelling out to `dvc`, which
would contend for `.dvc/tmp/rwlock`.

"""

from __future__ import annotations

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
    """Return every Python-command stage that declares no fingerprint dependency.

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
        if not any(item.startswith(FINGERPRINT_PREFIX) for item in dependencies):
            missing.append(name)
    return missing


def test_every_python_stage_declares_a_semflow_fingerprint() -> None:
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

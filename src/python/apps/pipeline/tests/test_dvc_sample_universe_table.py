"""DVC contracts for generated sample-universe tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping

_ROOT = Path(__file__).parents[5]
_STAGE_OUTPUTS = {
    "p1_sample_universe_render": "${p1_tex}/generated/sample_universe.tex",
    "p3_sample_universe_render": "${p3_tex}/generated/sample_universe.tex",
    "p5_sample_universe_render": "${p5_tex}/generated/sample_universe.tex",
}
_SEMFLOW_DEP = ".semflow/hashes/entrypoints/pipeline/render-sample-universe-table.json"


def _stages() -> Mapping[str, object]:
    document = yaml.safe_load((_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    return cast("Mapping[str, object]", document["stages"])


def _paths(values: object) -> set[str]:
    paths: set[str] = set()
    for value in cast("list[object]", values):
        if isinstance(value, dict):
            if "path" in value:
                paths.add(str(value["path"]))
            else:
                paths.update(str(path) for path in value)
        else:
            paths.add(str(value))
    return paths


def test_sample_universe_stages_have_governed_roster_dependencies() -> None:
    """Keep all three generated roster tables attached to their data sources."""
    stages = _stages()
    for stage_name, output in _STAGE_OUTPUTS.items():
        stage = cast("Mapping[str, object]", stages[stage_name])
        command = str(stage["cmd"])
        dependencies = _paths(stage["deps"])
        outputs = _paths(stage["outs"])

        assert "render-sample-universe-table" in command
        assert _SEMFLOW_DEP in dependencies
        assert "data/universe.csv" in dependencies
        assert "${shared}/returns" in dependencies
        assert any("typed_distances" in dependency for dependency in dependencies)
        assert output in outputs


def test_sample_universe_stages_are_locked() -> None:
    """Ensure the checked-in lock records each generated table stage."""
    document = yaml.safe_load((_ROOT / "dvc.lock").read_text(encoding="utf-8"))
    stages = cast("Mapping[str, object]", document["stages"])
    assert set(_STAGE_OUTPUTS).issubset(stages)

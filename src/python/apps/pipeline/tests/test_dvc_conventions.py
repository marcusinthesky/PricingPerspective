"""`dvc.yaml` follows the stage, path, cache and params conventions in ARCHITECTURE.md.

A static parse (no `dvc` subprocess, so no `.dvc/tmp/rwlock` contention): each
rule is checked over every stage, so a new stage that breaks one fails here.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "dvc.yaml").is_file()
)
CONFIG = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
STAGES: dict[str, dict[str, Any]] = {
    name: stage.get("do", stage) for name, stage in CONFIG["stages"].items()
}
RAW_STAGES: dict[str, dict[str, Any]] = CONFIG["stages"]

PUBLISHED_INPUT_STAGE = "fixed_inputs"

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

PUBLICATION_ROOTS = {"render"}
CERTIFICATION = {
    "lean_claims",
    "lean_bindings_check",
    "refs_bib_freshness",
    "public_inputs_check",
}
SHARED_COMMANDS = {"compute-sample-descriptives", "render-sample-universe-table"}
PAPER_STAGE = re.compile(r"^p(?P<paper>[1356])_(?P<experiment>[a-z0-9_]+)$")
UNCACHED_PREFIXES = (
    "${p1_tex}/",
    "${p3_tex}/",
    "${p5_tex}/",
    "${p6_tex}/",
    "src/latex/",
    "src/lean/",
    "data/publication/certification/",
    ".semflow/hashes",
)


def _outs(stage: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (out, {}) if isinstance(out, str) else next(iter(out.items()))
        for out in stage.get("outs", [])
    ]


def _deps(stage: dict[str, Any]) -> list[str]:
    return list(stage.get("deps", []))


def _command(stage: dict[str, Any]) -> str | None:
    match = re.search(r"\$\{py_pipeline\}\s+([a-z0-9-]+)", stage["cmd"])
    return match.group(1) if match else None


def _is_publication_path(path: str) -> bool:
    return path.startswith(("${p1_tex}/", "${p3_tex}/", "${p5_tex}/", "${p6_tex}/"))


def test_publication_builds_only_the_four_arxiv_manuscripts() -> None:
    """The public build exposes one arXiv PDF and source archive per paper."""
    builds = RAW_STAGES["render"]["foreach"]
    assert set(builds) == {
        f"{paper}-arxiv"
        for paper in (
            "01_continuous_bounds",
            "03_distance_implied_mpt",
            "05_spatial_pricing",
            "06_information_motion",
        )
    }
    assert all(item["target"] == "arxiv" for item in builds.values())
    assert all(item["archive"] == "tar.gz" for item in builds.values())


def test_public_stages_recompute_from_the_fixed_inputs() -> None:
    """Every retained computation can execute from its declared numeric inputs."""
    frozen = {
        name
        for name, stage in RAW_STAGES.items()
        if stage.get("frozen") or stage.get("do", {}).get("frozen")
    }
    assert frozen == {PUBLISHED_INPUT_STAGE}
    fixed = RAW_STAGES[PUBLISHED_INPUT_STAGE]
    assert fixed["cmd"] == "false"
    assert not fixed.get("deps")
    assert not fixed.get("params")
    assert all(options.get("cache", True) for _, options in _outs(fixed))


@pytest.mark.parametrize(
    ("stage_name", "input_flag"),
    [
        ("sample_descriptives", "--embeddings-dir"),
        ("sample_descriptives", "--distance-artifact-dir"),
        ("p5_transport_dimension_ladder", "--embeddings-dir"),
    ],
)
def test_selector_inputs_are_declared_with_the_same_expression(
    stage_name: str, input_flag: str
) -> None:
    """Changing a selector must change both the command input and its dependency."""
    stage = STAGES[stage_name]
    tokens = shlex.split(stage["cmd"])
    input_path = tokens[tokens.index(input_flag) + 1]
    assert input_path in _deps(stage)


def test_factor_archive_dependencies_match_registered_sources() -> None:
    """Relocating a registered factor archive must also update its DVC inputs."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    registered_paths = {source["path"] for source in params["fama_french_sources"]}
    for name, stage in STAGES.items():
        for dep in _deps(stage):
            if dep.startswith("data/raw/fama_french/"):
                assert dep in registered_paths, (
                    f"{name}: unregistered factor input {dep}"
                )


@pytest.mark.parametrize("name", sorted(STAGES))
def test_stage_name_matches_its_role(name: str) -> None:
    """Match a producer's role and output owner, including directory outputs."""
    stage = STAGES[name]
    outs = [path for path, _ in _outs(stage)]
    if name in PUBLICATION_ROOTS | CERTIFICATION | {
        "semflow_refresh",
        PUBLISHED_INPUT_STAGE,
    }:
        return
    match = PAPER_STAGE.match(name)
    if match is None:
        assert all(
            path.startswith("${shared}/") or "window.root" in path for path in outs
        ), f"{name}: a shared stage writes only under data/shared"
        return
    paper, experiment = match.group("paper"), match.group("experiment")
    publishes = any(_is_publication_path(path) for path in outs)
    if publishes:
        assert experiment.endswith("_render") or experiment in {
            "numbers",
            "illustrations",
        }, f"{name}: use a `_render`, `_numbers` or `_illustrations` stage"
        assert all(_is_publication_path(path) for path in outs), (
            f"{name}: a publication stage writes no data"
        )
        return
    folder = experiment.removesuffix("_cells")
    root = f"${{p{paper}}}/{folder}"
    assert all(path == root or path.startswith(root + "/") for path in outs), (
        f"{name}: outputs live under data/papers/paper{paper}/{folder}/"
    )


@pytest.mark.parametrize("name", sorted(STAGES))
def test_command_matches_stage(name: str) -> None:
    """Keep each paper's public command predictable from its stage name."""
    stage = STAGES[name]
    command = _command(stage)
    match = PAPER_STAGE.match(name)
    if command is None or match is None or command in SHARED_COMMANDS:
        return
    paper, experiment = match.group("paper"), match.group("experiment")
    if experiment.endswith("_cells") and command.startswith(f"paper{paper}-"):
        return
    if experiment.endswith("_render") or experiment in {"numbers", "illustrations"}:
        expected = f"render-paper{paper}-" + experiment.removesuffix("_render").replace(
            "_", "-"
        )
    else:
        expected = f"paper{paper}-" + experiment.replace("_", "-")
    assert command == expected


@pytest.mark.parametrize("name", sorted(STAGES))
def test_uncached_outputs_are_publication_files(name: str) -> None:
    """Keep scientific data available for DVC restoration and run caching."""
    for path, options in _outs(STAGES[name]):
        if options.get("cache") is False:
            assert path.startswith(UNCACHED_PREFIXES), (
                f"{name}: {path} is data and must be cached so `dvc pull` restores it"
            )


@pytest.mark.parametrize("name", sorted(STAGES))
def test_params_are_declared_by_key(name: str) -> None:
    """Avoid invalidating unrelated experiments on arbitrary parameter edits."""
    assert "params.yaml" not in _deps(STAGES[name]), (
        f"{name}: list the params keys it reads instead of the whole file"
    )


@pytest.mark.parametrize("name", sorted(STAGES))
def test_code_deps_are_semflow_fingerprints(name: str) -> None:
    """Use the reachable source closure without whole-file dependency churn."""
    stage = STAGES[name]
    if "${py_" not in stage["cmd"] or name == "semflow_refresh":
        return
    if name in BYTE_SOURCE_STAGES:
        assert BYTE_SOURCE_DEPS.issubset(_deps(stage))
        assert not any(dep.startswith(".semflow/hashes/") for dep in _deps(stage))
        return
    for dep in _deps(stage):
        assert not dep.endswith(("uv.lock", ".py")), (
            f"{name}: depend on the semflow fingerprint, not {dep}"
        )
    command = _command(stage)
    if command is not None:
        assert f".semflow/hashes/entrypoints/pipeline/{command}.json" in _deps(stage)

"""Plan review-first DVC instrumentation with entrypoint semantic hashes."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import yaml

from semflow._yaml import (
    YamlPatchError,
    dependency_line_span,
    render_scalar,
    replace_spans,
    stage_nodes,
)
from semflow.analysis import AnalysisError, ModuleIndex, fingerprint_entrypoint
from semflow.analysis.index import ModuleIndexError
from semflow.discovery import (
    DiscoveryError,
    available_applications,
    discover_command,
    parse_authored_command,
)
from semflow.hashing import (
    HashingError,
    entrypoint_fingerprint_json,
    fingerprint_json,
    fingerprint_source,
)
from semflow.model import FileChange, PatchPlan
from semflow.tracking import tracking_spec

if TYPE_CHECKING:
    from semflow._yaml import DependencyNode, StageNode
    from semflow.config import Config
    from semflow.discovery import CommandTarget
    from semflow.model import EntrypointFingerprint
    from semflow.tracking import TrackSpec

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


class DvcPatchError(ValueError):
    """Raised when a DVC dependency cannot be instrumented unambiguously."""


def build_patch(config: Config) -> PatchPlan:
    """Compute the exact DVC YAML and fingerprint changes without writing them."""
    pipeline_text = config.pipeline_file.read_text(encoding="utf-8")
    try:
        authored_stages = stage_nodes(pipeline_text)
        variables = _pipeline_variables(pipeline_text)
    except (YamlPatchError, yaml.YAMLError) as error:
        raise DvcPatchError(str(error)) from error

    try:
        index = ModuleIndex(
            project_root=config.project_root,
            source_roots=config.source_roots,
        )
    except ModuleIndexError as error:
        raise DvcPatchError(str(error)) from error
    applications = available_applications(config.applications)
    replacements: list[tuple[int, int, str]] = []
    desired_files: dict[Path, str] = {}
    expected_fingerprints: set[Path] = set()
    entrypoint_cache: dict[tuple[str, tuple[str, ...]], EntrypointFingerprint] = {}
    found_python = False

    for stage in authored_stages:
        if stage.name in config.control_stages:
            continue
        raw_dependencies = _raw_python_dependencies(config, stage)
        entrypoint_dependencies = tuple(
            dependency
            for dependency in stage.dependencies
            if _is_entrypoint_dependency(config, dependency.value)
        )
        if not raw_dependencies and not entrypoint_dependencies:
            continue
        found_python = True
        _plan_stage(
            config,
            stage,
            raw_dependencies=raw_dependencies,
            fingerprint_dependencies=entrypoint_dependencies,
            variables=variables,
            applications=applications,
            pipeline_text=pipeline_text,
            index=index,
            cache=entrypoint_cache,
            desired_files=desired_files,
            expected_fingerprints=expected_fingerprints,
            replacements=replacements,
        )

    if not found_python:
        msg = f"No Python dependencies found in {config.pipeline_file}."
        raise DvcPatchError(msg)

    try:
        desired_pipeline = replace_spans(pipeline_text, replacements)
    except YamlPatchError as error:
        raise DvcPatchError(str(error)) from error
    changes = _desired_changes(desired_files)
    changes.extend(_orphan_changes(config, expected_fingerprints))
    if desired_pipeline != pipeline_text:
        changes.append(
            FileChange(
                path=config.pipeline_file,
                before=pipeline_text,
                after=desired_pipeline,
            )
        )
    changes.sort(key=lambda change: change.path.relative_to(config.project_root))
    return PatchPlan(
        root=config.project_root,
        pipeline_file=config.pipeline_file,
        changes=tuple(changes),
    )


def build_refresh(config: Config) -> PatchPlan:
    """Compute a JSON-only refresh after DVC instrumentation is current."""
    plan = build_patch(config)
    pipeline_changes = tuple(
        change for change in plan.changes if change.path == config.pipeline_file
    )
    if pipeline_changes:
        msg = (
            "DVC instrumentation is stale; run `semflow patch --apply` before "
            "running the semflow refresh stage."
        )
        raise DvcPatchError(msg)
    return PatchPlan(
        root=plan.root,
        pipeline_file=plan.pipeline_file,
        changes=tuple(
            change for change in plan.changes if change.path != config.pipeline_file
        ),
    )


def apply_patch(plan: PatchPlan) -> None:
    """Apply a previously reviewed plan using atomic per-file replacements."""
    _apply_plan(plan, allow_pipeline=True)


def apply_refresh(plan: PatchPlan) -> None:
    """Apply a JSON-only plan without changing the authored DVC YAML."""
    if any(change.path == plan.pipeline_file for change in plan.changes):
        msg = "A refresh plan must not contain a dvc.yaml change."
        raise DvcPatchError(msg)
    _apply_plan(plan, allow_pipeline=False)


def _apply_plan(plan: PatchPlan, *, allow_pipeline: bool) -> None:
    """Validate and atomically apply one semflow plan."""
    for change in plan.changes:
        if not change.path.resolve().is_relative_to(plan.root.resolve()):
            msg = f"Refusing to write outside patch root: {change.path}"
            raise DvcPatchError(msg)
        if not allow_pipeline and change.path == plan.pipeline_file:
            msg = "A JSON-only semflow plan cannot write dvc.yaml."
            raise DvcPatchError(msg)
        if _read_optional(change.path) != change.before:
            relative = change.path.relative_to(plan.root)
            msg = f"Patch is stale because {relative} changed after planning."
            raise DvcPatchError(msg)

    writes = [change for change in plan.changes if change.after is not None]
    writes.sort(key=lambda change: change.path == plan.pipeline_file)
    for change in writes:
        if change.after is not None:
            _atomic_write(change.path, change.after)

    for change in plan.changes:
        if change.after is None:
            change.path.unlink(missing_ok=True)
    _remove_empty_directories(plan)


def _plan_stage(
    config: Config,
    stage: StageNode,
    *,
    raw_dependencies: tuple[tuple[DependencyNode, Path], ...],
    fingerprint_dependencies: tuple[DependencyNode, ...],
    variables: dict[str, str],
    applications: frozenset[str],
    pipeline_text: str,
    index: ModuleIndex,
    cache: dict[tuple[str, tuple[str, ...]], EntrypointFingerprint],
    desired_files: dict[Path, str],
    expected_fingerprints: set[Path],
    replacements: list[tuple[int, int, str]],
) -> None:
    try:
        parsed = parse_authored_command(
            stage.command,
            variables=variables,
            applications=applications,
        )
    except DiscoveryError as error:
        message = f"Stage {stage.name}: {error}"
        raise DvcPatchError(message) from error
    if parsed is None:
        _plan_legacy_stage(
            config,
            raw_dependencies,
            desired_files=desired_files,
            expected_fingerprints=expected_fingerprints,
            replacements=replacements,
        )
        return
    try:
        target = discover_command(
            parsed[0],
            parsed[1],
            overrides=config.applications,
        )
    except DiscoveryError as error:
        message = f"Stage {stage.name}: {error}"
        raise DvcPatchError(message) from error
    spec = tracking_spec(target.callback)
    if spec is None and not config.auto_track_typer:
        _plan_legacy_stage(
            config,
            raw_dependencies,
            desired_files=desired_files,
            expected_fingerprints=expected_fingerprints,
            replacements=replacements,
        )
        return
    _plan_entrypoint_stage(
        config,
        stage,
        target,
        raw_dependencies=raw_dependencies,
        fingerprint_dependencies=fingerprint_dependencies,
        pipeline_text=pipeline_text,
        index=index,
        track_spec=spec,
        cache=cache,
        desired_files=desired_files,
        expected_fingerprints=expected_fingerprints,
        replacements=replacements,
    )


def _plan_entrypoint_stage(
    config: Config,
    stage: StageNode,
    target: CommandTarget,
    *,
    raw_dependencies: tuple[tuple[DependencyNode, Path], ...],
    fingerprint_dependencies: tuple[DependencyNode, ...],
    pipeline_text: str,
    index: ModuleIndex,
    track_spec: TrackSpec | None,
    cache: dict[tuple[str, tuple[str, ...]], EntrypointFingerprint],
    desired_files: dict[Path, str],
    expected_fingerprints: set[Path],
    replacements: list[tuple[int, int, str]],
) -> None:
    cache_key = (target.application, target.command)
    fingerprint = cache.get(cache_key)
    if fingerprint is None:
        try:
            fingerprint = fingerprint_entrypoint(
                target,
                index=index,
                policy=config.policy,
                track_spec=track_spec,
            )
        except (AnalysisError, HashingError, OSError, UnicodeError) as error:
            message = f"Stage {stage.name}: {error}"
            raise DvcPatchError(message) from error
        cache[cache_key] = fingerprint

    fingerprint_path = _entrypoint_path(config, target)
    desired = entrypoint_fingerprint_json(fingerprint)
    previous = desired_files.setdefault(fingerprint_path, desired)
    if previous != desired:
        msg = f"Conflicting fingerprints planned for {fingerprint_path}."
        raise DvcPatchError(msg)
    expected_fingerprints.add(fingerprint_path)

    candidates = (*fingerprint_dependencies, *(item[0] for item in raw_dependencies))
    if not candidates:
        msg = f"Stage {stage.name} has no replaceable Python dependency scalar."
        raise DvcPatchError(msg)
    keep = candidates[0]
    replacement = config.repository_path(fingerprint_path)
    if keep.value != replacement:
        replacements.append(
            (keep.start, keep.end, render_scalar(replacement, keep.style))
        )
    for dependency in candidates[1:]:
        start, end = dependency_line_span(
            pipeline_text,
            dependency,
            include_leading_comments=True,
        )
        replacements.append((start, end, ""))


def _plan_legacy_stage(
    config: Config,
    dependencies: tuple[tuple[DependencyNode, Path], ...],
    *,
    desired_files: dict[Path, str],
    expected_fingerprints: set[Path],
    replacements: list[tuple[int, int, str]],
) -> None:
    for dependency, source in dependencies:
        fingerprint_path = _source_fingerprint_path(config, source)
        expected_fingerprints.add(fingerprint_path)
        try:
            desired = fingerprint_json(
                fingerprint_source(
                    source,
                    project_root=config.project_root,
                    policy=config.policy,
                )
            )
        except (HashingError, OSError, UnicodeError) as error:
            msg = f"Could not fingerprint {source}: {error}"
            raise DvcPatchError(msg) from error
        desired_files.setdefault(fingerprint_path, desired)
        replacement = config.repository_path(fingerprint_path)
        replacements.append(
            (
                dependency.start,
                dependency.end,
                render_scalar(replacement, dependency.style),
            )
        )


def _raw_python_dependencies(
    config: Config,
    stage: StageNode,
) -> tuple[tuple[DependencyNode, Path], ...]:
    found: list[tuple[DependencyNode, Path]] = []
    for dependency in stage.dependencies:
        source = _source_dependency(config, dependency.value)
        if source is not None:
            found.append((dependency, source))
    return tuple(found)


def _source_dependency(config: Config, value: str) -> Path | None:
    if "${" in value or value.startswith(config.repository_path(config.hash_dir)):
        return None
    path = _repository_file(config, value)
    if not _inside_source_roots(config, path):
        return None
    if path.is_file() and path.suffix == ".py":
        return path
    return path if path.is_dir() else None


def _is_entrypoint_dependency(config: Config, value: str) -> bool:
    if "${" in value:
        return False
    path = _repository_file(config, value)
    if path.suffix != ".json" or not path.is_relative_to(config.hash_dir):
        return False
    relative = path.relative_to(config.hash_dir)
    return bool(relative.parts) and relative.parts[0] == "entrypoints"


def _entrypoint_path(config: Config, target: CommandTarget) -> Path:
    components = (target.application, *target.command)
    if not all(_SAFE_COMPONENT.fullmatch(component) for component in components):
        msg = f"Unsafe entrypoint path components: {components!r}"
        raise DvcPatchError(msg)
    parent = config.hash_dir.joinpath("entrypoints", *components[:-1])
    return parent / f"{components[-1]}.json"


def _source_fingerprint_path(config: Config, source: Path) -> Path:
    relative = config.repository_path(source)
    return config.hash_dir / f"{relative}.json"


def _pipeline_variables(source: str) -> dict[str, str]:
    document = yaml.safe_load(source)
    if not isinstance(document, dict):
        return {}
    authored = document.get("vars", [])
    if not isinstance(authored, list):
        return {}
    variables: dict[str, str] = {}
    for item in authored:
        if not isinstance(item, dict):
            continue
        variables.update(
            (str(name), str(value))
            for name, value in item.items()
            if isinstance(value, (str, int, float, bool))
        )
    return variables


def _desired_changes(desired_files: dict[Path, str]) -> list[FileChange]:
    return [
        FileChange(path=path, before=current, after=desired)
        for path, desired in desired_files.items()
        if (current := _read_optional(path)) != desired
    ]


def _orphan_changes(
    config: Config, expected_fingerprints: set[Path]
) -> list[FileChange]:
    if not config.hash_dir.is_dir():
        return []
    return [
        FileChange(path=orphan, before=orphan.read_text(encoding="utf-8"), after=None)
        for orphan in sorted(config.hash_dir.rglob("*.json"))
        if orphan not in expected_fingerprints
    ]


def _inside_source_roots(config: Config, path: Path) -> bool:
    return any(
        path == root or path.is_relative_to(root) for root in config.source_roots
    )


def _repository_file(config: Config, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        msg = f"DVC dependency must stay within project-root: {value}"
        raise DvcPatchError(msg)
    resolved = (config.project_root / Path(*pure.parts)).resolve()
    if not resolved.is_relative_to(config.project_root):
        msg = f"DVC dependency resolves outside project-root: {value}"
        raise DvcPatchError(msg)
    return resolved


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        _write_temporary(descriptor, Path(temporary), content, mode)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_temporary(descriptor: int, path: Path, content: str, mode: int) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def _remove_empty_directories(plan: PatchPlan) -> None:
    directories = {
        parent
        for change in plan.changes
        if change.after is None
        for parent in change.path.parents
        if parent != plan.root and parent.is_relative_to(plan.root)
    }
    ordered = sorted(directories, key=lambda path: len(path.parts), reverse=True)
    for directory in ordered:
        with suppress(OSError):
            directory.rmdir()

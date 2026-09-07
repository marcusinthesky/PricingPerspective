"""Load and validate semflow project configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from semflow.model import HashPolicy


class ConfigError(ValueError):
    """Raised when semflow configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Resolved paths and normalization policy for one repository."""

    project_root: Path
    pipeline_file: Path
    hash_dir: Path
    source_roots: tuple[Path, ...]
    policy: HashPolicy
    applications: dict[str, str]
    auto_track_typer: bool
    control_stages: frozenset[str]
    config_file: Path

    def repository_path(self, path: Path) -> str:
        """Return a POSIX path relative to the configured project root."""
        return path.relative_to(self.project_root).as_posix()


def find_config(start: Path | None = None) -> Path:
    """Find the nearest semflow.toml or configured pyproject.toml."""
    directory = (start or Path.cwd()).resolve()
    if directory.is_file():
        directory = directory.parent
    for parent in (directory, *directory.parents):
        standalone = parent / "semflow.toml"
        if standalone.is_file():
            return standalone
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file() and _has_pyproject_config(pyproject):
            return pyproject
    msg = "No semflow.toml or [tool.semflow] pyproject.toml found."
    raise ConfigError(msg)


def load_config(path: Path | None = None) -> Config:
    """Load a standalone semflow.toml or a pyproject [tool.semflow] table."""
    config_file = (path or find_config()).resolve()
    try:
        document = tomllib.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        msg = f"Could not read semflow configuration {config_file}: {error}"
        raise ConfigError(msg) from error

    table = _config_table(document, config_file)
    config_parent = config_file.parent
    root_value = _string(table, "project-root", default=".")
    project_root = (config_parent / root_value).resolve()
    pipeline_file = _inside(project_root, _string(table, "pipeline-file", "dvc.yaml"))
    hash_dir = _inside(project_root, _string(table, "hash-dir", ".semflow/hashes"))

    source_values = table.get("source-roots")
    if not isinstance(source_values, list) or not source_values:
        msg = "semflow source-roots must be a non-empty array of paths."
        raise ConfigError(msg)
    if not all(isinstance(value, str) and value for value in source_values):
        msg = "Every semflow source-roots entry must be a non-empty string."
        raise ConfigError(msg)
    source_roots = tuple(_inside(project_root, value) for value in source_values)

    hashing = table.get("hashing", {})
    if not isinstance(hashing, dict):
        msg = "semflow hashing must be a TOML table."
        raise ConfigError(msg)
    policy = HashPolicy(
        ignore_docstrings=_boolean(hashing, "ignore-docstrings", default=True),
        ignore_annotations=_boolean(hashing, "ignore-annotations", default=False),
        ignore_type_checking_blocks=_boolean(
            hashing, "ignore-type-checking-blocks", default=True
        ),
    )
    applications = _string_table(table, "applications")
    tracking = table.get("tracking", {})
    if not isinstance(tracking, dict):
        msg = "semflow tracking must be a TOML table."
        raise ConfigError(msg)
    auto_track_typer = _boolean(tracking, "auto-typer", default=True)
    control_stages = frozenset(_string_list(tracking, "control-stages"))

    if not project_root.is_dir():
        msg = f"semflow project-root does not exist: {project_root}"
        raise ConfigError(msg)
    if not pipeline_file.is_file():
        msg = f"semflow pipeline-file does not exist: {pipeline_file}"
        raise ConfigError(msg)
    missing_roots = [root for root in source_roots if not root.is_dir()]
    if missing_roots:
        joined = ", ".join(str(root) for root in missing_roots)
        msg = f"semflow source-roots do not exist: {joined}"
        raise ConfigError(msg)

    return Config(
        project_root=project_root,
        pipeline_file=pipeline_file,
        hash_dir=hash_dir,
        source_roots=source_roots,
        policy=policy,
        applications=applications,
        auto_track_typer=auto_track_typer,
        control_stages=control_stages,
        config_file=config_file,
    )


def _has_pyproject_config(path: Path) -> bool:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool = document.get("tool")
    return isinstance(tool, dict) and isinstance(tool.get("semflow"), dict)


def _config_table(document: dict[str, object], path: Path) -> dict[str, object]:
    if path.name == "pyproject.toml":
        tool = document.get("tool")
        if not isinstance(tool, dict) or not isinstance(tool.get("semflow"), dict):
            msg = f"Missing [tool.semflow] in {path}."
            raise ConfigError(msg)
        return tool["semflow"]
    semflow = document.get("semflow")
    if isinstance(semflow, dict):
        return semflow
    return document


def _string(table: dict[str, object], key: str, default: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value:
        msg = f"semflow {key} must be a non-empty string."
        raise ConfigError(msg)
    return value


def _boolean(table: dict[str, object], key: str, *, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        msg = f"semflow hashing.{key} must be a boolean."
        raise ConfigError(msg)
    return value


def _string_table(table: dict[str, object], key: str) -> dict[str, str]:
    value = table.get(key, {})
    if not isinstance(value, dict):
        msg = f"semflow {key} must be a TOML table."
        raise ConfigError(msg)
    if not all(
        isinstance(name, str) and name and isinstance(target, str) and target
        for name, target in value.items()
    ):
        msg = f"Every semflow {key} entry must map non-empty strings."
        raise ConfigError(msg)
    return {str(name): str(target) for name, target in value.items()}


def _string_list(table: dict[str, object], key: str) -> tuple[str, ...]:
    """Read a configured list of non-empty strings."""
    value = table.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        msg = f"Every semflow {key} entry must be a non-empty string."
        raise ConfigError(msg)
    return tuple(value)


def _inside(root: Path, value: str) -> Path:
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        msg = f"Configured path escapes project-root: {value}"
        raise ConfigError(msg)
    return resolved

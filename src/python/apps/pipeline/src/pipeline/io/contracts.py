"""Small, versioned contracts for persisted pipeline artifacts.

This module serializes provenance manifests only. It deliberately does not
read, rewrite, or normalize the bytes of the artifact named by a manifest;
DVC remains the authority for dependency closure and content-addressed output
tracking.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

MANIFEST_SCHEMA_VERSION = "1.0"
SUPPORTED_MANIFEST_VERSIONS = frozenset({MANIFEST_SCHEMA_VERSION})
_SHA256_HEX_LENGTH = 64
_REQUIRED_FIELDS = (
    "schema_version",
    "stage_key",
    "lane",
    "phase",
    "protocol_id",
    "code_identity",
    "parameter_identity",
    "seed_policy",
    "upstream_artifacts",
    "environment",
    "outputs",
    "risk_class",
)


class ContractError(ValueError):
    """Base error for malformed or unsupported artifact contracts."""

    @classmethod
    def invalid_field(cls, field: str, requirement: str) -> ContractError:
        """Build an error describing one invalid manifest field."""
        return cls(f"manifest field {field!r} must be {requirement}")

    @classmethod
    def invalid_output(cls, index: int, detail: str) -> ContractError:
        """Build an error describing one invalid output entry."""
        return cls(f"outputs[{index}]{detail}")

    @classmethod
    def missing_fields(cls, fields: list[str]) -> ContractError:
        """Build an error listing absent required fields."""
        return cls(f"manifest is missing required fields: {', '.join(fields)}")

    @classmethod
    def invalid_json(cls, path: Path, error: json.JSONDecodeError) -> ContractError:
        """Build an error that retains the invalid manifest's location."""
        return cls(f"invalid JSON manifest {path}: {error}")

    @classmethod
    def manifest_not_object(cls) -> ContractError:
        """Build an error for a non-object manifest root."""
        return cls("manifest must be an object")

    @classmethod
    def upstream_not_array(cls) -> ContractError:
        """Build an error for an invalid upstream-artifact collection."""
        return cls("manifest field 'upstream_artifacts' must be an array")

    @classmethod
    def environment_not_object(cls) -> ContractError:
        """Build an error for invalid environment metadata."""
        return cls("manifest field 'environment' must be an object")

    @classmethod
    def outputs_not_array(cls) -> ContractError:
        """Build an error for an invalid output collection."""
        return cls("manifest field 'outputs' must be a non-empty array")

    @classmethod
    def unknown_risk_class(cls) -> ContractError:
        """Build an error for an unsupported risk classification."""
        return cls("manifest field 'risk_class' has an unknown value")


class UnsupportedManifestVersionError(ContractError):
    """Raised when a consumer cannot safely interpret a manifest version."""

    def __init__(self, version: object) -> None:
        """Describe the unsupported version and the accepted versions."""
        supported = sorted(SUPPORTED_MANIFEST_VERSIONS)
        super().__init__(
            f"unsupported manifest schema_version {version!r}; "
            f"supported versions: {supported}"
        )


# Preserve the original public exception name for downstream consumers. The
# Error-suffixed name is canonical because Ruff's N818 rule requires it.
UnsupportedManifestVersion = UnsupportedManifestVersionError


def _check_non_empty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError.invalid_field(field, "a non-empty string")


def _check_identity(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ContractError.invalid_field(field, "an object")
    for required in ("kind", "value"):
        _check_non_empty_string(value.get(required), f"{field}.{required}")


def _validate_outputs(outputs: object) -> None:
    """Validate output identities and their optional byte counts."""
    if not isinstance(outputs, list) or not outputs:
        raise ContractError.outputs_not_array()
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ContractError.invalid_output(index, " must be an object")
        typed_output = cast("dict[str, object]", output)
        for field in ("path", "kind", "sha256"):
            _check_non_empty_string(
                typed_output.get(field), f"outputs[{index}].{field}"
            )
        digest = cast("str", typed_output["sha256"])
        if len(digest) != _SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ContractError.invalid_output(
                index, ".sha256 must be a lowercase SHA-256 hex digest"
            )
        byte_count = typed_output.get("bytes")
        if byte_count is not None and (
            not isinstance(byte_count, int) or byte_count < 0
        ):
            raise ContractError.invalid_output(
                index, ".bytes must be a non-negative integer"
            )


def _validate_nested_fields(result: dict[str, object]) -> None:
    """Validate structured manifest fields after scalar identity checks."""
    upstream = result["upstream_artifacts"]
    if not isinstance(upstream, list):
        raise ContractError.upstream_not_array()
    for index, identity in enumerate(upstream):
        _check_identity(identity, f"upstream_artifacts[{index}]")
    if not isinstance(result["environment"], dict):
        raise ContractError.environment_not_object()
    _validate_outputs(result["outputs"])


def validate_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Validate and return a JSON-compatible copy of a manifest."""
    if not isinstance(manifest, Mapping):
        raise ContractError.manifest_not_object()
    result = dict(manifest)
    version = result.get("schema_version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        raise UnsupportedManifestVersionError(version)
    missing = [field for field in _REQUIRED_FIELDS if field not in result]
    if missing:
        raise ContractError.missing_fields(missing)
    for field in ("stage_key", "lane", "phase", "protocol_id", "seed_policy"):
        _check_non_empty_string(result[field], field)
    _check_identity(result["code_identity"], "code_identity")
    _check_identity(result["parameter_identity"], "parameter_identity")
    _validate_nested_fields(result)
    if result["risk_class"] not in {"low", "moderate", "high", "protected"}:
        raise ContractError.unknown_risk_class()
    return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))


def write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically write a deterministic UTF-8 JSON manifest to *path*.

    Atomic replacement is required when an existing output is a read-only DVC
    hardlink: opening that inode for mutation would either fail or corrupt the
    cache, whereas replacing the directory entry creates a new output inode.
    """
    checked = validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(checked, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_manifest(path: Path) -> dict[str, object]:
    """Read and validate a JSON manifest without touching artifact outputs."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError.invalid_json(path, error) from error
    return validate_manifest(document)

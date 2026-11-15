"""Offline tests for the versioned pipeline artifact manifest contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pipeline.io.contracts import (
    ContractError,
    UnsupportedManifestVersion,
    UnsupportedManifestVersionError,
    read_manifest,
    validate_manifest,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict[str, object]:
    identity = {"kind": "sha256", "value": "a" * 64}
    return {
        "schema_version": "1.0",
        "stage_key": "compute-covariance",
        "lane": "substrate",
        "phase": "prepare",
        "protocol_id": "covariance.v1",
        "code_identity": identity,
        "parameter_identity": {"kind": "sha256", "value": "b" * 64},
        "seed_policy": "not_applicable",
        "upstream_artifacts": [identity],
        "environment": {"python": "3.13"},
        "outputs": [
            {
                "path": "covariance_matrix.parquet",
                "kind": "parquet",
                "sha256": "c" * 64,
                "bytes": 12,
            }
        ],
        "risk_class": "protected",
    }


def test_manifest_round_trip_preserves_values_and_protected_bytes(
    tmp_path: Path,
) -> None:
    """Round trips preserve manifest values and never rewrite artifact bytes."""
    artifact = tmp_path / "protected.parquet"
    artifact_bytes = b"protected artifact bytes"
    artifact.write_bytes(artifact_bytes)
    manifest_path = tmp_path / "protected.manifest.json"
    original = _manifest()

    write_manifest(manifest_path, original)

    assert read_manifest(manifest_path) == original
    assert artifact.read_bytes() == artifact_bytes
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_unsupported_manifest_version_fails_loudly(tmp_path: Path) -> None:
    """Unknown schema versions fail during direct and file-backed validation."""
    manifest = _manifest()
    manifest["schema_version"] = "2.0"
    with pytest.raises(
        UnsupportedManifestVersionError, match="unsupported manifest schema_version"
    ):
        validate_manifest(manifest)
    path = tmp_path / "future.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(UnsupportedManifestVersionError):
        read_manifest(path)


def test_original_manifest_exception_name_remains_compatible() -> None:
    """Keep the original public exception import as an exact class alias."""
    assert UnsupportedManifestVersion is UnsupportedManifestVersionError


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "upstream_artifacts",
            {},
            "manifest field 'upstream_artifacts' must be an array",
        ),
        ("environment", [], "manifest field 'environment' must be an object"),
        ("outputs", [], "manifest field 'outputs' must be a non-empty array"),
        ("outputs", {}, "manifest field 'outputs' must be a non-empty array"),
        ("risk_class", "unknown", "manifest field 'risk_class' has an unknown value"),
    ],
)
def test_manifest_rejects_malformed_fields(
    field: str, value: object, message: str
) -> None:
    """Malformed fields retain their public error type and exact message."""
    manifest = _manifest()
    manifest[field] = value

    with pytest.raises(ContractError) as error:
        validate_manifest(manifest)

    assert str(error.value) == message


def test_manifest_file_requires_an_object(tmp_path: Path) -> None:
    """A non-object JSON document fails at the manifest input boundary."""
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ContractError) as error:
        read_manifest(path)

    assert str(error.value) == "manifest must be an object"

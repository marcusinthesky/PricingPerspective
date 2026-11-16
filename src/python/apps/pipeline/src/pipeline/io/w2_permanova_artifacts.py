"""Publish and admit the paired W2 sector result and permutation archive."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np


def _validate_schema(summary: dict[str, Any]) -> None:
    version = summary.get("schema_version")
    if type(version) is not int or version != 1:
        message = f"unsupported W2 PERMANOVA schema_version {version!r}; expected 1"
        raise ValueError(message)


def _audit_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_w2_permanova_results(
    output_dir: Path, summary: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    """Atomically publish the archive, then its hash-bound JSON completion record.

    Replacing directory entries also supports read-only DVC cache links. A failed
    write invalidates the previous completion record before any archive changes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    results_file.unlink(missing_ok=True)
    _validate_schema(summary)
    audit_file = output_dir / "permutation_audit.npz"
    temporary_files: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir, prefix=".permutation_audit.", suffix=".tmp", delete=False
        ) as handle:
            temporary_audit = Path(handle.name)
            temporary_files.append(temporary_audit)
            np.savez_compressed(handle, allow_pickle=False, **audit)
        temporary_audit.replace(audit_file)
        result = {**summary, "audit_sha256": _audit_digest(audit_file)}
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".results.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_result = Path(handle.name)
            temporary_files.append(temporary_result)
            handle.write(serialized + "\n")
        temporary_result.replace(results_file)
        return result
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)


def read_w2_permanova_results(results_file: Path) -> dict[str, Any]:
    """Read schema-1 results only when the companion archive matches their hash."""
    document = json.loads(results_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "W2 PERMANOVA results must be a JSON object"
        raise TypeError(message)
    result = cast("dict[str, Any]", document)
    _validate_schema(result)
    expected = result.get("audit_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        message = "W2 PERMANOVA results require a lowercase SHA-256 audit digest"
        raise ValueError(message)
    audit_file = results_file.parent / "permutation_audit.npz"
    try:
        actual = _audit_digest(audit_file)
    except FileNotFoundError as error:
        message = f"W2 PERMANOVA permutation audit is missing: {audit_file}"
        raise ValueError(message) from error
    if actual != expected:
        message = "W2 PERMANOVA permutation audit checksum mismatch"
        raise ValueError(message)
    return result

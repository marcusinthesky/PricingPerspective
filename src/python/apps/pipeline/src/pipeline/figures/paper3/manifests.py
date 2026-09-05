"""Deterministic manifests for the Paper 3 publication leaves."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.io.contracts import write_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_kind(path: Path) -> str:
    if path.suffix == ".tex":
        return "latex"
    return path.suffix.removeprefix(".") or "file"


def write_paper3_manifest(
    path: Path,
    *,
    stage_key: str,
    phase: str,
    outputs: Iterable[Path],
    upstreams: Iterable[Path],
) -> None:
    """Write a contract-valid manifest after a Paper 3 leaf completes."""
    output_list = list(outputs)
    upstream_list = list(upstreams)
    write_manifest(
        path,
        {
            "schema_version": "1.0",
            "stage_key": stage_key,
            "lane": "p3",
            "phase": phase,
            "protocol_id": "paper3-publication-v1",
            "code_identity": {"kind": "module", "value": stage_key},
            "parameter_identity": {"kind": "file", "value": "params.yaml"},
            "seed_policy": "deterministic-from-declared-inputs",
            "upstream_artifacts": [
                {"kind": "path-sha256", "value": f"{item}:{_sha256(Path(item))}"}
                for item in sorted(str(item) for item in upstream_list)
                if Path(item).is_file()
            ],
            "environment": {"python": "workspace", "renderer": "matplotlib"},
            "outputs": [
                {
                    "path": str(item),
                    "kind": _output_kind(item),
                    "sha256": _sha256(item),
                    "bytes": item.stat().st_size,
                }
                for item in output_list
            ],
            "risk_class": "moderate",
        },
    )

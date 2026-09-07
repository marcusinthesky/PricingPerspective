"""Manifest-backed readers and writers for reusable typed geometry components."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np
import pandas as pd

from pipeline._kernels.typed_geometry import (
    GeometryComponentMetadata,
    KernelMeanMatrix,
    MeanGroundDistanceMatrix,
)
from pipeline.io.contracts import read_manifest, write_manifest

if TYPE_CHECKING:
    from pathlib import Path


class TypedGeometryArtifactError(ValueError):
    """Raised when a persisted raw geometry component is not self-consistent."""


def _artifact_error(message: str) -> NoReturn:
    raise TypedGeometryArtifactError(message)


def _identity(path: Path) -> dict[str, object]:
    """Return the stable byte identity used by summary and manifest."""
    content = path.read_bytes()
    return {
        "path": str(path),
        "kind": "parquet",
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _component_path(output_dir: Path, kind: str) -> Path:
    if kind == "mean_ground_distance":
        return output_dir / "mean_ground_distances.parquet"
    if kind == "kernel_mean":
        return output_dir / "kernel_means.parquet"
    _artifact_error(f"unknown typed geometry component kind {kind!r}")


def write_geometry_component_artifact(
    output_dir: Path,
    component: MeanGroundDistanceMatrix | KernelMeanMatrix,
    *,
    parameter_identity: str,
    code_identity: str,
) -> dict[str, object]:
    """Write one raw component, summary, and provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _component_path(output_dir, component.metadata.component_kind)
    rows = [
        {
            "item_i": component.item_ids[i],
            "item_j": component.item_ids[j],
            "value": float(component.values[i, j]),  # noqa: PD011
        }
        for i in range(len(component.item_ids))
        for j in range(len(component.item_ids))
    ]
    pd.DataFrame(rows, columns=["item_i", "item_j", "value"]).to_parquet(
        path, index=False
    )
    inventory = [_identity(path)]
    metadata = component.metadata.as_dict()
    summary: dict[str, object] = {
        "schema_version": "typed_geometry_component.v1",
        "artifact_type": "raw_geometry_component",
        "component_kind": component.metadata.component_kind,
        "geometry_id": component.metadata.geometry_id,
        "provider_id": component.metadata.provider_id,
        "representation_id": component.metadata.representation_id,
        "item_ids": list(component.item_ids),
        "shape": [len(component.item_ids), len(component.item_ids)],
        "parameter_identity": parameter_identity,
        "code_identity": code_identity,
        "metadata": metadata,
        "output_inventory": inventory,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "shared_typed_geometry_component",
            "lane": "pipeline",
            "phase": "compute",
            "protocol_id": "t80.typed_geometry_component.v1",
            "code_identity": {"kind": "sha256", "value": code_identity},
            "parameter_identity": {"kind": "sha256", "value": parameter_identity},
            "seed_policy": "deterministic-from-declared-inputs",
            "upstream_artifacts": [
                {"kind": "sha256", "value": source_hash}
                for source_hash in component.metadata.input_hashes
            ],
            "environment": {
                "input_dtype": component.metadata.input_dtype,
                "compute_dtype": component.metadata.compute_dtype,
                "output_dtype": component.metadata.output_dtype,
            },
            "outputs": inventory,
            "risk_class": "moderate",
            "metadata": {
                **metadata,
                "parameter_identity": parameter_identity,
                "code_identity": code_identity,
            },
        },
    )
    return summary


def read_geometry_component_artifact(  # noqa: C901, PLR0912, PLR0915
    output_dir: Path,
    *,
    expected_identity: dict[str, object] | None = None,
) -> MeanGroundDistanceMatrix | KernelMeanMatrix:
    """Read one component only after identity, coverage, and matrix checks."""
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(output_dir / "provenance.manifest.json")
    if not isinstance(summary, dict) or not isinstance(manifest.get("metadata"), dict):
        _artifact_error("typed geometry summary/manifest is malformed")
    manifest_metadata = cast("dict[str, object]", manifest["metadata"])
    metadata_dict = summary.get("metadata")
    if not isinstance(metadata_dict, dict):
        _artifact_error("typed geometry metadata is malformed")
    if any(manifest_metadata.get(key) != value for key, value in metadata_dict.items()):
        _artifact_error("typed geometry summary/manifest metadata mismatch")
    for field in ("component_kind", "geometry_id", "provider_id", "representation_id"):
        if summary.get(field) != metadata_dict.get(field):
            _artifact_error(f"typed geometry identity mismatch for {field}")
        if (
            expected_identity is not None
            and field in expected_identity
            and expected_identity[field] != summary.get(field)
        ):
            _artifact_error(f"typed geometry does not match expected {field}")
    if expected_identity is not None:
        for field in ("parameter_identity", "code_identity"):
            if (
                field in expected_identity
                and summary.get(field) != expected_identity[field]
            ):
                _artifact_error(f"typed geometry does not match expected {field}")
    kind = metadata_dict.get("component_kind")
    if not isinstance(kind, str):
        _artifact_error("typed geometry component kind is missing")
    path = _component_path(output_dir, kind)
    actual = _identity(path)
    for inventory_name, entries in (
        ("summary", summary.get("output_inventory")),
        ("manifest", manifest.get("outputs")),
    ):
        if not isinstance(entries, list) or len(entries) != 1:
            _artifact_error(
                f"typed geometry {inventory_name} output inventory is malformed"
            )
        declared = entries[0]
        if not isinstance(declared, dict) or any(
            declared.get(field) != actual[field]
            for field in ("kind", "sha256", "bytes")
        ):
            _artifact_error(f"typed geometry {inventory_name} output identity mismatch")
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"item_i", "item_j", "value"}:
        _artifact_error("typed geometry parquet schema is malformed")
    item_ids = metadata_dict.get("item_ids")
    if not isinstance(item_ids, list) or not item_ids:
        _artifact_error("typed geometry item IDs are malformed")
    expected_pairs = [(left, right) for left in item_ids for right in item_ids]
    actual_pairs = list(frame[["item_i", "item_j"]].itertuples(index=False, name=None))
    if actual_pairs != expected_pairs:
        _artifact_error("typed geometry parquet ordering or coverage is malformed")
    matrix = frame["value"].to_numpy().reshape(len(item_ids), len(item_ids))
    if not np.issubdtype(matrix.dtype, np.floating) or not np.all(np.isfinite(matrix)):
        _artifact_error("typed geometry component contains non-finite values")
    tolerance = (
        32.0 * np.finfo(matrix.dtype).eps * max(1.0, float(np.max(np.abs(matrix))))
    )
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=tolerance):
        _artifact_error("typed geometry component is not symmetric")
    metadata = dict(metadata_dict)
    metadata["item_ids"] = tuple(cast("list[str]", metadata["item_ids"]))
    metadata["input_hashes"] = tuple(
        cast("list[str]", metadata.get("input_hashes", []))
    )
    metadata["item_counts"] = tuple(
        int(count) for count in cast("list[int]", metadata.get("item_counts", []))
    )
    if metadata["item_counts"] and len(metadata["item_counts"]) != len(item_ids):
        _artifact_error("typed geometry item counts do not cover every item")
    typed_metadata = GeometryComponentMetadata(**metadata)
    if kind == "mean_ground_distance":
        return MeanGroundDistanceMatrix(
            item_ids=typed_metadata.item_ids, values=matrix, metadata=typed_metadata
        )
    if kind == "kernel_mean":
        return KernelMeanMatrix(
            item_ids=typed_metadata.item_ids, values=matrix, metadata=typed_metadata
        )
    _artifact_error(f"unknown typed geometry component kind {kind!r}")


__all__ = [
    "TypedGeometryArtifactError",
    "read_geometry_component_artifact",
    "write_geometry_component_artifact",
]

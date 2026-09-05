"""Deterministic Parquet and JSON boundaries for shared barycentre outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.io.contracts import read_manifest, write_manifest

RANK_TWO = 2
FLOAT32_INVARIANT_FACTOR = 32.0

if TYPE_CHECKING:
    from pipeline._kernels.typed_barycentre import BarycentreResult


@dataclass(frozen=True)
class BarycentreArtifactMetadata:
    """Typed provenance fields shared by target-projection consumers."""

    arm_id: str
    provider_id: str
    representation_id: str
    geometry_id: str
    feasible_set: str
    problem: str
    solver: str
    estimator: str
    input_hashes: tuple[str, ...]
    parameter_identity: str
    code_identity: str
    objective_semantics: str = "unspecified"
    solver_tol: float | None = None
    solver_maxiter: int | None = None
    input_dtype: str = "unknown"
    compute_dtype: str = "unknown"
    output_dtype: str = "unknown"
    seed: int | None = None
    provider_model_id: str | None = None
    provider_vintage: str | None = None
    provider_status: str | None = None
    provider_caveat: str | None = None
    #: Constraint *family* of ``feasible_set``, which is an id. The two coincide
    #: for every feasible set whose id was chosen to equal its kind, but they are
    #: different fields: two arms can share a kind while carrying different
    #: prescribed weight vectors, and therefore different ids. Guards must branch
    #: on this, never on the id.
    feasible_set_kind: str = "unspecified"

    def as_dict(self) -> dict[str, object]:
        """Return provenance metadata as JSON-compatible values."""
        return {
            "arm_id": self.arm_id,
            "provider_id": self.provider_id,
            "provider_model_id": self.provider_model_id,
            "provider_vintage": self.provider_vintage,
            "provider_status": self.provider_status,
            "provider_caveat": self.provider_caveat,
            "representation_id": self.representation_id,
            "feasible_set": self.feasible_set,
            "feasible_set_kind": self.feasible_set_kind,
            "geometry": self.geometry_id,
            "problem": self.problem,
            "solver": self.solver,
            "solver_tol": self.solver_tol,
            "solver_maxiter": self.solver_maxiter,
            "estimator": self.estimator,
            "objective_semantics": self.objective_semantics,
            "input_hashes": list(self.input_hashes),
            "parameter_identity": self.parameter_identity,
            "code_identity": self.code_identity,
            "input_dtype": self.input_dtype,
            "compute_dtype": self.compute_dtype,
            "output_dtype": self.output_dtype,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MeasureBarycentreResult:
    """A source-measure barycentre with explicit support and source weights."""

    support_ids: tuple[str, ...]
    support_values: np.ndarray
    source_ids: tuple[str, ...]
    source_weights: tuple[float, ...]
    objective: float
    initial_objective: float
    optimality_gap: float
    constraint_violation: float
    converged: bool
    iterations: int
    solver: str
    feasible_set: str

    def __post_init__(self) -> None:
        """Validate support arrays and prescribed source weights."""
        values = np.asarray(self.support_values)
        weights = np.asarray(self.source_weights, dtype=values.dtype)
        if values.ndim != RANK_TWO or values.shape[0] != len(self.support_ids):
            _artifact_error("measure barycentre support shape disagrees with IDs")
        if len(self.source_ids) != len(weights):
            _artifact_error("measure barycentre source weights disagree with IDs")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
            _artifact_error("measure barycentre contains non-finite values")
        tolerance = _dtype_tolerance(str(values.dtype))
        if np.any(weights < -tolerance) or not np.isclose(
            np.sum(weights), 1.0, atol=tolerance, rtol=0.0
        ):
            _artifact_error("measure barycentre source weights must be a simplex")


def _artifact_error(message: str) -> NoReturn:
    raise ValueError(message)


def _dtype_tolerance(dtype_name: object) -> float:
    """Return a structural tolerance tied to the declared realized dtype."""
    if not isinstance(dtype_name, str):
        _artifact_error("artifact output_dtype must be declared")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError:
        _artifact_error("artifact output_dtype is invalid")
    if not np.issubdtype(dtype, np.floating):
        _artifact_error("artifact output_dtype must be floating")
    return float(FLOAT32_INVARIANT_FACTOR * np.finfo(dtype).eps)


def _validate_manifest_metadata(
    summary: dict[str, object], manifest: dict[str, object], metadata: dict[str, object]
) -> None:
    """Make the manifest authoritative and reject any summary divergence."""
    for field, value in metadata.items():
        if summary.get(field) != value:
            _artifact_error(f"summary/manifest metadata mismatch for {field}")
    for manifest_field, metadata_field in (
        ("code_identity", "code_identity"),
        ("parameter_identity", "parameter_identity"),
    ):
        identity = manifest.get(manifest_field)
        if not isinstance(identity, dict) or identity.get("value") != metadata.get(
            metadata_field
        ):
            _artifact_error(f"manifest {manifest_field} disagrees with metadata")


def _validate_output_inventory(
    entries: object,
    output_dir: Path,
    source: str,
    expected_names: set[str],
) -> None:
    """Verify the declared Parquet identities against bytes on disk."""
    if not isinstance(entries, list):
        _artifact_error(f"{source} output_inventory must be an array")
    by_name: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            _artifact_error(f"{source} output_inventory has an invalid entry")
        name = Path(entry["path"]).name
        if name in by_name:
            _artifact_error(f"{source} output_inventory repeats {name}")
        by_name[name] = entry
    if set(by_name) != expected_names:
        _artifact_error(f"{source} output_inventory has incomplete outputs")
    for name in expected_names:
        actual = _output_identity(output_dir / name)
        declared = by_name[name]
        for field in ("kind", "sha256", "bytes"):
            if declared.get(field) != actual[field]:
                _artifact_error(f"{source} output identity mismatch for {name}")


def write_target_projection_artifact(
    output_dir: Path,
    result: BarycentreResult,
    metadata: BarycentreArtifactMetadata,
) -> dict[str, object]:
    """Write weights, diagnostics, summary, and a validated manifest."""
    realized_dtype = str(np.result_type(*(row.dtype for row in result.weights)))
    if metadata.output_dtype == "unknown":
        metadata = replace(
            metadata,
            input_dtype=realized_dtype,
            compute_dtype=realized_dtype,
            output_dtype=realized_dtype,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_rows: list[dict[str, object]] = []
    for target_index, (target_id, candidates, weights) in enumerate(
        zip(result.target_ids, result.candidate_ids, result.weights, strict=True)
    ):
        values = np.asarray(weights)
        for candidate_index, (candidate_id, value) in enumerate(
            zip(candidates, values, strict=True)
        ):
            weight_rows.append(
                {
                    "target": target_id,
                    "candidate": candidate_id,
                    "target_index": target_index,
                    "candidate_index": candidate_index,
                    "weight": float(value),
                }
            )
    weight_frame = pd.DataFrame(weight_rows).sort_values(
        ["target_index", "candidate_index"], kind="mergesort"
    )
    diagnostics_frame = pd.DataFrame(
        [diagnostic.as_dict() for diagnostic in result.diagnostics]
    ).sort_values("target_id", kind="mergesort")
    _write_parquet(weight_frame, output_dir / "weights.parquet")
    _write_parquet(diagnostics_frame, output_dir / "diagnostics.parquet")

    outputs = [
        _output_identity(output_dir / name)
        for name in ("weights.parquet", "diagnostics.parquet")
    ]
    summary: dict[str, object] = {
        "schema_version": "typed_barycentre.v1",
        "artifact_type": "target_projection_weights",
        "target_count": len(result.target_ids),
        "candidate_count_per_target": len(result.candidate_ids[0])
        if result.candidate_ids
        else 0,
        "output_inventory": outputs,
        **metadata.as_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "stage_key": "shared_barycentre",
        "lane": "pipeline",
        "phase": "compute",
        "protocol_id": "t80.barycentre.v1",
        "code_identity": {"kind": "sha256", "value": metadata.code_identity},
        "parameter_identity": {"kind": "sha256", "value": metadata.parameter_identity},
        "seed_policy": "declared"
        if metadata.seed is not None
        else "deterministic-from-declared-inputs",
        "upstream_artifacts": [
            {"kind": "sha256", "value": input_hash}
            for input_hash in metadata.input_hashes
        ],
        "environment": {
            "input_dtype": metadata.input_dtype,
            "compute_dtype": metadata.compute_dtype,
            "output_dtype": metadata.output_dtype,
        },
        "outputs": outputs,
        "risk_class": "moderate",
        "metadata": metadata.as_dict(),
    }
    write_manifest(output_dir / "provenance.manifest.json", manifest)
    return summary


def write_measure_barycentre_artifact(
    output_dir: Path,
    result: MeasureBarycentreResult,
    metadata: BarycentreArtifactMetadata,
) -> dict[str, object]:
    """Write a source-measure barycentre and its prescribed source weights."""
    realized_dtype = str(np.asarray(result.support_values).dtype)
    if metadata.output_dtype == "unknown":
        metadata = replace(
            metadata,
            input_dtype=realized_dtype,
            compute_dtype=realized_dtype,
            output_dtype=realized_dtype,
        )
    if metadata.problem != "source_barycentre":
        _artifact_error("measure barycentre metadata must declare source_barycentre")
    # Branch on the kind, not the id: several arms declare prescribed weights at
    # different arities, so they necessarily carry distinct feasible-set ids.
    if metadata.feasible_set_kind != "prescribed_source_weights":
        _artifact_error("measure barycentre metadata must declare prescribed weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    values = np.asarray(result.support_values)
    for support_index, support_id in enumerate(result.support_ids):
        for coordinate_index, coordinate in enumerate(values[support_index]):
            rows.append(
                {
                    "row_kind": "support",
                    "row_index": support_index,
                    "support_id": support_id,
                    "source_id": None,
                    "coordinate_index": coordinate_index,
                    "coordinate": float(coordinate),
                    "source_weight": None,
                }
            )
    for source_index, (source_id, weight) in enumerate(
        zip(result.source_ids, result.source_weights, strict=True)
    ):
        rows.append(
            {
                "row_kind": "source",
                "row_index": source_index,
                "support_id": None,
                "source_id": source_id,
                "coordinate_index": None,
                "coordinate": None,
                "source_weight": float(weight),
            }
        )
    frame = pd.DataFrame(rows)
    path = output_dir / "barycentre.parquet"
    _write_parquet(frame, path)
    output = _output_identity(path)
    summary: dict[str, object] = {
        "schema_version": "typed_barycentre.v1",
        "artifact_type": "source_measure_barycentre",
        "support_count": len(result.support_ids),
        "source_count": len(result.source_ids),
        "objective": result.objective,
        "initial_objective": result.initial_objective,
        "optimality_gap": result.optimality_gap,
        "constraint_violation": result.constraint_violation,
        "converged": result.converged,
        "iterations": result.iterations,
        "output_inventory": [output],
        **metadata.as_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "stage_key": "shared_barycentre",
        "lane": "pipeline",
        "phase": "compute",
        "protocol_id": "t80.barycentre.v1",
        "code_identity": {"kind": "sha256", "value": metadata.code_identity},
        "parameter_identity": {"kind": "sha256", "value": metadata.parameter_identity},
        "seed_policy": "declared"
        if metadata.seed is not None
        else "deterministic-from-declared-inputs",
        "upstream_artifacts": [
            {"kind": "sha256", "value": input_hash}
            for input_hash in metadata.input_hashes
        ],
        "environment": {
            "input_dtype": metadata.input_dtype,
            "compute_dtype": metadata.compute_dtype,
            "output_dtype": metadata.output_dtype,
        },
        "outputs": [output],
        "risk_class": "moderate",
        "metadata": metadata.as_dict(),
    }
    write_manifest(output_dir / "provenance.manifest.json", manifest)
    return summary


def read_target_projection_artifact(  # noqa: C901, PLR0912, PLR0915
    output_dir: Path,
    *,
    expected_identity: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Read and validate a shared target-projection artifact."""
    manifest = read_manifest(output_dir / "provenance.manifest.json")
    weights = pd.read_parquet(output_dir / "weights.parquet")
    diagnostics = pd.read_parquet(output_dir / "diagnostics.parquet")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        _artifact_error("summary.json must contain an object")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        _artifact_error("provenance manifest has no typed metadata")
    _validate_manifest_metadata(summary, manifest, metadata)
    _validate_output_inventory(
        summary.get("output_inventory"),
        output_dir,
        "summary",
        {"weights.parquet", "diagnostics.parquet"},
    )
    _validate_output_inventory(
        manifest.get("outputs"),
        output_dir,
        "manifest",
        {"weights.parquet", "diagnostics.parquet"},
    )
    identity_fields = (
        "arm_id",
        "provider_id",
        "representation_id",
        "geometry",
        "feasible_set",
        "objective_semantics",
        "projected_gradient_norm",
        "cnd_tangent_max_eigenvalue",
        "tangent_curvature_min",
        "simplex_sum_error",
        "min_weight",
    )
    for field in identity_fields:
        if summary.get(field) != metadata.get(field):
            _artifact_error(f"summary/manifest identity mismatch for {field}")
        if (
            expected_identity is not None
            and field in expected_identity
            and expected_identity[field] != summary.get(field)
        ):
            _artifact_error(f"artifact identity mismatch for {field}")
    required_weights = {
        "target",
        "candidate",
        "target_index",
        "candidate_index",
        "weight",
    }
    required_diagnostics = {
        "target_id",
        "objective",
        "initial_objective",
        "optimality_gap",
        "constraint_violation",
        "active_support_size",
        "converged",
        "iterations",
        "solver",
        "feasible_set",
    }
    if not required_weights.issubset(weights.columns):
        _artifact_error("weights.parquet has an incomplete schema")
    if not required_diagnostics.issubset(diagnostics.columns):
        _artifact_error("diagnostics.parquet has an incomplete schema")
    sorted_diagnostics = diagnostics.sort_values(
        "target_id", kind="mergesort"
    ).reset_index(drop=True)
    if not diagnostics.reset_index(drop=True).equals(sorted_diagnostics):
        _artifact_error("diagnostics.parquet rows are not deterministically ordered")
    if weights.empty or diagnostics.empty:
        _artifact_error("shared barycentre artifacts must not be empty")
    if weights[["target", "candidate"]].duplicated().any():
        _artifact_error("weights.parquet contains duplicate target/candidate IDs")
    if weights[["target_index", "candidate_index"]].duplicated().any():
        _artifact_error("weights.parquet contains duplicate target/candidate indices")
    sorted_weights = weights.sort_values(
        ["target_index", "candidate_index"], kind="mergesort"
    ).reset_index(drop=True)
    if not weights.reset_index(drop=True).equals(sorted_weights):
        _artifact_error("weights.parquet rows are not deterministically ordered")
    target_groups = weights.groupby("target_index", sort=True)
    target_count = int(target_groups.ngroups)
    if summary.get("target_count") != target_count:
        _artifact_error("summary target_count disagrees with weights.parquet")
    candidate_counts = target_groups.size().to_numpy(dtype=np.int64)
    if not np.all(candidate_counts == candidate_counts[0]):
        _artifact_error("targets have inconsistent candidate coverage")
    expected_candidates = int(candidate_counts[0])
    if summary.get("candidate_count_per_target") != expected_candidates:
        _artifact_error("summary candidate count disagrees with weights.parquet")
    for _, group in target_groups:
        candidate_indices = group["candidate_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(candidate_indices, np.arange(expected_candidates)):
            _artifact_error("candidate indices must be contiguous and zero-based")
    if target_groups["target"].nunique().max() != 1:
        _artifact_error("each target_index must have one target ID")
    weights_values = weights["weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weights_values)):
        _artifact_error("weights.parquet contains non-finite weights")
    feasible_set = summary.get("feasible_set")
    tolerance = _dtype_tolerance(summary.get("output_dtype"))
    sums = target_groups["weight"].sum().to_numpy(dtype=np.float64)
    if feasible_set == "simplex_nonnegative" and (
        np.any(weights_values < -tolerance)
        or not np.allclose(sums, 1.0, atol=tolerance, rtol=0.0)
    ):
        _artifact_error("simplex weights violate nonnegativity or unit row sums")
    if feasible_set == "affine_signed" and not np.allclose(
        sums, 1.0, atol=tolerance, rtol=0.0
    ):
        _artifact_error("affine-signed weights violate unit row sums")
    if diagnostics["target_id"].duplicated().any():
        _artifact_error("diagnostics.parquet contains duplicate target IDs")
    if diagnostics["target_id"].nunique() != target_count:
        _artifact_error("diagnostics target coverage disagrees with weights.parquet")
    if set(diagnostics["target_id"]) != set(weights["target"]):
        _artifact_error("diagnostics target IDs do not match weights.parquet")
    if (
        len(set(diagnostics["solver"])) != 1
        or set(diagnostics["solver"]) != {summary.get("solver")}
        or len(set(diagnostics["feasible_set"])) != 1
        or set(diagnostics["feasible_set"]) != {feasible_set}
    ):
        _artifact_error("diagnostics solver metadata disagrees with summary")
    if (
        not diagnostics["converged"]
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .all()
    ):
        _artifact_error("diagnostics converged values must be boolean")
    for field in ("active_support_size", "iterations"):
        values = diagnostics[field].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            _artifact_error(f"diagnostics {field} must be finite and nonnegative")
    for field in required_diagnostics - {
        "target_id",
        "converged",
        "solver",
        "feasible_set",
        "active_support_size",
        "iterations",
    }:
        if not np.all(np.isfinite(diagnostics[field].to_numpy(dtype=np.float64))):
            _artifact_error(f"diagnostics.parquet contains non-finite {field}")
    return weights, diagnostics, summary


def read_measure_barycentre_artifact(  # noqa: C901, PLR0912
    output_dir: Path,
    *,
    expected_identity: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read and validate a source-measure barycentre artifact."""
    manifest = read_manifest(output_dir / "provenance.manifest.json")
    frame = pd.read_parquet(output_dir / "barycentre.parquet")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        _artifact_error("summary.json must contain an object")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        _artifact_error("provenance manifest has no typed metadata")
    _validate_manifest_metadata(summary, manifest, metadata)
    _validate_output_inventory(
        summary.get("output_inventory"), output_dir, "summary", {"barycentre.parquet"}
    )
    _validate_output_inventory(
        manifest.get("outputs"), output_dir, "manifest", {"barycentre.parquet"}
    )
    for field in (
        "arm_id",
        "provider_id",
        "representation_id",
        "geometry",
        "feasible_set",
    ):
        if summary.get(field) != metadata.get(field):
            _artifact_error(f"summary/manifest identity mismatch for {field}")
        if expected_identity is not None and expected_identity.get(
            field
        ) != summary.get(field):
            _artifact_error(f"artifact identity mismatch for {field}")
    required = {
        "row_kind",
        "row_index",
        "support_id",
        "source_id",
        "coordinate_index",
        "coordinate",
        "source_weight",
    }
    if not required.issubset(frame.columns):
        _artifact_error("barycentre.parquet has an incomplete schema")
    if frame.empty or not frame["row_kind"].isin(["support", "source"]).all():
        _artifact_error("barycentre.parquet has invalid row kinds")
    support = frame.loc[frame["row_kind"] == "support"]
    sources = frame.loc[frame["row_kind"] == "source"]
    support_count = summary.get("support_count")
    source_count = summary.get("source_count")
    if not isinstance(support_count, int) or not isinstance(source_count, int):
        _artifact_error("summary source/support counts must be integers")
    if support_count <= 0 or source_count <= 0:
        _artifact_error("source-measure barycentre counts must be positive")
    if len(support) % support_count != 0 or len(sources) != source_count:
        _artifact_error("barycentre row counts disagree with summary")
    if list(frame["row_kind"]) != ["support"] * len(support) + ["source"] * len(
        sources
    ):
        _artifact_error("barycentre.parquet rows are not deterministically ordered")
    if support["support_id"].nunique() != support_count:
        _artifact_error("barycentre support IDs are incomplete or duplicated")
    if sources["source_id"].nunique() != source_count:
        _artifact_error("barycentre source IDs are incomplete or duplicated")
    weights = sources["source_weight"].to_numpy(dtype=np.float64)
    coordinates = support["coordinate"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(coordinates)):
        _artifact_error("barycentre.parquet contains non-finite values")
    tolerance = _dtype_tolerance(summary.get("output_dtype"))
    if np.any(weights < -tolerance) or not np.isclose(
        np.sum(weights), 1.0, atol=tolerance, rtol=0.0
    ):
        _artifact_error("barycentre source weights violate prescribed simplex")
    for field in (
        "objective",
        "initial_objective",
        "optimality_gap",
        "constraint_violation",
    ):
        value = summary.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            _artifact_error(f"summary contains non-finite {field}")
    return frame, summary


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a stable Arrow table with no wall-clock metadata."""
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=False,
        write_page_index=False,
    )


def _output_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "kind": "parquet" if path.suffix == ".parquet" else "json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


__all__ = [
    "BarycentreArtifactMetadata",
    "MeasureBarycentreResult",
    "read_measure_barycentre_artifact",
    "read_target_projection_artifact",
    "write_measure_barycentre_artifact",
    "write_target_projection_artifact",
]

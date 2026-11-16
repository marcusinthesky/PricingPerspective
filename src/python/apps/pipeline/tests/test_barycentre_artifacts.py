"""Structural validation tests for shared barycentre artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline.io.barycentre_artifacts import (
    BarycentreArtifactMetadata,
    MeasureBarycentreResult,
    read_measure_barycentre_artifact,
    read_target_projection_artifact,
    write_measure_barycentre_artifact,
)
from pipeline.io.contracts import write_manifest

if TYPE_CHECKING:
    from pathlib import Path
EXPECTED_WEIGHT_ROWS = 2
EXPECTED_DIAGNOSTIC_ROWS = 1
EXPECTED_MEASURE_ROWS = 6
EXPECTED_PSD_TANGENT_MIN_EIGENVALUE = 0.25


def _write_artifact(tmp_path: Path, weights: pd.DataFrame) -> Path:
    """Write a minimal manifest-backed target-projection artifact."""
    diagnostics = pd.DataFrame(
        [
            {
                "target_id": "A",
                "objective": 0.5,
                "initial_objective": 1.0,
                "constraint_violation": 0.0,
                "optimality_gap": 0.0,
                "iterations": 1,
                "solver": "energy_qp",
                "feasible_set": "simplex_nonnegative",
                "active_support_size": 2,
                "converged": True,
                "projected_gradient_norm": 0.0,
                "cnd_tangent_max_eigenvalue": 0.0,
                "tangent_curvature_min": 0.0,
                "simplex_sum_error": 0.0,
                "min_weight": 0.5,
            }
        ]
    )
    output_dir = tmp_path / "barycentre"
    output_dir.mkdir()
    weights.to_parquet(output_dir / "weights.parquet", index=False)
    diagnostics.to_parquet(output_dir / "diagnostics.parquet", index=False)
    input_hashes: list[str] = []
    metadata = {
        "arm_id": "energy_simplex",
        "provider_id": "provider",
        "representation_id": "representation",
        "geometry": "energy_v",
        "feasible_set": "simplex_nonnegative",
        "problem": "target_projection",
        "solver": "energy_qp",
        "solver_tol": 1e-8,
        "solver_maxiter": 10_000,
        "estimator": "v_statistic",
        "objective_semantics": "squared_statistical_distance",
        "input_hashes": input_hashes,
        "parameter_identity": "p" * 64,
        "code_identity": "c" * 64,
        "input_dtype": "float32",
        "compute_dtype": "float32",
        "output_dtype": "float32",
        "seed": None,
    }
    summary: dict[str, object] = {
        "schema_version": "typed_barycentre.v1",
        "artifact_type": "target_projection_weights",
        "target_count": 1,
        "candidate_count_per_target": 2,
        **metadata,
    }
    outputs: list[dict[str, object]] = []
    for name in ("weights.parquet", "diagnostics.parquet"):
        content = (output_dir / name).read_bytes()
        outputs.append(
            {
                "path": str(output_dir / name),
                "kind": "parquet",
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    summary["output_inventory"] = outputs
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_manifest(
        output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "shared_barycentre",
            "lane": "pipeline",
            "phase": "compute",
            "protocol_id": "t80.barycentre.v1",
            "code_identity": {"kind": "sha256", "value": "c" * 64},
            "parameter_identity": {"kind": "sha256", "value": "p" * 64},
            "seed_policy": "deterministic-from-declared-inputs",
            "upstream_artifacts": [],
            "environment": {
                "input_dtype": metadata["input_dtype"],
                "compute_dtype": metadata["compute_dtype"],
                "output_dtype": metadata["output_dtype"],
            },
            "outputs": outputs,
            "risk_class": "moderate",
            "metadata": metadata,
        },
    )
    return output_dir


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "A",
                "candidate": "B",
                "target_index": 0,
                "candidate_index": 0,
                "weight": 0.5,
            },
            {
                "target": "A",
                "candidate": "C",
                "target_index": 0,
                "candidate_index": 1,
                "weight": 0.5,
            },
        ]
    )


def test_reader_accepts_complete_simplex_artifact(tmp_path: Path) -> None:
    """A complete artifact passes schema, ordering, and simplex checks."""
    weights, diagnostics, summary = read_target_projection_artifact(
        _write_artifact(tmp_path, _weights()),
        expected_identity={"objective_semantics": "squared_statistical_distance"},
    )

    assert len(weights) == EXPECTED_WEIGHT_ROWS
    assert len(diagnostics) == EXPECTED_DIAGNOSTIC_ROWS
    assert summary["feasible_set"] == "simplex_nonnegative"


def test_measure_barycentre_preserves_prescribed_source_weights(
    tmp_path: Path,
) -> None:
    """Source-measure outputs use their own artifact variant and simplex."""
    result = MeasureBarycentreResult(
        support_ids=("s0", "s1"),
        support_values=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        source_ids=("A", "B"),
        source_weights=(0.5, 0.5),
        objective=1.0,
        initial_objective=2.0,
        optimality_gap=0.0,
        constraint_violation=0.0,
        converged=True,
        iterations=3,
        solver="wasserstein_ot",
        feasible_set="prescribed_source_weights",
    )
    metadata = BarycentreArtifactMetadata(
        arm_id="wasserstein_w1_measure",
        provider_id="provider",
        representation_id="representation",
        geometry_id="wasserstein_w1",
        feasible_set="prescribed_source_weights",
        feasible_set_kind="prescribed_source_weights",
        problem="source_barycentre",
        solver="wasserstein_ot",
        estimator="balanced_wasserstein_1",
        input_hashes=(),
        parameter_identity="p" * 64,
        code_identity="c" * 64,
    )

    summary = write_measure_barycentre_artifact(tmp_path / "measure", result, metadata)
    frame, loaded_summary = read_measure_barycentre_artifact(tmp_path / "measure")

    assert len(frame) == EXPECTED_MEASURE_ROWS
    assert summary == loaded_summary
    assert loaded_summary["artifact_type"] == "source_measure_barycentre"
    assert frame.loc[
        frame["row_kind"] == "source", "source_weight"
    ].sum() == pytest.approx(1.0)
    assert loaded_summary["feasible_set_kind"] == "prescribed_source_weights"


def test_measure_barycentre_accepts_a_distinct_prescribed_feasible_set_id(
    tmp_path: Path,
) -> None:
    """A prescribed-weights arm is admitted on its kind, not on its id.

    Arities differ across arms, so two prescribed-weights feasible sets cannot
    share one id. Guarding on the id would reject every arm but the first.
    """
    result = MeasureBarycentreResult(
        support_ids=("s0", "s1"),
        support_values=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        source_ids=("A", "B", "C"),
        source_weights=(1 / 3, 1 / 3, 1 / 3),
        objective=1.0,
        initial_objective=2.0,
        optimality_gap=0.0,
        constraint_violation=0.0,
        converged=True,
        iterations=7,
        solver="wasserstein_free_support",
        feasible_set="prescribed_source_weights_k3",
    )
    metadata = BarycentreArtifactMetadata(
        arm_id="wasserstein_w2_free_support",
        provider_id="provider",
        representation_id="representation",
        geometry_id="wasserstein_w2",
        feasible_set="prescribed_source_weights_k3",
        feasible_set_kind="prescribed_source_weights",
        problem="source_barycentre",
        solver="wasserstein_free_support",
        estimator="balanced_wasserstein_2",
        input_hashes=(),
        parameter_identity="p" * 64,
        code_identity="c" * 64,
    )

    summary = write_measure_barycentre_artifact(tmp_path / "k3", result, metadata)

    assert summary["feasible_set"] == "prescribed_source_weights_k3"
    assert summary["feasible_set_kind"] == "prescribed_source_weights"


def test_measure_barycentre_rejects_an_unset_feasible_set_kind(
    tmp_path: Path,
) -> None:
    """A caller that omits the kind fails loudly rather than defaulting through."""
    result = MeasureBarycentreResult(
        support_ids=("s0",),
        support_values=np.asarray([[0.0, 1.0]]),
        source_ids=("A", "B"),
        source_weights=(0.5, 0.5),
        objective=1.0,
        initial_objective=2.0,
        optimality_gap=0.0,
        constraint_violation=0.0,
        converged=True,
        iterations=1,
        solver="wasserstein_ot",
        feasible_set="prescribed_source_weights",
    )
    metadata = BarycentreArtifactMetadata(
        arm_id="wasserstein_w1_measure",
        provider_id="provider",
        representation_id="representation",
        geometry_id="wasserstein_w1",
        feasible_set="prescribed_source_weights",
        problem="source_barycentre",
        solver="wasserstein_ot",
        estimator="balanced_wasserstein_1",
        input_hashes=(),
        parameter_identity="p" * 64,
        code_identity="c" * 64,
    )

    with pytest.raises(ValueError, match="prescribed weights"):
        write_measure_barycentre_artifact(tmp_path / "unset", result, metadata)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate", "duplicate target/candidate IDs"),
        ("non_unit", "unit row sums"),
    ],
)
def test_reader_rejects_invalid_weight_structure(
    tmp_path: Path, change: str, message: str
) -> None:
    """Duplicate IDs and violated simplex sums fail before consumption."""
    weights = _weights()
    if change == "duplicate":
        weights.loc[1, "candidate"] = "B"
    else:
        weights.loc[1, "weight"] = 0.25

    with pytest.raises(ValueError, match=message):
        read_target_projection_artifact(_write_artifact(tmp_path, weights))


def test_reader_accepts_mmd_diagnostics(tmp_path: Path) -> None:
    """The shared target artifact preserves MMD-specific solver diagnostics."""
    output_dir = _write_artifact(tmp_path, _weights())
    diagnostics_path = output_dir / "diagnostics.parquet"
    diagnostics = pd.read_parquet(diagnostics_path)
    diagnostics["psd_tangent_min_eigenvalue"] = EXPECTED_PSD_TANGENT_MIN_EIGENVALUE
    diagnostics.to_parquet(diagnostics_path, index=False)
    # Rebuild only the test fixture inventories after the intentional byte change.
    summary = json.loads((output_dir / "summary.json").read_text())
    manifest_path = output_dir / "provenance.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = {
        "path": str(diagnostics_path),
        "kind": "parquet",
        "sha256": hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        "bytes": len(diagnostics_path.read_bytes()),
    }
    summary["output_inventory"][1] = identity
    manifest["outputs"][1] = identity
    (output_dir / "summary.json").write_text(json.dumps(summary))
    write_manifest(manifest_path, manifest)

    _, loaded_diagnostics, _ = read_target_projection_artifact(output_dir)

    assert loaded_diagnostics["psd_tangent_min_eigenvalue"].iloc[0] == pytest.approx(
        EXPECTED_PSD_TANGENT_MIN_EIGENVALUE
    )

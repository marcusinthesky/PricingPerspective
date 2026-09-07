"""Thin DVC orchestrators for the typed statistical-distance boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd

from pipeline._kernels.typed_barycentre import (
    solve_energy_target_projections,
    solve_mmd_target_projections,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
    solve_wasserstein_target_projections,
)
from pipeline._kernels.typed_distances import (
    compute_statistical_distance,
    statistical_distance_from_components,
)
from pipeline.io.barycentre_artifacts import (
    BarycentreArtifactMetadata,
    MeasureBarycentreResult,
    write_measure_barycentre_artifact,
    write_target_projection_artifact,
)
from pipeline.io.contracts import read_manifest, write_manifest
from pipeline.io.params import load_params
from pipeline.io.typed_analysis import (
    StatisticalDistanceSpec,
    load_dvc_scientific_config,
    load_typed_analysis,
    typed_analysis_parameter_identity,
)
from pipeline.io.typed_geometry_artifacts import read_geometry_component_artifact
from pipeline.io.typed_providers import load_provider_clouds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pipeline._kernels.typed_barycentre import MeasureBarycentreKernelResult
    from pipeline._kernels.typed_geometry import (
        EmbeddingCloud,
        KernelMeanMatrix,
        MeanGroundDistanceMatrix,
    )
    from pipeline.io.typed_analysis import (
        BarycentreArm,
        ProviderConfig,
        RepresentationSpec,
        TypedAnalysisConfig,
        TypedAnalysisRegistry,
    )


class TypedStageError(ValueError):
    """Raised when a shared stage receives an incompatible registry arm."""


def _stage_error(message: str) -> NoReturn:
    raise TypedStageError(message)


def run_typed_distance(
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
    output_dir: Path,
    component_dir: Path | None = None,
    params_file: Path = Path("params.yaml"),
) -> dict[str, object]:
    """Materialize one typed statistical-distance matrix."""
    config = load_typed_analysis(params_file)
    provider, representation, distance = _resolve_distance(
        config, provider_id, representation_id, distance_id
    )
    ground = config.ground_distance_for(distance)
    if distance.family in {"energy", "mmd"} and component_dir is not None:
        component = read_geometry_component_artifact(
            component_dir,
            expected_identity={
                "provider_id": provider_id,
                "representation_id": representation_id,
                "geometry_id": distance_id,
            },
        )
        result = statistical_distance_from_components(component, distance)
    else:
        clouds = load_provider_clouds(
            provider,
            representation,
            item_ids=_market_items(params_file),
            sample_size=distance.sample_size,
            window=config.window_for(distance),
        )
        result = compute_statistical_distance(clouds, distance, ground)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_i": result.item_ids[i],
            "item_j": result.item_ids[j],
            "value": float(result.values[i, j]),  # noqa: PD011
        }
        for i in range(len(result.item_ids))
        for j in range(len(result.item_ids))
    ]
    distance_path = output_dir / "distances.parquet"
    pd.DataFrame(rows).to_parquet(distance_path, index=False)
    params_identity = typed_analysis_parameter_identity(
        params_file,
        provider_ids=(provider_id,),
        representation_ids=(representation_id,),
        distance_ids=(distance_id,),
    )
    code_identity = _code_identity()
    metadata = {
        **result.metadata.as_dict(),
        "code_identity": code_identity,
        "layout": "full_symmetric",
        "sample_window": distance.sample_window,
        "shape": [len(result.item_ids), len(result.item_ids)],
    }
    summary = {
        "schema_version": "typed_distance.v1",
        "artifact_type": "statistical_distance_matrix",
        "provider_id": provider_id,
        "representation_id": representation_id,
        "distance_id": distance_id,
        "item_ids": list(result.item_ids),
        "metadata": metadata,
        "output_inventory": [_identity(distance_path)],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_manifest(
        output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "shared_distance",
            "lane": "pipeline",
            "phase": "compute",
            "protocol_id": "t80.distance.v1",
            "code_identity": {"kind": "sha256", "value": code_identity},
            "parameter_identity": {"kind": "sha256", "value": params_identity},
            "seed_policy": "deterministic-from-declared-inputs",
            "upstream_artifacts": [
                {"kind": "sha256", "value": source_hash}
                for source_hash in result.metadata.input_hashes
            ],
            "environment": {"dtype_policy": "float64-host; jax-float64-solver"},
            "outputs": [_identity(distance_path)],
            "risk_class": "moderate",
            "metadata": metadata,
        },
    )
    return cast("dict[str, object]", summary)


def run_typed_distance_roster(
    *,
    provider_id: str,
    roster_key: str,
    output_dir: Path,
    params_file: Path = Path("params.yaml"),
) -> None:
    """Materialize every configured representation/distance arm for a provider."""
    scientific = load_dvc_scientific_config(params_file)
    roster = scientific.typed_distance_providers.get(roster_key)
    if roster is None or roster.provider_id != provider_id:
        _stage_error(f"typed-distance roster {roster_key!r} does not match provider")
    representation_ids = [
        scientific.typed_analysis.representations[key].representation_id
        for key in roster.representation_keys
    ]
    distances = list(roster.distance_ids or scientific.typed_distance_ids)
    if not representation_ids or not distances:
        _stage_error("typed-distance roster axes must not be empty")
    cells = [
        (representation_id, distance_id)
        for representation_id in representation_ids
        for distance_id in distances
    ]
    full_representation_id = scientific.typed_analysis.representations[
        roster.representation_full_key
    ].representation_id
    cells.extend(
        (full_representation_id, distance_id)
        for distance_id in roster.extra_full_distance_ids
    )
    for representation_id, distance_id in cells:
        distance = scientific.typed_analysis.distances[distance_id]
        if not isinstance(distance, StatisticalDistanceSpec):
            _stage_error(
                f"typed-distance roster includes non-statistical {distance_id}"
            )
        artifact_dir = output_dir / representation_id / distance_id
        if all(
            (artifact_dir / filename).is_file()
            for filename in (
                "distances.parquet",
                "summary.json",
                "provenance.manifest.json",
            )
        ):
            try:
                read_typed_distance_artifact(
                    artifact_dir,
                    expected_identity={
                        "provider_id": provider_id,
                        "representation_id": representation_id,
                        "distance_id": distance_id,
                        "value_semantics": distance.value_semantics,
                    },
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                continue
        run_typed_distance(
            provider_id=provider_id,
            representation_id=representation_id,
            distance_id=distance_id,
            output_dir=artifact_dir,
            component_dir=(
                output_dir.parents[1]
                / "typed_geometry_components"
                / provider_id
                / representation_id
                / distance_id
            ),
            params_file=params_file,
        )


def read_typed_distance_artifact(  # noqa: C901, PLR0912
    output_dir: Path,
    *,
    expected_identity: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read a distance matrix only after identity, bytes, and shape validation."""
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(output_dir / "provenance.manifest.json")
    if not isinstance(summary, dict) or not isinstance(manifest.get("metadata"), dict):
        _stage_error("typed distance summary/manifest is malformed")
    metadata = cast("dict[str, object]", manifest["metadata"])
    for field, value in metadata.items():
        if summary.get("metadata", {}).get(field) != value:  # type: ignore[union-attr]
            _stage_error(f"distance summary/manifest mismatch for {field}")
    for field in ("provider_id", "representation_id", "distance_id"):
        if summary.get(field) != metadata.get(field):
            _stage_error(f"distance artifact identity mismatch for {field}")
        if field in expected_identity and expected_identity[field] != summary.get(
            field
        ):
            _stage_error(f"distance artifact does not match expected {field}")
    for field in (
        "representation_transform",
        "representation_dimension",
        "sample_size",
        "value_semantics",
    ):
        expected = expected_identity.get(field)
        if field in expected_identity and metadata.get(field) != expected:
            _stage_error(f"distance artifact does not match expected {field}")
    path = output_dir / "distances.parquet"
    actual = _identity(path)
    for source, entries in (
        ("summary", summary.get("output_inventory")),
        ("manifest", manifest.get("outputs")),
    ):
        if not isinstance(entries, list) or len(entries) != 1:
            _stage_error(f"distance {source} output inventory is malformed")
        declared = entries[0]
        if not isinstance(declared, dict) or any(
            declared.get(field) != actual[field]
            for field in ("kind", "sha256", "bytes")
        ):
            _stage_error(f"distance {source} output identity mismatch")
    frame = pd.read_parquet(path)
    required = {"item_i", "item_j", "value"}
    if set(frame.columns) != required or frame.empty:
        _stage_error("distance parquet schema is malformed")
    item_ids = summary.get("item_ids")
    shape = metadata.get("shape")
    if not isinstance(item_ids, list) or shape != [len(item_ids), len(item_ids)]:
        _stage_error("distance summary shape is malformed")
    if len(frame) != len(item_ids) ** 2:
        _stage_error("distance parquet coverage disagrees with summary")
    expected_pairs = [(left, right) for left in item_ids for right in item_ids]
    actual_pairs = list(frame[["item_i", "item_j"]].itertuples(index=False, name=None))
    if actual_pairs != expected_pairs:
        _stage_error("distance parquet ordering or coverage is malformed")
    values = frame["value"].to_numpy()
    matrix = values.reshape(len(item_ids), len(item_ids))
    if not np.all(np.isfinite(matrix)):
        _stage_error("distance matrix contains non-finite values")
    output_dtype = metadata.get("output_dtype")
    if not isinstance(output_dtype, str):
        _stage_error("distance artifact output dtype is malformed")
    dtype = np.dtype(output_dtype)
    tolerance = 32.0 * np.finfo(dtype).eps * max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=tolerance):
        _stage_error("distance matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=tolerance, rtol=0.0):
        _stage_error("distance matrix diagonal is not zero")
    return frame, summary


def _solve_source_barycentre(
    clouds: Sequence[EmbeddingCloud],
    geometry: StatisticalDistanceSpec,
    arm: BarycentreArm,
    config: TypedAnalysisRegistry,
) -> MeasureBarycentreKernelResult:
    """Dispatch a source barycentre to the solver its arm declares.

    Two arities, two solvers, chosen by the declared arm rather than by counting
    weights: the closed form is exact at two sources, and the free-support
    iteration reaches only a local optimum past that. The registry already
    refuses an exact arm carrying more than two weights, so this dispatch cannot
    silently downgrade an exact result to a heuristic one.
    """
    if geometry.family != "wasserstein" or arm.solver not in {
        "wasserstein_ot",
        "wasserstein_free_support",
    }:
        _stage_error("source barycentres require a Wasserstein arm")
    feasible_set = config.feasible_sets[arm.feasible_set]
    source_weights = feasible_set.source_weights
    if source_weights is None:
        _stage_error("source barycentres require prescribed source weights")
    source_count = len(source_weights)
    if len(clouds) < source_count:
        _stage_error("source barycentre has fewer clouds than source weights")
    sources = tuple(clouds[:source_count])
    if arm.solver == "wasserstein_ot":
        return solve_wasserstein_measure_barycentre(
            sources,
            tuple(source_weights),
            solver=arm.solver,
            feasible_set=arm.feasible_set,
            estimator=geometry.estimator,
        )
    if arm.maxiter is None or arm.tol is None:
        _stage_error("free-support barycentres require a declared tol and maxiter")
    return solve_wasserstein_free_support_barycentre(
        sources,
        tuple(source_weights),
        solver=arm.solver,
        feasible_set=arm.feasible_set,
        estimator=geometry.estimator,
        max_iterations=arm.maxiter,
        tol=arm.tol,
    )


def run_typed_barycentre(
    *,
    provider_id: str,
    representation_id: str,
    arm_id: str,
    output_dir: Path,
    geometry_component_dir: Path | None = None,
    params_file: Path = Path("params.yaml"),
) -> dict[str, object]:
    """Materialize one typed barycentre artifact.

    The solver budget is a registry fact (``tol``/``maxiter`` on the arm), not a
    command-line default, so it stays inside the DVC-tracked ``typed_analysis``
    parameter and is republished in the artifact's own provenance.
    """
    config = load_typed_analysis(params_file)
    provider, representation, geometry, arm = _resolve_barycentre(
        config, provider_id, representation_id, arm_id
    )
    clouds = load_provider_clouds(
        provider,
        representation,
        item_ids=_market_items(params_file),
        sample_size=(
            geometry.sample_size
            if geometry.family == "wasserstein" and arm.problem == "target_projection"
            else None
        ),
    )
    params_identity = typed_analysis_parameter_identity(
        params_file,
        provider_ids=(provider_id,),
        representation_ids=(representation_id,),
        arm_ids=(arm_id,),
    )
    code_identity = _code_identity()
    component = None
    if geometry.family in {"energy", "mmd"}:
        if geometry_component_dir is None:
            _stage_error(
                "energy and MMD barycentres require an explicit geometry component"
            )
        component = read_geometry_component_artifact(
            geometry_component_dir,
            expected_identity={
                "provider_id": provider_id,
                "representation_id": representation_id,
                "geometry_id": geometry.distance_id,
            },
        )
    metadata = BarycentreArtifactMetadata(
        arm_id=arm.arm_id,
        provider_id=provider.provider_id,
        representation_id=representation.representation_id,
        geometry_id=geometry.distance_id,
        feasible_set=arm.feasible_set,
        feasible_set_kind=config.feasible_sets[arm.feasible_set].kind,
        problem=arm.problem,
        solver=arm.solver,
        estimator=geometry.estimator,
        input_hashes=tuple(cloud.source_hash for cloud in clouds if cloud.source_hash),
        parameter_identity=params_identity,
        code_identity=code_identity,
        objective_semantics=(
            "squared_statistical_distance"
            if geometry.estimator != "balanced_wasserstein_1"
            else "statistical_distance"
        ),
        solver_tol=arm.tol,
        solver_maxiter=arm.maxiter,
        seed=arm.seed,
        provider_model_id=provider.model_id,
        provider_vintage=provider.vintage,
        provider_status=provider.status,
        provider_caveat=provider.caveat,
        input_dtype=str(
            np.result_type(*(np.asarray(cloud.values).dtype for cloud in clouds))
        ),
        compute_dtype=str(jnp.asarray(clouds[0].values).dtype),
        output_dtype=str(jnp.asarray(clouds[0].values).dtype),
    )
    if arm.problem == "source_barycentre":
        kernel_result = _solve_source_barycentre(clouds, geometry, arm, config)
        result = MeasureBarycentreResult(
            support_ids=kernel_result.support_ids,
            support_values=kernel_result.support_values,
            source_ids=kernel_result.source_ids,
            source_weights=kernel_result.source_weights,
            objective=kernel_result.objective,
            initial_objective=kernel_result.initial_objective,
            optimality_gap=kernel_result.optimality_gap,
            constraint_violation=kernel_result.constraint_violation,
            converged=kernel_result.converged,
            iterations=kernel_result.iterations,
            solver=kernel_result.solver,
            feasible_set=kernel_result.feasible_set,
        )
        summary = write_measure_barycentre_artifact(output_dir, result, metadata)
        return _publish_representation_identity(output_dir, summary, representation)
    ground = config.ground_distance_for(geometry)
    if geometry.family == "energy":
        energy_component = cast("MeanGroundDistanceMatrix | None", component)
        result = solve_energy_target_projections(
            clouds, geometry, arm, ground, component=energy_component
        )
    elif geometry.family == "mmd":
        mmd_component = cast("KernelMeanMatrix | None", component)
        result = solve_mmd_target_projections(
            clouds, geometry, arm, ground, component=mmd_component
        )
    elif geometry.family == "wasserstein":
        result = solve_wasserstein_target_projections(clouds, geometry, arm, ground)
    else:
        _stage_error("target barycentre stage received an unsupported geometry")
    summary = write_target_projection_artifact(output_dir, result, metadata)
    return _publish_representation_identity(output_dir, summary, representation)


def _publish_representation_identity(
    output_dir: Path,
    summary: dict[str, object],
    representation: RepresentationSpec,
) -> dict[str, object]:
    """Add complete representation identity to newly written v1 artifacts."""
    identity = {
        "representation_transform": representation.transform,
        "representation_dimension": representation.dimension,
    }
    summary.update(identity)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_dir / "provenance.manifest.json"
    manifest = read_manifest(manifest_path)
    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, dict):
        _stage_error("typed barycentre manifest metadata is malformed")
    manifest_metadata.update(identity)
    write_manifest(manifest_path, manifest)
    return summary


def _resolve_distance(
    config: TypedAnalysisConfig,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> tuple[ProviderConfig, RepresentationSpec, StatisticalDistanceSpec]:
    provider = _provider(config, provider_id)
    representation = _representation(config, representation_id)
    if representation.provider_id != provider.provider_id:
        _stage_error("representation/provider identity mismatch")
    distance = config.distances.get(distance_id)
    if not isinstance(distance, StatisticalDistanceSpec):
        _stage_error(f"{distance_id!r} is not a statistical distance")
    return provider, representation, distance


def _resolve_barycentre(
    config: TypedAnalysisConfig,
    provider_id: str,
    representation_id: str,
    arm_id: str,
) -> tuple[ProviderConfig, RepresentationSpec, StatisticalDistanceSpec, BarycentreArm]:
    provider = _provider(config, provider_id)
    representation = _representation(config, representation_id)
    if representation.provider_id != provider.provider_id:
        _stage_error("representation/provider identity mismatch")
    arm, geometry = config.barycentre_for(arm_id)
    return provider, representation, geometry, arm


def _provider(config: TypedAnalysisConfig, provider_id: str) -> ProviderConfig:
    provider = config.providers.get(provider_id)
    if provider is None:
        provider = next(
            (
                candidate
                for candidate in config.providers.values()
                if candidate.provider_id == provider_id
            ),
            None,
        )
    if provider is None:
        _stage_error(f"unknown provider {provider_id!r}")
    return provider


def _representation(
    config: TypedAnalysisConfig, representation_id: str
) -> RepresentationSpec:
    representation = config.representations.get(representation_id)
    if representation is None:
        representation = next(
            (
                candidate
                for candidate in config.representations.values()
                if candidate.representation_id == representation_id
            ),
            None,
        )
    if representation is None:
        _stage_error(f"unknown representation {representation_id!r}")
    return representation


def _market_items(params_file: Path) -> list[str]:
    params = load_params(params_file)
    items = params.get("market_symbols")
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        _stage_error("params.yaml market_symbols must be a list of ticker IDs")
    return cast("list[str]", items)


def _code_identity() -> str:
    package_root = Path(__file__).resolve().parents[2]
    barycentre_package = package_root / "_kernels/typed_barycentre"
    paths = (
        package_root / "stages/shared/typed_artifacts.py",
        package_root / "_kernels/typed_distances.py",
        package_root / "_kernels/typed_geometry.py",
        *sorted(
            barycentre_package.rglob("*.py"),
            key=lambda path: path.relative_to(package_root).as_posix(),
        ),
        package_root / "io/typed_analysis.py",
        package_root / "io/typed_providers.py",
        package_root / "io/typed_geometry_artifacts.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "kind": "parquet" if path.suffix == ".parquet" else "json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


__all__ = [
    "TypedStageError",
    "read_typed_distance_artifact",
    "run_typed_barycentre",
    "run_typed_distance",
    "run_typed_distance_roster",
]

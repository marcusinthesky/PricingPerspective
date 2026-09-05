"""DVC orchestration for reusable raw typed geometry components."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from pipeline._kernels.typed_geometry import compute_geometry_component
from pipeline.io.params import load_params
from pipeline.io.typed_analysis import (
    StatisticalDistanceSpec,
    load_typed_analysis,
    typed_analysis_parameter_identity,
)
from pipeline.io.typed_geometry_artifacts import (
    read_geometry_component_artifact,
    write_geometry_component_artifact,
)
from pipeline.io.typed_providers import load_provider_clouds

if TYPE_CHECKING:
    from pipeline.io.typed_analysis import (
        ProviderConfig,
        RepresentationSpec,
        SampleWindowSpec,
    )


class TypedGeometryStageError(ValueError):
    """Raised when the geometry component roster is invalid."""


def _stage_error(message: str) -> NoReturn:
    raise TypedGeometryStageError(message)


def _resolve(
    params_file: Path, provider_id: str, representation_id: str, geometry_id: str
) -> tuple[
    ProviderConfig, RepresentationSpec, StatisticalDistanceSpec, SampleWindowSpec | None
]:
    config = load_typed_analysis(params_file)
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
    if representation.provider_id != provider.provider_id:
        _stage_error("representation/provider identity mismatch")
    distance = config.distances.get(geometry_id)
    if not isinstance(distance, StatisticalDistanceSpec):
        _stage_error(f"{geometry_id!r} is not a statistical geometry")
    if distance.family == "wasserstein":
        _stage_error("Wasserstein has no raw reusable component")
    return provider, representation, distance, config.window_for(distance)


def _market_items(params_file: Path) -> list[str]:
    params = load_params(params_file)
    items = params.get("market_symbols")
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        _stage_error("params.yaml market_symbols must be ticker IDs")
    return items


def _code_identity() -> str:
    package_root = Path(__file__).resolve().parents[2]
    paths = (
        package_root / "stages/shared/typed_geometry_components.py",
        package_root / "_kernels/typed_geometry.py",
        package_root / "io/typed_geometry_artifacts.py",
        package_root / "io/typed_analysis.py",
        package_root / "io/typed_providers.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_typed_geometry_component(
    *,
    provider_id: str,
    representation_id: str,
    geometry_id: str,
    output_dir: Path,
    params_file: Path = Path("params.yaml"),
) -> dict[str, object]:
    """Materialize one raw energy or MMD geometry component."""
    provider, representation, distance, window = _resolve(
        params_file, provider_id, representation_id, geometry_id
    )
    clouds = load_provider_clouds(
        provider,
        representation,
        item_ids=_market_items(params_file),
        sample_size=distance.sample_size,
        window=window,
    )
    component = compute_geometry_component(
        clouds, distance, load_typed_analysis(params_file).ground_distance_for(distance)
    )
    parameter_identity = typed_analysis_parameter_identity(
        params_file,
        provider_ids=(provider_id,),
        representation_ids=(representation_id,),
        distance_ids=(geometry_id,),
    )
    return write_geometry_component_artifact(
        output_dir,
        component,
        parameter_identity=parameter_identity,
        code_identity=_code_identity(),
    )


def run_typed_geometry_component_roster(
    *,
    specs: list[dict[str, str]],
    output_root: Path,
    params_file: Path = Path("params.yaml"),
) -> None:
    """Materialize the explicit ragged component roster."""
    if not specs:
        _stage_error("typed geometry component roster must not be empty")
    for spec in specs:
        required = {"provider_id", "representation_id", "geometry_id"}
        if set(spec) != required:
            _stage_error(
                "typed geometry component specs must contain provider_id, "
                "representation_id, and geometry_id"
            )
        output_dir = (
            output_root
            / spec["provider_id"]
            / spec["representation_id"]
            / spec["geometry_id"]
        )
        if all(
            (output_dir / filename).is_file()
            for filename in ("summary.json", "provenance.manifest.json")
        ) and (
            (output_dir / "mean_ground_distances.parquet").is_file()
            or (output_dir / "kernel_means.parquet").is_file()
        ):
            try:
                read_geometry_component_artifact(
                    output_dir,
                    expected_identity={
                        "provider_id": spec["provider_id"],
                        "representation_id": spec["representation_id"],
                        "geometry_id": spec["geometry_id"],
                    },
                )
            except (OSError, KeyError, TypeError, ValueError):
                pass
            else:
                continue
        run_typed_geometry_component(
            provider_id=spec["provider_id"],
            representation_id=spec["representation_id"],
            geometry_id=spec["geometry_id"],
            output_dir=output_dir,
            params_file=params_file,
        )


__all__ = [
    "TypedGeometryStageError",
    "run_typed_geometry_component",
    "run_typed_geometry_component_roster",
]

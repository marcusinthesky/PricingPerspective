"""Pure reusable geometry components for typed statistical stages.

This module owns cloud construction and the raw pairwise components consumed by
both statistical-distance and target-projection solvers.  It deliberately has
no knowledge of paths, YAML, Parquet, or DVC artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import jax.numpy as jnp
import numpy as np
from jcor.discrepancy._mmd.components import _compute_energy_components
from jcor.discrepancy.mmd import BUILTIN_MMD_KERNELS
from jcor.ground.config import resolve_ground_distance
from jcor.ground.metrics import cdist
from jcor.ground.similarities import gram, rbf_kernel

GeometryComponentKind = Literal["mean_ground_distance", "kernel_mean"]
GroundMetric = Literal["euclidean", "cosine", "angular"]
RANK_TWO = 2
MINIMUM_CLOUDS = 2
MEDIAN_BANDWIDTH_MAX_POINTS = 2048


class TypedGeometryError(ValueError):
    """Raised when an in-memory typed geometry input is incompatible."""


# The old name remains an import-compatible alias for callers that only need to
# catch the shared validation boundary.  New code should use TypedGeometryError.
TypedDistanceError = TypedGeometryError


def _geometry_error(message: str) -> NoReturn:
    raise TypedGeometryError(message)


@dataclass(frozen=True)
class EmbeddingCloud:
    """One provider cloud with stable identity and representation metadata."""

    item_id: str
    values: np.ndarray
    provider_id: str
    representation_id: str
    representation_transform: str = "identity"
    representation_dimension: int | None = None
    source_dimension: int | None = None
    native_dimension: int | None = None
    realized_dimension: int | None = None
    source_hash: str | None = None
    provider_model_id: str | None = None
    provider_vintage: str | None = None
    provider_status: str | None = None
    provider_caveat: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != RANK_TWO or values.shape[0] == 0 or values.shape[1] == 0:
            _geometry_error(f"{self.item_id}: cloud must be a nonempty rank-2 array")
        if not np.all(np.isfinite(values)):
            _geometry_error(f"{self.item_id}: cloud contains non-finite values")
        source_dimension = self.source_dimension or values.shape[1]
        native_dimension = self.native_dimension or source_dimension
        realized_dimension = self.realized_dimension or values.shape[1]
        if source_dimension != native_dimension:
            _geometry_error(f"{self.item_id}: source/native dimensions disagree")
        if realized_dimension != values.shape[1]:
            _geometry_error(f"{self.item_id}: realized dimension disagrees with values")
        if realized_dimension > native_dimension:
            _geometry_error(f"{self.item_id}: realized dimension exceeds native width")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_dimension", source_dimension)
        object.__setattr__(self, "native_dimension", native_dimension)
        object.__setattr__(self, "realized_dimension", realized_dimension)


def apply_representation(
    values: np.ndarray, representation: RepresentationSpec
) -> np.ndarray:
    """Apply a declared representation transform without repairing inputs."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    if array.ndim != RANK_TWO or array.shape[0] == 0:
        _geometry_error("representation input must be a nonempty rank-2 array")
    if not np.all(np.isfinite(array)):
        _geometry_error("representation input contains non-finite values")
    if representation.dimension is not None:
        if representation.dimension > array.shape[1]:
            _geometry_error(
                f"representation dimension {representation.dimension} exceeds input "
                f"width {array.shape[1]}"
            )
        array = array[:, : representation.dimension]
    if representation.transform == "identity":
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        _geometry_error("l2_normalize_rows requires finite non-zero rows")
    return array / norms


def make_cloud(
    item_id: str,
    values: np.ndarray,
    provider_id: str,
    representation: RepresentationSpec,
    *,
    source_hash: str | None = None,
    native_dimension: int | None = None,
    provider_model_id: str | None = None,
    provider_vintage: str | None = None,
    provider_status: str | None = None,
    provider_caveat: str | None = None,
) -> EmbeddingCloud:
    """Create a transformed cloud at the typed representation boundary."""
    if representation.provider_id != provider_id:
        _geometry_error(
            f"representation {representation.representation_id!r} belongs to "
            f"provider {representation.provider_id!r}, not {provider_id!r}"
        )
    raw = np.asarray(values)
    if raw.ndim != RANK_TWO:
        _geometry_error("provider embedding values must be rank-2")
    source_dimension = raw.shape[1]
    if native_dimension is not None and source_dimension != native_dimension:
        _geometry_error(
            f"provider source width {source_dimension} disagrees with native dimension "
            f"{native_dimension}"
        )
    transformed = apply_representation(raw, representation)
    return EmbeddingCloud(
        item_id=item_id,
        values=transformed,
        provider_id=provider_id,
        representation_id=representation.representation_id,
        representation_transform=representation.transform,
        representation_dimension=representation.dimension,
        source_dimension=source_dimension,
        native_dimension=native_dimension or source_dimension,
        realized_dimension=transformed.shape[1],
        source_hash=source_hash,
        provider_model_id=provider_model_id,
        provider_vintage=provider_vintage,
        provider_status=provider_status,
        provider_caveat=provider_caveat,
    )


def validate_clouds(clouds: tuple[EmbeddingCloud, ...]) -> None:
    """Validate the shared provider and representation identity of clouds."""
    if len(clouds) < MINIMUM_CLOUDS:
        _geometry_error("at least two embedding clouds are required")
    if len({cloud.item_id for cloud in clouds}) != len(clouds):
        _geometry_error("embedding cloud IDs must be unique")
    if len({cloud.provider_id for cloud in clouds}) != 1:
        _geometry_error("a geometry component requires one provider")
    if len({cloud.representation_id for cloud in clouds}) != 1:
        _geometry_error("a geometry component requires one representation")
    identities = {
        (
            cloud.representation_transform,
            cloud.representation_dimension,
            cloud.native_dimension,
            cloud.realized_dimension,
            cloud.provider_model_id,
            cloud.provider_vintage,
            cloud.provider_status,
            cloud.provider_caveat,
        )
        for cloud in clouds
    }
    if len(identities) != 1:
        _geometry_error("embedding clouds must share one representation transform")
    if len({cloud.values.shape[1] for cloud in clouds}) != 1:
        _geometry_error("embedding clouds must share feature width")


def compute_point_distance(
    left: np.ndarray, right: np.ndarray, spec: PointDistanceSpec
) -> np.ndarray:
    """Compute a declared point-distance matrix for two clouds."""
    if (
        left.ndim != RANK_TWO
        or right.ndim != RANK_TWO
        or left.shape[1] != right.shape[1]
    ):
        _geometry_error("point-distance inputs must be rank-2 with equal width")
    if spec.domain == "unit_sphere":
        representation = cast("RepresentationSpec", _UnitRepresentation())
        left = apply_representation(left, representation)
        right = apply_representation(right, representation)
    metric = resolve_ground_distance(cast("GroundMetric", spec.metric))
    return np.asarray(cdist(jnp.asarray(left), jnp.asarray(right), metric=metric))


def median_positive_pairwise_bandwidth(
    clouds: tuple[EmbeddingCloud, ...] | list[Any],
    *,
    ground: PointDistanceSpec | None = None,
) -> float:
    """Resolve the capped, deterministic positive-pairwise RBF bandwidth."""
    if clouds and isinstance(clouds[0], EmbeddingCloud):
        points = np.concatenate(
            [np.asarray(cast("EmbeddingCloud", cloud).values) for cloud in clouds],
            axis=0,
        )
    else:
        points = np.concatenate([np.asarray(cloud) for cloud in clouds], axis=0)
    if points.shape[0] > MEDIAN_BANDWIDTH_MAX_POINTS:
        positions = np.linspace(
            0,
            points.shape[0] - 1,
            MEDIAN_BANDWIDTH_MAX_POINTS,
            dtype=np.int64,
        )
        points = points[positions]
    if ground is None:
        norms = np.einsum("ij,ij->i", points, points)
        squared = norms[:, None] + norms[None, :] - 2.0 * (points @ points.T)
        upper = np.sqrt(np.maximum(squared[np.triu_indices(points.shape[0], k=1)], 0.0))
    else:
        pairwise = compute_point_distance(points, points, ground)
        upper = pairwise[np.triu_indices(points.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        _geometry_error("RBF median bandwidth is not positive")
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        _geometry_error("RBF median bandwidth is not positive")
    return bandwidth


def resolve_mmd_kernel(
    distance: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
    clouds: tuple[EmbeddingCloud, ...] | None = None,
    *,
    bandwidth_override: float | None = None,
) -> tuple[Any, dict[str, object]]:
    """Resolve one registry MMD kernel and publish its effective parameters."""
    kernel_name = distance.kernel
    if kernel_name is None:
        _geometry_error("MMD geometry requires a kernel")
    key = (
        f"distance_induced[{ground.metric}]" if kernel_name == "energy" else kernel_name
    )
    if kernel_name == "rbf":
        if bandwidth_override is not None:
            bandwidth = bandwidth_override
        elif distance.bandwidth_rule == "fixed":
            if distance.bandwidth is None:
                _geometry_error("fixed-bandwidth RBF MMD requires bandwidth")
            bandwidth = distance.bandwidth
        else:
            if clouds is None:
                _geometry_error("derived RBF bandwidth requires embedding clouds")
            bandwidth = median_positive_pairwise_bandwidth(clouds, ground=ground)
        kernel = rbf_kernel(bandwidth)
        return kernel, {
            "name": "rbf",
            "bandwidth": bandwidth,
            "bandwidth_rule": distance.bandwidth_rule,
        }
    kernel = BUILTIN_MMD_KERNELS.get(key)
    if kernel is None:
        _geometry_error(f"unknown MMD kernel {kernel_name!r}")
    return kernel, {"name": kernel_name}


@dataclass(frozen=True)
class GeometryComponentMetadata:
    """Identity and provenance carried by one persisted raw component."""

    component_kind: GeometryComponentKind
    geometry_id: str
    family: str
    provider_id: str
    representation_id: str
    ground_distance: str
    exponent: float
    estimator: str
    kernel: str | None
    bandwidth: float | None
    bandwidth_rule: str | None
    item_ids: tuple[str, ...]
    input_hashes: tuple[str, ...] = ()
    # Row count actually averaged per item. Constant across items for a
    # full-sample component, ragged once a window drops observations — which is
    # exactly when a consumer needs it, because "how many articles backed this
    # cell" stops being derivable from the provider file.
    item_counts: tuple[int, ...] = ()
    sample_design: str = "full_sample"
    sample_window: str | None = None
    provider_model_id: str | None = None
    provider_vintage: str | None = None
    provider_status: str | None = None
    provider_caveat: str | None = None
    representation_transform: str = "identity"
    representation_dimension: int | None = None
    source_dimension: int | None = None
    native_dimension: int | None = None
    realized_dimension: int | None = None
    input_dtype: str = "unknown"
    compute_dtype: str = "unknown"
    output_dtype: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible component metadata."""
        return {
            "component_kind": self.component_kind,
            "geometry_id": self.geometry_id,
            "family": self.family,
            "provider_id": self.provider_id,
            "representation_id": self.representation_id,
            "ground_distance": self.ground_distance,
            "exponent": self.exponent,
            "estimator": self.estimator,
            "kernel": self.kernel,
            "bandwidth": self.bandwidth,
            "bandwidth_rule": self.bandwidth_rule,
            "item_ids": list(self.item_ids),
            "input_hashes": list(self.input_hashes),
            "item_counts": list(self.item_counts),
            "sample_design": self.sample_design,
            "sample_window": self.sample_window,
            "provider_model_id": self.provider_model_id,
            "provider_vintage": self.provider_vintage,
            "provider_status": self.provider_status,
            "provider_caveat": self.provider_caveat,
            "representation_transform": self.representation_transform,
            "representation_dimension": self.representation_dimension,
            "source_dimension": self.source_dimension,
            "native_dimension": self.native_dimension,
            "realized_dimension": self.realized_dimension,
            "input_dtype": self.input_dtype,
            "compute_dtype": self.compute_dtype,
            "output_dtype": self.output_dtype,
        }


@dataclass(frozen=True)
class MeanGroundDistanceMatrix:
    """Raw ``mean(cdist(X_i, X_j) ** exponent)`` components."""

    item_ids: tuple[str, ...]
    values: np.ndarray
    metadata: GeometryComponentMetadata

    def __post_init__(self) -> None:
        _validate_component_matrix(self.item_ids, self.values, self.metadata)
        if self.metadata.component_kind != "mean_ground_distance":
            _geometry_error("mean-distance component has the wrong component kind")


@dataclass(frozen=True)
class KernelMeanMatrix:
    """Raw empirical kernel-mean Gram components."""

    item_ids: tuple[str, ...]
    values: np.ndarray
    metadata: GeometryComponentMetadata

    def __post_init__(self) -> None:
        _validate_component_matrix(self.item_ids, self.values, self.metadata)
        if self.metadata.component_kind != "kernel_mean":
            _geometry_error("kernel-mean component has the wrong component kind")


def _validate_component_matrix(
    item_ids: tuple[str, ...], values: np.ndarray, metadata: GeometryComponentMetadata
) -> None:
    matrix = np.asarray(values)
    if matrix.shape != (len(item_ids), len(item_ids)):
        _geometry_error("geometry component shape disagrees with item IDs")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.all(np.isfinite(matrix)):
        _geometry_error("geometry component must be finite and floating")
    tolerance = (
        32.0 * np.finfo(matrix.dtype).eps * max(1.0, float(np.max(np.abs(matrix))))
    )
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=tolerance):
        _geometry_error("geometry component is not symmetric")
    if tuple(item_ids) != metadata.item_ids:
        _geometry_error("geometry component item identity disagrees with metadata")


def _component_metadata(
    clouds: tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    kind: GeometryComponentKind,
    values: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> GeometryComponentMetadata:
    first = clouds[0]
    return GeometryComponentMetadata(
        component_kind=kind,
        geometry_id=distance.distance_id,
        family=distance.family,
        provider_id=first.provider_id,
        representation_id=first.representation_id,
        ground_distance=distance.ground_distance,
        exponent=distance.exponent,
        estimator=distance.estimator,
        kernel=distance.kernel,
        bandwidth=bandwidth,
        bandwidth_rule=distance.bandwidth_rule if distance.kernel == "rbf" else None,
        item_ids=tuple(cloud.item_id for cloud in clouds),
        input_hashes=tuple(
            source_hash
            for source_hash in (cloud.source_hash for cloud in clouds)
            if source_hash
        ),
        item_counts=tuple(int(cloud.values.shape[0]) for cloud in clouds),
        sample_design=distance.sample_design,
        sample_window=distance.sample_window,
        provider_model_id=first.provider_model_id,
        provider_vintage=first.provider_vintage,
        provider_status=first.provider_status,
        provider_caveat=first.provider_caveat,
        representation_transform=first.representation_transform,
        representation_dimension=first.representation_dimension,
        source_dimension=first.source_dimension,
        native_dimension=first.native_dimension,
        realized_dimension=first.realized_dimension,
        input_dtype=str(np.result_type(*(cloud.values.dtype for cloud in clouds))),
        compute_dtype=str(np.asarray(jnp.asarray(clouds[0].values)).dtype),
        output_dtype=str(values.dtype),
    )


def compute_mean_ground_distance_component(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
) -> MeanGroundDistanceMatrix:
    """Compute one reusable raw mean-ground-distance matrix."""
    items = tuple(clouds)
    validate_clouds(items)
    if distance.family != "energy":
        _geometry_error("mean-ground-distance components require energy geometry")
    if distance.estimator != "v_statistic":
        _geometry_error(
            "persisted mean-ground-distance components require V-statistics"
        )
    if ground.distance_id != distance.ground_distance:
        _geometry_error("geometry and ground-distance IDs disagree")
    metric = resolve_ground_distance(cast("GroundMetric", ground.metric))
    # JCOR's public mean_distance_matrix intentionally admits equal-size clouds
    # only.  The canonical embeddings are ragged, so use its packed component
    # kernel to preserve the same masked V-statistic semantics without padding.
    values, _ = _compute_energy_components(
        jnp.asarray(items[0].values),
        [jnp.asarray(cloud.values) for cloud in items],
        metric=metric,
        exponent=distance.exponent,
    )
    values = np.asarray(values)
    metadata = _component_metadata(items, distance, "mean_ground_distance", values)
    return MeanGroundDistanceMatrix(
        item_ids=metadata.item_ids, values=values, metadata=metadata
    )


def compute_kernel_mean_component(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
) -> KernelMeanMatrix:
    """Compute one reusable raw empirical kernel-mean Gram matrix."""
    items = tuple(clouds)
    validate_clouds(items)
    if distance.family != "mmd" or distance.estimator != "v_statistic":
        _geometry_error("kernel-mean components require V-statistic MMD geometry")
    if ground.distance_id != distance.ground_distance:
        _geometry_error("geometry and ground-distance IDs disagree")
    kernel, kernel_meta = resolve_mmd_kernel(distance, ground, items)
    arrays = [jnp.asarray(item.values) for item in items]
    dtype = np.result_type(*(item.values.dtype for item in items))
    values = np.empty((len(items), len(items)), dtype=dtype)
    if distance.kernel == "rbf":
        bandwidth = cast("float", kernel_meta["bandwidth"])
        for left in range(len(items)):
            for right in range(left, len(items)):
                distances = compute_point_distance(
                    items[left].values, items[right].values, ground
                )
                value = np.mean(np.exp(-(distances**2) / (2.0 * bandwidth**2)))
                values[left, right] = value
                values[right, left] = values[left, right]
    else:
        reference = arrays[0][0]
        for left in range(len(items)):
            for right in range(left, len(items)):
                value = jnp.mean(
                    gram(arrays[left], arrays[right], kernel, reference=reference)
                )
                values[left, right] = float(value)
                values[right, left] = values[left, right]
    metadata = _component_metadata(
        items,
        distance,
        "kernel_mean",
        values,
        bandwidth=cast("float | None", kernel_meta.get("bandwidth")),
    )
    return KernelMeanMatrix(
        item_ids=metadata.item_ids, values=values, metadata=metadata
    )


def compute_geometry_component(
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
) -> MeanGroundDistanceMatrix | KernelMeanMatrix:
    """Dispatch one statistical geometry to its raw reusable component."""
    if distance.family == "energy":
        return compute_mean_ground_distance_component(clouds, distance, ground)
    if distance.family == "mmd":
        return compute_kernel_mean_component(clouds, distance, ground)
    _geometry_error("Wasserstein has no reusable mean geometry component")


def statistical_distance_values(
    component: MeanGroundDistanceMatrix | KernelMeanMatrix,
    distance: StatisticalDistanceSpec,
) -> np.ndarray:
    """Convert raw components to the final zero-diagonal statistic matrix."""
    matrix = np.asarray(component.values)
    if component.metadata.geometry_id != distance.distance_id:
        _geometry_error("component geometry does not match statistical distance")
    if component.metadata.component_kind == "mean_ground_distance":
        values = 2.0 * matrix - np.diag(matrix)[:, None] - np.diag(matrix)[None, :]
    else:
        values = np.diag(matrix)[:, None] + np.diag(matrix)[None, :] - 2.0 * matrix
        values = np.maximum(values, 0.0)
        if distance.normalization == "halved":
            values *= 0.5
        if distance.normalization == "rooted":
            values = np.sqrt(values)
    values = np.asarray(values, dtype=matrix.dtype)
    values[np.diag_indices_from(values)] = 0.0
    return values


class _UnitRepresentation:
    """Internal minimal representation used for point-domain validation."""

    transform = "l2_normalize_rows"
    dimension: int | None = None


if TYPE_CHECKING:
    from pipeline.io.typed_analysis import (
        PointDistanceSpec,
        RepresentationSpec,
        StatisticalDistanceSpec,
    )


__all__ = [
    "EmbeddingCloud",
    "GeometryComponentKind",
    "GeometryComponentMetadata",
    "KernelMeanMatrix",
    "MeanGroundDistanceMatrix",
    "TypedDistanceError",
    "TypedGeometryError",
    "apply_representation",
    "compute_geometry_component",
    "compute_kernel_mean_component",
    "compute_mean_ground_distance_component",
    "compute_point_distance",
    "make_cloud",
    "median_positive_pairwise_bandwidth",
    "resolve_mmd_kernel",
    "statistical_distance_values",
    "validate_clouds",
]

"""Pure final statistical-distance transforms.

Reusable raw cloud geometry lives in :mod:`pipeline._kernels.typed_geometry`.
This module keeps the public in-memory distance API and the final distance
matrix metadata, while persisted stages can convert precomputed components
without touching the clouds again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
from jcor.discrepancy.energy import energy_functional_matrix
from jcor.discrepancy.exact import (
    balanced_wasserstein_from_cost_jax,
    exact_balanced_wasserstein_jax,
)
from jcor.discrepancy.mmd import mmd_matrix_values, mmd_squared_matrix

from pipeline._kernels.typed_geometry import (
    EmbeddingCloud,
    TypedDistanceError,
    apply_representation,
    compute_point_distance,
    make_cloud,
    median_positive_pairwise_bandwidth,
    resolve_mmd_kernel,
    statistical_distance_values,
    validate_clouds,
)

RANK_TWO = 2


@jax.jit
def _exact_wasserstein_pair_jax(
    left: jax.Array,
    right: jax.Array,
    p: float,
) -> jax.Array:
    """Run the fixed-shape sphere-chord Hungarian core without a host round trip."""
    return exact_balanced_wasserstein_jax(left, right, p)


if TYPE_CHECKING:
    from pipeline._kernels.typed_geometry import (
        KernelMeanMatrix,
        MeanGroundDistanceMatrix,
    )
    from pipeline.io.typed_analysis import (
        PointDistanceSpec,
        StatisticalDistanceSpec,
    )


def _distance_error(message: str) -> NoReturn:
    raise TypedDistanceError(message)


@dataclass(frozen=True)
class DistanceMetadata:
    """Serializable provenance for a realized distance matrix."""

    distance_id: str
    family: str
    provider_id: str
    representation_id: str
    ground_distance: str | None
    estimator: str
    normalization: str
    sample_design: str
    value_semantics: str
    sample_size: int | None = None
    kernel: str | None = None
    bandwidth: float | None = None
    bandwidth_rule: str | None = None
    provider_model_id: str | None = None
    provider_vintage: str | None = None
    provider_status: str | None = None
    provider_caveat: str | None = None
    representation_transform: str = "identity"
    representation_dimension: int | None = None
    source_dimension: int | None = None
    native_dimension: int | None = None
    realized_dimension: int | None = None
    input_hashes: tuple[str, ...] = ()
    code_identity: str | None = None
    input_dtype: str = "unknown"
    compute_dtype: str = "unknown"
    output_dtype: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible distance metadata."""
        return {
            "distance_id": self.distance_id,
            "family": self.family,
            "provider_id": self.provider_id,
            "provider_model_id": self.provider_model_id,
            "provider_vintage": self.provider_vintage,
            "provider_status": self.provider_status,
            "provider_caveat": self.provider_caveat,
            "representation_id": self.representation_id,
            "representation_transform": self.representation_transform,
            "representation_dimension": self.representation_dimension,
            "source_dimension": self.source_dimension,
            "native_dimension": self.native_dimension,
            "realized_dimension": self.realized_dimension,
            "ground_distance": self.ground_distance,
            "estimator": self.estimator,
            "normalization": self.normalization,
            "value_semantics": self.value_semantics,
            "sample_design": self.sample_design,
            "sample_size": self.sample_size,
            "kernel": self.kernel,
            "bandwidth": self.bandwidth,
            "bandwidth_rule": self.bandwidth_rule,
            "input_hashes": list(self.input_hashes),
            "code_identity": self.code_identity,
            "input_dtype": self.input_dtype,
            "compute_dtype": self.compute_dtype,
            "output_dtype": self.output_dtype,
        }


@dataclass(frozen=True)
class DistanceMatrixResult:
    """Realized symmetric matrix plus its typed computation metadata."""

    item_ids: tuple[str, ...]
    values: np.ndarray
    metadata: DistanceMetadata

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values)
        n = len(self.item_ids)
        if matrix.shape != (n, n):
            _distance_error(f"distance matrix shape {matrix.shape} != {(n, n)}")
        if not np.all(np.isfinite(matrix)):
            _distance_error("distance matrix contains non-finite values")
        tolerance = _invariant_tolerance(matrix.dtype, matrix)
        if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=tolerance):
            _distance_error("distance matrix is not symmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=tolerance, rtol=0.0):
            _distance_error("distance matrix diagonal must be zero")
        object.__setattr__(self, "values", matrix)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible result metadata and inventory."""
        return {
            "item_ids": list(self.item_ids),
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "metadata": self.metadata.as_dict(),
        }


def _metadata(
    clouds: tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    values: np.ndarray,
) -> DistanceMetadata:
    input_dtype = np.result_type(*(cloud.values.dtype for cloud in clouds))
    compute_dtype = np.asarray(jnp.asarray(clouds[0].values)).dtype
    return DistanceMetadata(
        distance_id=distance.distance_id,
        family=distance.family,
        provider_id=clouds[0].provider_id,
        provider_model_id=clouds[0].provider_model_id,
        provider_vintage=clouds[0].provider_vintage,
        provider_status=clouds[0].provider_status,
        provider_caveat=clouds[0].provider_caveat,
        representation_id=clouds[0].representation_id,
        representation_transform=clouds[0].representation_transform,
        representation_dimension=clouds[0].representation_dimension,
        source_dimension=clouds[0].source_dimension,
        native_dimension=clouds[0].native_dimension,
        realized_dimension=clouds[0].realized_dimension,
        ground_distance=distance.ground_distance,
        estimator=distance.estimator,
        normalization=distance.normalization,
        sample_design=distance.sample_design,
        value_semantics=distance.value_semantics,
        sample_size=distance.sample_size,
        kernel=distance.kernel,
        bandwidth=distance.bandwidth,
        bandwidth_rule=distance.bandwidth_rule if distance.kernel == "rbf" else None,
        input_hashes=tuple(
            source_hash
            for source_hash in (cloud.source_hash for cloud in clouds)
            if source_hash
        ),
        input_dtype=str(input_dtype),
        compute_dtype=str(compute_dtype),
        output_dtype=str(values.dtype),
    )


def compute_statistical_distance(  # noqa: C901, PLR0912, PLR0915
    clouds: list[EmbeddingCloud] | tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
) -> DistanceMatrixResult:
    """Compute one declared energy, MMD, or balanced-Wasserstein matrix."""
    items = tuple(clouds)
    validate_clouds(items)
    if ground.distance_id != distance.ground_distance:
        _distance_error(
            f"{distance.distance_id!r} declares ground distance "
            f"{distance.ground_distance!r}, received {ground.distance_id!r}"
        )
    arrays = [jnp.asarray(cloud.values) for cloud in items]
    if distance.family == "energy":
        if distance.estimator != "v_statistic":
            _distance_error("matrix energy requires the V-statistic")
        values = np.asarray(
            energy_functional_matrix(
                arrays, exponent=distance.exponent, metric=ground.metric
            ).values
        )
    elif distance.family == "mmd":
        if distance.kernel is None:
            _distance_error("MMD matrix requires a kernel")
        if distance.kernel == "rbf":
            bandwidth = (
                distance.bandwidth
                if distance.bandwidth_rule == "fixed"
                else median_positive_pairwise_bandwidth(items, ground=ground)
            )
            if bandwidth is None:
                _distance_error("fixed-bandwidth RBF MMD requires bandwidth")
            pairwise = compute_point_distance(
                np.concatenate([cloud.values for cloud in items], axis=0),
                np.concatenate([cloud.values for cloud in items], axis=0),
                ground,
            )
            kernel_matrix = np.exp(-(pairwise**2) / (2.0 * bandwidth**2))
            boundaries = np.cumsum([0, *(cloud.values.shape[0] for cloud in items)])
            squared_values = np.zeros(
                (len(items), len(items)), dtype=kernel_matrix.dtype
            )
            for left in range(len(items)):
                for right in range(left + 1, len(items)):
                    left_slice = slice(boundaries[left], boundaries[left + 1])
                    right_slice = slice(boundaries[right], boundaries[right + 1])
                    value = (
                        np.mean(kernel_matrix[left_slice, left_slice])
                        + np.mean(kernel_matrix[right_slice, right_slice])
                        - 2.0 * np.mean(kernel_matrix[left_slice, right_slice])
                    )
                    squared_values[left, right] = value
                    squared_values[right, left] = value
            values = np.sqrt(np.maximum(squared_values, 0.0))
            if distance.normalization != "rooted":
                values = np.maximum(squared_values, 0.0)
                if distance.normalization == "halved":
                    values *= 0.5
                values[np.diag_indices_from(values)] = 0.0
        else:
            kernel, _ = resolve_mmd_kernel(distance, ground)
            if distance.normalization == "rooted":
                values = np.asarray(mmd_matrix_values(arrays, kernel=kernel))
            else:
                values = np.asarray(mmd_squared_matrix(arrays, kernel=kernel))
                values = np.maximum(values, 0.0)
                if distance.normalization == "halved":
                    values *= 0.5
                values[np.diag_indices_from(values)] = 0.0
    elif distance.family == "wasserstein":
        if any(cloud.values.shape[0] != items[0].values.shape[0] for cloud in items):
            _distance_error(
                "balanced Wasserstein comparison requires equal cloud sizes"
            )
        output_dtype = np.result_type(*(cloud.values.dtype for cloud in items))
        values = np.zeros((len(items), len(items)), dtype=output_dtype)
        # The transport order is carried by the estimator, not by `exponent`:
        # see `BALANCED_WASSERSTEIN_ESTIMATORS` in `pipeline.io.typed_analysis`.
        # W2 assigns on the squared ground cost, so the order cannot be applied
        # to a matched cost vector after a W1 solve.
        order_two = distance.estimator == "balanced_wasserstein_2"
        order = 2.0 if order_two else 1.0
        if ground.metric == "euclidean" and ground.domain == "unit_sphere":
            for left in range(len(items)):
                for right in range(left + 1, len(items)):
                    value = _exact_wasserstein_pair_jax(
                        jnp.asarray(items[left].values),
                        jnp.asarray(items[right].values),
                        order,
                    )
                    values[left, right] = float(value)
                    values[right, left] = values[left, right]
        else:
            for left in range(len(items)):
                for right in range(left + 1, len(items)):
                    costs = compute_point_distance(
                        items[left].values, items[right].values, ground
                    )
                    values[left, right] = float(
                        balanced_wasserstein_from_cost_jax(
                            jnp.asarray(costs),
                            order,
                        )
                    )
                    values[right, left] = values[left, right]
    else:  # pragma: no cover - Pydantic prevents this branch
        _distance_error(f"unsupported statistical family {distance.family!r}")
    return DistanceMatrixResult(
        item_ids=tuple(cloud.item_id for cloud in items),
        values=values,
        metadata=_metadata(items, distance, values),
    )


def statistical_distance_from_components(
    component: MeanGroundDistanceMatrix | KernelMeanMatrix,
    distance: StatisticalDistanceSpec,
) -> DistanceMatrixResult:
    """Build a final distance matrix from one persisted raw component."""
    values = statistical_distance_values(component, distance)
    metadata = DistanceMetadata(
        distance_id=distance.distance_id,
        family=distance.family,
        provider_id=component.metadata.provider_id,
        representation_id=component.metadata.representation_id,
        ground_distance=component.metadata.ground_distance,
        estimator=component.metadata.estimator,
        normalization=distance.normalization,
        sample_design=distance.sample_design,
        value_semantics=distance.value_semantics,
        kernel=component.metadata.kernel,
        bandwidth=component.metadata.bandwidth,
        bandwidth_rule=component.metadata.bandwidth_rule,
        provider_model_id=component.metadata.provider_model_id,
        provider_vintage=component.metadata.provider_vintage,
        provider_status=component.metadata.provider_status,
        provider_caveat=component.metadata.provider_caveat,
        representation_transform=component.metadata.representation_transform,
        representation_dimension=component.metadata.representation_dimension,
        source_dimension=component.metadata.source_dimension,
        native_dimension=component.metadata.native_dimension,
        realized_dimension=component.metadata.realized_dimension,
        input_hashes=component.metadata.input_hashes,
        input_dtype=component.metadata.input_dtype,
        compute_dtype=component.metadata.compute_dtype,
        output_dtype=str(values.dtype),
    )
    return DistanceMatrixResult(
        item_ids=component.item_ids, values=values, metadata=metadata
    )


def _invariant_tolerance(dtype: np.dtype[Any], values: np.ndarray) -> float:
    """Scale structural checks to realized floating precision and magnitude."""
    if not np.issubdtype(dtype, np.floating):
        _distance_error("distance matrix must have a floating dtype")
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(32.0 * np.finfo(dtype).eps * scale)


__all__ = [
    "DistanceMatrixResult",
    "DistanceMetadata",
    "EmbeddingCloud",
    "TypedDistanceError",
    "apply_representation",
    "compute_point_distance",
    "compute_statistical_distance",
    "make_cloud",
    "statistical_distance_from_components",
]

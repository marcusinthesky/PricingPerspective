"""Contracts for reusable raw typed geometry components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pipeline._kernels.typed_distances import (
    compute_statistical_distance,
    statistical_distance_from_components,
)
from pipeline._kernels.typed_geometry import (
    compute_kernel_mean_component,
    compute_mean_ground_distance_component,
    make_cloud,
)

if TYPE_CHECKING:
    from typing import Literal

    from pipeline._kernels.typed_geometry import EmbeddingCloud
from pipeline.io.typed_analysis import (
    PointDistanceSpec,
    RepresentationSpec,
    StatisticalDistanceSpec,
)

GROUND = PointDistanceSpec(
    distance_id="euclidean_chord", metric="euclidean", domain="unit_sphere"
)
REPRESENTATION = RepresentationSpec(
    representation_id="unit",
    provider_id="provider",
    transform="l2_normalize_rows",
    normalization="unit_rows",
)


def _numpy_point_distances(
    left: np.ndarray, right: np.ndarray, ground: PointDistanceSpec
) -> np.ndarray:
    """Independent point-pair oracle, including the declared sphere transform."""
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    if ground.domain == "unit_sphere" or ground.metric != "euclidean":
        left = left / np.linalg.norm(left, axis=1, keepdims=True)
        right = right / np.linalg.norm(right, axis=1, keepdims=True)
    if ground.metric == "euclidean":
        return np.linalg.norm(left[:, None] - right[None, :], axis=2)
    cosine = np.sum(left[:, None] * right[None, :], axis=2)
    if ground.metric == "cosine":
        return 1.0 - cosine
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _numpy_kernel_means(
    clouds: tuple[EmbeddingCloud, ...],
    distance: StatisticalDistanceSpec,
    ground: PointDistanceSpec,
) -> tuple[np.ndarray, float | None]:
    """Compute raw means without Statadic or pipeline numerical routines."""
    arrays = [cloud.values for cloud in clouds]
    bandwidth = distance.bandwidth
    if distance.kernel == "rbf" and distance.bandwidth_rule != "fixed":
        points = np.concatenate(arrays)
        pairwise = _numpy_point_distances(points, points, ground)
        upper = pairwise[np.triu_indices(len(points), k=1)]
        bandwidth = float(np.median(upper[upper > 0.0]))
    reference = arrays[0][:1]
    values = np.empty((len(arrays), len(arrays)))
    for i, left in enumerate(arrays):
        for j, right in enumerate(arrays):
            if distance.kernel == "linear":
                # The mean-feature identity is independent of Gram reduction.
                value = np.mean(left.astype(np.float64), axis=0) @ np.mean(
                    right.astype(np.float64), axis=0
                )
            elif distance.kernel == "energy":
                value = (
                    np.mean(_numpy_point_distances(left, reference, ground))
                    + np.mean(_numpy_point_distances(right, reference, ground))
                    - np.mean(_numpy_point_distances(left, right, ground))
                )
            else:
                assert bandwidth is not None
                point_distances = _numpy_point_distances(left, right, ground)
                value = np.mean(np.exp(-(point_distances**2) / (2 * bandwidth**2)))
            values[i, j] = value
    return values, bandwidth


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("kernel", "metric", "domain", "derived"),
    [
        ("linear", "euclidean", "euclidean", False),
        ("linear", "angular", "unit_sphere", False),
        ("energy", "euclidean", "euclidean", False),
        ("energy", "angular", "unit_sphere", False),
        *[
            ("rbf", metric, domain, derived)
            for metric, domain in (
                ("euclidean", "euclidean"),
                ("euclidean", "unit_sphere"),
                ("cosine", "unit_sphere"),
                ("angular", "unit_sphere"),
            )
            for derived in (False, True)
        ],
    ],
)
def test_ragged_kernel_means_match_independent_oracle(
    dtype: type[np.floating],
    kernel: Literal["linear", "energy", "rbf"],
    metric: Literal["euclidean", "angular", "cosine"],
    domain: Literal["euclidean", "unit_sphere"],
    *,
    derived: bool,
) -> None:
    """Raw means preserve diagonal, reference, ground, bandwidth and identity."""
    representation = RepresentationSpec(
        representation_id="raw", provider_id="provider", normalization="raw"
    )
    # Axis directions avoid arccos conditioning at approximate unit inner
    # products. Unequal lengths still exercise explicit sphere normalization.
    clouds = tuple(
        make_cloud(str(i), np.asarray(values, dtype=dtype), "provider", representation)
        for i, values in enumerate(
            (
                [[3, 0], [0, 2]],
                [[-4, 0]],
                [[0, -3], [2, 0], [0, 4]],
                [[-2, 0], [0, -4]],
            )
        )
    )
    ground = PointDistanceSpec(distance_id="ground", metric=metric, domain=domain)
    distance = StatisticalDistanceSpec(
        distance_id="mmd",
        family="mmd",
        ground_distance="ground",
        exponent=1.0,
        estimator="v_statistic",
        kernel=kernel,
        bandwidth=0.8 if kernel == "rbf" and not derived else None,
        bandwidth_rule="median_positive_pairwise" if derived else "fixed",
    )
    expected, bandwidth = _numpy_kernel_means(clouds, distance, ground)
    component = compute_kernel_mean_component(clouds, distance, ground)
    # At most nine terms are averaged; account for float32 computation even
    # when an x64-disabled caller supplied float64 arrays.
    tolerance = float(16 * np.finfo(component.metadata.compute_dtype).eps)
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(
        component.values, expected, atol=tolerance * scale, rtol=tolerance
    )
    np.testing.assert_array_equal(component.values, component.values.T)
    assert np.any(np.diag(component.values) > 0)
    assert component.values.dtype == dtype
    assert component.item_ids == ("0", "1", "2", "3")
    assert component.metadata.item_counts == (2, 1, 3, 2)
    assert component.metadata.input_dtype == np.dtype(dtype).name
    assert component.metadata.output_dtype == np.dtype(dtype).name
    if bandwidth is None:
        assert component.metadata.bandwidth is None
    else:
        assert component.metadata.bandwidth == pytest.approx(bandwidth, rel=tolerance)
    final = statistical_distance_from_components(component, distance)
    expected_final = np.maximum(
        np.diag(expected)[:, None] + np.diag(expected)[None, :] - 2 * expected, 0
    )
    np.testing.assert_allclose(
        final.values, expected_final, atol=4 * tolerance * scale, rtol=tolerance
    )


def _clouds() -> tuple[EmbeddingCloud, ...]:
    values = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [1.0, 1.0]]),
        np.asarray([[0.0, 1.0], [1.0, 1.0]]),
    )
    return tuple(
        make_cloud(str(index), value, "provider", REPRESENTATION)
        for index, value in enumerate(values)
    )


def test_mean_component_retains_nonzero_raw_diagonal_and_final_parity() -> None:
    """Raw V-statistic terms retain diagonal means; the final distance does not."""
    distance = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )
    component = compute_mean_ground_distance_component(_clouds(), distance, GROUND)

    assert np.any(np.diag(component.values) > 0.0)
    fresh = compute_statistical_distance(_clouds(), distance, GROUND)
    from_components = statistical_distance_from_components(component, distance)
    np.testing.assert_allclose(from_components.values, fresh.values, atol=1e-6)


def test_kernel_component_reuses_raw_means_for_final_mmd() -> None:
    """Kernel-mean components reproduce the declared linear MMD matrix."""
    distance = StatisticalDistanceSpec(
        distance_id="linear_mmd",
        family="mmd",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
        kernel="linear",
    )
    component = compute_kernel_mean_component(_clouds(), distance, GROUND)
    fresh = compute_statistical_distance(_clouds(), distance, GROUND)
    from_components = statistical_distance_from_components(component, distance)
    np.testing.assert_allclose(from_components.values, fresh.values, atol=1e-6)


def test_ragged_raw_components_preserve_cardinality_order_and_metadata() -> None:
    """The public Statadic path keeps raw V means and pipeline-owned identities."""
    representation = RepresentationSpec(
        representation_id="raw", provider_id="provider", normalization="raw"
    )
    clouds = tuple(
        make_cloud(
            item_id,
            np.asarray(values, dtype=np.float32)[:, None],
            "provider",
            representation,
            source_hash=item_id * 64,
            provider_model_id="model",
            provider_vintage="2026-10",
        )
        for item_id, values in (("c", [1, 3, 5]), ("a", [0, 2]), ("b", [4]))
    )
    ground = PointDistanceSpec(
        distance_id="euclidean", metric="euclidean", domain="euclidean"
    )
    distance = StatisticalDistanceSpec(
        distance_id="energy_raw",
        family="energy",
        ground_distance="euclidean",
        exponent=1.0,
        estimator="v_statistic",
    )
    component = compute_mean_ground_distance_component(clouds, distance, ground)
    expected = np.asarray(
        [
            [
                np.mean(
                    [abs(float(x[0] - y[0])) for x in left.values for y in right.values]
                )
                for right in clouds
            ]
            for left in clouds
        ]
    )
    np.testing.assert_allclose(component.values, expected, rtol=1e-6)
    np.testing.assert_array_equal(component.values, component.values.T)
    np.testing.assert_allclose(np.diag(component.values), [16 / 9, 1, 0], rtol=1e-6)
    assert component.item_ids == ("c", "a", "b")
    assert component.metadata.item_counts == (3, 2, 1)
    assert component.metadata.input_hashes == ("c" * 64, "a" * 64, "b" * 64)
    assert component.metadata.provider_model_id == "model"
    assert component.metadata.provider_vintage == "2026-10"
    assert component.metadata.representation_id == "raw"
    assert component.metadata.geometry_id == "energy_raw"
    assert component.metadata.component_kind == "mean_ground_distance"
    assert component.metadata.input_dtype == "float32"
    assert component.metadata.compute_dtype == "float32"
    assert component.metadata.output_dtype == "float32"
    final = statistical_distance_from_components(component, distance)
    np.testing.assert_allclose(
        final.values,
        2 * expected - np.diag(expected)[:, None] - np.diag(expected)[None, :],
        atol=1e-6,
    )


@pytest.mark.parametrize("metric", ["angular", "cosine"])
def test_raw_directional_components_keep_superunit_power(
    metric: Literal["angular", "cosine"],
) -> None:
    """Raw components retain registry-admitted powers above energy theorem range."""
    clouds = tuple(
        make_cloud(str(index), values, "provider", REPRESENTATION)
        for index, values in enumerate(
            (
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.asarray([[-1.0, 0.0]], dtype=np.float32),
            )
        )
    )
    ground = PointDistanceSpec(
        distance_id="direction", metric=metric, domain="unit_sphere"
    )
    distance = StatisticalDistanceSpec(
        distance_id="energy_direction",
        family="energy",
        ground_distance="direction",
        exponent=1.5,
        estimator="v_statistic",
    )
    component = compute_mean_ground_distance_component(clouds, distance, ground)
    cross_distances = np.asarray([np.pi, np.pi / 2] if metric == "angular" else [2, 1])
    assert np.isfinite(component.values).all()
    assert component.values[0, 1] == pytest.approx(
        np.mean(cross_distances**1.5), rel=1e-6
    )
    assert component.metadata.exponent == distance.exponent
    assert component.metadata.item_counts == (2, 1)

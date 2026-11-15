"""Manifest and identity checks for persisted raw geometry components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pipeline._kernels.typed_geometry import (
    compute_mean_ground_distance_component,
    make_cloud,
)
from pipeline.io.typed_analysis import (
    PointDistanceSpec,
    RepresentationSpec,
    StatisticalDistanceSpec,
)
from pipeline.io.typed_geometry_artifacts import (
    TypedGeometryArtifactError,
    read_geometry_component_artifact,
    write_geometry_component_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_component_artifact_round_trip_and_identity_guard(tmp_path: Path) -> None:
    """Separate raw output names and manifest identity survive a round trip."""
    representation = RepresentationSpec(
        representation_id="unit",
        provider_id="provider",
        transform="l2_normalize_rows",
        normalization="unit_rows",
    )
    clouds = tuple(
        make_cloud(str(index), values, "provider", representation)
        for index, values in enumerate(
            (np.eye(2), np.asarray([[1.0, 0.0], [1.0, 1.0]]))
        )
    )
    distance = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="ground",
        exponent=1.0,
        estimator="v_statistic",
    )
    ground = PointDistanceSpec(
        distance_id="ground", metric="euclidean", domain="unit_sphere"
    )
    component = compute_mean_ground_distance_component(clouds, distance, ground)
    output = tmp_path / "energy_v"
    write_geometry_component_artifact(
        output, component, parameter_identity="p" * 64, code_identity="c" * 64
    )

    loaded = read_geometry_component_artifact(
        output,
        expected_identity={
            "provider_id": "provider",
            "representation_id": "unit",
            "geometry_id": "energy_v",
            "parameter_identity": "p" * 64,
            "code_identity": "c" * 64,
        },
    )
    np.testing.assert_allclose(loaded.values, component.values)
    assert (output / "mean_ground_distances.parquet").is_file()
    assert not (output / "kernel_means.parquet").exists()

    with pytest.raises(TypedGeometryArtifactError, match="expected geometry_id"):
        read_geometry_component_artifact(
            output,
            expected_identity={"geometry_id": "wrong"},
        )


def test_ragged_component_artifact_preserves_raw_values_and_provenance(
    tmp_path: Path,
) -> None:
    """Unequal-size components persist their self-means and ordered sample identity."""
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
        for item_id, values in (("a", [0, 2]), ("b", [4]), ("c", [1, 3, 5]))
    )
    distance = StatisticalDistanceSpec(
        distance_id="ragged_energy",
        family="energy",
        ground_distance="euclidean",
        exponent=1.0,
        estimator="v_statistic",
    )
    ground = PointDistanceSpec(
        distance_id="euclidean", metric="euclidean", domain="euclidean"
    )
    component = compute_mean_ground_distance_component(clouds, distance, ground)
    output = tmp_path / "ragged_energy"
    summary = write_geometry_component_artifact(
        output, component, parameter_identity="p" * 64, code_identity="c" * 64
    )
    loaded = read_geometry_component_artifact(
        output,
        expected_identity={
            "provider_id": "provider",
            "representation_id": "raw",
            "geometry_id": "ragged_energy",
            "parameter_identity": "p" * 64,
            "code_identity": "c" * 64,
        },
    )
    expected = [[1, 3, 7 / 3], [3, 0, 5 / 3], [7 / 3, 5 / 3, 16 / 9]]
    np.testing.assert_allclose(loaded.values, expected, rtol=1e-6)
    assert loaded.item_ids == ("a", "b", "c")
    assert loaded.metadata.item_counts == (2, 1, 3)
    assert loaded.metadata.as_dict() == component.metadata.as_dict()
    assert summary["metadata"] == component.metadata.as_dict()
    assert loaded.metadata.input_hashes == ("a" * 64, "b" * 64, "c" * 64)
    assert loaded.metadata.component_kind == "mean_ground_distance"
    assert (output / "mean_ground_distances.parquet").is_file()

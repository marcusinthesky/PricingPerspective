"""Kernel-level tests for typed distance and barycentre semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from pipeline._kernels.typed_barycentre import (
    solve_energy_target_projections,
    solve_mmd_target_projections,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
    solve_wasserstein_target_projections,
    solve_wasserstein_target_projections_with_supports,
)
from pipeline._kernels.typed_barycentre.wasserstein.projection import (
    _anchored_objective_and_gradient,
    _transport_permutation,
)
from pipeline._kernels.typed_distances import (
    TypedDistanceError,
    compute_point_distance,
    compute_statistical_distance,
    make_cloud,
)
from pipeline.io.typed_analysis import (
    BarycentreArm,
    PointDistanceSpec,
    RepresentationSpec,
    StatisticalDistanceSpec,
)
from statadic.geometry.discrepancy.transport.exact import balanced_wasserstein_from_cost

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pipeline._kernels.typed_distances import EmbeddingCloud

CLOUD_COUNT = 3
MATRIX_SHAPE = (CLOUD_COUNT, CLOUD_COUNT)
MATRYOSHKA_DIMENSION = 2
NATIVE_DIMENSION = 3
PARITY_EPSILON_FACTOR = 128.0
PARITY_SOLVER_FACTOR = 10.0
PARITY_SOLVER_TOLERANCE = 1e-6
SIMPLEX_WEIGHT_TOLERANCE = 1e-12
CHORD = PointDistanceSpec(
    distance_id="euclidean_chord", metric="euclidean", domain="unit_sphere"
)


def _representation(provider_id: str = "provider") -> RepresentationSpec:
    return RepresentationSpec(
        representation_id="unit",
        provider_id=provider_id,
        transform="l2_normalize_rows",
        normalization="unit_rows",
    )


def _values() -> list[NDArray[np.float64]]:
    return [
        np.asarray([[1.0, 0.0], [0.5, 0.5]]),
        np.asarray([[0.5, 0.5], [0.0, 1.0]]),
        np.asarray([[1.0, 1.0], [1.0, 0.5]]),
    ]


def _clouds(provider_id: str = "provider") -> tuple[EmbeddingCloud, ...]:
    representation = _representation(provider_id)
    return tuple(
        make_cloud(str(index), values, provider_id, representation)
        for index, values in enumerate(_values())
    )


def test_point_distance_preserves_declared_unit_sphere_domain() -> None:
    """Point distances normalize rows before applying chord geometry."""
    values = np.asarray([[2.0, 0.0]])

    result = compute_point_distance(values, values, CHORD)

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(0.0)


def test_statistical_distance_rejects_a_substituted_ground_geometry() -> None:
    """A ground spec the statistical distance never declared is not accepted."""
    distance = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )
    cosine_ground = PointDistanceSpec(
        distance_id="cosine_ground", metric="cosine", domain="unit_sphere"
    )

    with pytest.raises(TypedDistanceError, match="ground distance"):
        compute_statistical_distance(_clouds(), distance, cosine_ground)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (
            StatisticalDistanceSpec(
                distance_id="energy_v",
                family="energy",
                ground_distance="euclidean_chord",
                exponent=1.0,
                estimator="v_statistic",
                normalization="unhalved",
            ),
            "squared_statistical_distance",
        ),
        (
            StatisticalDistanceSpec(
                distance_id="linear_mmd",
                family="mmd",
                ground_distance="euclidean_chord",
                exponent=1.0,
                estimator="v_statistic",
                normalization="unhalved",
                kernel="linear",
            ),
            "squared_statistical_distance",
        ),
        (
            StatisticalDistanceSpec(
                distance_id="wasserstein_w1",
                family="wasserstein",
                ground_distance="euclidean_chord",
                exponent=1.0,
                estimator="balanced_wasserstein_1",
                normalization="rooted",
            ),
            "statistical_distance",
        ),
    ],
)
def test_distance_metadata_declares_its_persisted_scale(
    distance: StatisticalDistanceSpec, expected: str
) -> None:
    """Consumers can interpret values without branching on distance family."""
    result = compute_statistical_distance(_clouds(), distance, CHORD)

    assert result.metadata.value_semantics == expected


def test_rbf_mmd_dispatches_through_the_declared_ground_metric() -> None:
    """Angular RBF MMD must not silently reuse Euclidean pairwise distances."""
    distance = StatisticalDistanceSpec(
        distance_id="rbf",
        family="mmd",
        ground_distance="ground",
        exponent=1.0,
        estimator="v_statistic",
        kernel="rbf",
        bandwidth=0.75,
    )
    euclidean = PointDistanceSpec(
        distance_id="ground", metric="euclidean", domain="unit_sphere"
    )
    angular = PointDistanceSpec(
        distance_id="ground", metric="angular", domain="unit_sphere"
    )

    euclidean_result = compute_statistical_distance(_clouds(), distance, euclidean)
    angular_result = compute_statistical_distance(_clouds(), distance, angular)

    assert not np.allclose(euclidean_result.values, angular_result.values)
    assert euclidean_result.metadata.output_dtype == str(euclidean_result.values.dtype)


def test_float32_distance_records_realized_precision() -> None:
    """Float32 inputs remain supported and publish actual compute/output dtype."""
    representation = _representation()
    clouds = tuple(
        make_cloud(str(index), values.astype(np.float32), "provider", representation)
        for index, values in enumerate(_values())
    )
    distance = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )

    result = compute_statistical_distance(clouds, distance, CHORD)

    assert result.values.dtype == np.float32
    assert result.metadata.input_dtype == "float32"
    assert result.metadata.compute_dtype == "float32"
    assert result.metadata.output_dtype == "float32"


def test_matryoshka_identity_records_dimension_and_transform() -> None:
    """Distinct truncation arms publish collision-resistant metadata."""
    representation = RepresentationSpec(
        representation_id="arm-2d",
        provider_id="provider",
        transform="l2_normalize_rows",
        normalization="unit_rows",
        dimension=MATRYOSHKA_DIMENSION,
    )
    values = [np.pad(item, ((0, 0), (0, 1))) for item in _values()]
    clouds = tuple(
        make_cloud(str(index), item, "provider", representation)
        for index, item in enumerate(values)
    )
    distance = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )

    result = compute_statistical_distance(clouds, distance, CHORD)

    assert result.metadata.representation_id == "arm-2d"
    assert result.metadata.representation_transform == "l2_normalize_rows"
    assert result.metadata.representation_dimension == MATRYOSHKA_DIMENSION
    assert result.metadata.native_dimension == NATIVE_DIMENSION
    assert result.metadata.realized_dimension == MATRYOSHKA_DIMENSION


def test_provider_provenance_survives_the_pure_distance_boundary() -> None:
    """Exploratory provider caveats remain available to artifact writers."""
    representation = _representation()
    clouds = tuple(
        make_cloud(
            str(index),
            values,
            "provider",
            representation,
            provider_model_id="encoder-model",
            provider_vintage="V3",
            provider_status="exploratory",
            provider_caveat="look-ahead-sensitive diagnostic",
        )
        for index, values in enumerate(_values())
    )
    distance = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )

    result = compute_statistical_distance(clouds, distance, CHORD)

    assert result.metadata.provider_model_id == "encoder-model"
    assert result.metadata.provider_vintage == "V3"
    assert result.metadata.provider_status == "exploratory"
    assert result.metadata.provider_caveat == "look-ahead-sensitive diagnostic"


@pytest.mark.parametrize(
    "distance",
    [
        StatisticalDistanceSpec(
            distance_id="energy_v",
            family="energy",
            ground_distance="euclidean_chord",
            exponent=1.0,
            estimator="v_statistic",
        ),
        StatisticalDistanceSpec(
            distance_id="linear_mmd",
            family="mmd",
            ground_distance="euclidean_chord",
            exponent=1.0,
            estimator="v_statistic",
            kernel="linear",
        ),
        StatisticalDistanceSpec(
            distance_id="energy_mmd",
            family="mmd",
            ground_distance="euclidean_chord",
            exponent=1.0,
            estimator="v_statistic",
            kernel="energy",
        ),
        StatisticalDistanceSpec(
            distance_id="rbf_mmd",
            family="mmd",
            ground_distance="euclidean_chord",
            exponent=1.0,
            estimator="v_statistic",
            kernel="rbf",
            bandwidth_rule="median_positive_pairwise",
        ),
        StatisticalDistanceSpec(
            distance_id="wasserstein_w1",
            family="wasserstein",
            ground_distance="euclidean_chord",
            exponent=1.0,
            estimator="balanced_wasserstein_1",
            normalization="rooted",
        ),
    ],
)
def test_statistical_families_emit_typed_symmetric_matrices(
    distance: StatisticalDistanceSpec,
) -> None:
    """Supported families share matrix invariants without sharing estimands."""
    result = compute_statistical_distance(_clouds(), distance, CHORD)

    assert result.values.shape == MATRIX_SHAPE
    assert np.all(np.isfinite(result.values))
    assert np.allclose(result.values, result.values.T)
    assert np.allclose(np.diag(result.values), 0.0)
    assert result.metadata.distance_id == distance.distance_id
    assert result.metadata.estimator == distance.estimator


def test_energy_graph_alias_matches_distance_induced_mmd() -> None:
    """The removed graph arm has the same matrix, weights, and objective."""
    clouds = _clouds()
    energy = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )
    induced_mmd = StatisticalDistanceSpec(
        distance_id="distance_induced_mmd",
        family="mmd",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
        kernel="energy",
    )
    energy_matrix = compute_statistical_distance(clouds, energy, CHORD)
    mmd_matrix = compute_statistical_distance(clouds, induced_mmd, CHORD)
    matrix_scale = max(1.0, float(np.max(np.abs(energy_matrix.values))))
    relative_tolerance = (
        PARITY_EPSILON_FACTOR * np.finfo(energy_matrix.values.dtype).eps
    )
    absolute_tolerance = relative_tolerance * matrix_scale
    np.testing.assert_allclose(
        energy_matrix.values,
        mmd_matrix.values,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )

    energy_result = solve_energy_target_projections(
        clouds,
        energy,
        BarycentreArm(
            arm_id="energy_simplex",
            problem="target_projection",
            geometry=energy.distance_id,
            feasible_set="simplex_nonnegative",
            solver="energy_qp",
            target_policy="leave_one_out",
            tol=PARITY_SOLVER_TOLERANCE,
            maxiter=10_000,
        ),
        CHORD,
    )
    mmd_result = solve_mmd_target_projections(
        clouds,
        induced_mmd,
        BarycentreArm(
            arm_id="distance_induced_mmd_simplex",
            problem="target_projection",
            geometry=induced_mmd.distance_id,
            feasible_set="simplex_nonnegative",
            solver="mmd_qp",
            target_policy="leave_one_out",
            tol=PARITY_SOLVER_TOLERANCE,
            maxiter=10_000,
        ),
        CHORD,
    )
    assert all(row.converged for row in energy_result.diagnostics)
    assert all(row.converged for row in mmd_result.diagnostics)
    solution_tolerance = max(
        PARITY_SOLVER_FACTOR * PARITY_SOLVER_TOLERANCE, relative_tolerance
    )
    for energy_weights, mmd_weights in zip(
        energy_result.weights, mmd_result.weights, strict=True
    ):
        np.testing.assert_allclose(
            energy_weights,
            mmd_weights,
            atol=solution_tolerance,
            rtol=solution_tolerance,
        )
    np.testing.assert_allclose(
        [row.objective for row in energy_result.diagnostics],
        [row.objective for row in mmd_result.diagnostics],
        atol=solution_tolerance * matrix_scale,
        rtol=solution_tolerance,
    )


def test_distance_rejects_mixed_provider_clouds() -> None:
    """A matrix cannot silently combine provider representations."""
    clouds = list(_clouds())
    clouds[-1] = make_cloud(
        "2", _values()[-1], "other-provider", _representation("other-provider")
    )

    with pytest.raises(TypedDistanceError, match="one provider"):
        compute_statistical_distance(
            clouds,
            StatisticalDistanceSpec(
                distance_id="energy_v",
                family="energy",
                ground_distance="euclidean_chord",
                exponent=1.0,
                estimator="v_statistic",
            ),
            CHORD,
        )


def test_signed_energy_projection_has_distinct_affine_constraint() -> None:
    """An outlying target requires a negative affine-signed coefficient."""
    representation = RepresentationSpec(
        representation_id="raw", provider_id="provider", normalization="raw"
    )
    positions = (0.0, 1.0, 100.0)
    clouds = tuple(
        make_cloud(
            str(index),
            np.asarray([[position, 0.0], [position + 0.1, 0.0]]),
            "provider",
            representation,
        )
        for index, position in enumerate(positions)
    )
    geometry = StatisticalDistanceSpec(
        distance_id="energy_v",
        family="energy",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
    )
    arm = BarycentreArm(
        arm_id="energy_affine_signed",
        problem="target_projection",
        geometry="energy_v",
        feasible_set="affine_signed",
        solver="energy_qp",
        target_policy="leave_one_out",
        tol=1e-8,
        maxiter=10_000,
    )

    result = solve_energy_target_projections(clouds, geometry, arm, CHORD)

    assert all(np.isclose(np.sum(weights), 1.0) for weights in result.weights)
    assert any(np.any(weights < 0.0) for weights in result.weights)
    assert all(
        diagnostic.feasible_set == "affine_signed" for diagnostic in result.diagnostics
    )

    simplex_result = solve_energy_target_projections(
        clouds,
        geometry,
        BarycentreArm(
            arm_id="energy_simplex",
            problem="target_projection",
            geometry="energy_v",
            feasible_set="simplex_nonnegative",
            solver="energy_qp",
            target_policy="leave_one_out",
            tol=1e-8,
            maxiter=10_000,
        ),
        CHORD,
    )
    assert all(
        np.isfinite(diagnostic.projected_gradient_norm)
        and np.isfinite(diagnostic.cnd_tangent_max_eigenvalue)
        and np.isfinite(diagnostic.tangent_curvature_min)
        and np.isfinite(diagnostic.simplex_sum_error)
        and np.isfinite(diagnostic.min_weight)
        for diagnostic in simplex_result.diagnostics
    )


def test_wasserstein_measure_barycentre_preserves_source_simplex() -> None:
    """The OT measure path uses prescribed sources, not energy QP weights."""
    result = solve_wasserstein_measure_barycentre(
        _clouds()[:2],
        (0.25, 0.75),
        estimator="balanced_wasserstein_2",
    )

    assert result.source_ids == ("0", "1")
    assert result.source_weights == pytest.approx((0.25, 0.75))
    assert result.support_values.shape == (2, 2)
    assert result.converged
    assert result.objective <= result.initial_objective


def test_wasserstein_measure_barycentre_refuses_skewed_order_one() -> None:
    """Order 1 is exact only at equal weights, so skewed weights are rejected.

    The order-1 barycentre objective is linear along the geodesic, so its
    minimum sits at the source carrying the larger weight and the interpolant
    this solver returns is not a minimizer. Refusing beats publishing that
    support under ``converged=True`` with a zero optimality gap.
    """
    with pytest.raises(TypedDistanceError, match="minimized elsewhere"):
        solve_wasserstein_measure_barycentre(
            _clouds()[:2],
            (0.25, 0.75),
            estimator="balanced_wasserstein_1",
        )


def test_wasserstein_measure_barycentre_allows_balanced_order_one() -> None:
    """Equal weights make the whole order-1 geodesic optimal, so it is allowed.

    This is the case ``params.yaml`` declares for ``wasserstein_w1_measure``.
    """
    result = solve_wasserstein_measure_barycentre(
        _clouds()[:2],
        (0.5, 0.5),
        estimator="balanced_wasserstein_1",
    )

    assert result.converged
    assert result.objective_semantics == "statistical_distance"
    # Exact: the closed-form two-source solver never iterates, so the residual
    # is the literal 0.0 rather than something merely small.
    assert result.support_shift == 0.0  # noqa: RUF069


def test_free_support_barycentre_matches_exact_two_source_solver() -> None:
    """At K=2 the iterative solver must reproduce the exact geodesic objective.

    The two-source case is exactly solvable (an optimal assignment followed by
    displacement interpolation *is* the W2 geodesic barycentre), so it is the
    only ground truth available without an external oracle. Agreement there is
    the primary correctness anchor for the K-source generalization.
    """
    pair = _clouds()[:2]
    weights = (0.25, 0.75)

    exact = solve_wasserstein_measure_barycentre(
        pair, weights, estimator="balanced_wasserstein_2"
    )
    iterative = solve_wasserstein_free_support_barycentre(
        pair, weights, estimator="balanced_wasserstein_2"
    )

    # PARITY_SOLVER_TOLERANCE, not 1e-9: both solvers run through JAX, which the
    # pipeline leaves at float32, so agreement is bounded by ~1e-8 no matter how
    # exact the geometry is. A 1e-9 bound tests the dtype, not the solver.
    assert iterative.objective == pytest.approx(
        exact.objective, abs=PARITY_SOLVER_TOLERANCE
    )
    assert iterative.converged


def test_free_support_barycentre_beats_every_source_placed_support() -> None:
    """Exhaustive oracle: no source cloud, used as-is, beats the solved support.

    Stands in for an external OT oracle without taking a numpy-only dependency.
    Placing the barycentre at any single source is a feasible support, so the
    solved objective must be no worse than the best of them. This catches a
    solver that converges to something arbitrary while still reporting a
    plausible number.
    """
    clouds = _clouds()
    weights = (0.2, 0.3, 0.5)

    result = solve_wasserstein_free_support_barycentre(
        clouds, weights, estimator="balanced_wasserstein_2"
    )

    sources = tuple(np.asarray(cloud.values) for cloud in clouds)
    for candidate in sources:
        placed = 0.0
        for weight, source in zip(weights, sources, strict=True):
            cost = np.linalg.norm(candidate[:, None, :] - source[None, :, :], axis=-1)
            # Exhaustive over the 2! assignments of this fixture, so the
            # comparison never depends on the solver's own matching routine.
            best = min(
                float(np.mean([cost[0, 0] ** 2, cost[1, 1] ** 2])),
                float(np.mean([cost[0, 1] ** 2, cost[1, 0] ** 2])),
            )
            placed += weight * best
        assert result.objective <= placed + 1e-9


def test_free_support_barycentre_reaches_a_genuine_fixed_point() -> None:
    """Perturbing the converged support must not lower the objective."""
    clouds = _clouds()
    weights = (0.2, 0.3, 0.5)

    result = solve_wasserstein_free_support_barycentre(
        clouds, weights, estimator="balanced_wasserstein_2"
    )

    sources = tuple(np.asarray(cloud.values) for cloud in clouds)

    def objective(support: np.ndarray) -> float:
        total = 0.0
        for weight, source in zip(weights, sources, strict=True):
            cost = np.linalg.norm(support[:, None, :] - source[None, :, :], axis=-1)
            total += weight * min(
                float(np.mean([cost[0, 0] ** 2, cost[1, 1] ** 2])),
                float(np.mean([cost[0, 1] ** 2, cost[1, 0] ** 2])),
            )
        return total

    generator = np.random.default_rng(0)
    for _ in range(16):
        step = generator.normal(scale=1e-3, size=result.support_values.shape)
        assert objective(result.support_values + step) >= result.objective - 1e-9


def test_free_support_barycentre_of_identical_sources_is_that_source() -> None:
    """Zero transport cost, and the support is the common cloud."""
    cloud = _clouds()[0]

    result = solve_wasserstein_free_support_barycentre(
        (cloud, cloud, cloud), (1 / 3, 1 / 3, 1 / 3)
    )

    assert result.objective == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        np.sort(result.support_values, axis=0),
        np.sort(np.asarray(cloud.values), axis=0),
        atol=1e-9,
    )


def test_free_support_barycentre_collapses_at_a_degenerate_weight() -> None:
    """All mass on one source recovers that source at zero cost."""
    clouds = _clouds()

    result = solve_wasserstein_free_support_barycentre(clouds, (1.0, 0.0, 0.0))

    assert result.objective == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        np.sort(result.support_values, axis=0),
        np.sort(np.asarray(clouds[0].values), axis=0),
        atol=1e-9,
    )


def test_free_support_barycentre_never_increases_the_objective() -> None:
    """Alternating minimization is non-increasing, so it cannot regress."""
    result = solve_wasserstein_free_support_barycentre(
        _clouds(), (0.2, 0.3, 0.5), estimator="balanced_wasserstein_2"
    )

    assert result.objective <= result.initial_objective + 1e-12
    assert result.optimality_gap >= 0.0
    assert result.iterations >= 1


def test_free_support_barycentre_does_not_certify_order_one() -> None:
    """The mean update minimizes the order-2 inner problem, not the order-1 one."""
    result = solve_wasserstein_free_support_barycentre(
        _clouds(), (0.2, 0.3, 0.5), estimator="balanced_wasserstein_1"
    )

    assert not result.converged


@pytest.mark.parametrize(
    ("clouds", "weights", "message"),
    [
        (1, (1.0,), "at least two sources"),
        (3, (0.5, 0.5), "one weight per source"),
        (3, (0.5, 0.5, 0.5), "simplex"),
        (3, (-0.5, 0.75, 0.75), "simplex"),
    ],
)
def test_free_support_barycentre_rejects_malformed_inputs(
    clouds: int, weights: tuple[float, ...], message: str
) -> None:
    """Arity and simplex violations fail loudly rather than being coerced."""
    with pytest.raises(TypedDistanceError, match=message):
        solve_wasserstein_free_support_barycentre(_clouds()[:clouds], weights)


def test_free_support_barycentre_rejects_unequal_support_sizes() -> None:
    """Equal support sizes are required, as in the exact two-source solver."""
    representation = _representation()
    ragged = make_cloud("ragged", np.asarray([[1.0, 0.0]]), "provider", representation)

    with pytest.raises(TypedDistanceError, match="equal non-empty shapes"):
        solve_wasserstein_free_support_barycentre(
            (*_clouds()[:2], ragged), (1 / 3, 1 / 3, 1 / 3)
        )


@pytest.mark.parametrize("kernel", ["linear", "energy", "rbf"])
def test_mmd_projection_preserves_order_and_diagnostics(kernel: str) -> None:
    """Every declared target-projection kernel emits ordered simplex weights."""
    geometry = StatisticalDistanceSpec(
        distance_id=f"{kernel}_mmd",
        family="mmd",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
        kernel=kernel,  # type: ignore[arg-type]
        bandwidth_rule="median_positive_pairwise",
    )
    arm = BarycentreArm(
        arm_id=f"{kernel}_mmd_simplex",
        problem="target_projection",
        geometry=geometry.distance_id,
        feasible_set="simplex_nonnegative",
        solver="mmd_qp",
        target_policy="leave_one_out",
        tol=1e-6,
        maxiter=10_000,
    )

    result = solve_mmd_target_projections(_clouds(), geometry, arm, CHORD)

    assert result.target_ids == ("0", "1", "2")
    assert result.candidate_ids == (("1", "2"), ("0", "2"), ("0", "1"))
    assert all(np.sum(weights) == pytest.approx(1.0) for weights in result.weights)
    assert all(diagnostic.converged for diagnostic in result.diagnostics)
    assert all(
        np.isfinite(diagnostic.projected_gradient_norm)
        and np.isfinite(diagnostic.psd_tangent_min_eigenvalue)
        for diagnostic in result.diagnostics
    )


def test_mmd_projection_dispatches_rbf_ground_metric() -> None:
    """RBF barycentre weights consume the declared point metric."""
    geometry = StatisticalDistanceSpec(
        distance_id="rbf_mmd",
        family="mmd",
        ground_distance="ground",
        exponent=1.0,
        estimator="v_statistic",
        kernel="rbf",
        bandwidth=0.75,
    )
    arm = BarycentreArm(
        arm_id="rbf_mmd_simplex",
        problem="target_projection",
        geometry="rbf_mmd",
        feasible_set="simplex_nonnegative",
        solver="mmd_qp",
        target_policy="leave_one_out",
        tol=1e-6,
        maxiter=10_000,
    )
    euclidean = PointDistanceSpec(
        distance_id="ground", metric="euclidean", domain="unit_sphere"
    )
    angular = PointDistanceSpec(
        distance_id="ground", metric="angular", domain="unit_sphere"
    )

    euclidean_result = solve_mmd_target_projections(_clouds(), geometry, arm, euclidean)
    angular_result = solve_mmd_target_projections(_clouds(), geometry, arm, angular)

    assert any(
        not np.allclose(left, right)
        for left, right in zip(
            euclidean_result.weights, angular_result.weights, strict=True
        )
    )


def test_float32_mmd_projection_preserves_realized_dtype() -> None:
    """MMD projection does not opt into x64 or upcast float32 clouds."""
    representation = _representation()
    clouds = tuple(
        make_cloud(str(i), values.astype(np.float32), "provider", representation)
        for i, values in enumerate(_values())
    )
    geometry = StatisticalDistanceSpec(
        distance_id="linear_mmd",
        family="mmd",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="v_statistic",
        kernel="linear",
    )
    arm = BarycentreArm(
        arm_id="linear_mmd_simplex",
        problem="target_projection",
        geometry="linear_mmd",
        feasible_set="simplex_nonnegative",
        solver="mmd_qp",
        target_policy="leave_one_out",
        tol=1e-6,
        maxiter=10_000,
    )

    result = solve_mmd_target_projections(clouds, geometry, arm, CHORD)

    assert all(weights.dtype == np.float32 for weights in result.weights)


def test_non_spherical_w2_is_invariant_to_cloud_row_order() -> None:
    """Identical Euclidean clouds in different row order have zero W2."""
    representation = RepresentationSpec(
        representation_id="raw",
        provider_id="provider",
        transform="identity",
        normalization="raw",
    )
    points = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    distance = StatisticalDistanceSpec(
        distance_id="wasserstein_w2",
        family="wasserstein",
        ground_distance="euclidean",
        exponent=1.0,
        estimator="balanced_wasserstein_2",
        normalization="rooted",
    )
    ground = PointDistanceSpec(
        distance_id="euclidean", metric="euclidean", domain="euclidean"
    )
    result = compute_statistical_distance(
        (
            make_cloud("left", points, "provider", representation),
            make_cloud("right", points[[2, 0, 1]], "provider", representation),
        ),
        distance,
        ground,
    )

    assert result.values[0, 1] == pytest.approx(0.0)


def test_non_spherical_w2_uses_the_squared_cost_assignment() -> None:
    """The W2 value follows its explicit optimal row/column assignment."""
    representation = RepresentationSpec(
        representation_id="raw",
        provider_id="provider",
        transform="identity",
        normalization="raw",
    )
    left = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    right = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    distance = StatisticalDistanceSpec(
        distance_id="wasserstein_w2",
        family="wasserstein",
        ground_distance="euclidean",
        exponent=1.0,
        estimator="balanced_wasserstein_2",
        normalization="rooted",
    )
    ground = PointDistanceSpec(
        distance_id="euclidean", metric="euclidean", domain="euclidean"
    )
    result = compute_statistical_distance(
        (
            make_cloud("left", left, "provider", representation),
            make_cloud("right", right, "provider", representation),
        ),
        distance,
        ground,
    )

    # The squared-cost optimum is rows -> columns (2, 0, 1), with squared
    # costs (1, 1, 1), so the rooted mean is exactly one.
    assert result.values[0, 1] == pytest.approx(1.0)


def test_wasserstein_measure_barycentre_rejects_unknown_estimator() -> None:
    """A measure barycentre must use one of the two declared transport orders."""
    with pytest.raises(TypedDistanceError, match="unsupported Wasserstein estimator"):
        solve_wasserstein_measure_barycentre(
            _clouds()[:2],
            (0.5, 0.5),
            estimator="unsupported",
        )


def test_wasserstein_measure_barycentre_publishes_squared_w2_objective() -> None:
    """W2 metadata identifies the squared transport objective."""
    result = solve_wasserstein_measure_barycentre(
        _clouds()[:2],
        (0.5, 0.5),
        estimator="balanced_wasserstein_2",
    )

    assert result.objective_semantics == "squared_statistical_distance"


_WASSERSTEIN_W2 = StatisticalDistanceSpec(
    distance_id="wasserstein_w2",
    family="wasserstein",
    ground_distance="euclidean_chord",
    exponent=1.0,
    estimator="balanced_wasserstein_2",
    normalization="rooted",
)


def test_wasserstein_w2_emits_a_typed_symmetric_matrix() -> None:
    """Share the matrix invariants of the order-1 arm without sharing values."""
    result = compute_statistical_distance(_clouds(), _WASSERSTEIN_W2, CHORD)

    assert result.values.shape == MATRIX_SHAPE
    assert np.all(np.isfinite(result.values))
    assert np.allclose(result.values, result.values.T)
    assert np.allclose(np.diag(result.values), 0.0)
    assert result.metadata.estimator == "balanced_wasserstein_2"
    assert result.metadata.value_semantics == "statistical_distance"


def test_wasserstein_generic_ground_path_delegates_to_shared_wp_core() -> None:
    """Match Pipeline's generic ground branch to Statadic's Wp-from-cost primitive."""
    ground = PointDistanceSpec(
        distance_id="raw_euclidean",
        metric="euclidean",
        domain="euclidean",
    )
    distance = StatisticalDistanceSpec(
        distance_id="raw_wasserstein_w2",
        family="wasserstein",
        ground_distance=ground.distance_id,
        exponent=1.0,
        estimator="balanced_wasserstein_2",
        normalization="rooted",
    )
    clouds = _clouds()[:2]
    result = compute_statistical_distance(clouds, distance, ground)
    costs = compute_point_distance(clouds[0].values, clouds[1].values, ground)
    expected = balanced_wasserstein_from_cost(costs, 2.0)
    assert result.values[0, 1] == pytest.approx(
        expected,
        abs=PARITY_SOLVER_TOLERANCE,
    )


def test_wasserstein_w2_dominates_w1_dyad_by_dyad() -> None:
    """Order the two typed arms by Cauchy--Schwarz on the matched cost vector."""
    order_one = StatisticalDistanceSpec(
        distance_id="wasserstein_w1",
        family="wasserstein",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="balanced_wasserstein_1",
        normalization="rooted",
    )
    w1 = compute_statistical_distance(_clouds(), order_one, CHORD).values
    w2 = compute_statistical_distance(_clouds(), _WASSERSTEIN_W2, CHORD).values

    assert np.all(w1 <= w2 + PARITY_SOLVER_TOLERANCE)


def test_wasserstein_w2_barycentre_matches_its_closed_form() -> None:
    """Pin the two-source W2 objective to w0*w1*W2^2 between the sources.

    Displacement interpolation puts the barycentre on the W2 geodesic, so the
    objective is available in closed form and does not need a solver certificate
    to be believed.
    """
    clouds = _clouds()[:2]
    weights = (0.25, 0.75)
    result = solve_wasserstein_measure_barycentre(
        clouds, weights, estimator="balanced_wasserstein_2"
    )
    squared = (
        compute_statistical_distance(clouds, _WASSERSTEIN_W2, CHORD).values[0, 1] ** 2
    )

    assert result.objective == pytest.approx(
        weights[0] * weights[1] * squared, abs=PARITY_SOLVER_TOLERANCE
    )
    assert result.objective <= result.initial_objective
    assert result.support_values.shape == (2, 2)


@pytest.mark.parametrize(
    ("estimator", "solver"),
    [
        ("balanced_wasserstein_1", "wasserstein_w1_target"),
        ("balanced_wasserstein_2", "wasserstein_w2_target"),
    ],
)
def test_wasserstein_target_projection_is_leave_one_out_simplex(
    estimator: Literal["balanced_wasserstein_1", "balanced_wasserstein_2"],
    solver: Literal["wasserstein_w1_target", "wasserstein_w2_target"],
) -> None:
    """W1/W2 target arms publish one valid candidate simplex per target."""
    geometry = StatisticalDistanceSpec(
        distance_id=f"{estimator}_geometry",
        family="wasserstein",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator=estimator,
        normalization="rooted",
    )
    arm = BarycentreArm(
        arm_id=f"{solver}_arm",
        problem="target_projection",
        geometry=geometry.distance_id,
        feasible_set="simplex_nonnegative",
        solver=solver,
        target_policy="leave_one_out",
        tol=1e-8,
        maxiter=1000,
    )

    result = solve_wasserstein_target_projections(_clouds(), geometry, arm, CHORD)

    assert len(result.target_ids) == CLOUD_COUNT
    assert all(
        len(candidates) == CLOUD_COUNT - 1 for candidates in result.candidate_ids
    )
    assert all(
        np.all(weights >= -SIMPLEX_WEIGHT_TOLERANCE) for weights in result.weights
    )
    assert all(np.sum(weights) == pytest.approx(1.0) for weights in result.weights)
    assert all(diagnostic.converged for diagnostic in result.diagnostics)
    if estimator == "balanced_wasserstein_2":
        assert all(
            diagnostic.assignment_support_size == len(_clouds()[0].values)
            for diagnostic in result.diagnostics
        )


def test_wasserstein_target_projection_support_extension_preserves_result() -> None:
    """The support-returning API is an additive view of the legacy solve."""
    geometry = StatisticalDistanceSpec(
        distance_id="wasserstein_w2",
        family="wasserstein",
        ground_distance="euclidean_chord",
        exponent=1.0,
        estimator="balanced_wasserstein_2",
        normalization="rooted",
    )
    arm = BarycentreArm(
        arm_id="wasserstein_w2_target_arm",
        problem="target_projection",
        geometry=geometry.distance_id,
        feasible_set="simplex_nonnegative",
        solver="wasserstein_w2_target",
        target_policy="leave_one_out",
        tol=1e-8,
        maxiter=1000,
    )

    legacy = solve_wasserstein_target_projections(_clouds(), geometry, arm, CHORD)
    extended = solve_wasserstein_target_projections_with_supports(
        _clouds(), geometry, arm, CHORD
    )

    assert extended.result.target_ids == legacy.target_ids
    assert extended.result.candidate_ids == legacy.candidate_ids
    assert all(
        np.allclose(left, right)
        for left, right in zip(extended.result.weights, legacy.weights, strict=True)
    )
    assert len(extended.supports) == len(legacy.target_ids)
    expected_shape = _clouds()[0].values.shape
    assert all(support.shape == expected_shape for support in extended.supports)


def test_anchored_w2_objective_matches_direct_formula() -> None:
    """Pin the implemented quadratic objective to its aligned-support equation."""
    target = np.array([[0.0, 0.0], [1.0, 1.0]])
    candidates = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 0.0], [2.0, 1.0]],
        ]
    )
    weights = np.array([0.25, 0.75])

    objective, _ = _anchored_objective_and_gradient(
        weights, target, candidates, order_two=True
    )
    support = np.einsum("k,kmd->md", weights, candidates)
    expected = np.mean(np.sum((target - support) ** 2, axis=1))

    assert objective == pytest.approx(expected)


def test_assignment_gap_is_best_to_second_best_cost() -> None:
    """Expose the finite matching margin used by the local Lean certificate."""
    left = np.array([[0.0], [2.0]])
    right = np.array([[0.0], [3.0]])

    permutation, gap = _transport_permutation(
        left, right, order_two=True, metric="euclidean"
    )

    np.testing.assert_array_equal(permutation, [0, 1])
    assert gap == pytest.approx(12.0)

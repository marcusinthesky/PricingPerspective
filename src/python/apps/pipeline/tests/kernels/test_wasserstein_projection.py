"""Numerical certificates and independent oracles for target-anchored W1/W2."""

from __future__ import annotations

from typing import Literal, cast

import cvxpy as cp
import numpy as np
import pipeline._kernels.typed_barycentre.wasserstein.projection as projection_module
import pytest
from pipeline._kernels.typed_barycentre.wasserstein.projection import (
    _anchored_objective_and_gradient,
    _simplex_optimality_certificate,
    _solve_anchored_wasserstein_row,
    _transport_permutation,
)
from pipeline._kernels.typed_geometry import EmbeddingCloud
from scipy.optimize import OptimizeResult
from scipy.spatial.distance import cdist as scipy_cdist

CERTIFICATE_TOLERANCE = 1.0e-6
PROJECTED_GRADIENT_TOLERANCE = 5.0e-6
ORACLE_OBJECTIVE_TOLERANCE = 2.0e-8
ORACLE_WEIGHT_TOLERANCE = 2.0e-5
REFINEMENT_ATTEMPTS = 2
GLOBAL_SCALE = 7.0
SUPPORT_THRESHOLD = 1.0e-8


def _cloud(item_id: str, values: np.ndarray) -> EmbeddingCloud:
    return EmbeddingCloud(
        item_id=item_id,
        values=np.asarray(values, dtype=np.float64),
        provider_id="test-provider",
        representation_id="test-representation",
    )


def _oracle_fixture() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    target = np.array([[0.97699, -1.02841], [1.28566, 0.01793], [-0.17221, 0.32438]])
    candidates = (
        np.array([[-1.97139, -0.05635], [1.85756, 1.32817], [-0.65755, 1.44895]]),
        np.array([[-1.67652, -0.25609], [-0.38548, 0.31772], [0.11567, -0.24570]]),
        np.array([[-0.43551, -0.57214], [-0.30299, -0.18927], [-1.03531, 0.87904]]),
    )
    return target, candidates


def _matched_fixture(*, order_two: bool) -> tuple[EmbeddingCloud, np.ndarray]:
    target_values, candidates = _oracle_fixture()
    matched = []
    for candidate in candidates:
        permutation, _ = _transport_permutation(
            target_values, candidate, order_two=order_two, metric="euclidean"
        )
        matched.append(candidate[permutation])
    return _cloud("target", target_values), np.stack(matched)


def _solve_cvxpy_oracle(
    target: np.ndarray, matched: np.ndarray, *, order_two: bool
) -> tuple[np.ndarray, float]:
    weights = cp.Variable(matched.shape[0])
    residuals = [
        target[row] - matched[:, row, :].T @ weights for row in range(target.shape[0])
    ]
    if order_two:
        terms = [cp.sum_squares(residual) for residual in residuals]
    else:
        terms = [cp.norm(residual, 2) for residual in residuals]
    constraints = cast(
        "list[cp.constraints.constraint.Constraint]",
        [weights >= 0.0, cp.sum(weights) == 1.0],  # noqa: RUF069
    )
    problem = cp.Problem(cp.Minimize(sum(terms) / target.shape[0]), constraints)
    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1.0e-10,
        tol_gap_rel=1.0e-10,
        tol_feas=1.0e-10,
    )
    assert problem.status == cp.OPTIMAL
    assert weights.value is not None
    assert problem.value is not None
    return np.asarray(weights.value, dtype=np.float64), float(problem.value)


def test_simplex_certificate_handles_an_active_boundary() -> None:
    """Use constrained KKT geometry rather than a centered raw gradient."""
    weights = np.array([1.0, 0.0])
    gradient = np.array([0.0, 1.0])

    gap, projected_gradient_norm = _simplex_optimality_certificate(weights, gradient)

    assert gap == pytest.approx(0.0)
    assert projected_gradient_norm == pytest.approx(0.0)
    assert np.linalg.norm(gradient - np.mean(gradient)) > 0.0


@pytest.mark.parametrize("order_two", [False, True])
def test_successful_slsqp_without_certificate_is_not_converged(
    monkeypatch: pytest.MonkeyPatch, *, order_two: bool
) -> None:
    """Regress the prior false-convergence path and pin the total retry budget."""
    calls: list[dict[str, float | int | bool]] = []

    def fake_minimize(
        objective: object,
        initial: np.ndarray,
        *,
        jac: object,
        method: str,
        bounds: list[tuple[float, float]],
        constraints: dict[str, object],
        options: dict[str, float | int | bool],
    ) -> OptimizeResult:
        del objective, initial, jac, method, bounds, constraints
        calls.append(options)
        return OptimizeResult(x=np.array([0.5, 0.5]), success=True, nit=1)

    monkeypatch.setattr(projection_module, "minimize", fake_minimize)
    target = _cloud("target", np.zeros((2, 1)))
    matched = np.array([np.zeros((2, 1)), np.ones((2, 1))])

    _, diagnostics = _solve_anchored_wasserstein_row(
        target,
        matched,
        order_two=order_two,
        tol=CERTIFICATE_TOLERANCE,
        maxiter=2,
        solver="wasserstein_w2_target" if order_two else "wasserstein_w1_target",
        feasible_set="simplex_nonnegative",
        assignment_gap_min=1.0,
    )

    assert not diagnostics.converged
    assert diagnostics.optimality_gap == pytest.approx(0.5)
    assert diagnostics.iterations == REFINEMENT_ATTEMPTS
    assert [call["maxiter"] for call in calls] == [REFINEMENT_ATTEMPTS, 1]
    assert calls[0]["ftol"] == pytest.approx(1.0e-12)
    assert calls[1]["ftol"] == pytest.approx(64.0 * np.finfo(np.float64).eps)


@pytest.mark.parametrize("order_two", [False, True])
def test_unique_boundary_optimum_preserves_the_scientific_support(
    *, order_two: bool
) -> None:
    """Only the zero candidate can reconstruct zero with nonnegative weights."""
    weights, diagnostics = _solve_anchored_wasserstein_row(
        _cloud("target", np.zeros((2, 1))),
        np.array([np.zeros((2, 1)), np.ones((2, 1)), np.full((2, 1), 2.0)]),
        order_two=order_two,
        tol=CERTIFICATE_TOLERANCE,
        maxiter=1000,
        solver="wasserstein_w2_target" if order_two else "wasserstein_w1_target",
        feasible_set="simplex_nonnegative",
        assignment_gap_min=1.0,
    )
    assert diagnostics.converged
    np.testing.assert_allclose(
        weights, [1.0, 0.0, 0.0], atol=SUPPORT_THRESHOLD, rtol=0.0
    )
    np.testing.assert_array_equal(weights > SUPPORT_THRESHOLD, [True, False, False])
    assert diagnostics.objective == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("order", "solver"),
    [
        ("w1", "wasserstein_w1_target"),
        ("w2", "wasserstein_w2_target"),
    ],
)
def test_target_anchored_solve_matches_cvxpy_clarabel(
    order: Literal["w1", "w2"],
    solver: Literal["wasserstein_w1_target", "wasserstein_w2_target"],
) -> None:
    """Match an independent SOCP/QP solve conditional on identical maps."""
    order_two = order == "w2"
    target, matched = _matched_fixture(order_two=order_two)
    oracle_weights, oracle_objective = _solve_cvxpy_oracle(
        target.values, matched, order_two=order_two
    )

    weights, diagnostics = _solve_anchored_wasserstein_row(
        target,
        matched,
        order_two=order_two,
        tol=CERTIFICATE_TOLERANCE,
        maxiter=1000,
        solver=solver,
        feasible_set="simplex_nonnegative",
        assignment_gap_min=1.0,
    )

    assert diagnostics.converged
    assert diagnostics.optimality_gap <= CERTIFICATE_TOLERANCE * (
        1.0 + abs(diagnostics.objective)
    )
    assert diagnostics.projected_gradient_norm <= PROJECTED_GRADIENT_TOLERANCE
    assert diagnostics.objective == pytest.approx(
        oracle_objective, abs=ORACLE_OBJECTIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        weights, oracle_weights, atol=ORACLE_WEIGHT_TOLERANCE, rtol=0.0
    )


@pytest.mark.reference
@pytest.mark.requires_optional("POT")
@pytest.mark.parametrize("order", ["w1", "w2"])
def test_balanced_assignment_matches_pot(order: Literal["w1", "w2"]) -> None:
    """Match POT's balanced Kantorovich cost and unique assignment plan."""
    ot = pytest.importorskip("ot")
    order_two = order == "w2"
    left = np.array([[-2.0, -2.0], [-3.0, -1.0], [3.0, -2.0]])
    right = np.array([[-2.0, 2.0], [0.0, 1.0], [-3.0, -1.0]])
    cost = scipy_cdist(left, right)
    if order_two:
        cost = cost**2
    mass = np.full(len(left), 1.0 / len(left))

    permutation, _ = _transport_permutation(
        left, right, order_two=order_two, metric="euclidean"
    )
    plan = np.asarray(ot.emd(mass, mass, cost), dtype=np.float64)
    pot_cost = float(ot.emd2(mass, mass, cost))
    assignment_cost = float(np.mean(cost[np.arange(len(left)), permutation]))

    assert assignment_cost == pytest.approx(pot_cost, abs=1.0e-12)
    np.testing.assert_array_equal(np.argmax(plan, axis=1), permutation)


@pytest.mark.parametrize("order", ["w1", "w2"])
def test_reported_certificate_recomputes_from_returned_weights(
    order: Literal["w1", "w2"],
) -> None:
    """Keep persisted diagnostic semantics tied to the solved row."""
    order_two = order == "w2"
    target, matched = _matched_fixture(order_two=order_two)
    weights, diagnostics = _solve_anchored_wasserstein_row(
        target,
        matched,
        order_two=order_two,
        tol=CERTIFICATE_TOLERANCE,
        maxiter=1000,
        solver="wasserstein_w2_target" if order_two else "wasserstein_w1_target",
        feasible_set="simplex_nonnegative",
        assignment_gap_min=1.0,
    )
    _, gradient = _anchored_objective_and_gradient(
        weights, target.values, matched, order_two=order_two
    )
    expected_gap, expected_projected_gradient = _simplex_optimality_certificate(
        weights, gradient
    )

    assert diagnostics.optimality_gap == pytest.approx(expected_gap, abs=1.0e-15)
    assert diagnostics.projected_gradient_norm == pytest.approx(
        expected_projected_gradient, abs=1.0e-15
    )


@pytest.mark.parametrize("order", ["w1", "w2"])
def test_global_embedding_rescaling_preserves_assignments_and_simplex_weights(
    order: Literal["w1", "w2"],
) -> None:
    """A common positive scale cannot manufacture W1/W2 separation."""
    order_two = order == "w2"
    target_values, candidates = _oracle_fixture()
    original_matched = []
    scaled_matched = []
    for candidate in candidates:
        original_permutation, _ = _transport_permutation(
            target_values,
            candidate,
            order_two=order_two,
            metric="euclidean",
        )
        scaled_permutation, _ = _transport_permutation(
            GLOBAL_SCALE * target_values,
            GLOBAL_SCALE * candidate,
            order_two=order_two,
            metric="euclidean",
        )
        np.testing.assert_array_equal(scaled_permutation, original_permutation)
        original_matched.append(candidate[original_permutation])
        scaled_matched.append(GLOBAL_SCALE * candidate[scaled_permutation])

    original_weights, original_diagnostics = _solve_anchored_wasserstein_row(
        _cloud("target", target_values),
        np.stack(original_matched),
        order_two=order_two,
        tol=CERTIFICATE_TOLERANCE,
        maxiter=1000,
        solver="wasserstein_w2_target" if order_two else "wasserstein_w1_target",
        feasible_set="simplex_nonnegative",
        assignment_gap_min=1.0,
    )
    scaled_weights, scaled_diagnostics = _solve_anchored_wasserstein_row(
        _cloud("target", GLOBAL_SCALE * target_values),
        np.stack(scaled_matched),
        order_two=order_two,
        tol=CERTIFICATE_TOLERANCE,
        maxiter=1000,
        solver="wasserstein_w2_target" if order_two else "wasserstein_w1_target",
        feasible_set="simplex_nonnegative",
        assignment_gap_min=1.0,
    )

    assert original_diagnostics.converged
    assert scaled_diagnostics.converged
    np.testing.assert_allclose(
        scaled_weights,
        original_weights,
        atol=ORACLE_WEIGHT_TOLERANCE,
        rtol=0.0,
    )

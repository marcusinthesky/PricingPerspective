"""Independent witnesses for the scoped directional-inference audit."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pipeline._kernels.paper6.inference_audit import (
    FIXTURE_VARIANCE_FACTOR,
    ONE_SIDED_Z,
    bound_diagnostics,
    iid_fixture,
    iid_replications,
    pooled_candidate,
)
from pipeline._kernels.paper6.reference_alignment import (
    analyze,
    annual_moments,
    triple_projection,
)
from pipeline.stages.papers.paper6.inference_audit import run_audit

if TYPE_CHECKING:
    from pathlib import Path


def test_contracted_candidate_matches_enumerated_triples_and_ratio_derivative() -> None:
    """Both firm roles and cross-year covariance survive the cheap contraction."""
    rng = np.random.default_rng(406)
    financial, approach = rng.normal(size=(2, 4, 7, 7))
    summary, arrays = analyze(financial, approach)
    candidate = pooled_candidate(financial, approach)
    assert candidate["beta"] == pytest.approx(summary["original_pooled_beta"])
    expected = np.std(arrays["pooled_beta_if"], ddof=1) / np.sqrt(7)
    assert candidate["candidate_se"] == pytest.approx(expected, rel=1e-13)
    projection = triple_projection(financial, approach)
    q, d = projection.moments.numerator, projection.moments.denominator
    step = 1e-5
    numerical = []
    for phi_q, phi_d in zip(
        projection.numerator_influence,
        projection.denominator_influence,
        strict=True,
    ):
        plus = (q + step * phi_q).sum() / (d + step * phi_d).sum()
        minus = (q - step * phi_q).sum() / (d - step * phi_d).sum()
        numerical.append((plus - minus) / (2 * step))
    # Central differences have O(step²) error on this well-conditioned fixture.
    np.testing.assert_allclose(numerical, arrays["pooled_beta_if"], atol=1e-10)
    assert candidate["candidate_lower_bound"] == pytest.approx(
        candidate["beta"] - ONE_SIDED_Z * expected
    )


def test_zero_denominator_and_degenerate_projection_are_distinct() -> None:
    """An unidentified year need not invalidate an identified pooled statistic."""
    rng = np.random.default_rng(407)
    financial, approach = rng.normal(size=(2, 4, 6, 6))
    financial[0] = 0
    candidate = pooled_candidate(financial, approach)
    projection = triple_projection(financial, approach)
    beta = projection.moments.beta
    assert beta is not None
    expected_if = (
        projection.numerator_influence - beta * projection.denominator_influence
    ).sum(axis=1) / projection.moments.denominator.sum()
    assert candidate["candidate_se"] == pytest.approx(
        np.std(expected_if, ddof=1) / np.sqrt(6)
    )
    undefined = pooled_candidate(np.zeros_like(financial), approach)
    assert undefined["status"] == "zero_denominator"
    zero_law = pooled_candidate(financial, np.zeros_like(approach))
    assert zero_law["status"] == "degenerate_projection"
    assert zero_law["candidate_se"] == 0
    assert zero_law["candidate_lower_bound"] is None


def test_iid_fixture_direct_ratio_and_shared_latents_are_paired() -> None:
    """Reference centering eliminates U, while the E target role remains active."""
    x = np.array([-2.0, -1.0, 0.25, 1.0, 1.5, 3.0])
    error = np.array([0.4, 1.0, -0.2, 0.5, -0.4, -1.0])
    financial, approach = iid_fixture(x, error, np.ones(6), beta=0.5)
    financial_other, approach_other = iid_fixture(x, error, -np.ones(6), beta=0.5)
    moments = annual_moments(financial, approach)
    other = annual_moments(financial_other, approach_other)
    np.testing.assert_allclose(moments.numerator, other.numerator, atol=1e-15)
    expected_q, expected_d = 0.0, 0.0
    for target in range(len(x)):
        included = np.arange(len(x)) != target
        xc = x[included] - x[included].mean()
        ec = error[included] - error[included].mean()
        expected_q += 0.5 * 10 * (xc @ xc) + 9 * (
            xc @ ec + 0.5 * error[target] * (xc @ xc)
        )
        expected_d += 10 * (xc @ xc)
    assert moments.beta == pytest.approx(expected_q / expected_d)
    arrays = iid_replications(replications=8, firms=6, seed=3)
    np.testing.assert_allclose(arrays["estimates"][1] - arrays["estimates"][0], 0.5)
    np.testing.assert_allclose(arrays["candidate_se"][1], arrays["candidate_se"][0])
    replay = iid_replications(replications=8, firms=6, seed=3)
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, replay[key])


def test_iid_population_influence_has_declared_analytic_variance() -> None:
    """Exact quadrature checks the known target and population first influence.

    Each coordinate appears to degree at most two in these expectations, so
    the independent +/-1 nodes integrate the required normal moments exactly.
    This enumerates population conditional expectations rather than estimating
    an asymptotic variance from a fortunate finite simulation.
    """
    types = tuple(product((-1.0, 1.0), repeat=2))
    c, g = np.array([1.0, 2.0, 1.0, 2.0]), np.array([1.0, 1.0, 2.0, 2.0])
    beta = 0.5
    conditional = []
    for fixed in types:
        kernels = []
        for first, second in product(types, repeat=2):
            x, error = np.array((fixed, first, second)).T
            q, d = np.zeros(4), np.zeros(4)
            for target, reference, other in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
                dr = c * (x[reference] - x[other])
                da = beta * dr + g * (
                    error[reference]
                    - error[other]
                    + 0.5 * error[target] * (x[reference] - x[other])
                )
                q += dr * da / 3
                d += dr * dr / 3
            kernels.append((q, d))
        conditional.append(np.mean(kernels, axis=0))
    conditional = np.array(conditional)
    uq, ud = conditional.mean(axis=0)
    np.testing.assert_allclose(uq, beta * ud)
    np.testing.assert_allclose(0.5 * ud, c**2)
    phi_q, phi_d = (1.5 * (conditional - conditional.mean(axis=0))).transpose(1, 0, 2)
    pooled_if = (phi_q - beta * phi_d).sum(axis=1) / (0.5 * ud.sum())
    x, error = np.array(types).T
    np.testing.assert_allclose(pooled_if, 0.9 * error * (x + 0.5), atol=1e-14)
    assert np.mean(pooled_if**2) == pytest.approx(FIXTURE_VARIANCE_FACTOR)


def test_candidate_rates_retain_undefined_draws_and_use_one_sided_quantile() -> None:
    """Dropping failed bounds or using 1.96 would change the rejection count."""
    diagnostics = bound_diagnostics([1.8, np.nan, 0.0, 2.0], [1.0, 1.0, 0.0, 1.0], 0)
    assert diagnostics["positive_bound"]["count"] == len([1.8, 2.0])
    assert diagnostics["positive_bound"]["attempts"] == len([1.8, np.nan, 0.0, 2.0])
    assert diagnostics["positive_bound"]["rate"] == pytest.approx(0.5)
    assert diagnostics["target_containment"]["rate"] == 0
    assert diagnostics["undefined_estimates"] == 1
    assert diagnostics["degenerate_or_negative_candidate_se"] == 1
    assert diagnostics["interval_length"] is None


def test_audit_labels_conditional_mismatch_and_stratifies_mcse(tmp_path: Path) -> None:
    """Enumerated histories do not contribute fictitious between-history MCSE."""
    rows = [
        {
            "scenario_id": "A",
            "regime": "common",
            "budget": 1,
            "history": history,
            "beta": estimate,
            "oracle_beta": target,
            "candidate_se": se,
            "exact_population_candidate_se": 0.1,
        }
        for history, target, estimates, se in (
            (0, 1.0, (1.0, 2.0), 0.1),
            (1, 0.0, (0.1, 0.1), 1.0),
        )
        for estimate in estimates
    ]
    result = run_audit(
        tmp_path,
        conditional_rows=rows,
        fixture_replications=8,
        fixture_firms=6,
        seed=3,
    )
    stress = result["conditional_stress"][0]
    assert stress["rates"]["target_containment"]["equal_history_rate"] == pytest.approx(
        0.75
    )
    assert stress["rates"]["target_containment"]["mcse"] == pytest.approx(0.25)
    interpretation = stress["per_history"][0]["positive_bound_interpretation"]
    assert "not a Type I error" in interpretation
    assert stress["per_history"][1]["exact_population_conditional_measurement_sd"] == 0
    assert not result["empirical_inference"]["eligible"]
    assert result["empirical_inference"]["lower_bound"] is None
    with np.load(tmp_path / "fixture-replicates.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["estimates"][1] - saved["estimates"][0], 0.5)


@pytest.mark.parametrize("scenario", ["B", "C_zero_reporting"])
def test_constructed_null_preserves_arithmetic_residue(
    tmp_path: Path, scenario: str
) -> None:
    """Floating residue does not turn a constructed null into an alternative."""
    residual = 1.5e-15
    rows = [
        {
            "scenario_id": scenario,
            "regime": "common",
            "budget": 1,
            "history": 0,
            "beta": 0.1,
            "oracle_beta": residual,
            "oracle_constructed_zero": True,
            "candidate_se": 1.0,
        }
        for _ in range(2)
    ]
    result = run_audit(tmp_path, conditional_rows=rows, fixture_replications=2)
    history = result["conditional_stress"][0]["per_history"][0]
    assert history["target"] == 0
    assert history["arithmetic_oracle_beta"] == pytest.approx(residual, abs=1e-18)
    assert history["oracle_constructed_zero"]
    assert "conditional zero target" in history["positive_bound_interpretation"]
    # New saved schemas can already canonicalize the scientific oracle to zero.
    for row in rows:
        row["oracle_beta"] = 0.0
        row["oracle_arithmetic_beta"] = residual
    result = run_audit(tmp_path, conditional_rows=rows, fixture_replications=2)
    history = result["conditional_stress"][0]["per_history"][0]
    assert history["arithmetic_oracle_beta"] == pytest.approx(residual, abs=1e-18)


def test_failed_oracle_stratum_remains_in_attempted_accounting(tmp_path: Path) -> None:
    """A failed history cannot disappear from a favorable equal-history average."""
    rows = [
        {
            "scenario_id": "D",
            "regime": "common",
            "budget": 1,
            "history": history,
            "beta": estimate,
            "oracle_beta": target,
            "candidate_se": standard_error,
        }
        for history, target, estimate, standard_error in (
            (0, 1.0, 1.0, 0.1),
            (1, None, None, None),
        )
        for _ in range(2)
    ]
    result = run_audit(tmp_path, conditional_rows=rows, fixture_replications=2)
    stress = result["conditional_stress"][0]
    failed = stress["per_history"][1]
    assert failed["target"] is None
    assert failed["target_status"] == "unavailable"
    assert failed["undefined_estimates"] == len([None, None])
    assert failed["target_containment"] is None
    assert failed["positive_bound"] is None
    aggregate = stress["rates"]["target_containment"]
    assert aggregate["attempts"] == len(rows)
    assert aggregate["unavailable_histories"] == [1]
    assert aggregate["equal_history_rate"] is None
    assert aggregate["mcse"] is None

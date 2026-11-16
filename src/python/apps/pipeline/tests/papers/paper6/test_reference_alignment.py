"""Independent finite identities and dated measurement boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from hotelling.dynamics.transport import axis_point_w2
from pipeline._kernels.paper6.recovery import estimate_panel
from pipeline._kernels.paper6.reference_alignment import (
    analyze,
    annual_moments,
    conditional_covariance,
    financial_tails,
    fixed_common_slope_influence,
    fixed_linear_gmm,
    pool_dated_laws,
    recovery_summary,
    strict_tail_readout,
    summarize,
    triple_projection,
)

MATERIAL_DIFFERENCE = 0.01
EIGENVALUE_ROUNDOFF = 1e-15


def test_pooled_ratio_matches_fixed_effect_least_squares_and_role_deletions() -> None:
    """Deleting a firm removes all three roles and recenters its references."""
    rng = np.random.default_rng(81)
    r, a = rng.normal(size=(2, 4, 6, 6))
    r[:, np.arange(6), np.arange(6)] = np.nan
    a[:, np.arange(6), np.arange(6)] = np.nan
    mask = ~np.eye(6, dtype=bool)
    effects = np.eye(24)[np.repeat(np.arange(24), 5)]
    design = np.column_stack((effects, r[:, mask].ravel()))
    expected = np.linalg.lstsq(design, a[:, mask].ravel(), rcond=None)[0][-1]
    distances = np.concatenate((np.zeros_like(a[:1]), -np.cumsum(a, axis=0)))
    summary = summarize(distances, r, list("ABCDEF"))
    assert summary["beta"] == pytest.approx(expected)
    projection = triple_projection(r, a)
    assert projection.moments.beta == pytest.approx(expected)
    np.testing.assert_allclose(
        projection.numerator_influence.mean(axis=0), 0, atol=1e-15
    )
    np.testing.assert_allclose(
        projection.denominator_influence.mean(axis=0), 0, atol=1e-15
    )
    for removed in range(6):
        keep = np.arange(6) != removed
        reduced = annual_moments(r[:, keep][:, :, keep], a[:, keep][:, :, keep])
        np.testing.assert_allclose(
            projection.deletion_numerator[removed], reduced.numerator
        )
        np.testing.assert_allclose(
            projection.deletion_denominator[removed], reduced.denominator
        )


def test_strict_tails_and_pooling_share_heterogeneous_date_counts() -> None:
    """Same-date news is excluded and each target uses its own sampled date mass."""
    counts = np.array([[3, 0], [0, 4], [1, 0]], dtype=np.int64)
    laws = np.broadcast_to(np.eye(2)[None, None], (1, 3, 2, 2)).copy()
    pooled = pool_dated_laws(laws, counts[None])
    np.testing.assert_array_equal(pooled[0], np.eye(2))
    readout = financial_tails(
        np.array([1.0, 2.0, 3.0]),
        counts,
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert readout[0, 1] == pytest.approx(1.5)
    assert readout[1, 0] == pytest.approx(1)
    zero = financial_tails(
        np.array([1.0]),
        np.ones((1, 2), dtype=np.int64),
        np.array([1.0]),
        np.ones((1, 2)),
    )
    np.testing.assert_array_equal(zero[~np.eye(2, dtype=bool)], 0)


def test_zero_pooled_denominator_and_undefined_draws_remain_visible() -> None:
    """Undefined estimates cannot disappear from a balanced recovery average."""
    assert annual_moments(np.ones((4, 6, 6)), np.ones((4, 6, 6))).beta is None
    estimates = np.ones((16, 128))
    estimates[0, 0] = np.nan
    result = recovery_summary(estimates, np.ones(16))
    assert result.undefined == 1
    assert result.normalized_rmse is None
    assert not result.recovered


def test_stratified_recovery_error_is_not_history_sampling_uncertainty() -> None:
    """Enumeration creates no between-history Monte Carlo variance."""
    oracle = np.linspace(1, 2, 16)
    estimates = np.broadcast_to(oracle[:, None] + 0.1, (16, 128))
    result = recovery_summary(estimates, oracle)
    assert result.normalized_rmse == pytest.approx(0.1 / np.sqrt(np.mean(oracle**2)))
    assert result.rmse_mcse == pytest.approx(0, abs=1e-16)
    assert result.recovered


def test_fixed_weight_common_slope_changes_target_under_annual_heterogeneity() -> None:
    """The appendix GMM target need not equal the main pooled ratio."""
    q, d = np.array([1.0, -1.0]), np.array([1.0, 2.0])
    weight = np.array([[2.0, 0.2], [0.2, 1.0]])
    fitted = fixed_linear_gmm(q, d[:, None], weight)
    expected = float(d @ weight @ q / (d @ weight @ d))
    assert fitted["theta"][0] == pytest.approx(expected)
    assert abs(expected - q.sum() / d.sum()) > MATERIAL_DIFFERENCE
    assert fitted["criterion"] > 0


def test_misspecified_common_slope_influence_uses_full_derivative() -> None:
    """Changes in annual dispersion multiply the nonzero moment residual."""
    q = np.array([0.2, -0.1, 0.4, 0.3])
    d = np.array([0.5, 0.8, 0.3, 1.1])
    phi_q = np.array([[0.3, 0.1, -0.2, 0.05], [-0.3, -0.1, 0.2, -0.05]])
    phi_d = np.array([[0.2, -0.1, 0.3, -0.2], [-0.2, 0.1, -0.3, 0.2]])
    weight = np.diag([1.0, 2, 3, 1])
    beta, full, restricted = fixed_common_slope_influence(q, d, phi_q, phi_d, weight)
    step = 1e-6
    for i in range(len(phi_q)):
        plus = fixed_linear_gmm(
            q + step * phi_q[i], (d + step * phi_d[i])[:, None], weight
        )["theta"][0]
        minus = fixed_linear_gmm(
            q - step * phi_q[i], (d - step * phi_d[i])[:, None], weight
        )["theta"][0]
        assert (plus - minus) / (2 * step) == pytest.approx(full[i], abs=1e-10)
    assert np.max(np.abs(full - restricted)) > MATERIAL_DIFFERENCE
    _, full_exact, restricted_exact = fixed_common_slope_influence(
        beta * d, d, phi_q, phi_d, weight
    )
    np.testing.assert_allclose(full_exact, restricted_exact, atol=1e-12)


def test_conditional_covariance_matches_joint_history_sample_covariance() -> None:
    """Annual and pooled influences share histories and retain cross-year terms."""
    rng = np.random.default_rng(82)
    financial, approach = rng.normal(size=(2, 4, 8, 8))
    _, arrays = analyze(financial, approach)
    influence = np.column_stack(
        (
            arrays["annual_beta_if"],
            arrays["pooled_beta_if"],
            arrays["common_beta_full_if"],
        )
    )
    expected = np.cov(influence, rowvar=False, ddof=1) / len(influence)
    candidate = conditional_covariance(influence)
    np.testing.assert_allclose(candidate["matrix"], expected, rtol=1e-14, atol=1e-16)
    assert candidate["eigenvalues"].min() >= -EIGENVALUE_ROUNDOFF
    assert "No coverage claim" in candidate["scope"]


def test_fractional_article_mass_and_count_tails_agree() -> None:
    """Historical fractional redraw masses use the same strict-tail computation."""
    dates = np.array([1.0, 2.0, 3.0])
    counts = np.array([[3, 0], [0, 4], [1, 0]], dtype=np.int64)
    news = np.array([[1.0, 2.0], [3, 4], [5, 6]])
    actual = financial_tails(dates, counts, dates, news)
    weighted = strict_tail_readout(
        dates, counts / counts.sum(axis=0, keepdims=True), dates, news
    )
    offdiag = ~np.eye(2, dtype=bool)
    np.testing.assert_allclose(actual[offdiag], weighted[offdiag], rtol=0, atol=0)


def test_complete_triple_diagnostics_reject_unidentified_annual_slopes() -> None:
    """A positive pooled denominator does not identify each annual coefficient."""
    rng = np.random.default_rng(83)
    financial, approach = rng.normal(size=(2, 2, 5, 5))
    financial[0] = 1
    assert annual_moments(financial, approach).beta is not None
    with pytest.raises(ValueError, match="each annual coefficient"):
        analyze(financial, approach)


def test_origin_law_is_shared_at_every_occurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drawn 2018 reference law remains fixed across all target years."""
    rng = np.random.default_rng(84)
    laws = rng.dirichlet(np.ones(4), size=(5, 6))
    counts = rng.integers(1, 5, size=(5, 12, 6), dtype=np.int64)
    news = rng.normal(size=(60, 6))
    calls = []

    def inspect_origin(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
        np.testing.assert_array_equal(reference, laws[0][None, None, :, :])
        np.testing.assert_array_equal(target, laws[:, :, None, :])
        calls.append(reference.copy())
        return axis_point_w2(target, reference)

    monkeypatch.setattr(
        "pipeline._kernels.paper6.recovery.axis_point_w2", inspect_origin
    )
    assert np.isfinite(estimate_panel(laws, counts, news))
    assert len(calls) == 1


def test_no_news_has_a_zero_tail_readout() -> None:
    """An empty news sequence creates finite zeros for every target-reference pair."""
    result = financial_tails(
        np.array([1.0, 2.0]),
        np.ones((2, 3), dtype=np.int64),
        np.empty(0),
        np.empty((0, 3)),
    )
    np.testing.assert_array_equal(result[~np.eye(3, dtype=bool)], 0)

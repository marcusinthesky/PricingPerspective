"""Tests for the fixed-W2 random-exposure envelope diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pipeline.stages.papers.paper1._dyadic.random_exposure import (
    RandomExposureInputError,
    RandomExposureInputs,
    fixed_w2_random_exposure_diagnostic,
)

if TYPE_CHECKING:
    import pandas as pd

# Two dyads whose bracket arithmetic is checkable by hand.  With
# ``c_actual = 2 - 2 rho``, ``w2 = [0.5, 1.0]`` and ``rho = [0.875, 0.0]`` give
# ``c_actual = [0.25, 2.0]``, so at ``(L, tau) = (1, 0)`` the total-return
# compatibility quantity is exactly ``[0.0, 1.0]``.
_W2 = np.array([0.5, 1.0])
_RHO = np.array([0.875, 0.0])
_FIRM_I = np.array(["a", "a"])
_FIRM_J = np.array(["b", "c"])
_TICKERS = ["a", "b", "c"]
# Draw 0 keeps every firm once; draw 1 drops firm ``c``, so the a--c dyad gets
# endpoint weight ``2 * 0 = 0`` and that draw's share collapses onto a--b alone.
_NODE_COUNTS = np.array([[1, 1, 1], [2, 1, 0]], dtype=np.int64)


def _run(
    w2: np.ndarray = _W2, rho: np.ndarray = _RHO
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run the diagnostic on the shared two-dyad fixture."""
    summary, frame = fixed_w2_random_exposure_diagnostic(
        RandomExposureInputs(
            w2=w2,
            correlation=rho,
            firm_i=_FIRM_I,
            firm_j=_FIRM_J,
            tickers=_TICKERS,
            node_counts=_NODE_COUNTS,
        )
    )
    return cast("list[dict[str, object]]", summary["scenarios"]), frame


def _scenario(
    scenarios: list[dict[str, object]], lipschitz: float, slack: float
) -> dict[str, object]:
    """Select one maintained-scale scenario row by its declared scales."""
    return next(
        row
        for row in scenarios
        if np.isclose(cast("float", row["L"]), lipschitz)
        and np.isclose(cast("float", row["tau"]), slack)
    )


def _dyads(frame: pd.DataFrame, lipschitz: float, slack: float) -> pd.DataFrame:
    """Select the dyad rows belonging to one scenario."""
    mask = np.isclose(frame["L"].to_numpy(), lipschitz) & np.isclose(
        frame["tau"].to_numpy(), slack
    )
    return cast("pd.DataFrame", frame.loc[mask])


def test_zero_slack_w2_interval_has_exact_total_return_diagnostic() -> None:
    """At L=1 and tau=0 the two diagnostic endpoints coincide."""
    scenarios, frame = _run()
    scenario = _scenario(scenarios, 1.0, 0.0)
    assert np.isclose(cast("float", scenario["unresolved_share"]), 0.0)
    dyads = _dyads(frame, 1.0, 0.0)
    np.testing.assert_allclose(
        dyads["delta_lower"].to_numpy(), dyads["delta_upper"].to_numpy()
    )
    # c_actual - W2^2 = [0.25 - 0.25, 2.0 - 1.0].
    np.testing.assert_allclose(dyads["delta_lower"].to_numpy(), [0.0, 1.0])


def test_bracket_endpoints_match_the_manuscript_transfer_interval() -> None:
    """Reproduce (w^-, w^+) and the excess endpoints for a slack scenario.

    At ``L = 1`` and ``tau = 0.05`` the common slack is ``tau_ij = 0.10``, so
    ``w^- = w2 - 0.10`` and ``w^+ = w2 + 0.10``.  The first dyad then straddles
    zero excess and must be reported unresolved rather than covered.
    """
    scenarios, frame = _run()
    dyads = _dyads(frame, 1.0, 0.05)
    np.testing.assert_allclose(dyads["w2_lower"].to_numpy(), [0.4, 0.9])
    np.testing.assert_allclose(dyads["w2_upper"].to_numpy(), [0.6, 1.1])
    # c_actual - (w^+)^2 and c_actual - (w^-)^2.
    np.testing.assert_allclose(dyads["delta_lower"].to_numpy(), [-0.11, 0.79])
    np.testing.assert_allclose(dyads["delta_upper"].to_numpy(), [0.09, 1.19])
    assert list(dyads["robust_coverage"]) == [False, True]
    assert list(dyads["unresolved"]) == [True, False]
    assert list(dyads["robust_violation"]) == [False, False]

    scenario = _scenario(scenarios, 1.0, 0.05)
    assert np.isclose(cast("float", scenario["coverage_share"]), 0.5)
    assert np.isclose(cast("float", scenario["unresolved_share"]), 0.5)


def test_lipschitz_scale_widens_the_bracket_monotonically() -> None:
    """L rescales both endpoints; it never moves a dyad back to resolved."""
    scenarios, frame = _run()
    dyads = _dyads(frame, 1.25, 0.0)
    np.testing.assert_allclose(dyads["w2_lower"].to_numpy(), [0.4, 0.8])
    np.testing.assert_allclose(dyads["w2_upper"].to_numpy(), [0.625, 1.25])
    unresolved = [
        cast("float", _scenario(scenarios, lipschitz, 0.0)["unresolved_share"])
        for lipschitz in (1.0, 1.25, 1.5)
    ]
    assert unresolved == sorted(unresolved)


def test_robust_violation_is_reported_when_returns_are_too_correlated() -> None:
    """A dyad whose realized cost falls below the lower bracket violates."""
    # c_actual = 0.1 while (w^-)^2 = 1.96 at (L, tau) = (1, 0).
    scenarios, frame = _run(w2=np.array([1.4, 1.0]), rho=np.array([0.95, 0.0]))
    dyads = _dyads(frame, 1.0, 0.0)
    assert list(dyads["robust_violation"]) == [True, False]
    assert np.isclose(
        cast("float", _scenario(scenarios, 1.0, 0.0)["violation_share"]), 0.5
    )


def test_node_bootstrap_weights_dyads_by_endpoint_multiplicity() -> None:
    """A draw that drops a firm must drop every dyad touching it."""
    scenarios, _frame = _run(w2=np.array([1.4, 1.0]), rho=np.array([0.95, 0.0]))
    scenario = _scenario(scenarios, 1.0, 0.0)
    # Draw 1 has multiplicities (a, b, c) = (2, 1, 0): the a--c dyad receives
    # weight 0, so that draw's violation share is the a--b indicator alone, 1.0.
    # Draw 0 weights both dyads equally, giving 0.5.  The two-draw percentile
    # interval therefore spans [0.5, 1.0].
    assert np.isclose(cast("float", scenario["violation_ci_low"]), 0.5, atol=0.05)
    assert np.isclose(cast("float", scenario["violation_ci_high"]), 1.0, atol=0.05)


def test_scenario_classification_partitions_dyads() -> None:
    """Coverage, violation, and unresolved classes are disjoint and exhaustive."""
    _summary, frame = fixed_w2_random_exposure_diagnostic(
        RandomExposureInputs(
            w2=np.array([0.2, 0.8, 1.1]),
            correlation=np.array([0.98, 0.2, -0.1]),
            firm_i=np.array(["a", "a", "b"]),
            firm_j=np.array(["b", "c", "c"]),
            tickers=["a", "b", "c"],
            node_counts=np.array([[1, 1, 1], [2, 1, 0], [0, 1, 2]], dtype=np.int64),
        )
    )
    classes = frame[["robust_coverage", "robust_violation", "unresolved"]].to_numpy()
    np.testing.assert_array_equal(classes.sum(axis=1), np.ones(len(frame)))


def test_zero_weight_bootstrap_draws_are_discarded_before_percentiles() -> None:
    """An all-zero node draw cannot create a NaN bootstrap interval."""
    summary, _frame = fixed_w2_random_exposure_diagnostic(
        RandomExposureInputs(
            w2=_W2,
            correlation=_RHO,
            firm_i=_FIRM_I,
            firm_j=_FIRM_J,
            tickers=_TICKERS,
            node_counts=np.vstack([_NODE_COUNTS, np.zeros(3, dtype=np.int64)]),
        )
    )

    for row in cast("list[dict[str, object]]", summary["scenarios"]):
        interval_values = [
            cast("float", row["coverage_ci_low"]),
            cast("float", row["coverage_ci_high"]),
            cast("float", row["violation_ci_low"]),
            cast("float", row["violation_ci_high"]),
            cast("float", row["unresolved_ci_low"]),
            cast("float", row["unresolved_ci_high"]),
        ]
        assert np.isfinite(interval_values).all()


def test_random_exposure_rejects_nonfinite_correlations() -> None:
    """Correlation validation occurs before clipping or summary computation."""
    with pytest.raises(RandomExposureInputError, match="correlations must be finite"):
        _run(rho=np.array([np.nan, 0.0]))


def test_random_exposure_rejects_misaligned_endpoint_vectors() -> None:
    """Endpoint vectors must align with the dyad-indexed vectors."""
    with pytest.raises(RandomExposureInputError, match="endpoint vectors"):
        fixed_w2_random_exposure_diagnostic(
            RandomExposureInputs(
                w2=_W2,
                correlation=_RHO,
                firm_i=np.array(["a"]),
                firm_j=_FIRM_J,
                tickers=_TICKERS,
                node_counts=_NODE_COUNTS,
            )
        )


def test_random_exposure_rejects_unknown_endpoint_tickers() -> None:
    """Every endpoint label must resolve uniquely in the ticker roster."""
    with pytest.raises(RandomExposureInputError, match="endpoint membership"):
        fixed_w2_random_exposure_diagnostic(
            RandomExposureInputs(
                w2=_W2,
                correlation=_RHO,
                firm_i=np.array(["a", "unknown"]),
                firm_j=_FIRM_J,
                tickers=_TICKERS,
                node_counts=_NODE_COUNTS,
            )
        )


def test_random_exposure_rejects_malformed_node_counts() -> None:
    """Bootstrap counts must be a two-dimensional draw-by-ticker array."""
    with pytest.raises(RandomExposureInputError, match="count shape"):
        fixed_w2_random_exposure_diagnostic(
            RandomExposureInputs(
                w2=_W2,
                correlation=_RHO,
                firm_i=_FIRM_I,
                firm_j=_FIRM_J,
                tickers=_TICKERS,
                node_counts=np.ones(3, dtype=np.int64),
            )
        )


def test_random_exposure_rejects_fractional_node_counts() -> None:
    """Bootstrap multiplicities cannot be silently truncated to integers."""
    with pytest.raises(RandomExposureInputError, match="count integrality"):
        fixed_w2_random_exposure_diagnostic(
            RandomExposureInputs(
                w2=_W2,
                correlation=_RHO,
                firm_i=_FIRM_I,
                firm_j=_FIRM_J,
                tickers=_TICKERS,
                node_counts=np.asarray([[1.5, 1.0, 1.0]]),
            )
        )


def test_independent_residual_noise_is_not_systematic_transport_excess() -> None:
    """Identical fixed exposures can have positive total-return distance.

    R_i = sqrt(1/2) F + sqrt(1/2) e_i for independent unit-variance F, e_i.
    The latent exposure distance and transport excess are zero, but the
    diagnostic is one because the residual variances sum to one.
    """
    _scenarios, frame = _run(w2=np.zeros(2), rho=np.full(2, 0.5))
    dyads = _dyads(frame, 1.0, 0.0)
    np.testing.assert_allclose(dyads["delta_lower"].to_numpy(), np.ones(2))
    np.testing.assert_allclose(dyads["delta_upper"].to_numpy(), np.ones(2))
    residual_variance_sum = 1.0
    np.testing.assert_allclose(
        dyads["delta_lower"].to_numpy() - residual_variance_sum, np.zeros(2)
    )
    assert dyads["robust_coverage"].all()

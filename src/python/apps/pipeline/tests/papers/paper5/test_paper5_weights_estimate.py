"""Paper 5 tests: W_flat row-stochasticity + zero-diagonal, |rho_hat|<1.

Small synthetic panels (not the governed 100-firm data) exercising the
:mod:`pipeline.stages.papers.paper5.weights`/:mod:`pipeline.stages.papers.paper5.estimate`
invariants required by the task acceptance criteria.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from pipeline.stages.papers.paper5.co_mentions import load_co_mentions_w
from pipeline.stages.papers.paper5.estimate import (
    NUMERICAL_FLOOR,
    SarEstimationError,
    _global_scalar_minimize,
    _log_det_jacobian,
    _polish_sar_profile,
    _sar_profile,
    _sar_profile_score,
    _SarProfile,
    sar_gmm,
    sar_qmle,
)
from pipeline.stages.papers.paper5.operators import load_shared_w_flat
from pipeline.stages.papers.paper5.weights import build_w_kernel

if TYPE_CHECKING:
    from numpy.typing import NDArray
LOG_DETERMINANT_TOLERANCE = 1e-13
COMPLEX_EIGENVALUE_THRESHOLD = 0.5
EXPECTED_RETAINED_PAIRS = 3
EXPECTED_RETAINED_WEIGHT = 6.0


def test_shared_w_flat_loads_weights_and_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paper 5 consumes the typed artifact without invoking a private solver."""
    tickers = ["A", "B", "C"]
    weights = pd.DataFrame(
        [
            {
                "target": "A",
                "target_index": 0,
                "candidate_index": 0,
                "candidate": "B",
                "weight": 0.25,
            },
            {
                "target": "A",
                "target_index": 0,
                "candidate_index": 1,
                "candidate": "C",
                "weight": 0.75,
            },
            {
                "target": "B",
                "target_index": 1,
                "candidate_index": 0,
                "candidate": "A",
                "weight": 0.60,
            },
            {
                "target": "B",
                "target_index": 1,
                "candidate_index": 1,
                "candidate": "C",
                "weight": 0.40,
            },
            {
                "target": "C",
                "target_index": 2,
                "candidate_index": 0,
                "candidate": "A",
                "weight": 0.10,
            },
            {
                "target": "C",
                "target_index": 2,
                "candidate_index": 1,
                "candidate": "B",
                "weight": 0.90,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "target_id": "A",
                "converged": True,
                "tangent_curvature_min": 0.5,
                "assignment_gap_min": 2.0,
                "assignment_support_size": 2,
            },
            {
                "target_id": "B",
                "converged": True,
                "tangent_curvature_min": 0.25,
                "assignment_gap_min": 1.0,
                "assignment_support_size": 2,
            },
            {
                "target_id": "C",
                "converged": True,
                "tangent_curvature_min": 0.125,
                "assignment_gap_min": 0.5,
                "assignment_support_size": 2,
            },
        ]
    )
    monkeypatch.setattr(
        "pipeline.stages.papers.paper5.operators.read_target_projection_artifact",
        lambda *_args, **_kwargs: (weights, diagnostics, {}),
    )

    matrix, loaded_diagnostics = load_shared_w_flat(
        Path("unused"),
        tickers,
        provider_id="qwen3-embedding-8b",
        representation_id="qwen3-embedding-8b-unit",
        arm_id="energy_simplex",
        geometry_id="energy_v",
    )

    np.testing.assert_allclose(
        matrix,
        [[0.0, 0.25, 0.75], [0.6, 0.0, 0.4], [0.1, 0.9, 0.0]],
    )
    assert [item.tangent_curvature_min for item in loaded_diagnostics] == [
        0.5,
        0.25,
        0.125,
    ]
    assert all(item.converged for item in loaded_diagnostics)


def test_w_kernel_row_normalized_zero_diagonal() -> None:
    """W_h is row-normalized and zero-diagonal, bandwidth positive."""
    rng = np.random.default_rng(1)
    n = 6
    pts = rng.standard_normal((n, 3))
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, 0.0)
    tickers = [f"T{i}" for i in range(n)]

    w_h, h = build_w_kernel(d2, tickers)
    assert h > 0
    np.testing.assert_allclose(w_h.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.diag(w_h), 0.0, atol=1e-12)


def test_co_mentions_loads_symmetric_counts_in_ticker_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The undirected edge list becomes a weighted row-stochastic comparator."""
    edges = pd.DataFrame(
        {
            "firm_i": ["A", "A", "B", "OUT"],
            "firm_j": ["B", "C", "C", "A"],
            "weight": [2, 1, 3, 99],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: edges)

    matrix, diagnostics = load_co_mentions_w(Path("unused"), ["C", "A", "B"])

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    np.testing.assert_allclose(matrix[1], [1 / 3, 0.0, 2 / 3])
    assert diagnostics["retained_undirected_pairs"] == EXPECTED_RETAINED_PAIRS
    np.testing.assert_allclose(
        diagnostics["retained_pair_weight"], EXPECTED_RETAINED_WEIGHT
    )
    assert diagnostics["certified"] is False


def test_co_mentions_rejects_isolated_priced_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row-normalized comparison is undefined when a priced node is isolated."""
    edges = pd.DataFrame({"firm_i": ["A"], "firm_j": ["B"], "weight": [1]})
    monkeypatch.setattr(pd, "read_parquet", lambda _path: edges)

    with pytest.raises(ValueError, match=r"isolates priced tickers.*C"):
        load_co_mentions_w(Path("unused"), ["A", "B", "C"])


def test_ord_log_jacobian_matches_slogdet_for_asymmetric_weights() -> None:
    """Retain complex eigenvalue components for a directed interaction cycle."""
    w = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    eigvals = cast("NDArray[np.complex128]", np.linalg.eigvals(w))
    assert np.max(np.abs(eigvals.imag)) > COMPLEX_EIGENVALUE_THRESHOLD

    for rho in (-0.75, 0.5, 0.9):
        _sign, expected = np.linalg.slogdet(np.eye(w.shape[0]) - rho * w)
        actual = _log_det_jacobian(rho, eigvals)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=LOG_DETERMINANT_TOLERANCE,
            atol=LOG_DETERMINANT_TOLERANCE,
        )


def test_ord_log_jacobian_is_negative_infinity_at_singularity() -> None:
    """Do not replace a singular Jacobian with a clipped finite likelihood."""
    eigvals = np.array([1.0 + 0.0j], dtype=np.complex128)
    assert np.isneginf(_log_det_jacobian(1.0, eigvals))


def test_sar_qmle_and_gmm_respect_stationarity_bound() -> None:
    """SAR QMLE/GMM estimates always satisfy |rho_hat| < 1, even near-degenerate W."""
    rng = np.random.default_rng(2)
    n, t = 6, 100
    pts = rng.standard_normal((n, 3))
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, 0.0)
    tickers = [f"T{i}" for i in range(n)]
    w_h, _ = build_w_kernel(d2, tickers)

    r = rng.standard_normal((t, n)) * 0.01

    qmle = sar_qmle(r, w_h)
    with pytest.raises(SarEstimationError, match="unidentified"):
        sar_gmm(r, w_h)

    assert abs(qmle["rho_hat"]) < 1.0
    assert np.isfinite(qmle["loglik"])


def _nilpotent_sar_panel(rho: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Give the concentrated likelihood a known interior or boundary optimum."""
    rng = np.random.default_rng(246)
    driver = rng.normal(size=300)
    noise = rng.normal(size=300)
    # Orthogonal centered noise leaves the OLS/profile optimum exactly at rho.
    driver -= np.mean(driver)
    noise -= np.mean(noise)
    noise -= driver * (noise @ driver) / (driver @ driver)
    returns = np.column_stack((rho * driver + noise, driver))
    return returns, np.array([[0.0, 1.0], [0.0, 0.0]])


@pytest.mark.parametrize("rho", [-0.65, 0.0, 0.57])
def test_sar_score_matches_likelihood_derivative(rho: float) -> None:
    """The full complex Jacobian and intercept profile have the right score."""
    rng = np.random.default_rng(751)
    returns = rng.normal(size=(70, 3))
    weights = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    profile = _sar_profile(returns, weights)
    step = 1e-5
    derivative = (profile.objective(rho + step) - profile.objective(rho - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        _sar_profile_score(profile, rho), -derivative, rtol=1e-8, atol=1e-8
    )


def test_sar_score_matches_scale_floor_derivative() -> None:
    """The innovation-variance floor remains part of the same likelihood."""
    returns, weights = _nilpotent_sar_panel(0.37)
    returns *= np.sqrt(NUMERICAL_FLOOR) / 100
    profile = _sar_profile(returns, weights)
    step = 1e-3
    rho = 0.1
    assert profile.nuisance(rho)[1] == NUMERICAL_FLOOR
    derivative = (profile.objective(rho + step) - profile.objective(rho - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        _sar_profile_score(profile, rho), -derivative, rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("offset", [0.0, 1e12])
def test_sar_polish_resolves_large_constant_objective_plateau(
    monkeypatch: pytest.MonkeyPatch, offset: float
) -> None:
    """A likelihood constant must not change the resolved coefficient."""
    returns, weights = _nilpotent_sar_panel(0.37123456789)
    objective = _SarProfile.objective
    monkeypatch.setattr(
        _SarProfile, "objective", lambda profile, rho: objective(profile, rho) + offset
    )
    fit = sar_qmle(returns, weights, compute_se=False)
    np.testing.assert_allclose(fit["rho_hat"], 0.37123456789, rtol=0.0, atol=2e-14)
    profile = _sar_profile(returns, weights)
    assert fit["loglik"] == -profile.objective(fit["rho_hat"])
    assert fit["converged"]


def test_sar_polish_retains_admissible_boundary() -> None:
    """An increasing likelihood without a score root keeps its boundary optimum."""
    returns, weights = _nilpotent_sar_panel(1.2)
    profile = _sar_profile(returns, weights)
    original = _global_scalar_minimize(profile.objective, profile.bounds)
    assert original.x == profile.bounds[1]
    assert _polish_sar_profile(profile, original) is original
    assert sar_qmle(returns, weights, compute_se=False)["rho_hat"] == profile.bounds[1]


def test_sar_polish_retains_flat_unbracketed_profile() -> None:
    """An unidentified zero interaction must not acquire an arbitrary root."""
    returns, _weights = _nilpotent_sar_panel(0.37)
    profile = _sar_profile(returns, np.zeros((2, 2)))
    original = _global_scalar_minimize(profile.objective, profile.bounds)
    np.testing.assert_equal(_sar_profile_score(profile, original.x), 0.0)
    assert _polish_sar_profile(profile, original) is original


def test_sar_polish_preserves_search_failure_status() -> None:
    """Resolving a local score must not conceal an earlier failed refinement."""
    returns, weights = _nilpotent_sar_panel(0.37123456789)
    profile = _sar_profile(returns, weights)
    original = replace(
        _global_scalar_minimize(profile.objective, profile.bounds), converged=False
    )
    polished = _polish_sar_profile(profile, original)
    assert not polished.converged
    np.testing.assert_allclose(polished.x, 0.37123456789, rtol=0.0, atol=2e-14)


def test_sar_polish_rejects_materially_worse_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A score root cannot replace the global winner with a worse basin."""
    returns, weights = _nilpotent_sar_panel(0.37)
    profile = _sar_profile(returns, weights)
    original = _global_scalar_minimize(profile.objective, profile.bounds)
    displaced_root = original.x + 0.01
    monkeypatch.setattr(
        "pipeline.stages.papers.paper5.estimate._sar_profile_score",
        lambda _profile, rho: displaced_root - rho,
    )
    assert _polish_sar_profile(profile, original) is original

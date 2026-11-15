"""Boundary tests for the Paper 5 estimator-validation surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray


def _validation_module() -> ModuleType:
    """Import the validation surface only when its optional oracles exist."""
    pytest.importorskip("libpysal")
    pytest.importorskip("spreg")
    from . import estimator_validation as validate_estimators  # noqa: PLC0415

    return validate_estimators


def test_synthetic_validation_materializes_numpy_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass a NumPy matrix to validation-only libpysal/spreg adapters."""
    validate_estimators = _validation_module()
    seen_weights: list[NDArray[np.float64]] = []

    def fake_local_estimator(
        _returns: NDArray[np.float64], _weights: NDArray[np.float64]
    ) -> dict[str, float]:
        return {"rho_hat": 0.0}

    def fake_spreg(
        _returns: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> dict[str, Any]:
        assert isinstance(weights, np.ndarray)
        seen_weights.append(weights)
        return {
            "ml_lag_rho": 0.0,
            "gm_lag_rho": 0.0,
            "gm_lag_singular_instruments": False,
        }

    monkeypatch.setattr(validate_estimators, "sar_qmle", fake_local_estimator)
    monkeypatch.setattr(validate_estimators, "sar_gmm", fake_local_estimator)
    monkeypatch.setattr(validate_estimators, "_spreg_estimates", fake_spreg)

    results = validate_estimators.run_synthetic_grid(n=4, t_obs=8, smoke=True)

    assert len(results) == 1
    assert len(seen_weights) == 1
    assert seen_weights[0].shape == (4, 4)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.requires_optional("libpysal")
@pytest.mark.requires_optional("spreg")
def test_spreg_adapter_preserves_mean_cross_section_and_weights() -> None:
    """Match the adapter's ML-Lag rho to ``spreg`` on one fixed panel."""
    validate_estimators = _validation_module()
    from libpysal.weights import full2W  # noqa: PLC0415
    from spreg import ML_Lag  # noqa: PLC0415

    weights_matrix = np.array(
        [
            [0.0, 0.65, 0.35, 0.0, 0.0],
            [0.0, 0.0, 0.55, 0.45, 0.0],
            [0.0, 0.0, 0.0, 0.60, 0.40],
            [0.30, 0.0, 0.0, 0.0, 0.70],
            [0.40, 0.60, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    returns = np.array(
        [
            [1.1, -0.2, 0.7, 1.8, -1.3],
            [1.3, -0.1, 0.6, 1.9, -1.1],
            [0.9, -0.3, 0.8, 1.7, -1.4],
        ],
        dtype=np.float64,
    )

    observed = validate_estimators._spreg_estimates(  # noqa: SLF001
        returns, weights_matrix
    )

    reference_weights = full2W(weights_matrix)
    reference_weights.transform = "O"
    np.testing.assert_allclose(
        np.asarray(reference_weights.full()[0]),
        weights_matrix,
        rtol=0.0,
        atol=0.0,
    )
    reference = ML_Lag(
        returns.mean(axis=0).reshape(-1, 1),
        np.zeros((weights_matrix.shape[0], 0)),
        w=reference_weights,
        method="full",
    )
    np.testing.assert_allclose(
        observed["ml_lag_rho"],
        float(reference.rho),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["full", "LU", "ord"])
def test_local_qmle_matches_spreg_single_cross_section(
    *, directed: bool, method: str
) -> None:
    """Compare the local estimator, not the oracle adapter, on identical data."""
    pytest.importorskip("libpysal")
    pytest.importorskip("spreg")
    from libpysal.weights import full2W  # noqa: PLC0415
    from pipeline.stages.papers.paper5.estimate import (  # noqa: PLC0415
        sar_qmle,
        sar_rho_se,
    )
    from spreg import ML_Lag  # noqa: PLC0415

    rng = np.random.default_rng(126)
    n = 40
    w = np.zeros((n, n))
    for i in range(n):
        w[i, (i + 1) % n] = 0.8 if directed else 0.5
        # Nonreciprocal support makes spreg ORD select its complex-spectrum
        # path. Reciprocal support with unequal edge weights is incorrectly
        # classified as symmetrizable by that oracle's topology-only check.
        w[i, (i + 2 if directed else i - 1) % n] = 0.2 if directed else 0.5
    y = np.linalg.solve(np.eye(n) - 0.4 * w, rng.normal(size=n))
    local = sar_qmle(np.asarray(y[None, :], dtype=np.float64), w)
    np.testing.assert_allclose(
        sar_rho_se(np.asarray(y[None, :], dtype=np.float64), w, local["rho_hat"]),
        local["rho_se"],
        atol=1e-12,
    )
    reference = ML_Lag(y[:, None], np.empty((n, 0)), full2W(w), method=method)
    np.testing.assert_allclose(local["rho_hat"], reference.rho, atol=2e-6)
    np.testing.assert_allclose(local["alpha_hat"], reference.betas[0, 0], atol=2e-6)
    np.testing.assert_allclose(local["sigma2_hat"], reference.sig2, atol=2e-6)
    np.testing.assert_allclose(local["loglik"], reference.logll, atol=2e-6)

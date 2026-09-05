"""Pure numerical kernel for Paper 3's diversification certificate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MATRIX_NDIM = 2
_MIN_RETURN_OBSERVATIONS = 2


def _certificate_error(message: str) -> None:
    raise ValueError(message)


@dataclass(frozen=True, slots=True)
class CertificateCell:
    """One portfolio-by-transmission-scenario certificate evaluation."""

    portfolio: str
    lipschitz_inverse_scale: float
    common_slack: float
    certificate_credit: float
    variance_cap: float


def _validate_distance_matrix(distance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distance, dtype=np.float64)
    if (
        matrix.ndim != _MATRIX_NDIM
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.size
    ):
        _certificate_error("distance matrix must be nonempty and square")
    if not np.isfinite(matrix).all():
        _certificate_error("distance matrix must contain only finite values")
    if np.any(matrix < 0.0):
        _certificate_error("distance matrix must be nonnegative")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        _certificate_error("distance matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
        _certificate_error("distance matrix diagonal must be zero")
    return matrix


def exposure_floor(
    distance: np.ndarray,
    lipschitz_inverse_scale: float,
    common_slack: float,
) -> np.ndarray:
    r"""Return ``[W_ij / L - 2 tau]_+`` for a common slack scenario."""
    matrix = _validate_distance_matrix(distance)
    if not np.isfinite(lipschitz_inverse_scale) or lipschitz_inverse_scale <= 0.0:
        _certificate_error("lipschitz_inverse_scale must be finite and positive")
    if not np.isfinite(common_slack) or common_slack < 0.0:
        _certificate_error("common_slack must be finite and nonnegative")
    floor = np.maximum(0.0, matrix / lipschitz_inverse_scale - 2.0 * common_slack)
    np.fill_diagonal(floor, 0.0)
    return floor


def certificate_credit(floor: np.ndarray, weights: np.ndarray) -> float:
    r"""Compute ``0.5 q' (floor squared) q`` for long-only weights."""
    matrix = _validate_distance_matrix(floor)
    q = np.asarray(weights, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] != matrix.shape[0]:
        _certificate_error("weights must match the distance-matrix roster")
    if not np.isfinite(q).all() or np.any(q < 0.0):
        _certificate_error("weights must be finite and nonnegative")
    mass = float(q.sum())
    if mass <= 0.0:
        _certificate_error("weights must have positive mass")
    q = q / mass
    return float(0.5 * q @ np.square(matrix) @ q)


def equal_weights(n_assets: int) -> np.ndarray:
    """Return equal long-only weights for a nonempty asset roster."""
    if n_assets <= 0:
        _certificate_error("n_assets must be positive")
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float64)


def inverse_volatility_weights(returns: np.ndarray) -> np.ndarray:
    """Return full-sample inverse-volatility weights for a dense return panel."""
    panel = np.asarray(returns, dtype=np.float64)
    if (
        panel.ndim != _MATRIX_NDIM
        or panel.shape[0] < _MIN_RETURN_OBSERVATIONS
        or panel.shape[1] == 0
    ):
        _certificate_error("returns must contain at least two dates and one asset")
    if not np.isfinite(panel).all():
        _certificate_error("returns must contain only finite values")
    volatility = np.std(panel, axis=0, ddof=1)
    if not np.isfinite(volatility).all() or np.any(volatility <= 0.0):
        _certificate_error("every asset must have finite positive volatility")
    inverse = np.reciprocal(volatility)
    return inverse / inverse.sum()


def certificate_surface(
    distance: np.ndarray,
    portfolios: dict[str, np.ndarray],
    lipschitz_inverse_scales: tuple[float, ...],
    common_slacks: tuple[float, ...],
) -> tuple[CertificateCell, ...]:
    """Evaluate a deterministic grid of portfolio certificate scenarios."""
    matrix = _validate_distance_matrix(distance)
    if not portfolios:
        _certificate_error("at least one portfolio is required")
    if not lipschitz_inverse_scales or not common_slacks:
        _certificate_error("the L and tau grids must both be nonempty")
    cells: list[CertificateCell] = []
    for portfolio, weights in portfolios.items():
        if not portfolio:
            _certificate_error("portfolio labels must be nonempty")
        for scale in lipschitz_inverse_scales:
            for slack in common_slacks:
                floor = exposure_floor(matrix, scale, slack)
                credit = certificate_credit(floor, weights)
                cells.append(
                    CertificateCell(
                        portfolio=portfolio,
                        lipschitz_inverse_scale=float(scale),
                        common_slack=float(slack),
                        certificate_credit=credit,
                        variance_cap=1.0 - credit,
                    )
                )
    return tuple(cells)

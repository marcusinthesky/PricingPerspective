"""Pure contracts for Paper 3's frozen chronological certificate validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, t

_MIN_INTERVAL_OBSERVATIONS = 2
_SUM_TOLERANCE = 1e-8
_BOUND_TOLERANCE = 1e-9
_MIN_BLOCK_SERIES = 2
_SHORT_BLOCK_SERIES = 8
_AUTOCORRELATION_CUTOFF = 0.1
_KKT_TOLERANCE = 1e-6


class CertificateValidationError(ValueError):
    """Raised when the chronological validation contract is violated."""


def _validation_error(message: str) -> None:
    raise CertificateValidationError(message)


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    """Calibration evidence for one predeclared transmission cell."""

    lipschitz_scale: float
    slack: float
    margins: np.ndarray
    credits: np.ndarray


@dataclass(frozen=True, slots=True)
class CertificateOptimization:
    """Convex-program solution and fail-closed numerical diagnostics."""

    weights: np.ndarray
    credit: float
    success: bool
    iterations: int
    kkt_residual: float
    minimum_centered_kernel_eigenvalue: float


def capital_to_budget_weights(
    capital: np.ndarray, volatility: np.ndarray
) -> np.ndarray:
    """Map capital weights x to perfect-dependence volatility-budget shares q."""
    x = np.asarray(capital, dtype=np.float64)
    sigma = np.asarray(volatility, dtype=np.float64)
    if x.ndim != 1 or sigma.shape != x.shape or np.any(x < 0) or np.any(sigma <= 0):
        _validation_error(
            "capital and positive volatility vectors must have equal shape"
        )
    budget = x * sigma
    if not np.isfinite(budget).all() or budget.sum() <= 0:
        _validation_error("volatility budget must be finite and positive")
    return budget / budget.sum()


def one_sided_mean_lower(values: np.ndarray, confidence: float) -> float:
    """Student-t one-sided lower endpoint for a mean."""
    sample = np.asarray(values, dtype=np.float64)
    if (
        sample.ndim != 1
        or sample.size < _MIN_INTERVAL_OBSERVATIONS
        or not np.isfinite(sample).all()
    ):
        _validation_error("interval requires at least two finite observations")
    return float(
        sample.mean()
        - t.ppf(confidence, sample.size - 1) * sample.std(ddof=1) / np.sqrt(sample.size)
    )


def coverage_lower(successes: int, trials: int, confidence: float) -> float:
    """Exact one-sided Clopper--Pearson lower endpoint."""
    if trials <= 0 or not 0 <= successes <= trials:
        _validation_error("coverage counts are invalid")
    return (
        0.0
        if successes == 0
        else float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))
    )


def select_calibration_cell(
    cells: tuple[CalibrationCell, ...], *, confidence: float, coverage_target: float
) -> tuple[float, float]:
    """Select only on calibration evidence; fall back to the zero-credit benchmark."""
    passing: list[tuple[float, float, float]] = []
    for cell in cells:
        covered = int(np.count_nonzero(cell.margins >= 0.0))
        if (
            one_sided_mean_lower(cell.margins, confidence) >= 0.0
            and coverage_lower(covered, cell.margins.size, confidence)
            >= coverage_target
        ):
            passing.append(
                (
                    one_sided_mean_lower(cell.credits, confidence),
                    cell.lipschitz_scale,
                    cell.slack,
                )
            )
    if not passing:
        return (float("inf"), float("inf"))
    passing.sort(key=lambda row: (-row[0], row[1], row[2]))
    return passing[0][1:]


def stationary_block_indices(
    n: int, expected_length: int, draws: int, seed: int
) -> np.ndarray:
    """Deterministic stationary-bootstrap indices with a short-series fallback."""
    if n <= 0 or expected_length <= 0 or draws <= 0:
        _validation_error("bootstrap dimensions must be positive")
    block = min(expected_length, n)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, n), dtype=np.int64)
    restart = 1.0 / block
    for draw in range(draws):
        out[draw, 0] = rng.integers(n)
        for pos in range(1, n):
            out[draw, pos] = (
                rng.integers(n)
                if rng.random() < restart
                else (out[draw, pos - 1] + 1) % n
            )
    return out


def select_stationary_block_length(series: np.ndarray) -> int:
    """Choose a deterministic block length from an ordered scalar series.

    The first lag whose sample autocorrelation falls below ``0.1`` closes the
    dependence run.  Series shorter than eight observations use the explicit
    cube-root fallback, bounded by the observed length.
    """
    values = np.asarray(series, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < _MIN_BLOCK_SERIES
        or not np.isfinite(values).all()
    ):
        _validation_error("block-length selection requires a finite scalar series")
    fallback = min(
        values.size, max(_MIN_BLOCK_SERIES, round(values.size ** (1.0 / 3.0)))
    )
    if (
        values.size < _SHORT_BLOCK_SERIES
        or float(np.std(values)) <= np.finfo(values.dtype).eps
    ):
        return fallback
    centered = values - values.mean()
    denominator = float(centered @ centered)
    maximum_lag = min(values.size // 2, int(np.ceil(np.sqrt(values.size))))
    for lag in range(1, maximum_lag + 1):
        autocorrelation = float(centered[:-lag] @ centered[lag:] / denominator)
        if autocorrelation < _AUTOCORRELATION_CUTOFF:
            return lag
    return max(1, maximum_lag)


def solve_certificate(squared_floor: np.ndarray, cap: float) -> CertificateOptimization:
    """Maximize certificate credit on the capped simplex as a convex program.

    A squared Hilbert distance is conditionally negative definite.  Therefore
    ``0.5 q' D q`` is concave on the affine hyperplane ``sum(q) = 1`` and its
    negative is a convex QP.  The centered Gram matrix is the fail-closed gate;
    checking ``D`` itself for positive definiteness would test the wrong object.
    """
    matrix = np.asarray(squared_floor, dtype=np.float64)
    n = matrix.shape[0]
    if (
        matrix.shape != (n, n)
        or cap * n < 1.0
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0)
        or not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0)
    ):
        _validation_error("kernel/cap cannot define a feasible capped simplex")
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * centering @ matrix @ centering
    eigenvalues = np.linalg.eigvalsh(gram)
    scale = max(1.0, float(np.linalg.norm(gram, ord=2)))
    tolerance = 1e-10 * scale
    if (
        eigenvalues[0] < -tolerance
        or np.count_nonzero(eigenvalues > tolerance) != n - 1
    ):
        _validation_error("centered certificate kernel is not positive definite")

    def objective(q: np.ndarray) -> float:
        return -0.5 * float(q @ matrix @ q)

    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        jac=lambda q: -(matrix @ q),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints={"type": "eq", "fun": lambda q: q.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if (
        not result.success
        or abs(result.x.sum() - 1.0) > _SUM_TOLERANCE
        or result.x.min() < -_BOUND_TOLERANCE
        or result.x.max() > cap + _BOUND_TOLERANCE
    ):
        _validation_error("certificate optimization failed convergence or feasibility")
    weights = np.asarray(result.x)
    gradient = -(matrix @ weights)
    free = (weights > _BOUND_TOLERANCE) & (weights < cap - _BOUND_TOLERANCE)
    lagrange = -float(np.mean(gradient[free])) if np.any(free) else 0.0
    stationarity = gradient + lagrange
    residuals = [abs(weights.sum() - 1.0)]
    residuals.extend(np.abs(stationarity[free]))
    residuals.extend(np.maximum(0.0, -stationarity[weights <= _BOUND_TOLERANCE]))
    residuals.extend(np.maximum(0.0, stationarity[weights >= cap - _BOUND_TOLERANCE]))
    kkt_residual = float(max(residuals, default=0.0))
    if kkt_residual > _KKT_TOLERANCE:
        _validation_error(
            f"certificate optimization failed KKT gate: {kkt_residual:.3e}"
        )
    return CertificateOptimization(
        weights=weights,
        credit=float(-result.fun),
        success=True,
        iterations=int(result.nit),
        kkt_residual=kkt_residual,
        minimum_centered_kernel_eigenvalue=float(eigenvalues[1]),
    )


def optimize_certificate(kernel: np.ndarray, cap: float) -> tuple[np.ndarray, float]:
    """Compatibility wrapper returning the optimized weights and point credit."""
    result = solve_certificate(kernel, cap)
    return result.weights, result.credit


def residual_bound(point_bound: float, delta: float) -> float:
    """Add an assumed residual-covariance budget to a systematic bound."""
    if delta < 0:
        _validation_error("residual budget must be nonnegative")
    return point_bound + delta

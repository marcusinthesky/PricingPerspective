"""Numerical witnesses for the Paper 5 SAR asymptotic assumptions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pipeline.stages.papers.paper5._gate.driver import _load_gate_operator

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True)
class Paper5NetworkCertificatePaths:
    """Inputs and output for the network-theory certificate."""

    shared_barycentre_dir: Path
    qmle_results: Path
    output_json: Path


@dataclass(frozen=True)
class Paper5NetworkCertificateConfig:
    """Frozen operator identity and numerical tolerances."""

    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    barycentre_arm_id: str = "wasserstein_w2_loo"
    geometry_id: str = "wasserstein_w2"
    tolerance: float = 1e-12
    max_iterations: int = 100_000


def operator_infinity_norm(matrix: NDArray[np.float64]) -> float:
    """Return the maximum absolute row-sum norm used by the Lean SAR layer."""
    return float(np.max(np.sum(np.abs(matrix), axis=1)))


def dobrushin_coefficient(weights: NDArray[np.float64]) -> float:
    """Return the total-variation contraction coefficient of a Markov matrix."""
    pairwise_l1 = np.sum(
        np.abs(weights[:, None, :] - weights[None, :, :]),
        axis=2,
    )
    return float(0.5 * np.max(pairwise_l1))


def stationary_distribution(
    weights: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int]:
    """Compute the stationary row distribution by deterministic power iteration."""
    distribution = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        updated = distribution @ weights
        if np.linalg.norm(updated - distribution, ord=1) <= tolerance:
            updated /= updated.sum()
            return updated, iteration
        distribution = updated
    message = "stationary-distribution iteration did not converge"
    raise RuntimeError(message)


def build_network_certificate(
    weights: NDArray[np.float64],
    pooled_rhos: dict[str, float],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> dict[str, object]:
    """Build structural, Perron, and finite-rho numerical witnesses."""
    matrix = np.asarray(weights, dtype=np.float64)
    n_assets = len(matrix)
    minimum_market_size = 2
    if matrix.shape != (n_assets, n_assets) or n_assets < minimum_market_size:
        message = "weights must be a nontrivial square matrix"
        raise ValueError(message)
    row_sum_residual = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    minimum_weight = float(matrix.min())
    if row_sum_residual > tolerance or minimum_weight < -tolerance:
        message = "weights are not nonnegative and row stochastic"
        raise ValueError(message)

    squared = matrix @ matrix
    minimum_squared_weight = float(squared.min())
    primitive_power_two = minimum_squared_weight > tolerance
    if not primitive_power_two:
        message = "W squared is not strictly positive at the certificate tolerance"
        raise ValueError(message)

    delta = dobrushin_coefficient(matrix)
    if not 0.0 <= delta < 1.0:
        message = "Dobrushin coefficient does not certify contraction"
        raise ValueError(message)
    pi, iterations = stationary_distribution(
        matrix,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    projection = np.ones((n_assets, 1), dtype=np.float64) * pi[None, :]
    stationary_residual = float(np.linalg.norm(pi @ matrix - pi, ord=1))

    finite_rho: dict[str, object] = {}
    identity = np.eye(n_assets, dtype=np.float64)
    for variant, rho in pooled_rhos.items():
        if not 0.0 <= rho < 1.0:
            message = f"{variant} rho must lie in [0, 1)"
            raise ValueError(message)
        normalized = (1.0 - rho) * np.linalg.inv(identity - rho * matrix)
        actual_error = operator_infinity_norm(normalized - projection)
        contraction_bound = 2.0 * (1.0 - rho) / (1.0 - rho * delta)
        if actual_error > contraction_bound + 100.0 * tolerance:
            message = f"{variant} finite-rho error exceeds contraction bound"
            raise RuntimeError(message)
        complement_norm = operator_infinity_norm(rho * (matrix - projection))
        lean_neumann_bound: float | None = None
        if complement_norm < 1.0:
            lean_neumann_bound = (
                abs(1.0 - rho)
                * (1.0 / (1.0 - complement_norm))
                * operator_infinity_norm(identity - projection)
            )
        finite_rho[variant] = {
            "rho": rho,
            "normalized_resolvent_error_linf": actual_error,
            "dobrushin_upper_bound_linf": contraction_bound,
            "bound_slack": contraction_bound - actual_error,
            "complement_norm": complement_norm,
            "lean_neumann_region": complement_norm < 1.0,
            "lean_neumann_upper_bound_linf": lean_neumann_bound,
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "matrix_norm": "maximum absolute row sum (L-infinity operator norm)",
        "n_assets": n_assets,
        "structural": {
            "row_sum_residual_max": row_sum_residual,
            "minimum_weight": minimum_weight,
            "maximum_diagonal_weight": float(np.max(np.abs(np.diag(matrix)))),
            "primitive_witness_power": 2,
            "minimum_w_squared_weight": minimum_squared_weight,
            "w_squared_strictly_positive": primitive_power_two,
        },
        "perron": {
            "dobrushin_coefficient": delta,
            "geometric_bound_constant": 2.0,
            "geometric_bound_rate": delta,
            "stationary_distribution_minimum": float(pi.min()),
            "stationary_distribution_maximum": float(pi.max()),
            "stationary_sum_residual": float(abs(pi.sum() - 1.0)),
            "stationary_l1_residual": stationary_residual,
            "power_iterations": iterations,
        },
        "finite_rho": finite_rho,
        "formal_scope": {
            "lean_predicate": "HasGeometricPerronBound",
            "lean_bridge": "isPerronLimit_of_geometric_bound",
            "lean_finite_rho_theorem": "normalized_leontief_error_le",
            "numerical_role": (
                "floating-point witness for the frozen operator; not a kernel-checked "
                "proof of the parquet artifact"
            ),
        },
    }
    return payload


def _pooled_rhos_from_qmle_payload(payload: object) -> dict[str, float]:
    """Validate the stored pooled-QMLE structure before extracting rho values."""
    if not isinstance(payload, dict):
        message = "QMLE results JSON must be an object"
        raise TypeError(message)
    results = payload.get("results")
    if not isinstance(results, dict):
        message = "QMLE results JSON is missing its results object"
        raise TypeError(message)
    pooled = results.get("pooled_2023_2026")
    if not isinstance(pooled, dict):
        message = "QMLE results JSON is missing pooled_2023_2026"
        raise TypeError(message)

    pooled_rhos: dict[str, float] = {}
    for variant, record in pooled.items():
        if not isinstance(variant, str) or not isinstance(record, dict):
            message = "QMLE pooled results must map variant names to objects"
            raise TypeError(message)
        rho_hat = record.get("rho_hat")
        if isinstance(rho_hat, bool) or not isinstance(rho_hat, (int, float)):
            message = f"QMLE pooled result {variant!r} has no numeric rho_hat"
            raise TypeError(message)
        pooled_rhos[variant] = float(rho_hat)
    return pooled_rhos


def run_paper5_network_certificate(
    paths: Paper5NetworkCertificatePaths,
    config: Paper5NetworkCertificateConfig,
) -> dict[str, object]:
    """Load the frozen operator and pooled QMLE estimates, then write the witness."""
    _tickers, weights, _summary = _load_gate_operator(
        paths.shared_barycentre_dir,
        config,
    )
    qmle_payload: object = json.loads(paths.qmle_results.read_text(encoding="utf-8"))
    pooled_rhos = _pooled_rhos_from_qmle_payload(qmle_payload)
    payload = build_network_certificate(
        weights,
        pooled_rhos,
        tolerance=config.tolerance,
        max_iterations=config.max_iterations,
    )
    paths.output_json.parent.mkdir(parents=True, exist_ok=True)
    paths.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "Paper5NetworkCertificateConfig",
    "Paper5NetworkCertificatePaths",
    "build_network_certificate",
    "dobrushin_coefficient",
    "operator_infinity_norm",
    "run_paper5_network_certificate",
    "stationary_distribution",
]

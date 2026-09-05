"""Fixed-W2 diagnostics for the random-exposure covariance envelope.

The formal model supplies a sharp covariance envelope for the latent exposure
law.  Article clouds are observed in unit-chord coordinates, so this module
uses the corresponding standardized total-return geometry,
``c_actual = 2 - 2 rho``.  ``L`` and ``tau`` are maintained scenario scales;
they are not estimated from the return panel.

The substitution of the standardized *total*-return correlation for the
systematic covariance is a maintained restriction, not an identity: where
idiosyncratic variance is present and cross-firm residuals are uncorrelated it
makes the reported coverage share an upper bound on the coverage of the
systematic claim.  The manuscript states this where the table is discussed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

SCENARIO_LIPSCHITZ = (1.0, 1.25, 1.5)
SCENARIO_SLACK = (0.0, 0.05, 0.10)
_BOOTSTRAP_QUANTILES = (2.5, 97.5)
_ALIGNED_VECTORS_ERROR = (
    "W2, correlation, and endpoint vectors must be aligned one-dimensional arrays"
)
_NUMERIC_VECTORS_ERROR = "W2 distances and correlations must be finite numeric vectors"
_CORRELATION_ERROR = "correlations must be finite before clipping"
_W2_ERROR = "W2 distances must be finite and nonnegative"
_UNIQUE_TICKERS_ERROR = "endpoint membership requires a unique ticker list"
_ENDPOINT_MEMBERSHIP_ERROR = (
    "endpoint membership requires every endpoint ticker in the roster"
)
_NODE_COUNT_SHAPE_ERROR = "node count shape must be (n_draws, len(tickers))"
_NODE_COUNT_INTEGRALITY_ERROR = (
    "node count integrality requires finite, nonnegative, integer-valued counts"
)
_NODE_COUNT_DIMENSIONS = 2


class RandomExposureInputError(ValueError):
    """Report an invalid fixed-W2 diagnostic input."""


@dataclass(frozen=True)
class RandomExposureInputs:
    """Aligned dyadic inputs for the fixed-W2 envelope diagnostic.

    The five arrays travel together because they are one dyad-indexed table
    read column-wise: ``w2`` and ``correlation`` are per-dyad values, the two
    endpoint vectors index into ``tickers``, and ``node_counts`` is the shared
    bootstrap schedule those endpoints are weighted by.  Passing them
    separately would let a caller misalign the text side against the return
    side without any type saying so.
    """

    w2: NDArray[np.float64]
    correlation: NDArray[np.float64]
    firm_i: NDArray[np.str_]
    firm_j: NDArray[np.str_]
    tickers: list[str]
    node_counts: NDArray[np.int64]


def _weighted_share(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a node-bootstrap weighted mean for a dyadic indicator."""
    denominator = float(np.sum(weights))
    return (
        float(np.sum(weights * values) / denominator) if denominator else float("nan")
    )


def _dyad_weights(inputs: RandomExposureInputs) -> NDArray[np.float64]:
    """Return the per-draw endpoint-multiplicity weights for every dyad.

    Computed once: the weights depend only on the resampling schedule and the
    endpoint index, not on the maintained scenario, so recomputing them inside
    the scenario loop would repeat the same product nine times.
    """
    positions = {ticker: index for index, ticker in enumerate(inputs.tickers)}
    i_positions = np.array(
        [positions[str(value)] for value in inputs.firm_i], dtype=int
    )
    j_positions = np.array(
        [positions[str(value)] for value in inputs.firm_j], dtype=int
    )
    counts = np.asarray(inputs.node_counts, dtype=np.int64)
    return np.asarray(counts[:, i_positions] * counts[:, j_positions], dtype=np.float64)


def _bootstrap_interval(
    indicator: np.ndarray, weights: NDArray[np.float64]
) -> tuple[float, float]:
    """Return the percentile interval of a weighted share across draws."""
    shares = np.asarray(
        [
            _weighted_share(indicator.astype(float), draw_weights)
            for draw_weights in weights
            if float(np.sum(draw_weights)) > 0.0
        ],
        dtype=float,
    )
    if shares.size == 0:
        message = "node bootstrap contains no positive-weight draws"
        raise RandomExposureInputError(message)
    return (
        float(np.percentile(shares, _BOOTSTRAP_QUANTILES[0])),
        float(np.percentile(shares, _BOOTSTRAP_QUANTILES[1])),
    )


def _validated_node_counts(
    inputs: RandomExposureInputs, ticker_count: int
) -> NDArray[np.int64]:
    """Validate and coerce the node-bootstrap count matrix."""
    try:
        counts_raw = np.asarray(inputs.node_counts)
    except (TypeError, ValueError) as exc:
        raise RandomExposureInputError(_NODE_COUNT_SHAPE_ERROR) from exc
    if (
        counts_raw.ndim != _NODE_COUNT_DIMENSIONS
        or counts_raw.shape[0] == 0
        or counts_raw.shape[1] != ticker_count
    ):
        raise RandomExposureInputError(_NODE_COUNT_SHAPE_ERROR)
    try:
        counts = np.asarray(counts_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RandomExposureInputError(_NODE_COUNT_INTEGRALITY_ERROR) from exc
    if (
        not np.all(np.isfinite(counts))
        or np.any(counts < 0.0)
        or not np.all(counts == np.floor(counts))
        or np.any(counts > np.iinfo(np.int64).max)
    ):
        raise RandomExposureInputError(_NODE_COUNT_INTEGRALITY_ERROR)
    return counts.astype(np.int64)


def _validated_inputs(inputs: RandomExposureInputs) -> RandomExposureInputs:
    """Validate every aligned array before lossy coercion or clipping."""
    try:
        w2_raw = np.asarray(inputs.w2)
        correlation_raw = np.asarray(inputs.correlation)
        firm_i_raw = np.asarray(inputs.firm_i)
        firm_j_raw = np.asarray(inputs.firm_j)
    except (TypeError, ValueError) as exc:
        raise RandomExposureInputError(_ALIGNED_VECTORS_ERROR) from exc
    if any(
        array.ndim != 1 for array in (w2_raw, correlation_raw, firm_i_raw, firm_j_raw)
    ) or not (
        w2_raw.shape == correlation_raw.shape == firm_i_raw.shape == firm_j_raw.shape
    ):
        raise RandomExposureInputError(_ALIGNED_VECTORS_ERROR)
    try:
        w2_values = np.asarray(w2_raw, dtype=float)
        correlation_values = np.asarray(correlation_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RandomExposureInputError(_NUMERIC_VECTORS_ERROR) from exc
    if not np.all(np.isfinite(correlation_values)):
        raise RandomExposureInputError(_CORRELATION_ERROR)
    if not np.all(np.isfinite(w2_values)) or np.any(w2_values < 0.0):
        raise RandomExposureInputError(_W2_ERROR)

    tickers = [str(ticker) for ticker in inputs.tickers]
    if len(set(tickers)) != len(tickers):
        raise RandomExposureInputError(_UNIQUE_TICKERS_ERROR)
    firm_i = np.asarray(firm_i_raw, dtype=str)
    firm_j = np.asarray(firm_j_raw, dtype=str)
    ticker_set = set(tickers)
    if any(value not in ticker_set for value in (*firm_i, *firm_j)):
        raise RandomExposureInputError(_ENDPOINT_MEMBERSHIP_ERROR)
    return RandomExposureInputs(
        w2=w2_values,
        correlation=np.clip(correlation_values, -1.0, 1.0),
        firm_i=firm_i,
        firm_j=firm_j,
        tickers=tickers,
        node_counts=_validated_node_counts(inputs, len(tickers)),
    )


def _scenario_rows(
    inputs: RandomExposureInputs, actual_cost: NDArray[np.float64]
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Compute observed and node-bootstrap scenario classifications."""
    w2 = inputs.w2
    weights = _dyad_weights(inputs)
    observed_rows: list[dict[str, object]] = []
    dyad_frames: list[pd.DataFrame] = []
    for lipschitz in SCENARIO_LIPSCHITZ:
        for slack in SCENARIO_SLACK:
            tau_ij = 2.0 * slack
            lower_w2 = np.maximum(w2 / lipschitz - tau_ij, 0.0)
            upper_w2 = lipschitz * w2 + tau_ij
            delta_lower = actual_cost - upper_w2**2
            delta_upper = actual_cost - lower_w2**2
            covered = delta_lower >= 0.0
            violated = delta_upper < 0.0
            unresolved = ~(covered | violated)
            coverage_ci = _bootstrap_interval(covered, weights)
            violation_ci = _bootstrap_interval(violated, weights)
            unresolved_ci = _bootstrap_interval(unresolved, weights)
            observed_rows.append(
                {
                    "L": float(lipschitz),
                    "tau": float(slack),
                    "tau_ij": float(tau_ij),
                    "n_dyads": int(w2.size),
                    "coverage_share": float(np.mean(covered)),
                    "violation_share": float(np.mean(violated)),
                    "unresolved_share": float(np.mean(unresolved)),
                    "coverage_ci_low": coverage_ci[0],
                    "coverage_ci_high": coverage_ci[1],
                    "violation_ci_low": violation_ci[0],
                    "violation_ci_high": violation_ci[1],
                    "unresolved_ci_low": unresolved_ci[0],
                    "unresolved_ci_high": unresolved_ci[1],
                    "delta_lower_mean": float(np.mean(delta_lower)),
                    "delta_upper_mean": float(np.mean(delta_upper)),
                    "delta_lower_min": float(np.min(delta_lower)),
                    "delta_upper_max": float(np.max(delta_upper)),
                }
            )
            dyad_frames.append(
                pd.DataFrame(
                    {
                        "ticker_i": inputs.firm_i,
                        "ticker_j": inputs.firm_j,
                        "L": float(lipschitz),
                        "tau": float(slack),
                        "w2_distance": w2,
                        "actual_cost": actual_cost,
                        "w2_lower": lower_w2,
                        "w2_upper": upper_w2,
                        "delta_lower": delta_lower,
                        "delta_upper": delta_upper,
                        "robust_coverage": covered,
                        "robust_violation": violated,
                        "unresolved": unresolved,
                    }
                )
            )
    return observed_rows, pd.concat(dyad_frames, ignore_index=True)


def fixed_w2_random_exposure_diagnostic(
    inputs: RandomExposureInputs,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return the fixed-W2 envelope diagnostic and its auditable dyad frame.

    The null-compatible quantity is the transport excess
    ``Delta = c_actual - W2(P_i, P_j)^2 >= 0``.  Since the latent exposure W2 is
    only bracketed by the bi-Lipschitz/slack transfer, each scenario reports
    robust coverage, robust violation, or unresolved dyads.

    Args:
        inputs: Aligned dyadic W2, return-correlation, endpoint, and
            node-bootstrap inputs.

    Returns:
        The scenario summary and the per-dyad classification frame.

    Raises:
        RandomExposureInputError: If any vector, endpoint roster, or bootstrap
            count contract is invalid.

    """
    checked = _validated_inputs(inputs)
    actual_cost = 2.0 - 2.0 * checked.correlation
    scenarios, dyads = _scenario_rows(checked, actual_cost)
    summary: dict[str, object] = {
        "status": "computed",
        "theory_test": "Delta_ij = c_actual_ij - W2_exposure_ij^2 >= 0",
        "distance_id": "wasserstein_w2",
        "distance_scope": "qwen3-embedding-8b unit representation",
        "return_moment_geometry": (
            "standardized total-return moments: v_i=v_j=1 and kappa_ij=rho_ij; "
            "therefore c_actual=2-2*rho. Substituting the total-return "
            "correlation for the systematic covariance is a maintained "
            "restriction, so coverage is an upper bound on the systematic claim"
        ),
        "scale_contract": (
            "L and tau are maintained dimensionless scenario scales; no return-based "
            "estimation, holdout, or cross-fitting is used"
        ),
        "classification": {
            "robust_coverage": "c_actual - (L W2 + 2 tau)^2 >= 0",
            "robust_violation": "c_actual - ((W2/L - 2 tau)_+)^2 < 0",
            "unresolved": "otherwise",
        },
        "bootstrap": "multinomial node bootstrap using the dyadic stage schedule",
        "scenarios": scenarios,
    }
    return summary, dyads


__all__ = [
    "SCENARIO_LIPSCHITZ",
    "SCENARIO_SLACK",
    "RandomExposureInputError",
    "RandomExposureInputs",
    "fixed_w2_random_exposure_diagnostic",
]

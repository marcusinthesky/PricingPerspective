"""Pure weighting and loss functions for the Paper 5 economic gate."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pipeline.stages.papers.paper5._gate.contracts import raise_gate_error

FloatMatrix = NDArray[np.float64]
_MATRIX_DIMENSIONS = 2


def validate_weight_matrix(weights: FloatMatrix, *, tolerance: float = 1e-10) -> None:
    """Validate the simplex and leave-one-out invariants needed by the gate."""
    if weights.ndim != _MATRIX_DIMENSIONS or weights.shape[0] != weights.shape[1]:
        raise_gate_error("weight matrix must be square")
    if not np.all(np.isfinite(weights)) or np.min(weights) < -tolerance:
        raise_gate_error("weight matrix must be finite and nonnegative")
    if np.max(np.abs(np.diag(weights))) > tolerance:
        raise_gate_error("weight matrix must have a zero diagonal")
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > tolerance:
        raise_gate_error("every basket must be fully invested")


def equal_support_weights(
    weights: FloatMatrix,
    *,
    active_tolerance: float,
) -> FloatMatrix:
    """Give every active W-flat constituent equal weight, row by row."""
    support = weights > active_tolerance
    support[np.diag_indices_from(support)] = False
    counts = support.sum(axis=1)
    if np.any(counts == 0):
        raise_gate_error("W-flat contains an empty active row")
    result = support.astype(np.float64) / counts[:, None]
    validate_weight_matrix(result)
    return result


def industry_weights(
    tickers: list[str],
    industry_of: dict[str, str],
    sector_of: dict[str, str],
) -> FloatMatrix:
    """Build an industry basket with sector and priced-universe fallbacks."""
    n_assets = len(tickers)
    result = np.zeros((n_assets, n_assets), dtype=np.float64)
    for index, ticker in enumerate(tickers):
        industry = industry_of.get(ticker)
        peers = [
            peer_index
            for peer_index, peer in enumerate(tickers)
            if peer_index != index and industry_of.get(peer) == industry
        ]
        if not peers:
            sector = sector_of.get(ticker)
            peers = [
                peer_index
                for peer_index, peer in enumerate(tickers)
                if peer_index != index and sector_of.get(peer) == sector
            ]
        if not peers:
            peers = [
                peer_index for peer_index in range(n_assets) if peer_index != index
            ]
        result[index, peers] = 1.0 / len(peers)
    validate_weight_matrix(result)
    return result


def correlation_knn_weights(returns: FloatMatrix, *, k: int) -> FloatMatrix:
    """Fit a nonnegative correlation-kNN basket on training returns only."""
    n_assets = returns.shape[1]
    if not 1 <= k < n_assets:
        raise_gate_error("kNN size must lie between one and n_assets - 1")
    correlation = np.asarray(np.corrcoef(returns, rowvar=False), dtype=np.float64)
    np.fill_diagonal(correlation, -np.inf)
    result = np.zeros((n_assets, n_assets), dtype=np.float64)
    for index in range(n_assets):
        neighbours = np.argsort(correlation[index], kind="stable")[::-1][:k]
        positive = np.clip(correlation[index, neighbours], 0.0, None)
        if float(positive.sum()) <= 0.0:
            result[index, neighbours] = 1.0 / k
        else:
            result[index, neighbours] = positive / positive.sum()
    validate_weight_matrix(result)
    return result


def random_matched_sparsity_weights(
    weights: FloatMatrix,
    *,
    draws: int,
    seed: int,
    active_tolerance: float,
) -> NDArray[np.float64]:
    """Draw fixed-seed Dirichlet baskets with W-flat's row sparsities."""
    n_assets = weights.shape[0]
    support_sizes = np.sum(weights > active_tolerance, axis=1)
    if np.any(support_sizes < 1) or np.any(support_sizes >= n_assets):
        raise_gate_error("matched random supports require 1..n_assets-1 entries")
    rng = np.random.default_rng(seed)
    result = np.zeros((draws, n_assets, n_assets), dtype=np.float64)
    for draw in range(draws):
        for index, support_size in enumerate(support_sizes):
            candidates = np.delete(np.arange(n_assets), index)
            selected = rng.choice(candidates, size=int(support_size), replace=False)
            result[draw, index, selected] = rng.dirichlet(np.ones(int(support_size)))
    for matrix in result:
        validate_weight_matrix(matrix)
    return result


def fit_market_betas(training_returns: FloatMatrix) -> NDArray[np.float64]:
    """Estimate demeaned one-factor betas from a training window."""
    market = training_returns.mean(axis=1)
    centered_market = market - market.mean()
    denominator = float(centered_market @ centered_market)
    if denominator <= np.finfo(np.float64).eps:
        raise_gate_error("training market proxy has zero variance")
    centered_returns = training_returns - training_returns.mean(axis=0)
    return np.asarray(
        centered_market @ centered_returns / denominator,
        dtype=np.float64,
    )


def tracking_losses(
    evaluation_returns: FloatMatrix,
    weights: FloatMatrix,
    betas: FloatMatrix,
) -> dict[str, FloatMatrix | float]:
    """Compute raw and market-residualized tracking losses without refitting."""
    market = evaluation_returns.mean(axis=1)
    basket_returns = evaluation_returns @ weights.T
    raw_errors = evaluation_returns - basket_returns
    target_residuals = evaluation_returns - market[:, None] * betas[None, :]
    basket_betas = weights @ betas
    basket_residuals = basket_returns - market[:, None] * basket_betas[None, :]
    systematic_errors = target_residuals - basket_residuals
    raw_asset_mse = np.mean(np.square(raw_errors), axis=0)
    systematic_asset_mse = np.mean(np.square(systematic_errors), axis=0)
    raw_date_mse = np.mean(np.square(raw_errors), axis=1)
    systematic_date_mse = np.mean(np.square(systematic_errors), axis=1)
    return {
        "raw_rmse": float(np.sqrt(np.mean(raw_date_mse))),
        "systematic_rmse": float(np.sqrt(np.mean(systematic_date_mse))),
        "raw_asset_rmse": np.sqrt(raw_asset_mse),
        "systematic_asset_rmse": np.sqrt(systematic_asset_mse),
        "raw_date_mse": raw_date_mse,
        "systematic_date_mse": systematic_date_mse,
    }


def covariance_row_tv(
    evaluation_returns: FloatMatrix,
    weights: FloatMatrix,
    *,
    active_tolerance: float,
) -> tuple[float, FloatMatrix]:
    """Compare each row with positive realized correlation on matched support."""
    correlation = np.asarray(
        np.corrcoef(evaluation_returns, rowvar=False), dtype=np.float64
    )
    np.fill_diagonal(correlation, 0.0)
    positive = np.clip(correlation, 0.0, None)
    distances = np.empty(weights.shape[0], dtype=np.float64)
    for index in range(weights.shape[0]):
        support = weights[index] > active_tolerance
        support[index] = False
        target = np.where(support, positive[index], 0.0)
        target_mass = float(target.sum())
        if target_mass <= 0.0:
            target = support.astype(np.float64) / int(np.count_nonzero(support))
        else:
            target /= target_mass
        candidate = np.where(support, weights[index], 0.0)
        candidate /= candidate.sum()
        distances[index] = 0.5 * np.abs(candidate - target).sum()
    return float(distances.mean()), distances


def evaluate_weights(
    training_returns: FloatMatrix,
    evaluation_returns: FloatMatrix,
    weights: FloatMatrix,
    *,
    active_tolerance: float,
) -> dict[str, FloatMatrix | float]:
    """Evaluate all three preregistered estimands for one weighting scheme."""
    betas = fit_market_betas(training_returns)
    result = tracking_losses(evaluation_returns, weights, betas)
    covariance_tv, covariance_asset_tv = covariance_row_tv(
        evaluation_returns,
        weights,
        active_tolerance=active_tolerance,
    )
    result["covariance_tv"] = covariance_tv
    result["covariance_asset_tv"] = covariance_asset_tv
    return result


__all__ = [
    "correlation_knn_weights",
    "covariance_row_tv",
    "equal_support_weights",
    "evaluate_weights",
    "fit_market_betas",
    "industry_weights",
    "random_matched_sparsity_weights",
    "tracking_losses",
    "validate_weight_matrix",
]

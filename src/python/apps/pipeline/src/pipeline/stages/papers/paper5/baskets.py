"""Pure weighting functions for Paper 5 comparator baskets."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
_MATRIX_DIMENSIONS = 2


class Paper5OperatorError(ValueError):
    """Report a violated Paper 5 operator or comparator-basket contract."""


def raise_operator_error(message: str) -> NoReturn:
    """Raise the operator domain error without embedding literals at call sites."""
    raise Paper5OperatorError(message)


def validate_weight_matrix(weights: FloatMatrix, *, tolerance: float = 1e-10) -> None:
    """Validate the simplex and leave-one-out invariants of a Paper 5 operator."""
    if weights.ndim != _MATRIX_DIMENSIONS or weights.shape[0] != weights.shape[1]:
        raise_operator_error("weight matrix must be square")
    if not np.all(np.isfinite(weights)) or np.min(weights) < -tolerance:
        raise_operator_error("weight matrix must be finite and nonnegative")
    if np.max(np.abs(np.diag(weights))) > tolerance:
        raise_operator_error("weight matrix must have a zero diagonal")
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > tolerance:
        raise_operator_error("every basket must be fully invested")


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
        raise_operator_error("W-flat contains an empty active row")
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
        raise_operator_error("matched random supports require 1..n_assets-1 entries")
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


__all__ = [
    "Paper5OperatorError",
    "equal_support_weights",
    "industry_weights",
    "raise_operator_error",
    "random_matched_sparsity_weights",
    "validate_weight_matrix",
]

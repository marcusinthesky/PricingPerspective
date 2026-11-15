"""Construct the diffusion interaction matrix for Paper 5.

``W^h`` is the pre-registered diffusion kernel ``∝ exp(-D^2/h)``, row
normalized and zero-diagonal. Bandwidth ``h`` is fixed by the median
off-diagonal squared W2 distance and recorded alongside the matrix.

The replication operator ``W♭`` is loaded from the shared typed barycentre
artifact by the Paper 5 runner. This module only computes the diffusion
variant; it does not solve barycentre programs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]


def build_w_kernel(squared_distances: Float, tickers: list[str]) -> tuple[Float, float]:
    """Build ``W^h`` — the row-normalized diffusion kernel ``∝ exp(-D^2/h)``.

    Bandwidth ``h`` is the pre-registered rule: the median of the
    off-diagonal squared statistical distances (the standard diffusion-maps
    default bandwidth).

    Args:
        squared_distances: Squared statistical-distance matrix with zero diagonal.
        tickers: Ticker order (only used for shape/logging).

    Returns:
        Tuple ``(W_h, h)`` — ``W_h`` row-normalized, zero-diagonal.

    """
    expected_shape = (len(tickers), len(tickers))
    if squared_distances.shape != expected_shape:
        message = (
            f"distance matrix shape {squared_distances.shape} does not match "
            f"{len(tickers)} tickers"
        )
        raise ValueError(message)
    n = squared_distances.shape[0]
    iu = np.triu_indices(n, k=1)
    h = float(np.median(squared_distances[iu]))
    if h <= 0:
        message = "Degenerate bandwidth h<=0: statistical-distance matrix collapsed"
        raise ValueError(message)
    k = np.exp(-squared_distances / h)
    np.fill_diagonal(k, 0.0)
    row_sums = k.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w_h: Float = k / row_sums
    return w_h, h

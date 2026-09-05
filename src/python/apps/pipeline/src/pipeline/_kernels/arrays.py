"""Pure, I/O-free array kernels shared across stages."""

from __future__ import annotations

import numpy as np

_MATRIX_NDIM = 2


class InvalidNormalizationRowsError(ValueError):
    """Raised when row-wise normalization is not mathematically defined."""


class InvalidSimplexProjectionError(ValueError):
    """Raised when a simplex projection has no defined result."""


def project_simplex(weights: np.ndarray) -> np.ndarray:
    """Project a finite vector onto the unit nonnegative simplex.

    Duchi et al. (2008), in float64 on purpose. A JAX/optax projection exists
    (``jcor.optimize`` routes through one) but resolves to float32 unless the
    caller owns ``jax_enable_x64``, which this repository never sets. Both
    callers here -- the anchored Wasserstein target solver and the paper-2
    tightest-cap polish loop -- run float64 precisely to escape that, so they
    share this NumPy implementation rather than the jitted one.

    Raises:
        InvalidSimplexProjectionError: If the input is not a finite, nonempty
            one-dimensional vector.

    """
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        message = "simplex projection requires a finite nonempty vector"
        raise InvalidSimplexProjectionError(message)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    active = np.nonzero(ordered * np.arange(1, len(values) + 1) > cumulative - 1.0)[0]
    if len(active) == 0:
        message = "simplex projection found no active coordinate"
        raise InvalidSimplexProjectionError(message)
    threshold = (cumulative[active[-1]] - 1.0) / float(active[-1] + 1)
    return np.maximum(values - threshold, 0.0)


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Scale finite, nonzero rows to the L2 unit sphere.

    This is an eager pipeline policy boundary. A zero or non-finite row has no
    direction and is rejected; it is never repaired into purported unit-sphere
    data. The input dtype is preserved.

    Raises:
        InvalidNormalizationRowsError: If the input is not a nonempty matrix,
            has non-finite entries, or contains a zero/non-finite row norm.

    """
    if arr.ndim != _MATRIX_NDIM or 0 in arr.shape:
        message = f"row normalization requires a nonempty matrix, got {arr.shape}"
        raise InvalidNormalizationRowsError(message)
    if not np.all(np.isfinite(arr)):
        message = "row normalization requires finite values"
        raise InvalidNormalizationRowsError(message)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm == 0):
        message = "row normalization requires finite, nonzero row norms"
        raise InvalidNormalizationRowsError(message)
    return arr / norm


def l2_normalize_rows_then_mean(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize rows, mean-pool them, then normalize the pooled prototype.

    The order is part of the statistical feature definition. Normalizing the
    pooled raw rows would weight articles by embedding magnitude; this adapter
    first projects each article onto the L2 sphere, then gives every article
    equal weight in the empirical mean. A zero pooled mean (for example an
    antipodal cloud) has no direction and is rejected through the same eager
    boundary as an invalid row.

    Args:
        arr: Nonempty matrix of finite, nonzero embedding rows.

    Returns:
        The unit-normalized rows and their unit-normalized mean prototype.

    Raises:
        InvalidNormalizationRowsError: If row normalization or pooled-mean
            normalization is undefined.

    """
    rows = l2_normalize_rows(arr)
    pooled = rows.mean(axis=0, keepdims=True)
    prototype = l2_normalize_rows(pooled)[0]
    return rows, prototype

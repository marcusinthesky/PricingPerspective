"""Prototype-chord door used at host artifact boundaries.

Formerly a float64 door: it coerced both inputs with
``np.asarray(x, dtype=np.float64)`` regardless of caller configuration. t65.4
removed that upcast along with NumPy itself — the arithmetic is one norm of a
difference, and it evaluates at whatever precision the caller owns.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from jcor.core.typing import ArrayLike  # noqa: TC001  # runtime numerical contract
from jcor.discrepancy._mmd.contracts import (
    NonfiniteMeanEmbeddingError,
    ZeroNormMeanEmbeddingError,
)

__all__ = ["cosine_mean_mmd_functional"]


def cosine_mean_mmd_functional(left: ArrayLike, right: ArrayLike) -> float:
    """Compute chord distance between two caller-normalized unit prototypes."""
    left_unit = jnp.asarray(left)
    right_unit = jnp.asarray(right)
    if (
        left_unit.ndim != 1
        or right_unit.ndim != 1
        or left_unit.shape != right_unit.shape
    ):
        message = "prototype chord requires two vectors with the same shape"
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(left_unit))) or not bool(
        jnp.all(jnp.isfinite(right_unit))
    ):
        raise NonfiniteMeanEmbeddingError
    left_norm = float(jnp.linalg.norm(left_unit))
    right_norm = float(jnp.linalg.norm(right_unit))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ZeroNormMeanEmbeddingError
    if not math.isclose(left_norm, 1.0, abs_tol=1e-8) or not math.isclose(
        right_norm, 1.0, abs_tol=1e-8
    ):
        message = "prototype vectors must already be unit-normalized by the caller"
        raise ValueError(message)
    return float(jnp.linalg.norm(left_unit - right_unit))

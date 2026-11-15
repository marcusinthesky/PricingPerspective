"""Nested Sobol geometry with dimension-normalized Euclidean distances."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from statadic.core.typing import (  # noqa: TC002  # runtime shape contracts
    Array,
    ArrayLike,
    Float,
)
from statadic.geometry.ground.metrics import squared_euclidean_distance

POINT_ARRAY_RANK = 2


def sobol_points(
    count: int, dimension: int, seed: int, stream: int
) -> Float[Array, "point dimension"]:
    """Take a nested prefix from a separate reproducible scrambled stream."""
    sequence = np.random.SeedSequence([seed, stream])
    engine = qmc.Sobol(d=dimension, scramble=True, rng=np.random.default_rng(sequence))
    return jnp.asarray(engine.random_base2(math.ceil(math.log2(count)))[:count])


def squared_distance(x: ArrayLike, y: ArrayLike) -> Float[Array, "n m"]:
    """Squared vector distance divided by opportunity-space dimension."""
    x, y = jnp.asarray(x), jnp.asarray(y)
    if (
        x.ndim != POINT_ARRAY_RANK
        or y.ndim != POINT_ARRAY_RANK
        or x.shape[1] != y.shape[1]
        or x.shape[1] == 0
    ):
        msg = "point arrays must have the same positive vector dimension"
        raise ValueError(msg)
    return squared_euclidean_distance(x, y) / x.shape[1]


def interaction(x: ArrayLike, y: ArrayLike, tau: float) -> Float[Array, "n m"]:
    """Gaussian opportunity or congestion kernel using every coordinate."""
    return jnp.exp(-tau * squared_distance(x, y))


def centered_direction(value: Float[Array, "*axes"]) -> Float[Array, "*axes"]:
    """Center and bound a deterministic signed shock field by one."""
    centered = value - value.mean()
    scale = jnp.max(jnp.abs(centered))
    return centered / jnp.where(scale > 0, scale, 1)

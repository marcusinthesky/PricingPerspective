"""Key-only spatial-autoregressive (SAR) return-panel generator.

Every random draw is a pure function of a JAX key.  Callers derive the cell key
from semantic labels and fold in the replication index, so grid ordering and
replication-count changes do not perturb existing streams.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float  # noqa: TC002

__all__ = ["SarDGP", "make_synthetic_w", "simulate_sar"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("weights", "rho", "alpha", "sigma"),
    meta_fields=("n_observations",),
)
@dataclass(frozen=True)
class SarDGP:
    """Structural parameters for one simulated SAR return panel."""

    weights: Float[Array, "n n"]
    rho: Float[ArrayLike, ""] | Float[ArrayLike, " n"]
    alpha: Float[ArrayLike, ""]
    sigma: Float[ArrayLike, ""]
    n_observations: int


@partial(jax.jit, static_argnames=("n", "k"))
def make_synthetic_w(
    key: jax.Array,
    n: int,
    k: int = 5,
) -> Float[Array, "n n"]:
    """Draw a row-stochastic zero-diagonal k-NN matrix on random 2-D points."""
    coords = jax.random.normal(key, shape=(n, 2))
    differences = coords[:, None, :] - coords[None, :, :]
    distances = jnp.linalg.norm(differences, axis=2)
    distances = distances.at[jnp.diag_indices(n)].set(jnp.inf)
    neighbor_count = min(k, n - 1)
    neighbors = jnp.argsort(distances, axis=1)[:, :neighbor_count]
    rows = jnp.arange(n)[:, None]
    weights = jnp.zeros((n, n), dtype=coords.dtype)
    weights = weights.at[rows, neighbors].set(1.0 / distances[rows, neighbors])
    row_sums = jnp.sum(weights, axis=1, keepdims=True)
    return weights / jnp.where(row_sums > 0.0, row_sums, 1.0)


@jax.jit
def simulate_sar(
    key: jax.Array,
    dgp: SarDGP,
) -> Float[Array, "observations n"]:
    """Draw one SAR panel under a known ``(rho, W, alpha, sigma)``."""
    n = dgp.weights.shape[0]
    rho = jnp.asarray(dgp.rho)
    spatial = rho * dgp.weights if rho.ndim == 0 else rho[:, None] * dgp.weights
    operator = jnp.eye(n, dtype=dgp.weights.dtype) - spatial
    innovations = dgp.sigma * jax.random.normal(
        key,
        shape=(dgp.n_observations, n),
        dtype=dgp.weights.dtype,
    )
    rhs = dgp.alpha + innovations
    return jnp.linalg.solve(operator, rhs.T).T

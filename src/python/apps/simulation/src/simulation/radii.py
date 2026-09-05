"""Key-only empirical-sampling radii.

The numerical API is array-only and transformable.  Entity labels and mapping
conversion belong to pipeline/report boundaries, never to a traced bootstrap.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002

__all__ = ["bootstrap_stat_radius", "stat_radius_matrix"]


@partial(jax.jit, static_argnames=("n_boot", "quantile"))
def bootstrap_stat_radius(
    key: jax.Array,
    embeddings: Float[Array, "entities observations features"],
    n_boot: int = 300,
    quantile: float = 0.95,
) -> Float[Array, " entities"]:
    """Estimate one mean-embedding radius per entity by iid bootstrap."""
    n_entities, n_observations, _ = embeddings.shape
    entity_keys = jax.random.split(key, n_entities)

    def entity_radius(
        entity_key: jax.Array,
        observations: Float[Array, "observations features"],
    ) -> Float[Array, ""]:
        indices = jax.random.randint(
            entity_key,
            shape=(n_boot, n_observations),
            minval=0,
            maxval=n_observations,
        )
        mean = jnp.mean(observations, axis=0)
        bootstrap_means = jnp.mean(observations[indices], axis=1)
        deviations = jnp.linalg.norm(bootstrap_means - mean, axis=1)
        return jnp.quantile(deviations, quantile)

    return jax.vmap(entity_radius)(entity_keys, embeddings)


@jax.jit
def stat_radius_matrix(
    radii: Float[Array, " entities"],
) -> Float[Array, "entities entities"]:
    """Compose entity radii into ``eps_ij = eps_i + eps_j`` with zero diagonal."""
    matrix = radii[:, None] + radii[None, :]
    return matrix.at[jnp.diag_indices(radii.shape[0])].set(0.0)

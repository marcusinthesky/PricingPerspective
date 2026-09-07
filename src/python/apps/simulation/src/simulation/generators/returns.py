"""Return-panel generators.

All generators are pure functions of a :class:`jax.random.PRNGKey` and static
config args, producing **one replication**.  Batch via :func:`jax.vmap`.

Source: ``src/python/apps/simulation/src/simulation/validation/pricefree.py``,
``compute_true_covariance`` and ``simulate_returns``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Imported at RUNTIME, deliberately not under `if TYPE_CHECKING:`. Paired with
# `from __future__ import annotations`, a guarded `Float`/`Array` is an
# unresolvable name at runtime and beartype then **silently skips** the
# annotation rather than failing -- probed and recorded in
# `packages/jcor/src/jcor/core/typing.py:10-27`. Un-guarding is what puts the
# six shape sites below under `tests/conftest.py`'s import hook.
# `flake8-type-checking` in strict mode wants the import back inside the block;
# the `noqa` is that cost.
from jaxtyping import Array, Float  # noqa: TC002


def factor_covariance(
    betas: Float[Array, "n d"],
    sigma_eps: float,
) -> Float[Array, "n n"]:
    """Compute the factor-model covariance matrix.

    ``Σ = β βᵀ + σ_ε² I``

    This is the *true* covariance under the factor model: systematic
    component ``β βᵀ`` plus idiosyncratic noise ``σ_ε² I``.

    Args:
        betas: Factor-loading matrix, shape ``(n, d)``.
        sigma_eps: Idiosyncratic noise standard deviation.

    Returns:
        Covariance matrix of shape ``(n, n)``.

    """
    return betas @ betas.T + sigma_eps**2 * jnp.eye(betas.shape[0])


def _cholesky_with_jitter(
    sigma: Float[Array, "n n"],
    jitter: float = 1e-10,
) -> Float[Array, "n n"]:
    """Lower-triangular Cholesky factor of ``Sigma + jitter * I``."""
    n = sigma.shape[0]
    return jnp.linalg.cholesky(sigma + jitter * jnp.eye(n))


def gaussian_panel(
    key: jax.Array,
    betas: Float[Array, "n d"],
    sigma_eps: float,
    n_observations: int,
) -> Float[Array, "n_observations n"]:
    """Simulate ``T`` i.i.d. Gaussian return vectors ``r_t ~ N(0, Σ)``.

    ``Σ = β βᵀ + σ_ε² I`` is formed from ``betas`` and ``sigma_eps``,
    Cholesky-factorised with a ``1e-10`` jitter, and used in a single
    matmul ``z @ L.T`` where ``z ~ N(0, I_{T×n})``.

    Args:
        key: JAX PRNG key.
        betas: Factor-loading matrix, shape ``(n, d)``.
        sigma_eps: Idiosyncratic noise standard deviation.
        n_observations: Number of time periods to simulate.

    Returns:
        Return panel of shape ``(T, n)``.

    """
    n = betas.shape[0]
    sigma = factor_covariance(betas, sigma_eps)
    cholesky_factor = _cholesky_with_jitter(sigma)  # (n, n)
    z = jax.random.normal(key, shape=(n_observations, n))
    return z @ cholesky_factor.T

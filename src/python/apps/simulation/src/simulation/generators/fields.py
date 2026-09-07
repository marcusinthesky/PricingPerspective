"""Gaussian random field samplers on a regular grid.

All public functions are pure, jit/vmap-safe.  Shape arguments are static
(mark ``resolution``, ``d``, ``n_draws`` as ``static_argnames`` at call
sites that use :func:`jax.jit`).

Design note — Cholesky-on-grid vs FFT circulant embedding
----------------------------------------------------------
Cholesky factorisation is used here for correctness and simplicity on small
grids (M = resolution^d ≤ ~1 000).  For large grids the FFT circulant
embedding method is O(M log M) instead of O(M³) and is documented as the
upgrade path in ARCHITECTURE.md; it is not implemented here to keep the
initial implementation auditable.

White-noise fast path
---------------------
When ``kind="white"`` in :func:`gaussian_field`, the Cholesky matrix is
diagonal (σ·I) and we bypass the ``z @ chol.T`` matmul entirely, instead
calling the cheaper :func:`white_field`.  Callers may also call
:func:`white_field` directly if they know the field is white.

Source: ROADMAP §4.1 (new field samplers for §4 simulation stages).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Imported at RUNTIME, deliberately not under `if TYPE_CHECKING:`. Paired with
# `from __future__ import annotations`, a guarded `Float`/`Array` is an
# unresolvable name at runtime and beartype then **silently skips** the
# annotation rather than failing -- probed and recorded in
# `packages/jcor/src/jcor/core/typing.py:10-27`. Un-guarding is what puts the
# eight shape sites below under `tests/conftest.py`'s import hook.
# `flake8-type-checking` in strict mode wants the import back inside the block;
# the `noqa` is that cost.
from jaxtyping import Array, Float  # noqa: TC002

# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def grid_points(resolution: int, d: int) -> Float[Array, "M d"]:
    """Cell centres of a regular ``resolution^d`` grid on ``[0, 1]^d``.

    Args:
        resolution: Number of cells per dimension.
        d: Number of dimensions.

    Returns:
        Array of shape ``(M, d)`` where ``M = resolution^d``, containing
        the centre coordinates of every cell in row-major order.

    """
    edges = jnp.linspace(0.5 / resolution, 1.0 - 0.5 / resolution, resolution)
    # Build a meshgrid over d dimensions and stack into (M, d).
    grids = jnp.meshgrid(*[edges] * d, indexing="ij")
    return jnp.stack([g.ravel() for g in grids], axis=-1)  # (M, d)


# ---------------------------------------------------------------------------
# Kernel matrices
# ---------------------------------------------------------------------------


def kernel_matrix(
    x: Float[Array, "M d"],
    kind: str,
    lengthscale: float,
    sigma: float = 1.0,
) -> Float[Array, "M M"]:
    """Compute a stationary kernel matrix between grid points.

    Args:
        x: Grid points, shape ``(M, d)``.
        kind: Kernel type — one of ``"white"``, ``"exponential"``,
            ``"matern32"``.
        lengthscale: Characteristic length-scale ``ℓ > 0``.
        sigma: Marginal standard deviation (output scale).  Default ``1.0``.

    Returns:
        Kernel matrix ``K`` of shape ``(M, M)``.

    Raises:
        ValueError: If ``kind`` is not recognised.

    """
    # Pairwise squared Euclidean distances, (M, M).
    diff = x[:, None, :] - x[None, :, :]  # (M, M, d)
    r2 = jnp.sum(diff**2, axis=-1)  # (M, M)

    sigma2 = sigma**2

    if kind == "white":
        n_grid_points = x.shape[0]
        return sigma2 * jnp.eye(n_grid_points)
    if kind == "exponential":
        r = jnp.sqrt(r2 + 1e-30)
        return sigma2 * jnp.exp(-r / lengthscale)
    if kind == "matern32":
        r = jnp.sqrt(r2 + 1e-30)
        s = jnp.sqrt(3.0) * r / lengthscale
        return sigma2 * (1.0 + s) * jnp.exp(-s)
    message = (
        f"Unknown kernel kind {kind!r}. Choose from: white, exponential, matern32."
    )
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Cholesky factorisation
# ---------------------------------------------------------------------------


def field_cholesky(
    kernel_matrix: Float[Array, "M M"],
    jitter: float = 1e-8,
) -> Float[Array, "M M"]:
    """Lower-triangular Cholesky factor of ``kernel_matrix + jitter * I``.

    Args:
        kernel_matrix: Symmetric positive-semidefinite kernel matrix with shape
            ``(M, M)``.
        jitter: Diagonal regularisation added before factorisation.

    Returns:
        Lower-triangular Cholesky factor ``L`` such that ``L @ L.T ≈ kernel_matrix``.

    """
    n_grid_points = kernel_matrix.shape[0]
    return jnp.linalg.cholesky(kernel_matrix + jitter * jnp.eye(n_grid_points))


# ---------------------------------------------------------------------------
# Field draws
# ---------------------------------------------------------------------------


def white_field(
    key: jax.Array,
    n_grid_points: int,
    n_draws: int,
    sigma: float = 1.0,
) -> Float[Array, "n_draws M"]:
    """Fast white-noise field draw (no Cholesky required).

    Each draw is an i.i.d. ``N(0, σ²)`` vector of length ``M``.

    Args:
        key: JAX PRNG key.
        n_grid_points: Number of grid points.
        n_draws: Number of independent draws.
        sigma: Standard deviation of each component.

    Returns:
        Array of shape ``(n_draws, M)``.

    """
    return sigma * jax.random.normal(key, shape=(n_draws, n_grid_points))


def gaussian_field(
    key: jax.Array,
    chol: Float[Array, "M M"],
    n_draws: int,
) -> Float[Array, "n_draws M"]:
    """Draw ``n_draws`` realisations of a Gaussian field.

    Uses the single matmul ``z @ chol.T`` where ``z ~ N(0, I_{n_draws × M})``.

    For white-noise fields, prefer :func:`white_field` directly (or pass
    ``sigma * jnp.eye(M)`` as ``chol`` — the matmul degenerates to a scale).
    This function does **not** special-case white noise; callers that know
    the field is white should call :func:`white_field` instead.

    Args:
        key: JAX PRNG key.
        chol: Lower-triangular Cholesky factor, shape ``(M, M)``.
        n_draws: Number of independent field realisations to draw.

    Returns:
        Array of shape ``(n_draws, M)``.

    """
    n_grid_points = chol.shape[0]
    z = jax.random.normal(key, shape=(n_draws, n_grid_points))
    return z @ chol.T

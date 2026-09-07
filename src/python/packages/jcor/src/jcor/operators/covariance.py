"""Construct covariance operators from distances and marginal variances.

The ``*_kernel`` functions are the validation-free JAX numerical primitives;
they preserve input dtype and compose with ``jit``/``vmap`` and the shared
``jcor.optimize.ridge_psd`` repair.  The historical names remain eager
validation adapters, but return JAX arrays so the public surface stays
composable with transforms.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jcor.core.typing import (  # noqa: TC001  # runtime numerical contract
    Array,
    ArrayLike,
    Float,
    ScalarLike,
)

_MATRIX_NDIM = 2
_DISTANCE_TOLERANCE = 1e-12


def _validate_inputs(
    d2: ArrayLike,
    sigma_sq: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Validate a squared-distance matrix and matching marginal variances."""
    squared_distances = jnp.asarray(d2)
    if (
        squared_distances.ndim != _MATRIX_NDIM
        or squared_distances.shape[0] != squared_distances.shape[1]
        or squared_distances.shape[0] < 1
    ):
        message = "d2 must be a nonempty square matrix."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(squared_distances))):
        message = "d2 must contain only finite values."
        raise ValueError(message)
    if not bool(
        jnp.allclose(
            squared_distances,
            squared_distances.T,
            rtol=1e-10,
            atol=_DISTANCE_TOLERANCE,
        )
    ):
        message = "d2 must be symmetric."
        raise ValueError(message)
    if bool(jnp.any(squared_distances < -_DISTANCE_TOLERANCE)):
        message = "d2 must be nonnegative."
        raise ValueError(message)
    if not bool(
        jnp.allclose(
            jnp.diag(squared_distances),
            0.0,
            rtol=0.0,
            atol=_DISTANCE_TOLERANCE,
        )
    ):
        message = "d2 must have a zero diagonal."
        raise ValueError(message)
    variances = jnp.asarray(sigma_sq)
    if variances.shape != (squared_distances.shape[0],):
        message = "sigma_sq must be a vector matching d2."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(variances))) or bool(jnp.any(variances < 0.0)):
        message = "sigma_sq must contain finite nonnegative variances."
        raise ValueError(message)
    cleaned = jnp.maximum(
        0.5 * (squared_distances + squared_distances.T),
        0.0,
    )
    cleaned = jnp.fill_diagonal(cleaned, 0.0, inplace=False)
    return cleaned, variances


@jax.jit
def sigma_dist_kernel(
    d2: Float[Array, "n n"],  # noqa: F722  # jaxtyping shape
    sigma_sq: Float[Array, "n"],  # noqa: F821, UP037  # jaxtyping shape
    kappa: ScalarLike = 1.0,
) -> Float[Array, "n n"]:  # noqa: F722  # jaxtyping shape
    """Return the transformable scalar-scale covariance construction."""
    return 0.5 * (sigma_sq[:, None] + sigma_sq[None, :] - kappa**2 * d2)


def sigma_dist(
    d2: ArrayLike,
    sigma_sq: ArrayLike,
    kappa: float = 1.0,
) -> Float[Array, "n n"]:  # noqa: F722  # jaxtyping shape
    """Construct a distance-implied covariance with one scale parameter.

    ``Sigma[i,j] = 0.5 * (sigma_i^2 + sigma_j^2 - kappa^2 D^2[i,j])``.

    Args:
        d2: Squared distance matrix, shape ``(n, n)``.
        sigma_sq: Marginal variances, shape ``(n,)``.
        kappa: Common distance scale.

    Returns:
        JAX covariance operator, not necessarily PSD.

    """
    squared_distances, variances = _validate_inputs(d2, sigma_sq)
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa < 0.0:
        message = "kappa must be finite and nonnegative."
        raise ValueError(message)
    return sigma_dist_kernel(
        squared_distances,
        variances,
        kappa,
    )


@jax.jit
def sigma_dist_hetero_kappa_kernel(
    d2: Float[Array, "n n"],  # noqa: F722  # jaxtyping shape
    sigma_sq: Float[Array, "n"],  # noqa: F821, UP037  # jaxtyping shape
    kappa_vec: Float[Array, "n"],  # noqa: F821, UP037  # jaxtyping shape
) -> Float[Array, "n n"]:  # noqa: F722  # jaxtyping shape
    """Return the transformable per-item-scale covariance construction."""
    scale_products = kappa_vec[:, None] * kappa_vec[None, :]
    return 0.5 * (sigma_sq[:, None] + sigma_sq[None, :] - scale_products * d2)


def sigma_dist_hetero_kappa(
    d2: ArrayLike,
    sigma_sq: ArrayLike,
    kappa_vec: ArrayLike,
) -> Float[Array, "n n"]:  # noqa: F722  # jaxtyping shape
    """Construct a distance-implied covariance with per-item scales.

    ``Sigma[i,j] = 0.5 * (sigma_i^2 + sigma_j^2 - kappa_i kappa_j D^2[i,j])``.

    Args:
        d2: Squared distance matrix, shape ``(n, n)``.
        sigma_sq: Marginal variances, shape ``(n,)``.
        kappa_vec: Per-item distance scales, shape ``(n,)``.

    Returns:
        JAX covariance operator, not necessarily PSD.

    """
    squared_distances, variances = _validate_inputs(d2, sigma_sq)
    scales = jnp.asarray(kappa_vec)
    if scales.shape != variances.shape:
        message = "kappa_vec must be a vector matching sigma_sq."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(scales))) or bool(jnp.any(scales < 0.0)):
        message = "kappa_vec must contain finite nonnegative scales."
        raise ValueError(message)
    return sigma_dist_hetero_kappa_kernel(
        squared_distances,
        variances,
        scales,
    )


__all__ = [
    "sigma_dist",
    "sigma_dist_hetero_kappa",
    "sigma_dist_hetero_kappa_kernel",
    "sigma_dist_kernel",
]

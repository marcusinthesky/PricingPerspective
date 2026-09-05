"""Sample-covariance and constant-correlation Ledoit-Wolf estimators."""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jcor.core.typing import Array, ArrayLike, Static  # noqa: TC001
from jcor.core.typing import Float as JaxFloat  # noqa: TC001
from jcor.model._covariance.validation import (
    MIN_COVARIANCE_OBSERVATIONS,
    validate_observation_panel,
)

__all__ = [
    "LedoitWolfKernelResult",
    "LedoitWolfResult",
    "ledoit_wolf_sample",
    "ledoit_wolf_sample_kernel",
    "sample_covariance",
    "sample_covariance_kernel",
]


@partial(jax.jit, static_argnames=("ddof",))
def sample_covariance_kernel(
    returns: JaxFloat[Array, "t n"],
    ddof: Static[int] = 1,
) -> JaxFloat[Array, "n n"]:
    """Return a fixed-rank sample covariance entirely on the JAX graph."""
    centered = returns - jnp.mean(returns, axis=0, keepdims=True)
    return centered.T @ centered / (returns.shape[0] - ddof)


def sample_covariance(returns: ArrayLike, ddof: int = 1) -> JaxFloat[Array, "n n"]:
    """Return the ordinary sample covariance of an observation panel."""
    panel = validate_observation_panel(returns, ddof=ddof)
    return sample_covariance_kernel(panel, ddof)


class LedoitWolfResult(NamedTuple):
    """Constant-correlation Ledoit-Wolf estimate and shrinkage intensity."""

    sigma: JaxFloat[Array, "n n"]
    shrinkage: float
    target: JaxFloat[Array, "n n"]


class LedoitWolfKernelResult(NamedTuple):
    """Fixed-structure JAX constant-correlation shrinkage result."""

    sigma: JaxFloat[Array, "n n"]
    shrinkage: JaxFloat[Array, ""]
    target: JaxFloat[Array, "n n"]


def _constant_correlation_target_kernel(
    sample: JaxFloat[Array, "n n"],
) -> JaxFloat[Array, "n n"]:
    """Build the constant-correlation target without traced boolean slicing."""
    n_variables = sample.shape[0]
    variance = jnp.diag(sample)
    standard_deviation = jnp.sqrt(variance)
    correlation = sample / jnp.outer(standard_deviation, standard_deviation)
    mask = 1.0 - jnp.eye(n_variables, dtype=sample.dtype)
    mean_correlation = jnp.sum(correlation * mask) / (n_variables * (n_variables - 1))
    target = mean_correlation * jnp.outer(standard_deviation, standard_deviation)
    diagonal = jnp.diag_indices(n_variables)
    return target.at[diagonal].set(variance)


def _rho_hat_kernel(
    sample: JaxFloat[Array, "n n"],
    centered: JaxFloat[Array, "t n"],
    deviations: JaxFloat[Array, "t n n"],
    entry_variance: JaxFloat[Array, "n n"],
) -> JaxFloat[Array, ""]:
    """Accumulate the constant-correlation asymptotic covariance term."""
    n_variables = sample.shape[0]
    variance = jnp.diag(sample)
    standard_deviation = jnp.sqrt(variance)
    correlation = sample / jnp.outer(standard_deviation, standard_deviation)
    mask = 1.0 - jnp.eye(n_variables, dtype=sample.dtype)
    mean_correlation = jnp.sum(correlation * mask) / (n_variables * (n_variables - 1))
    squared = centered**2
    theta_ii = jnp.mean(
        (squared[:, :, None] - variance[None, :, None]) * deviations,
        axis=0,
    )
    theta_jj = jnp.mean(
        (squared[:, None, :] - variance[None, None, :]) * deviations,
        axis=0,
    )
    coefficient = (
        0.5
        * mean_correlation
        * (
            jnp.sqrt(jnp.outer(1.0 / variance, variance)) * theta_ii
            + jnp.sqrt(jnp.outer(variance, 1.0 / variance)) * theta_jj
        )
    )
    return jnp.trace(entry_variance) + jnp.sum(coefficient * mask)


@jax.jit
def ledoit_wolf_sample_kernel(
    returns: JaxFloat[Array, "t n"],
) -> LedoitWolfKernelResult:
    """Estimate constant-correlation Ledoit-Wolf shrinkage on graph."""
    n_observations, n_variables = returns.shape
    sample = sample_covariance_kernel(returns, ddof=0)
    if n_variables == 1:
        return LedoitWolfKernelResult(
            sigma=sample,
            shrinkage=jnp.asarray(0.0, dtype=returns.dtype),
            target=sample,
        )
    target = _constant_correlation_target_kernel(sample)
    centered = returns - jnp.mean(returns, axis=0, keepdims=True)
    products = centered[:, :, None] * centered[:, None, :]
    deviations = products - sample[None, :, :]
    entry_variance = jnp.mean(deviations**2, axis=0)
    pi_hat = jnp.sum(entry_variance)
    rho_hat = _rho_hat_kernel(sample, centered, deviations, entry_variance)
    gamma_hat = jnp.sum((target - sample) ** 2)
    kappa = jnp.where(gamma_hat > 0.0, (pi_hat - rho_hat) / gamma_hat, 0.0)
    shrinkage = jnp.clip(kappa / n_observations, 0.0, 1.0)
    sigma = shrinkage * target + (1.0 - shrinkage) * sample
    return LedoitWolfKernelResult(
        sigma=sigma,
        shrinkage=shrinkage,
        target=target,
    )


def ledoit_wolf_sample(returns: ArrayLike) -> LedoitWolfResult:
    """Shrink sample covariance toward a constant-correlation target."""
    panel = validate_observation_panel(
        returns,
        ddof=0,
        min_observations=MIN_COVARIANCE_OBSERVATIONS,
    )
    _, n_variables = panel.shape
    sample = sample_covariance_kernel(panel, ddof=0)
    if n_variables == 1:
        # `sigma` and `target` were distinct NumPy buffers here (the second a
        # `.copy()`); JAX arrays are immutable, so sharing one is equivalent.
        return LedoitWolfResult(sigma=sample, shrinkage=0.0, target=sample)
    if bool(jnp.any(jnp.diag(sample) <= 0.0)):
        message = "ledoit_wolf_sample requires strictly positive sample variances."
        raise ValueError(message)
    result = ledoit_wolf_sample_kernel(panel)
    return LedoitWolfResult(
        sigma=result.sigma,
        shrinkage=float(result.shrinkage),
        target=result.target,
    )

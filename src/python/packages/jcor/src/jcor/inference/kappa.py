"""κ inference: delta-method variances, Jacobian concentration, boundary p-value.

Delta-method propagation of ``κ̂`` uncertainty into covariance-entry variances, a
descriptive moment-Jacobian concentration diagnostic, and a boundary-robust
p-value near ``κ ≈ 0``.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from jcor.core.typing import Array, ArrayLike, Bool, Float, Static  # noqa: TC001
from jcor.inference._common import (
    _MATRIX_NDIM,
    _WEAK_IDENTIFICATION_THRESHOLD,
    _validate_bandwidth,
    _validate_full_rank_covariance,
    _validate_moment_panel,
)
from jcor.operators.longrun import newey_west_kernel, optimal_bandwidth

__all__ = [
    "KappaDeltaKernelResult",
    "KappaDeltaResult",
    "MomentJacobianKernelResult",
    "MomentJacobianResult",
    "boundary_robust_pvalue",
    "boundary_robust_pvalue_kernel",
    "kappa_delta_kernel",
    "kappa_delta_method",
    "moment_jacobian_concentration",
    "moment_jacobian_concentration_kernel",
]


class KappaDeltaKernelResult(NamedTuple):
    """Fixed-structure traced delta-method result."""

    entry_var: Float[Array, "n n"]
    gradient: Float[Array, "n n"]


class MomentJacobianKernelResult(NamedTuple):
    """Fixed-structure traced moment-sensitivity diagnostic."""

    concentration: Float[Array, ""]
    flagged_weak: Bool[Array, ""]
    jacobian_norm: Float[Array, ""]
    covariance: Float[Array, "m m"]


@jax.jit
def kappa_delta_kernel(
    d2: Float[Array, "n n"],
    kappa_hat: Float[Array, ""],
    kappa_se: Float[Array, ""],
) -> KappaDeltaKernelResult:
    """Propagate κ uncertainty using autodifferentiation of the covariance map."""
    distances = jnp.asarray(d2)

    def covariance_entry(kappa: jax.Array, distance: jax.Array) -> jax.Array:
        return -0.5 * kappa**2 * distance

    derivative = jax.grad(covariance_entry, argnums=0)
    gradient = jax.vmap(derivative, in_axes=(None, 0))(
        kappa_hat,
        distances.reshape(-1),
    ).reshape(distances.shape)
    return KappaDeltaKernelResult(
        entry_var=gradient**2 * kappa_se**2,
        gradient=gradient,
    )


@partial(jax.jit, static_argnames=("bandwidth",))
def moment_jacobian_concentration_kernel(
    g: Float[Array, "t m"],
    dg_dkappa: Float[Array, "t m"],
    bandwidth: Static[int],
) -> MomentJacobianKernelResult:
    """Compute the complete descriptive Jacobian diagnostic on the JAX graph."""
    d_bar = jnp.mean(dg_dkappa, axis=0)
    covariance = newey_west_kernel(g, bandwidth)
    concentration = g.shape[0] * d_bar @ jnp.linalg.pinv(covariance) @ d_bar
    return MomentJacobianKernelResult(
        concentration=concentration,
        flagged_weak=concentration < _WEAK_IDENTIFICATION_THRESHOLD,
        jacobian_norm=jnp.linalg.norm(d_bar),
        covariance=covariance,
    )


@jax.jit
def boundary_robust_pvalue_kernel(
    kappa_hat: Float[Array, ""],
    kappa_se: Float[Array, ""],
    kappa_null: Float[Array, ""],
) -> Float[Array, ""]:
    """Evaluate the inclusive ½χ²₀+½χ²₁ boundary tail on graph."""
    z = (kappa_hat - kappa_null) / kappa_se
    return jnp.where(z <= 0.0, 1.0, norm.sf(z))


class KappaDeltaResult(NamedTuple):
    """Return type of :func:`kappa_delta_method`.

    Attributes:
        entry_var: Covariance-entry variances induced by ``κ̂`` uncertainty,
            shape ``(n, n)``.
        kappa_hat: The κ estimate used.
        kappa_se: The κ standard error used.

    """

    entry_var: Float[Array, "n n"]
    kappa_hat: float
    kappa_se: float


def kappa_delta_method(
    d2: ArrayLike,
    kappa_hat: float,
    kappa_se: float,
) -> KappaDeltaResult:
    """Delta-method variance of ``Σ_dist`` entries from ``κ̂`` uncertainty.

    With ``Σ_dist[i,j] = ½(σ_i² + σ_j² - κ²·D²[i,j])`` and treating ``σ_i²`` as
    fixed, the only stochastic input is ``κ̂``.  The delta method gives

        ∂Σ_ij/∂κ = -κ·D²[i,j]  ⇒  Var(Σ_ij) ≈ (κ·D²[i,j])² · Var(κ̂).

    Args:
        d2: Squared energy-distance matrix, shape ``(n, n)``.
        kappa_hat: Point estimate ``κ̂``.
        kappa_se: Standard error of ``κ̂``.

    Returns:
        :class:`KappaDeltaResult` (diagonal is 0 since ``D²[i,i] = 0``).

    """
    distances = jnp.asarray(d2)
    if (
        distances.ndim != _MATRIX_NDIM
        or distances.shape[0] != distances.shape[1]
        or distances.shape[0] < 1
        or not bool(jnp.all(jnp.isfinite(distances)))
    ):
        message = "d2 must be a nonempty finite square matrix."
        raise ValueError(message)
    if not bool(jnp.allclose(distances, distances.T, rtol=1e-10, atol=1e-12)):
        message = "d2 must be symmetric."
        raise ValueError(message)
    kappa_hat = float(kappa_hat)
    kappa_se = float(kappa_se)
    if not math.isfinite(kappa_hat) or kappa_hat < 0.0:
        message = "kappa_hat must be finite and nonnegative."
        raise ValueError(message)
    if not math.isfinite(kappa_se) or kappa_se < 0.0:
        message = "kappa_se must be finite and nonnegative."
        raise ValueError(message)
    result = kappa_delta_kernel(
        distances,
        jnp.asarray(kappa_hat),
        jnp.asarray(kappa_se),
    )
    return KappaDeltaResult(
        entry_var=result.entry_var,
        kappa_hat=kappa_hat,
        kappa_se=kappa_se,
    )


class MomentJacobianResult(NamedTuple):
    """Descriptive result for :func:`moment_jacobian_concentration`.

    Attributes:
        concentration: Moment-Jacobian concentration proxy.
        flagged_weak: Whether the heuristic proxy is below its descriptive
            threshold. This is not an inferential weak-instrument test.
        jacobian_norm: Norm of the mean moment-Jacobian w.r.t. κ.

    """

    concentration: float
    flagged_weak: bool
    jacobian_norm: float


def moment_jacobian_concentration(
    g: ArrayLike,
    dg_dkappa: ArrayLike,
    bandwidth: int | None = None,
) -> MomentJacobianResult:
    """Return a descriptive concentration proxy for the moment Jacobian.

    Forms the scale-free sensitivity proxy

        C = n_observations · (D̄ᵀ Ŝ⁻¹ D̄)

    where ``D̄`` is the sample-mean Jacobian of the moments w.r.t. κ and ``Ŝ``
    is the moment HAC.  Small ``C`` (heuristic threshold ``< 10``) flags low
    local moment sensitivity. Because this implementation does not specify an
    instrumental-variables first stage or a Kleibergen--Paap testing model,
    ``C`` is a descriptive diagnostic rather than a formal weak-instrument
    statistic and should not be compared with published IV thresholds.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        dg_dkappa: Per-period derivative of moments with respect to κ, with
            shape ``(n_observations, m)``.
        bandwidth: HAC truncation lag; ``None`` → data-driven.

    Returns:
        :class:`MomentJacobianResult`.

    """
    moments, n_observations, m = _validate_moment_panel(g)
    dg = jnp.asarray(dg_dkappa)
    if dg.shape != moments.shape or not bool(jnp.all(jnp.isfinite(dg))):
        message = "dg_dkappa must be finite and have the same shape as g."
        raise ValueError(message)
    validated_bandwidth = _validate_bandwidth(bandwidth, n_observations)
    resolved_bandwidth = (
        optimal_bandwidth(n_observations)
        if validated_bandwidth is None
        else validated_bandwidth
    )
    result = moment_jacobian_concentration_kernel(
        moments,
        dg,
        resolved_bandwidth,
    )
    # The kernel's covariance goes straight in: `_validate_full_rank_covariance`
    # takes `ArrayLike` and coerces itself, so the former `np.asarray` round
    # trip through the host bought nothing.
    _validate_full_rank_covariance(result.covariance, dimension=m)
    return MomentJacobianResult(
        concentration=float(result.concentration),
        flagged_weak=bool(result.flagged_weak),
        jacobian_norm=float(result.jacobian_norm),
    )


def boundary_robust_pvalue(
    kappa_hat: float,
    kappa_se: float,
    kappa_null: float = 0.0,
) -> float:
    """Boundary-robust one-sided p-value for ``H0: κ = κ_null`` near ``κ ≥ 0``.

    Because κ is a scale bounded below by 0, its estimator is on the boundary
    of the parameter space when the true κ ≈ 0, and the usual two-sided normal
    test is invalid. The corresponding squared positive-part statistic has the
    ½·χ²₀ + ½·χ²₁ null law (Andrews 2001). For ``z > 0`` its upper tail is
    ``½·P(χ²₁ ≥ z²) = Φ(-z)``; at or inside the boundary the statistic is zero
    and its inclusive upper-tail p-value is one.

    Args:
        kappa_hat: Estimate ``κ̂ ≥ 0``.
        kappa_se: Standard error of ``κ̂`` (must be > 0).
        kappa_null: Null value (default 0, the boundary).

    Returns:
        Boundary-robust one-sided p-value.

    """
    kappa_hat = float(kappa_hat)
    kappa_se = float(kappa_se)
    kappa_null = float(kappa_null)
    if not math.isfinite(kappa_hat) or kappa_hat < 0.0:
        message = "kappa_hat must be finite and nonnegative."
        raise ValueError(message)
    if not math.isfinite(kappa_null) or kappa_null < 0.0:
        message = "kappa_null must be finite and nonnegative."
        raise ValueError(message)
    if not math.isfinite(kappa_se) or kappa_se <= 0.0:
        message = "kappa_se must be finite and strictly positive."
        raise ValueError(message)
    return float(
        boundary_robust_pvalue_kernel(
            jnp.asarray(kappa_hat),
            jnp.asarray(kappa_se),
            jnp.asarray(kappa_null),
        )
    )

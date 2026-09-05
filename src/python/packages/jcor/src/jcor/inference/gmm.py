"""GMM equality-restriction tests: Wald and Hansen J.

Both test ``H0: E[g_t] = 0`` over the same moment process, differing in the HAC
weighting: the Wald uses plain Newey-West, the ``J`` uses Hall centered-HAC
because plain Newey-West is inconsistent under misspecification.  A Hansen-J
p-value is licensed only when the reported parameter count corresponds to
parameters estimated by the same GMM criterion; :func:`hansen_j` makes that
contract explicit and :func:`linear_hansen_j` performs the estimation for
moments affine in their parameters.
"""

# ruff: noqa: F722, F821, UP037  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.stats import chi2

from jcor.core.precision import require_x64
from jcor.core.typing import Array, ArrayLike, Bool, Float, Static  # noqa: TC001
from jcor.inference._common import (
    _MATRIX_NDIM,
    _coerce_affine_jacobian,
    _validate_moment_panel,
    as_index,
)
from jcor.operators.longrun import (
    hall_centered_hac_kernel,
    newey_west_kernel,
    optimal_bandwidth,
)

__all__ = [
    "HansenJKernelResult",
    "HansenJResult",
    "LinearHansenJKernelResult",
    "LinearHansenJResult",
    "WaldKernelResult",
    "WaldResult",
    "gmm_wald",
    "gmm_wald_kernel",
    "hansen_j",
    "hansen_j_kernel",
    "linear_hansen_j",
    "linear_hansen_j_kernel",
]


class WaldKernelResult(NamedTuple):
    """Fixed-structure JAX result for a GMM Wald statistic."""

    stat: Float[Array, ""]
    pvalue: Float[Array, ""]
    gbar: Float[Array, "m"]
    covariance: Float[Array, "m m"]


class HansenJKernelResult(NamedTuple):
    """Fixed-structure JAX result for a Hansen-J statistic."""

    stat: Float[Array, ""]
    pvalue: Float[Array, ""]
    covariance: Float[Array, "m m"]


class LinearHansenJKernelResult(NamedTuple):
    """Fixed-structure iterated-GMM result with convergence evidence."""

    stat: Float[Array, ""]
    pvalue: Float[Array, ""]
    params: Float[Array, "k"]
    covariance: Float[Array, "m m"]
    relative_change: Float[Array, ""]
    converged: Bool[Array, ""]


@partial(jax.jit, static_argnames=("bandwidth",))
def gmm_wald_kernel(
    g: Float[Array, "t m"],
    bandwidth: Static[int],
) -> WaldKernelResult:
    """Evaluate the complete Wald statistic and χ² tail on the JAX graph."""
    moments = jnp.asarray(g)
    n_observations, n_moments = moments.shape
    gbar = jnp.mean(moments, axis=0)
    covariance = newey_west_kernel(moments, bandwidth)
    inverse = jnp.linalg.pinv(covariance)
    stat = n_observations * gbar @ inverse @ gbar
    return WaldKernelResult(
        stat=stat,
        pvalue=chi2.sf(stat, n_moments),
        gbar=gbar,
        covariance=covariance,
    )


@partial(jax.jit, static_argnames=("n_params", "bandwidth", "centered"))
def hansen_j_kernel(
    g: Float[Array, "t m"],
    n_params: Static[int],
    bandwidth: Static[int],
    *,
    centered: Static[bool],
) -> HansenJKernelResult:
    """Evaluate a Hansen-J statistic and χ² tail on the JAX graph."""
    moments = jnp.asarray(g)
    n_observations, n_moments = moments.shape
    covariance = (
        hall_centered_hac_kernel(moments, bandwidth)
        if centered
        else newey_west_kernel(moments, bandwidth)
    )
    gbar = jnp.mean(moments, axis=0)
    stat = n_observations * gbar @ jnp.linalg.pinv(covariance) @ gbar
    return HansenJKernelResult(
        stat=stat,
        pvalue=chi2.sf(stat, n_moments - n_params),
        covariance=covariance,
    )


@partial(
    jax.jit,
    static_argnames=("bandwidth", "centered", "max_iter", "tolerance"),
)
def linear_hansen_j_kernel(
    moments0: Float[Array, "t m"],
    derivative_panel: Float[Array, "t m k"],
    bandwidth: Static[int],
    *,
    centered: Static[bool],
    max_iter: Static[int],
    tolerance: Static[float],
) -> LinearHansenJKernelResult:
    """Fit affine moments by fixed-budget iterated GMM and evaluate Hansen J."""
    _, n_moments = moments0.shape
    n_params = derivative_panel.shape[2]
    dbar = jnp.mean(derivative_panel, axis=0)
    g0bar = jnp.mean(moments0, axis=0)

    def solve(weight: jax.Array) -> jax.Array:
        normal = dbar.T @ weight @ dbar
        score = dbar.T @ weight @ g0bar
        return -jnp.linalg.pinv(normal) @ score

    theta0 = solve(jnp.eye(n_moments, dtype=moments0.dtype))

    def update(
        _: jax.Array,
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        theta, relative_change, converged = state
        fitted = moments0 + jnp.einsum("tmk,k->tm", derivative_panel, theta)
        covariance = (
            hall_centered_hac_kernel(fitted, bandwidth)
            if centered
            else newey_west_kernel(fitted, bandwidth)
        )
        updated = solve(jnp.linalg.pinv(covariance))
        candidate_change = jnp.linalg.norm(updated - theta) / jnp.maximum(
            1.0,
            jnp.linalg.norm(theta),
        )
        next_theta = jnp.where(converged, theta, updated)
        next_change = jnp.where(converged, relative_change, candidate_change)
        return next_theta, next_change, converged | (candidate_change < tolerance)

    theta, relative_change, converged = lax.fori_loop(
        0,
        max_iter,
        update,
        (theta0, jnp.asarray(jnp.inf), jnp.zeros((), dtype=jnp.bool_)),
    )
    fitted = moments0 + jnp.einsum("tmk,k->tm", derivative_panel, theta)
    result = hansen_j_kernel(
        fitted,
        n_params,
        bandwidth,
        centered=centered,
    )
    return LinearHansenJKernelResult(
        stat=result.stat,
        pvalue=result.pvalue,
        params=theta,
        covariance=result.covariance,
        relative_change=relative_change,
        converged=converged,
    )


def _resolved_bandwidth(bandwidth: int | None, n_observations: int) -> int:
    """Resolve and validate the static HAC bandwidth for a numerical kernel."""
    resolved = optimal_bandwidth(n_observations) if bandwidth is None else bandwidth
    index = as_index(resolved)
    if index is None or index < 0 or index >= n_observations:
        message = "bandwidth must be an integer in [0, n_observations)."
        raise ValueError(message)
    return index


class WaldResult(NamedTuple):
    """Return type of :func:`gmm_wald`.

    Attributes:
        stat: Wald statistic ``T·ḡᵀ Ŝ⁻¹ ḡ``.
        dof: Degrees of freedom (number of moments ``m``).
        pvalue: Upper-tail χ²(dof) p-value.
        gbar: Sample-mean moment vector, shape ``(m,)``.

    """

    stat: float
    dof: int
    pvalue: float
    gbar: Float[Array, " m"]


def gmm_wald(g: ArrayLike, bandwidth: int | None = None) -> WaldResult:
    """GMM Wald test of ``H0: E[g_t] = 0`` with HAC weighting.

    Uses the plain Newey-West HAC (moments are mean-zero under the null for a
    Wald test).  Statistic ``W = T · ḡᵀ Ŝ⁻¹ ḡ ~ χ²(m)`` under H0.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        bandwidth: HAC truncation lag; ``None`` → data-driven.

    Returns:
        :class:`WaldResult`.

    """
    moments, n_observations, m = _validate_hansen_dimensions(g, n_params=0)
    require_x64("inference.gmm")
    resolved_bandwidth = _resolved_bandwidth(bandwidth, n_observations)
    result = gmm_wald_kernel(jnp.asarray(moments), resolved_bandwidth)
    covariance = result.covariance
    _require_full_rank_moment_covariance(covariance, m, statistic="GMM Wald")
    return WaldResult(
        stat=float(result.stat),
        dof=m,
        pvalue=float(result.pvalue),
        gbar=result.gbar,
    )


class HansenJResult(NamedTuple):
    """Return type of :func:`hansen_j`.

    Attributes:
        stat: Hansen ``J`` statistic.
        dof: Over-identifying degrees of freedom (``m - k``).
        pvalue: Upper-tail χ²(dof) p-value.
        centered: Whether Hall centering was applied.

    """

    stat: float
    dof: int
    pvalue: float
    centered: bool


class LinearHansenJResult(NamedTuple):
    """Hansen-J result for an affine moment model fitted by iterated GMM.

    Attributes:
        stat: Hansen ``J`` statistic at the GMM estimate.
        dof: Over-identifying degrees of freedom ``m - k``.
        pvalue: Upper-tail χ² p-value.
        centered: Whether Hall centering was applied.
        params: GMM parameter estimate, shape ``(k,)``.

    """

    stat: float
    dof: int
    pvalue: float
    centered: bool
    params: Float[Array, " k"]


def _validate_hansen_dimensions(
    g: ArrayLike,
    n_params: int,
) -> tuple[Float[Array, "t m"], int, int]:
    """Validate the moment panel and over-identification dimensions.

    The panel half is :func:`jcor.inference._common._validate_moment_panel`;
    only the over-identification pair below is specific to Hansen J.
    """
    moments, n_observations, m = _validate_moment_panel(g)
    if n_params < 0:
        message = "n_params must be nonnegative."
        raise ValueError(message)
    if m <= n_params:
        message = (
            "Hansen J requires over-identification: the number of moments "
            f"m={m} must exceed n_params={n_params}."
        )
        raise ValueError(message)
    return moments, n_observations, m


def _require_full_rank_moment_covariance(
    covariance: ArrayLike,
    m: int,
    *,
    statistic: str,
) -> None:
    """Fail closed when a chi-square reference has fewer than `m` directions."""
    covariance_rank = int(jnp.linalg.matrix_rank(jnp.asarray(covariance)))
    if covariance_rank < m:
        message = (
            f"{statistic} requires a full-rank moment covariance; "
            f"observed rank={covariance_rank} for m={m}."
        )
        raise ValueError(message)


def hansen_j(
    g: ArrayLike,
    n_params: int = 0,
    bandwidth: int | None = None,
    *,
    centered: bool = True,
    estimated_by_gmm: bool = False,
) -> HansenJResult:
    """Hansen ``J`` over-identification test with Hall centered-HAC weighting.

    ``J = n_observations · ḡᵀ Ŝ⁻¹ ḡ`` where ``Ŝ`` is the Hall centered HAC when
    ``centered=True`` (consistent under misspecification), or the plain
    Newey-West HAC when ``centered=False`` (the inconsistent baseline, kept for
    the size-power comparison).  ``J ~ χ²(m - k)`` under correct specification
    only when the ``k`` parameters were estimated by the corresponding GMM
    criterion.  Merely evaluating moments at an external estimate and
    subtracting ``k`` does not have that reference distribution, so this
    function rejects that call unless ``estimated_by_gmm=True``.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        n_params: Number of estimated parameters ``k`` (e.g. 1 when κ is
            estimated).  Over-identifying dof is ``m - k``.
        bandwidth: HAC truncation lag; ``None`` → data-driven.
        centered: Use Hall centering (default ``True``).
        estimated_by_gmm: Explicit confirmation that any ``n_params > 0``
            parameters were estimated with the same GMM moments/criterion.
            Prefer :func:`linear_hansen_j` when the moments are affine.

    Returns:
        :class:`HansenJResult`.

    """
    moments, n_observations, m = _validate_hansen_dimensions(g, n_params)
    if n_params > 0 and not estimated_by_gmm:
        message = (
            "Hansen J with n_params > 0 requires parameters estimated by the "
            "same GMM criterion; use linear_hansen_j for affine moments or pass "
            "estimated_by_gmm=True only for an already-fitted GMM estimate."
        )
        raise ValueError(message)
    require_x64("inference.gmm")
    resolved_bandwidth = _resolved_bandwidth(bandwidth, n_observations)
    result = hansen_j_kernel(
        jnp.asarray(moments),
        n_params,
        resolved_bandwidth,
        centered=centered,
    )
    covariance = result.covariance
    _require_full_rank_moment_covariance(covariance, m, statistic="Hansen J")
    dof = m - n_params
    return HansenJResult(
        stat=float(result.stat),
        dof=dof,
        pvalue=float(result.pvalue),
        centered=centered,
    )


def linear_hansen_j(
    g0: ArrayLike,
    jacobian: ArrayLike,
    bandwidth: int | None = None,
    *,
    centered: bool = True,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> LinearHansenJResult:
    """Estimate an affine moment model by GMM, then compute Hansen ``J``.

    The supplied model is

    ``g_t(theta) = g0_t + D_t @ theta``.

    ``jacobian`` may be the common matrix ``D`` with shape ``(m, k)`` or a
    per-observation array with shape ``(n_observations, m, k)``.  Iterated GMM
    updates ``theta`` using the same Hall-centered (or explicit plain-HAC
    comparator) weight used by the final J statistic.  This closes the
    estimator-contract gap that arises when moments are evaluated at an
    external OLS estimate.

    The parameter space here is unconstrained.  If a structural parameter is
    on a boundary (for example ``theta = kappa² = 0``), the ordinary interior
    ``chi²(m-k)`` reference is not licensed and the caller must use a
    boundary-robust procedure instead.

    Args:
        g0: Moment contributions at ``theta = 0``, shape
            ``(n_observations, m)``.
        jacobian: Affine parameter derivative, shape ``(m, k)`` or
            ``(n_observations, m, k)``.
        bandwidth: HAC truncation lag; ``None`` uses the data-driven default.
        centered: Use Hall-centered HAC (default ``True``).
        max_iter: Maximum iterated-GMM weight updates.
        tol: Relative parameter-change tolerance.

    Returns:
        :class:`LinearHansenJResult` at the GMM estimate.

    """
    moments0 = jnp.asarray(g0)
    if moments0.ndim != _MATRIX_NDIM:
        message = "g0 must be a two-dimensional (n_observations, m) array."
        raise ValueError(message)
    derivative_panel = _coerce_affine_jacobian(moments0, jacobian)
    n_params = derivative_panel.shape[2]
    _validate_hansen_dimensions(moments0, n_params)
    if n_params < 1:
        message = "linear_hansen_j requires at least one parameter."
        raise ValueError(message)
    if max_iter < 1 or not math.isfinite(tol) or tol <= 0.0:
        message = "max_iter must be positive and tol must be finite and positive."
        raise ValueError(message)
    dbar = derivative_panel.mean(axis=0)
    if jnp.linalg.matrix_rank(dbar) < n_params:
        message = "the mean moment jacobian does not identify every parameter."
        raise ValueError(message)
    require_x64("inference.gmm")
    resolved_bandwidth = _resolved_bandwidth(bandwidth, moments0.shape[0])
    result = linear_hansen_j_kernel(
        jnp.asarray(moments0),
        jnp.asarray(derivative_panel),
        resolved_bandwidth,
        centered=centered,
        max_iter=max_iter,
        tolerance=tol,
    )
    if not bool(result.converged):
        message = (
            "iterated linear GMM did not converge within "
            f"{max_iter} iterations "
            f"(relative change={float(result.relative_change):.3e})."
        )
        raise RuntimeError(message)
    covariance = result.covariance
    _require_full_rank_moment_covariance(
        covariance,
        moments0.shape[1],
        statistic="Hansen J",
    )
    return LinearHansenJResult(
        stat=float(result.stat),
        dof=moments0.shape[1] - n_params,
        pvalue=float(result.pvalue),
        centered=centered,
        params=result.params,
    )

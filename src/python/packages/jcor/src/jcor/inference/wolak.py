"""Wolak (1989) inequality test with Monte-Carlo chi-bar-squared weights.

One-sided test of the proven lower bounds ``g_ij >= 0``, with the mixing weights
computed by simulation under the least-favourable null.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import chi2

from jcor.core.precision import require_x64
from jcor.core.random import cell_key
from jcor.core.typing import Array, ArrayLike, Bool, Float  # noqa: TC001
from jcor.inference._common import (
    _BINDING_TOLERANCE,
    _chi2_survival,
    _require_projection_solution,
    _validate_bandwidth,
    _validate_full_rank_covariance,
    _validate_moment_panel,
    _validate_positive_integer,
)
from jcor.inference._projection import (
    POLISH_ROUNDS,
    PROJECTION_TOLERANCE,
    refine_projection_solution,
)
from jcor.operators.longrun import hall_centered_hac, hall_centered_hac_kernel
from jcor.optimize.simplex import batched_nonnegative_quadratic_program

if TYPE_CHECKING:
    from jcor.optimize.simplex import NonnegativeQPSolution

#: Cyclic-coordinate sweeps for the orthant projection.
#:
#: ``batched_nonnegative_quadratic_program`` defaults to 256, which is not
#: enough for this caller: a *well-conditioned* projection from the H1 pilot
#: (``cond(Q) = 149``, ``K = 3``) reports ``converged=False`` at 256 sweeps with
#: residual ``6.113e-03``, and ``_require_projection_solution`` then fails
#: closed. The budget, not the tolerance, is the binding constraint — loosening
#: ``active_tolerance`` by four orders changes nothing, while 4096 sweeps
#: converge to residual ``2.274e-12`` at the tightest tolerance and 65536 sweeps
#: reproduce that same residual, so the iteration count is saturated here.
#:
#: The solver's sweep loop is ``lax.fori_loop`` (``optimize/simplex.py:195``)
#: and therefore does **not** exit early, so this budget is paid in full on
#: every call. It is set here rather than on the shared default because this
#: caller's problems are tiny (``K`` = number of moments), making 4096 sweeps of
#: ``O(K²)`` work negligible, whereas raising the global default would impose an
#: unconditional 16x cost on every other consumer.
#:
#: 4096 is also the budget at which the shared active-set refinement in
#: :mod:`jcor.inference._projection` can finish the
#: job. Measured on the Paper 3 chi-bar calibration (``m = 15``, 3000 draws,
#: ``cond(Q) = 1840``): the polish clears 940/3000 rejected lanes to 0/3000 from
#: a 4096-sweep iterate, but only 3000 - 89 lanes from a 256-sweep one, because
#: a 256-sweep iterate is still too far away for the active set to be
#: identified. The budget therefore cannot revert to the shared default of 256.
_PROJECTION_MAX_SWEEPS = 4096

#: Active-set rounds needed by the transformable production MC kernel after its
#: well-scaled coordinate solve. It reaches the identical binding tally in two
#: rounds on the pinned Paper-3 covariance and sampled production cells. Eager
#: inference retains the shared conservative default: Paper 2's frontier
#: statistic includes less-well-scaled one-lane projections that need more than
#: two ratio-test steps.
_KERNEL_PROJECTION_POLISH_ROUNDS = 2


class WolakResult(NamedTuple):
    """Return type of :func:`wolak_test`.

    Attributes:
        stat: Wolak (distance / IU) statistic under the inequality null.
        pvalue: Monte-Carlo chi-bar-squared p-value.
        weights: Estimated χ̄² mixing weights ``w_j``, length ``m + 1``.
        n_binding: Number of moments estimated as binding (ĝ_j < 0 direction).

    """

    stat: float
    pvalue: float
    weights: Float[Array, " m1"]
    n_binding: int


class ChiBarWeightsKernelResult(NamedTuple):
    """Array-only result of the transformable chi-bar calibration."""

    weights: Float[Array, " m1"]
    converged: Bool[Array, ""]


class WolakKernelResult(NamedTuple):
    """Array-only result of :func:`wolak_test_kernel`."""

    stat: Float[Array, ""]
    pvalue: Float[Array, ""]
    weights: Float[Array, " m1"]
    n_binding: Array
    converged: Bool[Array, ""]


def _nonnegative_projection_solution(
    targets: Float[Array, "b k"],
    w_inv: Float[Array, "k k"],
    *,
    max_sweeps: int = _PROJECTION_MAX_SWEEPS,
    polish_rounds: int = POLISH_ROUNDS,
) -> NonnegativeQPSolution:
    """Return the shared solver carrier before eager convergence validation."""
    target_matrix = jnp.asarray(targets)
    symmetric_inverse = 0.5 * (w_inv + w_inv.T)
    linear = -(target_matrix @ symmetric_inverse)
    solution = batched_nonnegative_quadratic_program(
        symmetric_inverse,
        linear,
        active_tolerance=_BINDING_TOLERANCE,
        tolerance=PROJECTION_TOLERANCE,
        max_sweeps=max_sweeps,
    )
    return refine_projection_solution(
        solution,
        symmetric_inverse,
        linear,
        active_tolerance=_BINDING_TOLERANCE,
        polish_rounds=polish_rounds,
    )


def _project_nonnegative_quadform_batch_kernel(
    targets: Float[Array, "b k"],
    w_inv: Float[Array, "k k"],
    *,
    max_sweeps: int = _PROJECTION_MAX_SWEEPS,
) -> tuple[Float[Array, "b k"], Bool[Array, "b k"], Bool[Array, " b"]]:
    """Project a batch onto the orthant without a host validation boundary."""
    solution = _nonnegative_projection_solution(
        targets,
        w_inv,
        max_sweeps=max_sweeps,
        polish_rounds=_KERNEL_PROJECTION_POLISH_ROUNDS,
    )
    return (
        solution.primal,
        jnp.asarray(solution.active_mask, dtype=bool),
        jnp.asarray(solution.converged, dtype=bool),
    )


def _project_nonnegative_quadform_batch(
    targets: Float[Array, "b k"],
    w_inv: Float[Array, "k k"],
    *,
    context: str,
    max_sweeps: int = _PROJECTION_MAX_SWEEPS,
) -> tuple[Float[Array, "b k"], Bool[Array, "b k"]]:
    """Eager validated adapter over the transformable projection kernel."""
    solution = _nonnegative_projection_solution(
        targets,
        w_inv,
        max_sweeps=max_sweeps,
    )
    _require_projection_solution(solution, context=context)
    return solution.primal, jnp.asarray(solution.active_mask, dtype=bool)


def wolak_statistic(
    gbar: ArrayLike, covariance: ArrayLike, n_observations: int
) -> tuple[float, int]:
    """Wolak IU (distance) statistic for ``H0: E[g] ≥ 0`` vs unrestricted.

    ``IU = n_observations · min_{t ≥ 0} (t - ḡ)ᵀ Ŝ⁻¹ (t - ḡ)`` is the squared
    distance from the estimate to the non-negative orthant in the ``Ŝ⁻¹``
    metric. A large value is evidence against the inequality restriction.

    Args:
        gbar: Sample-mean moment vector ``ḡ``, shape ``(m,)``.
        covariance: HAC covariance of ``√n_observations·ḡ``, shape ``(m, m)``.
        n_observations: Sample size.

    Returns:
        Tuple ``(statistic, n_binding)``.

    """
    mean = jnp.asarray(gbar)
    if mean.ndim != 1 or mean.size < 1 or not bool(jnp.all(jnp.isfinite(mean))):
        message = "gbar must be a nonempty finite vector."
        raise ValueError(message)
    n_observations = _validate_positive_integer(
        n_observations,
        name="n_observations",
    )
    covariance = _validate_full_rank_covariance(
        covariance,
        dimension=mean.size,
    )
    s_inv = jnp.linalg.pinv(covariance)
    projection, active_mask = _project_nonnegative_quadform_batch(
        mean[None, :],
        s_inv,
        context="Wolak statistic",
    )
    t = projection[0]
    diff = t - mean
    stat = float(n_observations * diff @ s_inv @ diff)
    n_binding = int(jnp.sum(active_mask[0]))
    return stat, n_binding


def chi_bar_squared_weights_mc(
    covariance: ArrayLike,
    n_mc: int = 5000,
    seed: int = 0,
) -> Float[Array, " m1"]:
    """Monte-Carlo χ̄² mixing weights ``w_j`` under the least-favourable null.

    Under ``H0`` with all inequalities binding at zero, the Wolak statistic

        ``IU = min_{t ≥ 0} (t - standardized)ᵀ Ŝ⁻¹ (t - standardized)``

    is distributed as a mixture ``Σ_{j=0}^{m} w_j χ²_j``.  Because ``IU`` is the
    squared distance from ``standardized`` to the non-negative orthant, its
    χ²(j) component
    has ``j = `` the number of **binding (zero) coordinates** of the projection
    ``t̂`` (the coordinates that ``ARE`` constrained to the boundary), which is
    the COMPLEMENT of the strictly-positive (free) coordinates.  The weight
    ``w_j`` is therefore the probability that exactly ``j`` coordinates of the
    projection of ``standardized ~ N(0, covariance)`` are at the boundary.
    We estimate the weights by Monte Carlo: draw, project, and tally binding
    coordinates.

    Args:
        covariance: HAC covariance of the (whitened) moments, shape ``(m, m)``.
        n_mc: Number of Monte-Carlo draws.
        seed: RNG seed.

    Returns:
        Weight vector ``w`` of length ``m + 1`` summing to 1.

    """
    covariance = _validate_full_rank_covariance(covariance)
    n_mc = _validate_positive_integer(n_mc, name="n_mc")
    m = covariance.shape[0]
    inverse = jnp.linalg.pinv(covariance)
    key = cell_key(seed, "inference", "chi_bar_squared_weights_mc")
    standard_normals = jax.random.normal(key, shape=(n_mc, m))
    cholesky_factor = jnp.linalg.cholesky(covariance + 1e-12 * jnp.eye(m))
    standardized = (cholesky_factor @ standard_normals.T).T
    _, active_mask = _project_nonnegative_quadform_batch(
        standardized,
        inverse,
        context="Wolak chi-bar calibration",
    )
    n_binding = jnp.sum(active_mask, axis=1)
    counts = jnp.bincount(n_binding, length=m + 1).astype(standardized.dtype)
    return counts / counts.sum()


@partial(jax.jit, static_argnames=("n_mc", "max_sweeps"))
def chi_bar_squared_weights_mc_kernel(
    key: Array,
    covariance: Float[Array, "m m"],
    *,
    n_mc: int,
    max_sweeps: int = _PROJECTION_MAX_SWEEPS,
) -> ChiBarWeightsKernelResult:
    """Calibrate chi-bar weights from an explicit key on the JAX graph."""
    m = covariance.shape[0]
    s_inv = jnp.linalg.pinv(covariance)
    standard_normals = jax.random.normal(key, shape=(n_mc, m))
    cholesky_factor = jnp.linalg.cholesky(covariance + 1e-12 * jnp.eye(m))
    standardized = (cholesky_factor @ standard_normals.T).T
    _, active_mask, converged = _project_nonnegative_quadform_batch_kernel(
        standardized,
        s_inv,
        max_sweeps=max_sweeps,
    )
    # dof of the χ²(j) component = number of BINDING (zero) coordinates of
    # the projection onto the non-negative orthant (the complement of the
    # strictly-positive free coordinates).
    n_binding = jnp.sum(active_mask, axis=1)
    # `length=` is the jnp spelling of `minlength=` and is required, not
    # cosmetic: `jnp.bincount` needs a static output size. `m + 1` already is
    # one, so this is the same fixed-width histogram the NumPy call produced.
    counts = jnp.bincount(n_binding, length=m + 1).astype(standardized.dtype)
    return ChiBarWeightsKernelResult(
        weights=counts / counts.sum(),
        converged=jnp.all(converged),
    )


@jax.jit
def chi_bar_squared_sf_kernel(
    stat: Float[Array, ""],
    weights: Float[Array, " m1"],
) -> Float[Array, ""]:
    """Evaluate the chi-bar survival function without host scalar transfers."""
    degrees = jnp.arange(1, weights.shape[0])
    continuous = jnp.sum(weights[1:] * chi2.sf(stat, degrees))
    return continuous + jnp.where(stat <= 0.0, weights[0], 0.0)


def chi_bar_squared_sf(stat: float, weights: ArrayLike) -> float:
    """Upper-tail probability of a χ̄² mixture at ``stat``.

    ``P(χ̄² > stat) = Σ_j w_j · P(χ²_j > stat)`` with ``χ²_0 ≡ 0`` (so its
    survival is 0 for ``stat > 0``).

    Args:
        stat: Observed statistic.
        weights: χ̄² mixing weights, length ``m + 1`` (index = dof).

    Returns:
        Mixture p-value.

    """
    stat = float(stat)
    weights = jnp.asarray(weights)
    if not math.isfinite(stat) or stat < 0.0:
        message = "stat must be finite and nonnegative."
        raise ValueError(message)
    if (
        weights.ndim != 1
        or weights.size < 1
        or not bool(jnp.all(jnp.isfinite(weights)))
        or bool(jnp.any(weights < 0.0))
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-8)
    ):
        message = "weights must be a finite nonnegative vector summing to one."
        raise ValueError(message)
    p = 0.0
    for j, wj in enumerate(weights):
        if wj <= 0.0:
            continue
        # dof 0 is the point mass at zero, whose survival is 0 for any stat > 0.
        sf = _chi2_survival(stat, j) if j > 0 else float(stat <= 0.0)
        p += wj * sf
    return float(p)


def wolak_test(
    g: ArrayLike,
    bandwidth: int | None = None,
    n_mc: int = 5000,
    seed: int = 0,
) -> WolakResult:
    """Wolak one-sided test of ``H0: E[g_t] ≥ 0`` with MC χ̄² weights.

    Args:
        g: Per-period moment process with shape ``(n_observations, m)``, already
            clustered to a manageable ``m`` via :func:`cluster_pairs`.
        bandwidth: HAC truncation lag; ``None`` → data-driven.
        n_mc: Monte-Carlo draws for the χ̄² weight calibration.
        seed: RNG seed for the weight calibration.

    Returns:
        :class:`WolakResult`.

    """
    require_x64("inference.wolak.wolak_test")
    g, n_observations, n_moments = _validate_moment_panel(g)
    n_mc = _validate_positive_integer(n_mc, name="n_mc")
    bandwidth = _validate_bandwidth(bandwidth, n_observations)
    mean = g.mean(axis=0)
    covariance = _validate_full_rank_covariance(
        hall_centered_hac(g, bandwidth=bandwidth),
        dimension=n_moments,
    )
    stat, n_binding = wolak_statistic(mean, covariance, n_observations)
    weights = chi_bar_squared_weights_mc(covariance, n_mc=n_mc, seed=seed)
    pvalue = chi_bar_squared_sf(stat, weights)
    return WolakResult(stat=stat, pvalue=pvalue, weights=weights, n_binding=n_binding)


@partial(jax.jit, static_argnames=("bandwidth", "n_mc", "max_sweeps"))
def wolak_test_kernel(
    key: Array,
    g: Float[Array, "t m"],
    *,
    bandwidth: int | None,
    n_mc: int,
    max_sweeps: int = _PROJECTION_MAX_SWEEPS,
) -> WolakKernelResult:
    """Evaluate the complete Wolak test from an explicit key on the JAX graph."""
    moments = jnp.asarray(g)
    n_observations = moments.shape[0]
    mean = jnp.mean(moments, axis=0)
    covariance = hall_centered_hac_kernel(moments, bandwidth)
    inverse = jnp.linalg.pinv(covariance)
    projection, active_mask, statistic_converged = (
        _project_nonnegative_quadform_batch_kernel(
            mean[None, :],
            inverse,
            max_sweeps=max_sweeps,
        )
    )
    difference = projection[0] - mean
    stat = n_observations * difference @ inverse @ difference
    calibration = chi_bar_squared_weights_mc_kernel(
        key,
        covariance,
        n_mc=n_mc,
        max_sweeps=max_sweeps,
    )
    return WolakKernelResult(
        stat=stat,
        pvalue=chi_bar_squared_sf_kernel(stat, calibration.weights),
        weights=calibration.weights,
        n_binding=jnp.sum(active_mask[0]),
        converged=jnp.all(statistic_converged) & calibration.converged,
    )

"""Indicator-GMS diagnostics for moment inequalities.

The implementation uses an Andrews--Barwick-style indicator selection rule with
MMM or QLR statistics. It does not implement the Andrews--Barwick Table-I RMS
size correction; zero-correction results are therefore labelled uncorrected
indicator GMS.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jcor.core.random import cell_key
from jcor.core.typing import Array, ArrayLike, Bool, Float  # noqa: TC001
from jcor.inference._common import (
    _BINDING_TOLERANCE,
    _require_projection_solution,
    _validate_bandwidth,
    _validate_full_rank_covariance,
    _validate_moment_panel,
    _validate_positive_integer,
)
from jcor.inference._projection import (
    PROJECTION_TOLERANCE,
    refine_projection_solution,
)
from jcor.operators.longrun import hall_centered_hac
from jcor.optimize.simplex import batched_nonnegative_quadratic_program

#: Cyclic-coordinate sweeps for the QLR orthant projection.
#:
#: The shared default of 256 is not enough here for the same reason it is not
#: enough in :mod:`jcor.inference.wolak`: the sweep loop is a ``lax.fori_loop``
#: with no early exit and only linear convergence at rate ``~(1 - 1/cond(Q))``.
#: Measured on a ``k = 15`` QLR-shaped batch of 3001 lanes with
#: ``cond(Omega_inv) = 2052``, before and after the active-set refinement:
#:
#: ===========  ==================  =================
#: max_sweeps   solver alone        solver + polish
#: ===========  ==================  =================
#: 256          2898/3001 rejected  104/3001 rejected
#: 1024         1908/3001 rejected  6/3001 rejected
#: 4096         959/3001 rejected   0/3001 rejected
#: 16384        0/3001 rejected     0/3001 rejected
#: ===========  ==================  =================
#:
#: **4096 was tuned on that one batch and does not generalise** (raised to 16384
#: on 2026-08-08). "The smallest budget from which the refinement reaches every
#: lane" was true of the qwen3-embedding-4b baseline batch it was measured on and
#: false of the qwen3-embedding-8b ablation arm, whose 2001-lane batch leaves
#: lane 465 at a projected-KKT residual of ``31.1464`` against a ``4.23935``
#: threshold. That lane is not rescuable by refinement: 32, 128 and 384 polish
#: rounds all return the identical 31.1464, because a 4096-sweep iterate is too
#: far from the solution for the active set to be identified — the same
#: mechanism :mod:`jcor.inference.wolak` records for its 256-sweep case, one
#: batch further along. Measured on the failing batch:
#:
#: ===========  =============  =============  ==================
#: max_sweeps   polish rounds  rejected       lane 465 residual
#: ===========  =============  =============  ==================
#: 4096         32/128/384     1/2001         3.11464e+01
#: 16384        32/128/384     **0/2001**     3.81470e-06
#: ===========  =============  =============  ==================
#:
#: 16384 is where the table above shows the *solver alone* reaching 0/3001, so it
#: no longer depends on the refinement finishing the job on any particular batch.
#: Cost on the 2001-lane batch is 0.48 s -> 1.64 s per call, ~10 s across the
#: whole pipeline — bought against a failure that published a referee-facing
#: robustness arm as ``0.000000``.
#:
#: Do not re-tune this downward against a single batch. The 4096 figure was
#: correct for its measurement and wrong as a constant, which is the failure mode
#: to avoid rather than the number.
_PROJECTION_MAX_SWEEPS = 16384


class GmsResult(NamedTuple):
    """Return type of :func:`indicator_gms`.

    Attributes:
        stat: Test statistic ``S`` (modified-method-of-moments / QLR-type).
        crit: GMS data-dependent critical value at ``1 - alpha``.
        reject: Whether ``stat > crit`` (inequality restriction rejected).
        pvalue: GMS p-value (smallest α at which the null is rejected).
        n_selected: Number of moments selected as (near-)binding by the
            φ-function.
        kappa_n: Moment-selection tuning constant ``κ_n`` used.
        eta: RMS size-correction factor added to the simulated critical value.
        calibration: Explicit calibration label. The default is
            ``"indicator_gms_uncorrected"``.
        rms_table_i_calibrated: Always false: this implementation does not
            validate a supplied correction against Andrews--Barwick Table I.

    """

    stat: float
    crit: float
    reject: bool
    pvalue: float
    n_selected: int
    kappa_n: float
    eta: float
    calibration: str
    rms_table_i_calibrated: bool


class _GmsSetup(NamedTuple):
    """Validated arrays and controls for one indicator-GMS run."""

    moments: Float[Array, "t m"]
    n_observations: int
    n_moments: int
    n_mc: int
    bandwidth: int | None
    alpha: float
    eta: float
    kappa_n: float


def _validate_gms_setup(
    g: ArrayLike,
    *,
    alpha: float,
    bandwidth: int | None,
    n_mc: int,
    kappa_n: float | None,
    eta: float,
    statistic: str,
) -> _GmsSetup:
    """Validate public GMS inputs before covariance estimation or simulation."""
    moments, n_observations, n_moments = _validate_moment_panel(g)
    n_mc = _validate_positive_integer(n_mc, name="n_mc")
    bandwidth = _validate_bandwidth(bandwidth, n_observations)
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        message = "alpha must be finite and lie strictly between 0 and 1."
        raise ValueError(message)
    eta = float(eta)
    if not math.isfinite(eta) or eta < 0.0:
        message = "eta must be finite and nonnegative."
        raise ValueError(message)
    if statistic not in ("qlr", "mmm"):
        message = f"unknown GMS statistic {statistic!r}"
        raise ValueError(message)
    selection_tuning = (
        math.sqrt(0.3 * math.log(max(n_observations, 3)))
        if kappa_n is None
        else float(kappa_n)
    )
    if not math.isfinite(selection_tuning) or selection_tuning <= 0.0:
        message = "kappa_n must be finite and strictly positive."
        raise ValueError(message)
    return _GmsSetup(
        moments=moments,
        n_observations=n_observations,
        n_moments=n_moments,
        n_mc=n_mc,
        bandwidth=bandwidth,
        alpha=alpha,
        eta=eta,
        kappa_n=selection_tuning,
    )


class _GmsMomentGeometry(NamedTuple):
    """Centered HAC moment geometry shared by the statistic and the draws."""

    gbar: Float[Array, " k"]
    covariance: Float[Array, "k k"]
    sig: Float[Array, " k"]
    precision: Float[Array, "k k"]


def _gms_moment_geometry(
    moments: Float[Array, "t m"],
    *,
    bandwidth: int | None,
    n_moments: int,
) -> _GmsMomentGeometry:
    """Estimate the centered HAC moment covariance and its derived scalings.

    Args:
        moments: Validated moment panel of shape ``(T, k)``.
        bandwidth: Validated HAC truncation lag, or ``None`` for data-driven.
        n_moments: Number of moments ``k``.

    Returns:
        The moment mean, HAC covariance, coordinate scales ``σ`` and the
        pseudo-inverse metric ``Ω̂⁻¹``.

    """
    gbar = moments.mean(axis=0)
    # Inequality-null moments may be strictly slack, so their nonzero mean is
    # not sampling variation. Center before estimating Var(sqrt(T) * gbar).
    covariance = _validate_full_rank_covariance(
        hall_centered_hac(moments, bandwidth=bandwidth),
        dimension=n_moments,
    )
    sig = jnp.sqrt(jnp.clip(jnp.diag(covariance), 1e-12, None))
    return _GmsMomentGeometry(
        gbar=gbar,
        covariance=covariance,
        sig=sig,
        precision=jnp.linalg.pinv(covariance),
    )


def _gms_moment_selection(
    root_t: float,
    geometry: _GmsMomentGeometry,
    kappa_n: float,
) -> tuple[Bool[Array, " k"], int]:
    """Apply the ``φ⁽¹⁾`` indicator selection ``ξ_j > 1 ⇒ slack``.

    Args:
        root_t: ``√T`` scaling of the moment mean.
        geometry: Centered HAC moment geometry.
        kappa_n: Moment-selection tuning constant ``κ_n``.

    Returns:
        Tuple of the per-moment slack mask and the number of selected
        (near-binding) moments.

    """
    xi = (root_t * geometry.gbar / geometry.sig) / kappa_n
    slack = xi > 1.0
    return slack, int(jnp.sum(~slack))


def _gms_null_draws(
    geometry: _GmsMomentGeometry,
    slack: Bool[Array, " k"],
    *,
    n_moments: int,
    n_mc: int,
    seed: int,
) -> Float[Array, "b k"]:
    """Draw GMS-recentered ``N(0, Ω̂)`` replicates for the critical value.

    GMS recentering pushes slack coordinates far positive so they never bind,
    retaining only the selected, near-binding set. The shift was an in-place
    masked add on a NumPy buffer, which a ``jax.Array`` cannot do; ``where``
    broadcasts ``slack`` over the leading axis and is unconditional, so the
    former ``if np.any(slack)`` guard is redundant rather than dropped.

    Args:
        geometry: Centered HAC moment geometry.
        slack: Per-moment slack mask from the ``φ⁽¹⁾`` selection.
        n_moments: Number of moments ``k``.
        n_mc: Number of Monte-Carlo draws.
        seed: RNG seed.

    Returns:
        Recentered draws of shape ``(n_mc, k)``.

    """
    covariance = geometry.covariance
    w_eig, eigenvectors = jnp.linalg.eigh(0.5 * (covariance + covariance.T))
    w_eig = jnp.maximum(w_eig, 1e-14)
    cholesky_factor = (
        (eigenvectors * jnp.sqrt(w_eig)) @ eigenvectors.T
    )  # cholesky_factor cholesky_factor' = covariance (symmetric PSD sqrt)
    big = 1e6 * float(jnp.max(geometry.sig))

    key = cell_key(seed, "inference", "indicator_gms")
    normal_draws = jax.random.normal(key, shape=(n_moments, n_mc))
    standardized = (cholesky_factor @ normal_draws).T  # (n_mc, k) ~ N(0, covariance)
    return jnp.where(slack[None, :], standardized + big, standardized)


def _gms_observed_and_null(
    statistic: str,
    *,
    root_t: float,
    geometry: _GmsMomentGeometry,
    standardized: Float[Array, "b k"],
) -> tuple[float, Float[Array, " b"]]:
    """Evaluate the observed statistic and its null replicates.

    Args:
        statistic: ``"qlr"`` (correlation-aware) or ``"mmm"`` (diagonal only).
        root_t: ``√T`` scaling of the moment mean.
        geometry: Centered HAC moment geometry.
        standardized: GMS-recentered null draws of shape ``(n_mc, k)``.

    Returns:
        Tuple of the observed statistic and the simulated null values.

    """
    if statistic == "mmm":
        stat = _mmm_statistic(
            root_t * geometry.gbar / geometry.sig,
            geometry.covariance,
        )
        sims = jnp.sum(
            jnp.minimum(standardized / geometry.sig[None, :], 0.0) ** 2,
            axis=1,
        )
        return stat, sims
    # The observed vector and all Monte Carlo draws share one device dispatch.
    # The random key, shape, dtype, and downstream ordering remain unchanged;
    # only the B+1 scalar NNLS host calls disappear.
    m_data = root_t * geometry.gbar  # √T·ḡ (variance ≈ covariance under H0)
    qlr_values = _qlr_statistics(
        jnp.concatenate((m_data[None, :], standardized), axis=0),
        geometry.precision,
    )
    return float(qlr_values[0]), qlr_values[1:]


def _finite_mc_calibration(
    simulations: Float[Array, " b"],
    statistic: float,
    *,
    eta: float,
    alpha: float,
) -> tuple[float, float, bool]:
    """Return a mutually consistent plus-one p-value and order-statistic cutoff."""
    adjusted = simulations + eta
    n_mc = adjusted.size
    extreme = int(jnp.count_nonzero(adjusted >= statistic))
    pvalue = float((extreme + 1) / (n_mc + 1))
    max_extreme = math.floor(alpha * (n_mc + 1) - 1.0 + 1e-12)
    if max_extreme < 0:
        critical = float("inf")
    else:
        critical_index = n_mc - min(max_extreme, n_mc - 1) - 1
        critical = float(jnp.sort(adjusted)[critical_index])
    reject = bool(statistic > critical)
    if reject != (pvalue <= alpha):  # defensive tie/order-statistic invariant
        message = "Monte Carlo critical value and plus-one p-value disagree."
        raise RuntimeError(message)
    return critical, pvalue, reject


def _mmm_statistic(z: Float[Array, " k"], omega: Float[Array, "k k"]) -> float:
    """Modified-method-of-moments (MMM / S₃) statistic for ``H0: E[m] ≥ 0``.

    ``S₃ = Σ_j [min(z_j, 0)]²`` is the sum of squared *negative parts* of
    the studentized moments (Andrews-Barwick 2012).
    Ignores cross-moment correlation (diagonal only).  Zero when all moments are
    non-negative; large when several are strongly negative.

    Args:
        z: Studentized moment vector ``√n·m̄`` (unit-variance), shape ``(k,)``.
        omega: Correlation matrix (unused here; kept for signature parity).

    Returns:
        The MMM statistic value.

    """
    _ = omega
    neg = jnp.minimum(z, 0.0)
    return float(jnp.sum(neg * neg))


def _qlr_statistics(
    targets: Float[Array, "b k"],
    omega_inv: Float[Array, "k k"],
) -> Float[Array, " b"]:
    """Compute adjusted QLR statistics for a batch under ``H0: E[m]≥0``.

    ``S = min_{t ≥ 0} (z - t)ᵀ Ω⁻¹ (z - t)`` is the squared distance from the
    moment vectors ``z`` to the non-negative orthant in the ``Ω⁻¹`` metric.
    All leading lanes share one matrix and enter one internally-vmapped JAX
    solve, replacing the previous per-lane SciPy NNLS host calls.

    Args:
        targets: Moment vectors ``√n·ḡ`` and/or null draws, shape ``(B, k)``.
        omega_inv: Inverse moment covariance ``Ω⁻¹``, shape ``(k, k)``.

    Returns:
        QLR statistic values of shape ``(B,)``.

    """
    target_matrix = jnp.asarray(targets)
    omega = 0.5 * (omega_inv + omega_inv.T)
    linear = -(target_matrix @ omega)
    solution = batched_nonnegative_quadratic_program(
        omega,
        linear,
        active_tolerance=_BINDING_TOLERANCE,
        tolerance=PROJECTION_TOLERANCE,
        max_sweeps=_PROJECTION_MAX_SWEEPS,
    )
    solution = refine_projection_solution(
        solution,
        omega,
        linear,
        active_tolerance=_BINDING_TOLERANCE,
    )
    _require_projection_solution(solution, context="indicator GMS QLR")
    difference = target_matrix - solution.primal
    return jnp.einsum("bi,ij,bj->b", difference, omega, difference)


def indicator_gms(
    g: ArrayLike,
    alpha: float = 0.05,
    bandwidth: int | None = None,
    n_mc: int = 5000,
    seed: int = 0,
    kappa_n: float | None = None,
    eta: float = 0.0,
    statistic: str = "qlr",
) -> GmsResult:
    """Uncorrected indicator-GMS procedure for ``H0: E[g_t] ≥ 0``.

    Instead
    of taking the χ̄² weights under *all* inequalities binding (the least-
    favourable null — conservative and low-power when some are slack), GMS uses
    the data to select which moments are near-binding and simulates the critical
    value only for those, adjusting the slack ones out. The default combines an
    adjusted-QLR statistic with the ``φ⁽¹⁾`` indicator selection function, but
    ``eta=0`` omits the Andrews--Barwick Table-I RMS size correction. Therefore
    neither the default rejection nor p-value is represented as the exact
    Andrews--Barwick RMS calibration.

    Statistic (``statistic="qlr"``, recommended):
        ``S = min_{t ≥ 0} (z - t)ᵀ Ω̂⁻¹ (z - t)`` on studentized moments
        ``z = √n·ḡ/σ`` — correlation-aware, coinciding with the Wolak IU
        projection. The simulated indicator-selection critical value is
        uncorrected unless the caller supplies ``eta``; even then this function
        does not certify Table-I calibration. ``statistic="mmm"`` selects the
        diagonal-only ``Σ min(z_j,0)²`` form instead.

    Selection: moment ``j`` is *slack* (dropped from the binding set) when

        ξ_j = √n · ḡ_j / (σ_j · κ_n) > 1,

    with ``κ_n = √(0.3 · ln n)`` (Andrews-Barwick recommended tuning).  The
    critical value is the ``1 - alpha`` quantile of the statistic over
    ``N(0, Ω̂)`` draws with the *selected* (near-binding) coordinates recentered
    at the boundary and the slack coordinates pushed far positive (so they never
    bind — the GMS φ⁽¹⁾ recentering), plus the optional RMS size-correction
    ``η`` (default 0). A nonzero caller-supplied value is recorded, but this
    function cannot establish that it matches the appropriate Table-I setting
    and consequently never labels the result Table-I calibrated.

    Args:
        g: Per-period moment process with shape ``(T, k)``, already
            clustered to a manageable ``k``.
        alpha: Nominal size.
        bandwidth: HAC truncation lag; ``None`` → data-driven.
        n_mc: Monte-Carlo draws for the critical value.
        seed: RNG seed.
        kappa_n: Selection tuning; ``None`` → ``√(0.3·ln T)``.
        eta: User-supplied additive critical-value correction. Zero means the
            indicator-GMS calibration is explicitly uncorrected.
        statistic: ``"qlr"`` (recommended, correlation-aware) or ``"mmm"``.

    Returns:
        :class:`GmsResult`.

    """
    setup = _validate_gms_setup(
        g,
        alpha=alpha,
        bandwidth=bandwidth,
        n_mc=n_mc,
        kappa_n=kappa_n,
        eta=eta,
        statistic=statistic,
    )
    geometry = _gms_moment_geometry(
        setup.moments,
        bandwidth=setup.bandwidth,
        n_moments=setup.n_moments,
    )

    # Work in the raw √T·ḡ scale with the covariance⁻¹ metric,
    # identical to
    # :func:`wolak_statistic` / :func:`chi_bar_squared_weights_mc`, so that GMS
    # with no selection (``kappa_n → ∞``) reproduces the least-favourable Wolak
    # test exactly.  (Studentizing to the correlation metric first breaks this
    # equivalence when ``covariance`` is ill-conditioned — the moment covariance has
    # near-collinear directions whose pinv truncation differs between metrics.)
    root_t = math.sqrt(setup.n_observations)

    slack, n_selected = _gms_moment_selection(root_t, geometry, setup.kappa_n)
    standardized = _gms_null_draws(
        geometry,
        slack,
        n_moments=setup.n_moments,
        n_mc=setup.n_mc,
        seed=seed,
    )
    stat, sims = _gms_observed_and_null(
        statistic,
        root_t=root_t,
        geometry=geometry,
        standardized=standardized,
    )

    crit, pvalue, reject = _finite_mc_calibration(
        sims,
        stat,
        eta=setup.eta,
        alpha=setup.alpha,
    )
    calibration = (
        "indicator_gms_uncorrected"
        if math.isclose(setup.eta, 0.0, abs_tol=1e-12)
        else "indicator_gms_user_eta"
    )
    return GmsResult(
        stat=stat,
        crit=crit,
        reject=reject,
        pvalue=pvalue,
        n_selected=n_selected,
        kappa_n=setup.kappa_n,
        eta=setup.eta,
        calibration=calibration,
        rms_table_i_calibrated=False,
    )


def andrews_barwick_gms(
    g: ArrayLike,
    alpha: float = 0.05,
    bandwidth: int | None = None,
    n_mc: int = 5000,
    seed: int = 0,
    kappa_n: float | None = None,
    eta: float = 0.0,
    statistic: str = "qlr",
) -> GmsResult:
    """Compatibility alias for :func:`indicator_gms`.

    The historical name overstated the default ``eta=0`` implementation as the
    exact Andrews--Barwick RMS procedure. The returned metadata now identifies
    it as uncorrected indicator GMS (or as using an unverified caller-supplied
    correction), and never as Table-I calibrated.

    """
    return indicator_gms(
        g,
        alpha=alpha,
        bandwidth=bandwidth,
        n_mc=n_mc,
        seed=seed,
        kappa_n=kappa_n,
        eta=eta,
        statistic=statistic,
    )

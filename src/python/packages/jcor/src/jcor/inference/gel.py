"""Generalized Empirical Likelihood over-identification test.

The public :func:`gel_test` entry point; the carrier and inner-sup Newton
iteration it drives live in :mod:`jcor.inference._gel_carrier`.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

# Eager arrays and nested solver reports; t56.5 owns boundary behaviour.
import jax.numpy as jnp
import optimistix as ox

from jcor.core.typing import Array, ArrayLike, Float  # noqa: TC001
from jcor.inference._common import (
    _chi2_survival,
    _coerce_affine_jacobian,
    _validate_moment_panel,
    _validate_positive_integer,
    as_index,
)
from jcor.inference._gel_carrier import (
    _GEL_RCOND,
    _gel_identified_subspace,
    _gel_inner_sup,
    _gel_profile_objective_kernel,
    _require_host_gel,
)

if TYPE_CHECKING:
    import jax


class GelResult(NamedTuple):
    """Return type of :func:`gel_test`.

    Attributes:
        stat: GEL over-identification statistic (LR/LM/Wald-type per ``kind``).
        dof: Degrees of freedom: the identified moment rank (all parameters are
            fixed in this entry point).
        pvalue: Upper-tail χ²(dof) p-value.
        criterion: The GEL carrier used (``"cue"``, ``"el"`` or ``"et"``).
        kind: Statistic form (``"lr"``, ``"lm"`` or ``"wald"``).
        lam: Estimated Lagrange multiplier ``λ̂`` (inner-sup solution), shape
            ``(m,)``.
        converged: Whether the inner-sup Newton iteration converged.
        reference_valid: Whether the caller explicitly licensed the iid
            observation-level chi-square reference.

    """

    stat: float
    dof: int
    pvalue: float
    criterion: str
    kind: str
    lam: Float[Array, " m"]
    converged: bool
    reference_valid: bool


class LinearGelResult(NamedTuple):
    """Profiled GEL result for moments affine in their parameters.

    ``converged`` is true only when both the outer parameter minimization and
    the final inner multiplier supremum converge. Nonconverged fits expose
    ``NaN`` for the statistic and p-value so callers cannot consume them as
    valid chi-square tests.

    Attributes:
        stat: Profiled GEL likelihood-ratio over-identification statistic.
        dof: Over-identifying degrees of freedom ``r - k``, where ``r`` is the
            identified stochastic moment rank.
        pvalue: Upper-tail chi-square p-value.
        criterion: GEL carrier (``"cue"``, ``"el"`` or ``"et"``).
        kind: Statistic form; currently always ``"lr"``.
        lam: Inner-sup multiplier at the profiled parameter, shape ``(m,)``.
        params: Profiled parameter estimate, shape ``(k,)``.
        converged: Whether every layer converged — outer minimization, inner
            supremum, and the identified-subspace stability check.
        inner_converged: Whether the final inner multiplier solve converged.
        outer_converged: Whether the outer parameter minimization converged.
        subspace_stable: Whether the identified moment rank at the profiled
            ``theta`` still matches the rank frozen at the starting value. The
            profile is defined over a *fixed* set of identified directions; if
            that set moves along the profile path the retained rank — and hence
            ``dof`` — is a property of where the optimizer stopped rather than of
            the model, and no chi-square reference is licensed.
        reference_valid: Whether both solvers converged and the caller
            explicitly licensed the iid observation-level chi-square reference.

    """

    stat: float
    dof: int
    pvalue: float
    criterion: str
    kind: str
    lam: Float[Array, " m"]
    params: Float[Array, " k"]
    converged: bool
    inner_converged: bool
    outer_converged: bool
    subspace_stable: bool
    reference_valid: bool


def _validate_fixed_moment_setup(
    n_params: int,
    criterion: str,
    kind: str,
    max_iter: int,
    *,
    assume_iid: bool,
) -> int:
    """Validate the fixed-moment carrier, statistic and solver controls."""
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    if not isinstance(assume_iid, bool):
        message = "assume_iid must be a boolean."
        raise TypeError(message)
    if criterion not in ("cue", "el", "et"):
        message = f"unknown GEL criterion {criterion!r}"
        raise ValueError(message)
    if kind not in ("lr", "lm", "wald"):
        message = f"unknown GEL statistic kind {kind!r}"
        raise ValueError(message)
    n_params_index = as_index(n_params)
    if n_params_index is None or n_params_index < 0:
        message = "n_params must be a nonnegative integer."
        raise ValueError(message)
    if n_params_index > 0:
        message = (
            "gel_test receives pre-formed moments and cannot subtract externally "
            "estimated parameter dimensions; use linear_gel_test for affine "
            "moments."
        )
        raise ValueError(message)
    return max_iter


def _gel_moment_quadratic(
    g: Float[Array, "t m"],
    n_observations: int,
) -> float:
    """Score/Wald quadratic ``T·ḡ'Ω̂⁻¹ḡ`` in the carrier's moment metric."""
    omega = (
        g.T @ g
    ) / n_observations  # empirical second-moment (GEL variance estimate)
    # Use the exact row-space cutoff that licensed `effective_rank` and `dof`.
    # A looser pseudoinverse would reintroduce discarded near-null directions
    # into LM/Wald while still reporting the reduced chi-square dimension.
    omega_inv = jnp.linalg.pinv(omega, rtol=_GEL_RCOND)
    gbar = g.mean(axis=0)
    return float(n_observations * gbar @ omega_inv @ gbar)


def _validate_profile_setup(
    moments0: Float[Array, "t m"],
    derivative_panel: Float[Array, "t m k"],
    criterion: str,
    kind: str,
    tol: float,
) -> int:
    """Validate identification, carrier, statistic, and solver controls."""
    m = moments0.shape[1]
    n_params = derivative_panel.shape[2]
    if m <= n_params:
        message = (
            "profiled GEL requires over-identification: the number of moments "
            f"m={m} must exceed the number of parameters k={n_params}."
        )
        raise ValueError(message)
    if jnp.linalg.matrix_rank(derivative_panel.mean(axis=0)) < n_params:
        message = "the mean moment jacobian does not identify every parameter."
        raise ValueError(message)
    if criterion not in ("cue", "el", "et"):
        message = f"unknown GEL criterion {criterion!r}"
        raise ValueError(message)
    if kind != "lr":
        message = "linear_gel_test currently supports only kind='lr'."
        raise ValueError(message)
    if not math.isfinite(tol) or tol <= 0.0:
        message = "tol must be finite and positive."
        raise ValueError(message)
    return n_params


def gel_test(
    g: ArrayLike,
    n_params: int = 0,
    criterion: str = "cue",
    kind: str = "lr",
    max_iter: int = 100,
    *,
    assume_iid: bool = False,
) -> GelResult:
    """Generalized Empirical Likelihood over-identification test of ``E[g]=0``.

    GEL profiles a non-parametric likelihood supported on the data subject to
    moment constraints via the saddle-point

        min_θ max_λ (1/n_observations) Σ_t [rho(λ'g_t(θ)) - rho(0)],

    This entry point tests a fixed, pre-formed moment vector and solves only the
    inner supremum over ``λ``. It cannot license subtracting parameters
    estimated outside this criterion: ``n_params > 0`` fails closed. Use
    :func:`linear_gel_test` to estimate parameters inside an affine GEL
    criterion.

    The carrier uses the observation-level second moment ``g'g/T``, not a
    long-run covariance. Its chi-square reference is therefore licensed only
    for iid (or otherwise serially uncorrelated) moment contributions. The
    caller must acknowledge that condition with ``assume_iid=True``; otherwise
    the objective statistic remains available descriptively but ``pvalue`` is
    ``NaN`` and ``reference_valid`` is false.

    Three asymptotically-equivalent fixed-moment statistic forms (all
    ``~χ²(r)``), where ``r`` is the identified moment rank retained by the
    carrier solver:

    * ``lr`` — likelihood-ratio ``2·T·P̂(λ̂)``. This is the only form
      that reads the
      inner-sup objective, so the only one that differs across carriers and the
      one to use as the robustness statistic.
    * ``lm``   — score/Lagrange-multiplier: the score at ``λ = 0`` is ``-ḡ`` with
      metric ``Ω̂⁻¹``, giving ``n_observations·ḡ'Ω̂⁻¹ḡ``.
    * ``wald`` — ``n_observations·ḡ'Ω̂⁻¹ḡ`` in the moment metric, equivalent to
      CUE-GMM Wald at bandwidth 0. With the moment vector pre-formed and no
      structural parameter
      profiled inside, the GEL ``lm`` and ``wald`` forms **coincide exactly**
      (both are the score/quadratic in ``ḡ`` under the same ``Ω̂⁻¹`` metric);
      they are kept as distinct labels for API parity with the LR form.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        n_params: Must be zero. Kept only to turn historical pre-formed calls
            that subtracted externally estimated parameters into explicit
            errors rather than silently miscalibrated tests.
        criterion: GEL carrier — ``"cue"`` (continuous-updating, recommended and
            most stable), ``"el"`` (empirical likelihood) or ``"et"``
            (exponential tilting).
        kind: Statistic form — ``"lr"``, ``"lm"`` or ``"wald"``.
        max_iter: Newton iterations for the inner sup.
        assume_iid: Explicitly acknowledge the iid/serially-uncorrelated moment
            assumption required by the observation-level GEL reference.

    Returns:
        :class:`GelResult`.

    """
    _require_host_gel(g, name="gel_test")
    g, n_observations, _m = _validate_moment_panel(g)
    max_iter = _validate_fixed_moment_setup(
        n_params,
        criterion,
        kind,
        max_iter,
        assume_iid=assume_iid,
    )

    lam, obj, converged, effective_rank = _gel_inner_sup(
        g, criterion, max_iter=max_iter
    )
    if effective_rank < 1:
        message = "GEL requires at least one identified stochastic moment direction."
        raise ValueError(message)
    quadratic = _gel_moment_quadratic(g, n_observations)

    dof = effective_rank
    if not converged:
        # The inner sup did not genuinely converge (λ pinned at the trust-region
        # cap / non-finite objective on an ill-posed moment set).  The resulting
        # statistic is garbage; report NaN so callers cannot consume it as a
        # valid test.  The honest `converged=False` flag is preserved.
        return GelResult(
            stat=float("nan"),
            dof=dof,
            pvalue=float("nan"),
            criterion=criterion,
            kind=kind,
            lam=lam,
            converged=False,
            reference_valid=False,
        )

    if kind == "lr":
        stat = float(2.0 * n_observations * obj)
    elif kind == "wald":
        stat = quadratic
    else:  # lm: score at λ=0 is -ḡ, metric Ω⁻¹
        stat = quadratic
    stat = max(stat, 0.0)
    pvalue = _chi2_survival(stat, dof) if assume_iid else float("nan")
    return GelResult(
        stat=stat,
        dof=dof,
        pvalue=pvalue,
        criterion=criterion,
        kind=kind,
        lam=lam,
        converged=converged,
        reference_valid=bool(assume_iid),
    )


class _LinearGelSetup(NamedTuple):
    """Validated inputs and solver controls for :func:`linear_gel_test`.

    Attributes:
        moments0: Validated moment panel at ``theta=0``, shape ``(t, m)``.
        derivative_panel: Coerced affine derivative, shape ``(t, m, k)``.
        n_observations: Panel length ``t``.
        n_params: Parameter count ``k``.
        max_iter: Validated inner-solve iteration cap.
        outer_max_iter: Validated outer-minimization iteration cap.

    """

    moments0: Float[Array, "t m"]
    derivative_panel: Float[Array, "t m k"]
    n_observations: int
    n_params: int
    max_iter: int
    outer_max_iter: int


class _ProfileStart(NamedTuple):
    """Outer starting parameter and the identified subspace frozen there.

    Attributes:
        theta0: Least-squares starting parameter, shape ``(k,)``.
        basis: Orthonormal ``(m, m)`` rotation of ``Ω(theta0)``.
        keep: ``(m,)`` 0/1 retention mask of identified directions.
        rank: Number of retained stochastic moment directions.

    """

    theta0: Float[Array, " k"]
    basis: Float[Array, "m m"]
    keep: Float[Array, " m"]
    rank: int


class _ProfileScore(NamedTuple):
    """Inner-sup score of the profiled fit in the frozen subspace.

    Attributes:
        lam: Inner-sup multiplier at the profiled parameter, shape ``(m,)``.
        objective: Inner-sup objective value at the profiled parameter.
        inner_converged: Whether the final inner multiplier solve converged.
        effective_rank: Retained stochastic moment rank.
        subspace_stable: Whether the identified rank at the profiled parameter
            still matches the rank frozen at the starting value.

    """

    lam: Float[Array, " m"]
    objective: float
    inner_converged: bool
    effective_rank: int
    subspace_stable: bool


def _prepare_linear_gel(
    g0: ArrayLike,
    jacobian: ArrayLike,
    criterion: str,
    kind: str,
    max_iter: int,
    outer_max_iter: int,
    tol: float,
    *,
    assume_iid: bool,
) -> _LinearGelSetup:
    """Validate the profiled entry point's panels, carrier and solver controls."""
    _require_host_gel(g0, name="linear_gel_test")
    _require_host_gel(jacobian, name="linear_gel_test")
    moments0, n_observations, _m = _validate_moment_panel(g0, name="g0")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    outer_max_iter = _validate_positive_integer(
        outer_max_iter,
        name="outer_max_iter",
    )
    if not isinstance(assume_iid, bool):
        message = "assume_iid must be a boolean."
        raise TypeError(message)
    derivative_panel = _coerce_affine_jacobian(moments0, jacobian)
    # Kept at the call site, not in the shared helper: the parameter-count floor
    # is this entry point's contract, and `_validate_profile_setup` cannot catch
    # it (`m <= 0` and `rank < 0` are both false for an empty parameter axis).
    if derivative_panel.shape[2] < 1:
        message = "linear_gel_test requires at least one parameter."
        raise ValueError(message)
    n_params = _validate_profile_setup(
        moments0,
        derivative_panel,
        criterion,
        kind,
        tol,
    )
    return _LinearGelSetup(
        moments0=moments0,
        derivative_panel=derivative_panel,
        n_observations=n_observations,
        n_params=n_params,
        max_iter=max_iter,
        outer_max_iter=outer_max_iter,
    )


def _affine_moments(
    setup: _LinearGelSetup,
    theta: Float[Array, " k"],
) -> Float[Array, "t m"]:
    """Evaluate the affine moment panel ``g_t(theta) = g0_t + D_t @ theta``."""
    return setup.moments0 + jnp.einsum("tmk,k->tm", setup.derivative_panel, theta)


def _gel_profile_start(setup: _LinearGelSetup) -> _ProfileStart:
    """Least-squares starting parameter and the subspace frozen there.

    The identified subspace is decided once, at the starting parameter. Deciding
    it inside the objective instead would make the profile discontinuous in
    theta — see :func:`_gel_profile_objective_kernel` — and would put the
    non-differentiable ``eigh`` on the outer solver's tape.
    """
    dbar = setup.derivative_panel.mean(axis=0)
    g0bar = setup.moments0.mean(axis=0)
    theta0 = jnp.linalg.lstsq(dbar, -g0bar)[0]
    basis, keep, reference_rank = _gel_identified_subspace(
        _affine_moments(setup, theta0), _GEL_RCOND
    )
    return _ProfileStart(theta0=theta0, basis=basis, keep=keep, rank=reference_rank)


def _minimise_profile(
    setup: _LinearGelSetup,
    start: _ProfileStart,
    criterion: str,
    tol: float,
) -> tuple[Float[Array, " k"], bool]:
    """Minimize the profiled GEL objective over ``theta``.

    Returns:
        Tuple ``(theta, outer_converged)`` with ``theta`` the profiled parameter
        and ``outer_converged`` true only for a successful solve at a finite
        profiled objective.

    """
    max_iter = setup.max_iter
    panel_arg = jnp.asarray(setup.derivative_panel)
    moments_arg = jnp.asarray(setup.moments0)
    # The retention *decision* is made in float64 above; the basis itself is
    # carried at the working precision of the carrier loop.
    basis_arg = jnp.asarray(start.basis, dtype=moments_arg.dtype)
    keep_arg = jnp.asarray(start.keep, dtype=moments_arg.dtype)

    def profile_objective(
        theta: jax.Array,
        args: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        moments, derivative, subspace, mask = args
        return _gel_profile_objective_kernel(
            moments,
            derivative,
            theta,
            subspace,
            mask,
            criterion,
            max_iter,
        )

    solver_args = (moments_arg, panel_arg, basis_arg, keep_arg)
    fit = ox.minimise(
        profile_objective,
        ox.BFGS(rtol=tol, atol=tol),
        jnp.asarray(start.theta0),
        args=solver_args,
        max_steps=setup.outer_max_iter,
        throw=False,
    )
    theta = jnp.asarray(fit.value)
    profile_value = float(profile_objective(jnp.asarray(theta), solver_args))
    outer_converged = bool(
        fit.result == ox.RESULTS.successful and math.isfinite(profile_value)
    )
    return theta, outer_converged


def _score_profile(
    setup: _LinearGelSetup,
    start: _ProfileStart,
    theta: Float[Array, " k"],
    criterion: str,
) -> _ProfileScore:
    """Score the profiled fit in the subspace the profile was minimized over.

    Scoring in the *same* frozen subspace keeps ``stat`` and ``dof`` a
    description of one model rather than two. The retained rank at ``theta`` is
    checked separately: if it moved along the path, ``dof`` is a property of
    where the optimizer halted rather than of the model, and no chi-square
    reference is licensed.

    Returns:
        :class:`_ProfileScore`.

    """
    fitted = _affine_moments(setup, theta)
    lam, objective, inner_converged, effective_rank = _gel_inner_sup(
        fitted,
        criterion,
        max_iter=setup.max_iter,
        subspace=(start.basis, start.keep, start.rank),
    )
    _, _, rank_at_theta = _gel_identified_subspace(fitted, _GEL_RCOND)
    return _ProfileScore(
        lam=lam,
        objective=objective,
        inner_converged=bool(inner_converged),
        effective_rank=effective_rank,
        subspace_stable=bool(rank_at_theta == start.rank),
    )


def linear_gel_test(
    g0: ArrayLike,
    jacobian: ArrayLike,
    criterion: str = "cue",
    kind: str = "lr",
    *,
    max_iter: int = 100,
    outer_max_iter: int = 500,
    tol: float = 1e-8,
    assume_iid: bool = False,
) -> LinearGelResult:
    """Profile an affine parameter inside a GEL over-identification test.

    The supplied moment model is ``g_t(theta) = g0_t + D_t @ theta``. For each
    trial parameter the carrier's multiplier is maximized by
    :func:`_gel_inner_sup`; an outer minimization then profiles ``theta`` using
    the same criterion. The likelihood-ratio statistic is
    ``2*T*min_theta max_lambda P_T(theta, lambda)`` and has ``r-k``
    over-identifying degrees of freedom for an interior, identified solution,
    where ``r`` is the retained stochastic moment rank.

    This is observation-level GEL and does not estimate a long-run covariance.
    Its chi-square p-value therefore requires iid (or serially uncorrelated)
    moment contributions. Unless the caller explicitly passes
    ``assume_iid=True``, the profiled LR objective is returned only as a
    descriptive statistic and the p-value is withheld.

    Only the LR form is exposed here. A profiled GEL LM or Wald statistic
    requires an estimator-specific sandwich construction not available from a
    pre-formed moment panel, so those requests fail closed rather than reuse the
    fixed-moment quadratic.

    Args:
        g0: Moment contributions at ``theta=0``, shape
            ``(n_observations, m)``.
        jacobian: Affine derivative ``D``, shape ``(m, k)`` or
            ``(n_observations, m, k)``.
        criterion: GEL carrier — ``"cue"``, ``"el"`` or ``"et"``.
        kind: Statistic form. Only ``"lr"`` is supported.
        max_iter: Maximum iterations for each inner multiplier solve.
        outer_max_iter: Maximum outer parameter-minimization iterations.
        tol: Absolute parameter and objective tolerance for the outer solve.
        assume_iid: Explicitly acknowledge the iid/serially-uncorrelated moment
            assumption required by the observation-level GEL reference.

    Returns:
        :class:`LinearGelResult`. Failed inner or outer optimization produces
        ``converged=False`` and ``NaN`` inferential values.

    """
    setup = _prepare_linear_gel(
        g0,
        jacobian,
        criterion,
        kind,
        max_iter,
        outer_max_iter,
        tol,
        assume_iid=assume_iid,
    )
    start = _gel_profile_start(setup)
    theta, outer_converged = _minimise_profile(setup, start, criterion, tol)
    score = _score_profile(setup, start, theta, criterion)
    converged = bool(
        outer_converged
        and score.inner_converged
        and score.subspace_stable
        and math.isfinite(score.objective)
    )
    dof = score.effective_rank - setup.n_params
    if dof < 1:
        message = (
            "profiled GEL requires the identified moment rank "
            f"r={score.effective_rank} to exceed the parameter count "
            f"k={setup.n_params}."
        )
        raise ValueError(message)
    if not converged:
        return LinearGelResult(
            stat=float("nan"),
            dof=dof,
            pvalue=float("nan"),
            criterion=criterion,
            kind=kind,
            lam=score.lam,
            params=theta,
            converged=False,
            inner_converged=score.inner_converged,
            outer_converged=outer_converged,
            subspace_stable=score.subspace_stable,
            reference_valid=False,
        )

    stat = max(float(2.0 * setup.n_observations * score.objective), 0.0)
    pvalue = _chi2_survival(stat, dof) if assume_iid else float("nan")
    return LinearGelResult(
        stat=stat,
        dof=dof,
        pvalue=pvalue,
        criterion=criterion,
        kind=kind,
        lam=score.lam,
        params=theta,
        converged=True,
        inner_converged=True,
        outer_converged=True,
        subspace_stable=True,
        reference_valid=bool(assume_iid),
    )

"""GEL carrier ``rho`` and the Newton inner-sup solver over ``λ``.

Private machinery behind :mod:`jcor.inference.gel`; depends only on
:mod:`jcor.inference._common`.

Layout follows the house *traceable core + eager door* pattern
(``docs/numerical-host-boundaries.md``): :func:`_gel_rho_kernel`,
:func:`_gel_subspace_kernel` and :func:`_gel_inner_sup_kernel` are fixed-shape,
``jit``/``vmap``-safe and free of NumPy; :func:`_gel_identified_subspace` and
:func:`_gel_inner_sup` are the eager doors that own the float64 decision and the
NumPy report contract.

The carrier has no eager door. ``_gel_rho`` was one until the Newton solver
became traceable and left it with no caller outside the test suite; it was
deleted rather than kept as a NumPy-carrying surface nothing shipped. Call
:func:`_gel_rho_kernel` under a caller-owned ``jax_enable_x64=True`` for the
float64 evaluation it used to provide.
"""

# ruff: noqa: F722, F821, UP037  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp

from jcor.core.typing import Array, ArrayLike, Static  # noqa: TC001
from jcor.core.typing import Float as JaxFloat  # noqa: TC001

_GEL_RCOND = 1e-6
_GEL_LAM_CAP = 1e4
_GEL_LINE_SEARCH_STEPS = 12
_GEL_EL_FEASIBILITY_MARGIN = 1e-8
_GEL_EL_STEP_FRACTION = 0.9
# Scale-relative gradient test. A fixed absolute tolerance cannot be honoured in
# float32 — the attainable gradient norm floors near 1e-8, so the historical
# 1e-9 would never fire and every fit would report ``converged=False``. The
# default defers entirely to the dtype-derived floor; ``rtol`` only ever
# *loosens* it, since no caller can ask for tighter than the precision allows.
_GEL_NEWTON_RTOL = 0.0
type GelCriterion = Literal["cue", "et", "el"]


def _require_host_gel(value: object, *, name: str) -> None:
    """Reject tracing at the named GEL host boundary before NumPy transfer."""
    if isinstance(value, jax.core.Tracer):
        message = (
            f"{name} is a host-only GEL solver boundary: data-dependent rank "
            "selection and feasibility line search cannot run under JAX tracing"
        )
        raise RuntimeError(message)  # noqa: TRY004  # tracing is an execution mode


def _guard_outside_derivative(value: jax.Array, guarded: jax.Array) -> jax.Array:
    """Use a guarded primal value while differentiating the carrier coordinate.

    GEL feasibility belongs to the line search. This straight-through guard
    preserves the historical finite saturation values when a trial crosses a
    numerical guard, while keeping the analytic carrier derivative used to
    steer the iterate back toward its feasible set.
    """
    return jax.lax.stop_gradient(guarded) + (value - jax.lax.stop_gradient(value))


def _gel_scalar_rho(v: jax.Array, criterion: GelCriterion) -> jax.Array:
    """Define one scalar carrier; all derivatives are generated from this map.

    Conventions (Newey-Smith 2004): the GEL objective is
    ``P̂(λ) = (1/T) Σ_t [rho(λ'g_t) - rho(0)]`` **maximized** over ``λ``. Each
    carrier satisfies ``rho'(0) = rho''(0) = -1`` so all three share the same
    second-order (CUE/GMM) behaviour and only differ in higher-order (finite-
    sample) terms.

    * ``cue`` — quadratic ``rho(v) = -v - v²/2`` (continuous-updating GMM).
    * ``et``  — exponential tilting ``rho(v) = -e^{v} + 1`` (defined for all v).
    * ``el``  — empirical likelihood ``rho(v) = ln(1 - v)`` (needs ``v < 1``).
    """
    if criterion == "cue":
        return -v - 0.5 * v * v
    if criterion == "et":
        guarded = _guard_outside_derivative(v, jnp.clip(v, -50.0, 50.0))
        return -jnp.exp(guarded) + 1.0
    if criterion == "el":
        argument = 1.0 - v
        guarded = _guard_outside_derivative(argument, jnp.maximum(argument, 1e-12))
        return jnp.log(guarded)
    message = f"unknown GEL criterion {criterion!r}"
    raise ValueError(message)


@partial(jax.jit, static_argnames=("criterion",))
def _gel_rho_kernel(
    v: JaxFloat[Array, "t"],
    criterion: Static[GelCriterion],
) -> tuple[
    JaxFloat[Array, "t"],
    JaxFloat[Array, "t"],
    JaxFloat[Array, "t"],
]:
    """Return carrier values and autodiff first/second derivatives on graph."""

    def scalar(value: jax.Array) -> jax.Array:
        return _gel_scalar_rho(value, criterion)

    rho, rho1 = jax.vmap(jax.value_and_grad(scalar))(v)
    rho2 = jax.vmap(jax.grad(jax.grad(scalar)))(v)
    return rho, rho1, rho2


@partial(jax.jit, static_argnames=("rcond",))
def _gel_subspace_kernel(
    moments: JaxFloat[Array, "t m"],
    rcond: Static[float] = _GEL_RCOND,
) -> tuple[JaxFloat[Array, "m m"], JaxFloat[Array, "m"]]:
    """Return the eigenbasis of ``Ω`` and its identified-direction mask.

    ``Ω = g'g/n_observations`` is symmetrized before ``eigh`` so the basis is
    orthonormal to working precision. The mask keeps directions whose
    eigenvalue exceeds ``rcond · λ_max``; the returned basis is the **full**
    ``(m, m)`` rotation, so masking rather than slicing keeps every shape static
    and the whole construction ``vmap``-safe.
    """
    n_observations = moments.shape[0]
    omega = (moments.T @ moments) / n_observations
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (omega + omega.T))
    threshold = rcond * jnp.maximum(eigenvalues[-1], 1e-30)
    keep = (eigenvalues > threshold).astype(moments.dtype)
    return eigenvectors, keep


def _gel_identified_subspace(
    moments: ArrayLike,
    rcond: float = _GEL_RCOND,
) -> tuple[jax.Array, jax.Array, int]:
    """Eager door: decide the identified moment subspace.

    The retained rank propagates to ``dof`` and therefore to a chi-square
    reference, so the decision is worth one ``m x m`` ``eigh`` outside the
    ``T``-sized work.

    **Precision is the caller's.** This door previously coerced ``moments`` to
    float64 unconditionally, on the argument that a rank decision feeding a
    chi-square dof must be taken in float64 even when the carrier loop runs in
    float32. t65.1 removed that upcast: a library that silently re-rounds its
    caller's data is the inversion t56.7 deleted the x64 doors to fix, and the
    stage that needs float64 here (``paper3-empirical``) owns it explicitly via
    ``pipeline.precision.own_float64``. A float32 caller now gets a float32
    rank decision, which is the honest answer rather than a hidden promotion.

    Returns:
        Tuple ``(basis, keep, rank)`` with ``basis`` the ``(m, m)`` orthonormal
        rotation, ``keep`` its ``(m,)`` 0/1 retention mask, and ``rank`` the
        host-side integer count of identified directions.

    """
    basis, keep = _gel_subspace_kernel(jnp.asarray(moments), rcond)
    rank = int(jnp.sum(keep))
    return basis, keep, rank


def _gel_objective(
    reduced: jax.Array,
    beta: jax.Array,
    criterion: GelCriterion,
) -> jax.Array:
    """Carrier mean at ``beta``, with non-finite values mapped to ``-inf``."""
    rho, _, _ = _gel_rho_kernel(reduced @ beta, criterion)
    value = jnp.mean(rho)
    return jnp.where(jnp.isfinite(value), value, -jnp.inf)


def _gel_trust_region_step(
    beta: jax.Array,
    direction: jax.Array,
    lam_cap: jax.Array,
) -> jax.Array:
    """Largest ``alpha >= 0`` with ``‖beta + alpha·direction‖ <= lam_cap``.

    Closed form: the positive root of the quadratic in ``alpha``. Replaces the
    historical "halve until the cap is satisfied" search, so a diverging carrier
    reaches the trust-region boundary in one step instead of dozens.
    """
    beta_dot = beta @ direction
    direction_sq = jnp.maximum(direction @ direction, jnp.finfo(beta.dtype).tiny)
    slack = lam_cap * lam_cap - beta @ beta
    discriminant = jnp.maximum(beta_dot * beta_dot + direction_sq * slack, 0.0)
    return jnp.maximum((-beta_dot + jnp.sqrt(discriminant)) / direction_sq, 0.0)


def _gel_feasible_step(
    reduced: jax.Array,
    beta: jax.Array,
    direction: jax.Array,
    criterion: GelCriterion,
) -> jax.Array:
    """Largest EL-feasible step, in closed form.

    Empirical likelihood needs ``v_t = λ'g_t < 1`` for every observation. With
    ``s = reduced @ direction`` the binding constraint is
    ``min over {t : s_t > 0} of (1 - δ - v_t)/s_t``; a single masked reduction
    replaces the historical 80-step feasibility halving. ``cue`` and ``et`` are
    defined on all of ``R`` and take the full step.
    """
    if criterion != "el":
        return jnp.asarray(jnp.inf, dtype=beta.dtype)
    values = reduced @ beta
    slope = reduced @ direction
    room = jnp.where(
        slope > 0.0,
        (1.0 - _GEL_EL_FEASIBILITY_MARGIN - values)
        / jnp.where(slope > 0.0, slope, 1.0),
        jnp.inf,
    )
    return _GEL_EL_STEP_FRACTION * jnp.min(room)


def _gel_cue_supremum(
    reduced: jax.Array,
    frozen: jax.Array,
    cap: jax.Array,
    criterion: GelCriterion,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Closed-form CUE supremum ``β* = -Ω⁺ḡ``, capped at the trust region."""
    n_observations = reduced.shape[0]
    omega = (reduced.T @ reduced) / n_observations
    mean_moment = jnp.mean(reduced, axis=0)
    beta = jnp.linalg.solve(omega + frozen, -mean_moment)
    beta = jnp.where(jnp.isfinite(beta), beta, 0.0)
    norm = jnp.linalg.norm(beta)
    # Preserve the trust-region contract: report the capped multiplier and
    # an honest non-convergence rather than an unbounded one.
    beta = jnp.where(norm > cap, beta * (cap / jnp.maximum(norm, 1e-30)), beta)
    objective = _gel_objective(reduced, beta, criterion)
    converged = jnp.isfinite(objective) & (norm < cap)
    return beta, objective, converged


def _gel_gradient_tolerance(
    reduced: jax.Array,
    keep: jax.Array,
    rtol: float,
    atol: float,
) -> jax.Array:
    """Scale-relative gradient tolerance for the Newton inner-sup test.

    Scale the gradient test by the *moment* magnitude, not by ``‖ḡ‖``: under the
    null the mean moment vanishes, so a ``‖ḡ‖``-relative tolerance would collapse
    to zero and no fit would ever converge. ``sqrt(trace(Ω_r)/r)`` is the RMS
    size of a retained moment coordinate and stays O(1) there.

    The floor is ``sqrt(eps)`` of that scale — the classical limit, since an
    objective evaluated to relative accuracy ``eps`` pins its stationary point
    only to ``sqrt(eps)``. Anything tighter is unreachable: the float64 Newton
    iteration plateaus near 1e-11 and would report a spurious non-convergence
    on fits whose objective is correct to every printed digit.

    It costs nothing statistically. At a concave optimum the objective is
    second-order insensitive to the multiplier, so a residual gradient ``τ``
    perturbs the reported statistic ``2T·P`` by only ``O(T·τ²/|H|)`` — ~1e-13
    in float64 and ~1e-4 in float32, against a statistic of order one.
    """
    n_observations = reduced.shape[0]
    moment_scale = jnp.sqrt(
        jnp.sum(reduced * reduced) / (n_observations * jnp.maximum(jnp.sum(keep), 1.0))
    )
    resolution = jnp.sqrt(jnp.finfo(reduced.dtype).eps)
    return jnp.maximum(rtol, resolution) * moment_scale + atol


def _gel_newton_direction(
    reduced: jax.Array,
    beta: jax.Array,
    frozen: jax.Array,
    criterion: GelCriterion,
) -> tuple[jax.Array, jax.Array]:
    """Return the carrier gradient at ``beta`` and its damped Newton direction."""
    n_observations, n_moments = reduced.shape
    dtype = reduced.dtype
    _, rho1, rho2 = _gel_rho_kernel(reduced @ beta, criterion)
    gradient = jnp.mean(rho1[:, None] * reduced, axis=0)

    hessian = (reduced * rho2[:, None]).T @ reduced / n_observations
    # The Hessian is symmetric negative definite on the identified subspace;
    # damping subtracts a positive multiple of the identity to steepen it.
    ridge = jnp.sqrt(jnp.finfo(dtype).eps) * jnp.abs(jnp.trace(hessian)) / n_moments
    system = hessian - ridge * jnp.eye(n_moments, dtype=dtype) - frozen
    direction = -jnp.linalg.solve(system, gradient)
    return gradient, jnp.where(jnp.isfinite(direction), direction, 0.0)


def _gel_newton_line_search(
    reduced: jax.Array,
    beta: jax.Array,
    direction: jax.Array,
    base: jax.Array,
    alpha0: jax.Array,
    criterion: GelCriterion,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Backtrack a carrier step, retaining the first improving candidate."""

    def backtrack(
        _step: jax.Array,
        search: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        candidate, candidate_value, alpha = search
        trial = beta + alpha * direction
        trial_value = _gel_objective(reduced, trial, criterion)
        accepted = (alpha > 0.0) & jnp.isfinite(trial_value) & (trial_value >= base)
        take = accepted & ~jnp.isfinite(candidate_value)
        return (
            jnp.where(take, trial, candidate),
            jnp.where(take, trial_value, candidate_value),
            alpha * 0.5,
        )

    candidate, candidate_value, _ = jax.lax.fori_loop(
        0,
        _GEL_LINE_SEARCH_STEPS,
        backtrack,
        (beta, jnp.asarray(-jnp.inf, dtype=dtype), alpha0),
    )
    return candidate, candidate_value


def _gel_newton_body(
    reduced: jax.Array,
    beta: jax.Array,
    criterion: GelCriterion,
    tolerance: jax.Array,
    cap: jax.Array,
    frozen: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate one Newton iteration, including its feasible line search."""
    gradient, direction = _gel_newton_direction(reduced, beta, frozen, criterion)
    converged = jnp.linalg.norm(gradient) < tolerance
    base = _gel_objective(reduced, beta, criterion)
    alpha0 = jnp.minimum(
        jnp.asarray(1.0, dtype=reduced.dtype),
        jnp.minimum(
            _gel_trust_region_step(beta, direction, cap),
            _gel_feasible_step(reduced, beta, direction, criterion),
        ),
    )
    candidate, candidate_value = _gel_newton_line_search(
        reduced,
        beta,
        direction,
        base,
        alpha0,
        criterion,
        reduced.dtype,
    )
    improved = jnp.isfinite(candidate_value)
    return jnp.where(converged, beta, candidate), converged, ~converged & ~improved


@partial(
    jax.jit,
    static_argnames=("criterion", "max_iter", "lam_cap", "rtol", "atol"),
)
def _gel_inner_sup_kernel(
    reduced: JaxFloat[Array, "t m"],
    keep: JaxFloat[Array, "m"],
    criterion: Static[GelCriterion],
    max_iter: Static[int] = 200,
    lam_cap: Static[float] = _GEL_LAM_CAP,
    rtol: Static[float] = _GEL_NEWTON_RTOL,
    atol: Static[float] = 0.0,
) -> tuple[JaxFloat[Array, "m"], JaxFloat[Array, ""], jax.Array]:
    """Traceable inner supremum over the reduced multiplier ``beta``.

    ``reduced`` is ``g @ basis`` with dropped columns already zeroed, so every
    shape is ``m``-wide and static regardless of the identified rank.

    ``cue`` is solved in closed form: ``rho(v) = -v - v²/2`` makes the objective
    exactly quadratic, ``P(β) = -ḡ'β - β'Ωβ/2``, so the supremum is the single
    linear solve ``β* = -Ω⁺ḡ`` and no iteration is required. ``el`` and ``et``
    run a damped Newton iteration with a closed-form feasible/trust-region step
    and a short Armijo backtrack.

    ``converged`` is ``True`` only when the gradient genuinely vanishes inside
    the cap — never when ``β`` is pinned at the trust-region boundary or the
    objective is non-finite. That is the honest signal that the inner sup is
    ill-posed on this moment set, and callers key their ``NaN`` reporting off it.

    Returns:
        Tuple ``(beta, objective, converged)`` in the reduced coordinates.

    """
    n_moments = reduced.shape[1]
    dtype = reduced.dtype
    cap = jnp.asarray(lam_cap, dtype=dtype)
    # Dropped directions carry an exactly-zero column, hence a zero gradient and
    # a zero Hessian row. Freezing them with a unit diagonal keeps the Newton
    # system nonsingular and pins their step at exactly zero; a uniform tiny
    # ridge cannot, and underflows to a fully singular solve in float32.
    frozen = jnp.diag(1.0 - keep).astype(dtype)

    if criterion == "cue":
        return _gel_cue_supremum(reduced, frozen, cap, criterion)

    # See `_gel_gradient_tolerance` for why the test is relative to the moment
    # scale and floored at `sqrt(eps)` rather than at a fixed absolute value.
    tolerance = _gel_gradient_tolerance(reduced, keep, rtol, atol)

    def newton_body(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        beta, _converged, _stalled, iteration = state
        beta, converged, stalled = _gel_newton_body(
            reduced,
            beta,
            criterion,
            tolerance,
            cap,
            frozen,
        )
        return beta, converged, stalled, iteration + 1

    def newton_cond(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        _beta, converged, stalled, iteration = state
        return ~converged & ~stalled & (iteration < max_iter)

    beta, converged, _stalled, _iteration = jax.lax.while_loop(
        newton_cond,
        newton_body,
        (
            jnp.zeros((n_moments,), dtype=dtype),
            jnp.zeros((), dtype=bool),  # converged
            jnp.zeros((), dtype=bool),  # line search stalled
            jnp.zeros((), dtype=int),
        ),
    )
    objective = _gel_objective(reduced, beta, criterion)
    return beta, objective, converged & (jnp.linalg.norm(beta) < cap)


def _gel_inner_sup(
    g: ArrayLike,
    criterion: str,
    max_iter: int = 200,
    tol: float = 0.0,
    rcond: float = _GEL_RCOND,
    lam_cap: float = _GEL_LAM_CAP,
    subspace: tuple[jax.Array, jax.Array, int] | None = None,
) -> tuple[JaxFloat[Array, "m"], float, bool, int]:
    """Solve the regularized GEL inner supremum (eager float64 door).

    The inner problem is concave in ``λ`` for every carrier, but only over the
    numerical **row space** of the moment second-moment
    ``Ω = g'g/n_observations``. When the moments are near-collinear (so ``Ω`` is
    rank-deficient) the null-space directions are unidentified: the objective is
    *flat* along them, ``λ`` is not pinned down, and the Newton system is
    singular. A plain ``pinv(hessian)`` step masks this and wanders.

    A **separate** failure mode is that the sup itself can be unbounded: EL's
    ``rho(v) = ln(1-v)`` diverges as ``v → -∞``, which happens when the origin is
    not interior to the convex hull of the moment contributions. That is a
    statement about the moments' location, not about ``Ω``'s rank, and the
    row-space projection does nothing about it.

    The two are handled separately: (i) ``λ`` is restricted to the eigen-subspace
    of ``Ω`` with eigenvalues ``> rcond · λ_max``, which fixes the rank problem;
    and (ii) a trust-region cap ``‖λ‖ ≤ lam_cap`` bounds the divergence, with
    ``converged`` reported ``True`` only when the gradient genuinely vanishes
    strictly inside the cap.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        criterion: GEL carrier name.
        max_iter: Maximum Newton iterations (ignored for ``cue``, which is
            solved in closed form).
        tol: Absolute floor added to the scale-relative gradient tolerance
            ``rtol·‖ḡ‖``. Defaults to 0.0 — the relative term is the portable
            criterion; a fixed absolute floor is unreachable in float32.
        rcond: Relative eigenvalue floor for the ``Ω`` row-space projection.
            Ignored when ``subspace`` is supplied.
        lam_cap: Trust-region cap on ``‖λ‖``.
        subspace: Optional precomputed ``(basis, keep, rank)`` from
            :func:`_gel_identified_subspace`. Profiled callers pass the subspace
            frozen at their reference parameter so that the reported statistic,
            its ``dof`` and the profiled objective all describe the *same* set of
            identified directions.

    Returns:
        Tuple ``(lam, objective, converged, effective_rank)`` — ``lam`` in the
        *original* ``m``-space (zero in the dropped directions), and
        ``effective_rank`` the number of retained stochastic moment directions.

    """
    _require_host_gel(g, name="_gel_inner_sup")
    carrier = cast("GelCriterion", criterion)
    moments = jnp.asarray(g)
    if subspace is None:
        basis, keep = _gel_subspace_kernel(moments, rcond)
        rank = int(jnp.sum(keep))
    else:
        basis, keep, rank = subspace
        basis = jnp.asarray(basis, dtype=moments.dtype)
        keep = jnp.asarray(keep, dtype=moments.dtype)
    reduced = (moments @ basis) * keep
    beta, objective, converged = _gel_inner_sup_kernel(
        reduced,
        keep,
        carrier,
        max_iter,
        lam_cap,
        _GEL_NEWTON_RTOL,
        tol,
    )
    lam = basis @ beta
    return (
        lam,
        float(objective),
        bool(converged),
        rank,
    )


@partial(jax.jit, static_argnames=("criterion", "max_iter"))
def _gel_profile_objective_kernel(
    moments0: JaxFloat[Array, "t m"],
    derivative_panel: JaxFloat[Array, "t m k"],
    theta: JaxFloat[Array, "k"],
    basis: JaxFloat[Array, "m m"],
    keep: JaxFloat[Array, "m"],
    criterion: Static[GelCriterion],
    max_iter: Static[int],
) -> JaxFloat[Array, ""]:
    """Traceable fixed-shape profile objective for the outer GEL solver.

    Two properties make this differentiable and cheap, and both are load-bearing
    for correctness rather than merely for speed.

    **The identified subspace is frozen.** ``basis`` and ``keep`` are supplied by
    the caller, decided once at a reference parameter. Recomputing them from
    ``Ω(θ)`` inside this objective would make the profile *discontinuous*:
    ``Ω(θ) = Ω(0) + θ(ḡD' + Dḡ') + θ²DD'`` moves eigenvalue mass along the
    Jacobian direction, and a direction crossing ``rcond·λ_max`` changes the
    retained rank — so the objective jumps, and the outer minimizer can halt on
    the jump and report success. Freezing the subspace also keeps ``dof`` a
    property of the model rather than of where the optimizer stopped.

    **The inner argmax is frozen (envelope theorem).** At the inner supremum
    ``∂P/∂β = 0``: in the retained coordinates by convergence, and in the dropped
    ones because their columns of ``reduced`` are exactly zero. So
    ``dP/dθ = ∂P/∂θ`` evaluated at ``β*``, and ``stop_gradient(β*)`` yields the
    exact gradient without taping the solver — which also keeps the
    non-differentiable ``eigh`` and the Newton ``while_loop`` off the tape
    entirely.

    The envelope gradient is exact *at* the argmax; an unconverged inner solve
    biases it first-order in the inner residual, which is why the caller must
    read ``converged`` from :func:`_gel_inner_sup_kernel` rather than assume it.
    """
    moments = moments0 + jnp.einsum("tmk,k->tm", derivative_panel, theta)
    frozen_moments = jax.lax.stop_gradient(moments)
    beta, _objective, _converged = _gel_inner_sup_kernel(
        (frozen_moments @ basis) * keep,
        keep,
        criterion,
        max_iter,
    )
    beta = jax.lax.stop_gradient(beta)
    return _gel_objective((moments @ basis) * keep, beta, criterion)

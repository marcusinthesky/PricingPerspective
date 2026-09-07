"""Positive-semidefinite matrix repairs at the rank-0 numerical boundary.

Finite-sample covariance estimates and other symmetric operators can be
indefinite even when their population counterpart is positive semidefinite.
Every repair here is traceable JAX: there is no NumPy backend and no host
control flow, so each one composes under ``jit``, ``vmap``, and ``grad`` and
runs at whatever precision the caller has configured.

The repairs
-----------
1. :func:`nearest_correlation` — the Frobenius-nearest correlation matrix,
   computed by the Qi-Sun (2006) semismooth Newton method on the Lagrangian
   dual.  PREFERRED.  :func:`nearest_covariance` wraps it in correlation space
   so marginal variances are preserved exactly.
2. :func:`ridge_psd` — eigenvalue shift ``Σ + ε̃·I``.  The cheap baseline that
   ridging represents; changes every eigenvalue and inflates total variance.
3. :func:`gerber_shrink_psd` — diagonal shrink until PSD
   (``Σ(α) = (1-α)Σ + α·diag(Σ)``).  *Gerber-inspired* only: **not** a
   reimplementation of Gerber IQ / *Squeezing Financial Noise* (corpus
   ``gerber_iq_2025``, SSRN 4986939).
4. :func:`eig_clip` — eigenvalue clipping, included as the diagnostic
   counter-example: it clips negative eigenvalues to ``eps`` and thereby
   changes both diagonal scale and off-diagonals.

:func:`repair_diagnostics` reports condition number and off-diagonal Frobenius
distortion so callers can quantify the trade-off.

Why Newton rather than Dykstra alternating projections
------------------------------------------------------
Measured, not assumed.  The previous implementation ran Higham's Dykstra
alternating projections under a fixed 200-iteration budget with a ``1e-10``
relative-change tolerance.  On the production ``Σ_dist`` shape (52×52, Paper 3)
that tolerance was **never** reached: all 20 sampled matrices exhausted the
budget at ``max(dx, dy, dxy) ≈ 3e-5``, and the returned iterate sat ``7e-4``
(max entry) away from the same iteration run to convergence.  Dykstra converges
linearly, so the budget could not be tightened cheaply.

Qi-Sun Newton reaches the KKT point of the dual in 8-11 Newton steps on the
same 20 matrices — against 200 alternating projections — and lands ``4.7e-12``
from a Dykstra oracle run to 5000 iterations rather than ``7e-4``.

**Those 20 seeds are not the whole story, and this module does not state an
unconditional nearest-matrix contract.** ``converged`` is returned as a leaf
precisely because a real production matrix can fall outside what the sampled
seeds represent; callers that need *nearest* rather than *feasible* must read
it. The energy-robustness ``robust-higham`` arm does not read it, and describes
itself as a heuristic accordingly.

**The Paper 3 "boundary plateau" was a solver defect, fixed 2026-08-08 — it was
never a property of the input.** Two earlier notes here read the same stall as a
degenerate generalized Jacobian at the cone boundary, then as a state that "no
longer exists"; both were wrong, and the second was written from a single run
that happened to land on the good side of a coin flip.

Two independent defects, both required for the stall:

1. **The CG ridge was floored.** It read ``√eps · max(‖g‖, 1)``, pinning the
   regularisation at ``√eps`` = 1.49e-08 for *every* iterate with ``‖g‖ < 1`` —
   24 800x the ``n²·eps`` = 6.00e-13 stopping tolerance — so the Newton
   direction could not resolve the last ~1e-08 of gradient at any budget. It is
   now ``√eps · ‖g‖``, which tracks the accuracy still being demanded.
2. **The Armijo test ran below its own objective's resolution.** At
   ``‖g‖ ≈ 3.6e-08`` the decrement ``1e-4·t·slope`` is 1e-18 to 1e-28 against a
   dual objective of magnitude 251.76 whose float64 granularity is 5.59e-14: the
   line search was comparing bit-identical numbers, accepting on rounding noise
   for three steps and then exhausting all 30 halvings, forcing ``step_size = 0``
   and freezing the iterate. That is why the residual was bit-identical at 25,
   50, 100, 200 and 400 Newton steps and read as a plateau. Backtracking is now
   skipped once the demanded decrement falls below ``|value|·eps``.

Because the verdict turned on rounding, it was not reproducible. Before the fix,
a ``1e-16`` relative (one ulp) symmetric perturbation of the production
``Σ_dist`` flipped it: 6 of 8 draws converged at ~7.2e-14, 2 of 8 stalled — the
same 2-in-8 rate at 1e-15, 1e-14 and 1e-12. That is the whole history of the
published ``psd-converged`` flag: ``False@4.0977e-08``, then
``True@7.329280e-14``, then ``False@3.612131e-08``, across runs whose inputs
differed only in the last bits. Fixing the ridge alone moved the production
matrix off the knife edge but left the rate at 2-3 in 8 on perturbed draws; both
fixes together give **33 of 33** convergences at ~6.6e-14 over the same sweep.

Measured at the ``panel.py`` call site, one process, per dtype:

===========  ==========  ===================  ==========  ==========
input dtype  tolerance   residual             converged   iterations
===========  ==========  ===================  ==========  ==========
float64      6.0041e-13  **6.741998e-14**     True        7
float32      3.2234e-04  2.183340e-04         False       5
===========  ==========  ===================  ==========  ==========

The float32 row reports ``False`` by :data:`_NEAREST_CLAIM_CEILING`, not by its
stopping tolerance, which it does meet. Stopping at the dtype's roundoff floor
is not the same as being nearest, and the dual gradient ``diag(X) - e`` is
dimensionless, so the claim is gated on an absolute bar as well as a scaled one.

Precision policy
----------------
The working dtype is the input's; nothing here promotes to float64 or mutates
the caller's ``jax_enable_x64``.  The dual convergence tolerance, the Cholesky
eigenvalue floor, and the Newton system's Tikhonov level are all derived from
``n`` and the input dtype (:func:`_dual_tolerance`, :func:`_relative_floor`),
rather than pinning float64 constants into a float32 computation.

**The nearest-matrix contract is stated at float64.**  Over those 20 seeds,
float64 converges on 20/20; float32 converges on 11/20 and lands ``4.2e-03``
from the oracle.  That gap is the problem, not the solver: the nearest
correlation matrix is boundary-singular, so its smallest eigenvalues sit at
float32 epsilon and no amount of iteration resolves them.  In float32 the
return is a feasible unit-diagonal PSD repair with ``converged=False`` saying
so — the honest answer rather than a tolerance widened until it reads True.
Every production caller enables x64.

Convergence is reported as a leaf, never raised, so a capped iteration is
visible to a traced caller instead of aborting it.

References:
    Higham (2002; corpus ``higham_functions_2002``), *Computing the nearest
    correlation matrix*, IMA Journal of Numerical Analysis 22(3) — the problem
    and the alternating-projection method.
    Qi & Sun (2006), *A quadratically convergent Newton method for computing
    the nearest correlation matrix*, SIAM J. Matrix Anal. Appl. 28(2) — the
    dual semismooth Newton method implemented here.
    Gerber et al. (2025; corpus ``gerber_iq_2025``, SSRN 4986939) — full IQ /
    squeezing estimator; :func:`gerber_shrink_psd` is a lightweight
    diagonal-shrink *proxy*, not that estimator.

"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jcor.core.typing import (  # noqa: TC001  # runtime annotations
    Array,
    Bool,
    Float,
    Int,
    Static,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_RIDGE_EPS",
    "NearestCorrelationResult",
    "RepairDiagnostics",
    "ShrinkResult",
    "eig_clip",
    "gerber_shrink_psd",
    "nearest_correlation",
    "nearest_covariance",
    "repair_diagnostics",
    "ridge_psd",
]

#: Shared default ridge for :func:`ridge_psd`.
DEFAULT_RIDGE_EPS = 1e-8

_MATRIX_NDIM = 2
#: ``is_psd_after`` classification tolerance. Tight enough that a matrix
#: ``cholesky`` rejects is not rubber-stamped as PSD.
_PSD_CLASSIFICATION_TOLERANCE = 1e-10
#: Armijo sufficient-decrease fraction for the dual line search.
_ARMIJO_SLOPE_FRACTION = 1e-4
#: Backtracking cap. Reaching it means the step is rejected, not accepted.
_MAX_LINE_SEARCH_HALVINGS = 30
#: Multiple of ``n²·eps`` giving the dual stopping tolerance.
_TOLERANCE_SCALE = 1.0
#: Absolute ceiling on the dual residual for ``converged`` to be reported.
#:
#: The stopping tolerance is dtype-scaled because ``n²·eps`` is the achievable
#: roundoff floor, but *reporting convergence* against that floor lets a float32
#: run claim success at ``2.4e-04`` — a unit diagonal wrong in its fourth
#: decimal — purely because the bar moved with the dtype. The dual gradient is
#: ``diag(X) - e``, dimensionless with natural scale 1, so a nearest-matrix
#: claim admits an absolute bar. float64 clears this by four and a half orders
#: (measured 6.7e-14 on the Paper 3 panel); float32 does not clear it at any
#: budget, which is what the module means by "float32 is supported and reports
#: ``converged=False`` rather than overclaiming".
_NEAREST_CLAIM_CEILING = 1e-9
#: Multiple of ``n·eps`` giving the eigenvalue floor as a fraction of ``λ_max``.
_FLOOR_SCALE = 8.0


# ---------------------------------------------------------------------------
# Dtype-derived numerical policy
# ---------------------------------------------------------------------------


def _dual_tolerance(n: int, dtype: jnp.dtype) -> float:
    """Return a reachable stopping tolerance for ``‖diag(X₊) - 1‖``.

    The dual gradient is a diagonal residual against a unit target, so its
    natural scale is 1 and the achievable floor is roundoff accumulating
    through an ``n``-dimensional eigendecomposition. ``n²·eps`` is 2.2e-12 in
    float64 and 1.2e-03 in float32 at the production ``n = 100`` (it was 6.0e-13
    and 3.2e-04 at the retired ``n = 52``; the tolerance is computed from ``n``,
    so it tracked the roster change on its own).

    Measured across 20 production-shaped seeds, the solver *achieves* whatever
    tolerance it is given down to at least 3e-14 in float64 — the residual
    tracks the request rather than plateauing, so this default is a genuine
    accuracy choice and not a concession to noise. It is nonetheless only a
    request: :func:`nearest_correlation` returns the best iterate it saw, so
    setting it too tight costs Newton steps, never accuracy.

    Those seeds do not cover every real input, so "the residual tracks the
    request" remains a property of what was sampled rather than a guarantee:
    read ``converged`` rather than assuming the tolerance was met.

    This is a *stopping* tolerance only. Whether the nearest-matrix claim is
    reported rests additionally on :data:`_NEAREST_CLAIM_CEILING`, because in
    float32 this value is ``1.2e-03`` at the production ``n = 100`` and stopping
    there is not the same as being nearest. The gap widened with the roster: the
    float32 stopping tolerance is now ~1.2e+06 times the 1e-9 claim ceiling,
    against ~3.2e+05 at ``n = 52``, so a float32 run is further from supporting
    the nearest-matrix claim than it was. float64 remains well inside it.
    """
    return _TOLERANCE_SCALE * n * n * float(jnp.finfo(dtype).eps)


def _relative_floor(n: int, dtype: jnp.dtype) -> float:
    """Return the eigenvalue floor for Cholesky safety, as a fraction of λ_max.

    The Frobenius-nearest correlation matrix generically sits *on* the PSD-cone
    boundary, so the exact solution is singular and some floor is unavoidable
    if ``cholesky`` is to succeed. Expressing it relative to ``λ_max`` rather
    than as the former absolute ``1e-12`` removes the implicit assumption that
    the input's spectrum is ``O(1)``, and makes the floor track the working
    dtype instead of pinning a float64 constant into a float32 computation.

    It does **not** improve conditioning, and is not intended to: measured
    ``cond`` after repair is ~7e13 either way, because the near-singularity is
    a property of the solution, not of the floor. Callers who need a
    well-conditioned matrix must ridge explicitly (:func:`ridge_psd`), which is
    what the Paper 3 optimizer does before its linear solve.
    """
    return _FLOOR_SCALE * n * float(jnp.finfo(dtype).eps)


def _symmetrise(matrix: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Average a matrix with its transpose.

    ``eigh``/``eigvalsh`` read one triangle only, so without this the spectrum
    is computed from a different matrix than the one returned.
    """
    return 0.5 * (matrix + matrix.T)


def _require_square(matrix: Float[Array, "n n"], *, name: str) -> None:
    """Reject a nonsquare or empty operand at trace time."""
    if matrix.ndim != _MATRIX_NDIM or matrix.shape[0] != matrix.shape[1]:
        message = f"{name} must be a square matrix."
        raise ValueError(message)
    if matrix.shape[0] < 1:
        message = f"{name} must be nonempty."
        raise ValueError(message)


def _require_positive_finite(value: float, *, name: str) -> float:
    """Validate a strictly positive finite scalar tuning parameter."""
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        message = f"{name} must be finite and strictly positive."
        raise ValueError(message)
    return scalar


def _require_integer_at_least(value: int, *, name: str, minimum: int) -> int:
    """Validate a static iteration limit."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{name} must be an integer of at least {minimum}."
        raise ValueError(message)
    return value


# ---------------------------------------------------------------------------
# Spectral primitives
# ---------------------------------------------------------------------------


def _project_psd(
    matrix: Float[Array, "n n"],
    floor: float | Float[Array, ""] = 0.0,
) -> Float[Array, "n n"]:
    """Project a symmetric matrix onto the eigenvalue-floor cone."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetrise(matrix))
    return (eigenvectors * jnp.maximum(eigenvalues, floor)) @ eigenvectors.T


def _unit_diagonal(matrix: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Set the diagonal to one without host mutation."""
    return matrix.at[jnp.diag_indices(matrix.shape[0])].set(1.0)


def _loewner_weights(eigenvalues: Float[Array, " n"]) -> Float[Array, "n n"]:
    """Return the Löwner weights of the PSD projection's generalized Jacobian.

    For the projection ``X ↦ X₊`` at a symmetric ``X = P diag(λ) Pᵀ``, the
    Clarke generalized Jacobian acts as ``H ↦ P (Ω ∘ (Pᵀ H P)) Pᵀ`` with

    ``Ω_ij = 1`` if ``λ_i, λ_j > 0``, ``0`` if both are ``≤ 0``, and
    ``λ_i / (λ_i - λ_j)`` when exactly one is positive.

    All three branches collapse to ``(λ_i₊ + λ_j₊) / (|λ_i| + |λ_j|)``, which is
    branch-free and therefore cheap under ``vmap``. The zero denominator (both
    eigenvalues exactly zero) is the degenerate cone vertex and maps to 0.
    """
    positive = jnp.maximum(eigenvalues, 0.0)
    numerator = positive[:, None] + positive[None, :]
    magnitude = jnp.abs(eigenvalues)
    denominator = magnitude[:, None] + magnitude[None, :]
    singular = denominator <= 0.0
    return jnp.where(singular, 0.0, numerator / jnp.where(singular, 1.0, denominator))


def _jacobian_action(
    direction: Float[Array, " n"],
    eigenvectors: Float[Array, "n n"],
    weights: Float[Array, "n n"],
    ridge: Float[Array, ""],
) -> Float[Array, " n"]:
    """Apply the dual Hessian ``V(y) h = diag(P (Ω ∘ (Pᵀ diag(h) P)) Pᵀ) + rhoh``.

    Never materialises ``V`` itself: the dual variable is a vector of length
    ``n``, so the operator costs two ``n × n`` products per application and the
    Newton system is solved by conjugate gradients against it.

    ``V`` is singular wherever the active set is degenerate, which for a
    boundary solution is everywhere, so the Tikhonov term ``rho`` is load-bearing
    rather than cosmetic. Its level was chosen by measurement across 20
    production-shaped seeds and four stopping tolerances: ``√eps·max(‖g‖, 1)``
    hit the requested tolerance on every seed in ≤11 Newton steps, while
    ``rho = 0``, ``rho ∝ ‖g‖²``, ``rho ∝ tol`` and two fixed levels each stalled near
    ``1e-8`` on 3-8 of the 20 and exhausted the iteration budget. Tying ``rho`` to
    the stopping tolerance in particular makes the solver fragile — it couples
    the accuracy request to the conditioning of the system solved to meet it.
    """
    inner = (eigenvectors.T * direction[None, :]) @ eigenvectors
    product = eigenvectors @ (weights * inner)
    return jnp.sum(product * eigenvectors, axis=1) + ridge * direction


def _conjugate_gradient(
    operator: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    rhs: Float[Array, " n"],
    max_iterations: int,
    tolerance: Float[Array, ""],
) -> Float[Array, " n"]:
    """Solve ``V h = rhs`` for the PSD operator ``V`` by conjugate gradients."""

    def keep_going(
        state: tuple[Int[Array, ""], jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> Bool[Array, ""]:
        index, _, _, _, residual_norm_squared = state
        return (index < max_iterations) & (residual_norm_squared > tolerance)

    def step(
        state: tuple[Int[Array, ""], jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[Int[Array, ""], jax.Array, jax.Array, jax.Array, jax.Array]:
        index, solution, residual, search, residual_norm_squared = state
        curvature = operator(search)
        denominator = jnp.sum(search * curvature)
        positive = denominator > 0.0
        alpha = jnp.where(positive, residual_norm_squared, 0.0) / jnp.where(
            positive, denominator, 1.0
        )
        solution = solution + alpha * search
        residual = residual - alpha * curvature
        updated = jnp.sum(residual * residual)
        beta = updated / jnp.where(
            residual_norm_squared > 0.0, residual_norm_squared, 1.0
        )
        return index + 1, solution, residual, residual + beta * search, updated

    initial = jnp.sum(rhs * rhs)
    return lax.while_loop(
        keep_going,
        step,
        (jnp.zeros((), jnp.int32), jnp.zeros_like(rhs), rhs, rhs, initial),
    )[1]


# ---------------------------------------------------------------------------
# Nearest correlation matrix (Qi & Sun 2006)
# ---------------------------------------------------------------------------


class NearestCorrelationResult(NamedTuple):
    """A nearest-matrix repair and the evidence for its nearest-ness.

    Attributes:
        matrix: The repaired matrix, shape ``(n, n)``.
        residual: ``‖diag((C + diag(y))₊) - 1‖₂`` at the returned dual point —
            the KKT residual of the nearest-correlation problem. Compare it
            against the tolerance the call used, not against zero.
        converged: Whether ``residual`` fell below that tolerance within the
            Newton budget. When false, ``matrix`` is still a feasible PSD
            repair with the requested diagonal; it is simply not certified
            nearest.
        iterations: Newton steps actually taken.

    """

    matrix: Float[Array, "n n"]
    residual: Float[Array, ""]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]


def _newton_direction(
    gradient: Float[Array, " n"],
    gradient_norm: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    eigenvectors: Float[Array, "n n"],
    max_cg: int,
    dtype: jnp.dtype,
) -> tuple[Float[Array, " n"], Float[Array, ""]]:
    """Return the globalised Newton direction and its slope along the gradient."""
    weights = _loewner_weights(eigenvalues)
    # The CG regularisation must track the accuracy still being demanded.
    # It previously read ``jnp.maximum(gradient_norm, 1.0)``, which pins the
    # ridge at ``√eps`` (1.49e-08 in float64) for every iterate with
    # ``‖g‖ < 1`` — 24 800x the ``n²·eps`` = 6.00e-13 stopping tolerance, so
    # the Newton direction could not resolve the last ~1e-08 of gradient at
    # any budget. Scaling with ``‖g‖`` instead is what retires the Paper 3
    # plateau: that input converges in 6 steps to 6.74e-14 where the floored
    # ridge froze at 3.61e-08 through 400 steps.
    ridge = jnp.sqrt(jnp.finfo(dtype).eps) * gradient_norm
    direction = _conjugate_gradient(
        lambda h: _jacobian_action(h, eigenvectors, weights, ridge),
        -gradient,
        max_cg,
        jnp.square(jnp.minimum(0.1, gradient_norm) * gradient_norm),
    )
    slope = jnp.sum(gradient * direction)
    # A CG step truncated on a singular Hessian can fail to be a descent
    # direction; steepest descent always is, so the line search below
    # cannot stall.
    ascending = slope >= 0.0
    return (
        jnp.where(ascending, -gradient, direction),
        jnp.where(ascending, -jnp.square(gradient_norm), slope),
    )


def _armijo_step_size(
    objective: Callable[[Float[Array, " n"]], Float[Array, ""]],
    dual: Float[Array, " n"],
    direction: Float[Array, " n"],
    value: Float[Array, ""],
    slope: Float[Array, ""],
    dtype: jnp.dtype,
) -> Float[Array, ""]:
    """Return the backtracked step size, or zero when no halving is accepted."""

    def searching(
        search: tuple[Int[Array, ""], jax.Array, jax.Array],
    ) -> Bool[Array, ""]:
        halvings, _, accepted = search
        return (halvings < _MAX_LINE_SEARCH_HALVINGS) & ~accepted

    def halve(
        search: tuple[Int[Array, ""], jax.Array, jax.Array],
    ) -> tuple[Int[Array, ""], jax.Array, jax.Array]:
        halvings, step_size, _ = search
        decrement = _ARMIJO_SLOPE_FRACTION * step_size * slope
        # Armijo compares two evaluations of ``dual_value``, so it is only
        # meaningful while the decrement it demands exceeds that objective's
        # own representation granularity. Near the solution it does not: on
        # the Paper 3 panel the decrement is 1e-18 to 1e-28 against
        # ``|value|·eps`` = 5.59e-14, i.e. the test compares bit-identical
        # numbers and its verdict is rounding luck. Backtracking on that
        # verdict is what froze the iterate at ``step_size = 0``. Below the
        # granularity the semismooth Newton step is accepted unbacktracked —
        # globalisation is for the region where descent is still decidable.
        sufficient = (jnp.abs(decrement) < jnp.abs(value) * jnp.finfo(dtype).eps) | (
            objective(dual + step_size * direction) <= (value + decrement)
        )
        return (
            halvings + 1,
            jnp.where(sufficient, step_size, 0.5 * step_size),
            sufficient,
        )

    _, step_size, accepted = lax.while_loop(
        searching,
        halve,
        (jnp.zeros((), jnp.int32), jnp.ones((), dtype), jnp.zeros((), bool)),
    )
    return jnp.where(accepted, step_size, jnp.zeros((), dtype))


def _nearest_correlation_result(
    target: Float[Array, "n n"],
    dual: Float[Array, " n"],
    floor_fraction: float,
    tolerance: float,
    iterations: Int[Array, ""],
) -> NearestCorrelationResult:
    """Return the primal repair and KKT evidence at a converged dual point."""
    # One eigendecomposition serves the whole tail: the returned point's KKT
    # residual, the scale the floor is relative to, and the floored
    # reconstruction. Note the residual is measured *at* the returned dual, not
    # the pre-step value the loop guard last tested.
    eigenvalues, eigenvectors = jnp.linalg.eigh(target + jnp.diag(dual))
    positive = jnp.maximum(eigenvalues, 0.0)
    residual = jnp.linalg.norm(
        jnp.sum((eigenvectors * positive) * eigenvectors, axis=1) - 1.0
    )
    floor = floor_fraction * jnp.max(positive)
    floored = (eigenvectors * jnp.maximum(positive, floor)) @ eigenvectors.T
    scale = jnp.sqrt(jnp.diag(floored))
    normalised = _symmetrise(floored / jnp.outer(scale, scale))
    return NearestCorrelationResult(
        matrix=_unit_diagonal(normalised),
        residual=residual,
        converged=(residual <= tolerance) & (residual <= _NEAREST_CLAIM_CEILING),
        iterations=iterations,
    )


@partial(
    jax.jit,
    static_argnames=("max_newton", "max_cg", "tol", "eigenvalue_floor"),
)
def nearest_correlation(
    corr: Float[Array, "n n"],
    *,
    max_newton: Static[int] = 25,
    max_cg: Static[int] = 40,
    tol: Static[float | None] = None,
    eigenvalue_floor: Static[float | None] = None,
) -> NearestCorrelationResult:
    """Frobenius-nearest correlation matrix by semismooth Newton on the dual.

    Solves ``min ½‖X - C‖²_F`` over symmetric PSD ``X`` with ``diag(X) = 1``.
    The Lagrangian dual is the unconstrained, convex, continuously (but not
    twice) differentiable

    ``θ(y) = ½‖(C + diag(y))₊‖²_F - eᵀy``,

    whose gradient is ``diag((C + diag(y))₊) - e``. Newton steps use an element
    of the Clarke generalized Jacobian (:func:`_loewner_weights`), solved
    matrix-free by conjugate gradients and globalised by an Armijo backtracking
    line search with a steepest-descent fallback — so descent is guaranteed
    unconditionally, and the quadratic local rate of Qi & Sun (2006, Thm 5.3)
    applies under their constraint-nondegeneracy assumption.

    On convergence the primal iterate ``(C + diag(y))₊`` satisfies
    ``diag(X) = 1`` to working precision *before* any projection, which is why
    the returned matrix is the nearest one rather than a feasible repair near
    it. A relative eigenvalue floor and a congruence normalisation are applied
    last for Cholesky safety; both move the solution by ``O(n·eps·λ_max)``.

    Args:
        corr: Symmetric input matrix, shape ``(n, n)``. The diagonal is
            *ignored* — it enters the objective as a constant — so passing a
            covariance and passing its unit-diagonal form give the same answer.
        max_newton: Cap on Newton steps. Reaching it returns ``converged=False``
            rather than raising.
        max_cg: Cap on conjugate-gradient iterations per Newton step.
        tol: Stopping tolerance on the dual gradient norm. ``None`` derives a
            reachable one from ``n`` and the input dtype.
        eigenvalue_floor: Final eigenvalue floor as a fraction of ``λ_max``.
            ``None`` derives it from ``n`` and the input dtype.

    Returns:
        :class:`NearestCorrelationResult` with an exact unit diagonal.

    """
    _require_square(corr, name="corr")
    max_newton = _require_integer_at_least(max_newton, name="max_newton", minimum=1)
    max_cg = _require_integer_at_least(max_cg, name="max_cg", minimum=1)
    n = corr.shape[0]
    dtype = jnp.result_type(corr)
    tolerance = (
        _dual_tolerance(n, dtype)
        if tol is None
        else _require_positive_finite(tol, name="tol")
    )
    floor_fraction = (
        _relative_floor(n, dtype)
        if eigenvalue_floor is None
        else _require_positive_finite(eigenvalue_floor, name="eigenvalue_floor")
    )
    target = _symmetrise(corr)

    def dual_value(dual: Float[Array, " n"]) -> Float[Array, ""]:
        eigenvalues = jnp.linalg.eigvalsh(target + jnp.diag(dual))
        positive = jnp.maximum(eigenvalues, 0.0)
        return 0.5 * jnp.sum(positive**2) - jnp.sum(dual)

    def dual_value_and_gradient(
        dual: Float[Array, " n"],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        eigenvalues, eigenvectors = jnp.linalg.eigh(target + jnp.diag(dual))
        positive = jnp.maximum(eigenvalues, 0.0)
        value = 0.5 * jnp.sum(positive**2) - jnp.sum(dual)
        gradient = jnp.sum((eigenvectors * positive) * eigenvectors, axis=1) - 1.0
        return value, gradient, eigenvalues, eigenvectors

    def keep_going(
        state: tuple[Int[Array, ""], jax.Array, jax.Array, jax.Array],
    ) -> Bool[Array, ""]:
        index, _, _, best_residual = state
        return (index < max_newton) & (best_residual > tolerance)

    def newton_step(
        state: tuple[Int[Array, ""], jax.Array, jax.Array, jax.Array],
    ) -> tuple[Int[Array, ""], jax.Array, jax.Array, jax.Array]:
        index, dual, best_dual, best_residual = state
        value, gradient, eigenvalues, eigenvectors = dual_value_and_gradient(dual)
        gradient_norm = jnp.linalg.norm(gradient)
        # Keep the best point seen. Past the roundoff plateau the semismooth
        # iteration wanders in the degenerate directions of the active set and
        # the residual *rises* — measured 1.1e-14 at step 9 against 1.1e-09 at
        # step 60 on one production-shaped seed. Retaining the best iterate
        # makes a too-tight tolerance cost time rather than accuracy.
        improved = gradient_norm < best_residual
        best_dual = jnp.where(improved, dual, best_dual)
        best_residual = jnp.where(improved, gradient_norm, best_residual)
        direction, slope = _newton_direction(
            gradient, gradient_norm, eigenvalues, eigenvectors, max_cg, dtype
        )
        step_size = _armijo_step_size(dual_value, dual, direction, value, slope, dtype)
        return index + 1, dual + step_size * direction, best_dual, best_residual

    initial_dual = jnp.zeros(n, dtype)
    iterations, _, dual, _ = lax.while_loop(
        keep_going,
        newton_step,
        (
            jnp.zeros((), jnp.int32),
            initial_dual,
            initial_dual,
            jnp.asarray(jnp.inf, dtype),
        ),
    )
    return _nearest_correlation_result(
        target, dual, floor_fraction, tolerance, iterations
    )


@partial(
    jax.jit,
    static_argnames=("max_newton", "max_cg", "tol", "eigenvalue_floor"),
)
def nearest_covariance(
    sigma: Float[Array, "n n"],
    *,
    max_newton: Static[int] = 25,
    max_cg: Static[int] = 40,
    tol: Static[float | None] = None,
    eigenvalue_floor: Static[float | None] = None,
) -> NearestCorrelationResult:
    """Diagonal-preserving PSD covariance repair in correlation space.

    Rescales ``Σ`` to correlation form, runs :func:`nearest_correlation`, then
    rescales back with the original standard deviations. Total variance (the
    diagonal) is preserved exactly. Nearest is claimed in **correlation-space**
    Frobenius norm; after variance rescaling the result is generally not the
    nearest covariance in unweighted Frobenius norm, and that is deliberate —
    an unweighted-nearest covariance is free to move marginal variances, which
    for a portfolio target is the one thing it must not do.

    Args:
        sigma: Symmetric covariance, shape ``(n, n)``, with a strictly positive
            diagonal. A nonpositive diagonal yields nonfinite output rather
            than an exception; validate at the application boundary (checkify)
            if runtime data can violate it.
        max_newton: Cap on Newton steps.
        max_cg: Cap on conjugate-gradient iterations per Newton step.
        tol: Dual stopping tolerance; ``None`` derives one from ``n`` and dtype.
        eigenvalue_floor: Relative eigenvalue floor; ``None`` derives one.

    Returns:
        :class:`NearestCorrelationResult` whose ``matrix`` carries the original
        diagonal exactly.

    """
    _require_square(sigma, name="sigma")
    symmetric = _symmetrise(sigma)
    diagonal = jnp.diag(symmetric)
    deviation = jnp.sqrt(diagonal)
    scale = jnp.outer(deviation, deviation)
    result = nearest_correlation(
        symmetric / scale,
        max_newton=max_newton,
        max_cg=max_cg,
        tol=tol,
        eigenvalue_floor=eigenvalue_floor,
    )
    repaired = _symmetrise(result.matrix * scale)
    return result._replace(
        matrix=repaired.at[jnp.diag_indices(repaired.shape[0])].set(diagonal)
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def ridge_psd(
    sigma: Float[Array, "K K"],
    eps: float = DEFAULT_RIDGE_EPS,
) -> Float[Array, "K K"]:
    """Shift a matrix to a positive-definite spectrum.

    Returns ``Sigma + epsilon_tilde I`` where
    ``epsilon_tilde = max(0, -lambda_min(Sigma)) + eps``. The input is
    symmetrised first so the eigenspectrum and returned matrix use the same
    triangle.

    Args:
        sigma: Finite square matrix of shape ``(K, K)``. Eager callers are
            responsible for this value precondition; under ``jax.jit``, use
            checkify at the application boundary if runtime data can be
            nonfinite.
        eps: Positive finite minimum ridge after shifting the spectrum. This
            numerical tuning parameter must be concrete when the function is
            traced; close over it or mark it static when using `jax.jit`.

    Returns:
        Symmetric ridge-regularised matrix of the same shape.

    """
    _require_square(sigma, name="sigma")
    eps_value = _require_positive_finite(eps, name="eps")
    symmetric = _symmetrise(sigma)
    lam_min = jnp.min(jnp.linalg.eigvalsh(symmetric))
    ridge = jnp.maximum(0.0, -lam_min) + eps_value
    return symmetric + ridge * jnp.eye(symmetric.shape[0], dtype=symmetric.dtype)


def eig_clip(
    sigma: Float[Array, "n n"],
    eps: float = 1e-10,
) -> Float[Array, "n n"]:
    """Eigenvalue clipping — the variance-inflating counter-example.

    Clips eigenvalues below ``eps`` up to ``eps`` and reconstructs. Provided
    for the diagnostic comparison ONLY: unlike :func:`nearest_covariance` it
    changes the diagonal, so total variance is inflated rather than preserved.
    Prefer :func:`nearest_covariance`.

    Args:
        sigma: Symmetric matrix, shape ``(n, n)``.
        eps: Absolute floor for clipped eigenvalues.

    Returns:
        PSD matrix of the same shape.

    """
    _require_square(sigma, name="sigma")
    return _project_psd(sigma, floor=_require_positive_finite(eps, name="eps"))


# ---------------------------------------------------------------------------
# Gerber-inspired shrink-to-PSD
# ---------------------------------------------------------------------------


class ShrinkResult(NamedTuple):
    """A diagonal-shrink repair and the intensity it required.

    Attributes:
        matrix: ``Σ(α) = (1-α)Σ + α·diag(Σ)`` at the selected ``α``, shape
            ``(n, n)``.
        alpha: The smallest grid intensity meeting the PSD floor. When no grid
            point does, this is ``0.0`` and ``matrix`` is the unshrunk input —
            check ``feasible`` before reading either.
        feasible: Whether the grid contained an intensity meeting the floor.
            False means no repair was found, not that a partial one was.

    """

    matrix: Float[Array, "n n"]
    alpha: Float[Array, ""]
    feasible: Bool[Array, ""]


@partial(jax.jit, static_argnames=("step", "max_iter", "eps"))
def gerber_shrink_psd(
    sigma: Float[Array, "n n"],
    step: Static[float] = 0.05,
    max_iter: Static[int] = 100,
    eps: Static[float] = 1e-10,
) -> ShrinkResult:
    """Diagonal shrink until PSD (Gerber-*inspired* proxy, not full Gerber IQ).

    ``Σ(α) = (1 - α)·Σ + α·diag(Σ)`` — returns the smallest ``α`` on the grid
    ``min(1, k·step)``, ``k = 0…max_iter``, whose smallest eigenvalue clears
    ``eps``. Diagonal variances are preserved exactly (the shrink target *is*
    the diagonal) and off-diagonals are uniformly deflated.

    ``λ_min(Σ(α))`` is concave in ``α`` (a pointwise minimum of affine
    functions) and ``Σ(1) = diag(Σ)``, so the feasible set is an interval with
    right endpoint 1 whenever the diagonal itself clears ``eps``. The whole
    grid is therefore evaluated at once and the first feasible index taken —
    identical selection to a sequential scan, without its host control flow.

    Args:
        sigma: Symmetric covariance, shape ``(n, n)``. Every diagonal entry
            must be at least ``eps`` or no intensity can succeed; that shows up
            as ``feasible=False`` rather than an exception.
        step: Grid increment of the shrink intensity.
        max_iter: Number of grid increments past zero.
        eps: PSD floor the smallest eigenvalue must clear.

    Returns:
        :class:`ShrinkResult`.

    """
    _require_square(sigma, name="sigma")
    step_value = _require_positive_finite(step, name="step")
    if step_value > 1.0:
        message = "step must not exceed 1."
        raise ValueError(message)
    eps_value = _require_positive_finite(eps, name="eps")
    max_iter = _require_integer_at_least(max_iter, name="max_iter", minimum=0)
    symmetric = _symmetrise(sigma)
    target = jnp.diag(jnp.diag(symmetric))
    grid = jnp.minimum(
        1.0, jnp.arange(max_iter + 1, dtype=jnp.result_type(sigma)) * step_value
    )

    def candidate(alpha: Float[Array, ""]) -> Float[Array, "n n"]:
        return (1.0 - alpha) * symmetric + alpha * target

    smallest = jax.vmap(lambda a: jnp.min(jnp.linalg.eigvalsh(candidate(a))))(grid)
    feasible = smallest >= eps_value
    index = jnp.argmax(feasible)
    alpha = grid[index]
    return ShrinkResult(
        matrix=candidate(alpha),
        alpha=alpha,
        feasible=jnp.any(feasible),
    )


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


class RepairDiagnostics(NamedTuple):
    """Effect of a PSD repair relative to the raw (possibly non-PSD) matrix.

    Attributes:
        min_eig_before: Smallest eigenvalue of the raw matrix.
        min_eig_after: Smallest eigenvalue after repair.
        cond_before: Condition number of the raw matrix in absolute
            eigenvalues (``inf`` when singular).
        cond_after: Condition number after repair.
        total_var_before: Trace (total variance) of the raw matrix.
        total_var_after: Trace after repair — equal to ``total_var_before`` for
            diagonal-preserving repairs.
        offdiag_distortion: ``‖offdiag(Σ_repaired - Σ_raw)‖_F``.
        is_psd_after: Whether the repaired matrix is PSD at the documented
            ``-1e-10`` classification tolerance.
        converged: Convergence evidence supplied by an iterative repair, or
            ``None`` for noniterative calls. Pass
            :attr:`NearestCorrelationResult.converged` when the nearest-matrix
            claim matters.

    """

    min_eig_before: Float[Array, ""]
    min_eig_after: Float[Array, ""]
    cond_before: Float[Array, ""]
    cond_after: Float[Array, ""]
    total_var_before: Float[Array, ""]
    total_var_after: Float[Array, ""]
    offdiag_distortion: Float[Array, ""]
    is_psd_after: Bool[Array, ""]
    converged: Bool[Array, ""] | None = None


def repair_diagnostics(
    raw: Float[Array, "n n"],
    repaired: Float[Array, "n n"],
    *,
    converged: Bool[Array, ""] | bool | None = None,
) -> RepairDiagnostics:
    """Quantify a PSD repair's effect on conditioning and off-diagonals.

    Args:
        raw: The raw (possibly indefinite) matrix, shape ``(n, n)``.
        repaired: The repaired matrix, same shape.
        converged: Optional convergence leaf from an iterative repair.

    Returns:
        :class:`RepairDiagnostics` with array leaves.

    """
    _require_square(raw, name="raw")
    _require_square(repaired, name="repaired")
    if raw.shape != repaired.shape:
        message = "raw and repaired must have the same shape."
        raise ValueError(message)
    before = jnp.linalg.eigvalsh(_symmetrise(raw))
    after = jnp.linalg.eigvalsh(_symmetrise(repaired))

    def condition(eigenvalues: Float[Array, " n"]) -> Float[Array, ""]:
        magnitude = jnp.abs(eigenvalues)
        smallest = jnp.min(magnitude)
        return jnp.where(
            smallest > 0.0,
            jnp.max(magnitude) / jnp.where(smallest > 0.0, smallest, 1.0),
            jnp.inf,
        )

    off_diagonal = ~jnp.eye(raw.shape[0], dtype=bool)
    distortion = jnp.linalg.norm(jnp.where(off_diagonal, repaired - raw, 0.0))
    return RepairDiagnostics(
        min_eig_before=jnp.min(before),
        min_eig_after=jnp.min(after),
        cond_before=condition(before),
        cond_after=condition(after),
        total_var_before=jnp.trace(raw),
        total_var_after=jnp.trace(repaired),
        offdiag_distortion=distortion,
        is_psd_after=jnp.min(after) >= -_PSD_CLASSIFICATION_TOLERANCE,
        converged=None if converged is None else jnp.asarray(converged, dtype=bool),
    )

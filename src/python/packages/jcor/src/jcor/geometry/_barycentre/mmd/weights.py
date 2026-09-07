"""Energy barycentre weights: the public mixture-weight solvers.

The single, batched, and pairwise entry points that approximate a target
distribution as a convex combination of candidate clouds.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
from jax import vmap
from optax.projections import projection_simplex

from jcor.discrepancy._mmd.components import (
    _compute_energy_components,
    _compute_mmd_components,
)
from jcor.geometry._barycentre.mmd._common import (
    _DEGENERATE_CURVATURE,
    _MAX_DISTANCE_EXPONENT,
    _MIN_CANDIDATES,
    _STACKED_CLOUD_NDIM,
)
from jcor.geometry._barycentre.mmd.solve import (
    EnergyBarycentreDiagnostics,
    _BarycentreOutcome,
    _candidate_list,
    _certify_barycentre,
    _certify_kernel_barycentre,
    _energy_value,
    _exact_barycentre_qp_core,
    _simplex_tangent_geometry,
    _solve_barycentre,
)
from jcor.ground.config import (  # noqa: TC001  # runtime annotations
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.ground.metrics import cdist
from jcor.ground.similarities import SimilarityStrategy

if TYPE_CHECKING:
    from jcor.geometry._barycentre.mmd.solve import (
        KernelBarycentreDiagnostics,
        _BarycentreGeometry,
    )
    from jcor.ground._similarity_strategy import PositiveSemidefiniteKernelLaw


def _validate_energy_barycentre_request(
    exponent: float, solver: str, tol: float, cnd_tol: float, maxiter: int
) -> None:
    """Validate scalar options shared by the eager energy-barycentre API."""
    if not 0.0 < exponent < _MAX_DISTANCE_EXPONENT:
        message = f"exponent must be in (0, 2), got {exponent}"
        raise ValueError(message)
    if solver not in {"qp", "exact", "pgd"}:
        message = (
            "Unknown solver: use 'qp' (exact interior-point) for energy barycentres"
        )
        raise ValueError(message)
    if tol <= 0 or cnd_tol < 0 or maxiter <= 0:
        message = "tol and maxiter must be positive; cnd_tol must be nonnegative"
        raise ValueError(message)


def _energy_barycentre_components(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection,
    exponent: float,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve candidate clouds and produce the scalar barycentre components."""
    resolved_metric = resolve_ground_distance(metric)
    candidates_list = _candidate_list(target, candidates)
    if precomputed_components is None:
        gram, cross = _compute_energy_components(
            target, candidates_list, resolved_metric, exponent
        )
    else:
        gram, cross = precomputed_components
    gram = (gram + gram.T) / 2
    target_self = jnp.mean(cdist(target, target, metric=resolved_metric) ** exponent)
    return candidates_list, gram, cross, target_self


def _validate_kernel_barycentre_request(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    solver: str,
    tol: float,
    psd_tol: float,
    maxiter: int,
) -> None:
    """Validate scalar options shared by the eager kernel-barycentre API."""
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    if solver not in {"qp", "exact"}:
        message = "solver must be 'qp' or 'exact' for kernel barycentres"
        raise ValueError(message)
    if tol <= 0 or psd_tol < 0 or maxiter <= 0:
        message = "tol and maxiter must be positive; psd_tol must be nonnegative"
        raise ValueError(message)


def _validate_energy_cnd(
    geometry: _BarycentreGeometry,
    *,
    validate_cnd: bool,
    cnd_tol: float,
) -> None:
    """Reject an energy Gram matrix outside the requested CND tolerance."""
    cnd_scale = jnp.maximum(1.0, geometry.eigenvalue_norm)
    if validate_cnd and float(geometry.maximum_eigenvalue) > cnd_tol * float(cnd_scale):
        message = (
            "candidate distance Gram matrix is not conditionally negative "
            "definite (tangent maximum eigenvalue="
            f"{float(geometry.maximum_eigenvalue):.3e})"
        )
        raise ValueError(message)


def _validate_kernel_psd(
    geometry: _BarycentreGeometry,
    *,
    validate_psd: bool,
    psd_tol: float,
) -> None:
    """Reject a kernel Gram matrix outside the requested PSD tolerance."""
    psd_scale = jnp.maximum(1.0, geometry.eigenvalue_norm)
    if validate_psd and float(geometry.maximum_eigenvalue) > psd_tol * float(psd_scale):
        message = (
            "candidate kernel Gram matrix is not positive semidefinite on the "
            "simplex tangent (minimum eigenvalue="
            f"{float(-geometry.maximum_eigenvalue):.3e})"
        )
        raise ValueError(message)


def _solve_kernel_barycentre(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    target_self: jnp.ndarray,
    qp_gram: jnp.ndarray,
    qp_cross: jnp.ndarray,
    geometry: _BarycentreGeometry,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, KernelBarycentreDiagnostics]:
    """Solve the kernel barycentre QP and certify its returned weights."""
    target_term = -target_self
    candidate_count = len(cross)
    initial_weights = jnp.ones(candidate_count, dtype=gram.dtype) / candidate_count
    initial_mmd_squared = _energy_value(initial_weights, qp_cross, qp_gram, target_term)
    weights, iterations = _solve_barycentre(qp_gram, qp_cross, geometry, tol, maxiter)
    mmd_squared = _energy_value(weights, qp_cross, qp_gram, target_term)
    gradient = 2.0 * (gram @ weights - cross)
    diagnostics = _certify_kernel_barycentre(
        _BarycentreOutcome(
            weights,
            initial_mmd_squared,
            mmd_squared,
            gradient,
            geometry.maximum_eigenvalue,
            iterations,
        ),
        tol,
    )
    return weights, diagnostics


def _kernel_barycentre_components(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve candidate clouds and produce the kernel barycentre components."""
    candidates_list = _candidate_list(target, candidates)
    if precomputed_components is None:
        gram, cross, target_self = _compute_mmd_components(
            target, candidates_list, kernel, reference
        )
    else:
        gram, cross, target_self = precomputed_components
    return (gram + gram.T) / 2.0, cross, target_self


def _validate_batched_energy_inputs(
    grams: jnp.ndarray,
    *,
    validate_cnd: bool,
    cnd_tol: float,
) -> jnp.ndarray:
    """Validate shape and per-lane CND conditions for the batched wrapper."""
    _t, candidate_count, candidate_count_2 = grams.shape
    if candidate_count != candidate_count_2:
        message = (
            "grams must be square per lane, got "
            f"({candidate_count}, {candidate_count_2})"
        )
        raise ValueError(message)
    if candidate_count < _MIN_CANDIDATES:
        message = (
            "energy_barycentre_weights_batched requires k >= 2 candidates per "
            "lane; k == 1 (degenerate tangent space) must be handled by the "
            "caller outside the batched path"
        )
        raise ValueError(message)
    symmetrized = (grams + jnp.swapaxes(grams, -1, -2)) / 2
    eye = jnp.eye(candidate_count, dtype=grams.dtype)
    raw_basis = (
        eye[:, : candidate_count - 1] - eye[:, candidate_count - 1 : candidate_count]
    )
    tangent_basis, _ = jnp.linalg.qr(raw_basis, mode="reduced")
    tangent_max, tangent_norm = _batched_tangent_statistics(symmetrized, tangent_basis)
    cnd_violation = tangent_max > cnd_tol * jnp.maximum(1.0, tangent_norm)
    if validate_cnd and bool(jnp.any(cnd_violation)):
        bad_lanes = [i for i in range(symmetrized.shape[0]) if bool(cnd_violation[i])]
        message = (
            "candidate distance Gram matrix is not conditionally negative "
            f"definite for lanes {bad_lanes} (tangent maximum eigenvalue(s)="
            f"{[float(tangent_max[i]) for i in bad_lanes]})"
        )
        raise ValueError(message)
    return symmetrized


def _batched_tangent_statistics(
    grams: jnp.ndarray, tangent_basis: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return maximum and norm eigenvalues on the shared simplex tangent."""

    def _stats(gram: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        tangent_gram = tangent_basis.T @ gram @ tangent_basis
        tangent_gram = (tangent_gram + tangent_gram.T) / 2
        eigenvalues = jnp.linalg.eigvalsh(tangent_gram)
        return jnp.max(eigenvalues), jnp.max(jnp.abs(eigenvalues))

    return vmap(_stats)(grams)


def _batched_energy_gradient(
    gram: jnp.ndarray, cross: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Return the energy objective gradient for one or more lanes."""
    return 2 * cross - 2 * (gram @ weights)


def _batched_degenerate_weights(
    grams: jnp.ndarray, crosses: jnp.ndarray, initial_weights: jnp.ndarray, tol: float
) -> jnp.ndarray:
    """Return uniform-on-ties weights for degenerate tangent geometries."""

    def _one(gram: jnp.ndarray, cross: jnp.ndarray) -> jnp.ndarray:
        gradient = _batched_energy_gradient(gram, cross, initial_weights)
        minimum = jnp.min(gradient)
        ties = jnp.isclose(gradient, minimum, rtol=0.0, atol=tol)
        return ties.astype(gram.dtype) / jnp.sum(ties)

    return vmap(_one)(grams, crosses)


def _batched_energy_diagnostics(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
    weights: jnp.ndarray,
    initial_energy: jnp.ndarray,
    iterations: jnp.ndarray,
    tangent_max: jnp.ndarray,
    tol: float,
) -> EnergyBarycentreDiagnostics:
    """Compute traceable batched energy values and convergence diagnostics."""
    energy = vmap(lambda g, c, b, w: _energy_value(w, c, g, b))(
        grams, crosses, target_selfs, weights
    )
    gradient = vmap(_batched_energy_gradient)(grams, crosses, weights)
    frank_wolfe_gap = jnp.sum(weights * gradient, axis=-1) - jnp.min(gradient, axis=-1)
    projected = vmap(lambda w, g: w - projection_simplex(w - g))(weights, gradient)
    projected_norm = jnp.linalg.norm(projected, axis=-1)
    converged = frank_wolfe_gap <= tol * (1.0 + jnp.abs(energy))
    return EnergyBarycentreDiagnostics(
        converged=converged,
        iterations=iterations,
        energy=energy,
        initial_energy=initial_energy,
        frank_wolfe_gap=frank_wolfe_gap,
        projected_gradient_norm=projected_norm,
        cnd_tangent_max_eigenvalue=tangent_max,
        tangent_curvature_min=-2 * tangent_max,
        simplex_sum_error=jnp.abs(jnp.sum(weights, axis=-1) - 1.0),
        min_weight=jnp.min(weights, axis=-1),
    )


def _batched_energy_geometry(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Build the shared tangent basis and its per-lane curvature statistics."""
    _t, candidate_count, _ = grams.shape
    initial_weights = jnp.ones(candidate_count, dtype=grams.dtype) / candidate_count
    initial_energy = vmap(
        lambda gram, cross, target: _energy_value(initial_weights, cross, gram, target)
    )(grams, crosses, target_selfs)
    eye = jnp.eye(candidate_count, dtype=grams.dtype)
    raw_basis = (
        eye[:, : candidate_count - 1] - eye[:, candidate_count - 1 : candidate_count]
    )
    tangent_basis, _ = jnp.linalg.qr(raw_basis, mode="reduced")
    tangent_max, tangent_norm = _batched_tangent_statistics(grams, tangent_basis)
    return initial_weights, initial_energy, tangent_basis, tangent_max, tangent_norm


@overload
def energy_barycentre_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    return_diagnostics: bool = False,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray: ...


@overload
def energy_barycentre_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    return_diagnostics: bool = True,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]: ...


def energy_barycentre_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
    return_diagnostics: bool = False,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Minimize energy distance from a target to a simplex mixture.

    This is an eager API: it performs host-side input validation and, when
    requested, materializes Python-valued diagnostics.  For prevalidated
    batched components that must compose under JAX transformations, use
    :func:`_energy_barycentre_weights_batched_core`.

    With ``Q[k,l] = E d(Y_k,Y_l)^alpha``, ``c[k] = E d(X,Y_k)^alpha``, and
    ``b = E d(X,X')^alpha``, this solves

        min_w  2 c^T w - w^T Q w - b
        s.t.   1^T w = 1, w >= 0.

    Conditional negative definiteness of ``Q`` makes the objective convex on
    the simplex tangent space. The solver validates that property rather than
    replacing ``Q`` by a different positive-semidefinite objective, then solves
    the exact program with a deterministic primal-dual interior-point method
    (:func:`_exact_barycentre_qp`) on the tangent-reduced Hessian.

    Args:
        target: Target observations with shape ``(n, d)``.
        candidates: Candidate observation clouds, stacked or provided as a list.
        metric: Distance metric name or pairwise metric callable.
        exponent: Distance exponent in the open interval ``(0, 2)``.
        solver: Exact QP or projected-gradient solver selection.
        tol: Numerical convergence tolerance.
        maxiter: Maximum solver iterations.
        validate_cnd: Whether to verify conditional negative definiteness.
        cnd_tol: Conditional-negative-definiteness tolerance.
        return_diagnostics: Whether to return the numerical certificate.
        precomputed_components: Optional ``(XX, YX)`` pair (as returned by
            :func:`_compute_energy_components`) to reuse instead of
            recomputing the Gram/cross terms from scratch — e.g. sliced from
            a full pairwise :func:`jcor.discrepancy.energy.mean_distance_matrix`
            precomputed once for a batch of leave-one-out targets. When
            supplied, its entries must equal what
            ``_compute_energy_components(target, candidates, metric,
            exponent)`` would return for the same arguments up to
            summation-order/XLA-tiling tolerance (~1e-8 float64 / ~1e-6
            float32): diagonal and ``i < j`` Gram entries are bit-identical,
            while ``j > i`` entries are mirrored from ``i < j`` (author-approved
            contract relaxation 2026-07-18).

    Returns:
        Simplex weights, optionally paired with numerical diagnostics.

    """
    _validate_energy_barycentre_request(exponent, solver, tol, cnd_tol, maxiter)
    target = jnp.asarray(target)
    candidates_list, gram, cross, target_self = _energy_barycentre_components(
        target, candidates, metric, exponent, precomputed_components
    )

    candidate_count = len(candidates_list)
    initial_weights = jnp.ones(candidate_count, dtype=gram.dtype) / candidate_count
    initial_energy = _energy_value(initial_weights, cross, gram, target_self)

    geometry = _simplex_tangent_geometry(gram)
    _validate_energy_cnd(geometry, validate_cnd=validate_cnd, cnd_tol=cnd_tol)

    weights, iterations = _solve_barycentre(gram, cross, geometry, tol, maxiter)
    energy = _energy_value(weights, cross, gram, target_self)
    gradient = 2 * cross - 2 * (gram @ weights)
    diagnostics = _certify_barycentre(
        _BarycentreOutcome(
            weights,
            initial_energy,
            energy,
            gradient,
            geometry.maximum_eigenvalue,
            iterations,
        ),
        tol,
    )
    if return_diagnostics:
        return weights, diagnostics
    return weights


@overload
def kernel_barycentre_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None = None,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_psd: bool = True,
    psd_tol: float = 1e-8,
    return_diagnostics: Literal[False] = False,
    precomputed_components: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
    | None = None,
) -> jnp.ndarray: ...


@overload
def kernel_barycentre_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None = None,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_psd: bool = True,
    psd_tol: float = 1e-8,
    return_diagnostics: Literal[True] = True,
    precomputed_components: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
    | None = None,
) -> tuple[jnp.ndarray, KernelBarycentreDiagnostics]: ...


def kernel_barycentre_weights(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: jnp.ndarray | None = None,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_psd: bool = True,
    psd_tol: float = 1e-8,
    return_diagnostics: bool = False,
    precomputed_components: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
    | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, KernelBarycentreDiagnostics]:
    """Minimize a typed-kernel MMD squared distance to a simplex mixture.

    For candidate Gram means ``G``, target--candidate means ``c``, and target
    self mean ``b``, this minimizes ``b - 2 c.T w + w.T G w`` over the
    simplex. The existing exact simplex QP is reused after negating the PSD
    Gram and cross terms, which exposes the same tangent-space curvature
    certificate without creating a second optimizer.
    """
    _validate_kernel_barycentre_request(kernel, solver, tol, psd_tol, maxiter)
    target = jnp.asarray(target)
    gram, cross, target_self = _kernel_barycentre_components(
        target, candidates, kernel, reference, precomputed_components
    )

    # The MMD objective is convex because G is PSD. Negating it puts the
    # objective into the shared energy-QP sign convention.
    qp_gram = -gram
    qp_cross = -cross
    geometry = _simplex_tangent_geometry(qp_gram)
    _validate_kernel_psd(geometry, validate_psd=validate_psd, psd_tol=psd_tol)
    weights, diagnostics = _solve_kernel_barycentre(
        gram,
        cross,
        target_self,
        qp_gram,
        qp_cross,
        geometry,
        tol,
        maxiter,
    )
    if return_diagnostics:
        return weights, diagnostics
    return weights


@partial(jax.jit, static_argnums=(3, 4))
def _energy_barycentre_weights_batched_core(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Transformation-safe batched energy-barycentre kernel.

    Inputs must already be shape-validated and conditionally negative definite
    on the simplex tangent space.  The eager
    :func:`energy_barycentre_weights_batched` wrapper owns those checks and
    converts their failures into informative Python exceptions.

    Returns:
        Pair ``(weights, diagnostics)`` of JAX arrays; every diagnostics field
        is traceable, so this function composes under :func:`jax.jit`,
        :func:`jax.vmap`, and JAX control-flow transforms.

    """
    grams = (grams + jnp.swapaxes(grams, -1, -2)) / 2
    w0, initial_energy, tangent_basis, tangent_max, tangent_norm = (
        _batched_energy_geometry(grams, crosses, target_selfs)
    )
    curvature_scale = 2.0 * tangent_norm

    w_qp, iters_qp = vmap(
        lambda g, c: _exact_barycentre_qp_core(g, c, tangent_basis, tol, maxiter)
    )(grams, crosses)
    w_degenerate = _batched_degenerate_weights(grams, crosses, w0, tol)

    is_degenerate = curvature_scale <= _DEGENERATE_CURVATURE
    weights = jnp.where(is_degenerate[:, None], w_degenerate, w_qp)
    iterations = jnp.where(
        is_degenerate, jnp.ones(grams.shape[0], dtype=iters_qp.dtype), iters_qp
    ).astype(jnp.int32)

    diagnostics = _batched_energy_diagnostics(
        grams,
        crosses,
        target_selfs,
        weights,
        initial_energy,
        iterations,
        tangent_max,
        tol,
    )
    return weights, diagnostics


def energy_barycentre_weights_batched(
    grams: jnp.ndarray,
    crosses: jnp.ndarray,
    target_selfs: jnp.ndarray,
    tol: float,
    maxiter: int,
    *,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics]:
    """Eager validation wrapper for the batched energy-barycentre kernel.

    Solves the exact energy-barycentre QP for a batch of ``T`` lanes that
    share one candidate count ``K`` (as within a hedging-error cell, where
    ``K = n_assets - 1`` is constant across targets), using ``jax.vmap``
    across the lane axis of the interior-point solve and every diagnostic
    computed by :func:`energy_barycentre_weights`. This eliminates the
    per-target Python dispatch of a plain ``for`` loop over
    :func:`energy_barycentre_weights` while reproducing, lane-for-lane, the
    same arithmetic that loop performs — only ``qpax``'s ``lax.while_loop``
    now runs to the batch-max iteration count rather than each lane's own
    count (a wall-clock difference only; the converged iterate does not
    depend on how many extra no-op iterations follow it, since the KKT
    system already at the fixed point stays there — see the module's
    ``_exact_barycentre_qp`` docstring for the interior-point method).

    Only the ``k >= 2`` (non-degenerate tangent space) case is supported;
    callers must handle ``k == 1`` lanes separately (unused by the current
    hedging-error caller, whose ``K = n_assets - 1 >= 2`` by construction).

    Args:
        grams: Batched, symmetrised or not, Gram matrices, shape ``(T, K, K)``.
        crosses: Batched cross terms, shape ``(T, K)``.
        target_selfs: Batched target self-energy terms ``b``, shape ``(T,)``.
        tol: Frank-Wolfe-gap convergence tolerance (shared across lanes).
        maxiter: Maximum interior-point iterations (shared across lanes).
        validate_cnd: Whether to reject any lane whose candidate Gram matrix
            is not conditionally negative definite on the simplex tangent
            space.
        cnd_tol: Relative tolerance for the per-lane CND check.

    Returns:
        Pair ``(weights, diagnostics)`` where ``weights`` has shape
        ``(T, K)`` and every field of ``diagnostics`` is a length-``T``
        array (one entry per lane), matching the scalar fields
        :func:`energy_barycentre_weights` returns per call.

    Raises:
        ValueError: If ``k < 2``, or (when ``validate_cnd``) any lane's Gram
            matrix fails the CND check — the message names the offending
            lane indices.

    """
    grams = _validate_batched_energy_inputs(
        jnp.asarray(grams), validate_cnd=validate_cnd, cnd_tol=cnd_tol
    )
    crosses = jnp.asarray(crosses)
    target_selfs = jnp.asarray(target_selfs)

    return _energy_barycentre_weights_batched_core(
        grams, crosses, target_selfs, tol, maxiter
    )


def energy_barycentre_weights_pairwise(
    distributions: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    solver: str = "qp",
    tol: float = 1e-8,
    maxiter: int = 10_000,
    validate_cnd: bool = True,
    cnd_tol: float = 1e-8,
) -> jnp.ndarray:
    """Compute exact energy-barycentre weights for every leave-one-out target.

    Returns:
        Matrix whose rows contain leave-one-out barycentre weights.

    """
    if isinstance(distributions, jnp.ndarray):
        if distributions.ndim != _STACKED_CLOUD_NDIM:
            message = "stacked distributions must have shape (K, m, d)"
            raise ValueError(message)
        distributions_list = [distributions[i] for i in range(distributions.shape[0])]
    else:
        distributions_list = [
            jnp.asarray(distribution) for distribution in distributions
        ]
    if len(distributions_list) < _MIN_CANDIDATES:
        message = "at least two distributions are required"
        raise ValueError(message)

    resolved_metric = resolve_ground_distance(metric)
    weights = []
    for i, target in enumerate(distributions_list):
        candidates = [
            distribution for j, distribution in enumerate(distributions_list) if j != i
        ]
        weights.append(
            energy_barycentre_weights(
                target,
                candidates,
                metric=resolved_metric,
                exponent=exponent,
                solver=solver,
                tol=tol,
                maxiter=maxiter,
                validate_cnd=validate_cnd,
                cnd_tol=cnd_tol,
            )
        )
    return jnp.stack(weights)

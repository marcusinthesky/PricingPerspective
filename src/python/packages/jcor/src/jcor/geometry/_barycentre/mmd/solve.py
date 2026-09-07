"""Geometry, certification, and the reduced-QP solver for the energy barycentre.

Private machinery behind :mod:`jcor.geometry._barycentre.mmd.weights`; depends
only on :mod:`jcor.geometry._barycentre.mmd._common`. (Named
``jcor.kernels.{weights,_common}`` before t46.6 dissolved that package — the
exact barycentre needs the conditionally-negative-definite geometry of the
candidate Gram matrix, which is a geometry object, not a discrepancy one.)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import qpax
from optax.projections import projection_simplex

from jcor.geometry._barycentre.mmd._common import (
    _DEGENERATE_CURVATURE,
    _MATRIX_NDIM,
    _QP_MAX_ITER,
    _QP_RIDGE_REL,
    _QP_TOL_FACTOR,
    _STACKED_CLOUD_NDIM,
)


class EnergyBarycentreDiagnostics(NamedTuple):
    """Numerical certificate returned by the energy-barycentre solver.

    Fields are Python scalars on the per-call (serial) solvers and length-``T``
    arrays on the ``vmap``-batched solver (:func:`energy_barycentre_weights_batched`),
    hence the ``scalar | jnp.ndarray`` unions; the batched caller indexes the
    array fields per lane.
    """

    converged: bool | jnp.ndarray
    iterations: int | jnp.ndarray
    energy: float | jnp.ndarray
    initial_energy: float | jnp.ndarray
    frank_wolfe_gap: float | jnp.ndarray
    projected_gradient_norm: float | jnp.ndarray
    cnd_tangent_max_eigenvalue: float | jnp.ndarray
    tangent_curvature_min: float | jnp.ndarray
    simplex_sum_error: float | jnp.ndarray
    min_weight: float | jnp.ndarray


class KernelBarycentreDiagnostics(NamedTuple):
    """Numerical certificate for a typed kernel-MMD barycentre."""

    converged: bool
    iterations: int
    mmd_squared: float
    initial_mmd_squared: float
    frank_wolfe_gap: float
    projected_gradient_norm: float
    psd_tangent_min_eigenvalue: float
    tangent_curvature_min: float
    simplex_sum_error: float
    min_weight: float


class _BarycentreGeometry(NamedTuple):
    """Tangent-space geometry of a candidate distance Gram matrix."""

    basis: jnp.ndarray
    eigenvalue_norm: jnp.ndarray
    maximum_eigenvalue: jnp.ndarray


class _BarycentreOutcome(NamedTuple):
    """Numerical state required to certify a barycentre solution."""

    weights: jnp.ndarray
    initial_energy: jnp.ndarray
    energy: jnp.ndarray
    gradient: jnp.ndarray
    tangent_maximum: jnp.ndarray
    iterations: int


def _simplex_tangent_geometry(gram: jnp.ndarray) -> _BarycentreGeometry:
    """Return an orthonormal simplex-tangent basis and its Gram spectrum."""
    candidate_count = gram.shape[0]
    if candidate_count == 1:
        return _BarycentreGeometry(
            jnp.zeros((candidate_count, 0), dtype=gram.dtype),
            jnp.asarray(0.0, dtype=gram.dtype),
            jnp.asarray(0.0, dtype=gram.dtype),
        )

    eye = jnp.eye(candidate_count, dtype=gram.dtype)
    raw_basis = (
        eye[:, : candidate_count - 1] - eye[:, candidate_count - 1 : candidate_count]
    )
    tangent_basis, _ = jnp.linalg.qr(raw_basis, mode="reduced")
    tangent_gram = tangent_basis.T @ gram @ tangent_basis
    tangent_gram = (tangent_gram + tangent_gram.T) / 2
    eigenvalues = jnp.linalg.eigvalsh(tangent_gram)
    return _BarycentreGeometry(
        tangent_basis,
        jnp.max(jnp.abs(eigenvalues)),
        jnp.max(eigenvalues),
    )


def _solve_barycentre(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    geometry: _BarycentreGeometry,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, int]:
    """Solve the flat or strictly convex simplex barycentre objective."""
    candidate_count = gram.shape[0]
    weights = jnp.ones(candidate_count, dtype=gram.dtype) / candidate_count
    gradient = 2 * cross - 2 * (gram @ weights)
    curvature_scale = 2 * float(geometry.eigenvalue_norm)
    if curvature_scale <= _DEGENERATE_CURVATURE:
        minimum_gradient = jnp.min(gradient)
        ties = jnp.isclose(gradient, minimum_gradient, rtol=0.0, atol=tol)
        return ties.astype(gram.dtype) / jnp.sum(ties), 1
    return _exact_barycentre_qp(
        gram,
        cross,
        geometry.basis,
        tol,
        maxiter,
    )


def _certify_barycentre(
    outcome: _BarycentreOutcome, tol: float
) -> EnergyBarycentreDiagnostics:
    """Build the eager numerical certificate for a barycentre solution."""
    weights = outcome.weights
    frank_wolfe_gap = jnp.dot(weights, outcome.gradient) - jnp.min(outcome.gradient)
    projected_gradient_norm = jnp.linalg.norm(
        weights - projection_simplex(weights - outcome.gradient)
    )
    converged = float(frank_wolfe_gap) <= tol * (1.0 + abs(float(outcome.energy)))
    return EnergyBarycentreDiagnostics(
        converged=converged,
        iterations=outcome.iterations,
        energy=float(outcome.energy),
        initial_energy=float(outcome.initial_energy),
        frank_wolfe_gap=float(frank_wolfe_gap),
        projected_gradient_norm=float(projected_gradient_norm),
        cnd_tangent_max_eigenvalue=float(outcome.tangent_maximum),
        tangent_curvature_min=float(-2 * outcome.tangent_maximum),
        simplex_sum_error=abs(float(jnp.sum(weights)) - 1.0),
        min_weight=float(jnp.min(weights)),
    )


def _certify_kernel_barycentre(
    outcome: _BarycentreOutcome, tol: float
) -> KernelBarycentreDiagnostics:
    """Build the MMD certificate from the shared simplex-QP outcome."""
    weights = outcome.weights
    frank_wolfe_gap = jnp.dot(weights, outcome.gradient) - jnp.min(outcome.gradient)
    projected_gradient_norm = jnp.linalg.norm(
        weights - projection_simplex(weights - outcome.gradient)
    )
    converged = float(frank_wolfe_gap) <= tol * (1.0 + abs(float(outcome.energy)))
    return KernelBarycentreDiagnostics(
        converged=converged,
        iterations=outcome.iterations,
        mmd_squared=float(outcome.energy),
        initial_mmd_squared=float(outcome.initial_energy),
        frank_wolfe_gap=float(frank_wolfe_gap),
        projected_gradient_norm=float(projected_gradient_norm),
        psd_tangent_min_eigenvalue=float(-outcome.tangent_maximum),
        tangent_curvature_min=float(-2 * outcome.tangent_maximum),
        simplex_sum_error=abs(float(jnp.sum(weights)) - 1.0),
        min_weight=float(jnp.min(weights)),
    )


def _candidate_list(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
) -> list[jnp.ndarray]:
    """Normalize and validate stacked/list candidate inputs.

    Returns:
        Validated list of two-dimensional candidate arrays.

    """
    target = jnp.asarray(target)
    if target.ndim != _MATRIX_NDIM or target.shape[0] == 0:
        message = "target must be a non-empty array of shape (n, d)"
        raise ValueError(message)

    if isinstance(candidates, jnp.ndarray):
        if candidates.ndim != _STACKED_CLOUD_NDIM:
            message = "stacked candidates must have shape (K, m, d)"
            raise ValueError(message)
        candidates_list = [candidates[i] for i in range(candidates.shape[0])]
    else:
        candidates_list = [jnp.asarray(candidate) for candidate in candidates]

    if not candidates_list:
        message = "at least one candidate distribution is required"
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(target))):
        message = "target contains non-finite values"
        raise ValueError(message)
    for candidate in candidates_list:
        if (
            candidate.ndim != _MATRIX_NDIM
            or candidate.shape[0] == 0
            or candidate.shape[1] != target.shape[1]
        ):
            message = (
                "each candidate must be non-empty with the target feature dimension"
            )
            raise ValueError(message)
        if not bool(jnp.all(jnp.isfinite(candidate))):
            message = "candidates contain non-finite values"
            raise ValueError(message)
    return candidates_list


def _energy_value(
    weights: jnp.ndarray,
    cross: jnp.ndarray,
    gram: jnp.ndarray,
    target_self: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the full target-to-mixture energy objective.

    Returns:
        Scalar empirical squared energy distance.

    """
    return 2 * jnp.dot(cross, weights) - jnp.dot(weights, gram @ weights) - target_self


def _exact_barycentre_qp(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    tangent_basis: jnp.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, int]:
    """Solve the exact energy-barycentre QP on the simplex tangent space.

    Minimizing ``2 cᵀw - wᵀQw`` over the simplex is convex because ``Q`` is
    conditionally negative definite on the zero-sum tangent space. Writing
    ``w = w₀ + B u`` for the uniform vertex ``w₀`` and an orthonormal tangent
    basis ``B`` (so ``1ᵀw ≡ 1`` for all ``u``) removes the equality constraint
    and yields a genuinely convex reduced program

        min_u  ½ uᵀH u + gᵀu,     H = -2 BᵀQB ⪰ 0,   g = 2 Bᵀ(c - Q w₀),

    subject to the box image ``-B u ≤ w₀`` (i.e. ``w ≥ 0``). Because ``H`` is
    positive semidefinite in the ambient space (unlike ``-2Q``, which is
    indefinite along ``1``), the primal-dual interior-point method in
    :mod:`qpax` solves it deterministically to a machine-precision optimality
    gap — replacing the earlier projected-gradient iteration, which stalled at
    loose, non-reproducible gaps.

    Args:
        gram: Symmetrised energy Gram matrix ``Q`` of shape ``(K, K)``.
        cross: Target-to-candidate energy vector ``c`` of shape ``(K,)``.
        tangent_basis: Orthonormal basis ``B`` of the zero-sum tangent space,
            shape ``(K, K-1)``.
        tol: Interior-point convergence tolerance.
        maxiter: Caller iteration budget; the interior-point cap is
            ``min(maxiter, _QP_MAX_ITER)``.

    Returns:
        Pair ``(weights, iterations)`` with simplex-feasible ``weights``.

    """
    weights, iterations = _exact_barycentre_qp_core(
        gram, cross, tangent_basis, tol, maxiter
    )
    return weights, int(iterations)


def _exact_barycentre_qp_core(
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    tangent_basis: jnp.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Traceable core of :func:`_exact_barycentre_qp` (no host-side ``int()`` cast).

    Identical arithmetic to :func:`_exact_barycentre_qp`; split out so that
    :func:`energy_barycentre_weights_batched` can ``vmap`` it directly —
    ``int(iterations)`` inside the non-batched wrapper requires a concrete
    (non-traced) value and is therefore incompatible with ``vmap``/``jit``
    tracing.

    Returns:
        Pair ``(weights, iterations)`` where ``iterations`` is a traced
        scalar array (not a Python ``int``).

    """
    k = gram.shape[0]
    w0 = jnp.ones(k, dtype=gram.dtype) / k
    tangent_gram = tangent_basis.T @ gram @ tangent_basis
    tangent_gram = (tangent_gram + tangent_gram.T) / 2
    hessian = -2.0 * tangent_gram
    # Negligible ridge (relative to the Hessian scale) guarantees strict
    # convexity when candidates are degenerate (identical clouds → an exact
    # zero tangent direction), without perturbing the optimum beyond ~1e-12.
    ridge = _QP_RIDGE_REL * jnp.maximum(1.0, jnp.max(jnp.abs(jnp.diag(hessian))))
    hessian = hessian + ridge * jnp.eye(k - 1, dtype=gram.dtype)
    linear = 2.0 * (tangent_basis.T @ (cross - gram @ w0))
    no_eq = jnp.zeros((0, k - 1), dtype=gram.dtype)
    no_eq_rhs = jnp.zeros((0,), dtype=gram.dtype)
    reduced, _slack, _dual_ineq, _dual_eq, _converged, iterations = qpax.solve_qp(
        hessian,
        linear,
        no_eq,
        no_eq_rhs,
        -tangent_basis,
        w0,
        solver_tol=tol * _QP_TOL_FACTOR,
        max_iter=int(min(maxiter, _QP_MAX_ITER)),
    )
    weights = w0 + tangent_basis @ reduced
    weights = jnp.clip(weights, 0.0, None)
    weights = weights / jnp.sum(weights)
    return weights, iterations

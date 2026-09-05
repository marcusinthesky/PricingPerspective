"""Weight selection, block splitting, and grouped energy shared across the tests.

This module is the sink of the :mod:`jcor.inference` permutation machinery: it
imports no sibling, so every other module may depend on it cycle-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jcor.core.axioms import Law  # noqa: TC001  # runtime annotations
from jcor.discrepancy.balancing import energy_distance_kernel
from jcor.geometry import energy_barycentre_weights
from jcor.ground._strategy import GroundDistance  # noqa: TC001  # runtime annotations

if TYPE_CHECKING:
    from jcor.geometry import EnergyBarycentreDiagnostics


def _optimal_weights(
    x: jnp.ndarray,
    candidates: list[jnp.ndarray],
    exponent: float,
    metric: GroundDistance[Law],
    solver: str,
    tol: float,
    maxiter: int,
    weight_method: str,
) -> tuple[jnp.ndarray, EnergyBarycentreDiagnostics | None, str]:
    """Select simplex weights using the requested, explicit objective.

    Returns:
        Tuple of optimal simplex weights, exact-solver diagnostics when
        available, and the solver actually used.

    """
    if weight_method == "energy_barycentre":
        exact_solver = "pgd" if solver in {"boxosqp", "eqcp"} else solver
        weights, diagnostics = energy_barycentre_weights(
            x,
            candidates,
            metric=metric,
            exponent=exponent,
            solver=exact_solver,
            tol=tol,
            maxiter=maxiter,
            return_diagnostics=True,
        )
        if not diagnostics.converged:
            message = (
                "energy-barycentre solver did not satisfy the KKT tolerance "
                f"after {diagnostics.iterations} iterations"
            )
            raise RuntimeError(message)
        return weights, diagnostics, exact_solver
    if weight_method != "kernel_balance":
        message = "weight_method must be 'kernel_balance' or 'energy_barycentre'"
        raise ValueError(message)

    return (
        energy_distance_kernel(
            x,
            candidates,
            metric=metric,
            exponent=exponent,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
        ),
        None,
        solver,
    )


def _split_blocks(n: int, frac_a: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Contiguous block split into selection-A (first frac) and test-B indices.

    Contiguous (not random) so that serial dependence is respected — the
    selection sample and the test sample are non-overlapping time blocks.

    Returns:
        Tuple (selection-A indices, test-B indices).

    """
    n_a = max(1, round(frac_a * n))
    n_a = min(n_a, n - 1)  # keep at least one point for B
    idx = jnp.arange(n)
    return idx[:n_a], idx[n_a:]


def _energy_from_group_blocks(
    block_sums: jnp.ndarray, sizes: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Mixture energy V-statistic from pooled group-block distance sums.

    ``block_sums[a, b] = Σ_{i∈a, j∈b} d(i, j)`` over the ``(K+1)`` pooled groups
    (group 0 is the target ``X``); dividing by ``sizes[a]·sizes[b]`` recovers the
    block means, so ``E = 2 wᵀ·XY - term_xx - wᵀ·YY·w`` exactly as in
    :func:`jcor.discrepancy.energy.mixture_energy_distance` (self-distances contribute
    zero and are kept in the diagonal counts, matching the V-statistic
    normalization).

    Returns:
        Scalar mixture energy distance ``E(X, Y_w)``.

    """
    block_means = block_sums / (sizes[:, None] * sizes[None, :])
    return (
        2.0 * jnp.dot(weights, block_means[0, 1:])
        - block_means[0, 0]
        - jnp.dot(weights, block_means[1:, 1:] @ weights)
    )

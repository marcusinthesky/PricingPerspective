"""Studentized mixture statistic and its jitted core.

Divides the grouped energy statistic by an inverse-variance effective-sample-size
scale so the null distribution is stable across split geometries.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

from jcor.discrepancy.balancing import inverse_variance_m_eff
from jcor.discrepancy.energy import _pooled_from_candidates
from jcor.ground.config import (  # noqa: TC001  # runtime annotations
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.ground.metrics import cdist
from jcor.inference._permutation.blocks import grouped_block_reduction
from jcor.inference._permutation.common import _energy_from_group_blocks


@partial(jax.jit, static_argnums=(4, 5))
def _studentized_core(
    distances: jnp.ndarray,
    group_ids: jnp.ndarray,
    weights: jnp.ndarray,
    group_variances: jnp.ndarray | None,
    n_groups: int,
    eps: float,
) -> jnp.ndarray:
    """Vectorized studentized mixture statistic from a pooled distance matrix.

    Thin front end over :func:`_studentized_from_reduction`: it builds the one-hot
    membership for the pooled row order and takes the ``(B, R)`` reduction of the
    *unpermuted* matrix. A permutation replicate wants the same statistic from a
    reindexed matrix, which
    :mod:`jcor.inference._permutation.blocks` supplies without gathering ``D`` —
    hence the split, so both callers share one jackknife implementation.

    Args:
        distances: Pooled ``(N, N)`` pre-exponentiated distance matrix.
        group_ids: ``(N,)`` group label per pooled row.
        weights: Mixture weights ``(K,)``.
        group_variances: Optional per-candidate variances for the
            heteroscedastic effective size.
        n_groups: Static group count ``K+1``.
        eps: Variance floor.

    Returns:
        Scalar studentized statistic ``√(scale)·Ê/σ̂``.

    """
    membership = jax.nn.one_hot(group_ids, n_groups, dtype=distances.dtype)
    # `grouped_block_reduction(Mᵀ, D)` is exactly the former
    # `row_sums = D @ M; block_sums = Mᵀ @ row_sums` — same two matmuls, same
    # order, so this front end is bit-identical to the pre-refactor core.
    block_sums, row_sums = grouped_block_reduction(membership.T, distances)
    return _studentized_from_reduction(
        block_sums,
        row_sums,
        membership,
        group_ids,
        weights,
        group_variances,
        n_groups,
        eps,
    )


def _studentized_from_reduction(
    block_sums: jnp.ndarray,
    row_sums: jnp.ndarray,
    membership: jnp.ndarray,
    group_ids: jnp.ndarray,
    weights: jnp.ndarray,
    group_variances: jnp.ndarray | None,
    n_groups: int,
    eps: float,
) -> jnp.ndarray:
    """Studentized mixture statistic from an already-reduced distance matrix.

    Algebraically identical to the delete-1 jackknife of
    :func:`studentized_mixture_statistic`, but computed in closed form: block
    sums ``B = Mᵀ D M`` and per-point group row-sums ``R = D M`` (with ``M`` the
    one-hot group membership) let every leave-one-out replicate be obtained by a
    rank-structured update ``B - (e_aᵀr + rᵀe_a)`` instead of re-slicing the
    distance matrix.  This removes the ``O(n + Σ m_k)`` Python/eager-op loop and
    makes the whole statistic a single jittable, ``vmap``-friendly kernel.

    Taking ``(B, R)`` as arguments rather than ``D`` is what lets a permutation
    replicate reuse this jackknife: under a permutation the reduction is
    ``B = C D Cᵀ`` and ``R = (D Cᵀ)[perm]``, so the caller supplies the
    replicate's reduction and ``membership``/``group_ids`` stay the fixed
    slot-order vectors. ``R`` must therefore be in **slot** order, not pooled
    order.

    Args:
        block_sums: ``(K+1, K+1)`` group-block distance sums for the replicate.
        row_sums: ``(N, K+1)`` per-slot group distance sums, in slot order.
        membership: ``(N, K+1)`` one-hot slot membership.
        group_ids: ``(N,)`` group label per slot.
        weights: Mixture weights ``(K,)``.
        group_variances: Optional per-candidate variances for the
            heteroscedastic effective size.
        n_groups: Static group count ``K+1``.
        eps: Variance floor.

    Returns:
        Scalar studentized statistic ``√(scale)·Ê/σ̂``.

    """
    n_pooled = row_sums.shape[0]
    sizes = membership.sum(0)  # (K+1,) = [n, m_1, ..., m_K]
    n = sizes[0]
    e_hat = _energy_from_group_blocks(block_sums, sizes, weights)

    # `p` is the vmap-traced point index, not a Python int (t46.1).
    def _energy_leave_one_out(p: jnp.ndarray) -> jnp.ndarray:
        r = row_sums[p]
        e_a = jax.nn.one_hot(group_ids[p], n_groups, dtype=row_sums.dtype)
        # Dropping point p (group a) removes its row/col from every block it
        # touches: subtract r from row a and column a of the block-sum matrix.
        delta = e_a[:, None] * r[None, :] + r[:, None] * e_a[None, :]
        return _energy_from_group_blocks(block_sums - delta, sizes - e_a, weights)

    e_loo = vmap(_energy_leave_one_out)(jnp.arange(n_pooled))  # (N,)
    # Per-group jackknife variance summed over X and each candidate group:
    #   jack_var = Σ_g (n_g - 1)/n_g · Σ_{p∈g} (e_loo[p] - mean_g)²
    group_mean = (membership.T @ e_loo) / sizes
    group_ss = membership.T @ (e_loo - group_mean[group_ids]) ** 2
    jack_var = jnp.sum((sizes - 1.0) / sizes * group_ss)
    sigma = jnp.sqrt(jnp.maximum(jack_var, eps))

    m_eff = inverse_variance_m_eff(weights, sizes[1:], group_variances)
    scale = (n * m_eff) / (n + m_eff)
    return jnp.sqrt(jnp.maximum(scale, 0.0)) * e_hat / sigma


def studentized_mixture_statistic(
    x: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    weights: jnp.ndarray,
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
    group_variances: jnp.ndarray | None = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Studentized energy statistic for the weighted-mixture two-sample problem.

    Correction #4 (studentize) composed with Correction #1 (inverse-variance
    effective size).  Returns ``T = √(scale) · Ê / σ̂`` where:

    - ``Ê = E(X, Y_w)`` is the mixture energy distance,
    - ``scale = (n·m_eff)/(n+m_eff)`` with the **inverse-variance** ``m_eff``
      (``1/m_eff = Σ_k w_k²/m_k`` or the heteroscedastic form) — the harmonic
      outer factor kept from the correct two-sample normalization,
      - ``σ̂`` is a multi-sample jackknife SE of ``Ê``: leave-one-out over ``X``
      and over each candidate ``Y_k`` separately (sum of group jackknife vars).

    Studentization is what restores **asymptotic** validity under the composite
    mixture null when the pooled sample is *not* exchangeable (heterogeneous
    ``m_k``). It is exact only under the identical-distribution null ``H0: F_X =
    F_{Y_k} ∀k``. Ground: Chung & Romano (2013): a studentized (asymptotically
    pivotal) statistic makes the permutation distribution mimic the sampling
    distribution under the weak/parameter null, retaining exact level when P=Q.

    .. warning::
        This studentizes the **non-degenerate** part. Under the exact null the
        energy statistic is degenerate (``Σ_k λ_k Z_k²``); no scalar normalizer
        is exact. Use permutation/bootstrap calibration for finite-sample size.

    Args:
        x: Target sample, shape (n, d).
        candidates: K candidate samples (list or (K, m, d) array).
        weights: Mixture weights, shape (K,).
        exponent: Distance exponent α in (0, 2).
        metric: Distance metric.
        group_variances: Optional per-candidate variances for heteroscedastic
            ``m_eff``.
        eps: Numerical floor on σ̂.

    Returns:
        Scalar studentized statistic (larger ⇒ more distributional difference).

    """
    clist, _ = _pooled_from_candidates(candidates)
    resolved_metric = resolve_ground_distance(metric)
    sizes = [len(x)] + [len(c) for c in clist]

    # Pool once and compute the full pairwise distance matrix a single time; the
    # jackknife delete-1 replicates are then closed-form rank updates on the
    # group-block sums (see :func:`_studentized_core`) rather than an eager
    # per-point Python loop that re-slices ``cdist`` blocks.
    pooled = jnp.concatenate([x, *clist], axis=0)
    distances = cdist(pooled, pooled, metric=resolved_metric) ** exponent
    group_ids = jnp.concatenate(
        [jnp.full(size, g, dtype=jnp.int32) for g, size in enumerate(sizes)]
    )
    gvar = None if group_variances is None else jnp.asarray(group_variances)
    return _studentized_core(
        distances, group_ids, jnp.asarray(weights), gvar, len(sizes), float(eps)
    )

"""Per-split execution of the corrected mixture test.

Private machinery behind :mod:`jcor.inference.mixture`: runs one train/test
split end to end, from weight selection through the studentized statistic and
its bootstrap interval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from jcor.core.axioms import Law  # noqa: TC001  # runtime annotations
from jcor.discrepancy.balancing import inverse_variance_m_eff
from jcor.ground._strategy import GroundDistance  # noqa: TC001  # runtime annotations
from jcor.ground.metrics import cdist
from jcor.inference._permutation.blocks import (
    grouped_block_reduction,
    map_replicates,
    replicate_counts,
)
from jcor.inference._permutation.common import (
    _energy_from_group_blocks,
    _optimal_weights,
)
from jcor.inference._results import MixtureTestResult
from jcor.inference.bootstrap import _bootstrap_energy_ci
from jcor.inference.studentized import (
    _studentized_core,
    _studentized_from_reduction,
    studentized_mixture_statistic,
)

if TYPE_CHECKING:
    from jcor.geometry import EnergyBarycentreDiagnostics


class _PooledGeometry(NamedTuple):
    """Pooled-sample layout and its single reused pairwise distance matrix.

    Attributes:
        pooled: Row-concatenated ``[X, C_1, ..., C_K]`` sample.
        distances: Pooled pairwise matrix ``cdist(pooled, pooled) ** exponent``.
        sizes: Per-group row counts, target first.
        offsets: Cumulative group row offsets, shape ``(K + 2,)``.
        group_ids: Contiguous-block group label per pooled row.
        membership: One-hot group membership, shape ``(N, K + 1)``.
        size_arr: Float group sizes, shape ``(K + 1,)``.
        n_groups: Number of groups ``K + 1``.

    """

    pooled: jnp.ndarray
    distances: jnp.ndarray
    sizes: list[int]
    offsets: jnp.ndarray
    group_ids: jnp.ndarray
    membership: jnp.ndarray
    size_arr: jnp.ndarray
    n_groups: int


def _pooled_geometry(
    x: jnp.ndarray,
    clist: list[jnp.ndarray],
    exponent: float,
    metric: GroundDistance[Law],
) -> _PooledGeometry:
    """Pool the samples once and build the shared pairwise distance matrix.

    D-reuse (see :func:`_run_test_on_split`): ``group_ids`` is the fixed
    contiguous-block label vector matching the ``[X, C_1, ..., C_K]`` row order
    of ``pooled``/``distances``.

    Args:
        x: Target sample, shape (n, d).
        clist: Candidate samples.
        exponent: Distance exponent α in (0, 2).
        metric: Ground distance.

    Returns:
        The pooled layout together with its reusable distance matrix.

    """
    sizes = [len(x)] + [len(c) for c in clist]
    offsets = jnp.cumsum(jnp.array([0, *sizes]))
    pooled = jnp.concatenate([x, *clist], axis=0)
    n_groups = len(sizes)
    distances = cdist(pooled, pooled, metric=metric) ** exponent
    group_ids = jnp.concatenate(
        [jnp.full(size, g, dtype=jnp.int32) for g, size in enumerate(sizes)]
    )
    membership = jax.nn.one_hot(group_ids, n_groups, dtype=distances.dtype)
    return _PooledGeometry(
        pooled=pooled,
        distances=distances,
        sizes=sizes,
        offsets=offsets,
        group_ids=group_ids,
        membership=membership,
        size_arr=membership.sum(0),  # (K+1,) float group sizes
        n_groups=n_groups,
    )


def _reoptimized_null(
    geometry: _PooledGeometry,
    w_star: jnp.ndarray,
    keys: jnp.ndarray,
    exponent: float,
    metric: GroundDistance[Law],
    group_variances: jnp.ndarray | None,
    *,
    solver: str,
    tol: float,
    maxiter: int,
    weight_method: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the serial re-optimize-in-permutation null.

    NOT vectorizable: :func:`_optimal_weights` re-solves the QP per replicate
    (qpax/PGD), which has no jittable/vmap-able form. This stays the original
    serial Python loop over ``_stat_from_pool``. The observed statistic here
    re-optimizes ``w`` on the full pooled sample, so it CANNOT be sourced from
    the fixed-``w_star`` hoisted ``D``.

    Args:
        geometry: Pooled layout and distance matrix.
        w_star: Selection-stage mixture weights (loop seed only).
        keys: Per-replicate PRNG keys.
        exponent: Distance exponent α in (0, 2).
        metric: Ground distance.
        group_variances: Optional per-candidate variances.
        solver: Mixture-weight solver.
        tol: Numerical solver tolerance.
        maxiter: Maximum solver iterations.
        weight_method: Mixture-weight objective selector.

    Returns:
        Tuple of the observed studentized statistic and the null distribution.

    """
    sizes = geometry.sizes
    offsets = geometry.offsets
    pooled = geometry.pooled

    def _stat_from_pool(
        pool: jnp.ndarray, w: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        xp = pool[: sizes[0]]
        cp = [pool[offsets[i] : offsets[i + 1]] for i in range(1, len(sizes))]
        w, _, _ = _optimal_weights(
            xp,
            cp,
            exponent,
            metric,
            solver,
            tol,
            maxiter,
            weight_method,
        )
        return studentized_mixture_statistic(
            xp, cp, w, exponent, metric, group_variances
        ), w

    t_obs, _ = _stat_from_pool(pooled, w_star)
    null_stats = []
    for key in keys:
        perm = random.permutation(key, pooled.shape[0])
        t_b, _ = _stat_from_pool(pooled[perm], w_star)
        null_stats.append(t_b)
    return t_obs, jnp.array(null_stats)


def _hoisted_null(
    geometry: _PooledGeometry,
    w_arr: jnp.ndarray,
    gvar: jnp.ndarray | None,
    keys: jnp.ndarray,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the fixed-``w*`` null on the hoisted pooled distance matrix.

    Permutation-invariant distance hoist (t05.2): the pooled pairwise distance
    matrix D is invariant under a joint (row, label) permutation of the block
    sums ``B = Mᵀ D M`` (see ``tests/migration/_parity.py``::
    ``assert_reduction_identity`` for the bit-exact identity that licenses
    this). The null replicate loop permutes the *membership* while keeping
    ``group_ids`` as the fixed contiguous-block vector -- this reproduces the
    oracle's per-replicate ``pooled[perm]`` draw element-wise (up to the
    documented ~1e-8 XLA matmul-tiling drift), instead of recomputing ``cdist``
    from scratch on every replicate. Permuting the membership rather than
    gathering ``D[perm][:, perm]`` is the same identity read in the other
    direction, and avoids materialising an (N, N) view per replicate.
    ``t_obs`` is fed the same ``distances`` array to the same core, so it is
    bit-identical to the pre-D-reuse value.

    Args:
        geometry: Pooled layout and distance matrix.
        w_arr: Fixed mixture weights.
        gvar: Optional per-candidate variances.
        keys: Per-replicate PRNG keys, shape ``(B, 2)``.
        eps: Studentization floor.

    Returns:
        Tuple of the observed studentized statistic and the null distribution.

    """
    distances = geometry.distances
    group_ids = geometry.group_ids
    n_groups = geometry.n_groups
    t_obs = _studentized_core(distances, group_ids, w_arr, gvar, n_groups, eps)

    perms = jax.vmap(lambda k: random.permutation(k, geometry.pooled.shape[0]))(keys)
    n_pooled = distances.shape[0]

    def _null_stat(perm: jnp.ndarray) -> jnp.ndarray:
        # Push the replicate into the membership matrix instead of D: with
        # C the permuted membership, B = C D Cᵀ and the per-slot row sums
        # are (D Cᵀ)[perm]. Both are what the former
        # `_studentized_core(distances[perm][:, perm], ...)` computed from
        # its gathered view, so the jackknife downstream is unchanged.
        counts = replicate_counts(perm, group_ids, n_groups, n_pooled, distances.dtype)
        block_sums, pooled_group_sums = grouped_block_reduction(counts, distances)
        return _studentized_from_reduction(
            block_sums,
            pooled_group_sums[perm],
            geometry.membership,
            group_ids,
            w_arr,
            gvar,
            n_groups,
            eps,
        )

    return t_obs, map_replicates(_null_stat, perms)


def _run_test_on_split(
    x: jnp.ndarray,
    clist: list[jnp.ndarray],
    w_star: jnp.ndarray,
    num_permutations: int,
    exponent: float,
    metric: GroundDistance[Law],
    group_variances: jnp.ndarray | None,
    confidence_level: float,
    num_bootstrap: int,
    rng: jnp.ndarray,
    *,
    reoptimized: bool,
    solver: str,
    tol: float,
    maxiter: int,
    reoptimize: bool,
    weight_method: str,
    selection_diagnostics: EnergyBarycentreDiagnostics | None,
    solver_used: str,
) -> MixtureTestResult:
    """Shared studentized-permutation machinery for both #3 architectures.

    **Pooled distance-matrix reuse (t05.2 / "D-reuse", author-approved
    2026-07-18).** The pooled pairwise matrix ``D = cdist(pooled, pooled,
    metric) ** exponent`` is computed **ONCE** here and reused for three
    consumers:

    - the observed studentized statistic ``t_obs`` (non-reoptimize path):
      fed as the *same array* to the *same* :func:`_studentized_core` the old
      :func:`studentized_mixture_statistic` call would have built internally,
      so ``t_obs`` is **bit-identical** to the pre-D-reuse value;
    - the energy point estimate ``Ê`` via block sums ``B = Mᵀ D M`` +
      :func:`_energy_from_group_blocks` (the V-statistic block means keep the
      zero diagonals, so this is algebraically identical to
      :func:`mixture_energy_distance` — ≤~1e-8 f64 drift vs a fresh ``cdist``);
    - the bootstrap distribution of ``Ê`` (see :func:`_bootstrap_energy_ci`),
      where each within-group resample is a **reindexed view** ``D[idx][:,
      idx]`` of the same ``D`` rather than a fresh per-replicate ``cdist``.

    The point estimate and bootstrap drift at the documented ~1e-8
    XLA-matmul-tiling class (reindexed view vs fresh ``cdist``; see
    ``jax-batch-matmul-tiling-nonreproducible``); the observed statistic and
    the permutation null are unchanged (bit-identical). The ``reoptimize=True``
    permutation-null loop stays the serial QP-per-replicate path.

    Returns:
        A :class:`jcor.inference.mixture.MixtureTestResult` with the studentized
        statistic, its permutation p-value, the energy point estimate and
        bootstrap CI.

    """
    eps = 1e-8
    # D-reuse: pool once, build the pooled pairwise distance matrix a single
    # time (see this function's docstring).
    geometry = _pooled_geometry(x, clist, exponent, metric)
    gvar = None if group_variances is None else jnp.asarray(group_variances)
    w_arr = jnp.asarray(w_star)

    keys = random.split(rng, num_permutations)

    if reoptimize:
        t_obs, null = _reoptimized_null(
            geometry,
            w_star,
            keys,
            exponent,
            metric,
            group_variances,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
            weight_method=weight_method,
        )
    else:
        t_obs, null = _hoisted_null(geometry, w_arr, gvar, keys, eps)

    # One-sided (greater): large studentized statistic ⇒ difference.
    pvalue = float((1.0 + jnp.sum(null >= t_obs)) / (1.0 + num_permutations))

    # Ê from the reused D via block sums B = M^T D M (algebraically identical
    # to mixture_energy_distance; ~1e-8 reindexed-view drift, author-approved).
    block_sums_obs = geometry.membership.T @ geometry.distances @ geometry.membership
    e_hat = float(_energy_from_group_blocks(block_sums_obs, geometry.size_arr, w_arr))
    ci, boot_dist = _bootstrap_energy_ci(
        geometry.distances,
        geometry.membership,
        geometry.size_arr,
        geometry.sizes,
        w_arr,
        confidence_level,
        num_bootstrap,
        rng,
    )
    m_sizes = jnp.array([len(c) for c in clist], dtype=jnp.float32)
    m_eff = float(inverse_variance_m_eff(w_star, m_sizes, group_variances))

    return MixtureTestResult(
        energy_point_estimate=e_hat,
        energy_ci=ci,
        studentized_statistic=float(t_obs),
        pvalue=pvalue,
        weights=w_star,
        m_eff=m_eff,
        null_distribution=null,
        bootstrap_distribution=boot_dist,
        reoptimized_in_permutation=reoptimized,
        studentized=True,
        equivalence_framed=False,
        num_permutations=num_permutations,
        weight_method=weight_method,
        solver_used=solver_used,
        solver_converged=(
            bool(selection_diagnostics.converged)
            if selection_diagnostics is not None
            else None
        ),
        solver_frank_wolfe_gap=(
            float(selection_diagnostics.frank_wolfe_gap)
            if selection_diagnostics is not None
            else None
        ),
        solver_cnd_tangent_max_eigenvalue=(
            float(selection_diagnostics.cnd_tangent_max_eigenvalue)
            if selection_diagnostics is not None
            else None
        ),
        solver_simplex_sum_error=(
            float(selection_diagnostics.simplex_sum_error)
            if selection_diagnostics is not None
            else None
        ),
        solver_min_weight=(
            float(selection_diagnostics.min_weight)
            if selection_diagnostics is not None
            else None
        ),
    )

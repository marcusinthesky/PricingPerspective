"""Moment-dimension reduction by averaging pair moments into clusters.

Reduces ``C(n,2)`` off-diagonal moments into a user-selected number of grouped
moments, which can make covariance estimation tractable when the raw moment
dimension is large relative to the observation count.

Why this module is eager and stays eager
----------------------------------------
:func:`cluster_pairs` is the only function in ``jcor`` whose **output shape**
depends on its input *values* rather than their shapes: empty clusters are
dropped, so the column count is decided by the contents of ``assignments``.
Boolean-mask selection with a concrete mask is legal in eager JAX and produces
exactly that variable width, so t65.1 converted it in place — the documented
``result.shape[1]`` contract is unchanged and no caller moved. It is **not**
jittable, and wrapping it in :func:`jax.jit` raises
``NonConcreteBooleanIndexError`` rather than silently padding.

A traceable variant would have to return the full ``(n_observations,
n_clusters)`` matrix plus an ``(n_clusters,)`` occupancy mask and let the
caller compact — the masking pattern ``_gel_subspace_kernel`` already uses for
rank selection. That is an additive API change, deliberately out of scope here.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import jax.numpy as jnp

from jcor.core.typing import Array, ArrayLike, Float  # noqa: TC001


def cluster_pairs(
    pair_moments: ArrayLike,
    assignments: ArrayLike,
    n_clusters: int,
) -> Float[Array, "t clusters"]:
    """Average per-observation pair moments into grouped moments.

    Averaging within each cluster yields at most ``n_clusters`` moment columns
    and can avoid a singular estimated moment covariance when the raw pair
    count exceeds the available observations.

    Any cluster label in ``[0, n_clusters)`` that receives **no** pairs would
    otherwise leave an all-zero moment column, making the downstream HAC
    rank-deficient (silently absorbed by ``pinv`` and corrupting the χ² dof).
    Empty clusters are therefore **dropped** and the result is re-indexed to
    only the non-empty labels — so it never contains an all-zero column.  Use
    ``result.shape[1]`` to recover the effective number of clusters.

    Evaluates at the caller's dtype; it does not promote to float64.

    Args:
        pair_moments: Per-period pair moment process with shape
            ``(n_observations, P)``, where ``P`` is the number of pairs.
        assignments: Integer cluster index (in ``[0, n_clusters)``) for each of
            the ``P`` pairs, shape ``(P,)``. Typed ``ArrayLike`` rather than
            ``Int[Array, …]`` because ``apps/pipeline`` builds this label vector
            in NumPy, and a ``jax.Array``-only annotation rejects it under the
            jaxtyping import hook.
        n_clusters: Number of clusters ``≤ P``.

    Returns:
        Clustered moment process, shape ``(n_observations, n_nonempty)`` where
        ``n_nonempty ≤ n_clusters`` is the number of clusters that actually
        received at least one pair (columns ordered by ascending label).

    Raises:
        ValueError: If no cluster receives any pair (empty result).

    """
    gradient = jnp.asarray(pair_moments)
    a = jnp.asarray(assignments)
    cols = []
    for c in range(n_clusters):
        mask = a == c
        # `mask` is concrete, so this data-dependent selection is legal eagerly.
        if bool(jnp.any(mask)):
            cols.append(gradient[:, mask].mean(axis=1))
    if not cols:
        message = (
            "cluster_pairs: no cluster received any pair; "
            "check `assignments` and `n_clusters`."
        )
        raise ValueError(message)
    return jnp.column_stack(cols)

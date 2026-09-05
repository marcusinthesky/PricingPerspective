"""S4 — spatial (geometric) medians and the spread of a cloud around them.

Position
--------
rank 4 · geometry. Consumes *coordinates* — an embedding, typically from
:mod:`jcor.geometry.embedding` — and produces per-group medians plus the
PERMDISP-style homogeneity-of-dispersion F-ratio measured against them.

Why the F-ratio lives here and not in ``association/``
-------------------------------------------------------
The t46.6 module plan put the whole DISCO family in
``association/dispersion.py``. The import fence forbids it. ``association`` and
``geometry`` are **rank-tied at 4** in ``JCOR_STAGE_RANK``, and
``_classify_jcor_stage`` only short-circuits on ``src_stage == dst_stage``, so a
tie between two *different* stage names is a ``stage-imports-later-stage``
violation in both directions. :func:`dispersion_f_ratio` calls
:func:`weiszfeld_medians`, and :func:`dispersion_permutation_null` recomputes
the medians for every permuted label vector, so the dependency cannot be
hoisted out by passing medians in.

The taxonomy agrees with the fence: this statistic consumes an ``(n, k)``
coordinate cloud, never a distance matrix, so it was never an ``association``
object. What *is* in :mod:`jcor.association.dispersion` is the DISCO
decomposition proper; Paper 1's ``disco.py`` composes the two, which is legal
because ``pipeline`` sits above all of ``jcor``.

Numerics are verbatim from the pre-t46 ``paper1/disco.py`` — same iteration
count, same ``1e-12`` softening inside the square roots, same jit boundaries —
because Paper 1's committed ``summary.yaml`` is a DVC output that cannot be
regenerated here.
"""

from __future__ import annotations

from functools import partial
from typing import Final

import jax
import jax.numpy as jnp

from jcor.core.random import cell_key, permutation_batch
from jcor.core.typing import ArrayLike  # noqa: TC001

__all__ = [
    "WEISZFELD_ITERATIONS",
    "dispersion_f_ratio",
    "dispersion_permutation_null",
    "weiszfeld_medians",
]

#: Fixed Weiszfeld iteration count. A fixed budget rather than a convergence
#: test keeps the routine ``jit``/``vmap``-safe: the permutation null runs it
#: once per permutation inside a single traced computation.
WEISZFELD_ITERATIONS: Final = 30

#: Softening added under every square root so the gradient and the reciprocal
#: weight stay finite when a point coincides with the running median.
_DISTANCE_EPSILON: Final = 1e-12


@partial(jax.jit, static_argnames=("n_groups", "iters"))
def weiszfeld_medians(
    coords: jnp.ndarray,
    group_codes: jnp.ndarray,
    n_groups: int,
    iters: int = WEISZFELD_ITERATIONS,
) -> jnp.ndarray:
    """Compute per-group spatial medians with vectorized Weiszfeld iterations.

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.
        iters: Fixed Weiszfeld iteration budget.

    Returns:
        Per-group spatial medians, shape ``(n_groups, k)``, initialised at the
        group centroid.

    """

    def median_for_group(g: jnp.ndarray) -> jnp.ndarray:
        mask = (group_codes == g).astype(coords.dtype)
        w_sum = jnp.sum(mask)
        init = jnp.sum(mask[:, None] * coords, axis=0) / w_sum

        def body(m: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
            diffs = coords - m[None, :]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=1) + _DISTANCE_EPSILON)
            w = mask / dists
            new_m = jnp.sum(w[:, None] * coords, axis=0) / jnp.sum(w)
            return new_m, None

        m_final, _ = jax.lax.scan(body, init, xs=None, length=iters)
        return m_final

    return jax.vmap(median_for_group)(jnp.arange(n_groups))


@partial(jax.jit, static_argnames=("n_groups",))
def dispersion_f_ratio(
    coords: jnp.ndarray, group_codes: jnp.ndarray, n_groups: int
) -> jnp.ndarray:
    """Compute the PERMDISP-style homogeneity-of-dispersion F-ratio.

    Distance from each point to its own group's spatial median, summarised by
    a one-way-ANOVA F-ratio across groups (Anderson's PERMDISP, with the
    spatial median as the centroid).

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.

    Returns:
        The F-ratio as a JAX scalar.

    """
    medians = weiszfeld_medians(coords, group_codes, n_groups)
    per_point_median = medians[group_codes]
    dist_to_med = jnp.sqrt(
        jnp.sum((coords - per_point_median) ** 2, axis=1) + _DISTANCE_EPSILON
    )

    def group_stats(g: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mask = (group_codes == g).astype(coords.dtype)
        n_g = jnp.sum(mask)
        mean_g = jnp.sum(dist_to_med * mask) / n_g
        ss_g = jnp.sum(mask * (dist_to_med - mean_g) ** 2)
        return n_g, mean_g, ss_g

    n_g_arr, mean_g_arr, ss_within_arr = jax.vmap(group_stats)(jnp.arange(n_groups))
    grand_mean = jnp.mean(dist_to_med)
    ss_between = jnp.sum(n_g_arr * (mean_g_arr - grand_mean) ** 2)
    ss_within = jnp.sum(ss_within_arr)
    n = coords.shape[0]
    return (ss_between / (n_groups - 1)) / (ss_within / (n - n_groups))


@partial(jax.jit, static_argnames=("n_groups", "n_permutations"))
def _dispersion_permutation_null_kernel(
    coords: jnp.ndarray,
    group_codes: jnp.ndarray,
    key: jax.Array,
    n_groups: int,
    n_permutations: int,
) -> jnp.ndarray:
    """Generate labels and evaluate every PERMDISP null draw in one transform."""
    perm_codes = permutation_batch(key, group_codes, n_permutations)
    return jax.vmap(dispersion_f_ratio, in_axes=(None, 0, None))(
        coords, perm_codes, n_groups
    )


def dispersion_permutation_null(
    coords: ArrayLike,
    group_codes: ArrayLike,
    n_groups: int,
    n_permutations: int,
    random_state: int,
) -> jax.Array:
    """Draw the label-permutation null of :func:`dispersion_f_ratio`.

    Coordinates are held fixed and only the labels are permuted, which is the
    exchangeability the PERMDISP null assumes.

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Observed group codes, shape ``(n,)``.
        n_groups: Number of distinct groups.
        n_permutations: Number of label permutations to draw.
        random_state: Base seed for :func:`jcor.core.random.cell_key`.

    Returns:
        Permuted F-ratios, shape ``(n_permutations,)``.

    """
    if n_permutations <= 0:
        message = "n_permutations must be positive."
        raise ValueError(message)
    key = cell_key(random_state, "geometry", "dispersion_permutation_null")
    return _dispersion_permutation_null_kernel(
        jnp.asarray(coords),
        jnp.asarray(group_codes),
        key,
        n_groups,
        n_permutations,
    )

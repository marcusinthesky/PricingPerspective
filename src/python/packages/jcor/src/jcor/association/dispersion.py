r"""S4 — DISCO: the energy-distance decomposition of total dispersion.

Position
--------
rank 4 · association. Consumes one distance matrix plus a grouping and returns
a *decomposition* — an ANOVA-style table, not a distance. That is why it sits a
stage above ``discrepancy`` rather than beside it, and it is the concept a
package named for distance *correlation* never had a home for.

The statistic
-------------
Rizzo & Székely (2010), "DISCO analysis: a nonparametric extension of analysis
of variance" — the energy-statistics-native analogue of PERMANOVA. For groups
:math:`g = 1..K` with sizes :math:`n_g`, :math:`N = \sum_g n_g`, and pairwise
distances :math:`d_{ij}`, let :math:`\bar a_{XY}` be the mean of :math:`d_{ij}`
over :math:`i \in X, j \in Y`. Within-group dispersion is
:math:`S_W = \sum_g (n_g/2)\,\bar a_{gg}`, total dispersion is
:math:`T = (N/2)\,\bar a`, and :math:`S_B = T - S_W`. Reported as the DISCO
index :math:`R^2_E = S_B / T` and the F-ratio
:math:`F = [S_B/(K-1)]\,/\,[S_W/(N-K)]`.

The ``g_alpha(A, A)`` above is a **V-statistic**: the zero diagonal is included
and the ordered sum is divided by :math:`n_g^2`, not :math:`n_g(n_g-1)`.

What is *not* here
------------------
The PERMDISP-style dispersion **companion** — distance from each point to its
group's spatial median in an embedding — lives in
:mod:`jcor.geometry.median`, and the embedding itself in
:mod:`jcor.geometry.embedding`. The t46.6 plan put all three together in this
module; the import fence forbids it, because ``association`` and ``geometry``
are rank-tied at 4 in ``JCOR_STAGE_RANK`` and a tie between two differently
named stages is a violation in both directions. See
:mod:`jcor.geometry.median` for the full argument. Paper 1's ``disco.py``
composes the two halves, which is legal in the other direction: ``pipeline``
sits above all of ``jcor``.

This module owns only the pure decomposition kernel. Label resampling and
add-one calibration live in :mod:`jcor.inference.disco`; keeping them here
would let an association-layer result bypass the hypothesis and calibration
axes of the typed inference API.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = [
    "disco_decomposition",
]


@partial(jax.jit, static_argnames=("n_groups",))
def disco_decomposition(
    dist: jnp.ndarray, group_codes: jnp.ndarray, n_groups: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Decompose total dispersion into within- and between-group parts.

    Args:
        dist: Square distance matrix, shape ``(n, n)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.

    Returns:
        ``(S_W, S_B, T, R2_E, F)`` as JAX scalars.

    """
    n = dist.shape[0]

    def group_stats(g: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mask = (group_codes == g).astype(dist.dtype)
        n_g = jnp.sum(mask)
        # The cited DISCO g_alpha(A, A) is a V-statistic: its zero diagonal is
        # included and the ordered sum is divided by n_g^2, not n_g(n_g-1).
        sum_gg = mask @ dist @ mask
        count_gg = n_g * n_g
        abar_gg = jnp.where(
            count_gg > 0, sum_gg / jnp.where(count_gg > 0, count_gg, 1.0), 0.0
        )
        return n_g, abar_gg

    n_g_arr, abar_gg_arr = jax.vmap(group_stats)(jnp.arange(n_groups))
    s_w = jnp.sum(n_g_arr / 2.0 * abar_gg_arr)

    n_total = jnp.asarray(n, dtype=dist.dtype)
    sum_all = jnp.sum(dist)
    count_all = n_total * n_total
    abar_all = sum_all / count_all
    t_total = (n_total / 2.0) * abar_all

    s_b = t_total - s_w
    k = jnp.asarray(n_groups, dtype=dist.dtype)
    f_ratio = (s_b / (k - 1.0)) / (s_w / (n_total - k))
    r2_e = s_b / t_total
    return s_w, s_b, t_total, r2_e, f_ratio

"""HAC / Newey-West long-run covariance and circular block-bootstrap indices.

Stage S5 (waist 2): everything here produces a **structured operator a
downstream model consumes** -- a symmetric long-run covariance matrix, or the
resample index set a bootstrap gathers rows with. That is the same role
``operator/covariance.py`` and ``operator/psd.py`` play, and it is what makes
``inference/gmm -> operator/longrun`` a legal S7 -> S5 edge rather than the
sibling crossing it was under the rejected subdiscipline axis.

Key distinction (ROADMAP 4.5, deep-research):

* :func:`newey_west` -- the *plain* Newey-West HAC of the moment process.  This
  is the right long-run covariance for a **Wald** test where the moments are
  centred at their own mean by construction.
* :func:`hall_centered_hac` -- Hall's (1987) **centered** HAC, which subtracts
  the *sample mean moment* ``gbar`` before forming the autocovariances.  Under
  misspecification the moments do not have mean zero, and the plain Newey-West
  Hansen-``J`` weighting matrix diverges at rate ``O(T / b_T)``; centering
  restores consistency of the ``J`` statistic.  Use this for the Hansen ``J``.

:func:`newey_west_variability` and :func:`hall_centered_variability` are the
same two estimators with that distinction carried in the **return type** rather
than in this docstring: they brand their output as
:class:`jcor.core.equations.VariabilityMatrix` at
:class:`~jcor.core.equations.PlainLongRun` and
:class:`~jcor.core.equations.MeanCenteredLongRun` respectively, which are
nominal siblings.  A consumer that needs the centered form can then demand it
instead of trusting a ``centered=True`` argument to have been passed.

Bandwidth: Newey-West (1994) / Andrews (1991) plug-in via
:func:`optimal_bandwidth` (Bartlett kernel), with a fixed override available.

Block length for :func:`circular_block_bootstrap_indices` is an S7 concern and
lives in :mod:`jcor.inference.block_length` -- this module never picks one.

The ``*_kernel`` entry points are typed JAX primitives, compiled with genuinely
shape-determining arguments (bandwidth, observation count, resample count)
static.  Block length is **not** among them: it is data-derived at every call
site, so holding it static re-keyed the compilation cache on a continuous
quantity -- see :func:`circular_block_bootstrap_indices_kernel`.  Anything a
caller computes *from the data* belongs in the traced set unless it truly
determines an output shape.  They accept PRNG keys and return device arrays, so
they compose under nested ``jit`` and leading-axis ``vmap``.  The established
eager entry points remain seed-based adapters for GMM/GMS/Sharpe compatibility,
but return JAX arrays so the public operator surface remains transformable.

Corpus keys (when PDFs packaged): Newey-West primary HAC source still open
(``newey_west_1987`` -- wrong IER PDF must not be used); Hall centered critic
still open (``hall_large_1987``); the circular / moving-block resampler is the
Politis-Romano (1992) family. Hansen ``J`` weighting context: ``hansen1982``.
"""

from __future__ import annotations

from functools import partial
from math import floor

import jax
import jax.numpy as jnp
from jax import lax

from jcor.core.equations import (  # noqa: TC001  # runtime: return brands
    EquationDescriptor,
    EstimatingFunction,
    MeanCenteredLongRun,
    PlainLongRun,
    VariabilityMatrix,
    _variability_matrix,
)
from jcor.core.random import cell_key
from jcor.core.typing import (  # noqa: TC001  # runtime numerical contract
    Array,
    ArrayLike,
    Float,
    Int,
    PRNGKey,
    Static,
)

__all__ = [
    "bartlett_weights",
    "bartlett_weights_kernel",
    "circular_block_bootstrap_indices",
    "circular_block_bootstrap_indices_kernel",
    "hall_centered_hac",
    "hall_centered_hac_kernel",
    "hall_centered_variability",
    "newey_west",
    "newey_west_kernel",
    "newey_west_variability",
    "optimal_bandwidth",
]


# ---------------------------------------------------------------------------
# Kernels & bandwidth
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("bandwidth",))
def bartlett_weights_kernel(
    bandwidth: Static[int],
) -> Float[Array, "lags"]:  # noqa: F821, UP037  # jaxtyping shape
    """Return compiled float64 Bartlett weights for a static lag count."""
    lag = jnp.arange(bandwidth + 1, dtype=jnp.float64)
    return 1.0 - lag / (bandwidth + 1.0)


def bartlett_weights(bandwidth: int) -> Float[Array, "lags"]:  # noqa: F821, UP037
    """Bartlett (triangular) kernel lag weights ``w_k = 1 - k/(b+1)``.

    Args:
        bandwidth: Truncation lag ``b`` (number of nonzero autocovariance
            lags used).  ``b = 0`` returns just the contemporaneous weight.

    Returns:
        Array of length ``bandwidth + 1`` with ``w[0] = 1``.

    """
    b = int(bandwidth)
    return bartlett_weights_kernel(b)


def optimal_bandwidth(n_observations: int) -> int:
    """Newey-West rule-of-thumb bandwidth ``floor(4*(T/100)^{2/9})``.

    Args:
        n_observations: Number of time-series observations.

    Returns:
        Non-negative integer truncation lag.

    """
    return floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0))


# ---------------------------------------------------------------------------
# HAC long-run covariance
# ---------------------------------------------------------------------------


def _hac_from_moments_kernel(
    g: Float[Array, "observations moments"],  # noqa: F722  # jaxtyping shape
    bandwidth: Static[int],
    *,
    center: Static[bool],
) -> Float[Array, "moments moments"]:  # noqa: F722  # jaxtyping shape
    """Bartlett-kernel HAC of a moment matrix ``g`` of shape ``(T, m)``.

    Args:
        g: Moment process, rows are time periods, columns moment components.
        bandwidth: Bartlett truncation lag.
        center: If ``True`` subtract the column-mean (Hall centering) before
            forming autocovariances; else use the raw moments.

    Returns:
        HAC long-run covariance estimate, shape ``(m, m)``.

    """
    moments = jnp.asarray(g)
    n_observations = moments.shape[0]
    u = moments - moments.mean(axis=0, keepdims=True) if center else moments
    time = jnp.arange(n_observations)
    lag_zero = (u.T @ u) / n_observations

    def add_lag(
        lag: Int[Array, ""],  # noqa: F722  # jaxtyping scalar
        covariance: Array,
    ) -> Array:
        """Accumulate one fixed-shape lag covariance under ``lax.fori_loop``."""
        valid = time >= lag
        lagged = u[(time - lag) % n_observations]
        current = jnp.where(valid[:, None], u, 0.0)
        lag_covariance = (current.T @ lagged) / n_observations
        weight = 1.0 - lag.astype(u.dtype) / (bandwidth + 1.0)
        return covariance + weight * (lag_covariance + lag_covariance.T)

    long_run_covariance = lax.fori_loop(
        1,
        bandwidth + 1,
        add_lag,
        lag_zero,
    )
    return 0.5 * (long_run_covariance + long_run_covariance.T)


@partial(jax.jit, static_argnames=("bandwidth",))
def newey_west_kernel(
    g: Float[Array, "observations moments"],  # noqa: F722  # jaxtyping shape
    bandwidth: Static[int | None] = None,
) -> Float[Array, "moments moments"]:  # noqa: F722  # jaxtyping shape
    """Return the transformable plain Newey-West HAC device array."""
    b = optimal_bandwidth(g.shape[0]) if bandwidth is None else bandwidth
    return _hac_from_moments_kernel(g, b, center=False)


@partial(jax.jit, static_argnames=("bandwidth",))
def hall_centered_hac_kernel(
    g: Float[Array, "observations moments"],  # noqa: F722  # jaxtyping shape
    bandwidth: Static[int | None] = None,
) -> Float[Array, "moments moments"]:  # noqa: F722  # jaxtyping shape
    """Return the transformable Hall-centered HAC device array."""
    b = optimal_bandwidth(g.shape[0]) if bandwidth is None else bandwidth
    return _hac_from_moments_kernel(g, b, center=True)


def newey_west(
    g: ArrayLike,
    bandwidth: int | None = None,
) -> Float[Array, "moments moments"]:  # noqa: F722
    """Plain Newey-West HAC long-run covariance of a moment process.

    Does NOT subtract the sample mean (the moments are assumed mean-zero, as
    for a correctly-specified Wald test).

    Args:
        g: Moment process, shape ``(T, m)``.
        bandwidth: Bartlett truncation lag; ``None`` uses
            :func:`optimal_bandwidth`.

    Returns:
        HAC covariance, shape ``(m, m)``.

    """
    b = None if bandwidth is None else int(bandwidth)
    moments = jnp.asarray(g)
    return newey_west_kernel(moments, b)


def hall_centered_hac(
    g: ArrayLike,
    bandwidth: int | None = None,
) -> Float[Array, "moments moments"]:  # noqa: F722
    """Hall (1987) **centered** HAC for the Hansen ``J`` under misspecification.

    Subtracts the sample-mean moment ``gbar`` before forming autocovariances.
    Consistent for the long-run variance of ``g_t - E[g_t]`` even when
    ``E[g_t] != 0`` (misspecified moment condition), avoiding the
    ``O(T/b_T)`` divergence of the plain Newey-West ``J`` weighting matrix.

    Args:
        g: Moment process, shape ``(T, m)``.
        bandwidth: Bartlett truncation lag; ``None`` uses
            :func:`optimal_bandwidth`.

    Returns:
        Centered HAC covariance, shape ``(m, m)``.

    """
    b = None if bandwidth is None else int(bandwidth)
    moments = jnp.asarray(g)
    return hall_centered_hac_kernel(moments, b)


# ---------------------------------------------------------------------------
# Typed meat doors (waist 2)
# ---------------------------------------------------------------------------
#
# The two functions above are the *numerics*; the two below are the same
# numerics with the fact that distinguishes them moved out of prose and into
# the return type.  Neither adds an operation: each delegates to the kernel
# directly above it and brands the result, so results are bit-identical and
# the pair costs one array wrap.
#
# That is the whole content of the waist-2 probe.  ``Omega`` is produced here
# at rank 5, consumed by a statistic at rank 7, and the rule relating them is
# stated at rank 0 in :mod:`jcor.core.equations` -- exactly the arrangement
# ``ground`` -> ``DMat`` -> ``discrepancy`` already has for waist 1.  The
# untyped ``newey_west`` / ``hall_centered_hac`` doors stay: ``inference/gmm``
# and the ``simulation`` app call them today and are migrated by their own
# node, not by this one.


def newey_west_variability[EquationA: EquationDescriptor](
    psi: EstimatingFunction[EquationA],
    bandwidth: int | None = None,
) -> VariabilityMatrix[EquationA, PlainLongRun]:
    """Brand the plain Newey-West HAC as the meat of one estimating equation.

    The equation descriptor rides through from ``psi``, so the resulting
    ``Omega`` cannot be paired with a bread from an unrelated equation, and the
    :class:`~jcor.core.equations.PlainLongRun` scheme is fixed by this
    function's return type rather than by a caller-supplied flag.

    Args:
        psi: The evaluated estimating function, shape ``(T, m)``.
        bandwidth: Bartlett truncation lag; ``None`` uses
            :func:`optimal_bandwidth`.

    Returns:
        The variability matrix at the plain, uncentered scheme.

    """
    return _variability_matrix(newey_west(psi.values, bandwidth))


def hall_centered_variability[EquationA: EquationDescriptor](
    psi: EstimatingFunction[EquationA],
    bandwidth: int | None = None,
) -> VariabilityMatrix[EquationA, MeanCenteredLongRun]:
    """Brand the Hall-centered HAC as the meat of one estimating equation.

    The scheme brand is what a Hansen ``J`` consumer should require: centering
    is not a tuning choice there but the condition under which the weighting
    matrix stays consistent under misspecification.

    Args:
        psi: The evaluated estimating function, shape ``(T, m)``.
        bandwidth: Bartlett truncation lag; ``None`` uses
            :func:`optimal_bandwidth`.

    Returns:
        The variability matrix at the mean-centered scheme.

    """
    return _variability_matrix(hall_centered_hac(psi.values, bandwidth))


# ---------------------------------------------------------------------------
# Circular block bootstrap
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=("n_observations", "n_boot"),
)
def circular_block_bootstrap_indices_kernel(
    key: PRNGKey,
    n_observations: Static[int],
    block_length: ArrayLike,
    n_boot: Static[int],
) -> Int[Array, "bootstrap observations"]:  # noqa: F722  # jaxtyping shape
    """Draw circular-block indices under a single compiled batched transform.

    ``block_length`` is deliberately **traced**, not static.  Every caller
    derives it from the data -- ``optimal_block_length`` in
    :mod:`jcor.decision._sharpe.multiple`, ``heuristic_block_length`` in
    :mod:`jcor.decision._sharpe.difference` -- so making it shape-determining
    re-keyed the compilation cache on a continuous quantity: a measured
    ``dvc repro`` spent 105 s across 261 compiles of this one kernel, 0.40 s
    each.  Keeping it traced costs one compile per ``(n_observations, n_boot)``
    pair, which are configuration, not data.

    The reformulation that allows it: output position ``i`` of a circular block
    resample is ``(starts[i // L] + i % L) % n``, so ``L`` only ever needs to
    appear in *arithmetic*.  The original built an explicit
    ``(n_blocks, L)`` grid and truncated it, which forced ``L`` into a shape.
    ``starts`` is instead drawn at the fixed upper bound ``n_observations``
    (``n_blocks = ceil(n / L) <= n`` for every ``L >= 1``), and the surplus tail
    is simply never indexed.

    Over-drawing is safe *and* exactly draw-preserving because threefry is
    counter-based: ``randint(key, (n,))[:k] == randint(key, (k,))``, verified
    directly plus 72/72 bit-identical index matrices against the previous
    static-shape implementation over ``n in {40, 105, 251, 1326}``,
    ``L in {1, 2, 3, 5, 7, 11, 17, 23, 40}``, ``n_boot in {1, 8}``.  Bootstrap
    outputs and their DVC hashes are therefore unchanged by this rewrite -- do
    not "simplify" the fixed-size draw back to ``(n_blocks,)`` without
    re-checking that, since a shorter draw would silently move every result.
    """
    if n_boot == 0:
        return jnp.empty((0, n_observations), dtype=jnp.int64)
    keys = jax.random.split(key, n_boot)
    positions = jnp.arange(n_observations, dtype=jnp.int64)
    length = jnp.asarray(block_length, dtype=jnp.int64)

    def draw_resample(
        draw_key: PRNGKey,
    ) -> Int[Array, "observations"]:  # noqa: F821, UP037  # jaxtyping shape
        starts = jax.random.randint(
            draw_key,
            shape=(n_observations,),
            minval=0,
            maxval=n_observations,
            dtype=jnp.int64,
        )
        return (starts[positions // length] + positions % length) % n_observations

    return jax.vmap(draw_resample)(keys)


def circular_block_bootstrap_indices(
    n_observations: int,
    block_length: int,
    n_boot: int,
    seed: int = 0,
) -> Int[Array, "bootstrap observations"]:  # noqa: F722
    """Circular block bootstrap resample indices (Politis-Romano 1992).

    Wraps blocks around the end of the series so every observation has equal
    inclusion probability.  Returns index arrays; the caller gathers rows of
    the panel / moment process.

    Args:
        n_observations: Length of the original series.
        block_length: Block length ``l``.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        Integer index array of shape ``(n_boot, n_observations)`` with entries
        in ``[0, n_observations)``.

    """
    if n_boot == 0:
        return jnp.empty((0, n_observations), dtype=jnp.int64)
    key = cell_key(seed, "operators", "circular_block_bootstrap_indices")
    return circular_block_bootstrap_indices_kernel(
        key,
        int(n_observations),
        int(block_length),
        int(n_boot),
    )

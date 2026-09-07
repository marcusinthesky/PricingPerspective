"""Paired Sharpe-difference significance test and the overlap-robust Sharpe CI.

Both procedures are serial-dependence aware: the difference test resamples with
the circular block bootstrap (:mod:`jcor.operators.longrun`), the CI with the
stationary bootstrap (:mod:`jcor.inference.block_length`). The Sharpe kernel is
the single one in :mod:`jcor.decision._sharpe.ratio`; ``simulation.hac``'s
private ``_sharpe_ratio`` was retired against it at exact parity (t46.8).
T48.4 then moved the numerical reductions to x64 JAX within a measured
``5e-13`` relative / ``5e-15`` absolute parity envelope.

Corpus keys to stage (author acquires; do NOT fabricate @cite):
``ledoit_wolf_2008``, ``politis_stationary_1994``, ``politis_automatic_2004``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jcor.core.typing import (  # noqa: TC001  # runtime contract
    Array,
    ArrayLike,
    Float,
    Int,
    PRNGKey,
)
from jcor.decision._sharpe.ratio import _sharpe_ratio_axis
from jcor.inference.block_length import (
    heuristic_block_length,
    heuristic_block_length_result,
    stationary_bootstrap_ci,
)
from jcor.operators.longrun import (
    circular_block_bootstrap_indices,
    circular_block_bootstrap_indices_kernel,
)

type _ReturnVector = Float[Array, "observations"]  # noqa: F821  # jaxtyping
type _BootstrapMatrix = Int[Array, "bootstrap observations"]  # noqa: F722
type _FloatScalar = Float[Array, ""]  # noqa: F722  # jaxtyping scalar


class SharpeDifferenceKernelResult(NamedTuple):
    """Array-only carrier for a keyed paired-Sharpe test."""

    statistic: _FloatScalar
    pvalue: _FloatScalar
    block_length: Int[Array, ""]  # noqa: F722


@jax.jit
def _sharpe_difference_kernel(
    returns_a: _ReturnVector,
    returns_b: _ReturnVector,
    indices: _BootstrapMatrix,
) -> tuple[_FloatScalar, _FloatScalar]:
    """Evaluate the paired test on a fixed bootstrap index matrix.

    Args:
        returns_a: First return series, shape ``(T,)``.
        returns_b: Second return series, shape ``(T,)``.
        indices: Fixed bootstrap indices, shape ``(B, T)``.

    Returns:
        Studentized Sharpe difference and two-sided bootstrap p-value.

    """
    delta_hat = _sharpe_ratio_axis(returns_a, axis=0) - _sharpe_ratio_axis(
        returns_b, axis=0
    )
    delta_boot = _sharpe_ratio_axis(returns_a[indices], axis=1) - _sharpe_ratio_axis(
        returns_b[indices], axis=1
    )
    standard_error = jnp.std(delta_boot, ddof=1)
    statistic = jnp.where(standard_error > 0.0, delta_hat / standard_error, 0.0)
    centered = delta_boot - jnp.mean(delta_boot)
    pvalue = jnp.mean(
        jnp.abs(centered) >= jnp.abs(delta_hat),
        dtype=returns_a.dtype,
    )
    return statistic, jnp.clip(pvalue, 0.0, 1.0)


@partial(jax.jit, static_argnames=("n_boot",))
def sharpe_difference_test_kernel(
    key: PRNGKey,
    returns_a: _ReturnVector,
    returns_b: _ReturnVector,
    *,
    n_boot: int,
) -> SharpeDifferenceKernelResult:
    """Run the paired-Sharpe test from an explicit bootstrap key."""
    difference = returns_a - returns_b
    block_length = heuristic_block_length_result(difference)
    indices = circular_block_bootstrap_indices_kernel(
        key,
        returns_a.shape[0],
        block_length,
        n_boot,
    )
    statistic, pvalue = _sharpe_difference_kernel(returns_a, returns_b, indices)
    return SharpeDifferenceKernelResult(statistic, pvalue, block_length)


def sharpe_difference_test(
    returns_a: ArrayLike,
    returns_b: ArrayLike,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Serial-dependence-aware paired Sharpe-difference significance test.

    Tests ``H0: SR(returns_a) == SR(returns_b)`` for two paired strategy
    return series, following the Ledoit-Wolf (2008) HAC-robust/bootstrap
    approach to Sharpe-ratio comparison: the null is assessed via a circular
    block bootstrap (Politis-Romano 1992) over the *paired* observations
    (preserving cross-series dependence and each series' serial dependence),
    with the block length chosen from
    :func:`jcor.inference.block_length.heuristic_block_length` applied to the
    return differential ``d_t = a_t - b_t`` unless overridden.

    Args:
        returns_a: Strategy-A return series, shape ``(T,)``.
        returns_b: Strategy-B return series, shape ``(T,)``, paired with
            ``returns_a`` (same dates).
        n_boot: Number of circular-block-bootstrap resamples.
        block_length: Block length ``l``; ``None`` uses the first-lag
            heuristic on the differential series.
        seed: RNG seed for the bootstrap.

    Returns:
        Tuple ``(stat, pvalue)`` where ``stat`` is the studentized statistic
        ``dSR / se_boot`` (``dSR = SR(a) - SR(b)``, ``se_boot`` the bootstrap
        standard deviation of ``dSR``) and ``pvalue`` is the two-sided
        bootstrap p-value for ``H0: dSR = 0``.

    Raises:
        ValueError: If the two return series have different shapes.

    """
    a = jnp.asarray(returns_a, dtype=jnp.float64).ravel()
    b = jnp.asarray(returns_b, dtype=jnp.float64).ravel()
    if a.shape != b.shape:
        message = "returns_a and returns_b must have the same shape"
        raise ValueError(message)
    n_observations = a.size
    difference = a - b
    ell = (
        heuristic_block_length(jax.device_get(difference))
        if block_length is None
        else int(block_length)
    )
    ell = max(1, min(ell, n_observations))

    indices = jnp.asarray(
        circular_block_bootstrap_indices(
            n_observations,
            ell,
            n_boot,
            seed=seed,
        ),
        dtype=jnp.int64,
    )
    statistic, pvalue = _sharpe_difference_kernel(a, b, indices)
    return float(statistic), float(pvalue)


def sharpe_ratio_ci(
    returns: ArrayLike,
    confidence_level: float = 0.95,
    num_bootstrap: int = 999,
    seed: int = 0,
) -> tuple[float, float]:
    """Stationary-bootstrap CI for the per-observation Sharpe ratio.

    Thin, parity-preserving wrapper: routes through
    :func:`jcor.inference.block_length.stationary_bootstrap_ci` with the
    Politis-White automatic block length (its default), so overlapping-window
    autocorrelation is handled by the block bootstrap rather than an iid
    assumption.

    Args:
        returns: Return series, shape ``(n,)``.
        confidence_level: CI level.
        num_bootstrap: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(low, high)`` percentile CI for the Sharpe ratio.

    """

    def _sr(z: jnp.ndarray) -> jnp.ndarray:
        col = z[:, 0] if z.ndim > 1 else z
        sd = jnp.std(col, ddof=1)
        return jnp.where(sd > 0, jnp.mean(col) / sd, 0.0)

    values = jnp.asarray(returns, dtype=jnp.float64).ravel()
    return stationary_bootstrap_ci(
        values,
        _sr,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )

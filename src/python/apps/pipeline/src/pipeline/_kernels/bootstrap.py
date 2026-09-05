"""Shared circular block-bootstrap index helper.

Extracted from the identical ~5-line snippet duplicated across
``pipeline.stages.substrate.w2_exposure_frontier``, and
the energy-robustness stage: pick a default block length from the
equal-weight portfolio return series via
``jcor.inference.block_length.optimal_block_length_numpy`` (Politis-White automatic
selection), then draw circular block-bootstrap resample indices via
``jcor.operators.longrun.circular_block_bootstrap_indices``. Behavior-preserving --
same defaults, same rounding (``max(1, round(...))``), same seed handling.
"""

from __future__ import annotations

import jax
import numpy as np
from jcor.inference.block_length import optimal_block_length_numpy
from jcor.operators.longrun import circular_block_bootstrap_indices


def bootstrap_indices(
    returns: np.ndarray,
    n_boot: int,
    seed: int,
    block_length: int | None = None,
) -> tuple[np.ndarray, int]:
    """Circular block-bootstrap resample indices over a ``(T, N)`` return panel.

    Args:
        returns: ``(T, N)`` return panel; only used to derive ``T`` and, when
            ``block_length`` is None, the equal-weight portfolio series for
            automatic block-length selection.
        n_boot: Number of bootstrap replications.
        seed: RNG seed passed through to
            ``jcor.operators.longrun.circular_block_bootstrap_indices``.
        block_length: Circular block length; if None, selected via
            ``optimal_block_length_numpy`` on the equal-weighted portfolio
            return series (Politis-White).

    Returns:
        A tuple ``(idx, block_length)`` where ``idx`` is the
        ``(n_boot, T)`` resample index array and ``block_length`` is the
        (possibly auto-selected) block length used.

    """
    t_obs = returns.shape[0]
    # The block-length estimator is a float64 NumPy host door; the JAX index
    # result is materialized here because this helper returns a NumPy artifact.
    with jax.enable_x64(new_val=True):
        if block_length is None:
            equal_weight = returns.mean(axis=1)
            block_length = max(1, round(optimal_block_length_numpy(equal_weight)))
        idx = circular_block_bootstrap_indices(t_obs, block_length, n_boot, seed=seed)
    return np.asarray(idx), block_length

"""Shared eager calibration rules for inferential procedures."""

from __future__ import annotations

import jax.numpy as jnp

from jcor.core.typing import ArrayLike  # noqa: TC001

__all__ = ["add_one_p", "add_one_p_kernel"]


def add_one_p_kernel(observed: ArrayLike, null: ArrayLike) -> jnp.ndarray:
    """Right-tail Monte-Carlo p-value with add-one correction, transformable.

    The retired NumPy body filtered the reference set with a boolean mask
    (``null[np.isfinite(null)]``), whose output shape depends on the data and
    therefore cannot be traced. This form is shape-stable: non-finite draws are
    masked out of both the exceedance count and the valid-draw count instead of
    being removed, which is arithmetically identical and jit/vmap-safe.

    Args:
        observed: Observed statistic.
        null: Reference draws of any shape; flattened before counting.

    Returns:
        The p-value as a JAX scalar, or ``nan`` when the observed statistic is
        undefined or no reference draw is finite.

    """
    statistic = jnp.asarray(observed)
    draws = jnp.asarray(null).ravel()
    finite = jnp.isfinite(draws)
    n_valid = jnp.count_nonzero(finite)
    exceed = jnp.count_nonzero(finite & (draws >= statistic))
    pvalue = (1 + exceed) / (n_valid + 1)
    defined = jnp.isfinite(statistic) & (n_valid > 0)
    return jnp.where(defined, pvalue, jnp.nan)


def add_one_p(observed: float, null: ArrayLike) -> float:
    """Return a right-tail Monte-Carlo p-value with add-one correction.

    Non-finite draws are excluded from the reference set. An undefined observed
    statistic or an empty valid reference set returns ``nan``.

    This is the host adapter over :func:`add_one_p_kernel`; it exists because
    callers in ``apps/pipeline`` put the result straight into a result record.
    Call the kernel directly from anything that needs to stay transformable.
    """
    return float(add_one_p_kernel(observed, null))

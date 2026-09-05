"""The single Sharpe-ratio kernel every other module in this package calls.

t46.8 collapsed two implementations into this one:

* ``simulation.hac._sharpe_ratio`` (retired) -- ``mean/std(ddof=1)`` with no
  sample-size guard, so a length-1 series emitted a ``RuntimeWarning`` before
  falling through to ``0.0``.
* ``simulation.sharpe_tests.sharpe_ratio`` (survives) -- identical arithmetic
  with an explicit ``size > 1`` guard, hence a strict superset.

Measured parity of the collapse: exact (relative difference ``0.0`` over 200
seeded random series, and identical returns on the constant, ``n = 1`` and
``n = 2`` degenerate cases). The retired body was never numerically distinct.
T48.4 subsequently moved the survivor to x64 JAX: generic seeded gaps are at
most one ulp, while exactly constant decimal arrays now return the documented
``0.0`` instead of amplifying NumPy's residual standard deviation. An
ill-conditioned ``0.3 +/- 1e-12`` stress fixture remains nonzero, with a
measured ``1.54e-9`` relative gap after division by its near-zero deviation.

:func:`_sharpe_ratio_axis` is the compiled vectorised form the two bootstrap
procedures share. Precision is the caller's: jcor neither changes the
process-global flag nor opens a scope of its own, so the ``float64`` requests
below resolve to float32 unless the caller owns ``jax_enable_x64=True``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from jcor.core.typing import Array, ArrayLike, Static  # noqa: TC001  # runtime


def sharpe_ratio(returns: ArrayLike) -> float:
    """Per-observation sample Sharpe ratio ``mean / std`` (ddof=1).

    Args:
        returns: 1-D return series.

    Returns:
        Sharpe ratio (0.0 if the series has non-positive dispersion).

    """
    r = jnp.asarray(returns, dtype=jnp.float64).ravel()
    sd = jnp.std(r, ddof=1) if r.size > 1 else jnp.asarray(0.0)
    value = jnp.where(sd > 0.0, jnp.mean(r) / sd, 0.0)
    return float(value)


@partial(jax.jit, static_argnames=("axis",))
def _sharpe_ratio_axis(x: ArrayLike, axis: Static[int]) -> Array:
    """Vectorised Sharpe ratio along ``axis`` (0.0 where dispersion vanishes).

    Args:
        x: 2-D array of return series.
        axis: Time axis to reduce over.

    Returns:
        Sharpe ratios with ``axis`` removed.

    """
    values = jnp.asarray(x, dtype=jnp.float64)
    mean = jnp.mean(values, axis=axis)
    sd = jnp.std(values, axis=axis, ddof=1)
    return jnp.where(sd > 0.0, mean / sd, 0.0)


def _skew_kurtosis(returns: ArrayLike) -> tuple[float, float]:
    """Sample skewness and NON-excess kurtosis (normal -> kurtosis 3).

    Args:
        returns: 1-D return series.

    Returns:
        ``(skew, kurtosis)`` with kurtosis on the non-excess scale.

    """
    r = jnp.asarray(returns, dtype=jnp.float64).ravel()
    centered = r - jnp.mean(r)
    second = jnp.mean(centered**2)
    skew = jnp.mean(centered**3) / second**1.5
    kurt = jnp.mean(centered**4) / second**2
    return float(skew), float(kurt)

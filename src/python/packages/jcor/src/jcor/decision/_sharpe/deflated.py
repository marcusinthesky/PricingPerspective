"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

The Probabilistic Sharpe Ratio benchmarked against the *expected maximum*
Sharpe under a null of ``n_trials`` independent zero-skill trials, correcting
for non-normality (skew/kurtosis) and short samples. Answers "is this Sharpe
significant AFTER accounting for how many configs were tried?".

Corpus key to stage (author acquires; do NOT fabricate @cite):
``bailey_deflated_2014``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.scipy.stats import norm

from jcor.core.typing import Array, ArrayLike  # noqa: TC001  # runtime contract
from jcor.decision._sharpe.ratio import _skew_kurtosis, sharpe_ratio

_EULER_MASCHERONI = 0.5772156649015329
_MINIMUM_SAMPLE_SIZE = 2


@jax.jit
def _probabilistic_sharpe_ratio_kernel(
    sr_hat: Array,
    n_obs: Array,
    skew: Array,
    kurtosis: Array,
    benchmark_sr: Array,
) -> Array:
    """Evaluate the PSR normal CDF without leaving the JAX graph.

    Args:
        sr_hat: Estimated Sharpe ratio.
        n_obs: Number of observations, represented as a scalar array.
        skew: Biased sample skewness.
        kurtosis: Biased, non-excess sample kurtosis.
        benchmark_sr: Sharpe benchmark under the null.

    Returns:
        Scalar PSR probability.

    """
    denominator = jnp.sqrt(1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat**2)
    z = (sr_hat - benchmark_sr) * jnp.sqrt(n_obs - 1.0) / denominator
    return norm.cdf(z)


def probabilistic_sharpe_ratio(
    sr_hat: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    benchmark_sr: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio ``PSR(SR*)`` (Bailey & Lopez de Prado 2014).

    ``PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1) /
    sqrt(1 - g3 * SR_hat + ((g4 - 1) / 4) * SR_hat^2) )`` -- the probability that
    the true (per-observation) Sharpe exceeds the benchmark ``SR*``, adjusting
    for skew ``g3`` and NON-excess kurtosis ``g4`` and the sample length ``n``.

    Args:
        sr_hat: Estimated per-observation Sharpe ratio.
        n_obs: Number of return observations (``>= 2``).
        skew: Sample skewness of the returns.
        kurtosis: Sample NON-excess kurtosis (normal -> 3.0).
        benchmark_sr: Benchmark Sharpe ``SR*`` (per observation; default 0).

    Returns:
        ``PSR`` in ``[0, 1]``.

    Raises:
        ValueError: If ``n_obs`` is below two.

    """
    if n_obs < _MINIMUM_SAMPLE_SIZE:
        message = f"n_obs must be >= 2, got {n_obs}"
        raise ValueError(message)
    denom_var = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat * sr_hat
    if denom_var <= 0.0:
        # Degenerate estimation variance; fall back to a point verdict.
        return 1.0 if sr_hat > benchmark_sr else 0.0
    value = _probabilistic_sharpe_ratio_kernel(
        jnp.asarray(sr_hat, dtype=jnp.float64),
        jnp.asarray(n_obs, dtype=jnp.float64),
        jnp.asarray(skew, dtype=jnp.float64),
        jnp.asarray(kurtosis, dtype=jnp.float64),
        jnp.asarray(benchmark_sr, dtype=jnp.float64),
    )
    return float(value)


def expected_maximum_sharpe_ratio(n_trials: int, sr_variance: float) -> float:
    """Estimate the maximum Sharpe over ``n_trials`` zero-skill trials.

    ``E[max SR] ~ sqrt(V) * [ (1 - gamma) * Phi^{-1}(1 - 1/N)
    + gamma * Phi^{-1}(1 - 1/(N e)) ]`` with ``gamma`` the Euler-Mascheroni
    constant, ``V`` the cross-trial variance of the estimated Sharpe ratios and
    ``N = n_trials``. This is the deflation benchmark ``SR0``: the Sharpe you
    would expect the best of ``N`` truly-worthless strategies to post by luck.

    Args:
        n_trials: Number of independent trials/configurations tried (``>= 1``).
        sr_variance: Variance of the estimated Sharpe ratios across trials.

    Returns:
        The expected maximum Sharpe ``SR0`` (0.0 if ``N == 1`` or ``V == 0``).

    Raises:
        ValueError: If ``n_trials`` is below one.

    """
    if n_trials < 1:
        message = f"n_trials must be >= 1, got {n_trials}"
        raise ValueError(message)
    if n_trials == 1 or sr_variance <= 0.0:
        return 0.0
    n = float(n_trials)
    gamma = _EULER_MASCHERONI
    q1 = ndtri(jnp.asarray(1.0 - 1.0 / n, dtype=jnp.float64))
    q2 = ndtri(jnp.asarray(1.0 - 1.0 / (n * math.e), dtype=jnp.float64))
    value = jnp.sqrt(sr_variance) * ((1.0 - gamma) * q1 + gamma * q2)
    return float(value)


class DeflatedSharpeResult(NamedTuple):
    """Result of a Deflated-Sharpe computation.

    Attributes:
        sr_hat: Estimated per-observation Sharpe ratio.
        n_obs: Number of return observations.
        skew: Sample skewness.
        kurtosis: Sample NON-excess kurtosis.
        n_trials: Number of trials the deflation accounts for.
        sr0: Expected-maximum-Sharpe deflation benchmark.
        psr_zero: PSR against a zero benchmark (undeflated significance).
        deflated_sharpe: PSR against ``sr0`` -- the Deflated Sharpe Ratio.

    """

    sr_hat: float
    n_obs: int
    skew: float
    kurtosis: float
    n_trials: int
    sr0: float
    psr_zero: float
    deflated_sharpe: float


def deflated_sharpe_ratio(
    returns: ArrayLike,
    n_trials: int,
    sr_variance_across_trials: float | None = None,
    benchmark_sr: float = 0.0,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio of a return series (Bailey & Lopez de Prado 2014).

    Computes ``SR_hat``, skew, kurtosis and ``n`` from ``returns``; the
    deflation benchmark ``SR0 = E[max SR]`` over ``n_trials`` from
    :func:`expected_maximum_sharpe_ratio`; and ``DSR = PSR(SR0)`` -- the
    probability the strategy's true Sharpe exceeds what the best of
    ``n_trials`` worthless trials would post by luck.

    Args:
        returns: Strategy return series, shape ``(n,)``.
        n_trials: Number of configurations tried (the snooping breadth). Must
            be ``>= 1``. ``1`` means no deflation (``DSR == PSR(benchmark)``).
        sr_variance_across_trials: Cross-trial variance ``V`` of the estimated
            Sharpe ratios. Required when ``n_trials > 1`` (there is no way to
            deflate without the trial dispersion); ignored when ``n_trials == 1``.
        benchmark_sr: Benchmark used for the undeflated ``psr_zero`` and, when
            ``n_trials == 1``, for the deflated verdict.

    Returns:
        A :class:`DeflatedSharpeResult`.

    Raises:
        ValueError: If ``n_trials > 1`` without a cross-trial Sharpe variance.

    """
    r = jnp.asarray(returns, dtype=jnp.float64).ravel()
    n_obs = int(r.size)
    sr_hat = sharpe_ratio(r)
    skew, kurt = _skew_kurtosis(r)

    if n_trials > 1 and sr_variance_across_trials is None:
        message = (
            "sr_variance_across_trials is required when n_trials > 1 "
            "(the deflation benchmark needs the cross-trial Sharpe dispersion)."
        )
        raise ValueError(message)
    sr0 = (
        benchmark_sr
        if n_trials == 1
        else expected_maximum_sharpe_ratio(
            n_trials, float(sr_variance_across_trials or 0.0)
        )
    )
    psr_zero = probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurt, benchmark_sr)
    dsr = probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurt, sr0)
    return DeflatedSharpeResult(
        sr_hat=sr_hat,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt,
        n_trials=n_trials,
        sr0=sr0,
        psr_zero=psr_zero,
        deflated_sharpe=dsr,
    )

"""Block-length selection for dependent-series resampling (stage S7).

Two selectors live here and they are **different estimators, not two
implementations of one** -- t46.8 measured them 3.4%-710% apart on the existing
seeded fixtures. :func:`optimal_block_length` is the Politis-White (2004)
flat-top lag-window selector with the Patton-Politis-White (2009) variance
constant, and the repository's only Politis-White body
(:func:`optimal_block_length_numpy` is a boundary adapter delegating to it); use
it for production CIs. :func:`heuristic_block_length` is the cheap
``T^{1/3}(1 + 2|rho_1|)`` first-lag rule backing the paired Sharpe-difference
bootstrap only.

Scheme constants (same g-hat / G-hat(0) pipeline; pick the D that matches the
resampler): ``stationary`` -> ``D_SB = 2 g^2(0)`` (Patton-Politis-White SB
correction); ``circular`` -> ``D_MB = (4/3) g^2(0)`` (fixed / circular /
moving-block). Do **not** pair a stationary-bootstrap length with circular block
bootstrap or the reverse: the SB/MB D ratio is ``(3/2)^{1/3} ~ 1.14`` on
``b_opt``.

Corpus: ``politis_automatic_2004``, ``patton_correction_2009``; and
``politis_stationary_1994`` for the :func:`stationary_bootstrap_ci` theory.
"""

from __future__ import annotations

import math
from collections.abc import Callable  # runtime type alias
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax, random

from jcor.core.precision import require_x64
from jcor.core.random import cell_key
from jcor.core.results import EstimateResult
from jcor.core.typing import (  # noqa: TC001  # runtime numerical contract
    Array,
    ArrayLike,
    Float,
    Int,
    PRNGKey,
    Static,
)

BlockLengthScheme = Literal["stationary", "circular"]

__all__ = [
    "BlockLengthScheme",
    "heuristic_block_length",
    "heuristic_block_length_result",
    "optimal_block_length",
    "optimal_block_length_numpy",
    "optimal_block_length_result",
    "stationary_bootstrap_ci",
    "stationary_bootstrap_estimate",
]

type BlockStatistic = Callable[[Array], Float[Array, ""]]  # noqa: F722
type EagerBlockStatistic = Callable[[Array], float | Array]

_MIN_BLOCK_SERIES = 2


def _validate_scheme(scheme: BlockLengthScheme) -> None:
    """Validate a static bootstrap-family selector."""
    if scheme not in ("stationary", "circular"):
        message = f"scheme must be 'stationary' or 'circular', got {scheme!r}"
        raise ValueError(message)


def _validate_bootstrap_config(confidence_level: float, num_bootstrap: int) -> None:
    """Validate static stationary-bootstrap report configuration."""
    if not 0.0 < confidence_level < 1.0:
        message = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(message)
    if num_bootstrap < 1:
        message = f"num_bootstrap must be positive, got {num_bootstrap}"
        raise ValueError(message)


def _lag_autocorrelations(z: Array, sample_count: int, max_lag_value: int) -> Array:
    """Return the FFT autocorrelations of a centered series up to ``max_lag``.

    Vectorized ACF via FFT (O(T log T) instead of O(T*max_lag)).

    Args:
        z: Mean-centered 1-D series, shape ``(T,)``.
        sample_count: Static number of observations ``T``.
        max_lag_value: Static (already clipped) cap on the lag search.

    Returns:
        Autocorrelations at lags ``0 .. max_lag_value``.

    """
    variance = jnp.sum(z * z)
    rho0 = jnp.where(variance > 0, 1.0, 0.0)
    fft_len = 2 ** math.ceil(math.log2(2 * sample_count))
    fz = jnp.fft.rfft(z, n=fft_len)
    ac_full = jnp.fft.irfft(fz * jnp.conj(fz), n=fft_len)[:sample_count].real
    denom = jnp.where(variance > 0, variance, 1.0)
    rho_all = jnp.concatenate([jnp.array([rho0]), ac_full[1:sample_count] / denom])
    return rho_all[: max_lag_value + 1]


def _flat_top_bandwidth(rho: Array, sample_count: int, max_lag_value: int) -> Array:
    """Return the flat-top lag-window bandwidth ``2*m_hat`` (capped at max lag).

    ``m_hat`` is the smallest lag past which ``|rho|`` is negligible, i.e. the
    first lag whose next ``K_N ~ 5`` correlations all fall inside the
    significance band ``~ 2/sqrt(T)`` (Politis-White heuristic). The
    candidate/window arrays have static max-lag shapes; missing tail positions
    are neutral ``True`` values, matching the retired shorter slices.

    Args:
        rho: Autocorrelations at lags ``0 .. max_lag_value``.
        sample_count: Static number of observations ``T``.
        max_lag_value: Static (already clipped) cap on the lag search.

    Returns:
        Scalar bandwidth array.

    """
    band = 2.0 / jnp.sqrt(sample_count)
    small = jnp.abs(rho[1:]) < band
    if max_lag_value > 1:
        starts = jnp.arange(max_lag_value - 1)[:, None]
        positions = starts + jnp.arange(5)[None, :]
        valid = positions < max_lag_value
        clipped = jnp.minimum(positions, max_lag_value - 1)
        windows = jnp.where(
            valid,
            small[clipped],
            jnp.ones_like(small[clipped], dtype=bool),
        )
        all_small = jnp.all(windows, axis=1)
        first_small = jnp.argmax(all_small) + 1
        m_hat = jnp.where(jnp.any(all_small), first_small, max_lag_value)
    else:
        m_hat = jnp.asarray(max_lag_value, dtype=jnp.int32)
    return jnp.minimum(2 * m_hat, max_lag_value)


def _flat_top_lag_moments(
    rho: Array,
    bandwidth: Array,
    max_lag_value: int,
) -> tuple[Array, Array]:
    """Return the flat-top-weighted ``g_hat`` and ``G_hat(0)`` lag sums.

    The lag grid stays fixed at ``2*max_lag+1`` and is masked outside the
    selected dynamic bandwidth, so no data-dependent array shape enters the
    trace.

    Args:
        rho: Autocorrelations at lags ``0 .. max_lag_value``.
        bandwidth: Scalar flat-top bandwidth.
        max_lag_value: Static (already clipped) cap on the lag search.

    Returns:
        Tuple of ``g_hat`` and the spectral density at zero.

    """
    lags = jnp.arange(-max_lag_value, max_lag_value + 1)
    absolute_lags = jnp.abs(lags)
    rho_full = rho[absolute_lags]
    # Flat-top (trapezoidal) weights.
    safe_bandwidth = jnp.maximum(bandwidth, 1)
    taper = 2.0 - 2.0 * absolute_lags / safe_bandwidth
    weights = jnp.where(
        absolute_lags > bandwidth,
        0.0,
        jnp.where(
            2 * absolute_lags <= bandwidth,
            1.0,
            taper,
        ),
    )
    weights = jnp.clip(weights, 0.0, 1.0)
    return jnp.sum(weights * absolute_lags * rho_full), jnp.sum(weights * rho_full)


@partial(jax.jit, static_argnames=("max_lag", "scheme"))
def optimal_block_length_result(
    series: Float[Array, "observations"],  # noqa: F821, UP037  # jaxtyping shape
    max_lag: Static[int | None] = None,
    scheme: Static[BlockLengthScheme] = "stationary",
) -> Float[Array, ""]:  # noqa: F722  # jaxtyping scalar
    """Return the transformable Politis-White/Patton block-length scalar.

    Politis & White (2004) flat-top lag-window selector WITH the
    Patton-Politis-White (2009) variance constant for the chosen bootstrap:

    - ``scheme="stationary"``: ``D_SB = 2 g²(0)`` (default; stationary bootstrap)
    - ``scheme="circular"``: ``D_MB = (4/3) g²(0)`` (fixed / circular / moving block)

    SCOPED to non-degenerate functionals (e.g. correlation-gap / Sharpe CI);
    NOT for calibrating the degenerate energy null.

    Args:
        series: 1-D time series, shape (T,).
        max_lag: Optional cap on the lag search (default ~ ``ceil(sqrt(T)) + ...``).
        scheme: Bootstrap family for the D constant (default stationary).

    Returns:
        Scalar JAX array containing the optimal expected (or fixed) block
        length ``b_opt >= 1``.

    Raises:
        ValueError: If ``series`` has fewer than two observations or ``scheme``
            is not recognized.

    """
    _validate_scheme(scheme)

    z = jnp.ravel(series)
    sample_count = z.shape[0]
    if sample_count < _MIN_BLOCK_SERIES:
        message = "series must contain at least two observations"
        raise ValueError(message)
    z = z - jnp.mean(z)
    if max_lag is None:
        max_lag_value = math.ceil(math.sqrt(sample_count)) + math.ceil(
            math.log10(sample_count + 1)
        )
    else:
        max_lag_value = max_lag
    max_lag_value = max(0, min(max_lag_value, sample_count - 1))

    rho = _lag_autocorrelations(z, sample_count, max_lag_value)
    bandwidth = _flat_top_bandwidth(rho, sample_count, max_lag_value)
    g_hat, spectral_density_zero = _flat_top_lag_moments(
        rho,
        bandwidth,
        max_lag_value,
    )
    # Patton D: SB uses 2 G0²; fixed/circular/moving block uses (4/3) G0².
    variance_constant = (
        (4.0 / 3.0) * spectral_density_zero**2
        if scheme == "circular"
        else 2.0 * spectral_density_zero**2
    )
    variance_constant = jnp.maximum(variance_constant, 1e-8)
    b_opt = (2.0 * g_hat**2 / variance_constant) ** (1.0 / 3.0) * sample_count ** (
        1.0 / 3.0
    )
    return jnp.maximum(jnp.abs(b_opt), 1.0)


def optimal_block_length(
    series: ArrayLike,
    max_lag: int | None = None,
    scheme: BlockLengthScheme = "stationary",
) -> float:
    """Return the eager Python-scalar adapter for automatic block length.

    Transformable callers should use :func:`optimal_block_length_result` and
    retain its scalar JAX array on device.

    Args:
        series: One-dimensional numerical time series.
        max_lag: Optional static cap on the lag search.
        scheme: Bootstrap family for the Patton variance constant.

    Returns:
        Optimal expected (or fixed) block length as a Python float.

    """
    result = optimal_block_length_result(
        jnp.asarray(series),
        max_lag=max_lag,
        scheme=scheme,
    )
    return float(result)


def optimal_block_length_numpy(
    series: ArrayLike,
    max_lag: int | None = None,
    scheme: BlockLengthScheme = "stationary",
) -> float:
    """NumPy-boundary adapter for :func:`optimal_block_length`.

    Converts a float series to a JAX array, runs the full PW+Patton selector,
    and returns a Python float. For pipelines that prefer not to thread JAX
    arrays through hot NumPy paths.

    Args:
        series: 1-D numerical series.
        max_lag: Optional lag cap (forwarded).
        scheme: Bootstrap family for the D constant (forwarded).

    Returns:
        Optimal block length ``b_opt`` (float, >= 1).

    """
    require_x64("inference.block_length.optimal_block_length_numpy")
    return float(
        optimal_block_length_result(
            jnp.asarray(series, dtype=jnp.float64),
            max_lag=max_lag,
            scheme=scheme,
        )
    )


@jax.jit
def heuristic_block_length_result(
    x: Float[Array, "observations"],  # noqa: F821, UP037  # jaxtyping shape
) -> Int[Array, ""]:  # noqa: F722  # jaxtyping scalar
    """Return the transformable first-lag heuristic as an integer JAX scalar.

    Args:
        x: On-device univariate series.

    Returns:
        Positive integer block length.

    """
    values = jnp.ravel(x)
    n_observations = values.shape[0]
    centered = values - jnp.mean(values)
    denominator = jnp.sum(centered**2)
    rho1 = jnp.where(
        denominator > 0,
        jnp.sum(centered[1:] * centered[:-1]) / denominator,
        0.0,
    )
    base = n_observations ** (1.0 / 3.0)
    scale = 1.0 + 2.0 * jnp.abs(rho1)
    return jnp.maximum(1, jnp.rint(base * scale)).astype(jnp.int32)


def heuristic_block_length(x: ArrayLike) -> int:
    """Return the eager float64 adapter for the first-lag heuristic.

    Uses ``l* ~ T^{1/3} (1 + 2|rho_1|)``. This is **not** the Politis-White
    spectral selector and never converges to it -- the two sit 3.4%-710% apart
    on this repository's seeded fixtures (t46.8). It is the cheap rule
    :func:`jcor.decision.sharpe.sharpe_difference_test` is calibrated against;
    prefer :func:`optimal_block_length` for production CIs.

    Transformable callers should use :func:`heuristic_block_length_result`;
    this compatibility adapter performs the one intentional scalar transfer.

    Args:
        x: Univariate series, shape ``(T,)``.

    Returns:
        Positive integer block length.

    """
    require_x64("inference.block_length.heuristic_block_length")
    result = heuristic_block_length_result(
        jnp.asarray(x, dtype=jnp.float64),
    )
    return int(result)


@partial(
    jax.jit,
    static_argnames=("statistic_fn", "confidence_level", "num_bootstrap"),
)
def stationary_bootstrap_estimate(
    z: Float[Array, "observations ..."],  # noqa: F722  # jaxtyping shape
    statistic_fn: Static[BlockStatistic],
    key: PRNGKey,
    block_length: Array | float,
    confidence_level: Static[float] = 0.95,
    num_bootstrap: Static[int] = 999,
) -> EstimateResult[Float[Array, ""]]:  # noqa: F722  # jaxtyping scalar
    """Return the transformable stationary-bootstrap estimate pytree.

    Args:
        z: On-device series, shape ``(T,)`` or ``(T, p)``.
        statistic_fn: Traceable scalar statistic over a resampled series.
        key: Typed JAX PRNG key; no seed conversion occurs in this core.
        block_length: Expected block length, as a scalar array or Python value.
        confidence_level: Static percentile interval level.
        num_bootstrap: Static number of resamples.

    Returns:
        Registered estimate pytree with the observed statistic, bootstrap
        standard error, percentile interval, convergence flag, and static
        observation count.

    """
    _validate_bootstrap_config(confidence_level, num_bootstrap)
    sample_count = z.shape[0]
    transition_probability = 1.0 / jnp.maximum(
        jnp.asarray(block_length, dtype=z.dtype),
        1.0,
    )

    def resample(resample_key: PRNGKey) -> Float[Array, ""]:  # noqa: F722
        """Evaluate the statistic on one stationary-bootstrap resample."""
        key_initial, key_scan = random.split(resample_key)
        start = random.randint(key_initial, (), 0, sample_count)

        def step(
            previous: Int[Array, ""],  # noqa: F722  # jaxtyping scalar
            scan_key: PRNGKey,
        ) -> tuple[Int[Array, ""], Int[Array, ""]]:  # noqa: F722
            key_jump, key_index = random.split(scan_key)
            jump = random.uniform(key_jump, ()) < transition_probability
            new_index = random.randint(key_index, (), 0, sample_count)
            next_index = jnp.where(
                jump,
                new_index,
                (previous + 1) % sample_count,
            )
            return next_index, next_index

        scan_keys = random.split(key_scan, sample_count - 1)
        _, tail = lax.scan(step, start, scan_keys)
        indices = jnp.concatenate([jnp.reshape(start, (1,)), tail])
        return jnp.asarray(statistic_fn(z[indices]))

    keys = random.split(key, num_bootstrap)
    distribution: Float[
        Array,
        "resamples",  # noqa: F821, UP037  # jaxtyping shape
    ] = jax.vmap(resample)(keys)
    alpha = 1.0 - confidence_level
    lower = jnp.percentile(distribution, 100 * alpha / 2)
    upper = jnp.percentile(distribution, 100 * (1.0 - alpha / 2))
    ddof = 1 if num_bootstrap > 1 else 0
    return EstimateResult(
        estimate=jnp.asarray(statistic_fn(z)),
        se=jnp.std(distribution, ddof=ddof),
        ci=(lower, upper),
        converged=jnp.ones((), dtype=bool),
        n_obs=sample_count,
    )


def stationary_bootstrap_ci(
    z: ArrayLike,
    statistic_fn: EagerBlockStatistic,
    block_length: float | None = None,
    confidence_level: float = 0.95,
    num_bootstrap: int = 999,
    seed: int | None = None,
) -> tuple[float, float]:
    """Stationary-bootstrap CI for a functional of a dependent series.

    Politis-Romano (1994) stationary bootstrap: random geometric-length blocks
    (mean ``block_length``), so the resampled series is itself stationary; valid
    under weak dependence. SCOPED to the correlation-gap effect size (a
    non-degenerate functional). Block length defaults to
        :func:`optimal_block_length_result` on the first column of ``z``.

    Args:
        z: Series, shape (T,) or (T, p) (rows are time steps).
        statistic_fn: Maps a resampled (T,...) block series to a scalar (e.g. the
            weighted-minus-unweighted correlation gap).
        block_length: Expected block length; default via
            :func:`optimal_block_length_result`.
        confidence_level: CI level.
        num_bootstrap: Resamples.
        seed: RNG seed.

    Returns:
        (low, high) percentile CI.

    """
    _validate_bootstrap_config(confidence_level, num_bootstrap)
    values = jnp.asarray(z)
    if block_length is None:
        column = values[:, 0] if values.ndim > 1 else values
        selected_block_length: Array | float = optimal_block_length_result(column)
    else:
        selected_block_length = block_length

    def array_statistic(sample: Array) -> Float[Array, ""]:  # noqa: F722
        """Narrow an eager scalar statistic to the JAX numerical contract."""
        return jnp.asarray(statistic_fn(sample))

    result = stationary_bootstrap_estimate(
        values,
        array_statistic,
        cell_key(
            seed if seed is not None else 0,
            "inference",
            "stationary_bootstrap_ci",
        ),
        selected_block_length,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
    )
    return float(result.ci[0]), float(result.ci[1])

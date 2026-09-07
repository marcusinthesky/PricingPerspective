"""Familywise multiple Sharpe test: Romano-Wolf stepwise + White Reality Check.

Which strategies beat a benchmark with familywise-error control, and the joint
"best strategy does not beat the benchmark" p-value, via the circular block
bootstrap (Politis-Romano) with the Politis-White automatic block length.
Answers "which method/window actually beats 1/N once the whole method x window
grid is a snooping surface?".

Corpus keys to stage (author acquires; do NOT fabricate @cite):
``romano_wolf_2005``, ``white_reality_2000``, ``hansen_spa_2005``,
``patton_correction_2009``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jcor.core.typing import (  # noqa: TC001  # runtime contract
    Array,
    ArrayLike,
    Float,
    Int,
    PRNGKey,
)
from jcor.decision._sharpe.ratio import _sharpe_ratio_axis
from jcor.inference.block_length import (
    optimal_block_length,
    optimal_block_length_result,
)
from jcor.operators.longrun import (
    circular_block_bootstrap_indices,
    circular_block_bootstrap_indices_kernel,
)

type _ReturnVector = Float[Array, "observations"]  # noqa: F821  # jaxtyping
type _StrategyPanel = Float[Array, "observations strategies"]  # noqa: F722
type _StrategyVector = Float[Array, "strategies"]  # noqa: F821  # jaxtyping
type _BootstrapMatrix = Int[Array, "bootstrap observations"]  # noqa: F722
type _FloatScalar = Float[Array, ""]  # noqa: F722  # jaxtyping scalar


class MultipleSharpeResult(NamedTuple):
    """Result of a familywise multiple Sharpe-superiority test.

    Attributes:
        strategy_labels: Labels aligned with the statistics arrays.
        sharpe_diff: Observed Sharpe difference ``SR(strategy) - SR(benchmark)``
            per strategy.
        studentized: Studentized statistic ``sharpe_diff / bootstrap_se`` per
            strategy.
        rw_adjusted_pvalues: Romano-Wolf stepwise FWER-adjusted, one-sided
            p-values (H0: strategy does NOT beat the benchmark).
        reality_check_pvalue: White (2000) Reality-Check joint p-value for
            ``H0: the best strategy does not beat the benchmark``.
        block_length: Circular-block-bootstrap block length used.
        n_boot: Number of bootstrap resamples.

    """

    strategy_labels: tuple[str, ...]
    sharpe_diff: _StrategyVector
    studentized: _StrategyVector
    rw_adjusted_pvalues: _StrategyVector
    reality_check_pvalue: float
    block_length: int
    n_boot: int


class MultipleSharpeKernelResult(NamedTuple):
    """Array-only carrier for a keyed multiple-Sharpe test."""

    sharpe_diff: _StrategyVector
    studentized: _StrategyVector
    rw_adjusted_pvalues: _StrategyVector
    reality_check_pvalue: _FloatScalar
    block_length: Int[Array, ""]  # noqa: F722


@jax.jit
def _paired_sharpe_diffs(
    block: _StrategyPanel,
    bench: _ReturnVector,
) -> _StrategyVector:
    """Sharpe difference of every column of ``block`` vs ``bench`` (all (T,)).

    Args:
        block: Strategy return panel, shape ``(T, k)``.
        bench: Benchmark return series, shape ``(T,)``.

    Returns:
        Per-strategy Sharpe difference, shape ``(k,)``.

    """
    return _sharpe_ratio_axis(block, axis=0) - _sharpe_ratio_axis(bench, axis=0)


@jax.jit
def _multiple_sharpe_kernel(
    strategy_returns: _StrategyPanel,
    benchmark_returns: _ReturnVector,
    indices: _BootstrapMatrix,
) -> tuple[
    _StrategyVector,
    _StrategyVector,
    _StrategyVector,
    _FloatScalar,
]:
    """Evaluate bootstrap Sharpe statistics on a fixed resample index matrix.

    Args:
        strategy_returns: Strategy panel, shape ``(T, k)``.
        benchmark_returns: Benchmark series, shape ``(T,)``.
        indices: Fixed bootstrap indices, shape ``(B, T)``.

    Returns:
        Observed differences, studentized statistics, Romano-Wolf adjusted
        p-values, and the scalar Reality-Check p-value.

    """
    diff_hat = _paired_sharpe_diffs(strategy_returns, benchmark_returns)
    strategy_boot = strategy_returns[indices]
    benchmark_boot = benchmark_returns[indices]
    diff_boot = (
        _sharpe_ratio_axis(strategy_boot, axis=1)
        - _sharpe_ratio_axis(benchmark_boot, axis=1)[:, None]
    )

    standard_error = jnp.std(diff_boot, axis=0, ddof=1)
    standard_error = jnp.where(standard_error > 0.0, standard_error, jnp.inf)
    studentized = diff_hat / standard_error
    null_boot = (diff_boot - diff_hat[None, :]) / standard_error[None, :]

    reality_check = jnp.mean(
        jnp.max(null_boot, axis=1) >= jnp.max(studentized),
        dtype=strategy_returns.dtype,
    )

    order = jnp.argsort(-studentized)
    ordered_null = null_boot[:, order]
    ordered_statistics = studentized[order]
    step_maxima = lax.associative_scan(
        jnp.maximum,
        ordered_null,
        axis=1,
        reverse=True,
    )
    raw_pvalues = jnp.mean(
        step_maxima >= ordered_statistics[None, :],
        axis=0,
        dtype=strategy_returns.dtype,
    )
    monotone_pvalues = lax.associative_scan(jnp.maximum, raw_pvalues)
    adjusted = jnp.empty_like(monotone_pvalues).at[order].set(monotone_pvalues)
    return diff_hat, studentized, adjusted, reality_check


@partial(jax.jit, static_argnames=("n_boot",))
def multiple_sharpe_test_kernel(
    key: PRNGKey,
    strategy_returns: _StrategyPanel,
    benchmark_returns: _ReturnVector,
    *,
    n_boot: int,
) -> MultipleSharpeKernelResult:
    """Run the multiple-Sharpe test from an explicit bootstrap key."""
    observed = _paired_sharpe_diffs(strategy_returns, benchmark_returns)
    best = jnp.argmax(observed)
    block_length = jnp.rint(
        optimal_block_length_result(
            strategy_returns[:, best] - benchmark_returns,
            scheme="circular",
        )
    ).astype(jnp.int32)
    block_length = jnp.clip(block_length, 1, strategy_returns.shape[0])
    indices = circular_block_bootstrap_indices_kernel(
        key,
        strategy_returns.shape[0],
        block_length,
        n_boot,
    )
    diff_hat, studentized, adjusted, reality_check = _multiple_sharpe_kernel(
        strategy_returns,
        benchmark_returns,
        indices,
    )
    return MultipleSharpeKernelResult(
        diff_hat,
        studentized,
        adjusted,
        reality_check,
        block_length,
    )


def multiple_sharpe_test(
    strategy_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    strategy_labels: tuple[str, ...] | None = None,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> MultipleSharpeResult:
    """Romano-Wolf stepwise + White Reality-Check multiple Sharpe test.

    Tests, for each strategy ``k``, ``H0_k: SR_k <= SR_benchmark`` against the
    one-sided alternative that strategy ``k`` beats the benchmark, controlling
    the familywise error rate over the whole set. The base statistic is the
    Sharpe difference ``d_k = SR_k - SR_bench``, studentized by its circular
    block-bootstrap standard error (Ledoit-Wolf 2008 style). The bootstrap null
    is imposed by recentering (Politis-Romano circular block bootstrap over the
    PAIRED rows, preserving cross-strategy and serial dependence). The
    Romano-Wolf (2005) stepwise max-statistic procedure yields FWER-adjusted
    p-values; the single-step joint p-value on ``max_k d_k`` is White's (2000)
    Reality Check.

    Args:
        strategy_returns: Strategy return panel, shape ``(T, k)`` (columns are
            strategies). A 1-D array is treated as a single strategy.
        benchmark_returns: Benchmark return series, shape ``(T,)`` (e.g. 1/N).
        strategy_labels: Optional labels for the ``k`` strategies.
        n_boot: Number of circular-block-bootstrap resamples.
        block_length: Block length ``ell``; ``None`` uses the Politis-White
            automatic length
            (:func:`jcor.inference.block_length.optimal_block_length`,
            circular scheme) on the best strategy's differential series.
        seed: RNG seed for the bootstrap.

    Returns:
        A :class:`MultipleSharpeResult`.

    Raises:
        ValueError: If the benchmark length or the label count does not match
            the strategy panel.

    """
    strat = jnp.asarray(strategy_returns, dtype=jnp.float64)
    if strat.ndim == 1:
        strat = strat.reshape(-1, 1)
    bench = jnp.asarray(benchmark_returns, dtype=jnp.float64).ravel()
    n_obs, n_strat = strat.shape
    if bench.shape[0] != n_obs:
        message = "benchmark and strategies must share the same length"
        raise ValueError(message)
    labels = strategy_labels or tuple(f"strategy_{j}" for j in range(n_strat))
    if len(labels) != n_strat:
        message = "strategy_labels length must match the number of strategies"
        raise ValueError(message)

    observed = _paired_sharpe_diffs(strat, bench)
    if block_length is None:
        best = int(jnp.argmax(observed))
        ell = round(
            optimal_block_length(
                strat[:, best] - bench,
                scheme="circular",
            )
        )
    else:
        ell = int(block_length)
    ell = max(1, min(ell, n_obs))

    indices = jnp.asarray(
        circular_block_bootstrap_indices(n_obs, ell, n_boot, seed=seed),
        dtype=jnp.int64,
    )
    diff_hat, studentized, adjusted, reality_check = _multiple_sharpe_kernel(
        strat,
        bench,
        indices,
    )

    return MultipleSharpeResult(
        strategy_labels=labels,
        sharpe_diff=diff_hat,
        studentized=studentized,
        rw_adjusted_pvalues=adjusted,
        reality_check_pvalue=float(reality_check),
        block_length=ell,
        n_boot=n_boot,
    )

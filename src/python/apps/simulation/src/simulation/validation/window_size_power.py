"""Monte-Carlo SIZE/POWER of the overlap-robust Sharpe-superiority test vs T.

Backtest workstream deliverable #3 (decision brief D6): validates
:func:`jcor.operators.longrun.sharpe_difference_test` (and, for the multiple-testing
control, :func:`jcor.decision.sharpe.multiple_sharpe_test`) as a function
of the evaluation-series length ``T`` — the empirical backbone for "what
minimal T gives adequate power" against a target Sharpe gap.

ADDITIVE ONLY: this module reuses the existing overlap-robust bootstrap
machinery (:mod:`jcor.operators.longrun`, :mod:`jcor.decision.sharpe`) and the
shared harness summary primitives (:mod:`simulation.harness.stats`); it
re-implements no test statistic and overwrites no published number.

Models its structure on
:mod:`simulation.validation.size_power` (draw_* DGP fns, ``_size_sim`` /
``_power_sim`` task fns, ``run_*_experiment`` aggregators returning
DataFrames).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
from jax import lax, random
from jcor.core.random import cell_key
from jcor.decision.sharpe import (
    multiple_sharpe_test_kernel,
    sharpe_difference_test_kernel,
)

from simulation.harness.replications import execute_replication_cell, map_replications
from simulation.harness.stats import binomial_se, power_summary, size_summary
from simulation.io import write_parquet_atomic, write_yaml_summary

_SIZE_Z_SCORE_LIMIT = 2.0
_ADEQUATE_POWER_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


def draw_paired_ar1_returns(
    key: jax.Array,
    n_obs: int,
    phi: float,
    mu_candidate: float,
    mu_benchmark: float,
    sigma: float,
) -> tuple[jax.Array, jax.Array]:
    """Paired AR(1) candidate/benchmark return series.

    Each series is an AR(1) process with autoregressive coefficient ``phi``
    (serial dependence, to exercise the circular block bootstrap and the
    overlap-robustness of :func:`jcor.operators.longrun.sharpe_difference_test`) built
    from a shared common-factor innovation plus idiosyncratic noise, so the
    two series are positively (paired) correlated — the realistic backtest
    case where candidate and benchmark trade the same market.

    Innovation scale is chosen so each series has (approximately) stationary
    unconditional standard deviation ``sigma``: for an AR(1) with i.i.d.
    innovations of variance ``s^2``, the stationary variance is
    ``s^2 / (1 - phi^2)``, so innovations are scaled by
    ``sigma * sqrt(1 - phi^2)``.

    The TRUE per-period Sharpe gap between the two series is
    ``(mu_candidate - mu_benchmark) / sigma`` (both series share the same
    dispersion ``sigma`` by construction).

    Args:
        n_obs: Series length ``T``.
        phi: AR(1) coefficient, ``|phi| < 1``.
        mu_candidate: Candidate series per-period mean.
        mu_benchmark: Benchmark series per-period mean.
        sigma: Target stationary standard deviation, shared by both series.
        key: Threefry key dedicated to this paired-series draw.

    Returns:
        ``(candidate, benchmark)``, each shape ``(n_obs,)``.

    """
    if not (-1.0 < phi < 1.0):
        message = f"phi must be in (-1, 1), got {phi}"
        raise ValueError(message)
    innov_scale = sigma * jnp.sqrt(1.0 - phi * phi)

    # Common factor (shared market shock) + idiosyncratic noise per series,
    # combined so each idiosyncratic AR(1) innovation still has stationary
    # std `sigma` after adding a half-weight common factor.
    common_key, candidate_key, benchmark_key = random.split(key, 3)
    common = random.normal(common_key, (n_obs,)) * innov_scale * jnp.sqrt(0.5)
    idio_c = random.normal(candidate_key, (n_obs,)) * innov_scale * jnp.sqrt(0.5)
    idio_b = random.normal(benchmark_key, (n_obs,)) * innov_scale * jnp.sqrt(0.5)
    e_c = common + idio_c
    e_b = common + idio_b

    initial = jnp.asarray((mu_candidate + e_c[0], mu_benchmark + e_b[0]))

    def step(
        previous: jax.Array, innovations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        means = jnp.asarray((mu_candidate, mu_benchmark))
        current = means + phi * (previous - means) + innovations
        return current, current

    _, tail = lax.scan(step, initial, jnp.stack((e_c[1:], e_b[1:]), axis=1))
    series = jnp.concatenate((initial[None, :], tail), axis=0)
    return series[:, 0], series[:, 1]


# ---------------------------------------------------------------------------
# SIZE experiment (H0: true Sharpe gap == 0)
# ---------------------------------------------------------------------------


def _size_replication(
    key: jax.Array,
    *,
    t_value: int,
    phi: float,
    sigma: float,
    n_boot: int,
) -> jax.Array:
    """Return one null paired-Sharpe p-value."""
    data_key, bootstrap_key = random.split(key)
    candidate, benchmark = draw_paired_ar1_returns(
        data_key, t_value, phi, 0.0, 0.0, sigma
    )
    return sharpe_difference_test_kernel(
        bootstrap_key, candidate, benchmark, n_boot=n_boot
    ).pvalue


@partial(
    jax.jit,
    static_argnames=(
        "t_value",
        "phi",
        "sigma",
        "n_boot",
        "n_sims",
        "batch_size",
    ),
)
def _size_cell(
    key: jax.Array,
    *,
    t_value: int,
    phi: float,
    sigma: float,
    n_boot: int,
    n_sims: int,
    batch_size: int,
) -> jax.Array:
    kernel = partial(
        _size_replication,
        t_value=t_value,
        phi=phi,
        sigma=sigma,
        n_boot=n_boot,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def run_window_size_experiment(
    t_values: list[int],
    phi: float,
    alpha_levels: list[float],
    n_sims: int,
    n_boot: int,
    seed: int,
    sigma: float = 0.02,
    replication_batch_size: int = 64,
) -> pd.DataFrame:
    """Empirical SIZE of :func:`sharpe_difference_test` vs ``T`` under H0.

    For each ``T`` in ``t_values``, draws ``n_sims`` paired series with
    ``mu_candidate == mu_benchmark`` (true Sharpe gap 0, the null), runs
    ``sharpe_difference_test``, and reports the empirical rejection rate at
    each level in ``alpha_levels`` via
    :func:`simulation.harness.stats.size_summary`.

    Args:
        t_values: Evaluation-window lengths to sweep.
        phi: AR(1) serial-dependence coefficient shared by all draws.
        alpha_levels: Nominal significance levels to check.
        n_sims: Monte-Carlo replications per ``T``.
        n_boot: Bootstrap resamples per ``sharpe_difference_test`` call.
        seed: Base seed (position-independent per-cell seeding via
            :func:`jcor.cell_key`).
        sigma: Shared per-period return standard deviation for both series.
        replication_batch_size: Maximum replications evaluated in one device batch.

    Returns:
        One row per ``(T, alpha)`` with columns ``T``, ``alpha``,
        ``n_sims``, ``empirical_size``, ``se``, ``z_score``, and ``size_ok``.

    """
    rows: list[dict[str, Any]] = []
    for t_value in t_values:
        key = cell_key(seed, "mc_window_size_power", "size", "T", t_value)
        pvalues = execute_replication_cell(
            _size_cell,
            key,
            name="mc_window_size_power.size",
            replication_count=n_sims,
            replication_batch_size=replication_batch_size,
            static={"T": t_value},
            t_value=t_value,
            phi=phi,
            sigma=sigma,
            n_boot=n_boot,
            n_sims=n_sims,
            batch_size=replication_batch_size,
        )
        rows.extend(
            [
                {
                    "T": t_value,
                    "alpha": entry["nominal_alpha"],
                    "n_sims": n_sims,
                    "empirical_size": entry["empirical_rejection_rate"],
                    "se": entry["binom_se"],
                    "z_score": entry["z_score"],
                    "size_ok": abs(entry["z_score"]) <= _SIZE_Z_SCORE_LIMIT,
                }
                for entry in size_summary(pvalues, alpha_levels, n_sims)
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# POWER experiment (H1: true Sharpe gap == delta)
# ---------------------------------------------------------------------------


def _power_replication(
    key: jax.Array,
    *,
    t_value: int,
    phi: float,
    sigma: float,
    delta: float,
    n_boot: int,
) -> jax.Array:
    """Return one alternative paired-Sharpe p-value."""
    data_key, bootstrap_key = random.split(key)
    mu_candidate = delta * sigma
    candidate, benchmark = draw_paired_ar1_returns(
        data_key, t_value, phi, mu_candidate, 0.0, sigma
    )
    return sharpe_difference_test_kernel(
        bootstrap_key, candidate, benchmark, n_boot=n_boot
    ).pvalue


@partial(
    jax.jit,
    static_argnames=(
        "t_value",
        "phi",
        "sigma",
        "delta",
        "n_boot",
        "n_sims",
        "batch_size",
    ),
)
def _power_cell(
    key: jax.Array,
    *,
    t_value: int,
    phi: float,
    sigma: float,
    delta: float,
    n_boot: int,
    n_sims: int,
    batch_size: int,
) -> jax.Array:
    kernel = partial(
        _power_replication,
        t_value=t_value,
        phi=phi,
        sigma=sigma,
        delta=delta,
        n_boot=n_boot,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def run_window_power_experiment(
    t_values: list[int],
    phi: float,
    deltas: list[float],
    alpha: float,
    n_sims: int,
    n_boot: int,
    seed: int,
    sigma: float = 0.02,
    replication_batch_size: int = 64,
) -> pd.DataFrame:
    """Empirical POWER of :func:`sharpe_difference_test` vs ``T`` and ``delta``.

    For each ``T`` and each true per-period Sharpe gap ``delta`` (candidate
    mean set to ``delta * sigma`` above the zero-mean benchmark), draws
    ``n_sims`` paired series and reports the rejection rate at ``alpha`` via
    :func:`simulation.harness.stats.power_summary`. Power rises with ``T``
    for fixed ``delta > 0``.

    Args:
        t_values: Evaluation-window lengths to sweep.
        phi: AR(1) serial-dependence coefficient.
        deltas: True Sharpe gaps (candidate minus benchmark, in per-period
            Sharpe units) to sweep.
        replication_batch_size: Maximum replications evaluated in one device batch.
        alpha: Significance level at which power is evaluated.
        n_sims: Monte-Carlo replications per ``(T, delta)`` cell.
        n_boot: Bootstrap resamples per ``sharpe_difference_test`` call.
        seed: Base seed.
        sigma: Shared per-period return standard deviation for both series.

    Returns:
        One row per ``(T, delta)`` with columns ``T``, ``delta``, ``alpha``,
        ``n_sims``, ``power``, ``se``.

    """
    rows: list[dict[str, Any]] = []
    for t_value in t_values:
        for delta in deltas:
            key = cell_key(
                seed,
                "mc_window_size_power",
                "power",
                "T",
                t_value,
                "delta",
                delta,
            )
            pvalues = execute_replication_cell(
                _power_cell,
                key,
                name="mc_window_size_power.power",
                replication_count=n_sims,
                replication_batch_size=replication_batch_size,
                static={"T": t_value, "delta": delta},
                t_value=t_value,
                phi=phi,
                sigma=sigma,
                delta=delta,
                n_boot=n_boot,
                n_sims=n_sims,
                batch_size=replication_batch_size,
            )
            summary = power_summary(pvalues, alpha, n_sims)
            rows.append(
                {
                    "T": t_value,
                    "delta": delta,
                    "alpha": alpha,
                    "n_sims": n_sims,
                    "power": summary["empirical_power"],
                    "se": summary["binom_se"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multiple-testing FWER check (Romano-Wolf / Reality Check)
# ---------------------------------------------------------------------------


def _fwer_replication(
    key: jax.Array,
    *,
    t_value: int,
    phi: float,
    sigma: float,
    k_strategies: int,
    n_boot: int,
) -> jax.Array:
    """Return the minimum Romano-Wolf p-value under the global null."""
    draws_key, bootstrap_key = random.split(key)
    draw_keys = random.split(draws_key, k_strategies)
    candidates, benchmarks = jax.vmap(
        lambda draw_key: draw_paired_ar1_returns(
            draw_key, t_value, phi, 0.0, 0.0, sigma
        )
    )(draw_keys)
    strategies = candidates.T
    result = multiple_sharpe_test_kernel(
        bootstrap_key,
        strategies,
        benchmarks[0],
        n_boot=n_boot,
    )
    return jnp.min(result.rw_adjusted_pvalues)


@partial(
    jax.jit,
    static_argnames=(
        "t_value",
        "phi",
        "sigma",
        "k_strategies",
        "n_boot",
        "n_sims",
        "batch_size",
    ),
)
def _fwer_cell(
    key: jax.Array,
    *,
    t_value: int,
    phi: float,
    sigma: float,
    k_strategies: int,
    n_boot: int,
    n_sims: int,
    batch_size: int,
) -> jax.Array:
    kernel = partial(
        _fwer_replication,
        t_value=t_value,
        phi=phi,
        sigma=sigma,
        k_strategies=k_strategies,
        n_boot=n_boot,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def run_multiple_testing_fwer(
    t_value: int,
    phi: float,
    k_strategies: int,
    alpha: float,
    n_sims: int,
    n_boot: int,
    seed: int,
    sigma: float = 0.02,
    replication_batch_size: int = 64,
) -> dict[str, Any]:
    """Familywise error rate of :func:`multiple_sharpe_test` under a global null.

    Draws ``k_strategies`` NULL candidate series (all with true Sharpe gap 0
    against a shared benchmark) per simulation, runs
    :func:`jcor.decision.sharpe.multiple_sharpe_test`, and reports the
    fraction of simulations in which *any* Romano-Wolf adjusted p-value is
    below ``alpha`` (reject-any). Familywise-error control implies this
    fraction stays at or below (approximately) ``alpha``.

    Args:
        t_value: Evaluation-window length ``T``.
        phi: AR(1) serial-dependence coefficient shared by all draws.
        k_strategies: Number of null candidate strategies in the family.
        alpha: Significance level for the familywise test.
        n_sims: Monte-Carlo replications.
        n_boot: Bootstrap resamples per :func:`multiple_sharpe_test` call.
        seed: Base seed.
        sigma: Shared per-period return standard deviation.
        replication_batch_size: Maximum replications evaluated in one device batch.

    Returns:
        Dict with ``T``, ``k_strategies``, ``alpha``, ``n_sims``,
        ``familywise_error_rate``, ``se``.

    """
    if k_strategies < 1:
        message = "k_strategies must be positive"
        raise ValueError(message)
    key = cell_key(
        seed,
        "mc_window_size_power",
        "fwer",
        "T",
        t_value,
        "strategies",
        k_strategies,
    )
    min_pvalues = execute_replication_cell(
        _fwer_cell,
        key,
        name="mc_window_size_power.fwer",
        replication_count=n_sims,
        replication_batch_size=replication_batch_size,
        static={"T": t_value, "strategies": k_strategies},
        t_value=t_value,
        phi=phi,
        sigma=sigma,
        k_strategies=k_strategies,
        n_boot=n_boot,
        n_sims=n_sims,
        batch_size=replication_batch_size,
    )
    fwer = float(jnp.mean(min_pvalues < alpha))
    return {
        "T": t_value,
        "k_strategies": k_strategies,
        "alpha": alpha,
        "n_sims": n_sims,
        "familywise_error_rate": fwer,
        "se": binomial_se(fwer, n_sims),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_window_size_power(
    t_values: list[int],
    phi: float,
    deltas: list[float],
    alpha_levels: list[float],
    n_sims: int,
    n_boot: int,
    seed: int,
    k_strategies: int,
    output_dir: str | None = None,
    replication_batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Orchestrate the SIZE, POWER, and FWER window-length validations.

    Runs :func:`run_window_size_experiment`, :func:`run_window_power_experiment`
    (at the primary alpha, ``alpha_levels[0]``), and
    :func:`run_multiple_testing_fwer` (at the largest ``t_value`` in ``t_values``),
    then builds a summary dict recording the DGP config, per-alpha size
    control, and the smallest ``T`` at which power reaches ``>= 0.5`` for the
    largest ``delta`` — a data-driven "minimal adequate t_value" readout for
    decision brief D6.

    Args:
        t_values: Evaluation-window lengths to sweep.
        phi: AR(1) serial-dependence coefficient.
        deltas: True Sharpe gaps to sweep in the power experiment.
        alpha_levels: Nominal significance levels; ``alpha_levels[0]`` is used
            as the primary level for the power and FWER experiments.
        n_sims: Monte-Carlo replications per cell.
        n_boot: Bootstrap resamples per test call.
        seed: Base seed.
        k_strategies: Number of null strategies in the FWER family.
        output_dir: If given, writes ``size_results.parquet``,
            ``power_results.parquet``, and ``summary.yaml`` there.
        replication_batch_size: Maximum replications evaluated in one device batch.

    Returns:
        ``(size_df, power_df, summary)``.

    """
    alpha_primary = alpha_levels[0]

    size_df = run_window_size_experiment(
        t_values=t_values,
        phi=phi,
        alpha_levels=alpha_levels,
        n_sims=n_sims,
        n_boot=n_boot,
        seed=seed,
        replication_batch_size=replication_batch_size,
    )
    power_df = run_window_power_experiment(
        t_values=t_values,
        phi=phi,
        deltas=deltas,
        alpha=alpha_primary,
        n_sims=n_sims,
        n_boot=n_boot,
        seed=seed,
        replication_batch_size=replication_batch_size,
    )
    fwer = run_multiple_testing_fwer(
        t_value=max(t_values),
        phi=phi,
        k_strategies=k_strategies,
        alpha=alpha_primary,
        n_sims=n_sims,
        n_boot=n_boot,
        seed=seed,
        replication_batch_size=replication_batch_size,
    )

    size_control: dict[str, Any] = {}
    for alpha in alpha_levels:
        subset = size_df[size_df["alpha"] == alpha]
        size_control[f"alpha_{float(alpha)}"] = {
            "nominal": float(alpha),
            "controlled": bool(subset["size_ok"].all()),
            "max_abs_z": float(subset["z_score"].abs().max()) if len(subset) else None,
        }

    min_adequate_t: int | None = None
    if deltas:
        largest_delta = max(deltas)
        at_delta = power_df[power_df["delta"] == largest_delta].sort_values("T")
        above = at_delta[at_delta["power"] >= _ADEQUATE_POWER_THRESHOLD]
        if not above.empty:
            min_adequate_t = int(above.iloc[0]["T"])

    summary: dict[str, Any] = {
        "dgp": {
            "phi": phi,
            "t_values": list(t_values),
            "deltas": list(deltas),
            "alpha_levels": list(alpha_levels),
            "n_sims": n_sims,
            "n_boot": n_boot,
            "seed": seed,
            "k_strategies": k_strategies,
        },
        "size_control": size_control,
        "fwer": fwer,
        "minimal_adequate_T": min_adequate_t,
    }

    if output_dir is not None:
        out = Path(output_dir)
        write_parquet_atomic(size_df, out / "size_results.parquet")
        write_parquet_atomic(power_df, out / "power_results.parquet")
        write_yaml_summary(summary, out / "summary.yaml")

    return size_df, power_df, summary

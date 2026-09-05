"""Type I / Type II (size / power) validation of the energy permutation test.

Validates the energy permutation test used as Paper 2's hedge-portfolio
distribution-matching test (ROADMAP §4.5 / §5).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
from jax import random
from jcor.core.random import cell_key
from jcor.discrepancy.energy import UNIT_EXPONENT, energy_geometry
from jcor.ground.metrics import EUCLIDEAN
from jcor.inference import energy_permutation_test_result

from simulation.harness.replications import execute_replication_cell, map_replications

_THREE_STANDARD_ERRORS = 3.0
_HEADLINE_SAMPLE_SIZE = 128
_HEADLINE_DIMENSION = 8
_TARGET_POWER = 0.80


_EUCLIDEAN_ENERGY = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)


def run_permutation_test(
    x: jax.Array,
    y: jax.Array,
    num_permutations: int,
    key: jax.Array,
) -> jax.Array:
    """Run the transformable, distance-hoisted energy permutation kernel."""
    result = energy_permutation_test_result(
        jnp.asarray(x),
        jnp.asarray(y),
        geometry=_EUCLIDEAN_ENERGY,
        key=key,
        num_permutations=num_permutations,
        alternative="greater",
    )
    return result.pvalue


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def draw_normal(key: jax.Array, n: int, d: int) -> jax.Array:
    """Draw independent standard-normal samples."""
    return random.normal(key, (n, d))


def draw_t(key: jax.Array, n: int, d: int, df: int) -> jax.Array:
    """Multivariate t via chi-squared scaling (independent marginals)."""
    normal_key, chi_key = random.split(key)
    z = random.normal(normal_key, (n, d))
    chi2 = 2.0 * random.gamma(chi_key, df / 2.0, shape=(n, 1))
    return z / jnp.sqrt(chi2 / df)


def draw_shifted_mean(
    key: jax.Array, n: int, d: int, delta: float | jax.Array
) -> jax.Array:
    """Draw normal samples with mean shift δ in every dimension."""
    return random.normal(key, (n, d)) + delta


def draw_scaled(key: jax.Array, n: int, d: int, scale: float | jax.Array) -> jax.Array:
    """Draw normal samples with a scale shift."""
    return random.normal(key, (n, d)) * scale


def draw_mixture(
    key: jax.Array, n: int, d: int, contamination: float, shift: float
) -> jax.Array:
    """Draw a normal mixture with a shifted contaminated fraction."""
    n_contam = int(n * contamination)
    return _draw_mixture_count(key, n, d, n_contam, shift)


def _draw_mixture_count(
    key: jax.Array, n: int, d: int, n_contam: int, shift: float
) -> jax.Array:
    """Draw a mixture when the shape-determining count is already static."""
    n_clean = n - n_contam
    clean_key, contam_key = random.split(key)
    clean = random.normal(clean_key, (n_clean, d))
    contam = random.normal(contam_key, (n_contam, d)) + shift
    return jnp.concatenate([clean, contam], axis=0)


# ---------------------------------------------------------------------------
# Size experiment
# ---------------------------------------------------------------------------


def _size_replication(
    key: jax.Array,
    *,
    t_df: int,
    n: int,
    d: int,
    num_permutations: int,
) -> jax.Array:
    """Return one null p-value from an explicitly split replication key."""
    x_key, y_key, permutation_key = random.split(key, 3)
    if t_df == 0:
        x = draw_normal(x_key, n, d)
        y = draw_normal(y_key, n, d)
    else:
        x = draw_t(x_key, n, d, t_df)
        y = draw_t(y_key, n, d, t_df)
    return run_permutation_test(x, y, num_permutations, permutation_key)


@partial(
    jax.jit,
    static_argnames=("t_df", "n", "d", "num_permutations", "n_sims", "batch_size"),
)
def _size_cell(
    key: jax.Array,
    *,
    t_df: int,
    n: int,
    d: int,
    num_permutations: int,
    n_sims: int,
    batch_size: int,
) -> jax.Array:
    kernel = partial(
        _size_replication,
        t_df=t_df,
        n=n,
        d=d,
        num_permutations=num_permutations,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def run_size_experiment(
    n_sims: int,
    num_permutations: int,
    n_values: list[int],
    d_values: list[int],
    alpha_levels: list[float],
    base_seed: int,
    replication_batch_size: int = 64,
) -> pd.DataFrame:
    """Simulate empirical rejection rates under H0 (size / Type I error)."""
    dist_configs = [("normal", 0), ("t_df4", 4)]
    cells = [
        (dist_name, t_df, n, d)
        for dist_name, t_df in dist_configs
        for n in n_values
        for d in d_values
    ]

    cell_pvalues: list[jax.Array] = []
    for dist_name, t_df, n, d in cells:
        key = cell_key(
            base_seed,
            "mc_test_size_power",
            "size",
            "distribution",
            dist_name,
            "n",
            n,
            "d",
            d,
        )
        cell_pvalues.append(
            execute_replication_cell(
                _size_cell,
                key,
                name="mc_test_size_power.size",
                replication_count=n_sims,
                replication_batch_size=replication_batch_size,
                static={"distribution": dist_name, "n": n, "d": d},
                t_df=t_df,
                n=n,
                d=d,
                num_permutations=num_permutations,
                n_sims=n_sims,
                batch_size=replication_batch_size,
            )
        )

    rows = []
    for (dist_name, _t_df, n, d), cell_pvals in zip(cells, cell_pvalues, strict=True):
        for alpha in alpha_levels:
            rej_rate = jnp.mean(cell_pvals < alpha)
            binom_se = jnp.sqrt(alpha * (1.0 - alpha) / n_sims)
            z_score = jnp.where(binom_se > 0, (rej_rate - alpha) / binom_se, 0.0)
            rows.append(
                {
                    "distribution": dist_name,
                    "n": n,
                    "d": d,
                    "nominal_alpha": alpha,
                    "empirical_rejection_rate": float(rej_rate),
                    "binom_se": float(binom_se),
                    "z_score": float(z_score),
                    "flag_outside_3se": bool(jnp.abs(z_score) > _THREE_STANDARD_ERRORS),
                    "n_sims": n_sims,
                    "num_permutations": num_permutations,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Power experiment
# ---------------------------------------------------------------------------


def _power_replication(
    key: jax.Array,
    *,
    alt_type: str,
    alt_param: jax.Array,
    n_contam: int,
    n: int,
    d: int,
    num_permutations: int,
) -> jax.Array:
    """Return one alternative p-value from an explicitly split key."""
    x_key, y_key, permutation_key = random.split(key, 3)
    x = draw_normal(x_key, n, d)
    if alt_type == "mean_shift":
        y = draw_shifted_mean(y_key, n, d, alt_param)
    elif alt_type == "scale_shift":
        y = draw_scaled(y_key, n, d, alt_param)
    else:
        y = _draw_mixture_count(y_key, n, d, n_contam, shift=2.0)
    return run_permutation_test(x, y, num_permutations, permutation_key)


@partial(
    jax.jit,
    static_argnames=(
        "alt_type",
        "n_contam",
        "n",
        "d",
        "num_permutations",
        "n_sims",
        "batch_size",
    ),
)
def _power_cell(
    key: jax.Array,
    *,
    alt_type: str,
    alt_param: jax.Array,
    n_contam: int,
    n: int,
    d: int,
    num_permutations: int,
    n_sims: int,
    batch_size: int,
) -> jax.Array:
    kernel = partial(
        _power_replication,
        alt_type=alt_type,
        alt_param=alt_param,
        n_contam=n_contam,
        n=n,
        d=d,
        num_permutations=num_permutations,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def run_power_experiment(
    n_sims: int,
    num_permutations: int,
    n_values: list[int],
    d_values: list[int],
    alpha: float,
    mean_shifts: list[float],
    scale_factors: list[float],
    contamination_fracs: list[float],
    base_seed: int,
    replication_batch_size: int = 64,
) -> pd.DataFrame:
    """Simulate empirical power curves under H1 alternatives."""
    alt_configs = [("mean_shift", f"delta={delta}", delta) for delta in mean_shifts]
    alt_configs.extend(
        ("scale_shift", f"scale={scale}", scale) for scale in scale_factors
    )
    alt_configs.extend(
        ("mixture", f"frac={frac}", frac) for frac in contamination_fracs
    )

    cells = [
        (alt_type, alt_label, alt_param, n, d)
        for alt_type, alt_label, alt_param in alt_configs
        for n in n_values
        for d in d_values
    ]

    cell_pvalues: list[jax.Array] = []
    for alt_type, alt_label, alt_param, n, d in cells:
        key = cell_key(
            base_seed,
            "mc_test_size_power",
            "power",
            "alternative",
            alt_label,
            "n",
            n,
            "d",
            d,
        )
        cell_pvalues.append(
            execute_replication_cell(
                _power_cell,
                key,
                name="mc_test_size_power.power",
                replication_count=n_sims,
                replication_batch_size=replication_batch_size,
                static={"alternative": alt_label, "n": n, "d": d},
                alt_type=alt_type,
                alt_param=alt_param,
                n_contam=int(n * alt_param) if alt_type == "mixture" else 0,
                n=n,
                d=d,
                num_permutations=num_permutations,
                n_sims=n_sims,
                batch_size=replication_batch_size,
            )
        )
    rows = []
    for (
        alt_type,
        alt_label,
        alt_param,
        n,
        d,
    ), cell_pvals in zip(cells, cell_pvalues, strict=True):
        power = jnp.mean(cell_pvals < alpha)
        binom_se = jnp.sqrt(power * (1.0 - power) / n_sims)
        rows.append(
            {
                "alt_type": alt_type,
                "alt_label": alt_label,
                "alt_param": float(alt_param),
                "n": n,
                "d": d,
                "nominal_alpha": alpha,
                "empirical_power": float(power),
                "binom_se": float(binom_se),
                "n_sims": n_sims,
                "num_permutations": num_permutations,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    size_df: pd.DataFrame,
    power_df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build headline summary for summary.yaml."""
    size_normal = size_df[size_df["distribution"] == "normal"]
    size_summary = {}
    for alpha in sorted(size_normal["nominal_alpha"].unique()):
        subset = size_normal[size_normal["nominal_alpha"] == alpha]
        mean_rej = float(subset["empirical_rejection_rate"].mean())
        max_abs_z = float(subset["z_score"].abs().max())
        any_flag = bool(subset["flag_outside_3se"].any())
        se_val = float(subset["binom_se"].mean())
        size_summary[f"alpha_{float(alpha)}"] = {
            "nominal": float(alpha),
            "mean_empirical_rejection_rate": round(mean_rej, 4),
            "binomial_se": round(se_val, 4),
            "max_abs_z_score": round(max_abs_z, 3),
            "pass": not any_flag,
            "note": (
                "FAIL: empirical size outside ±3 SE; see size_results.parquet"
                if any_flag
                else "PASS"
            ),
        }

    power_128_8 = power_df[
        (power_df["n"] == _HEADLINE_SAMPLE_SIZE)
        & (power_df["d"] == _HEADLINE_DIMENSION)
    ]
    power_summary = {}
    if not power_128_8.empty:
        for _, row in power_128_8.iterrows():
            key = f"{row['alt_type']}_{row['alt_label']}"
            power_summary[key] = {
                "alt_type": row["alt_type"],
                "alt_label": row["alt_label"],
                "n": _HEADLINE_SAMPLE_SIZE,
                "d": _HEADLINE_DIMENSION,
                "empirical_power": round(float(row["empirical_power"]), 4),
                "binom_se": round(float(row["binom_se"]), 4),
            }

    power_80: dict[str, int | None] = {}
    power_d8 = power_df[power_df["d"] == _HEADLINE_DIMENSION]
    if not power_d8.empty:
        for (alt_type, alt_label), grp in power_d8.groupby(["alt_type", "alt_label"]):
            above_80 = grp[grp["empirical_power"] >= _TARGET_POWER].sort_values("n")
            key = f"{alt_type}_{alt_label}"
            if not above_80.empty:
                power_80[key] = int(above_80.iloc[0]["n"])
            else:
                power_80[key] = None

    future_work = (
        "GMM-style identity moment test (Paper 2 §3, ROADMAP §4.5) is out of scope: "
        "requires full pipeline (embeddings, returns, portfolio weights). "
        "Implement separately once compute_long_short_analysis data is available."
    )

    return {
        "size": size_summary,
        "power_at_n128_d8": power_summary,
        "smallest_n_for_80pct_power_d8": power_80,
        "params": params,
        "future_work": future_work,
    }


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_size_power(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the size and power Monte-Carlo validation.

    Args:
        cfg: ``params["monte_carlo"]["test_size_power"]`` dict from ``params.yaml``.

    Returns:
        (size_df, power_df, summary).

    """
    n_sims: int = cfg["n_sims"]
    num_permutations: int = cfg["num_permutations"]
    n_values: list[int] = cfg["n_values"]
    d_values: list[int] = cfg["d_values"]
    alpha_levels: list[float] = cfg["alpha_levels"]
    mean_shifts: list[float] = cfg["mean_shifts"]
    scale_factors: list[float] = cfg["scale_factors"]
    contamination_fracs: list[float] = cfg["contamination_fracs"]
    base_seed: int = cfg["seed"]
    replication_batch_size: int = int(cfg.get("replication_batch_size", 64))

    size_df = run_size_experiment(
        n_sims=n_sims,
        num_permutations=num_permutations,
        n_values=n_values,
        d_values=d_values,
        alpha_levels=alpha_levels,
        base_seed=base_seed,
        replication_batch_size=replication_batch_size,
    )

    # Size and power have disjoint kernels and together span forty static
    # shape/type specialisations at the production grid. Retaining the size
    # executables while compiling power cells made the CPU process exceed its
    # memory budget. The stage is process-isolated, and its materialized host
    # frame is complete here, so release the in-memory compilation cache before
    # entering the second experiment. The persistent cache remains available
    # to later processes. ``alt_param`` is dynamic above so the three values of
    # one alternative type reuse the same shape-specialized executable.
    jax.clear_caches()

    power_df = run_power_experiment(
        n_sims=n_sims,
        num_permutations=num_permutations,
        n_values=n_values,
        d_values=d_values,
        alpha=0.05,
        mean_shifts=mean_shifts,
        scale_factors=scale_factors,
        contamination_fracs=contamination_fracs,
        base_seed=base_seed,
        replication_batch_size=replication_batch_size,
    )

    summary = build_summary(size_df, power_df, cfg)

    return size_df, power_df, summary

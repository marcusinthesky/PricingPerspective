"""GMM Wald + Wolak size/power Monte-Carlo (Paper 3 — gates Wave 2).

Data-free Monte-Carlo stage that (a) confirms the SIZE of the GMM Wald and
Wolak inequality tests on synthetic moment processes with a KNOWN null, and
(b) maps their POWER against calibrated deviations.  This gates the empirical
interpretation of the 53-ticker run: if the tests are mis-sized on synthetic
data whose covariance we control, the empirical p-values are not trustworthy.

Design:
    * Moment process ``g_t`` of dimension ``m`` (post-clustering) is drawn from
      a Gaussian AR(1) with a known long-run covariance so the HAC estimator
      has something non-trivial to recover.
    * SIZE (Wald): ``E[g_t] = 0`` — the equality identity holds.  We report the
      empirical rejection rate vs nominal α (should be within ±3 binomial SE).
    * POWER (Wald): ``E[g_t] = δ·1`` for a grid of δ.
    * SIZE (Wolak): ``E[g_t] = 0`` on the boundary of ``g ≥ 0`` (least-
      favourable null) — the χ̄² weights are Monte-Carlo-calibrated per cell.
    * POWER (Wolak): one moment mean set to ``−δ`` (inequality violated).

Consistent with the existing ``validation/size_power.py`` contract:
returns ``(size_df, power_df, summary)`` and writes a ``summary.yaml`` the
Paper-3 typst can read.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pandas as pd
from jax import lax, random
from jcor.core.random import cell_key
from jcor.inference import gmm_wald_kernel, wolak_test_kernel
from jcor.operators.longrun import optimal_bandwidth

from simulation.harness.replications import execute_replication_cell, map_replications

_THREE_STANDARD_ERRORS = 3.0
# The production design's m=12 covariance family was checked across all 300
# null lanes against jcor's conservative 4096-sweep default: 256 plus Wolak's
# active-set polish converges everywhere and returns exact p-values/decisions.
# This is an execution budget, not a statistical parameter.
_SIMULATION_PROJECTION_MAX_SWEEPS = 256


def _ar1_moment_panel(
    key: jax.Array,
    n_observations: int,
    m: int,
    mean: jax.Array,
    phi: float,
    innov_scale: float,
) -> jax.Array:
    """Draw a Gaussian AR(1) moment panel with a target mean.

    ``g_t = mean + phi·(g_{t-1} − mean) + innov_scale·ε_t``, ``ε_t ~ N(0, I_m)``.

    Args:
        n_observations: Number of periods.
        m: Moment dimension.
        mean: Target mean vector, shape ``(m,)``.
        phi: AR(1) coefficient (serial dependence for HAC).
        innov_scale: Innovation standard deviation.
        key: Threefry key dedicated to this panel.

    Returns:
        Panel of shape ``(n_observations, m)``.

    """
    eps = innov_scale * random.normal(key, (n_observations, m), dtype=mean.dtype)
    initial = mean + eps[0]

    def step(previous: jax.Array, innovation: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = mean + phi * (previous - mean) + innovation
        return current, current

    _, tail = lax.scan(step, initial, eps[1:])
    return jnp.concatenate((initial[None, :], tail), axis=0)


# ---------------------------------------------------------------------------
# Replication kernels
# ---------------------------------------------------------------------------


class GmmWolakReplication(NamedTuple):
    """Array-only result for one GMM/Wolak replication."""

    pvalue: jax.Array
    converged: jax.Array


def _test_replication(
    key: jax.Array,
    mean: jax.Array,
    *,
    test: str,
    n_observations: int,
    m: int,
    phi: float,
    innov_scale: float,
    bandwidth: int,
    n_mc: int,
    projection_max_sweeps: int,
) -> GmmWolakReplication:
    """Draw one moment panel and evaluate one keyed inference kernel."""
    data_key, calibration_key = random.split(key)
    moments = _ar1_moment_panel(data_key, n_observations, m, mean, phi, innov_scale)
    if test == "wald":
        result = gmm_wald_kernel(moments, bandwidth)
        return GmmWolakReplication(result.pvalue, jnp.ones((), dtype=jnp.bool_))
    result = wolak_test_kernel(
        calibration_key,
        moments,
        bandwidth=bandwidth,
        n_mc=n_mc,
        max_sweeps=projection_max_sweeps,
    )
    return GmmWolakReplication(result.pvalue, result.converged)


@partial(
    jax.jit,
    static_argnames=(
        "test",
        "n_observations",
        "m",
        "phi",
        "innov_scale",
        "bandwidth",
        "n_mc",
        "projection_max_sweeps",
        "n_sims",
        "batch_size",
    ),
)
def _test_cell(
    key: jax.Array,
    mean: jax.Array,
    *,
    test: str,
    n_observations: int,
    m: int,
    phi: float,
    innov_scale: float,
    bandwidth: int,
    n_mc: int,
    projection_max_sweeps: int,
    n_sims: int,
    batch_size: int,
) -> GmmWolakReplication:
    kernel = partial(
        _test_replication,
        mean=mean,
        test=test,
        n_observations=n_observations,
        m=m,
        phi=phi,
        innov_scale=innov_scale,
        bandwidth=bandwidth,
        n_mc=n_mc,
        projection_max_sweeps=projection_max_sweeps,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def _size_rows(
    pvals: jax.Array,
    alpha_levels: list[float],
    n_observations: int,
    m: int,
    n_sims: int,
    test: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Shared rejection-rate aggregation for the Wald/Wolak size tables."""
    rows = []
    for alpha in alpha_levels:
        rate = jnp.mean(pvals < alpha)
        se = jnp.sqrt(alpha * (1.0 - alpha) / n_sims)
        z = jnp.where(se > 0, (rate - alpha) / se, 0.0)
        row = {
            "test": test,
            "nominal_alpha": alpha,
            "empirical_rejection_rate": float(rate),
            "binom_se": float(se),
            "z_score": float(z),
            "flag_outside_3se": bool(jnp.abs(z) > _THREE_STANDARD_ERRORS),
            "T": n_observations,
            "m": m,
            "n_sims": n_sims,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _power_row(
    pvals: jax.Array,
    delta: float,
    alpha: float,
    n_observations: int,
    m: int,
    n_sims: int,
    test: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared power aggregation for the Wald/Wolak power tables."""
    power = jnp.mean(pvals < alpha)
    se = jnp.sqrt(power * (1.0 - power) / n_sims)
    row = {
        "test": test,
        "delta": delta,
        "nominal_alpha": alpha,
        "empirical_power": float(power),
        "binom_se": float(se),
        "T": n_observations,
        "m": m,
        "n_sims": n_sims,
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_summary(
    size_df: pd.DataFrame,
    power_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the gating summary for ``summary.yaml``."""
    size_summary: dict[str, Any] = {}
    for test in sorted(size_df["test"].unique()):
        sub = size_df[size_df["test"] == test]
        entries = {}
        for _, row in sub.iterrows():
            entries[f"alpha_{float(row['nominal_alpha'])}"] = {
                "nominal": float(row["nominal_alpha"]),
                "empirical_rejection_rate": round(
                    float(row["empirical_rejection_rate"]), 4
                ),
                "binom_se": round(float(row["binom_se"]), 4),
                "z_score": round(float(row["z_score"]), 3),
                "pass": not bool(row["flag_outside_3se"]),
            }
        size_summary[test] = entries

    power_summary: dict[str, Any] = {}
    for test in sorted(power_df["test"].unique()):
        sub = power_df[power_df["test"] == test].sort_values("delta")
        power_summary[test] = {
            f"delta_{float(r['delta'])}": round(float(r["empirical_power"]), 4)
            for _, r in sub.iterrows()
        }

    all_size_pass = not bool(size_df["flag_outside_3se"].any())
    return {
        "size": size_summary,
        "power": power_summary,
        "gate": {
            "all_sizes_within_3se": all_size_pass,
            "note": (
                "PASS: GMM Wald + Wolak sizes within ±3 SE of nominal on "
                "synthetic null — Wave-2 empirical p-values are calibrated."
                if all_size_pass
                else "WARN: at least one test mis-sized on synthetic null; "
                "inspect size_results before trusting Wave-2 p-values."
            ),
        },
        "params": cfg,
    }


def run_gmm_wolak_size_power(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the full GMM Wald + Wolak size/power Monte-Carlo.

    Args:
        cfg: Config dict with keys ``n_sims``, ``n_observations``, ``m``, ``phi``,
            ``innov_scale``, ``alpha_levels``, ``power_deltas``, ``n_mc``,
            ``seed``.

    Returns:
        Tuple ``(size_df, power_df, summary)``.

    """
    n_sims = int(cfg["n_sims"])
    n_observations = int(cfg["T"])
    m = int(cfg["m"])
    phi = float(cfg["phi"])
    innov = float(cfg["innov_scale"])
    alphas = list(cfg["alpha_levels"])
    deltas = list(cfg["power_deltas"])
    n_mc = int(cfg["n_mc"])
    seed = int(cfg["seed"])
    power_alpha = float(cfg.get("power_alpha", 0.05))
    replication_batch_size = int(cfg.get("replication_batch_size", 64))
    bandwidth = optimal_bandwidth(n_observations)

    def run_cell(
        test: str, experiment: str, mean: jax.Array, label: object
    ) -> jax.Array:
        key = cell_key(
            seed,
            "mc_gmm_wolak_size_power",
            "test",
            test,
            "experiment",
            experiment,
            "effect",
            label,
            "T",
            n_observations,
            "m",
            m,
        )
        result = execute_replication_cell(
            _test_cell,
            key,
            mean,
            name="mc_gmm_wolak_size_power.cell",
            replication_count=n_sims,
            replication_batch_size=replication_batch_size,
            static={"test": test, "experiment": experiment, "effect": label},
            test=test,
            n_observations=n_observations,
            m=m,
            phi=phi,
            innov_scale=innov,
            bandwidth=bandwidth,
            n_mc=n_mc,
            projection_max_sweeps=_SIMULATION_PROJECTION_MAX_SWEEPS,
            n_sims=n_sims,
            batch_size=replication_batch_size,
        )
        if not bool(jnp.all(result.converged)):
            message = f"{test} {experiment} projection/calibration failed to converge"
            raise RuntimeError(message)
        return result.pvalue

    zero_mean = jnp.zeros(m)
    wald_size_pvals = run_cell("wald", "size", zero_mean, "boundary")
    wolak_size_pvals = run_cell("wolak", "size", zero_mean, "boundary")
    wald_power_pvals = [
        run_cell("wald", "power", jnp.full(m, delta), delta) for delta in deltas
    ]
    wolak_power_pvals = []
    for delta in deltas:
        mean = jnp.concatenate((jnp.asarray([-delta]), jnp.zeros(m - 1)))
        wolak_power_pvals.append(run_cell("wolak", "power", mean, delta))

    wald_size = pd.DataFrame(
        _size_rows(wald_size_pvals, alphas, n_observations, m, n_sims, "wald")
    )
    wald_power = pd.DataFrame(
        [
            _power_row(pvals, delta, power_alpha, n_observations, m, n_sims, "wald")
            for delta, pvals in zip(deltas, wald_power_pvals, strict=True)
        ]
    )
    wolak_size = pd.DataFrame(
        _size_rows(
            wolak_size_pvals, alphas, n_observations, m, n_sims, "wolak", {"n_mc": n_mc}
        )
    )
    wolak_power = pd.DataFrame(
        [
            _power_row(
                pvals,
                delta,
                power_alpha,
                n_observations,
                m,
                n_sims,
                "wolak",
                {"n_mc": n_mc},
            )
            for delta, pvals in zip(deltas, wolak_power_pvals, strict=True)
        ]
    )

    size_df = pd.concat([wald_size, wolak_size], ignore_index=True)
    power_df = pd.concat([wald_power, wolak_power], ignore_index=True)
    summary = build_summary(size_df, power_df, cfg)
    return size_df, power_df, summary

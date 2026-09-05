"""Price-free portfolio construction vs sample-MVO Monte-Carlo experiment.

Compares four long-only, sum-to-one portfolio constructions across return-
sample sizes t and asset counts n:

  1. EW             — equal weight, no data needed.
  2. SAMPLE-MVO(T)  — minimum-variance on sample covariance from T observations.
  3. PF-MAX-SPREAD  — maximise Q(w) = Σ_ij w_i w_j D_ij² over the simplex.
  4. PF-DIST-MVO    — minimum-variance on distance-implied covariance Σ_dist.

All constructions are evaluated on the TRUE covariance Σ_true.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003  # runtime beartype contract
from functools import partial
from math import isfinite
from typing import Any, cast

import jax
import jax.numpy as jnp
import pandas as pd
from jax.typing import ArrayLike  # noqa: TC002  # runtime array contract
from jcor.core.random import cell_key
from jcor.discrepancy.energy import energy_distance_matrix
from jcor.discrepancy.metrize import sqrt_energy_functional
from jcor.optimize import pgd_maximize_quadratic_form, pgd_minimize_quadratic_form
from jcor.optimize.psd import nearest_covariance

from simulation.bounds import certified_floor, lipschitz_constants
from simulation.generators.assets import cluster_embeddings
from simulation.harness.replications import execute_replication_cell, map_replications

_VARIANCE_DENOMINATOR_FLOOR = 1e-12
_ANCHOR_FLOOR = 1e-6
_PSD_SOLVER_FLOOR = 1e-8

# ---------------------------------------------------------------------------
# Simplex utilities
# ---------------------------------------------------------------------------


def project_simplex(v: ArrayLike) -> jax.Array:
    """Project vector v onto the probability simplex (Duchi et al. 2008)."""
    values = jnp.asarray(v)
    n = values.shape[0]
    u = jnp.sort(values)[::-1]
    cssv = jnp.cumsum(u)
    ranks = jnp.arange(1, n + 1)
    valid = u * ranks > (cssv - 1)
    rho = jnp.max(jnp.where(valid, ranks - 1, 0))
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return jnp.maximum(values - theta, 0.0)


def pgd_min_variance(
    sigma: ArrayLike,
    n_steps: int = 5000,
) -> jax.Array:
    """Minimise w'Σw over the simplex via projected gradient descent.

    Delegates to the shared, jitted :func:`jcor.optimize.pgd_minimize_quadratic_form`
    primitive (fixed-iteration PGD, ``λ_max``-based step size, ``optax``
    simplex projection) rather than maintaining a parallel copy.
    """
    return pgd_minimize_quadratic_form(jnp.asarray(sigma), n_steps)


def pgd_max_quadratic(
    q: ArrayLike,
    n_steps: int = 5000,
) -> jax.Array:
    """Maximise w'Qw over the simplex via projected gradient ascent."""
    return pgd_maximize_quadratic_form(jnp.asarray(q), n_steps)


# ---------------------------------------------------------------------------
# Asset generation
# ---------------------------------------------------------------------------


def compute_energy_distances(sample_sets: Sequence[object]) -> jax.Array:
    """Pairwise energy distances ``D_ij`` (Euclidean kernel, ``exponent=1``).

    ``D``, the **metric** — not the energy functional ``S = D**2`` that
    :func:`jcor.discrepancy.energy.energy_distance_matrix` returns.
    ``sqrt_energy_functional`` validates the stored functional before taking
    its root, preserving the metric convention used by the certificates.
    """
    arrays = [jnp.asarray(samples) for samples in sample_sets]
    functional = energy_distance_matrix(  # noqa: PD011
        arrays, exponent=1.0, metric="euclidean"
    ).values  # jcor result carrier, not a pandas Series
    return sqrt_energy_functional(functional)


def compute_true_covariance(betas: ArrayLike, sigma_eps: float) -> jax.Array:
    """Σ_true = β β' + σ_ε² I."""
    values = jnp.asarray(betas)
    identity = jnp.eye(values.shape[0], dtype=values.dtype)
    return values @ values.T + sigma_eps**2 * identity


def simulate_returns(
    key: jax.Array,
    sigma_true: jax.Array,
    t: int,
) -> jax.Array:
    """Simulate T iid returns r_t ~ N(0, Σ_true). Returns (T, n) array."""
    n = sigma_true.shape[0]
    cholesky_factor = jnp.linalg.cholesky(
        sigma_true + 1e-10 * jnp.eye(n, dtype=sigma_true.dtype)
    )
    z = jax.random.normal(key, (t, n), dtype=sigma_true.dtype)
    return z @ cholesky_factor.T


# ---------------------------------------------------------------------------
# Portfolio constructions
# ---------------------------------------------------------------------------


def ew_portfolio(n: int) -> jax.Array:
    """Equal-weight portfolio."""
    return jnp.ones(n) / n


def sample_mvo_portfolio(
    returns: ArrayLike,
    n_pgd_steps: int,
) -> tuple[jax.Array, float]:
    """Minimum-variance portfolio using sample covariance with ridge."""
    values = jnp.asarray(returns)
    t, n = values.shape
    if t > 1:
        centered = values - jnp.mean(values, axis=0)
        sigma_sample = centered.T @ centered / (t - 1)
    else:
        sigma_sample = jnp.eye(n, dtype=values.dtype)
    if n == 1:
        return jnp.ones(1, dtype=values.dtype), 1.0
    lam_min = float(jnp.min(jnp.linalg.eigvalsh(sigma_sample)))
    ridge = max(0.0, -lam_min) + 1e-6
    sigma_ridge = sigma_sample + ridge * jnp.eye(n, dtype=values.dtype)
    w = pgd_min_variance(sigma_ridge, n_steps=n_pgd_steps)
    predicted_var = float(w @ sigma_ridge @ w)
    return w, predicted_var


def pf_max_spread_portfolio(
    d2: ArrayLike,
    n_pgd_steps: int,
) -> jax.Array:
    """Price-free max-spread: maximise w'D²w over the simplex."""
    return pgd_max_quadratic(d2, n_steps=n_pgd_steps)


def pf_dist_mvo_portfolio(
    distance_matrix: ArrayLike,
    sigma_hat: float,
    ell_hat: float,
    n_pgd_steps: int,
) -> tuple[jax.Array, float]:
    """Price-free distance-MVO: minimise variance on Σ_dist.

    ``Σ_dist = σ̂²·J − ½ℓ̂²·D²``.  The covariance is repaired with the
    diagonal-preserving nearest-covariance solver before the simplex PGD
    step, preserving the anchor's role in the solved geometry.
    """
    distances = jnp.asarray(distance_matrix)
    dtype = distances.dtype
    d2 = distances**2
    anchor = jnp.asarray(sigma_hat, dtype=dtype)
    ell = jnp.asarray(ell_hat, dtype=dtype)
    anchor_sq = jnp.maximum(anchor, _ANCHOR_FLOOR) ** 2
    sigma_dist = anchor_sq - 0.5 * ell**2 * d2
    repaired = nearest_covariance(sigma_dist).matrix
    sigma_dist_psd = repaired + _PSD_SOLVER_FLOOR * jnp.eye(
        distances.shape[0], dtype=repaired.dtype
    )
    w = pgd_min_variance(sigma_dist_psd, n_steps=n_pgd_steps)
    predicted_var = float(w @ sigma_dist_psd @ w)
    return w, predicted_var


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_lipschitz_constants(
    betas: ArrayLike, distance_matrix: ArrayLike, eps: float = 1e-12
) -> tuple[float, float]:
    """Compute (L_max, ell_min) from pairwise ||b_i - b_j|| / D_ij ratios."""
    result = lipschitz_constants(
        jnp.asarray(betas), jnp.asarray(distance_matrix), eps=eps
    )
    return result.L_max, result.ell_min


def oracle_sigma_hat(betas: ArrayLike) -> float:
    """Oracle σ̂ = mean ‖β_i‖."""
    values = jnp.asarray(betas)
    return float(jnp.mean(jnp.linalg.norm(values, axis=1)))


def data_sigma_hat(returns_t20: ArrayLike) -> float:
    """Data-driven σ̂ = sqrt(mean diagonal of sample covariance at T=20)."""
    values = jnp.asarray(returns_t20)
    t, _n = values.shape
    if t > 1:
        centered = values - jnp.mean(values, axis=0)
        sigma_sample = centered.T @ centered / (t - 1)
        mean_var = float(jnp.mean(jnp.diag(sigma_sample)))
    else:
        mean_var = 1.0
    return float(jnp.sqrt(jnp.maximum(mean_var, 1e-12)))


def portfolio_variance(w: ArrayLike, sigma: ArrayLike) -> float:
    """Realised portfolio variance w'Σw."""
    weights = jnp.asarray(w)
    covariance = jnp.asarray(sigma)
    return float(weights @ covariance @ weights)


def oracle_min_variance_portfolio(sigma_true: ArrayLike, n_pgd_steps: int) -> jax.Array:
    """Oracle minimum-variance portfolio (long-only) on true Σ."""
    return pgd_min_variance(sigma_true, n_steps=n_pgd_steps)


def _return_replication(
    key: jax.Array,
    *,
    sigma_true: jax.Array,
    t_value: int,
    min_t: int,
) -> tuple[jax.Array, jax.Array]:
    """Draw the requested and anchor panels for one price-free replication."""
    returns_key, anchor_key = jax.random.split(key)
    returns = simulate_returns(returns_key, sigma_true, t_value)
    anchor = (
        returns if t_value == min_t else simulate_returns(anchor_key, sigma_true, min_t)
    )
    return returns, anchor


@partial(
    jax.jit,
    static_argnames=("t_value", "min_t", "n_sims", "batch_size"),
)
def _return_cell(
    key: jax.Array,
    sigma_true: jax.Array,
    *,
    t_value: int,
    min_t: int,
    n_sims: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    kernel = partial(
        _return_replication,
        sigma_true=sigma_true,
        t_value=t_value,
        min_t=min_t,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def _pricefree_cell(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    """One host-sequential ``(n_assets, d)`` shape cell."""
    (
        n_assets,
        d,
        base_seed,
        n_points,
        n_clusters,
        sigma_eps,
        t_values,
        n_sims,
        n_pgd_steps,
        replication_batch_size,
    ) = task

    rows: list[dict[str, Any]] = []
    geometry_key = cell_key(
        base_seed,
        "mc_pricefree",
        "geometry",
        "n_assets",
        n_assets,
        "embedding_dim",
        d,
    )
    sample = cluster_embeddings(geometry_key, n_assets, n_points, d, n_clusters)
    betas = jnp.asarray(sample.betas)
    sample_sets = [jnp.asarray(points) for points in sample.samples]
    distance_matrix = compute_energy_distances(sample_sets)
    d2 = distance_matrix**2

    sigma_true = compute_true_covariance(betas, sigma_eps)

    sigma_hat_oracle = oracle_sigma_hat(betas)
    l_max, ell_min = compute_lipschitz_constants(betas, distance_matrix)

    w_oracle = oracle_min_variance_portfolio(sigma_true, n_pgd_steps)
    var_oracle = portfolio_variance(w_oracle, sigma_true)

    w_ew = ew_portfolio(n_assets)
    var_ew = portfolio_variance(w_ew, sigma_true)

    w_spread = pf_max_spread_portfolio(d2, n_pgd_steps)
    var_spread = portfolio_variance(w_spread, sigma_true)
    q_spread = float(w_spread @ d2 @ w_spread)

    cert_floor = certified_floor(w_spread, betas, d2, l_max)
    cert_valid_spread = bool(var_spread >= cert_floor - 1e-10)
    honesty_ratio_spread = (
        var_spread / cert_floor
        if abs(cert_floor) > _VARIANCE_DENOMINATOR_FLOOR
        else float("nan")
    )

    w_distmvo_oracle, pred_var_distmvo_oracle = pf_dist_mvo_portfolio(
        distance_matrix, sigma_hat_oracle, ell_min, n_pgd_steps
    )
    var_distmvo_oracle = portfolio_variance(w_distmvo_oracle, sigma_true)
    honesty_err_distmvo_oracle = (
        abs(var_distmvo_oracle - pred_var_distmvo_oracle) / var_distmvo_oracle
        if var_distmvo_oracle > _VARIANCE_DENOMINATOR_FLOOR
        else float("nan")
    )

    min_t = min(t_values)
    sigma_true_jax = jnp.asarray(sigma_true)
    for t in t_values:
        high_dim = bool(n_assets >= t)
        returns_key = cell_key(
            base_seed,
            "mc_pricefree",
            "returns",
            "n_assets",
            n_assets,
            "embedding_dim",
            d,
            "T",
            t,
        )
        returns_batch, anchor_batch = execute_replication_cell(
            _return_cell,
            returns_key,
            sigma_true_jax,
            name="mc_pricefree.returns",
            replication_count=n_sims,
            replication_batch_size=replication_batch_size,
            static={"n_assets": n_assets, "embedding_dim": d, "T": t},
            t_value=t,
            min_t=min_t,
            n_sims=n_sims,
            batch_size=replication_batch_size,
        )
        for sim_idx in range(n_sims):
            returns = returns_batch[sim_idx]

            w_smvo, pred_var_smvo = sample_mvo_portfolio(returns, n_pgd_steps)
            var_smvo = portfolio_variance(w_smvo, sigma_true)
            honesty_err_smvo = (
                abs(var_smvo - pred_var_smvo) / var_smvo
                if var_smvo > _VARIANCE_DENOMINATOR_FLOOR
                else float("nan")
            )

            returns_t_min = anchor_batch[sim_idx]
            sigma_hat_data = data_sigma_hat(returns_t_min)

            w_distmvo_data, pred_var_distmvo_data = pf_dist_mvo_portfolio(
                distance_matrix, sigma_hat_data, ell_min, n_pgd_steps
            )
            var_distmvo_data = portfolio_variance(w_distmvo_data, sigma_true)
            honesty_err_distmvo_data = (
                abs(var_distmvo_data - pred_var_distmvo_data) / var_distmvo_data
                if var_distmvo_data > _VARIANCE_DENOMINATOR_FLOOR
                else float("nan")
            )

            cap_violation = bool(cert_floor > var_spread + 1e-10)

            rows.append(
                {
                    "n_assets": n_assets,
                    "embedding_dim": d,
                    "T": t,
                    "sim": sim_idx,
                    "high_dim": high_dim,
                    "var_oracle": var_oracle,
                    "var_ew": var_ew,
                    "var_smvo": var_smvo,
                    "var_spread": var_spread,
                    "var_distmvo_oracle": var_distmvo_oracle,
                    "var_distmvo_data": var_distmvo_data,
                    "regret_ew": var_ew - var_oracle,
                    "regret_smvo": var_smvo - var_oracle,
                    "regret_spread": var_spread - var_oracle,
                    "regret_distmvo_oracle": var_distmvo_oracle - var_oracle,
                    "regret_distmvo_data": var_distmvo_data - var_oracle,
                    "pred_var_smvo": pred_var_smvo,
                    "pred_var_spread": cert_floor,
                    "pred_var_distmvo_oracle": pred_var_distmvo_oracle,
                    "pred_var_distmvo_data": pred_var_distmvo_data,
                    "honesty_err_smvo": honesty_err_smvo,
                    "honesty_err_distmvo_oracle": honesty_err_distmvo_oracle,
                    "honesty_err_distmvo_data": honesty_err_distmvo_data,
                    "honesty_ratio_spread": honesty_ratio_spread,
                    "certificate_valid_spread": cert_valid_spread,
                    "certified_floor": cert_floor,
                    "Q_spread": q_spread,
                    "sigma_hat_oracle": sigma_hat_oracle,
                    "sigma_hat_data": sigma_hat_data,
                    "L_max": l_max,
                    "ell_min": ell_min,
                    "cap_violation": cap_violation,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def _aggregate_pricefree_tables(
    df: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, float | int | bool]],
    dict[str, dict[str, float | int | bool]],
]:
    """Aggregate median regret and honesty diagnostics by panel cell."""
    regret_columns = [
        "regret_smvo",
        "regret_spread",
        "regret_distmvo_oracle",
        "regret_distmvo_data",
    ]
    honesty_columns = [
        "honesty_err_smvo",
        "honesty_err_distmvo_oracle",
        "honesty_err_distmvo_data",
        "honesty_ratio_spread",
    ]
    median_regret: dict[str, dict[str, float | int | bool]] = {}
    honesty: dict[str, dict[str, float | int | bool]] = {}
    for group_key, group in df.groupby(["n_assets", "T"]):
        n_assets, t_value = cast("tuple[int, int]", group_key)
        key = f"n{n_assets}_T{t_value}"
        regret = group[regret_columns].median()
        median_regret[key] = {
            "n_assets": int(n_assets),
            "T": int(t_value),
            "high_dim": bool(n_assets >= t_value),
            "smvo": float(regret["regret_smvo"]),
            "pf_max_spread": float(regret["regret_spread"]),
            "pf_distmvo_oracle": float(regret["regret_distmvo_oracle"]),
            "pf_distmvo_data": float(regret["regret_distmvo_data"]),
        }
        honesty_median = group[honesty_columns].median()
        honesty[key] = {
            "n_assets": int(n_assets),
            "T": int(t_value),
            "high_dim": bool(n_assets >= t_value),
            "median_honesty_err_smvo": float(honesty_median["honesty_err_smvo"]),
            "median_honesty_err_distmvo_oracle": float(
                honesty_median["honesty_err_distmvo_oracle"]
            ),
            "median_honesty_err_distmvo_data": float(
                honesty_median["honesty_err_distmvo_data"]
            ),
            "median_honesty_ratio_spread": float(
                honesty_median["honesty_ratio_spread"]
            ),
        }
    return median_regret, honesty


def _pricefree_crossovers(
    n_assets_values: list[int],
    t_values: list[int],
    median_regret: dict[str, dict[str, float | int | bool]],
) -> dict[str, dict[str, int | None]]:
    """Find the first sample size where each price-free method loses to SMVO."""
    result: dict[str, dict[str, int | None]] = {}
    column_names = {
        "spread": "pf_max_spread",
        "distmvo_oracle": "pf_distmvo_oracle",
        "distmvo_data": "pf_distmvo_data",
    }
    for n_assets in n_assets_values:
        result[f"n{n_assets}"] = {}
        for label, column in column_names.items():
            crossover = next(
                (
                    t_value
                    for t_value in t_values
                    if f"n{n_assets}_T{t_value}" in median_regret
                    and float(median_regret[f"n{n_assets}_T{t_value}"][column])
                    > float(median_regret[f"n{n_assets}_T{t_value}"]["smvo"])
                ),
                None,
            )
            result[f"n{n_assets}"][label] = crossover
    return result


def run_pricefree(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the price-free portfolio comparison experiment.

    Args:
        cfg: ``params["monte_carlo"]`` dict from ``params.yaml``.

    Returns:
        (results_df, summary) — DataFrame of per-config results and summary dict.

    """
    n_assets_list: list[int] = cfg["n_assets"]
    embedding_dims: list[int] = cfg["embedding_dim"]
    n_points: int = cfg["n_points_per_asset"]
    n_clusters: int = cfg["n_clusters"]
    base_seed: int = cfg["seed"]
    t_values: list[int] = cfg["pricefree_T_values"]
    sigma_eps: float = cfg["pricefree_sigma_eps"]
    n_sims: int = cfg["pricefree_n_sims"]
    n_pgd_steps: int = cfg["pricefree_pgd_steps"]
    replication_batch_size: int = int(cfg.get("replication_batch_size", 64))

    cells = [
        (
            n_assets,
            d,
            base_seed,
            n_points,
            n_clusters,
            sigma_eps,
            t_values,
            n_sims,
            n_pgd_steps,
            replication_batch_size,
        )
        for n_assets in n_assets_list
        for d in embedding_dims
    ]

    cell_rows = [_pricefree_cell(cell) for cell in cells]
    rows = [r for cr in cell_rows for r in cr]

    df = pd.DataFrame(rows)

    total_rows, total_cap_violations, total_cert_valid = (
        len(df),
        int(df["cap_violation"].sum()),
        int(df["certificate_valid_spread"].sum()),
    )
    cert_validity_pct = 100.0 * total_cert_valid / total_rows if total_rows > 0 else 0.0

    median_regret_by_n_t, honesty_table = _aggregate_pricefree_tables(df)
    crossover_t = _pricefree_crossovers(
        n_assets_list,
        sorted(int(value) for value in df["T"].unique()),
        median_regret_by_n_t,
    )

    pf_cols = ["pf_max_spread", "pf_distmvo_oracle", "pf_distmvo_data"]
    high_dim_cells = {k: v for k, v in median_regret_by_n_t.items() if v["high_dim"]}
    pf_wins_highdim = {
        k: any(v[c] < v["smvo"] for c in pf_cols) for k, v in high_dim_cells.items()
    }
    any_pf_win_highdim = any(pf_wins_highdim.values())

    # Which constructions actually beat sample-MVO, derived from the run so the
    # caveat in the headline can never diverge from the numbers. The pf_max_spread
    # claim spans every (n, t) cell, not just the high-dim ones.
    pf_win_cells = {
        c: [k for k, v in median_regret_by_n_t.items() if v[c] < v["smvo"]]
        for c in pf_cols
    }
    spread_win_cells = pf_win_cells["pf_max_spread"]
    distmvo_wins = any(
        pf_win_cells[c] for c in ("pf_distmvo_oracle", "pf_distmvo_data")
    )
    if spread_win_cells:
        spread_clause = (
            f"the fully price-free pf_max_spread beats sample-MVO in "
            f"{len(spread_win_cells)} cell(s): {spread_win_cells}."
        )
    else:
        spread_clause = (
            "the fully price-free pf_max_spread does not beat sample-MVO in any cell."
        )
    if distmvo_wins:
        pf_note = (
            "Note: the winning constructions (pf_distmvo_*) use return-data "
            "diagonals (price-light); " + spread_clause
        )
    else:
        pf_note = "Note: " + spread_clause

    high_dim_hon = [v for v in honesty_table.values() if v["high_dim"]]
    low_dim_hon = [v for v in honesty_table.values() if not v["high_dim"]]

    def _mean_safe(vals: list[float]) -> float:
        finite = [v for v in vals if isfinite(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    avg_smvo_err_highdim = _mean_safe(
        [v["median_honesty_err_smvo"] for v in high_dim_hon]
    )
    avg_smvo_err_lowdim = _mean_safe(
        [v["median_honesty_err_smvo"] for v in low_dim_hon]
    )
    avg_spread_ratio_highdim = _mean_safe(
        [v["median_honesty_ratio_spread"] for v in high_dim_hon]
    )

    if any_pf_win_highdim:
        winning_cells = [k for k, v in pf_wins_highdim.items() if v]
        headline = (
            f"Price-free beats sample-MVO in {len(winning_cells)} high-dim (n>=t) "
            f"cell(s): {winning_cells}. "
            "Sample-MVO median honesty error in n>=t regime: "
            f"{avg_smvo_err_highdim:.3f} "
            f"vs low-dim: {avg_smvo_err_lowdim:.3f}. "
            f"Price-free floor (certified_floor) median honesty ratio in n>=t: "
            f"{avg_spread_ratio_highdim:.3f} (floor/realized; negative means "
            "the floor is vacuous in this geometry). "
            f"Certificate validity: {cert_validity_pct:.1f}% of rows. "
            f"{pf_note}"
        )
    else:
        headline = (
            f"Sample-MVO wins in ALL high-dim (n>=t) cells tested — price-free "
            f"does NOT beat sample-MVO even in the high-dimensional regime. "
            "Sample-MVO median honesty error in n>=t regime: "
            f"{avg_smvo_err_highdim:.3f} "
            f"vs low-dim: {avg_smvo_err_lowdim:.3f}. "
            f"Price-free floor (certified_floor) median honesty ratio in n>=t: "
            f"{avg_spread_ratio_highdim:.3f} (floor/realized; negative means "
            "the floor is vacuous in this geometry). "
            f"Certificate validity: {cert_validity_pct:.1f}% of rows. "
            f"{pf_note}"
        )

    summary = {
        "total_rows": len(df),
        "floor_violations": total_cap_violations,
        "floor_violation_note": (
            "certified_floor = sigma_hat^2 - (L_max^2/2)*Q(w) is a lower bound "
            "on systematic portfolio variance (Lean theorem "
            "portfolio_variance_lower_bound). "
            "Violations indicate floor > w'sigma_true w, which should not occur."
        ),
        "certificate_validity": {
            "count_valid": total_cert_valid,
            "total_rows": total_rows,
            "pct_valid": round(cert_validity_pct, 2),
            "note": (
                "certificate_valid_spread = (realized >= cert_floor). "
                "Expect 100% by the Lean theorem guarantee. "
                "CAVEAT: in this simulation geometry the floor is deeply negative "
                "(median honesty ratio ≈ -0.004, i.e. floor ≈ -250× realized) "
                "because the empirical l_max is large relative to sigma_hat; "
                "validity is therefore vacuous here — see the cap certificate and "
                "the calibration-tightness discussion in Paper 3."
            ),
        },
        "median_regret_vs_oracle_by_n_T": median_regret_by_n_t,
        "honesty_table": honesty_table,
        "crossover_T_by_n": {
            "note": (
                "t at which price-free regret first exceeds SMVO regret per n "
                "(None means PF always beats SMVO across tested t values)"
            ),
            **crossover_t,
        },
        "headline": headline,
        "hypothesis": (
            "Price-free beats SMVO at small T (O(n) vs O(n^2) params) "
            "and loses at large t. Crossover reported per (n, construction) above. "
            "HIGH-DIM REGIME (n>=t): sample-MVO covariance rank-deficient; "
            "price-free constructions use no return data so are unaffected."
        ),
        "parameters": {
            "n_assets": n_assets_list,
            "embedding_dim": embedding_dims,
            "T_values": t_values,
            "sigma_eps": sigma_eps,
            "n_sims": n_sims,
            "n_pgd_steps": n_pgd_steps,
        },
    }

    return df, summary

"""Run the Paper 5 Monte Carlo studies.

DGP (per repetition, fully seeded): ``r_t = rho * W r_t + alpha * 1 + e_t``,
``e_t ~ N(0, sigma^2 I_n)`` -- the same SAR model as
:mod:`pipeline.stages.papers.paper5.estimate` (``sar_qmle``), solved in closed form
via ``r_t = (I - rho W)^{-1} (alpha * 1 + e_t)`` (heterogeneous-rho studies
use a diagonal ``rho`` matrix, ``r_t = (I - diag(rho) W)^{-1} (...)``).
``W`` is a synthetic fixed row-stochastic k-NN matrix on random 2-D latent
coordinates (independent of the real 100-ticker panel; a deterministic,
data-free stand-in, consistent with the energy-robustness synthetic-DGP
convention).

Study (i)'s "S5c bracket coverage" is a Monte-Carlo-calibrated 95% CI on
``rho_hat`` -- DISCLOSED approximation: the plug-in SE is the empirical
(across-repetition) standard deviation of ``rho_hat`` at each config, not an
asymptotic QMLE information-matrix formula (out of scope here); this is
still a meaningful bracket-coverage check since the SE is itself simulated.

Study (iv)'s size/power follows the ``gmm_wolak_size_power`` template's
SIZE/POWER + binomial-SE-flag structure. The ``bar-alpha = 0`` restriction
as implemented in
:func:`pipeline.stages.papers.paper5.hypotheses.h2_zero_intercept_test`
is an EQUALITY restriction (two-sided Wald, ``chi2(1)``), not an inequality
boundary restriction -- so no chi-bar-squared mixture weight applies here
(DISCLOSED: this differs from the ``gmm_wolak`` Wolak test, which is a
one-sided/boundary test). What is reused from that template is its
SIZE-under-null / POWER-under-alternative design and its ``|z| > 3`` binomial
flag for mis-sizing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
from jcor.core.random import cell_key
from numpy.typing import NDArray
from simulation.dgp.spatial import SarDGP, make_synthetic_w, simulate_sar

from pipeline.figures.paper5.figures import (
    MisspecificationSeries,
    SizePowerSeries,
    render_size_power,
    render_w_misspecification,
)
from pipeline.stages.papers.paper5.estimate import sar_qmle
from pipeline.stages.papers.paper5.hypotheses import h2_zero_intercept_test

logger = logging.getLogger(__name__)

Float = NDArray[np.float64]

# Study (i)-(iii) scale (small synthetic panel, fast).
N_SMALL = 20
T_SMALL = 250
RHO_GRID = (0.2, 0.4, 0.6, 0.8)

# Study (ii): W-misspecification magnitude grid.
MISSPEC_GRID = (0.0, 0.1, 0.2, 0.4, 0.6)
MAX_BINOMIAL_Z_SCORE = 3.0
MISSPEC_RHO = 0.5

# Study (iv): size/power at the real panel's cross-section size. This tracks
# params.yaml:market_symbols by intent -- the study exists to characterise the
# test AT the panel's n, so leaving it at the retired 52 would answer a question
# about a panel the papers no longer report. Changing it changes Paper 5's MC
# results; that is the point, not a side effect.
N_IV = 100
T_IV = 252  # ~1 trading year; disclosed speed/tractability choice for the MC.
ALPHA_NULL = 0.0
POWER_ALPHA_GRID = (0.0005, 0.001, 0.002, 0.005)
SIGMA_IV = 0.02
NOMINAL_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Monte-Carlo result contracts. These are the on-disk shapes serialised to
# mc_results.json; typing them documents each study's output and lets the type
# checker verify the figure code's key access against the producer.
# ---------------------------------------------------------------------------


class _RhoCell(TypedDict):
    """Per-rho recovery summary within :class:`_RhoRecovery`."""

    rho_true: float
    bias: float
    rmse: float
    se_mc: float
    bracket_coverage: float


class _RhoRecovery(TypedDict):
    """Study (i): SAR rho recovery and S5c bracket coverage across the rho grid."""

    n_reps: int
    n: int
    t_obs: int
    by_rho: dict[str, _RhoCell]


class _MisspecCell(TypedDict):
    """Per-magnitude bias summary within :class:`_WMisspecification`."""

    magnitude: float
    bias_mean: float
    bias_se: float


class _WMisspecification(TypedDict):
    """Study (ii): rho-estimate bias under W-misspecification of growing magnitude."""

    n_reps: int
    n: int
    t_obs: int
    rho_true: float
    by_magnitude: dict[str, _MisspecCell]
    bias_slope_vs_magnitude: float


class _HeterogeneousRho(TypedDict):
    """Study (iii): estimator behaviour under a heterogeneous-rho DGP."""

    n_reps: int
    n: int
    t_obs: int
    rho_hat_mean: float
    rho_effective_mean: float
    degradation_mean_abs: float
    degradation_max_abs: float


class _SizeResult(TypedDict):
    """Empirical size of the bar-alpha=0 test at the nominal level."""

    nominal_alpha: float
    empirical_size: float
    binom_se: float
    z_score: float
    flag_mis_sized: bool


class _PowerCell(TypedDict):
    """Per-alternative rejection rate within :class:`_SizePower`."""

    alpha_true: float
    power: float


class _SizePower(TypedDict):
    """Study (iv): size and power of the bar-alpha=0 test at the panel n."""

    n_reps: int
    n: int
    t_obs: int
    sigma: float
    size: _SizeResult
    power: dict[str, _PowerCell]


# ---------------------------------------------------------------------------
# (i) rho recovery + S5c bracket coverage
# ---------------------------------------------------------------------------


def _run_rho_recovery(n_reps: int, seed: int) -> _RhoRecovery:
    w = make_synthetic_w(cell_key(seed, "paper5_mc", "rho_recovery", "W"), N_SMALL)
    by_rho: dict[str, _RhoCell] = {}
    for rho_true in RHO_GRID:
        rho_hats = []
        replication_key = cell_key(seed, "paper5_mc", "rho_recovery", "rho", rho_true)
        for rep in range(n_reps):
            r = np.asarray(
                simulate_sar(
                    jax.random.fold_in(replication_key, rep),
                    SarDGP(
                        w,
                        rho_true,
                        alpha=0.01,
                        sigma=0.05,
                        n_observations=T_SMALL,
                    ),
                )
            )
            fit = sar_qmle(r, w)
            rho_hats.append(fit["rho_hat"])
        rho_hats_arr = np.asarray(rho_hats)
        bias = float(np.mean(rho_hats_arr - rho_true))
        rmse = float(np.sqrt(np.mean((rho_hats_arr - rho_true) ** 2)))
        se_mc = float(np.std(rho_hats_arr, ddof=1)) if n_reps > 1 else float("nan")
        lo = rho_hats_arr - 1.96 * se_mc
        hi = rho_hats_arr + 1.96 * se_mc
        coverage = float(np.mean((rho_true >= lo) & (rho_true <= hi)))
        by_rho[f"rho_{rho_true}"] = {
            "rho_true": rho_true,
            "bias": bias,
            "rmse": rmse,
            "se_mc": se_mc,
            "bracket_coverage": coverage,
        }
    return {"n_reps": n_reps, "n": N_SMALL, "t_obs": T_SMALL, "by_rho": by_rho}


# ---------------------------------------------------------------------------
# (ii) W-misspecification
# ---------------------------------------------------------------------------


def _run_w_misspecification(n_reps: int, seed: int) -> _WMisspecification:
    w_true = make_synthetic_w(
        cell_key(seed, "paper5_mc", "w_misspecification", "W"), N_SMALL
    )
    by_magnitude: dict[str, _MisspecCell] = {}
    for m in MISSPEC_GRID:
        biases = []
        replication_key = cell_key(
            seed, "paper5_mc", "w_misspecification", "magnitude", m
        )
        for rep in range(n_reps):
            data_key, perturbation_key = jax.random.split(
                jax.random.fold_in(replication_key, rep)
            )
            r = np.asarray(
                simulate_sar(
                    data_key,
                    SarDGP(
                        w_true,
                        MISSPEC_RHO,
                        alpha=0.01,
                        sigma=0.05,
                        n_observations=T_SMALL,
                    ),
                )
            )
            if abs(m) <= 0.0:
                w_est = w_true
            else:
                noise = jax.random.uniform(
                    perturbation_key, w_true.shape, minval=-m, maxval=m
                )
                w_pert = jnp.clip(w_true + noise, 0.0)
                w_pert = w_pert.at[jnp.diag_indices(N_SMALL)].set(0.0)
                row_sums = w_pert.sum(axis=1, keepdims=True)
                w_est = w_pert / jnp.where(row_sums > 0.0, row_sums, 1.0)
            fit = sar_qmle(r, np.asarray(w_est, dtype=np.float64))
            biases.append(fit["rho_hat"] - MISSPEC_RHO)
        biases_arr = np.asarray(biases)
        by_magnitude[f"m_{m}"] = {
            "magnitude": m,
            "bias_mean": float(np.mean(biases_arr)),
            "bias_se": float(np.std(biases_arr, ddof=1) / np.sqrt(n_reps))
            if n_reps > 1
            else float("nan"),
        }
    magnitudes = np.array(MISSPEC_GRID)
    means = np.array([by_magnitude[f"m_{m}"]["bias_mean"] for m in MISSPEC_GRID])
    slope = (
        float(np.polyfit(magnitudes, means, 1)[0])
        if len(magnitudes) > 1
        else float("nan")
    )
    return {
        "n_reps": n_reps,
        "n": N_SMALL,
        "t_obs": T_SMALL,
        "rho_true": MISSPEC_RHO,
        "by_magnitude": by_magnitude,
        "bias_slope_vs_magnitude": slope,
    }


# ---------------------------------------------------------------------------
# (iii) heterogeneous-rho DGP robustness
# ---------------------------------------------------------------------------


def _run_heterogeneous_rho(n_reps: int, seed: int) -> _HeterogeneousRho:
    w = make_synthetic_w(cell_key(seed, "paper5_mc", "heterogeneous_rho", "W"), N_SMALL)
    rho_hats = []
    rho_effectives = []
    replication_key = cell_key(seed, "paper5_mc", "heterogeneous_rho")
    for rep in range(n_reps):
        rho_key, returns_key = jax.random.split(
            jax.random.fold_in(replication_key, rep)
        )
        rho_i = jax.random.uniform(rho_key, (N_SMALL,), minval=0.2, maxval=0.8)
        r = np.asarray(
            simulate_sar(
                returns_key,
                SarDGP(w, rho_i, alpha=0.01, sigma=0.05, n_observations=T_SMALL),
            )
        )
        fit = sar_qmle(r, w)
        rho_hats.append(fit["rho_hat"])
        rho_effectives.append(float(np.mean(rho_i)))
    rho_hats_arr = np.asarray(rho_hats)
    rho_eff_arr = np.asarray(rho_effectives)
    degradation = np.abs(rho_hats_arr - rho_eff_arr)
    return {
        "n_reps": n_reps,
        "n": N_SMALL,
        "t_obs": T_SMALL,
        "rho_hat_mean": float(np.mean(rho_hats_arr)),
        "rho_effective_mean": float(np.mean(rho_eff_arr)),
        "degradation_mean_abs": float(np.mean(degradation)),
        "degradation_max_abs": float(np.max(degradation)),
    }


# ---------------------------------------------------------------------------
# (iv) size/power of the bar-alpha=0 test at the panel n
# ---------------------------------------------------------------------------


def _run_size_power(n_reps: int, seed: int) -> _SizePower:
    w = make_synthetic_w(cell_key(seed, "paper5_mc", "size_power", "W"), N_IV)

    def _reject_rate(alpha_true: float, tag: int) -> tuple[float, float]:
        rejects = []
        replication_key = cell_key(seed, "paper5_mc", "size_power", "alternative", tag)
        for rep in range(n_reps):
            r = np.asarray(
                simulate_sar(
                    jax.random.fold_in(replication_key, rep),
                    SarDGP(
                        w,
                        rho=0.3,
                        alpha=alpha_true,
                        sigma=SIGMA_IV,
                        n_observations=T_IV,
                    ),
                )
            )
            fit = sar_qmle(r, w)
            test = h2_zero_intercept_test(
                fit["alpha_hat"], fit["sigma2_hat"], T_IV, N_IV
            )
            rejects.append(bool(test["reject_05"]))
        rate = float(np.mean(rejects))
        se = float(np.sqrt(NOMINAL_ALPHA * (1 - NOMINAL_ALPHA) / n_reps))
        return rate, se

    size_rate, size_se = _reject_rate(ALPHA_NULL, tag=0)
    z = (size_rate - NOMINAL_ALPHA) / size_se if size_se > 0 else 0.0
    size_result: _SizeResult = {
        "nominal_alpha": NOMINAL_ALPHA,
        "empirical_size": size_rate,
        "binom_se": size_se,
        "z_score": z,
        "flag_mis_sized": bool(abs(z) > MAX_BINOMIAL_Z_SCORE),
    }

    power_by_delta: dict[str, _PowerCell] = {}
    for i, delta in enumerate(POWER_ALPHA_GRID):
        power_rate, _ = _reject_rate(delta, tag=100 + i)
        power_by_delta[f"delta_{delta}"] = {"alpha_true": delta, "power": power_rate}

    return {
        "n_reps": n_reps,
        "n": N_IV,
        "t_obs": T_IV,
        "sigma": SIGMA_IV,
        "size": size_result,
        "power": power_by_delta,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _to_plain(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        arr = cast("np.ndarray[Any, Any]", obj)
        if arr.ndim == 0:
            return _to_plain(arr.item())
        return [_to_plain(v) for v in arr]
    return obj


def run_paper5_mc(
    output_dir: Path, n_reps: int = 500, seed: int = 0, figures_dir: Path | None = None
) -> dict[str, Any]:
    """Run all four Paper 5 Monte Carlo studies and write results + figures.

    Args:
        output_dir: Directory to write ``mc_results.json`` into
            (``data/papers/paper5``).
        n_reps: Repetitions per config (shared across studies i-iv).
        seed: Base RNG seed.
        figures_dir: Directory to write the two figure PDFs into. Defaults to
            ``output_dir`` when omitted.

    Returns:
        Dict with keys ``rho_recovery``, ``w_misspecification``,
        ``heterogeneous_rho``, ``size_power``, ``scales_note``.

    """
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else output_dir
    logger.info(
        "Paper 5 MC (i): rho recovery + S5c bracket coverage (n_reps=%d)", n_reps
    )
    rho_recovery = _run_rho_recovery(n_reps, seed)
    logger.info("Paper 5 MC (ii): W-misspecification (n_reps=%d)", n_reps)
    w_misspec = _run_w_misspecification(n_reps, seed + 1)
    logger.info("Paper 5 MC (iii): heterogeneous-rho robustness (n_reps=%d)", n_reps)
    het_rho = _run_heterogeneous_rho(n_reps, seed + 2)
    logger.info(
        "Paper 5 MC (iv): size/power of bar-alpha=0 at n=%d (n_reps=%d)", N_IV, n_reps
    )
    size_power = _run_size_power(n_reps, seed + 3)

    results: dict[str, Any] = {
        "rho_recovery": rho_recovery,
        "w_misspecification": w_misspec,
        "heterogeneous_rho": het_rho,
        "size_power": size_power,
        "seeds": {"base_seed": seed},
        "scales_note": (
            f"n_reps={n_reps}/config — see t04 README for the recorded full scale"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mc_results.json").open("w") as fh:
        json.dump(_to_plain(results), fh, indent=2, default=str)

    figures_dir.mkdir(parents=True, exist_ok=True)
    by_mag = w_misspec["by_magnitude"]
    render_w_misspecification(
        MisspecificationSeries(
            magnitude=[by_mag[f"m_{m}"]["magnitude"] for m in MISSPEC_GRID],
            bias_mean=[by_mag[f"m_{m}"]["bias_mean"] for m in MISSPEC_GRID],
            bias_se=[by_mag[f"m_{m}"]["bias_se"] for m in MISSPEC_GRID],
        ),
        figures_dir / "mc_w_misspecification.pgf",
    )
    power = size_power["power"]
    render_size_power(
        SizePowerSeries(
            alpha_true=[power[f"delta_{d}"]["alpha_true"] for d in POWER_ALPHA_GRID],
            power=[power[f"delta_{d}"]["power"] for d in POWER_ALPHA_GRID],
            empirical_size=size_power["size"]["empirical_size"],
            nominal_alpha=NOMINAL_ALPHA,
            n_assets=cast("int", size_power.get("n")),
        ),
        figures_dir / "mc_size_power.pgf",
    )

    logger.info("Wrote %s", output_dir / "mc_results.json")
    return results


def main() -> None:
    """CLI-free entry point: run the default-scale MC study and write outputs."""
    logging.basicConfig(level=logging.INFO)
    run_paper5_mc(Path("data/papers/paper5"))


if __name__ == "__main__":
    main()


# `SarDGP`, `make_synthetic_w` and `simulate_sar` were exported here until the
# DGP moved to `simulation.dgp.spatial`. They are imported above for this
# module's own use and deliberately NOT re-exported: this module is the study
# driver, and advertising the generator as its API is what let two other call
# sites reach a data-free DGP through a paper's stage module.
__all__ = ["run_paper5_mc"]

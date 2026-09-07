"""Orchestration and emit for the Paper 5 energy-robustness empirical stage."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
from jcor.model.covariance import sample_covariance
from jcor.operators.covariance import sigma_dist
from jcor.optimize.psd import nearest_covariance

from pipeline.stages.papers.paper3.empirical import (
    calibrate_kappa,
)
from pipeline.stages.papers.paper5._energy_robust.backtest import backtest
from pipeline.stages.papers.paper5._energy_robust.contracts import (
    Paper5EnergyRobustnessConfig,
    RobustBacktestConfig,
    RobustBacktestInputs,
    RobustCovarianceInputs,
)
from pipeline.stages.papers.paper5._energy_robust.coverage import (
    _coverage_block_bootstrap_ci,
)
from pipeline.stages.papers.paper5._energy_robust.materiality import (
    _materiality_gate,
    _to_plain,
)
from pipeline.stages.papers.paper5.energy_radius import (
    CovarianceBracketInputs,
    RidgeRadiusConfig,
    RidgeRadiusInputs,
    assemble_brackets,
    bootstrap_ridge_radius,
    bootstrap_stat_radius,
    frontier_envelope,
    implied_kappa_envelope_check,
    load_calibration_embeddings,
    pair_gmm_radius,
    stat_radius_matrix,
)
from pipeline.stages.substrate.energy_shared import _load_energy_distances
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import (
        Float,
        Paper5EnergyRobustnessPaths,
    )

logger = logging.getLogger(__name__)


def run_paper5_energy_robustness(
    paths: Paper5EnergyRobustnessPaths,
    config: Paper5EnergyRobustnessConfig | None = None,
) -> dict[str, object]:
    """Run Paper 5 energy-robustness empirics and write manifest data.

    Uses the shared typed-distance convention throughout: the persisted value is
    already the squared energy functional ``D2 = mathcal{E}^2`` and is therefore
    not squared again; ``D = sqrt(D2)`` is the metric. This matches the shared frontier
    H1 and Paper 3 empirical consumers and keeps T1's "radii add on the metric"
    discipline consistent across papers.

    Args:
        paths: Input and output artifact boundaries.
        config: Epoch, window, bootstrap, and seed controls.

    Returns:
        The full results dict (also serialized as JSON).

    """
    config = Paper5EnergyRobustnessConfig() if config is None else config
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    all_tickers, energy_distance = _load_energy_distances(
        paths.energy_tests_path,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        distance_id=config.distance_id,
    )
    missing_calibration = [
        ticker
        for ticker in all_tickers
        if not (paths.returns_dir / f"{ticker}.parquet").exists()
    ]
    missing_evaluation = [
        ticker
        for ticker in all_tickers
        if not (paths.eval_returns_dir / f"{ticker}.parquet").exists()
    ]
    if missing_calibration or missing_evaluation:
        message = (
            "Paper 5 energy-robustness requires the distance universe in both "
            "return panels; "
            f"missing calibration={missing_calibration}, "
            f"evaluation={missing_evaluation}"
        )
        raise FileNotFoundError(message)
    tickers = all_tickers
    d2 = np.array(energy_distance, dtype=np.float64, copy=True)
    np.fill_diagonal(d2, 0.0)
    n = len(tickers)
    logger.info("Paper 5 energy-robustness: %d tickers", n)

    _, r_calib = load_aligned_return_panel(
        paths.returns_dir, tickers, config.calib_start, config.calib_end
    )
    _, r_eval = load_aligned_return_panel(
        paths.eval_returns_dir, tickers, config.eval_start, config.eval_end
    )
    t_calib, t_eval = r_calib.shape[0], r_eval.shape[0]
    logger.info("Calibration T=%d, evaluation T=%d", t_calib, t_eval)

    # --- Radius calibration (calibration epoch only) ---
    embeddings = load_calibration_embeddings(
        paths.embeddings_dir, tickers, config.calib_start, config.calib_end
    )
    eps_stat = bootstrap_stat_radius(
        embeddings, n_boot=config.n_boot_stat, seed=config.seed
    )
    eps_stat_matrix = stat_radius_matrix(eps_stat, tickers)

    kappa_calib, kappa_calib_se = calibrate_kappa(r_calib, d2)
    eps_gmm_d2 = pair_gmm_radius(r_calib, d2, kappa_calib)

    frontier = frontier_envelope(paths.frontier_metrics_path)
    implied_kappa = implied_kappa_envelope_check(
        r_calib, d2, frontier["ell_lo"], frontier["L_hat"]
    )

    sigma_sq_calib = np.diag(sample_covariance(r_calib, ddof=1))
    # The PSD repair is traced, so Σ_dist stays on device through it; x64 is
    # required for its nearest-matrix contract and is what this stage runs at.
    with jax.enable_x64(new_val=True):
        sd_calib = np.array(
            nearest_covariance(
                sigma_dist(d2, sigma_sq_calib, kappa=kappa_calib)
            ).matrix,
            dtype=np.float64,
        )

    brackets_by_mode = {
        "calibrated": assemble_brackets(
            CovarianceBracketInputs(
                d2,
                eps_stat_matrix,
                eps_gmm_d2,
                sigma_sq_calib,
                abs(kappa_calib),
                abs(kappa_calib),
            )
        ),
        "certified": assemble_brackets(
            CovarianceBracketInputs(
                d2,
                eps_stat_matrix,
                np.zeros_like(d2),
                sigma_sq_calib,
                frontier["ell_lo"],
                frontier["L_hat"],
            )
        ),
    }

    ridge_radius = bootstrap_ridge_radius(
        RidgeRadiusInputs(r_calib, sd_calib, config.windows),
        RidgeRadiusConfig(n_boot=config.n_boot_ridge, seed=config.seed),
    )
    r_fitted = cast("dict[int, float]", ridge_radius["r_by_window_fitted"])
    r_by_window = {int(k): float(v) for k, v in r_fitted.items()}

    # --- Backtest (evaluation epoch, fixed calibration radii) ---
    bt = backtest(
        RobustBacktestInputs(
            r_eval,
            RobustCovarianceInputs(
                d2,
                eps_stat_matrix,
                eps_gmm_d2,
                frontier,
                r_by_window,
                kappa_calib,
                sigma_sq_calib,
            ),
        ),
        RobustBacktestConfig(
            windows=config.windows,
            n_boot=config.n_boot_sharpe,
            seed=config.seed,
        ),
    )
    bt.to_parquet(paths.output_dir / "results.parquet", index=False)

    # --- H1 coverage: calibration bracket vs held-out realized covariance,
    #     block-bootstrap CI, both bracket modes ---
    h1_coverage: dict[str, object] = {
        "nominal_target": 0.90,
        "headline_rule": (
            "pre-stated: the paper's headline bracket is the TIGHTER mode "
            "whose point coverage meets the nominal target; both reported"
        ),
    }
    for mode, br in brackets_by_mode.items():
        h1_coverage[mode] = _coverage_block_bootstrap_ci(
            cast("Float", br["Sigma_lo"]),
            cast("Float", br["Sigma_hi"]),
            r_eval,
            n_boot=300,
            seed=config.seed,
        )

    materiality_gate = _materiality_gate(
        brackets_by_mode, r_calib, t_ref=60, n_boot=200, seed=config.seed
    )

    results: dict[str, object] = {
        "universe": {"n_tickers": n, "tickers": tickers},
        "epochs": {
            "calibration": {
                "start": config.calib_start,
                "end": config.calib_end,
                "n_obs": int(t_calib),
            },
            "evaluation": {
                "start": config.eval_start,
                "end": config.eval_end,
                "n_obs": int(t_eval),
            },
        },
        "energy_distance_convention": (
            "The persisted typed value is D2 = E^2; Papers 2--4 consume it "
            "directly and use D = sqrt(D2) where the metric is required."
        ),
        "kappa_calibration": {"kappa_hat": kappa_calib, "kappa_se": kappa_calib_se},
        "radii": {
            "eps_stat_per_firm": eps_stat,
            "eps_stat_n_boot": config.n_boot_stat,
            "eps_gmm_d2_summary": {
                "median": float(np.median(eps_gmm_d2[np.triu_indices(n, 1)])),
                "mean": float(np.mean(eps_gmm_d2[np.triu_indices(n, 1)])),
            },
            "frontier_envelope": frontier,
            "ridge_radius": ridge_radius,
        },
        "implied_kappa_check": implied_kappa,
        "bracket_diagnostics_calibration": {
            mode: {
                k: v
                for k, v in br.items()
                if k not in ("Sigma_lo", "Sigma_hi", "D2_lo", "D2_hi")
            }
            for mode, br in brackets_by_mode.items()
        },
        "materiality_gate": materiality_gate,
        "h1_coverage": h1_coverage,
        "backtest": bt.to_dict(orient="records"),
    }

    with (paths.output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(_to_plain(results), fh, indent=2, default=str)
    logger.info("Wrote %s", paths.output_dir / "results.json")
    return results

"""Orchestration and emit for the Paper 3 empirical stage."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import yaml

from pipeline.stages.papers.paper3._empirical.backtest import backtest
from pipeline.stages.papers.paper3._empirical.backtest_cells import _load_pit_matrices
from pipeline.stages.papers.paper3._empirical.contracts import (
    BacktestConfig,
    BacktestInputs,
    IdentityTestConfig,
    IdentityTestInputs,
)
from pipeline.stages.papers.paper3._empirical.identity import identity_tests
from pipeline.stages.papers.paper3._empirical.manifest import _write_paper3_manifest
from pipeline.stages.papers.paper3._empirical.panel import (
    build_sigma_dist,
    calibrate_kappa,
    load_panel,
)
from pipeline.stages.papers.paper3._empirical.robustness import robustness_gel_gms
from pipeline.stages.papers.paper3._empirical.sharpe import _paired_sharpe_diff_summary
from pipeline.stages.substrate.energy_shared import _to_plain

if TYPE_CHECKING:
    from jcor.optimize.psd import RepairDiagnostics

    from pipeline.stages.papers.paper3._empirical.contracts import (
        Paper3EmpiricalConfig,
        Paper3EmpiricalPaths,
    )

logger = logging.getLogger(__name__)


def _temporary_sibling(path: Path) -> Path:
    """Allocate a writable sibling for atomic replacement of a DVC hardlink."""
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix
    )
    os.close(descriptor)
    return Path(name)


def _write_yaml_atomic(path: Path, value: object) -> None:
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, default_flow_style=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_sibling(path)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _run_paper3_empirical_impl(
    paths: Paper3EmpiricalPaths,
    config: Paper3EmpiricalConfig,
) -> None:
    """Body of :func:`run_paper3_empirical`, run inside the BLAS-thread-cap scope."""
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    tickers, returns, squared_distances, dates = load_panel(
        paths.returns_dir,
        paths.distance_artifact_dir,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        distance_id=config.distance_id,
    )
    n_observations, n = returns.shape
    logger.info("Panel: %d tickers, T=%d observations", n, n_observations)

    kappa_hat, kappa_se = calibrate_kappa(returns, squared_distances)
    logger.info("Calibrated kappa=%.4g (se=%.4g)", kappa_hat, kappa_se)

    step1 = build_sigma_dist(returns, squared_distances, kappa_hat)
    diag = cast("RepairDiagnostics", step1["diagnostics"])

    identity_inputs = IdentityTestInputs(
        returns,
        squared_distances,
        kappa_hat,
        kappa_se,
    )
    identity_config = IdentityTestConfig(
        n_clusters=config.n_clusters,
        n_mc=config.n_mc,
        seed=config.seed,
    )
    tests = identity_tests(identity_inputs, identity_config)

    # Robustness: descriptive profiled GEL plus uncorrected indicator-GMS on
    # the same zero-transport-excess benchmark moments and clustering.
    # Emitted as its own artifact so Paper 3's LaTeX render can cite it.
    gel_gms = robustness_gel_gms(identity_inputs, identity_config)
    gel_gms_plain = _to_plain(gel_gms)
    _write_yaml_atomic(paths.output_dir / "robustness_gel_gms.yaml", gel_gms_plain)
    logger.info("Wrote %s", paths.output_dir / "robustness_gel_gms.yaml")

    # OOS backtest recalibrates κ per training window (no look-ahead); the
    # full-panel kappa_hat is retained only for the in-sample benchmark tests.
    pit_matrices, pit_dates = _load_pit_matrices(paths.output_dir)
    bt = backtest(
        BacktestInputs(
            returns,
            squared_distances,
            dates=dates,
            pit_matrices=pit_matrices,
            pit_dates=pit_dates,
        ),
        BacktestConfig(
            windows=config.windows,
            n_boot=config.n_boot,
            seed=config.seed,
        ),
    )
    _write_parquet_atomic(paths.output_dir / "results.parquet", bt)

    sharpe_diff = _paired_sharpe_diff_summary(
        returns,
        squared_distances,
        n_boot=min(config.n_boot, 500),
        seed=config.seed,
    )

    summary = {
        "universe": {
            "n_tickers": n,
            "n_observations": int(n_observations),
            "tickers": tickers,
            "excluded": cast("list[str]", []),
        },
        "disclosure": {
            "machine_checked_headline": (
                "The W2 Frechet covariance ceiling, transport-excess identity, "
                "and excess nonnegativity are proved in Lean under their "
                "stated integrability and optimal-plan-certificate hypotheses."
            ),
            "transmission_restriction": (
                "The observed matrix is W2 between article-characteristic laws "
                "C_i. Interpreting kappa^2 W2(C_i,C_j)^2 as the exposure-law "
                "transport cost additionally imposes a scalar exact-transmission "
                "benchmark; treating the resulting pairwise ceiling as realized "
                "covariance further imposes zero transport excess."
            ),
        },
        "geometry": {
            "observed_law": "characteristic_law_C",
            "latent_law": "exposure_law_P=Law(T(X,U))",
            "distance_id": config.distance_id,
            "artifact_scale": "rooted_wasserstein_2",
            "model_scale": "squared_wasserstein_2_transport_cost",
        },
        "kappa": {"kappa_hat": kappa_hat, "kappa_se": kappa_se},
        "psd_repair": {
            "min_eig_before": diag.min_eig_before,
            "min_eig_after": diag.min_eig_after,
            "cond_before": diag.cond_before,
            "cond_after": diag.cond_after,
            "total_var_before": diag.total_var_before,
            "total_var_after": diag.total_var_after,
            "offdiag_distortion": diag.offdiag_distortion,
            "is_psd_after": diag.is_psd_after,
            # The Qi-Sun dual KKT verdict. `is_psd_after` only says the repair
            # landed in the cone; this says it landed on the *nearest* point,
            # which is the claim the manuscript makes. Recorded because the
            # nearest-correlation interpretation is certified per call rather
            # than assumed, and a reader cannot check it from the matrix alone.
            # Coerced here rather than left to `_to_plain`: the leaf is a JAX
            # array, which that helper passes through untouched and
            # `yaml.safe_dump` then refuses.
            "converged": (None if diag.converged is None else bool(diag.converged)),
            # Measured 4.0977e-08 against a 6.00e-13 stopping tolerance, flat
            # across 25, 50, 100, 200, 400 and 800 Newton steps -- the solver
            # stalls rather than running out of budget. Persisted so the
            # manuscript can quote the distance from nearest instead of
            # asserting a verdict.
            "dual_kkt_residual": step1["psd_residual"],
            "newton_iterations": step1["psd_iterations"],
        },
        "shrinkage": {
            "same_window_plugin_sample_weight": step1["lam"],
            "fixed_target_mse_optimal": False,
            "adjusted_gap_sum": step1["bias_sq_sum"],
            # Compatibility keys for existing generated-number readers.  The
            # names predate the audit; they do not restore fixed-target
            # optimality or turn the adjusted gap into target bias.
            "lambda_star_on_sample": step1["lam"],
            "var_S_sum": step1["var_S_sum"],
            "bias_sq_sum": step1["bias_sq_sum"],
        },
        "envelope_tests": tests,
        "backtest": _backtest_summary(bt),
        "sharpe_difference_test": sharpe_diff,
    }
    summary = _to_plain(summary)
    _write_yaml_atomic(paths.output_dir / "summary.yaml", summary)
    logger.info("Wrote %s", paths.output_dir / "summary.yaml")
    _write_paper3_manifest(paths, config)
    logger.info("Wrote %s", paths.output_dir / "provenance.manifest.json")


def _backtest_summary(bt: pd.DataFrame) -> dict[str, object]:
    """Compact nested backtest summary and Σ_dist-vs-sample honesty flags.

    Args:
        bt: Backtest results frame.

    Returns:
        Nested dict keyed by constraint → method → window → metrics, plus a
        ``sigma_dist_beats_sample`` boolean matrix (Sharpe at 10bps).

    """
    out: dict[str, object] = {}
    flags: dict[str, dict[int, bool]] = {}
    filtered = bt.loc[
        (bt.get("period", "full") == "full")
        & (bt.get("source", "baseline") == "baseline")
    ]
    for constraint_key, gc in filtered.groupby("constraint"):
        constraint = str(constraint_key)
        cd: dict[str, object] = {}
        for method_key, gm in gc.groupby("method"):
            method = str(method_key)
            md: dict[int, dict[str, float]] = {}
            for _, r in gm.iterrows():
                md[int(r["window"])] = {
                    "sharpe_net_10bps": float(r["sharpe_net_10bps"]),
                    "sharpe_net_50bps": float(r["sharpe_net_50bps"]),
                    "realized_vol": float(r["realized_vol"]),
                    "turnover": float(r["turnover"]),
                    "cond": float(r["cond"]),
                    "eff_n": float(r["eff_n"]),
                    "hhi": float(r["hhi"]),
                }
                if method == "sigma_shrink":
                    for key in (
                        "lambda_sample_mean",
                        "lambda_sample_median",
                        "lambda_sample_min",
                        "lambda_sample_max",
                    ):
                        if key in r and pd.notna(r[key]):
                            md[int(r["window"])][key] = float(r[key])
            cd[method] = md
        out[constraint] = cd
        # Honesty: does sigma_dist beat sample-cov (Sharpe net 10bps)?
        samp = gc[gc["method"] == "sample"].set_index("window")["sharpe_net_10bps"]
        dist = gc[gc["method"] == "sigma_dist"].set_index("window")["sharpe_net_10bps"]
        flags[constraint] = {
            int(w): bool(dist[w] > samp[w]) for w in samp.index if w in dist.index
        }
    out["sigma_dist_beats_sample_10bps"] = flags
    return out

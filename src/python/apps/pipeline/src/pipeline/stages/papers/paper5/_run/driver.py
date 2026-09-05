"""Artifact-I/O driver for the Paper 5 empirical stage."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper5 import hypotheses as hyp
from pipeline.stages.papers.paper5._run.contracts import (
    _CALIB_START,
    _IN_SAMPLE_FOLDS,
    _OOS_FOLDS,
    ACTIVE_WEIGHT_TOLERANCE,
    EXPECTED_TICKERS,
    Paper5EmpiricalConfig,
    Paper5EmpiricalError,
    _H3EvaluationInputs,
    _RhoFitInputs,
)
from pipeline.stages.papers.paper5._run.evaluation import (
    _evaluate_h3,
    _evaluate_h3_per_fold,
    _evaluate_h4,
)
from pipeline.stages.papers.paper5._run.rho import (
    _fit_rho_models,
    _rho_stability_records,
)
from pipeline.stages.papers.paper5._run.robustness import (
    _build_weight_variants,
    _robustness_cuts,
)
from pipeline.stages.papers.paper5.energy_radius import (
    bootstrap_stat_radius,
    load_calibration_embeddings,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.papers.paper5._run.contracts import Paper5EmpiricalPaths
    from pipeline.stages.papers.paper5._run.evaluation import H3VariantFamilies

logger = logging.getLogger(__name__)


def _load_distance_matrix(
    artifact_dir: Path, config: Paper5EmpiricalConfig
) -> tuple[list[str], np.ndarray]:
    """Load the typed geometry as squared transport cost for covariance work."""
    frame, summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity={
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "distance_id": config.distance_id,
        },
    )
    tickers = cast("list[str]", summary["item_ids"])
    values = frame["value"].to_numpy(dtype=np.float64)
    if config.distance_id == "wasserstein_w2":
        # The canonical W2 artifact is rooted; the covariance envelope and
        # diffusion kernel both consume the quadratic transport cost.
        values = values**2
    return tickers, values.reshape(len(tickers), len(tickers))


def _returns_seam_disclosure(
    returns_dir: Path,
    oos_returns_dir: Path,
    tickers: list[str],
) -> dict[str, object]:
    """Union in-sample + OOS returns and disclose the 2022/2023 seam.

    Validate ticker identity, a unique ``2022-12-30`` row, and the next
    ``2023-01-03`` trading day; report per-year rows and ticker composition.
    """
    oos_tickers = sorted(p.stem for p in oos_returns_dir.glob("*.parquet"))
    if oos_tickers != sorted(tickers):
        message = (
            f"OOS ticker set {oos_tickers} is not identity-equal to the "
            f"in-sample {len(tickers)}-ticker universe {sorted(tickers)}"
        )
        raise Paper5EmpiricalError(message)

    dates, _ = load_aligned_return_panel(
        returns_dir, tickers, oos_returns_dir=oos_returns_dir
    )
    dates = pd.to_datetime(dates)
    seam_ts = pd.Timestamp("2022-12-30")
    if int((dates == seam_ts).sum()) != 1:
        message = (
            f"Expected exactly one 2022-12-30 row in the unioned panel, "
            f"found {int((dates == seam_ts).sum())}"
        )
        raise Paper5EmpiricalError(message)
    seam_pos = int(np.searchsorted(dates.values, seam_ts.to_datetime64()))
    if seam_pos + 1 >= len(dates) or dates[seam_pos + 1] != pd.Timestamp("2023-01-03"):
        message = (
            "Abnormal trading-day gap across the 2022-12-30 -> 2023-01-03 "
            f"seam; next trading day after 2022-12-30 is "
            f"{dates[seam_pos + 1] if seam_pos + 1 < len(dates) else 'MISSING'}"
        )
        raise Paper5EmpiricalError(message)

    years = pd.Series(dates).dt.year
    per_year_rows = {
        int(cast("int", year)): int(count)
        for year, count in years.value_counts().sort_index().items()
    }
    per_year_tickers = {int(y): list(tickers) for y in per_year_rows}

    return {
        "n_tickers": len(tickers),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "per_year_row_counts": per_year_rows,
        "per_year_tickers": per_year_tickers,
    }


def _to_plain(obj: object) -> object:
    """Convert numpy/JAX containers and scalars to JSON-compatible values."""
    if isinstance(obj, dict):
        plain: object = {str(key): _to_plain(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        plain = [_to_plain(value) for value in obj]
    elif isinstance(obj, np.ndarray):
        plain = np.asarray(obj).tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        plain = obj.item()
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        plain = obj
    else:
        as_dict = getattr(obj, "_asdict", None)
        plain = _to_plain(as_dict()) if callable(as_dict) else str(obj)
    return plain


def _summarize_h3_bracket_width(
    h3_record: H3VariantFamilies,
) -> tuple[float | None, dict[str, object]]:
    """Return a nullable width and an explicit certification record.

    A fail-closed H3 evaluation intentionally omits ``sar_robust``.  Report
    that state as unavailable instead of manufacturing a numerical bracket.
    """
    sar_robust = h3_record.get("sar_robust")
    if isinstance(sar_robust, dict) and "median_bracket_width_sigma" in sar_robust:
        width = sar_robust["median_bracket_width_sigma"]
        if isinstance(width, bool) or not isinstance(width, (int, float)):
            message = "H3 SAR bracket width must be numeric"
            raise TypeError(message)
        return float(width), {
            "status": "computed",
            "available": True,
        }

    certification = h3_record.get("certification")
    if isinstance(certification, dict):
        return None, {
            "status": str(certification.get("status", "unavailable")),
            "available": False,
        }
    return None, {"status": "unavailable", "available": False}


def _build_w_comparison(
    rho_df: pd.DataFrame,
    w_flat: np.ndarray,
    w_h: np.ndarray,
    w_variants: dict[str, np.ndarray],
    h3_by_variant: dict[str, H3VariantFamilies],
) -> dict[str, object]:
    """Build the comparison arm without assuming every bracket is certified."""
    w_flat_width, w_flat_width_certification = _summarize_h3_bracket_width(
        h3_by_variant["w_flat"]
    )
    w_h_width, w_h_width_certification = _summarize_h3_bracket_width(
        h3_by_variant["w_h"]
    )

    def _eval2021_loglik(variant: str) -> float:
        return float(
            rho_df[
                (rho_df.w_variant == variant)
                & (rho_df.method == "qmle")
                & (rho_df.fold == "eval2021")
            ]["loglik"].iloc[0]
        )

    return {
        "fit_loglik_eval2021": {
            "w_flat": _eval2021_loglik("w_flat"),
            "w_h": _eval2021_loglik("w_h"),
            "w_co_mentions": _eval2021_loglik("w_co_mentions"),
        },
        "sparsity_active_entries": {
            "w_flat": int(np.sum(w_flat > ACTIVE_WEIGHT_TOLERANCE)),
            "w_h": int(np.sum(w_h > ACTIVE_WEIGHT_TOLERANCE)),
            "w_co_mentions": int(
                np.sum(w_variants["w_co_mentions"] > ACTIVE_WEIGHT_TOLERANCE)
            ),
        },
        "bracket_width_median": {
            "w_flat": w_flat_width,
            "w_h": w_h_width,
        },
        "bracket_certification": {
            "w_flat": w_flat_width_certification,
            "w_h": w_h_width_certification,
        },
    }


def run_paper5_empirical(
    paths: Paper5EmpiricalPaths,
    config: Paper5EmpiricalConfig | None = None,
) -> dict[str, object]:
    """End-to-end Paper 5 empirical run. Writes ``results.json`` + parquet.

    Args:
        paths: Input/output artifacts. Optional OOS returns activate the
            ``eval2023``..``eval2026`` folds and the disclosed seam guards.
        config: Bootstrap, neighborhood, and seed controls.

    Returns:
        The full results dict (also serialized as JSON).

    """
    config = Paper5EmpiricalConfig() if config is None else config
    folds: dict[str, tuple[str, str]] = dict(_IN_SAMPLE_FOLDS)
    if paths.oos_returns_dir is not None:
        folds.update(_OOS_FOLDS)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    tickers, d2 = _load_distance_matrix(paths.distance_artifact_dir, config)
    n = len(tickers)
    if n != EXPECTED_TICKERS:
        message = f"Expected {EXPECTED_TICKERS}-ticker intersection, got {n}"
        raise Paper5EmpiricalError(message)
    logger.info("Paper 5: %d tickers", n)

    universe = pd.read_csv(paths.universe_csv).set_index("Symbol")
    sector_of = universe["Sector"].to_dict()

    returns_seam: dict[str, object] | None = None
    if paths.oos_returns_dir is not None:
        returns_seam = _returns_seam_disclosure(
            paths.returns_dir, paths.oos_returns_dir, tickers
        )

    # --- Calibration-epoch radii (energy-robustness prerequisite) ---
    calib_embeddings = load_calibration_embeddings(
        paths.embeddings_dir, tickers, _CALIB_START, "2021-12-31"
    )
    eps_stat = bootstrap_stat_radius(
        calib_embeddings, n_boot=config.n_boot_stat, seed=config.seed
    )
    (
        frozen_w_variants,
        w_diagnostics,
        flat_bound,
        delta_w_h,
        bandwidth_h,
        w_flat,
        w_h,
    ) = _build_weight_variants(
        tickers,
        d2,
        eps_stat,
        sector_of,
        paths.shared_barycentre_dir,
        paths.wasserstein_w1_barycentre_dir,
        paths.co_mentions_adjacency,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        barycentre_arm_id=config.barycentre_arm_id,
        wasserstein_w1_arm_id=config.wasserstein_w1_arm_id,
        distance_id=config.distance_id,
    )
    w_variants = frozen_w_variants

    rho_df, fold_data = _fit_rho_models(
        _RhoFitInputs(
            folds,
            paths,
            tickers,
            frozen_w_variants,
            config.knn_k,
        )
    )
    rho_stability = _rho_stability_records(rho_df)

    h2_row = rho_df[
        (rho_df["w_variant"] == "w_flat")
        & (rho_df["method"] == "qmle")
        & (rho_df["fold"] == "eval2021")
    ].iloc[0]
    h2 = hyp.h2_zero_intercept_test(
        alpha_hat=float(h2_row["alpha_hat"]),
        sigma2_hat=float(h2_row["sigma2_hat"]),
        t_obs=int(fold_data["eval2021"]["train"].shape[0]),
        n=n,
    )

    energy_robustness_results_path = (
        paths.output_dir / "energy_robustness" / "results.json"
    )
    energy_robustness_results = json.loads(energy_robustness_results_path.read_text())
    lw_band = float(energy_robustness_results["materiality_gate"]["lw_band_width_t60"])
    energy_cal_cov = float(
        energy_robustness_results["h1_coverage"]["calibrated"]["point_coverage"]
    )
    energy_cert_cov = float(
        energy_robustness_results["h1_coverage"]["certified"]["point_coverage"]
    )

    coverage = (energy_cal_cov, energy_cert_cov)
    h3_inputs = _H3EvaluationInputs(
        fold_data,
        rho_df,
        w_variants,
        flat_bound,
        delta_w_h,
        coverage,
        lw_band,
    )
    h3_by_variant = _evaluate_h3(h3_inputs)

    h4_by_variant = _evaluate_h4(folds, fold_data, rho_df, w_variants, d2)

    w_comparison = _build_w_comparison(rho_df, w_flat, w_h, w_variants, h3_by_variant)

    # Per-asset heterogeneous-rho breakdown (local Moran's I decomposition,
    # eval2021, both W variants -- diagnostic, does not re-estimate rho).
    per_asset_rho: dict[str, dict[str, float]] = {}
    for w_name in ("w_flat", "w_h"):
        per_asset_rho[w_name] = hyp.local_moran_contributions(
            fold_data["eval2021"]["train"], w_variants[w_name], tickers
        )

    h3_per_fold = _evaluate_h3_per_fold(folds, h3_inputs)
    notation_panel = hyp.notation_panel_entries(
        n,
        folds,
        [
            "w_flat",
            "w_h",
            "w_w1",
            "w_co_mentions",
            "industry_adjacency",
            "correlation_knn",
            "rho_zero",
        ],
    )
    robustness_knn_k, robustness_bandwidth_h = _robustness_cuts(
        fold_data["eval2021"]["train"], d2, bandwidth_h
    )

    results: dict[str, object] = {
        "universe": {"n_tickers": n, "tickers": tickers},
        "folds": {name: {"start": s, "end": e} for name, (s, e) in folds.items()},
        "w_diagnostics": w_diagnostics,
        "h2": h2,
        "h3": h3_by_variant,
        "h4": h4_by_variant,
        "w_comparison": w_comparison,
        "per_asset_rho": per_asset_rho,
        "h3_per_fold": h3_per_fold,
        "notation_panel": notation_panel,
        "robustness": {
            "knn_k": robustness_knn_k,
            "bandwidth_h": robustness_bandwidth_h,
        },
        "rho_stability": rho_stability,
    }
    if returns_seam is not None:
        results["returns_seam"] = returns_seam
    plain_results = _to_plain(results)

    with (paths.output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(plain_results, fh, indent=1)
    logger.info("Wrote %s", paths.output_dir / "results.json")

    rho_df.to_parquet(paths.output_dir / "results.parquet", index=False)
    logger.info("Wrote %s", paths.output_dir / "results.parquet")

    return results

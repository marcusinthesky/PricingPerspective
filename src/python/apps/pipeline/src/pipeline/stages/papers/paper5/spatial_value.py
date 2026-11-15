"""Rigorous factor-only versus W1/W2 spatial-value comparisons."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from pipeline.io.fama_french import load_factor_data_config, load_factor_model
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5.bootstrap import (
    independent_stationary_bootstrap_indices,
    paired_refit_schedule,
)
from pipeline.stages.papers.paper5.transport_order import (
    Paper5TransportOrderConfig,
    Paper5TransportOrderPaths,
    _align_factor_panel,
    _eigenvalues,
    _load_operators,
)
from pipeline.stages.papers.paper5.two_field import quasi_likelihood_ratio
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline.io.fama_french import FactorModelId
    from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel

logger = logging.getLogger(__name__)

type FloatMatrix = NDArray[np.float64]
type IntVector = NDArray[np.int64]
type VarianceModel = Literal["spherical", "diagonal"]

_MODELS = ("none", "w1", "w2")
_SPATIAL_MODELS = ("w1", "w2")
_VARIANCE_MODELS: tuple[VarianceModel, ...] = ("spherical", "diagonal")
_ONE_HALF = 0.5


@dataclass(frozen=True, slots=True)
class Paper5SpatialValuePaths:
    """Input and output boundaries for the spatial-value stage."""

    returns_dir: Path
    oos_returns_dir: Path
    w1_barycentre_dir: Path
    w2_barycentre_dir: Path
    transport_results_json: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5SpatialValueConfig:
    """Frozen factor, timing, resampling, and inference contract."""

    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    confidence_level: float = 0.95
    qlr_alpha: float = 0.05
    factor_model: FactorModelId = "ff5_mom"
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    w1_barycentre_arm_id: str = "wasserstein_w1_loo"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w1_geometry_id: str = "wasserstein_w1"
    w2_geometry_id: str = "wasserstein_w2"

    def validate(self) -> None:
        """Reject invalid inference schedules before reading production data."""
        if self.n_bootstrap < 1 or self.block_length < 1:
            message = "bootstrap draws and block length must be positive"
            raise ValueError(message)
        if not 0.0 < self.confidence_level < 1.0:
            message = "confidence level must lie strictly between zero and one"
            raise ValueError(message)
        if not 0.0 < self.qlr_alpha < _ONE_HALF:
            message = "QLR alpha must lie strictly between zero and one half"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _AnalysisInputs:
    tickers: list[str]
    matrices: dict[str, FloatMatrix]
    eigenvalues: dict[str, NDArray[np.complex128]]
    calibration: _AlignedFactorPanel
    evaluation: _AlignedFactorPanel
    transport: dict[str, Any]


def _fit_models(
    panel: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, NDArray[np.complex128]],
    variance_model: VarianceModel,
    indices: IntVector | None = None,
) -> dict[str, est.FactorSarFit]:
    """Fit factor-only, W1, and W2 models on one common date sample."""
    selected = (
        np.arange(len(panel.dates), dtype=np.int64) if indices is None else indices
    )
    fits = {
        order: est.sar_factor_qmle(
            panel.excess_returns[selected],
            matrices[order],
            panel.factors[selected],
            compute_se=indices is None,
            eigvals_w=eigenvalues[order],
            variance_model=variance_model,
        )
        for order in _SPATIAL_MODELS
    }
    fits["none"] = est.sar_factor_fit_at_rho(
        panel.excess_returns[selected],
        matrices["w2"],
        panel.factors[selected],
        0.0,
        eigvals_w=eigenvalues["w2"],
        variance_model=variance_model,
    )
    return fits


def _scores(
    fits: dict[str, est.FactorSarFit],
    panel: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, NDArray[np.complex128]],
    indices: IntVector | None = None,
) -> dict[str, float]:
    """Return held-out mean quasi-log scores per firm-day."""
    selected = (
        np.arange(len(panel.dates), dtype=np.int64) if indices is None else indices
    )
    scores: dict[str, float] = {}
    for model in _MODELS:
        order = "w2" if model == "none" else model
        daily = est.sar_factor_daily_logscore(
            fits[model],
            panel.excess_returns[selected],
            matrices[order],
            panel.factors[selected],
            eigvals_w=eigenvalues[order],
        )
        scores[model] = float(np.mean(daily) / panel.excess_returns.shape[1])
    return scores


def _interval(values: pd.Series, confidence_level: float) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    return float(values.quantile(tail)), float(values.quantile(1.0 - tail))


def _paired_bootstrap(
    calibration: _AlignedFactorPanel,
    evaluation: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, NDArray[np.complex128]],
    config: Paper5SpatialValueConfig,
) -> pd.DataFrame:
    """Use shared date schedules for all models and variance specifications."""
    schedule = paired_refit_schedule(
        calibration_observations=len(calibration.dates),
        evaluation_observations=len(evaluation.dates),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed,
    )
    calibration_indices = schedule.calibration
    evaluation_fit_indices = schedule.evaluation_fit
    evaluation_score_indices = schedule.evaluation_score
    records: list[dict[str, object]] = []
    for draw_id in range(config.n_bootstrap):
        record: dict[str, object] = {"draw_id": draw_id}
        converged = True
        for variance_model in _VARIANCE_MODELS:
            calibration_fits = _fit_models(
                calibration,
                matrices,
                eigenvalues,
                variance_model,
                calibration_indices[draw_id],
            )
            evaluation_fits = _fit_models(
                evaluation,
                matrices,
                eigenvalues,
                variance_model,
                evaluation_fit_indices[draw_id],
            )
            predictive_scores = _scores(
                calibration_fits,
                evaluation,
                matrices,
                eigenvalues,
                evaluation_score_indices[draw_id],
            )
            for model in _MODELS:
                prefix = f"{variance_model}_{model}"
                record[f"rho_{prefix}"] = evaluation_fits[model].rho_hat
                record[f"loglik_{prefix}"] = evaluation_fits[model].loglik
                record[f"score_{prefix}"] = predictive_scores[model]
            for order in _SPATIAL_MODELS:
                record[f"score_delta_{variance_model}_{order}"] = (
                    predictive_scores[order] - predictive_scores["none"]
                )
            converged = converged and all(
                fit.converged
                for fit in (*calibration_fits.values(), *evaluation_fits.values())
            )
        record["all_solvers_converged"] = converged
        records.append(record)
    return pd.DataFrame.from_records(records)


def _independent_null_indices(
    *, n_dates: int, n_assets: int, config: Paper5SpatialValueConfig
) -> NDArray[np.int64]:
    """Build independent stationary-block paths across firms under the null."""
    return independent_stationary_bootstrap_indices(
        n_dates,
        n_assets,
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed + 1000,
    )


def _null_simulation_panel(
    panel: _AlignedFactorPanel,
    null_fit: est.FactorSarFit,
    indices: NDArray[np.int64],
) -> FloatMatrix:
    """Simulate a factor-only panel with independent firm innovations."""
    design = np.column_stack(
        (np.ones(len(panel.dates), dtype=np.float64), panel.factors)
    )
    fitted = design @ null_fit.coefficients
    residual = panel.excess_returns - fitted
    residual -= np.mean(residual, axis=0, keepdims=True)
    residual_scale = np.sqrt(
        np.maximum(np.mean(np.square(residual), axis=0), est.NUMERICAL_FLOOR)
    )
    standardized = residual / residual_scale[None, :]
    simulated = np.empty_like(residual)
    for asset in range(panel.excess_returns.shape[1]):
        simulated[:, asset] = standardized[indices[asset], asset]
    simulated *= np.sqrt(null_fit.innovation_variances)[None, :]
    return cast("FloatMatrix", fitted + simulated)


def _qlr_bootstrap(
    evaluation: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, NDArray[np.complex128]],
    config: Paper5SpatialValueConfig,
    observed_fits: dict[VarianceModel, dict[str, est.FactorSarFit]],
) -> tuple[pd.DataFrame, dict[tuple[VarianceModel, str], dict[str, object]]]:
    """Calibrate spatial-versus-factor-only QLR statistics under rho zero."""
    null_indices = _independent_null_indices(
        n_dates=len(evaluation.dates),
        n_assets=evaluation.excess_returns.shape[1],
        config=config,
    )
    records: list[dict[str, object]] = []
    for draw_id in range(config.n_bootstrap):
        for variance_model in _VARIANCE_MODELS:
            simulated_returns = _null_simulation_panel(
                evaluation,
                observed_fits[variance_model]["none"],
                null_indices[draw_id],
            )
            simulated_panel = type(evaluation)(
                dates=evaluation.dates,
                raw_returns=simulated_returns,
                excess_returns=simulated_returns,
                factors=evaluation.factors,
                dropped_return_dates=evaluation.dropped_return_dates,
            )
            fits = _fit_models(simulated_panel, matrices, eigenvalues, variance_model)
            if not all(fit.converged for fit in fits.values()):
                message = f"spatial-value QLR null draw {draw_id} did not converge"
                raise RuntimeError(message)
            records.extend(
                {
                    "draw_id": draw_id,
                    "variance_model": variance_model,
                    "operator": order,
                    "qlr": quasi_likelihood_ratio(
                        fits[order].loglik, fits["none"].loglik
                    ),
                    "rho_hat": fits[order].rho_hat,
                    "all_solvers_converged": all(
                        fit.converged for fit in fits.values()
                    ),
                }
                for order in _SPATIAL_MODELS
            )
    draws = pd.DataFrame.from_records(records)
    summaries: dict[tuple[VarianceModel, str], dict[str, object]] = {}
    for variance_model in _VARIANCE_MODELS:
        for order in _SPATIAL_MODELS:
            observed = quasi_likelihood_ratio(
                observed_fits[variance_model][order].loglik,
                observed_fits[variance_model]["none"].loglik,
            )
            selected = draws.loc[
                (draws["variance_model"] == variance_model)
                & (draws["operator"] == order),
                "qlr",
            ]
            exceedances = int(np.sum(selected >= observed))
            p_value = float((1 + exceedances) / (len(selected) + 1))
            summaries[(variance_model, order)] = {
                "statistic": observed,
                "bootstrap_p_value": p_value,
                "bootstrap_exceedances": exceedances,
                "p_value_floor": 1.0 / (len(selected) + 1),
                "inference_status": "working_null_calibration",
                "alpha": config.qlr_alpha,
                "decision": (
                    "reject_rho_zero"
                    if p_value < config.qlr_alpha
                    else "retain_rho_zero"
                ),
            }
    return draws, summaries


def _information_criteria(
    loglik: float, parameter_count: int, n_dates: int, n_assets: int
) -> dict[str, float]:
    """Return working AIC, HQC, and both disclosed panel-BIC conventions."""
    firm_days = n_dates * n_assets
    if n_dates <= 1 or firm_days <= 1:
        message = "HQC requires date and firm-day sample sizes greater than one"
        raise ValueError(message)
    return {
        "aic": -2.0 * loglik + 2.0 * parameter_count,
        "hqc_dates": -2.0 * loglik + 2.0 * parameter_count * np.log(np.log(n_dates)),
        "hqc_firm_days": -2.0 * loglik
        + 2.0 * parameter_count * np.log(np.log(firm_days)),
        "bic_dates": -2.0 * loglik + np.log(n_dates) * parameter_count,
        "bic_firm_days": -2.0 * loglik + np.log(firm_days) * parameter_count,
    }


def _factor_ladder_rows(transport: dict[str, Any]) -> list[dict[str, object]]:
    comparison = cast("dict[str, Any]", transport["factor_model_comparison"])
    results = cast("dict[str, dict[str, Any]]", comparison["results"])
    rows: list[dict[str, object]] = []
    for model_id in cast("list[str]", comparison["model_order"]):
        w2 = cast("dict[str, Any]", results[model_id]["post_corpus_results"])["w2"]
        fit = cast("dict[str, Any]", w2)
        n_dates = int(fit["trading_days"])
        n_assets = int(fit["n_assets"])
        n_factors = int(fit["n_factors"])
        parameters = n_assets * (n_factors + 1) + 2
        loglik = float(fit["loglik"])
        rows.append(
            {
                "scope": "factor_ladder",
                "variance_model": "spherical",
                "model": model_id,
                "n_factors": n_factors,
                "n_dates": n_dates,
                "n_assets": n_assets,
                "parameter_count": parameters,
                "loglik": loglik,
                **_information_criteria(loglik, parameters, n_dates, n_assets),
            }
        )
    return rows


def _identity_diagnostics(
    fit: est.FactorSarFit,
    panel: _AlignedFactorPanel,
    weights: FloatMatrix,
) -> dict[str, float]:
    design = np.column_stack(
        (np.ones(len(panel.dates), dtype=np.float64), panel.factors)
    )
    ols_coefficients, *_ = np.linalg.lstsq(design, panel.excess_returns, rcond=None)
    reduced = est.sar_factor_reduced_form_coefficients(fit, weights)
    return {
        "max_abs_coefficient_error": float(np.max(np.abs(reduced - ols_coefficients))),
        "max_abs_fitted_mean_error": float(
            np.max(np.abs(design @ reduced - design @ ols_coefficients))
        ),
    }


def _load_analysis_inputs(
    paths: Paper5SpatialValuePaths,
    config: Paper5SpatialValueConfig,
) -> _AnalysisInputs:
    transport_config = Paper5TransportOrderConfig(
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        w1_barycentre_arm_id=config.w1_barycentre_arm_id,
        w2_barycentre_arm_id=config.w2_barycentre_arm_id,
        w1_geometry_id=config.w1_geometry_id,
        w2_geometry_id=config.w2_geometry_id,
        factor_model=config.factor_model,
    )
    transport_paths = Paper5TransportOrderPaths(
        returns_dir=paths.returns_dir,
        oos_returns_dir=paths.oos_returns_dir,
        w1_barycentre_dir=paths.w1_barycentre_dir,
        w2_barycentre_dir=paths.w2_barycentre_dir,
        params_file=paths.params_file,
        output_dir=paths.output_dir,
    )
    tickers, matrices, _ = _load_operators(transport_paths, transport_config)
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    if config.factor_model != factor_config.primary_model:
        message = "spatial-value comparison must use the governed primary factor model"
        raise ValueError(message)
    factor_panel = load_factor_model(factor_config, config.factor_model)
    calibration_dates, calibration_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        config.calibration_start,
        config.calibration_end,
        oos_returns_dir=paths.oos_returns_dir,
    )
    evaluation_dates, evaluation_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        config.evaluation_start,
        config.evaluation_end,
        oos_returns_dir=paths.oos_returns_dir,
    )
    calibration = _align_factor_panel(
        cast("NDArray[np.datetime64]", calibration_dates),
        cast("FloatMatrix", calibration_returns),
        factor_panel,
    )
    evaluation = _align_factor_panel(
        cast("NDArray[np.datetime64]", evaluation_dates),
        cast("FloatMatrix", evaluation_returns),
        factor_panel,
    )
    if np.max(calibration.dates) >= np.min(evaluation.dates):
        message = "spatial-value calibration and evaluation dates must not overlap"
        raise ValueError(message)
    with paths.transport_results_json.open(encoding="utf-8") as handle:
        transport = cast("dict[str, Any]", json.load(handle))
    return _AnalysisInputs(
        tickers=tickers,
        matrices=matrices,
        eigenvalues=_eigenvalues(matrices),
        calibration=calibration,
        evaluation=evaluation,
        transport=transport,
    )


def _comparison_rows(
    evaluation_fits: dict[VarianceModel, dict[str, est.FactorSarFit]],
    predictive_scores: dict[VarianceModel, dict[str, float]],
    paired_draws: pd.DataFrame,
    qlr_summaries: dict[tuple[VarianceModel, str], dict[str, object]],
    transport: dict[str, Any],
    confidence_level: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_rows = _factor_ladder_rows(transport)
    contrast_rows: list[dict[str, object]] = []
    for variance_model in _VARIANCE_MODELS:
        for model in _MODELS:
            fit = evaluation_fits[variance_model][model]
            parameters = est.factor_sar_parameter_count(fit, spatial=model != "none")
            rho_lower, rho_upper = _interval(
                paired_draws[f"rho_{variance_model}_{model}"], confidence_level
            )
            model_rows.append(
                {
                    "scope": "spatial_value",
                    "variance_model": variance_model,
                    "model": model,
                    "n_factors": fit.n_factors,
                    "n_dates": fit.n_obs,
                    "n_assets": fit.n_assets,
                    "parameter_count": parameters,
                    "rho_hat": fit.rho_hat,
                    "rho_ci_lower": rho_lower,
                    "rho_ci_upper": rho_upper,
                    "loglik": fit.loglik,
                    "held_out_score": predictive_scores[variance_model][model],
                    **_information_criteria(
                        fit.loglik, parameters, fit.n_obs, fit.n_assets
                    ),
                }
            )
        for order in _SPATIAL_MODELS:
            score_delta = (
                predictive_scores[variance_model][order]
                - predictive_scores[variance_model]["none"]
            )
            score_lower, score_upper = _interval(
                paired_draws[f"score_delta_{variance_model}_{order}"],
                confidence_level,
            )
            contrast_rows.append(
                {
                    "variance_model": variance_model,
                    "operator": order,
                    "loglik_gain": (
                        evaluation_fits[variance_model][order].loglik
                        - evaluation_fits[variance_model]["none"].loglik
                    ),
                    "held_out_score_gain": score_delta,
                    "held_out_score_gain_ci_lower": score_lower,
                    "held_out_score_gain_ci_upper": score_upper,
                    "held_out_decision": (
                        "improves"
                        if score_lower > 0.0
                        else "harms"
                        if score_upper < 0.0
                        else "unresolved"
                    ),
                    **qlr_summaries[(variance_model, order)],
                }
            )
    return model_rows, contrast_rows


def run_paper5_spatial_value_comparison(
    paths: Paper5SpatialValuePaths,
    config: Paper5SpatialValueConfig,
) -> dict[str, object]:
    """Run governed factor-only, spatial, and variance-specification comparisons."""
    config.validate()
    inputs = _load_analysis_inputs(paths, config)

    evaluation_fits = {
        variance_model: _fit_models(
            inputs.evaluation, inputs.matrices, inputs.eigenvalues, variance_model
        )
        for variance_model in _VARIANCE_MODELS
    }
    calibration_fits = {
        variance_model: _fit_models(
            inputs.calibration, inputs.matrices, inputs.eigenvalues, variance_model
        )
        for variance_model in _VARIANCE_MODELS
    }
    predictive_scores = {
        variance_model: _scores(
            calibration_fits[variance_model],
            inputs.evaluation,
            inputs.matrices,
            inputs.eigenvalues,
        )
        for variance_model in _VARIANCE_MODELS
    }
    paired_draws = _paired_bootstrap(
        inputs.calibration,
        inputs.evaluation,
        inputs.matrices,
        inputs.eigenvalues,
        config,
    )
    if not bool(paired_draws["all_solvers_converged"].all()):
        message = "one or more paired spatial-value bootstrap fits did not converge"
        raise RuntimeError(message)
    qlr_draws, qlr_summaries = _qlr_bootstrap(
        inputs.evaluation,
        inputs.matrices,
        inputs.eigenvalues,
        config,
        evaluation_fits,
    )
    if not bool(qlr_draws["all_solvers_converged"].all()):
        message = "one or more restricted-null QLR bootstrap fits did not converge"
        raise RuntimeError(message)

    registered = cast("dict[str, Any]", inputs.transport["post_corpus_factor_qmle"])
    registered_results = cast("dict[str, dict[str, Any]]", registered["results"])
    for order in _SPATIAL_MODELS:
        if not np.isclose(
            evaluation_fits["spherical"][order].rho_hat,
            float(registered_results[order]["rho_hat"]),
            rtol=0.0,
            atol=1e-10,
        ):
            message = f"spherical {order} estimate does not reproduce transport-order"
            raise RuntimeError(message)

    model_rows, contrast_rows = _comparison_rows(
        evaluation_fits,
        predictive_scores,
        paired_draws,
        qlr_summaries,
        inputs.transport,
        config.confidence_level,
    )

    identities = {
        variance_model: {
            order: _identity_diagnostics(
                evaluation_fits[variance_model][order],
                inputs.evaluation,
                inputs.matrices[order],
            )
            for order in _SPATIAL_MODELS
        }
        for variance_model in _VARIANCE_MODELS
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_id": "paper5-spatial-value-comparison-v1",
        "primary_specification": {
            "factor_model": config.factor_model,
            "variance_model": "spherical",
            "status": "prespecified_primary_unchanged",
        },
        "sample": {
            "n_assets": len(inputs.tickers),
            "calibration_start": str(
                pd.Timestamp(inputs.calibration.dates.min()).date()
            ),
            "calibration_end": str(pd.Timestamp(inputs.calibration.dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(inputs.evaluation.dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(inputs.evaluation.dates.max()).date()),
            "calibration_dates": len(inputs.calibration.dates),
            "evaluation_dates": len(inputs.evaluation.dates),
        },
        "information_criteria": {
            "aic": "-2 loglik + 2 k",
            "hqc_dates": "-2 loglik + 2 k log(log(T))",
            "hqc_firm_days": "-2 loglik + 2 k log(log(NT))",
            "bic_dates": "-2 loglik + log(T) k",
            "bic_firm_days": "-2 loglik + log(NT) k",
            "status": "working_quasi_likelihood_criteria",
        },
        "reduced_form_identity": {
            "claim": "S(rho)^-1 C(rho) equals firm-wise OLS coefficients",
            "assumptions": [
                "balanced sample",
                "fixed W",
                "common regressor set",
                "unrestricted firm coefficients",
            ],
            "diagnostics": identities,
        },
        "model_rows": model_rows,
        "contrast_rows": contrast_rows,
        "bootstrap": {
            "draws": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "seed": config.seed,
            "confidence_level": config.confidence_level,
            "paired_predictive_date_schedules": True,
            "independent_firm_paths_under_qlr_null": True,
            "qlr_null_contract": (
                "standardized firm residual paths with fitted variance restoration"
            ),
            "qlr_p_value_floor": 1.0 / (config.n_bootstrap + 1),
            "qlr_inference_status": "working_null_calibration",
        },
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(model_rows).to_parquet(
        paths.output_dir / "model_rows.parquet", index=False
    )
    pd.DataFrame.from_records(contrast_rows).to_parquet(
        paths.output_dir / "contrast_rows.parquet", index=False
    )
    paired_draws.to_parquet(paths.output_dir / "bootstrap_draws.parquet", index=False)
    qlr_draws.to_parquet(paths.output_dir / "qlr_null_draws.parquet", index=False)
    (paths.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Wrote Paper 5 spatial-value comparison to %s", paths.output_dir)
    return payload

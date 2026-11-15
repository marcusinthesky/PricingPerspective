"""Factor-conditioned W1/W2 transport-order sensitivity for Paper 5."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from pipeline.io.fama_french import (
    load_factor_data_config,
    load_factor_model,
)
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5.bootstrap import (
    independent_stationary_bootstrap_indices,
    paired_refit_schedule,
    stationary_bootstrap_indices,
)
from pipeline.stages.papers.paper5.factor_two_field import (
    FACTOR_TWO_FIELD_DRAW_COLUMNS,
    build_factor_two_field_problem,
    factor_nested_qlr_bootstrap,
    factor_two_field_bootstrap,
    factor_two_field_qmle,
    factor_two_field_with_grid,
)
from pipeline.stages.papers.paper5.operators import load_operator
from pipeline.stages.papers.paper5.two_field import TwoFieldGrid
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline.io.fama_french import (
        FactorDataConfig,
        FactorModelId,
        FrenchFactorPanel,
    )
    from pipeline.stages.papers.paper5.factor_two_field import FactorTwoFieldFit
    from pipeline.stages.papers.paper5.two_field import NestedQLRBootstrap

logger = logging.getLogger(__name__)

type FloatMatrix = NDArray[np.float64]
type FloatVector = NDArray[np.float64]
type ComplexVector = NDArray[np.complex128]

_MINIMUM_DATES = 30
_ONE_HALF = 0.5
_MIN_PROFILE_POINTS = 2
_BOUNDARY_RHO_TOLERANCE = 1e-4
_BOUNDARY_LOGLIK_TOLERANCE = 1e-4
_CORRELATION_SCALE_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class Paper5TransportOrderPaths:
    """Input and output boundaries for the transport-order stage."""

    returns_dir: Path
    oos_returns_dir: Path
    w1_barycentre_dir: Path
    w2_barycentre_dir: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5TransportOrderConfig:
    """Frozen factor, timing, resampling, and decision-rule contract."""

    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    confidence_level: float = 0.95
    equivalence_alpha: float = 0.05
    predictive_equivalence_gain_share: float = 0.01
    minimum_signal_correlation: float = 0.98
    maximum_relative_signal_difference_sd: float = 0.10
    active_weight_tolerance: float = 1e-8
    incremental_qlr_alpha: float = 0.05
    joint_theta_points: int = 401
    joint_rho_points: int = 1000
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    w1_barycentre_arm_id: str = "wasserstein_w1_loo"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w1_geometry_id: str = "wasserstein_w1"
    w2_geometry_id: str = "wasserstein_w2"
    factor_model: FactorModelId = "ff5_mom"

    def validate(self) -> None:
        """Reject invalid inference schedules before reading production data."""
        if self.n_bootstrap < 1 or self.block_length < 1:
            message = "bootstrap draws and block length must be positive"
            raise ValueError(message)
        if not 0.0 < self.confidence_level < 1.0:
            message = "confidence level must lie strictly between zero and one"
            raise ValueError(message)
        if not 0.0 < self.equivalence_alpha < _ONE_HALF:
            message = "equivalence alpha must lie strictly between zero and one half"
            raise ValueError(message)
        if not 0.0 < self.predictive_equivalence_gain_share < 1.0:
            message = (
                "predictive equivalence share must lie strictly between zero and one"
            )
            raise ValueError(message)
        if not -1.0 < self.minimum_signal_correlation <= 1.0:
            message = "minimum signal correlation must lie in (-1, 1]"
            raise ValueError(message)
        if self.maximum_relative_signal_difference_sd <= 0.0:
            message = "relative signal-difference threshold must be positive"
            raise ValueError(message)
        if not 0.0 < self.incremental_qlr_alpha < _ONE_HALF:
            message = (
                "incremental QLR alpha must lie strictly between zero and one half"
            )
            raise ValueError(message)
        if (
            self.joint_theta_points < _MIN_PROFILE_POINTS
            or self.joint_rho_points < _MIN_PROFILE_POINTS
        ):
            message = "joint W1/W2 profile grids must each contain at least two points"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _OperatorIdentity:
    provider_id: str
    representation_id: str
    barycentre_arm_id: str
    geometry_id: str


@dataclass(frozen=True, slots=True)
class _AlignedFactorPanel:
    dates: NDArray[np.datetime64]
    raw_returns: FloatMatrix
    excess_returns: FloatMatrix
    factors: FloatMatrix
    dropped_return_dates: int


@dataclass(frozen=True, slots=True)
class _FactorModelAnalysis:
    """One nested factor set estimated under both transport orders."""

    factor_panel: FrenchFactorPanel
    calibration: _AlignedFactorPanel
    evaluation: _AlignedFactorPanel
    post_fits: dict[str, est.FactorSarFit]
    predictive_fits: dict[str, est.FactorSarFit]
    daily_scores: dict[str, FloatVector]
    bootstrap: pd.DataFrame
    post_results: dict[str, dict[str, object]]
    paired_contrast: dict[str, float]
    predictive_comparison: dict[str, float]


@dataclass(frozen=True, slots=True)
class _FactorJointAnalysis:
    """Primary-model joint W1/W2 estimates and retained inference draws."""

    result: dict[str, object]
    bootstrap: pd.DataFrame
    qlr_null_draws: pd.DataFrame


def _load_operators(
    paths: Paper5TransportOrderPaths,
    config: Paper5TransportOrderConfig,
) -> tuple[list[str], dict[str, FloatMatrix], dict[str, dict[str, object]]]:
    """Load target-anchored W1 and W2 fields with exact shared identity."""
    identities = {
        "w1": _OperatorIdentity(
            config.provider_id,
            config.representation_id,
            config.w1_barycentre_arm_id,
            config.w1_geometry_id,
        ),
        "w2": _OperatorIdentity(
            config.provider_id,
            config.representation_id,
            config.w2_barycentre_arm_id,
            config.w2_geometry_id,
        ),
    }
    directories = {"w1": paths.w1_barycentre_dir, "w2": paths.w2_barycentre_dir}
    tickers: list[str] | None = None
    matrices: dict[str, FloatMatrix] = {}
    summaries: dict[str, dict[str, object]] = {}
    for order in ("w1", "w2"):
        order_tickers, matrix, summary = load_operator(
            directories[order], identities[order]
        )
        if tickers is None:
            tickers = order_tickers
        elif tickers != order_tickers:
            message = "target-anchored W1 and W2 artifacts use different ticker orders"
            raise ValueError(message)
        matrices[order] = np.asarray(matrix, dtype=np.float64)
        summaries[order] = summary
    if tickers is None:
        message = "transport-order comparison loaded no operators"
        raise ValueError(message)
    return tickers, matrices, summaries


def _align_factor_panel(
    dates: NDArray[np.datetime64],
    returns: FloatMatrix,
    panel: FrenchFactorPanel,
) -> _AlignedFactorPanel:
    """Inner-align one return panel with factors and subtract the daily RF rate."""
    factor_frame = panel.frame.set_index("Date")
    aligned = factor_frame.reindex(pd.DatetimeIndex(pd.to_datetime(dates)))
    columns = (*panel.factor_columns, "RF")
    complete = aligned.loc[:, list(columns)].notna().all(axis=1).to_numpy()
    if int(np.sum(complete)) < _MINIMUM_DATES:
        message = "fewer than 30 factor-complete return dates remain after alignment"
        raise ValueError(message)
    values = aligned.loc[complete, list(columns)].to_numpy(dtype=np.float64)
    raw_returns = np.asarray(returns[complete], dtype=np.float64)
    factors = values[:, :-1]
    risk_free = values[:, -1]
    return _AlignedFactorPanel(
        dates=np.asarray(dates[complete], dtype="datetime64[ns]"),
        raw_returns=raw_returns,
        excess_returns=raw_returns - risk_free[:, None],
        factors=np.asarray(factors, dtype=np.float64),
        dropped_return_dates=int(len(dates) - np.sum(complete)),
    )


def _validate_nested_panel_identity(
    calibration: dict[FactorModelId, _AlignedFactorPanel],
    evaluation: dict[FactorModelId, _AlignedFactorPanel],
    primary_model: FactorModelId,
) -> None:
    """Require every nested factor set to use the primary model's return panels."""
    for sample_name, panels in (
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        reference = panels[primary_model]
        for model_id, panel in panels.items():
            if not np.array_equal(panel.dates, reference.dates):
                message = (
                    f"{sample_name} dates for {model_id} differ from {primary_model}"
                )
                raise ValueError(message)
            if not np.array_equal(panel.raw_returns, reference.raw_returns):
                message = (
                    f"{sample_name} raw returns for {model_id} differ from "
                    f"{primary_model}"
                )
                raise ValueError(message)
            if not np.array_equal(panel.excess_returns, reference.excess_returns):
                message = (
                    f"{sample_name} risk-free adjustment for {model_id} differs "
                    f"from {primary_model}"
                )
                raise ValueError(message)
    calibration_end = np.max(calibration[primary_model].dates)
    evaluation_start = np.min(evaluation[primary_model].dates)
    if calibration_end >= evaluation_start:
        message = "factor-model calibration and evaluation dates must not overlap"
        raise ValueError(message)


def _correlation(left: FloatMatrix, right: FloatMatrix) -> float:
    """Return a finite pooled Pearson correlation."""
    value = float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])
    if not np.isfinite(value):
        message = "transport-order comparison produced a degenerate correlation"
        raise ValueError(message)
    return value


def _operator_separation(
    w1: FloatMatrix,
    w2: FloatMatrix,
    excess_returns: FloatMatrix,
    config: Paper5TransportOrderConfig,
    factors: FloatMatrix | None = None,
) -> dict[str, object]:
    """Measure raw and factor-residualized signal separation."""
    n_assets = w1.shape[0]
    off_diagonal = ~np.eye(n_assets, dtype=bool)
    w1_values = w1[off_diagonal]
    w2_values = w2[off_diagonal]
    active_w1 = w1_values > config.active_weight_tolerance
    active_w2 = w2_values > config.active_weight_tolerance
    union = int(np.sum(active_w1 | active_w2))
    intersection = int(np.sum(active_w1 & active_w2))
    row_correlations = np.asarray(
        [
            _correlation(w1[index : index + 1], w2[index : index + 1])
            for index in range(n_assets)
        ],
        dtype=np.float64,
    )
    signal_w1 = excess_returns @ w1.T
    signal_w2 = excess_returns @ w2.T
    signal_difference = signal_w2 - signal_w1
    signal_correlation = _correlation(signal_w1, signal_w2)
    signal_scale = float(np.std(signal_w2, ddof=1))
    if signal_scale <= 0.0:
        message = "W2 induced signal has zero dispersion"
        raise ValueError(message)
    relative_difference_sd = float(np.std(signal_difference, ddof=1) / signal_scale)
    residualized = None
    if factors is not None:
        design = np.column_stack((np.ones(len(factors), dtype=np.float64), factors))
        signal_pair = np.column_stack((signal_w1, signal_w2))
        signal_coefficients, *_ = np.linalg.lstsq(design, signal_pair, rcond=None)
        residualized = cast("FloatMatrix", signal_pair - design @ signal_coefficients)
    if residualized is not None:
        residualized_w1 = residualized[:, :n_assets]
        residualized_w2 = residualized[:, n_assets:]
        residualized_difference = residualized_w2 - residualized_w1
        residualized_scale = float(np.std(residualized_w2, ddof=1))
        if residualized_scale <= 0.0:
            message = "factor-residualized W2 induced signal has zero dispersion"
            raise ValueError(message)
        residualized_correlation = _correlation(residualized_w1, residualized_w2)
        residualized_relative_difference = float(
            np.std(residualized_difference, ddof=1) / residualized_scale
        )
    else:
        residualized_correlation = None
        residualized_relative_difference = None
    weakly_separated = bool(
        signal_correlation >= config.minimum_signal_correlation
        and relative_difference_sd <= config.maximum_relative_signal_difference_sd
    )
    return {
        "off_diagonal_weight_correlation": _correlation(
            w1_values[None, :], w2_values[None, :]
        ),
        "off_diagonal_mean_absolute_difference": float(
            np.mean(np.abs(w2_values - w1_values))
        ),
        "off_diagonal_rms_difference": float(
            np.sqrt(np.mean(np.square(w2_values - w1_values)))
        ),
        "active_support_jaccard": float(intersection / union),
        "row_correlation_minimum": float(np.min(row_correlations)),
        "row_correlation_median": float(np.median(row_correlations)),
        "row_correlation_maximum": float(np.max(row_correlations)),
        "induced_signal_correlation": signal_correlation,
        "induced_signal_difference_sd": float(np.std(signal_difference, ddof=1)),
        "w2_induced_signal_sd": signal_scale,
        "relative_signal_difference_sd": relative_difference_sd,
        "factor_residualized_signal_correlation": residualized_correlation,
        "factor_residualized_relative_signal_difference_sd": (
            residualized_relative_difference
        ),
        "rule": {
            "minimum_signal_correlation": config.minimum_signal_correlation,
            "maximum_relative_signal_difference_sd": (
                config.maximum_relative_signal_difference_sd
            ),
            "combination": "both",
        },
        "decision": (
            "weakly_separated_do_not_orthogonalize" if weakly_separated else "separated"
        ),
        "weakly_separated": weakly_separated,
        "orthogonalization": (
            "not_performed_joint_unchanged_field_model_reported_separately"
        ),
    }


def _eigenvalues(matrices: dict[str, FloatMatrix]) -> dict[str, ComplexVector]:
    """Cache fixed-operator eigenvalues across all bootstrap refits."""
    return {
        order: cast("ComplexVector", np.linalg.eigvals(matrix))
        for order, matrix in matrices.items()
    }


def _fit_factor_models(
    panel: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, ComplexVector],
    indices: NDArray[np.int64] | None = None,
) -> dict[str, est.FactorSarFit]:
    """Fit both transport orders on one common panel or resample."""
    selected = (
        np.arange(len(panel.dates), dtype=np.int64) if indices is None else indices
    )
    return {
        order: est.sar_factor_qmle(
            panel.excess_returns[selected],
            matrix,
            panel.factors[selected],
            compute_se=indices is None,
            eigvals_w=eigenvalues[order],
        )
        for order, matrix in matrices.items()
    }


def _paired_bootstrap(
    calibration: _AlignedFactorPanel,
    evaluation: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, ComplexVector],
    *,
    draws: int,
    block_length: int,
    seed: int,
    model_id: str = "factor_model",
) -> pd.DataFrame:
    """Refit factors and rho on paired date-block draws for fit and score evidence."""
    schedule = paired_refit_schedule(
        calibration_observations=len(calibration.dates),
        evaluation_observations=len(evaluation.dates),
        draws=draws,
        expected_block_length=block_length,
        seed=seed,
    )
    calibration_fit_indices = schedule.calibration
    evaluation_fit_indices = schedule.evaluation_fit
    evaluation_score_indices = schedule.evaluation_score
    records: list[dict[str, object]] = []
    for draw_id in range(draws):
        fit_evaluation = _fit_factor_models(
            evaluation,
            matrices,
            eigenvalues,
            evaluation_fit_indices[draw_id],
        )
        fit_calibration = _fit_factor_models(
            calibration,
            matrices,
            eigenvalues,
            calibration_fit_indices[draw_id],
        )
        score_index = evaluation_score_indices[draw_id]
        predictive_scores = {
            order: est.sar_factor_daily_logscore(
                fit_calibration[order],
                evaluation.excess_returns[score_index],
                matrices[order],
                evaluation.factors[score_index],
                eigvals_w=eigenvalues[order],
            )
            for order in ("w1", "w2")
        }
        score_w1 = float(np.mean(predictive_scores["w1"]) / matrices["w1"].shape[0])
        score_w2 = float(np.mean(predictive_scores["w2"]) / matrices["w2"].shape[0])
        records.append(
            {
                "draw_id": draw_id,
                "rho_w1": fit_evaluation["w1"].rho_hat,
                "rho_w2": fit_evaluation["w2"].rho_hat,
                "rho_w2_minus_w1": (
                    fit_evaluation["w2"].rho_hat - fit_evaluation["w1"].rho_hat
                ),
                "loglik_w1": fit_evaluation["w1"].loglik,
                "loglik_w2": fit_evaluation["w2"].loglik,
                "loglik_w2_minus_w1": (
                    fit_evaluation["w2"].loglik - fit_evaluation["w1"].loglik
                ),
                "predictive_score_w1": score_w1,
                "predictive_score_w2": score_w2,
                "predictive_score_w2_minus_w1": score_w2 - score_w1,
                "all_solvers_converged": all(
                    fit.converged
                    for fit in (*fit_evaluation.values(), *fit_calibration.values())
                ),
            }
        )
        if (draw_id + 1) % 100 == 0 or draw_id + 1 == draws:
            logger.info(
                "Paper 5 %s transport-order bootstrap %d/%d",
                model_id,
                draw_id + 1,
                draws,
            )
    return pd.DataFrame.from_records(records)


def _interval(values: pd.Series, confidence_level: float) -> tuple[float, float]:
    """Return a central empirical bootstrap interval."""
    tail = (1.0 - confidence_level) / 2.0
    return float(values.quantile(tail)), float(values.quantile(1.0 - tail))


def _fit_record(
    fit: est.FactorSarFit,
    rho_interval: tuple[float, float],
) -> dict[str, object]:
    """Serialize one factor-profiled fit without persisting firm coefficients."""
    intercepts = fit.coefficients[0]
    loadings = fit.coefficients[1:]
    return {
        "rho_hat": fit.rho_hat,
        "rho_ci_lower": rho_interval[0],
        "rho_ci_upper": rho_interval[1],
        "rho_hessian_se": fit.rho_se,
        "sigma2_hat": fit.sigma2_hat,
        "loglik": fit.loglik,
        "loglik_per_firm_day": fit.loglik / (fit.n_obs * fit.n_assets),
        "converged": fit.converged,
        "refinement_gain": fit.refinement_gain,
        "trading_days": fit.n_obs,
        "n_assets": fit.n_assets,
        "n_factors": fit.n_factors,
        "firm_intercept_mean": float(np.mean(intercepts)),
        "firm_intercept_abs_mean": float(np.mean(np.abs(intercepts))),
        "factor_loading_abs_mean": float(np.mean(np.abs(loadings))),
    }


def _analyse_factor_model(
    factor_panel: FrenchFactorPanel,
    calibration: _AlignedFactorPanel,
    evaluation: _AlignedFactorPanel,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, ComplexVector],
    config: Paper5TransportOrderConfig,
) -> _FactorModelAnalysis:
    """Estimate and resample one nested factor specification."""
    post_fits = _fit_factor_models(evaluation, matrices, eigenvalues)
    predictive_fits = _fit_factor_models(calibration, matrices, eigenvalues)
    daily_scores = {
        order: est.sar_factor_daily_logscore(
            predictive_fits[order],
            evaluation.excess_returns,
            matrices[order],
            evaluation.factors,
            eigvals_w=eigenvalues[order],
        )
        for order in ("w1", "w2")
    }
    bootstrap = _paired_bootstrap(
        calibration,
        evaluation,
        matrices,
        eigenvalues,
        draws=config.n_bootstrap,
        block_length=config.block_length,
        seed=config.seed,
        model_id=factor_panel.model_id,
    )
    if not bool(bootstrap["all_solvers_converged"].all()):
        message = (
            f"one or more {factor_panel.model_id} factor-profiled bootstrap fits "
            "did not converge"
        )
        raise RuntimeError(message)

    rho_intervals = {
        order: _interval(bootstrap[f"rho_{order}"], config.confidence_level)
        for order in ("w1", "w2")
    }
    post_results = {
        order: _fit_record(post_fits[order], rho_intervals[order])
        for order in ("w1", "w2")
    }
    n_assets = matrices["w2"].shape[0]
    score_w1 = float(np.mean(daily_scores["w1"]) / n_assets)
    score_w2 = float(np.mean(daily_scores["w2"]) / n_assets)
    delta_rho_interval = _interval(
        bootstrap["rho_w2_minus_w1"], config.confidence_level
    )
    delta_loglik_interval = _interval(
        bootstrap["loglik_w2_minus_w1"], config.confidence_level
    )
    predictive_interval = _interval(
        bootstrap["predictive_score_w2_minus_w1"], config.confidence_level
    )
    return _FactorModelAnalysis(
        factor_panel=factor_panel,
        calibration=calibration,
        evaluation=evaluation,
        post_fits=post_fits,
        predictive_fits=predictive_fits,
        daily_scores=daily_scores,
        bootstrap=bootstrap,
        post_results=post_results,
        paired_contrast={
            "rho": post_fits["w2"].rho_hat - post_fits["w1"].rho_hat,
            "rho_ci_lower": delta_rho_interval[0],
            "rho_ci_upper": delta_rho_interval[1],
            "loglik": post_fits["w2"].loglik - post_fits["w1"].loglik,
            "loglik_ci_lower": delta_loglik_interval[0],
            "loglik_ci_upper": delta_loglik_interval[1],
        },
        predictive_comparison={
            "w1_score": score_w1,
            "w2_score": score_w2,
            "w2_minus_w1": score_w2 - score_w1,
            "w2_minus_w1_ci_lower": predictive_interval[0],
            "w2_minus_w1_ci_upper": predictive_interval[1],
        },
    )


def _factor_joint_fit_record(fit: FactorTwoFieldFit) -> dict[str, object]:
    """Serialize a factor-conditioned joint fit without firm coefficients."""
    return {
        "rho_w1": fit.rho_w1,
        "rho_w2": fit.rho_w2,
        "rho_total": fit.rho_total,
        "theta_w1": fit.theta_w1,
        "sigma2_hat": fit.sigma2_hat,
        "loglik": fit.loglik,
        "loglik_per_firm_day": fit.loglik / (fit.n_obs * fit.n_assets),
        "converged": fit.converged,
        "trading_days": fit.n_obs,
        "n_assets": fit.n_assets,
        "n_factors": fit.n_factors,
        "at_w1_only_boundary": fit.at_w1_boundary,
        "at_w2_only_boundary": fit.at_w2_boundary,
    }


def _factor_joint_bootstrap_summary(
    draws: pd.DataFrame,
    confidence_level: float,
) -> dict[str, object]:
    """Summarize joint coefficient uncertainty and boundary incidence."""
    rho_w1_sd = float(draws["rho_w1"].std(ddof=1))
    rho_w2_sd = float(draws["rho_w2"].std(ddof=1))
    if min(rho_w1_sd, rho_w2_sd) <= _CORRELATION_SCALE_FLOOR:
        correlation: float | None = None
        correlation_status = "undefined_zero_variance_at_boundary"
    else:
        correlation = float(draws["rho_w1"].corr(draws["rho_w2"]))
        if not np.isfinite(correlation):
            message = "factor-conditioned joint W1/W2 correlation is nonfinite"
            raise RuntimeError(message)
        correlation_status = "estimated"
    result: dict[str, object] = {
        "rho_channel_correlation": correlation,
        "rho_channel_correlation_status": correlation_status,
        "interval_status": "descriptive_boundary_bootstrap_quantiles",
        "rho_w1_bootstrap_sd": rho_w1_sd,
        "rho_w2_bootstrap_sd": rho_w2_sd,
        "w1_only_boundary_share": float(draws["at_w1_boundary"].mean()),
        "w2_only_boundary_share": float(draws["at_w2_boundary"].mean()),
    }
    for column in ("rho_w1", "rho_w2", "rho_total", "theta_w1"):
        lower, upper = _interval(draws[column], confidence_level)
        result[f"{column}_ci_lower"] = lower
        result[f"{column}_ci_upper"] = upper
    return result


def _factor_qlr_record(
    test: NestedQLRBootstrap,
    *,
    null: str,
    tested_channel: str,
    alpha: float,
) -> dict[str, object]:
    """Serialize one restricted-null incremental transport-order test."""
    reject = test.p_value < alpha
    return {
        "null": null,
        "tested_channel": tested_channel,
        "statistic": test.statistic,
        "bootstrap_p_value": test.p_value,
        "bootstrap_exceedances": test.exceedances,
        "p_value_floor": 1.0 / (test.draws.size + 1),
        "inference_status": "working_null_calibration",
        "alpha": alpha,
        "decision": "reject" if reject else "do_not_reject",
    }


def _incremental_geometry_conclusion(
    *,
    w2_given_w1_rejected: bool,
    w1_given_w2_rejected: bool,
) -> str:
    """Map the two directional tests to a bounded descriptive conclusion."""
    if w2_given_w1_rejected and w1_given_w2_rejected:
        return "both_orders_add_incremental_conditional_fit"
    if w2_given_w1_rejected:
        return "w2_adds_beyond_w1_but_not_reverse"
    if w1_given_w2_rejected:
        return "w1_adds_beyond_w2_but_not_reverse"
    return "neither_order_adds_detectable_incremental_conditional_fit"


def _analyse_factor_joint_model(
    primary: _FactorModelAnalysis,
    matrices: dict[str, FloatMatrix],
    config: Paper5TransportOrderConfig,
) -> _FactorJointAnalysis:
    """Estimate the primary FF5+MOM joint W1/W2 model and both nested QLRs."""
    joint_grid = TwoFieldGrid.build(
        theta_points=config.joint_theta_points,
        rho_points=config.joint_rho_points,
    )
    joint_problem = build_factor_two_field_problem(
        primary.evaluation.excess_returns,
        matrices["w1"],
        matrices["w2"],
        primary.evaluation.factors,
        joint_grid,
    )
    boundary_problems = {
        "w1_only": factor_two_field_with_grid(
            joint_problem,
            TwoFieldGrid.build(
                theta_points=1,
                rho_points=config.joint_rho_points,
                theta_lower=1.0,
                theta_upper=1.0,
            ),
        ),
        "w2_only": factor_two_field_with_grid(
            joint_problem,
            TwoFieldGrid.build(
                theta_points=1,
                rho_points=config.joint_rho_points,
                theta_lower=0.0,
                theta_upper=0.0,
            ),
        ),
    }
    joint_fit = factor_two_field_qmle(joint_problem)
    boundary_fits = {
        name: factor_two_field_qmle(problem)
        for name, problem in boundary_problems.items()
    }
    if not all(
        (
            joint_fit.converged,
            boundary_fits["w1_only"].converged,
            boundary_fits["w2_only"].converged,
        )
    ):
        message = "factor-conditioned observed joint W1/W2 fit did not converge"
        raise RuntimeError(message)

    for order, boundary_name in (("w1", "w1_only"), ("w2", "w2_only")):
        expected = primary.post_fits[order]
        boundary = boundary_fits[boundary_name]
        if (
            abs(boundary.rho_total - expected.rho_hat) > _BOUNDARY_RHO_TOLERANCE
            or abs(boundary.loglik - expected.loglik) > _BOUNDARY_LOGLIK_TOLERANCE
        ):
            message = (
                f"factor-conditioned {boundary_name} boundary does not reproduce "
                f"the {order} single-field fit"
            )
            raise RuntimeError(message)

    indices = stationary_bootstrap_indices(
        len(primary.evaluation.dates),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed,
    )
    logger.info("Estimating Paper 5 primary factor-conditioned joint W1/W2 bootstrap")
    bootstrap_values = factor_two_field_bootstrap(joint_problem, indices)
    bootstrap = pd.DataFrame(
        bootstrap_values,
        columns=list(FACTOR_TWO_FIELD_DRAW_COLUMNS),
    )
    bootstrap.insert(0, "draw_id", np.arange(len(bootstrap), dtype=np.int64))
    if not bool(bootstrap["converged"].all()):
        message = "one or more factor-conditioned joint W1/W2 refits did not converge"
        raise RuntimeError(message)

    null_indices = independent_stationary_bootstrap_indices(
        len(primary.evaluation.dates),
        primary.evaluation.excess_returns.shape[1],
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed + 10_000,
    )
    logger.info("Calibrating W2-given-W1 restricted-boundary QLR")
    w2_given_w1 = factor_nested_qlr_bootstrap(
        joint_problem,
        boundary_problems["w1_only"],
        null_indices,
    )
    logger.info("Calibrating W1-given-W2 restricted-boundary QLR")
    w1_given_w2 = factor_nested_qlr_bootstrap(
        joint_problem,
        boundary_problems["w2_only"],
        null_indices,
    )
    qlr_tests = {
        "w2_given_w1": _factor_qlr_record(
            w2_given_w1,
            null="rho_w2 = 0",
            tested_channel="W2 transport-order field",
            alpha=config.incremental_qlr_alpha,
        ),
        "w1_given_w2": _factor_qlr_record(
            w1_given_w2,
            null="rho_w1 = 0",
            tested_channel="W1 transport-order field",
            alpha=config.incremental_qlr_alpha,
        ),
    }
    conclusion = _incremental_geometry_conclusion(
        w2_given_w1_rejected=qlr_tests["w2_given_w1"]["decision"] == "reject",
        w1_given_w2_rejected=qlr_tests["w1_given_w2"]["decision"] == "reject",
    )
    boundary_records = {
        name: {
            "rho": fit.rho_total,
            "loglik": fit.loglik,
            "matches_single_field": True,
        }
        for name, fit in boundary_fits.items()
    }
    result: dict[str, object] = {
        "working_model": (
            "(I-rho_w1 W1-rho_w2 W2) r_excess = alpha_i + A_i f_t + epsilon"
        ),
        "factor_model": primary.factor_panel.model_id,
        "fields_entered_unchanged": True,
        "literal_matrix_orthogonalization": False,
        "parameter_region": "rho_w1 >= 0, rho_w2 >= 0, rho_w1 + rho_w2 < 1",
        "joint": _factor_joint_fit_record(joint_fit),
        "boundaries": boundary_records,
        "loglik_gain_vs_w1_only": (joint_fit.loglik - boundary_fits["w1_only"].loglik),
        "loglik_gain_vs_w2_only": (joint_fit.loglik - boundary_fits["w2_only"].loglik),
        "bootstrap": _factor_joint_bootstrap_summary(
            bootstrap, config.confidence_level
        ),
        "quasi_lr_tests": qlr_tests,
        "incremental_geometry_conclusion": conclusion,
        "claim_boundary": (
            "incremental conditional transport-geometry fit; not investment-cost "
            "curvature and not a causal channel decomposition"
        ),
        "follow_up_status": (
            "results-aware diagnostic added after the standalone W1/W2 comparison"
        ),
        "profile": {
            "theta_points": config.joint_theta_points,
            "rho_points": config.joint_rho_points,
            "theta_range": [0.0, 1.0],
            "rho_range": [0.0, 0.999],
            "refinement": "bounded Brent inside the grid cell of the maximum",
        },
    }
    qlr_null_draws = pd.concat(
        [
            pd.DataFrame(
                {
                    "test": name,
                    "draw_id": np.arange(len(test.draws), dtype=np.int64),
                    "qlr": test.draws,
                }
            )
            for name, test in (
                ("w2_given_w1", w2_given_w1),
                ("w1_given_w2", w1_given_w2),
            )
        ],
        ignore_index=True,
    )
    return _FactorJointAnalysis(
        result=result,
        bootstrap=bootstrap,
        qlr_null_draws=qlr_null_draws,
    )


def _primary_and_joint_analysis(
    analyses: dict[FactorModelId, _FactorModelAnalysis],
    primary_model: FactorModelId,
    matrices: dict[str, FloatMatrix],
    config: Paper5TransportOrderConfig,
) -> tuple[_FactorModelAnalysis, _FactorJointAnalysis]:
    """Resolve the primary factor model and run its joint W1/W2 extension."""
    primary = analyses[primary_model]
    return primary, _analyse_factor_joint_model(primary, matrices, config)


def _governed_model_ids(
    factor_config: FactorDataConfig,
    configured_primary: FactorModelId,
) -> tuple[FactorModelId, ...]:
    """Return the frozen diagnostic order after validating the primary model."""
    if configured_primary != factor_config.primary_model:
        message = (
            f"CLI primary factor model {configured_primary} disagrees with "
            f"the governed {factor_config.primary_model} model"
        )
        raise ValueError(message)
    return (*factor_config.diagnostic_models, factor_config.primary_model)


def _plain(value: object) -> object:
    """Convert NumPy values for deterministic JSON serialization."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run_paper5_transport_order_sensitivity(
    paths: Paper5TransportOrderPaths,
    config: Paper5TransportOrderConfig,
) -> dict[str, object]:
    """Run governed nested-factor and W1/W2 transport-order comparisons."""
    config.validate()
    tickers, matrices, operator_summaries = _load_operators(paths, config)
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    model_ids = _governed_model_ids(factor_config, config.factor_model)
    factor_panels = {
        model_id: load_factor_model(factor_config, model_id) for model_id in model_ids
    }

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
    calibration_panels = {
        model_id: _align_factor_panel(
            cast("NDArray[np.datetime64]", calibration_dates),
            cast("FloatMatrix", calibration_returns),
            factor_panel,
        )
        for model_id, factor_panel in factor_panels.items()
    }
    evaluation_panels = {
        model_id: _align_factor_panel(
            cast("NDArray[np.datetime64]", evaluation_dates),
            cast("FloatMatrix", evaluation_returns),
            factor_panel,
        )
        for model_id, factor_panel in factor_panels.items()
    }
    _validate_nested_panel_identity(
        calibration_panels,
        evaluation_panels,
        factor_config.primary_model,
    )

    eigenvalues = _eigenvalues(matrices)
    analyses: dict[FactorModelId, _FactorModelAnalysis] = {}
    for model_id in model_ids:
        logger.info("Estimating Paper 5 nested factor model %s", model_id)
        analyses[model_id] = _analyse_factor_model(
            factor_panels[model_id],
            calibration_panels[model_id],
            evaluation_panels[model_id],
            matrices,
            eigenvalues,
            config,
        )

    primary, joint_analysis = _primary_and_joint_analysis(
        analyses, factor_config.primary_model, matrices, config
    )
    raw_fits = {
        order: est.sar_qmle(primary.evaluation.raw_returns, matrix)
        for order, matrix in matrices.items()
    }
    null_fit = est.sar_factor_fit_at_rho(
        primary.calibration.excess_returns,
        matrices["w2"],
        primary.calibration.factors,
        0.0,
        eigvals_w=eigenvalues["w2"],
    )
    calibration_w2_scores = est.sar_factor_daily_logscore(
        primary.predictive_fits["w2"],
        primary.calibration.excess_returns,
        matrices["w2"],
        primary.calibration.factors,
        eigvals_w=eigenvalues["w2"],
    )
    calibration_null_scores = est.sar_factor_daily_logscore(
        null_fit,
        primary.calibration.excess_returns,
        matrices["w2"],
        primary.calibration.factors,
        eigvals_w=eigenvalues["w2"],
    )
    n_assets = len(tickers)
    calibration_gain = float(
        np.mean(calibration_w2_scores - calibration_null_scores) / n_assets
    )
    if calibration_gain <= 0.0:
        message = "W2 calibration score gain must be positive to calibrate equivalence"
        raise ValueError(message)
    equivalence_margin = config.predictive_equivalence_gain_share * calibration_gain

    bootstrap = primary.bootstrap
    equivalence_interval = (
        float(
            bootstrap["predictive_score_w2_minus_w1"].quantile(config.equivalence_alpha)
        ),
        float(
            bootstrap["predictive_score_w2_minus_w1"].quantile(
                1.0 - config.equivalence_alpha
            )
        ),
    )
    equivalent = bool(
        equivalence_interval[0] >= -equivalence_margin
        and equivalence_interval[1] <= equivalence_margin
    )
    operator_separation = _operator_separation(
        matrices["w1"],
        matrices["w2"],
        primary.evaluation.excess_returns,
        config,
        primary.evaluation.factors,
    )

    post_results = {order: dict(primary.post_results[order]) for order in ("w1", "w2")}
    for order in ("w1", "w2"):
        post_results[order]["unadjusted_raw_return_rho"] = raw_fits[order]["rho_hat"]
        post_results[order]["factor_conditioning_rho_change"] = primary.post_fits[
            order
        ].rho_hat - float(raw_fits[order]["rho_hat"])

    equivalence = {
        "status": "equivalent" if equivalent else "not_equivalent",
        "alpha": config.equivalence_alpha,
        "interval_level": 1.0 - 2.0 * config.equivalence_alpha,
        "interval_lower": equivalence_interval[0],
        "interval_upper": equivalence_interval[1],
        "margin": equivalence_margin,
        "calibration_rule": (
            "one per cent of the W2-versus-rho-zero mean calibration "
            "score gain per firm-day"
        ),
        "calibration_gain": calibration_gain,
        "gain_share": config.predictive_equivalence_gain_share,
        "registered_timing": (
            "frozen before governed production rerun after feasibility audit"
        ),
    }
    primary_predictive = {
        **primary.predictive_comparison,
        "equivalence": equivalence,
    }
    factor_comparison_results = {
        model_id: {
            "primary": model_id == factor_config.primary_model,
            "factor_columns": list(analysis.factor_panel.factor_columns),
            "source_hashes": analysis.factor_panel.source_hashes,
            "source_merge_dropped_dates": analysis.factor_panel.dropped_dates,
            "post_corpus_results": analysis.post_results,
            "paired_contrast_w2_minus_w1": analysis.paired_contrast,
            "predictive_comparison": analysis.predictive_comparison,
        }
        for model_id, analysis in analyses.items()
    }

    payload: dict[str, object] = {
        "schema_version": 3,
        "analysis_id": "paper5-nested-factor-transport-order-v3",
        "claim_label": "transport-order sensitivity",
        "claim_boundary": {
            "supported": (
                "transport-order sensitivity and nested incremental fit conditional "
                "on common factors"
            ),
            "not_supported": "linear-versus-convex investment adjustment costs",
            "paper6_structural_no_go": "unchanged",
        },
        "identity": {
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "factor_model": factor_config.primary_model,
            "factor_models": list(model_ids),
            "w1_barycentre_arm_id": config.w1_barycentre_arm_id,
            "w2_barycentre_arm_id": config.w2_barycentre_arm_id,
            "w1_geometry_id": config.w1_geometry_id,
            "w2_geometry_id": config.w2_geometry_id,
        },
        "factor_data": {
            "primary_model": factor_config.primary_model,
            "diagnostic_models": list(factor_config.diagnostic_models),
            "factor_columns": list(primary.factor_panel.factor_columns),
            "risk_free_subtracted": True,
            "firm_specific_intercepts": True,
            "firm_specific_factor_loadings": True,
            "source_hashes": primary.factor_panel.source_hashes,
            "source_merge_dropped_dates": primary.factor_panel.dropped_dates,
        },
        "sample": {
            "tickers": tickers,
            "n_assets": n_assets,
            "calibration": {
                "role": "fit predictive parameters before evaluation",
                "start": str(pd.Timestamp(primary.calibration.dates.min()).date()),
                "end": str(pd.Timestamp(primary.calibration.dates.max()).date()),
                "factor_complete_dates": len(primary.calibration.dates),
                "factor_missing_return_dates": (
                    primary.calibration.dropped_return_dates
                ),
            },
            "evaluation": {
                "role": "post-corpus estimation and held-out predictive scoring",
                "start": str(pd.Timestamp(primary.evaluation.dates.min()).date()),
                "end": str(pd.Timestamp(primary.evaluation.dates.max()).date()),
                "factor_complete_dates": len(primary.evaluation.dates),
                "factor_missing_return_dates": primary.evaluation.dropped_return_dates,
            },
        },
        "operator_artifacts": {
            order: {
                "path": str(
                    paths.w1_barycentre_dir
                    if order == "w1"
                    else paths.w2_barycentre_dir
                ),
                "summary": operator_summaries[order],
            }
            for order in ("w1", "w2")
        },
        "operator_separation": operator_separation,
        "factor_model_comparison": {
            "design": (
                "Matched official Fama-French factor-family controls; additions "
                "within a family are nested, while FF3 versus FF5 is not"
            ),
            "source_citation_key": "ge_news-implied_2023",
            "extension": (
                "FF3+MOM is included as an additional bridge between FF3 and "
                "FF5 specifications"
            ),
            "model_order": list(model_ids),
            "primary_model": factor_config.primary_model,
            "identical_return_dates_across_models": True,
            "common_bootstrap_schedule_across_models": True,
            "results": factor_comparison_results,
        },
        "post_corpus_factor_qmle": {
            "working_model": "(I-rho W) r_excess = alpha_i + A_i f_t + epsilon",
            "primary": True,
            "results": post_results,
            "paired_contrast_w2_minus_w1": primary.paired_contrast,
        },
        "post_corpus_factor_joint_qmle": joint_analysis.result,
        "predictive_comparison": {
            "temporal_design": "fit 2018-2022 parameters; score 2023-2026 dates",
            "score_unit": (
                "mean Gaussian quasi-log score per firm-day; higher is better"
            ),
            **primary_predictive,
        },
        "inference": {
            "method": "paired joint-date stationary bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "seed": config.seed,
            "factor_refit_each_draw": True,
            "rho_refit_each_draw": True,
            "predictive_training_refit_each_draw": True,
            "same_draw_indices_across_factor_models": True,
            "confidence_level": config.confidence_level,
            "joint_w1_w2": {
                "method": "paired joint-date stationary bootstrap",
                "factor_refit_each_draw": True,
                "fields_entered_unchanged": True,
                "nested_qlr_method": (
                    "restricted-null asset-wise stationary residual bootstrap"
                ),
                "nested_qlr_null_contract": (
                    "standardized firm residual paths with fitted variance restoration"
                ),
                "nested_qlr_factor_refit_each_null_draw": True,
                "nested_qlr_p_value_rule": ("(1 + exceedances) / (n_bootstrap + 1)"),
                "nested_qlr_p_value_floor": 1.0 / (config.n_bootstrap + 1),
                "alpha": config.incremental_qlr_alpha,
            },
        },
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "results.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bootstrap_frames: list[pd.DataFrame] = []
    predictive_frames: list[pd.DataFrame] = []
    for model_id, analysis in analyses.items():
        model_bootstrap = analysis.bootstrap.copy()
        model_bootstrap.insert(0, "factor_model", model_id)
        bootstrap_frames.append(model_bootstrap)
        predictive_frames.append(
            pd.DataFrame(
                {
                    "factor_model": model_id,
                    "Date": analysis.evaluation.dates,
                    "w1_score": analysis.daily_scores["w1"] / n_assets,
                    "w2_score": analysis.daily_scores["w2"] / n_assets,
                    "w2_minus_w1": (
                        analysis.daily_scores["w2"] - analysis.daily_scores["w1"]
                    )
                    / n_assets,
                }
            )
        )
    pd.concat(bootstrap_frames, ignore_index=True).to_parquet(
        paths.output_dir / "bootstrap_draws.parquet", index=False
    )
    pd.concat(predictive_frames, ignore_index=True).to_parquet(
        paths.output_dir / "predictive_scores.parquet", index=False
    )
    joint_analysis.bootstrap.to_parquet(
        paths.output_dir / "joint_bootstrap_draws.parquet", index=False
    )
    joint_analysis.qlr_null_draws.to_parquet(
        paths.output_dir / "joint_qlr_null_draws.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "factor_model": model_id,
                "primary_factor_model": model_id == factor_config.primary_model,
                "transport_order": order,
                **{
                    key: value
                    for key, value in (
                        post_results[order]
                        if model_id == factor_config.primary_model
                        else analysis.post_results[order]
                    ).items()
                    if not isinstance(value, (dict, list))
                },
                "predictive_score": analysis.predictive_comparison[f"{order}_score"],
            }
            for model_id, analysis in analyses.items()
            for order in ("w1", "w2")
        ]
    ).to_parquet(paths.output_dir / "results.parquet", index=False)
    return payload


__all__ = [
    "Paper5TransportOrderConfig",
    "Paper5TransportOrderPaths",
    "run_paper5_transport_order_sensitivity",
]

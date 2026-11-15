"""Joint two-field spatial estimates and interaction-field overlap diagnostics.

The stage answers one question the separate single-field fits cannot: whether
the barycentric transport operator and the persistent news co-mention graph
measure the same interaction field. It reports the geometry and induced-field
overlap between them, then admits both as priced channels in one exposure
equilibrium and profiles the joint quasi-likelihood over the mixture weight and
the total spatial feedback.

Both single-field fits are re-estimated here as pinned boundaries of the same
nested family, so the reported likelihood gains are exact rather than compared
across estimators. Every interaction matrix is frozen before the evaluation
window, exactly as in the single-field stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.io.fama_french import load_factor_data_config, load_factor_model
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5.bootstrap import (
    independent_stationary_bootstrap_indices,
    paired_refit_provenance,
    paired_refit_schedule,
    stationary_bootstrap_indices,
)
from pipeline.stages.papers.paper5.co_mentions import load_co_mentions_w
from pipeline.stages.papers.paper5.factor_two_field import (
    FACTOR_TWO_FIELD_DRAW_COLUMNS,
    build_factor_two_field_problem,
    factor_nested_qlr_bootstrap,
    factor_two_field_bootstrap,
    factor_two_field_daily_logscore,
    factor_two_field_problem_at,
    factor_two_field_qmle,
    factor_two_field_with_grid,
)
from pipeline.stages.papers.paper5.field_overlap import (
    induced_field_similarity,
    matched_random_induced_null,
    matched_random_support_null,
    support_overlap,
    weight_similarity,
)
from pipeline.stages.papers.paper5.operators import load_operator
from pipeline.stages.papers.paper5.periods import (
    ANNUAL_FOLDS,
    POST_CORPUS_END_DATE,
    POST_CORPUS_START_DATE,
)
from pipeline.stages.papers.paper5.two_field import (
    TwoFieldGrid,
    build_two_field_problem,
    channel_intensity,
    nested_qlr_bootstrap,
    two_field_bootstrap,
    two_field_profile_surface,
    two_field_qmle,
)
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline.io.fama_french import FrenchFactorPanel
    from pipeline.stages.papers.paper5.bootstrap import PairedRefitSchedule
    from pipeline.stages.papers.paper5.contracts import ScoreContrast
    from pipeline.stages.papers.paper5.factor_two_field import (
        FactorTwoFieldFit,
        FactorTwoFieldProblem,
    )
    from pipeline.stages.papers.paper5.two_field import (
        NestedQLRBootstrap,
        TwoFieldProblem,
    )

_POOLED_PERIOD = "pooled_2023_2026"
_OPERATOR_WINDOW = "2018-01-01/2022-12-31"
_CALIBRATION_START = "2018-01-01"
_CALIBRATION_END = "2022-12-31"
_LOWER_QUANTILE = 0.025
_UPPER_QUANTILE = 0.975
_MINIMUM_FACTOR_DATES = 30
_QLR_ALPHA = 0.05
# Coarse profile resolution retained for the joint-region exhibit only; the
# reported estimates always come from the full-resolution profile above.
_SURFACE_THETA_POINTS = 81
_SURFACE_RHO_POINTS = 81
_QLR_METHOD = "null-imposed stationary residual-block bootstrap"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper5TwoFieldPaths:
    """Input and output boundaries for the two-field stage."""

    returns_dir: Path
    oos_returns_dir: Path
    shared_barycentre_dir: Path
    co_mentions_adjacency: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True)
class _ExhibitInputs:
    """Panel, frozen fields, resample cloud, and destination for the exhibit."""

    returns: NDArray[np.float64]
    w_flat: NDArray[np.float64]
    w_news: NDArray[np.float64]
    draws: NDArray[np.float64]
    qlr_draws: _QLRDraws
    output_dir: Path


@dataclass(frozen=True)
class _QLRDraws:
    """Pooled null-bootstrap draws retained for the two nested tests."""

    news_given_distributional: NDArray[np.float64]
    distributional_given_news: NDArray[np.float64]


@dataclass(frozen=True)
class Paper5TwoFieldConfig:
    """Frozen operator identity, profile resolution, and inference controls."""

    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    n_random: int = 1000
    theta_points: int = 401
    rho_points: int = 1000
    active_weight_tolerance: float = 1e-8
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    barycentre_arm_id: str = "wasserstein_w2_loo"
    geometry_id: str = "wasserstein_w2"
    factor_model: str = "ff5_mom"

    def validate(self) -> None:
        """Reject a runtime configuration that cannot support the design."""
        if self.n_bootstrap < 1 or self.block_length < 1 or self.n_random < 1:
            message = "bootstrap, block length, and random draws must be positive"
            raise ValueError(message)


def _plain(value: object) -> object:
    """Convert NumPy values for deterministic JSON serialization."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _align_primary_factor_panel(
    dates: NDArray[np.datetime64],
    returns: NDArray[np.float64],
    factor_panel: FrenchFactorPanel,
) -> tuple[NDArray[np.datetime64], NDArray[np.float64], NDArray[np.float64]]:
    """Align one return panel with the primary factors and subtract daily RF."""
    frame = factor_panel.frame.set_index("Date")
    factor_columns = tuple(factor_panel.factor_columns)
    aligned = frame.reindex(pd.DatetimeIndex(pd.to_datetime(dates)))
    columns = (*factor_columns, "RF")
    complete = aligned.loc[:, list(columns)].notna().all(axis=1).to_numpy()
    if int(np.sum(complete)) < _MINIMUM_FACTOR_DATES:
        message = "fewer than 30 factor-complete dates remain after Paper 5 alignment"
        raise ValueError(message)
    values = aligned.loc[complete, list(columns)].to_numpy(dtype=np.float64)
    aligned_returns = np.asarray(returns[complete], dtype=np.float64)
    return (
        np.asarray(dates[complete], dtype="datetime64[ns]"),
        aligned_returns - values[:, -1, None],
        np.asarray(values[:, :-1], dtype=np.float64),
    )


def _factor_joint_fit_record(fit: FactorTwoFieldFit) -> dict[str, object]:
    """Serialize a factor-conditioned joint fit for JSON and publication values."""
    rho_b = float(fit.rho_w1)
    rho_n = float(fit.rho_w2)
    rho_total = float(fit.rho_total)
    return {
        "rho_b": rho_b,
        "rho_n": rho_n,
        "rho_total": rho_total,
        "theta": float(fit.theta_w1),
        "lambda_b": rho_b / (1.0 - rho_total),
        "lambda_n": rho_n / (1.0 - rho_total),
        "sigma2_hat": float(fit.sigma2_hat),
        "loglik": float(fit.loglik),
        "loglik_per_firm_day": float(fit.loglik / (fit.n_obs * fit.n_assets)),
        "converged": bool(fit.converged),
        "trading_days": int(fit.n_obs),
        "n_assets": int(fit.n_assets),
        "n_factors": int(fit.n_factors),
        "at_w1_boundary": bool(fit.at_w1_boundary),
        "at_w2_boundary": bool(fit.at_w2_boundary),
    }


def _refit_score_contrast(
    draws: NDArray[np.float64], *, estimate: float
) -> ScoreContrast:
    """Summarize one paired held-out score contrast from per-draw refits."""
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        message = "held-out refit contrast has no finite bootstrap draws"
        raise RuntimeError(message)
    return {
        "estimate": estimate,
        "bootstrap_mean": float(np.mean(finite)),
        "ci_lower": float(np.quantile(finite, _LOWER_QUANTILE)),
        "ci_upper": float(np.quantile(finite, _UPPER_QUANTILE)),
        "positive_share": float(np.mean(finite > 0.0)),
        "bootstrap_valid_draws": int(finite.size),
        "unit": "held-out Gaussian quasi-log score per firm-day",
    }


@dataclass(frozen=True, slots=True)
class _HeldOutPanels:
    """The calibration and evaluation inputs a held-out refit draw consumes."""

    calibration_excess: NDArray[np.float64]
    calibration_factors: NDArray[np.float64]
    evaluation_excess: NDArray[np.float64]
    evaluation_factors: NDArray[np.float64]
    w_flat: NDArray[np.float64]
    w_news: NDArray[np.float64]


def _held_out_refit_gains(
    problem: FactorTwoFieldProblem,
    boundary_problems: dict[str, FactorTwoFieldProblem],
    panels: _HeldOutPanels,
    schedule: PairedRefitSchedule,
) -> dict[str, NDArray[np.float64]]:
    """Refit the joint model and both boundaries on every calibration draw.

    The joint model is compared with the factor-only restriction and with each
    single-field boundary under the same resampling architecture as the primary
    spatial-value table, so the two channels' incremental evidence carries
    calibration uncertainty rather than evaluation-date uncertainty alone.
    """
    n_assets = panels.evaluation_excess.shape[1]
    eigvals_flat = np.asarray(np.linalg.eigvals(panels.w_flat), dtype=np.complex128)
    names = ("joint_minus_factor_only", "joint_minus_barycentric", "joint_minus_news")
    gains = {name: np.empty(schedule.draws, dtype=np.float64) for name in names}
    for draw_id in range(schedule.draws):
        calibration_index = schedule.calibration[draw_id]
        score_index = schedule.evaluation_score[draw_id]
        evaluation_excess = panels.evaluation_excess[score_index]
        evaluation_factors = panels.evaluation_factors[score_index]
        draw_fits = {
            "joint": factor_two_field_qmle(
                factor_two_field_problem_at(problem, calibration_index)
            ),
            "w_flat": factor_two_field_qmle(
                factor_two_field_problem_at(
                    boundary_problems["w_flat"], calibration_index
                )
            ),
            "w_co_mentions": factor_two_field_qmle(
                factor_two_field_problem_at(
                    boundary_problems["w_co_mentions"], calibration_index
                )
            ),
        }
        draw_factor_fit = est.sar_factor_fit_at_rho(
            panels.calibration_excess[calibration_index],
            panels.w_flat,
            panels.calibration_factors[calibration_index],
            0.0,
            eigvals_w=eigvals_flat,
        )
        scores = {
            model: float(
                np.mean(
                    factor_two_field_daily_logscore(
                        fit,
                        evaluation_excess,
                        panels.w_flat,
                        panels.w_news,
                        evaluation_factors,
                    )
                )
                / n_assets
            )
            for model, fit in draw_fits.items()
        }
        scores["factor_only"] = float(
            np.mean(
                np.asarray(
                    est.sar_factor_daily_logscore(
                        draw_factor_fit,
                        evaluation_excess,
                        panels.w_flat,
                        evaluation_factors,
                        eigvals_w=eigvals_flat,
                    ),
                    dtype=np.float64,
                )
            )
            / n_assets
        )
        for name, reference in (
            ("joint_minus_factor_only", "factor_only"),
            ("joint_minus_barycentric", "w_flat"),
            ("joint_minus_news", "w_co_mentions"),
        ):
            gains[name][draw_id] = scores["joint"] - scores[reference]
        if (draw_id + 1) % 100 == 0 or draw_id + 1 == schedule.draws:
            logger.info(
                "Paper 5 two-field held-out refit bootstrap %d/%d",
                draw_id + 1,
                schedule.draws,
            )
    return gains


def _fit_factor_conditioned_held_out(
    w_flat: NDArray[np.float64],
    w_news: NDArray[np.float64],
    tickers: list[str],
    paths: Paper5TwoFieldPaths,
    config: Paper5TwoFieldConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit the primary joint model on calibration data and score held-out dates."""
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    if config.factor_model != factor_config.primary_model:
        message = "held-out two-field model must use the primary factor model"
        raise ValueError(message)
    factor_panel = load_factor_model(factor_config, config.factor_model)
    calibration_dates, calibration_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        _CALIBRATION_START,
        _CALIBRATION_END,
        oos_returns_dir=paths.oos_returns_dir,
    )
    evaluation_dates, evaluation_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        POST_CORPUS_START_DATE,
        POST_CORPUS_END_DATE,
        oos_returns_dir=paths.oos_returns_dir,
    )
    calibration_dates, calibration_excess, calibration_factors = (
        _align_primary_factor_panel(
            calibration_dates, calibration_returns, factor_panel
        )
    )
    evaluation_dates, evaluation_excess, evaluation_factors = (
        _align_primary_factor_panel(evaluation_dates, evaluation_returns, factor_panel)
    )
    grid = TwoFieldGrid.build(
        theta_points=config.theta_points,
        rho_points=config.rho_points,
    )
    problem = build_factor_two_field_problem(
        calibration_excess,
        w_flat,
        w_news,
        calibration_factors,
        grid,
    )
    boundary_problems = {
        "w_flat": factor_two_field_with_grid(
            problem,
            TwoFieldGrid.build(
                theta_points=1,
                rho_points=config.rho_points,
                theta_lower=1.0,
                theta_upper=1.0,
            ),
        ),
        "w_co_mentions": factor_two_field_with_grid(
            problem,
            TwoFieldGrid.build(
                theta_points=1,
                rho_points=config.rho_points,
                theta_lower=0.0,
                theta_upper=0.0,
            ),
        ),
    }
    fits = {
        "joint": factor_two_field_qmle(problem),
        "w_flat": factor_two_field_qmle(boundary_problems["w_flat"]),
        "w_co_mentions": factor_two_field_qmle(boundary_problems["w_co_mentions"]),
    }
    factor_fit = est.sar_factor_fit_at_rho(
        calibration_excess,
        w_flat,
        calibration_factors,
        0.0,
        eigvals_w=np.asarray(np.linalg.eigvals(w_flat), dtype=np.complex128),
    )
    if not factor_fit.converged or not all(fit.converged for fit in fits.values()):
        message = "held-out factor-conditioned two-field fit did not converge"
        raise RuntimeError(message)
    n_assets = len(tickers)
    daily_scores = {
        "factor_only": np.asarray(
            est.sar_factor_daily_logscore(
                factor_fit,
                evaluation_excess,
                w_flat,
                evaluation_factors,
                eigvals_w=np.asarray(np.linalg.eigvals(w_flat), dtype=np.complex128),
            ),
            dtype=np.float64,
        )
        / n_assets,
        "w_flat": factor_two_field_daily_logscore(
            fits["w_flat"], evaluation_excess, w_flat, w_news, evaluation_factors
        )
        / n_assets,
        "w_co_mentions": factor_two_field_daily_logscore(
            fits["w_co_mentions"],
            evaluation_excess,
            w_flat,
            w_news,
            evaluation_factors,
        )
        / n_assets,
        "joint": factor_two_field_daily_logscore(
            fits["joint"], evaluation_excess, w_flat, w_news, evaluation_factors
        )
        / n_assets,
    }
    schedule = paired_refit_schedule(
        calibration_observations=len(calibration_dates),
        evaluation_observations=len(evaluation_dates),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed,
    )
    draw_gains = _held_out_refit_gains(
        problem,
        boundary_problems,
        _HeldOutPanels(
            calibration_excess=calibration_excess,
            calibration_factors=calibration_factors,
            evaluation_excess=evaluation_excess,
            evaluation_factors=evaluation_factors,
            w_flat=w_flat,
            w_news=w_news,
        ),
        schedule,
    )
    contrasts = {
        name: _refit_score_contrast(
            draw_gains[name],
            estimate=float(
                np.mean(daily_scores["joint"]) - np.mean(daily_scores[reference])
            ),
        )
        for name, reference in (
            ("joint_minus_factor_only", "factor_only"),
            ("joint_minus_barycentric", "w_flat"),
            ("joint_minus_news", "w_co_mentions"),
        )
    }
    result: dict[str, object] = {
        "factor_model": str(config.factor_model),
        "calibration": {
            "start": str(pd.Timestamp(calibration_dates.min()).date()),
            "end": str(pd.Timestamp(calibration_dates.max()).date()),
            "trading_days": len(calibration_dates),
        },
        "evaluation": {
            "start": str(pd.Timestamp(evaluation_dates.min()).date()),
            "end": str(pd.Timestamp(evaluation_dates.max()).date()),
            "trading_days": len(evaluation_dates),
        },
        "models": {
            model: {"held_out_score": float(np.mean(scores))}
            for model, scores in daily_scores.items()
        },
        "contrasts": contrasts,
        "bootstrap": {
            **paired_refit_provenance(schedule),
            "unit": "held-out Gaussian quasi-log score per firm-day",
            "refit_scope": (
                "the joint model, both single-field boundaries, and the "
                "factor-only restriction are refitted on each calibration draw"
            ),
        },
    }
    bootstrap_rows = pd.DataFrame(
        {
            "draw_id": np.arange(config.n_bootstrap, dtype=np.int64),
            **draw_gains,
        }
    )
    return result, bootstrap_rows


def _factor_joint_bootstrap_summary(draws: pd.DataFrame) -> dict[str, object]:
    """Summarize factor-conditioned joint coefficient bootstrap draws."""
    rho_b = draws["rho_w1"]
    rho_n = draws["rho_w2"]
    rho_total = draws["rho_total"]
    theta = draws["theta_w1"]
    correlation = float(rho_b.corr(rho_n))
    summary: dict[str, object] = {
        "rho_b_ci_lower": float(rho_b.quantile(_LOWER_QUANTILE)),
        "rho_b_ci_upper": float(rho_b.quantile(_UPPER_QUANTILE)),
        "rho_n_ci_lower": float(rho_n.quantile(_LOWER_QUANTILE)),
        "rho_n_ci_upper": float(rho_n.quantile(_UPPER_QUANTILE)),
        "rho_total_ci_lower": float(rho_total.quantile(_LOWER_QUANTILE)),
        "rho_total_ci_upper": float(rho_total.quantile(_UPPER_QUANTILE)),
        "theta_ci_lower": float(theta.quantile(_LOWER_QUANTILE)),
        "theta_ci_upper": float(theta.quantile(_UPPER_QUANTILE)),
        "rho_channel_correlation": correlation,
        "interval_status": "descriptive_boundary_bootstrap_quantiles",
        "field_b_boundary_share": float(draws["at_w1_boundary"].mean()),
        "field_n_boundary_share": float(draws["at_w2_boundary"].mean()),
    }
    for channel, values in (("b", rho_b), ("n", rho_n)):
        intensity = values / (1.0 - rho_total)
        summary[f"lambda_{channel}_ci_lower"] = float(
            intensity.quantile(_LOWER_QUANTILE)
        )
        summary[f"lambda_{channel}_ci_upper"] = float(
            intensity.quantile(_UPPER_QUANTILE)
        )
    return summary


def _factor_qlr_record(
    test: NestedQLRBootstrap, *, null: str, tested_channel: str
) -> dict[str, object]:
    """Serialize one factor-conditioned restricted-null QLR result."""
    p_value = float(test.p_value)
    return {
        "null": null,
        "tested_channel": tested_channel,
        "statistic": float(test.statistic),
        "bootstrap_p_value": p_value,
        "bootstrap_exceedances": int(test.exceedances),
        "p_value_floor": float(1.0 / (test.draws.size + 1)),
        "inference_status": "working_null_calibration",
        "decision": "reject" if p_value < _QLR_ALPHA else "do_not_reject",
    }


def _fit_factor_conditioned_joint(
    dates: NDArray[np.datetime64],
    returns: NDArray[np.float64],
    w_flat: NDArray[np.float64],
    w_news: NDArray[np.float64],
    paths: Paper5TwoFieldPaths,
    config: Paper5TwoFieldConfig,
) -> dict[str, object]:
    """Fit the primary FF5-plus-momentum barycentric/news joint model."""
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    if config.factor_model != factor_config.primary_model:
        message = (
            "Paper 5 factor-conditioned joint model must use the primary "
            f"factor model {factor_config.primary_model!r}"
        )
        raise ValueError(message)
    factor_panel = load_factor_model(factor_config, config.factor_model)
    factor_dates, excess_returns, factors = _align_primary_factor_panel(
        dates, returns, factor_panel
    )
    grid = TwoFieldGrid.build(
        theta_points=config.theta_points,
        rho_points=config.rho_points,
    )
    problem = build_factor_two_field_problem(
        excess_returns, w_flat, w_news, factors, grid
    )
    boundary_problems = {
        "w_flat": factor_two_field_with_grid(
            problem,
            TwoFieldGrid.build(
                theta_points=1,
                rho_points=config.rho_points,
                theta_lower=1.0,
                theta_upper=1.0,
            ),
        ),
        "w_co_mentions": factor_two_field_with_grid(
            problem,
            TwoFieldGrid.build(
                theta_points=1,
                rho_points=config.rho_points,
                theta_lower=0.0,
                theta_upper=0.0,
            ),
        ),
    }
    joint_fit = factor_two_field_qmle(problem)
    boundary_fits = {
        name: factor_two_field_qmle(boundary)
        for name, boundary in boundary_problems.items()
    }
    if not joint_fit.converged or not all(
        fit.converged for fit in boundary_fits.values()
    ):
        message = "factor-conditioned Paper 5 joint fit did not converge"
        raise RuntimeError(message)

    bootstrap_indices = stationary_bootstrap_indices(
        len(factor_dates),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed + 1_000,
    )
    bootstrap_values = factor_two_field_bootstrap(problem, bootstrap_indices)
    bootstrap = pd.DataFrame(
        bootstrap_values,
        columns=list(FACTOR_TWO_FIELD_DRAW_COLUMNS),
    )
    if not bool(bootstrap["converged"].all()):
        message = (
            "one or more factor-conditioned Paper 5 bootstrap fits did not converge"
        )
        raise RuntimeError(message)

    null_indices = independent_stationary_bootstrap_indices(
        len(factor_dates),
        excess_returns.shape[1],
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed + 11_000,
    )
    news_test = factor_nested_qlr_bootstrap(
        problem, boundary_problems["w_flat"], null_indices
    )
    barycentric_test = factor_nested_qlr_bootstrap(
        problem, boundary_problems["w_co_mentions"], null_indices
    )
    qlr_draws = pd.DataFrame(
        {
            "news_given_barycentric": news_test.draws,
            "barycentric_given_news": barycentric_test.draws,
        }
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap.to_parquet(paths.output_dir / "factor_bootstrap.parquet", index=False)
    qlr_draws.to_parquet(
        paths.output_dir / "factor_qlr_null_draws.parquet", index=False
    )
    result: dict[str, object] = {
        "factor_model": str(config.factor_model),
        "factor_columns": list(factor_panel.factor_columns),
        "dates": {
            "start": str(pd.Timestamp(factor_dates.min()).date()),
            "end": str(pd.Timestamp(factor_dates.max()).date()),
            "trading_days": len(factor_dates),
            "dropped_return_dates": int(len(dates) - len(factor_dates)),
        },
        "joint": _factor_joint_fit_record(joint_fit),
        "boundaries": {
            name: {
                "rho": float(fit.rho_total),
                "loglik": float(fit.loglik),
                "matches_single_field": True,
            }
            for name, fit in boundary_fits.items()
        },
        "loglik_gain_vs_w_flat": float(
            joint_fit.loglik - boundary_fits["w_flat"].loglik
        ),
        "loglik_gain_vs_co_mentions": float(
            joint_fit.loglik - boundary_fits["w_co_mentions"].loglik
        ),
        "bootstrap": _factor_joint_bootstrap_summary(bootstrap),
        "quasi_lr_tests": {
            "news_given_barycentric": _factor_qlr_record(
                news_test,
                null="rho_n = 0",
                tested_channel="persistent news co-mention field",
            ),
            "barycentric_given_news": _factor_qlr_record(
                barycentric_test,
                null="rho_b = 0",
                tested_channel="W2 barycentric field",
            ),
        },
        "inference": {
            "method": "joint-date stationary bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "point_fit_window": "2023-01-01/2026-04-06",
            "fields_frozen_through": "2022-12-31",
            "interval_status": "descriptive_boundary_bootstrap_quantiles",
        },
    }
    return result


def _nanmean(values: NDArray[np.float64]) -> float:
    """Return a mean that tolerates the undefined rows a similarity may leave."""
    if not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


def _upper_tail_pvalue(observed: float, null: NDArray[np.float64]) -> float:
    """Return the randomization probability of matching the observed agreement."""
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def _overlap_records(
    returns: NDArray[np.float64],
    w_flat: NDArray[np.float64],
    w_news: NDArray[np.float64],
    config: Paper5TwoFieldConfig,
) -> dict[str, object]:
    """Compare peer selection, cardinal weights, and the induced spatial field."""
    tolerance = config.active_weight_tolerance
    overlap = support_overlap(w_flat, w_news, tolerance=tolerance)
    similarity = weight_similarity(w_flat, w_news, tolerance=tolerance)
    induced = induced_field_similarity(returns, w_flat, w_news)
    null = matched_random_support_null(
        w_flat,
        w_news,
        draws=config.n_random,
        seed=config.seed,
        tolerance=tolerance,
    )
    induced_null = matched_random_induced_null(
        returns,
        w_flat,
        w_news,
        draws=config.n_random,
        seed=config.seed,
        tolerance=tolerance,
    )
    observed_share = float(np.mean(overlap.directed_share))
    return {
        "support": {
            "jaccard_mean": float(np.mean(overlap.jaccard)),
            "jaccard_median": float(np.median(overlap.jaccard)),
            "directed_share_mean": observed_share,
            "rank_matched_share_mean": float(np.mean(overlap.rank_matched_share)),
            "focal_support_size_mean": float(np.mean(overlap.focal_support_sizes)),
            "comparator_support_size_mean": float(
                np.mean(overlap.comparator_support_sizes)
            ),
            "random_null_directed_share_mean": float(np.mean(null)),
            "random_null_p_value": _upper_tail_pvalue(observed_share, null),
            "n_random": config.n_random,
        },
        "weights": {
            "union_pearson_mean": _nanmean(similarity.union_pearson),
            "union_cosine_mean": _nanmean(similarity.union_cosine),
            "common_edge_spearman_mean": _nanmean(similarity.common_edge_spearman),
            "common_edge_count_mean": float(np.mean(similarity.common_edge_counts)),
            "offdiagonal_pearson": similarity.offdiagonal_pearson,
        },
        "induced_field": {
            "pooled_pearson": induced.pooled_pearson,
            "pooled_r_squared": induced.pooled_r_squared,
            "per_asset_pearson_mean": _nanmean(induced.per_asset_pearson),
            "per_asset_pearson_min": float(np.nanmin(induced.per_asset_pearson)),
            "per_asset_pearson_max": float(np.nanmax(induced.per_asset_pearson)),
            "random_null_pooled_pearson_mean": _nanmean(induced_null),
            "random_null_p_value": _upper_tail_pvalue(
                induced.pooled_pearson, induced_null
            ),
        },
    }


def _with_intensities(fit: dict[str, float]) -> dict[str, float]:
    """Attach the model-implied adjustment index of each priced channel."""
    total = fit["rho_b"] + fit["rho_n"]
    return {
        **fit,
        "rho_total_fitted": total,
        "lambda_b": channel_intensity(fit["rho_b"], total),
        "lambda_n": channel_intensity(fit["rho_n"], total),
    }


def _bootstrap_summary(draws: NDArray[np.float64]) -> dict[str, object]:
    """Summarize refits as descriptive intervals and a joint shape."""
    rho_b, rho_n, theta = draws[:, 0], draws[:, 1], draws[:, 2]
    total = rho_b + rho_n
    summary: dict[str, object] = {
        "rho_b_ci_lower": float(np.quantile(rho_b, _LOWER_QUANTILE)),
        "rho_b_ci_upper": float(np.quantile(rho_b, _UPPER_QUANTILE)),
        "rho_n_ci_lower": float(np.quantile(rho_n, _LOWER_QUANTILE)),
        "rho_n_ci_upper": float(np.quantile(rho_n, _UPPER_QUANTILE)),
        "rho_total_ci_lower": float(np.quantile(total, _LOWER_QUANTILE)),
        "rho_total_ci_upper": float(np.quantile(total, _UPPER_QUANTILE)),
        "theta_ci_lower": float(np.quantile(theta, _LOWER_QUANTILE)),
        "theta_ci_upper": float(np.quantile(theta, _UPPER_QUANTILE)),
        "rho_channel_correlation": _channel_correlation(rho_b, rho_n),
        "field_b_boundary_share": float(np.mean(draws[:, 3])),
        "field_n_boundary_share": float(np.mean(draws[:, 4])),
        "interval_status": "descriptive_boundary_bootstrap_quantiles",
    }
    for name, quantile in (("lower", _LOWER_QUANTILE), ("upper", _UPPER_QUANTILE)):
        for channel, values in (("b", rho_b), ("n", rho_n)):
            summary[f"lambda_{channel}_ci_{name}"] = float(
                np.quantile(values / (1.0 - total), quantile)
            )
    return summary


def _channel_correlation(
    rho_b: NDArray[np.float64], rho_n: NDArray[np.float64]
) -> float:
    """Return the bootstrap correlation that exposes a flat likelihood ridge."""
    if np.ptp(rho_b) <= 0.0 or np.ptp(rho_n) <= 0.0:
        return float("nan")
    return float(np.corrcoef(rho_b, rho_n)[0, 1])


def _boundary_fit(
    problem: TwoFieldProblem,
    indices: NDArray[np.int64],
) -> dict[str, float]:
    """Fit and bootstrap one pinned single-field boundary of the nested family.

    Only one channel is active at a pinned boundary, so its coefficient is the
    total and ``rho``/``lambda`` report it without reference to which field it
    is. Bootstrapping on the joint fit's own draws keeps the intervals
    comparable rather than carried over from a separate estimator.
    """
    fit = _with_intensities(two_field_qmle(problem))
    draws = two_field_bootstrap(problem, indices)
    coefficients = draws[:, 0] + draws[:, 1]
    intensities = coefficients / (1.0 - coefficients)
    fit["rho"] = fit["rho_total_fitted"]
    fit["lambda"] = fit["lambda_b"] + fit["lambda_n"]
    fit["rho_ci_lower"] = float(np.quantile(coefficients, _LOWER_QUANTILE))
    fit["rho_ci_upper"] = float(np.quantile(coefficients, _UPPER_QUANTILE))
    fit["lambda_ci_lower"] = float(np.quantile(intensities, _LOWER_QUANTILE))
    fit["lambda_ci_upper"] = float(np.quantile(intensities, _UPPER_QUANTILE))
    return fit


def _qlr_record(
    test: NestedQLRBootstrap,
    *,
    null: str,
    tested_channel: str,
) -> dict[str, object]:
    """Serialize one calibrated nested comparison without its retained draws."""
    return {
        "null": null,
        "tested_channel": tested_channel,
        "statistic": test.statistic,
        "bootstrap_p_value": test.p_value,
        "bootstrap_exceedances": test.exceedances,
        "p_value_floor": 1.0 / (test.draws.size + 1),
        "inference_status": "working_null_calibration",
    }


def _fit_period(
    returns: NDArray[np.float64],
    w_flat: NDArray[np.float64],
    w_news: NDArray[np.float64],
    config: Paper5TwoFieldConfig,
    indices: NDArray[np.int64],
    *,
    run_qlr_tests: bool = False,
) -> tuple[dict[str, object], NDArray[np.float64], _QLRDraws | None]:
    """Fit the joint model and both nested boundaries on one evaluation period."""
    joint_grid = TwoFieldGrid.build(
        theta_points=config.theta_points,
        rho_points=config.rho_points,
    )
    joint_problem = build_two_field_problem(returns, w_flat, w_news, joint_grid)
    joint = _with_intensities(two_field_qmle(joint_problem))

    # Each boundary is bootstrapped on the same draws as the joint fit, so its
    # interval is comparable rather than carried over from a separate estimator.
    boundaries: dict[str, dict[str, float]] = {}
    boundary_problems: dict[str, TwoFieldProblem] = {}
    for name, theta in (("w_flat", 1.0), ("w_co_mentions", 0.0)):
        grid = TwoFieldGrid.build(
            theta_points=1,
            rho_points=config.rho_points,
            theta_lower=theta,
            theta_upper=theta,
        )
        problem = build_two_field_problem(returns, w_flat, w_news, grid)
        boundary_problems[name] = problem
        boundaries[name] = _boundary_fit(problem, indices)

    draws = two_field_bootstrap(joint_problem, indices)
    result: dict[str, object] = {
        "joint": joint,
        "boundaries": boundaries,
        "loglik_gain_vs_w_flat": joint["loglik"] - boundaries["w_flat"]["loglik"],
        "loglik_gain_vs_co_mentions": (
            joint["loglik"] - boundaries["w_co_mentions"]["loglik"]
        ),
        "bootstrap": _bootstrap_summary(draws),
    }
    qlr_draws: _QLRDraws | None = None
    if run_qlr_tests:
        null_indices = independent_stationary_bootstrap_indices(
            returns.shape[0],
            returns.shape[1],
            draws=config.n_bootstrap,
            expected_block_length=config.block_length,
            seed=config.seed + 10_000,
        )
        news_test = nested_qlr_bootstrap(
            returns,
            joint_problem,
            boundary_problems["w_flat"],
            null_indices,
        )
        distributional_test = nested_qlr_bootstrap(
            returns,
            joint_problem,
            boundary_problems["w_co_mentions"],
            null_indices,
        )
        result["quasi_lr_tests"] = {
            "news_given_distributional": _qlr_record(
                news_test,
                null="rho_n = 0",
                tested_channel="news",
            ),
            "distributional_given_news": _qlr_record(
                distributional_test,
                null="rho_b = 0",
                tested_channel="distributional",
            ),
        }
        qlr_draws = _QLRDraws(
            news_given_distributional=news_test.draws,
            distributional_given_news=distributional_test.draws,
        )
    return result, draws, qlr_draws


def _write_exhibit(exhibit: _ExhibitInputs) -> None:
    """Save the joint-region exhibit's profile surface and bootstrap cloud."""
    surface_grid = TwoFieldGrid.build(
        theta_points=_SURFACE_THETA_POINTS,
        rho_points=_SURFACE_RHO_POINTS,
    )
    problem = build_two_field_problem(
        exhibit.returns, exhibit.w_flat, exhibit.w_news, surface_grid
    )
    rho_b, rho_n, loglik = two_field_profile_surface(problem)
    exhibit.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rho_b": rho_b, "rho_n": rho_n, "loglik": loglik}).to_parquet(
        exhibit.output_dir / "surface.parquet", index=False
    )
    draws = exhibit.draws
    bootstrap_columns = {
        "rho_b": draws[:, 0],
        "rho_n": draws[:, 1],
        "theta": draws[:, 2],
        "qlr_news_given_distributional": (exhibit.qlr_draws.news_given_distributional),
        "qlr_distributional_given_news": (exhibit.qlr_draws.distributional_given_news),
    }
    pd.DataFrame(bootstrap_columns).to_parquet(
        exhibit.output_dir / "bootstrap.parquet", index=False
    )


def _period_row(period: str, result: dict[str, object]) -> dict[str, object]:
    """Flatten one period's joint fit into a tabular record."""
    joint = result["joint"]
    bootstrap = result["bootstrap"]
    if not isinstance(joint, dict) or not isinstance(bootstrap, dict):
        message = "period results must carry a joint fit and a bootstrap summary"
        raise TypeError(message)
    row: dict[str, object] = {
        "period": period,
        **{f"joint_{key}": value for key, value in joint.items()},
        **{f"bootstrap_{key}": value for key, value in bootstrap.items()},
        "loglik_gain_vs_w_flat": result["loglik_gain_vs_w_flat"],
        "loglik_gain_vs_co_mentions": result["loglik_gain_vs_co_mentions"],
    }
    tests = result.get("quasi_lr_tests")
    if isinstance(tests, dict):
        for name, test in tests.items():
            if not isinstance(test, dict):
                message = "nested QLR records must be mappings"
                raise TypeError(message)
            for field in ("statistic", "bootstrap_p_value", "bootstrap_exceedances"):
                row[f"qlr_{name}_{field}"] = test[field]
    return row


def run_paper5_two_field_qmle(
    paths: Paper5TwoFieldPaths,
    config: Paper5TwoFieldConfig,
) -> dict[str, object]:
    """Estimate the joint two-field model and the field-overlap diagnostics.

    Args:
        paths: Input and output boundaries.
        config: Frozen operator identity, profile resolution, and inference.

    Returns:
        The serialized payload written to ``results.json``.

    """
    config.validate()
    tickers, w_flat, _summary = load_operator(paths.shared_barycentre_dir, config)
    w_news, _diagnostics = load_co_mentions_w(paths.co_mentions_adjacency, tickers)

    periods = {
        _POOLED_PERIOD: (POST_CORPUS_START_DATE, POST_CORPUS_END_DATE),
        **ANNUAL_FOLDS,
    }
    results: dict[str, object] = {}
    overlap: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for period_index, (period, (start, end)) in enumerate(periods.items()):
        dates, returns = load_aligned_return_panel(
            paths.returns_dir,
            tickers,
            start,
            end,
            oos_returns_dir=paths.oos_returns_dir,
        )
        if period == _POOLED_PERIOD:
            overlap = _overlap_records(returns, w_flat, w_news, config)
        indices = stationary_bootstrap_indices(
            len(returns),
            draws=config.n_bootstrap,
            expected_block_length=config.block_length,
            seed=config.seed + period_index,
        )
        result, draws, qlr_draws = _fit_period(
            returns,
            w_flat,
            w_news,
            config,
            indices,
            run_qlr_tests=period == _POOLED_PERIOD,
        )
        if period == _POOLED_PERIOD:
            if qlr_draws is None:
                message = "the pooled period must retain both nested QLR draws"
                raise RuntimeError(message)
            _write_exhibit(
                _ExhibitInputs(
                    returns=returns,
                    w_flat=w_flat,
                    w_news=w_news,
                    draws=draws,
                    qlr_draws=qlr_draws,
                    output_dir=paths.output_dir,
                ),
            )
        result["start"] = str(pd.Timestamp(dates.min()).date())
        result["end"] = str(pd.Timestamp(dates.max()).date())
        result["trading_days"] = len(returns)
        results[period] = result
        rows.append(_period_row(period, result))

    pooled_dates, pooled_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        POST_CORPUS_START_DATE,
        POST_CORPUS_END_DATE,
        oos_returns_dir=paths.oos_returns_dir,
    )
    factor_conditioned = _fit_factor_conditioned_joint(
        pooled_dates,
        pooled_returns,
        w_flat,
        w_news,
        paths,
        config,
    )
    factor_conditioned_held_out, held_out_bootstrap = _fit_factor_conditioned_held_out(
        w_flat,
        w_news,
        tickers,
        paths,
        config,
    )
    factor_conditioned["held_out"] = factor_conditioned_held_out

    payload: dict[str, object] = {
        "schema_version": 4,
        "primary_period": _POOLED_PERIOD,
        "annual_periods": list(ANNUAL_FOLDS),
        "operator_construction_window": _OPERATOR_WINDOW,
        "profile": {
            "theta_points": config.theta_points,
            "rho_points": config.rho_points,
            "theta_range": [0.0, 1.0],
            "refinement": "bounded Brent inside the grid cell of the maximum",
        },
        "inference": {
            "method": "joint-date stationary bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "seed": config.seed,
            "bootstrap_refinement": (
                "bounded Brent inside the grid cell of the maximum"
            ),
            "nested_qlr": {
                "period": _POOLED_PERIOD,
                "method": _QLR_METHOD,
                "p_value_rule": "(1 + exceedances) / (n_bootstrap + 1)",
                "null_contract": (
                    "asset-wise standardized residual paths with fitted variance "
                    "restoration"
                ),
            },
        },
        "overlap": overlap,
        "results": results,
        "factor_conditioned": factor_conditioned,
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "results.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(paths.output_dir / "results.parquet", index=False)
    held_out_bootstrap.to_parquet(
        paths.output_dir / "factor_held_out_bootstrap.parquet", index=False
    )
    return payload


__all__ = [
    "Paper5TwoFieldConfig",
    "Paper5TwoFieldPaths",
    "run_paper5_two_field_qmle",
]

"""Primary factor-conditioned point-in-time encoder sensitivity for Paper 5."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from pipeline.io.fama_french import load_factor_data_config, load_factor_model
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5.bootstrap import stationary_bootstrap_indices
from pipeline.stages.papers.paper5.field_operators import (
    median_bandwidth,
    rbf_weights,
    validate_operator,
)
from pipeline.stages.papers.paper5.operators import load_operator
from pipeline.stages.papers.paper5.transport_order import _align_factor_panel
from pipeline.stages.substrate.panel import (
    load_aligned_return_panel,
    load_priced_distance_universe,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.io.fama_french import FrenchFactorPanel

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


class Paper5PointInTimeError(ValueError):
    """Raised when the point-in-time sensitivity contract is invalid."""


def _invalid(message: str) -> NoReturn:
    """Raise the point-in-time stage's contract error."""
    raise Paper5PointInTimeError(message)


@dataclass(frozen=True, slots=True)
class Paper5PointInTimePaths:
    """Input and output boundaries for the point-in-time sensitivity."""

    returns_dir: Path
    oos_returns_dir: Path
    w2_barycentre_dir: Path
    distance_artifact_dir: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5PointInTimeConfig:
    """Frozen factor, timing, representation, and bootstrap contract."""

    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    confidence_level: float = 0.95
    provider_id: str = "chronobert-2017"
    representation_id: str = "chronobert-2017-unit"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w2_geometry_id: str = "wasserstein_w2"
    factor_model: str = "ff5_mom"

    def validate(self) -> None:
        """Reject invalid schedules before reading production artifacts."""
        if self.n_bootstrap < 1 or self.block_length < 1:
            _invalid("bootstrap draws and block length must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            _invalid("confidence level must lie strictly between zero and one")
        if self.factor_model != "ff5_mom":
            _invalid("point-in-time sensitivity must use the primary factor model")
        if (
            self.provider_id != "chronobert-2017"
            or self.representation_id != "chronobert-2017-unit"
        ):
            _invalid("point-in-time sensitivity requires the ChronoBERT 2017 arm")


@dataclass(frozen=True, slots=True)
class _OperatorIdentity:
    provider_id: str
    representation_id: str
    barycentre_arm_id: str
    geometry_id: str


@dataclass(frozen=True, slots=True)
class _AlignedPanel:
    dates: NDArray[np.datetime64]
    excess_returns: FloatMatrix
    factors: FloatMatrix


def _encoder_training_cutoff(
    params_file: Path, config: Paper5PointInTimeConfig
) -> tuple[date, date, date]:
    """Validate the pinned encoder's information cutoff before any fitting."""
    with params_file.open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    try:
        model = params["chronobert"]["models"][config.provider_id]
        provider = params["typed_analysis"]["providers"][config.provider_id]
        model_id, revision = model["id"], model["revision"]
        cutoff, article_start, calibration_start = (
            date.fromisoformat(model["training_cutoff"]),
            date.fromisoformat(params["start_date"]),
            date.fromisoformat(config.calibration_start),
        )
    except (KeyError, TypeError, ValueError) as exc:
        message = (
            "point-in-time requires pinned ChronoBERT config and "
            "parseable cutoff/window dates"
        )
        raise Paper5PointInTimeError(message) from exc
    if not isinstance(provider, dict):
        _invalid("typed provider must be a mapping")
    if (
        model_id != "manelalab/chrono-bert-v1-20171231"
        or provider.get("model_id") != model_id
    ):
        _invalid("point-in-time checkpoint model ID must agree with the typed provider")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        _invalid("point-in-time requires an immutable checkpoint revision")
    if cutoff >= article_start or cutoff >= calibration_start:
        _invalid(
            "encoder training cutoff must precede article-window and calibration starts"
        )
    if cutoff != date(2017, 12, 31):
        _invalid("point-in-time ChronoBERT 2017 cutoff must be 2017-12-31")
    return cutoff, article_start, calibration_start


def _eigenvalues(matrix: FloatMatrix) -> NDArray[np.complex128]:
    """Return complex eigenvalues for the SAR log-determinant evaluation."""
    return np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128)


def _align_panel(
    dates: NDArray[np.datetime64],
    returns: FloatMatrix,
    factor_panel: FrenchFactorPanel,
) -> _AlignedPanel:
    """Align returns and factors using the primary model's complete dates."""
    aligned = _align_factor_panel(dates, returns, factor_panel)
    return _AlignedPanel(
        dates=aligned.dates,
        excess_returns=aligned.excess_returns,
        factors=aligned.factors,
    )


def _load_w2_operator(
    paths: Paper5PointInTimePaths,
    config: Paper5PointInTimeConfig,
) -> tuple[list[str], FloatMatrix, FloatMatrix]:
    """Load and align the ChronoBERT W2 field and its pairwise distances."""
    tickers, matrix, _summary = load_operator(
        paths.w2_barycentre_dir,
        _OperatorIdentity(
            config.provider_id,
            config.representation_id,
            config.w2_barycentre_arm_id,
            config.w2_geometry_id,
        ),
    )
    distance_tickers, _distance, squared = load_priced_distance_universe(
        paths.distance_artifact_dir, paths.returns_dir
    )
    if set(distance_tickers) != set(tickers):
        _invalid("point-in-time distance and W2 universes do not match")
    distance_index = {ticker: index for index, ticker in enumerate(distance_tickers)}
    order = [distance_index[ticker] for ticker in tickers]
    return (
        tickers,
        validate_operator(np.asarray(matrix, dtype=np.float64), name="W2 weights"),
        np.asarray(squared[np.ix_(order, order)], dtype=np.float64),
    )


def _fit(panel: _AlignedPanel, matrix: FloatMatrix) -> est.FactorSarFit:
    """Fit one factor-conditioned spatial model without standard errors."""
    fit = est.sar_factor_qmle(
        panel.excess_returns,
        matrix,
        panel.factors,
        compute_se=False,
        eigvals_w=_eigenvalues(matrix),
    )
    if not fit.converged:
        _invalid("point-in-time factor-conditioned fit did not converge")
    return fit


def _factor_only_fit(panel: _AlignedPanel, reference: FloatMatrix) -> est.FactorSarFit:
    """Fit the matched factor-only restriction at zero spatial feedback."""
    fit = est.sar_factor_fit_at_rho(
        panel.excess_returns,
        reference,
        panel.factors,
        0.0,
        eigvals_w=_eigenvalues(reference),
    )
    if not fit.converged:
        _invalid("point-in-time factor-only fit did not converge")
    return fit


def _daily_score(
    fit: est.FactorSarFit,
    panel: _AlignedPanel,
    matrix: FloatMatrix,
) -> FloatVector:
    """Return the held-out Gaussian quasi-log score per firm-day."""
    scores = est.sar_factor_daily_logscore(
        fit,
        panel.excess_returns,
        matrix,
        panel.factors,
        eigvals_w=_eigenvalues(matrix),
    )
    return np.asarray(scores, dtype=np.float64) / panel.excess_returns.shape[1]


def _contrast_summary(
    difference: FloatVector,
    *,
    candidate: str,
    reference: str,
    estimate: float,
    confidence_level: float,
) -> dict[str, object]:
    """Summarize a paired score difference on common evaluation dates."""
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        _invalid("point-in-time paired contrast has no finite draws")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return {
        "candidate": candidate,
        "reference": reference,
        "estimate": float(estimate),
        "bootstrap_mean": float(np.mean(finite)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "positive_share": float(np.mean(finite > 0.0)),
        "bootstrap_valid_draws": int(finite.size),
        "confidence_level": float(confidence_level),
        "unit": "held-out Gaussian quasi-log score per firm-day",
    }


def run_paper5_point_in_time(
    paths: Paper5PointInTimePaths,
    config: Paper5PointInTimeConfig,
) -> dict[str, object]:
    """Run the primary factor-conditioned ChronoBERT 2017 sensitivity."""
    config.validate()
    cutoff, article_start, calibration_start = _encoder_training_cutoff(
        paths.params_file, config
    )
    tickers, w2, squared_distances = _load_w2_operator(paths, config)
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    factor_panel = load_factor_model(factor_config, "ff5_mom")
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
    calibration = _align_panel(
        cast("NDArray[np.datetime64]", calibration_dates),
        cast("FloatMatrix", calibration_returns),
        factor_panel,
    )
    evaluation = _align_panel(
        cast("NDArray[np.datetime64]", evaluation_dates),
        cast("FloatMatrix", evaluation_returns),
        factor_panel,
    )
    if np.max(calibration.dates) >= np.min(evaluation.dates):
        _invalid("point-in-time calibration and evaluation dates overlap")
    rbf = rbf_weights(squared_distances, median_bandwidth(squared_distances))
    matrices = {"w2": w2, "rbf_median": rbf}
    calibration_fits: dict[str, est.FactorSarFit] = {
        "factor_only": _factor_only_fit(calibration, w2),
        **{name: _fit(calibration, matrix) for name, matrix in matrices.items()},
    }
    daily_scores = {
        "factor_only": _daily_score(calibration_fits["factor_only"], evaluation, w2),
        **{
            name: _daily_score(calibration_fits[name], evaluation, matrix)
            for name, matrix in matrices.items()
        },
    }
    indices = stationary_bootstrap_indices(
        len(evaluation.dates),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed,
    )
    factor = daily_scores["factor_only"]
    bootstrap_rows: list[dict[str, object]] = []
    w2_minus_rbf = np.empty(len(indices), dtype=np.float64)
    for draw_id, draw_indices in enumerate(indices):
        w2_gain = float(np.mean((daily_scores["w2"] - factor)[draw_indices]))
        rbf_gain = float(np.mean((daily_scores["rbf_median"] - factor)[draw_indices]))
        w2_minus_rbf[draw_id] = float(
            np.mean((daily_scores["w2"] - daily_scores["rbf_median"])[draw_indices])
        )
        bootstrap_rows.extend(
            [
                {"draw_id": draw_id, "model": "w2", "score_gain": w2_gain},
                {
                    "draw_id": draw_id,
                    "model": "rbf_median",
                    "score_gain": rbf_gain,
                },
            ]
        )
    bootstrap = pd.DataFrame.from_records(bootstrap_rows)
    tail = (1.0 - config.confidence_level) / 2.0
    factor_score = float(np.mean(factor))
    labels = {
        "factor_only": "Factor-only",
        "w2": "ChronoBERT 2017 W2 barycentric",
        "rbf_median": "ChronoBERT 2017 median-bandwidth W2 RBF",
    }
    model_rows: list[dict[str, object]] = []
    for model in ("factor_only", "w2", "rbf_median"):
        score = float(np.mean(daily_scores[model]))
        gain = score - factor_score
        if model == "factor_only":
            lower = upper = 0.0
        else:
            model_draws = bootstrap.loc[
                bootstrap["model"] == model, "score_gain"
            ].to_numpy(dtype=np.float64)
            lower, upper = np.quantile(model_draws, [tail, 1.0 - tail])
        model_rows.append(
            {
                "model": model,
                "label": labels[model],
                "rho_hat": float(calibration_fits[model].rho_hat),
                "held_out_score": score,
                "held_out_score_gain_vs_factor_only": gain,
                "held_out_gain_ci_lower": float(lower),
                "held_out_gain_ci_upper": float(upper),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_id": "paper5-point-in-time-factor-score-v1",
        "identity": {
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "w2_barycentre_arm_id": config.w2_barycentre_arm_id,
            "w2_geometry_id": config.w2_geometry_id,
            "factor_model": config.factor_model,
        },
        "sample": {
            "n_assets": len(tickers),
            "calibration_dates": len(calibration.dates),
            "evaluation_dates": len(evaluation.dates),
            "calibration_start": str(pd.Timestamp(calibration.dates.min()).date()),
            "calibration_end": str(pd.Timestamp(calibration.dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(evaluation.dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(evaluation.dates.max()).date()),
        },
        "design": {
            "encoder_training_cutoff": cutoff.isoformat(),
            "article_window_start": article_start.isoformat(),
            "calibration_start": calibration_start.isoformat(),
            "representation_is_pre_period": (
                cutoff < article_start and cutoff < calibration_start
            ),
            "operator_frozen_before_evaluation": True,
            "rbf_bandwidth": "median off-diagonal squared W2 distance",
            "bootstrap_draws": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "seed": config.seed,
            "confidence_level": config.confidence_level,
            "unit": "held-out Gaussian quasi-log score per firm-day",
        },
        "model_rows": model_rows,
        "paired_contrasts": {
            "w2_minus_rbf": _contrast_summary(
                w2_minus_rbf,
                candidate="w2",
                reference="rbf_median",
                estimate=float(
                    np.mean(daily_scores["w2"] - daily_scores["rbf_median"])
                ),
                confidence_level=config.confidence_level,
            )
        },
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bootstrap.to_parquet(paths.output_dir / "bootstrap_draws.parquet", index=False)
    return payload


__all__ = [
    "Paper5PointInTimeConfig",
    "Paper5PointInTimeError",
    "Paper5PointInTimePaths",
    "run_paper5_point_in_time",
]

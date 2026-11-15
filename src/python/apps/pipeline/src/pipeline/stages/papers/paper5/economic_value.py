"""Factor-conditioned economic interpretation diagnostics for Paper 5.

This stage is deliberately about the interaction field, not portfolio choice.
It compares fixed target-to-peer operators under the primary factor model and
records a small, deterministic panel explaining the W2 weights.  No return
information is used to construct an operator or to tune the RBF bandwidth
outside the registered calibration split.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from statadic.estimand.core import Covariance, Estimand
from statadic.estimation.covariance import LedoitWolf

from pipeline.io.fama_french import load_factor_data_config, load_factor_model
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5.baskets import industry_weights
from pipeline.stages.papers.paper5.bootstrap import (
    paired_refit_provenance,
    paired_refit_schedule,
)
from pipeline.stages.papers.paper5.field_operators import (
    FieldOperatorIdentity,
    FieldOperatorPaths,
    co_mention_operator,
    equal_support_operator,
    load_field_operator_inputs,
    median_bandwidth,
    rbf_median_operator,
    rbf_weights,
    validate_operator,
)
from pipeline.stages.papers.paper5.transport_order import _align_factor_panel
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.papers.paper5.contracts import PairedScoreContrast
    from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]
IntVector = NDArray[np.int64]

logger = logging.getLogger(__name__)

_MODELS = (
    "factor_only",
    "ledoit_wolf",
    "w1",
    "w2",
    "rbf_median",
    "rbf_tuned",
    "centroid",
    "equal_support",
    "co_mentions",
    "industry_sector",
)
_NON_FACTOR_MODELS = _MODELS[1:]
_FIELD_MODELS = _MODELS[2:]
_DEFAULT_RBF_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
_MINIMUM_DATES = 30
_MATRIX_DIMENSIONS = 2
_MINIMUM_FIRMS = 2
_TOP_PEERS = 5


def _raise(message: str, exception: type[Exception] = ValueError) -> NoReturn:
    """Raise a validation error while keeping messages at the call site small."""
    raise exception(message)


@dataclass(frozen=True, slots=True)
class Paper5EconomicValuePaths:
    """Input and output boundaries for the economic-value diagnostics."""

    returns_dir: Path
    oos_returns_dir: Path
    w1_barycentre_dir: Path
    w2_barycentre_dir: Path
    distance_artifact_dir: Path
    embeddings_dir: Path
    co_mentions_adjacency: Path
    universe_csv: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5EconomicValueConfig:
    """Frozen factor, split, tuning, and bootstrap contract."""

    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    inner_training_end: str = "2020-12-31"
    inner_validation_start: str = "2021-01-01"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    confidence_level: float = 0.95
    factor_model: str = "ff5_mom"
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    w1_barycentre_arm_id: str = "wasserstein_w1_loo"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w1_geometry_id: str = "wasserstein_w1"
    w2_geometry_id: str = "wasserstein_w2"
    rbf_multipliers: tuple[float, ...] = _DEFAULT_RBF_MULTIPLIERS
    active_weight_tolerance: float = 1e-8

    def validate(self) -> None:
        """Reject invalid schedules before reading production artifacts."""
        if self.n_bootstrap < 1 or self.block_length < 1:
            _raise("bootstrap draws and block length must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            _raise("confidence level must lie strictly between zero and one")
        if not self.rbf_multipliers or any(
            not np.isfinite(multiplier) or multiplier <= 0.0
            for multiplier in self.rbf_multipliers
        ):
            _raise("RBF bandwidth multipliers must be finite and positive")


@dataclass(frozen=True, slots=True)
class _Inputs:
    tickers: list[str]
    matrices: dict[str, FloatMatrix]
    squared_distances: FloatMatrix
    calibration: _AlignedFactorPanel
    evaluation: _AlignedFactorPanel
    co_mentions: FloatMatrix
    industry: dict[str, str]
    sector: dict[str, str]


def _eigenvalues(matrix: FloatMatrix) -> NDArray[np.complex128]:
    """Return explicitly complex eigenvalues for the SAR log determinant."""
    return cast(
        "NDArray[np.complex128]",
        np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128),
    )


def _load_centroids(embeddings_dir: Path, tickers: list[str]) -> FloatMatrix:
    """Load article-cloud means in the same representation as the geometry."""
    centroids: list[FloatVector] = []
    dimension: int | None = None
    for ticker in tickers:
        frame = pd.read_parquet(embeddings_dir / f"{ticker}.parquet")
        if "embedding" not in frame.columns or frame.empty:
            _raise(f"embedding artifact for {ticker} is missing or empty")
        embedding_rows = cast("list[FloatVector]", frame["embedding"].tolist())
        values = np.vstack(embedding_rows).astype(np.float64)
        if values.ndim != _MATRIX_DIMENSIONS or not np.all(np.isfinite(values)):
            _raise(f"embedding artifact for {ticker} is malformed")
        if dimension is None:
            dimension = values.shape[1]
        elif values.shape[1] != dimension:
            _raise("embedding artifacts have inconsistent dimensions")
        centroids.append(cast("FloatVector", values.mean(axis=0)))
    return cast("FloatMatrix", np.vstack(centroids))


def _centroid_weights(centroids: FloatMatrix) -> FloatMatrix:
    """Reconstruct each leave-one-out centroid on the peer simplex."""
    values = np.asarray(centroids, dtype=np.float64)
    n_assets = values.shape[0]
    if values.ndim != _MATRIX_DIMENSIONS or n_assets < _MINIMUM_FIRMS:
        _raise("centroid reconstruction requires at least two firms")
    gram = values @ values.T
    result = np.zeros((n_assets, n_assets), dtype=np.float64)
    constraint = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    for target in range(n_assets):
        peers = np.delete(np.arange(n_assets), target)
        peer_gram = gram[np.ix_(peers, peers)]
        linear = gram[peers, target]

        def objective(
            weights: NDArray[np.float64],
            peer_gram: NDArray[np.float64] = peer_gram,
            linear: NDArray[np.float64] = linear,
        ) -> float:
            return float(weights @ peer_gram @ weights - 2.0 * linear @ weights)

        def gradient(
            weights: NDArray[np.float64],
            peer_gram: NDArray[np.float64] = peer_gram,
            linear: NDArray[np.float64] = linear,
        ) -> NDArray[np.float64]:
            return cast("NDArray[np.float64]", 2.0 * (peer_gram @ weights - linear))

        initial = np.full(len(peers), 1.0 / len(peers), dtype=np.float64)
        fit = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(peers),
            constraints=constraint,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not fit.success or not np.all(np.isfinite(fit.x)):
            _raise(
                f"centroid reconstruction failed for target {target}",
                RuntimeError,
            )
        row = np.clip(np.asarray(fit.x, dtype=np.float64), 0.0, None)
        row_sum = float(row.sum())
        if row_sum <= 0.0:
            _raise(
                f"centroid reconstruction produced an empty target {target}",
                RuntimeError,
            )
        result[target, peers] = row / row_sum
    return validate_operator(result, name="centroid weights")


def _load_inputs(
    paths: Paper5EconomicValuePaths, config: Paper5EconomicValueConfig
) -> _Inputs:
    """Load all fields and the common factor-complete calibration/evaluation panel."""
    fields = load_field_operator_inputs(
        FieldOperatorPaths(
            returns_dir=paths.returns_dir,
            w1_barycentre_dir=paths.w1_barycentre_dir,
            w2_barycentre_dir=paths.w2_barycentre_dir,
            distance_artifact_dir=paths.distance_artifact_dir,
            co_mentions_adjacency=paths.co_mentions_adjacency,
            universe_csv=paths.universe_csv,
        ),
        FieldOperatorIdentity(
            provider_id=config.provider_id,
            representation_id=config.representation_id,
            w1_barycentre_arm_id=config.w1_barycentre_arm_id,
            w2_barycentre_arm_id=config.w2_barycentre_arm_id,
            w1_geometry_id=config.w1_geometry_id,
            w2_geometry_id=config.w2_geometry_id,
        ),
    )
    tickers = fields.tickers
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    if config.factor_model != factor_config.primary_model:
        _raise("economic-value stage must use the governed primary factor model")
    factor_panel = load_factor_model(factor_config, cast("Any", config.factor_model))
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
    return _Inputs(
        tickers=tickers,
        matrices=fields.matrices,
        squared_distances=fields.squared_distances,
        calibration=calibration,
        evaluation=evaluation,
        co_mentions=fields.co_mentions,
        industry=fields.industry,
        sector=fields.sector,
    )


def _selection(panel: _AlignedFactorPanel, indices: IntVector | None) -> IntVector:
    """Resolve a date selection, defaulting to the panel's own order."""
    if indices is None:
        return cast("IntVector", np.arange(len(panel.dates), dtype=np.int64))
    return indices


def _fit(
    panel: _AlignedFactorPanel,
    matrix: FloatMatrix,
    *,
    eigvals: NDArray[np.complex128] | None = None,
    indices: IntVector | None = None,
) -> est.FactorSarFit:
    """Fit one factor-conditioned spatial field without standard errors."""
    selected = _selection(panel, indices)
    fit = est.sar_factor_qmle(
        panel.excess_returns[selected],
        matrix,
        panel.factors[selected],
        compute_se=False,
        eigvals_w=_eigenvalues(matrix) if eigvals is None else eigvals,
    )
    if not fit.converged:
        _raise("economic-value comparator fit did not converge", RuntimeError)
    return fit


def _factor_only_fit(
    panel: _AlignedFactorPanel,
    reference: FloatMatrix,
    *,
    eigvals: NDArray[np.complex128] | None = None,
    indices: IntVector | None = None,
) -> est.FactorSarFit:
    """Fit the common factor-only restriction at rho=0."""
    selected = _selection(panel, indices)
    fit = est.sar_factor_fit_at_rho(
        panel.excess_returns[selected],
        reference,
        panel.factors[selected],
        0.0,
        eigvals_w=_eigenvalues(reference) if eigvals is None else eigvals,
    )
    if not fit.converged:
        _raise("factor-only comparator fit did not converge", RuntimeError)
    return fit


def _daily_score(
    fit: est.FactorSarFit,
    panel: _AlignedFactorPanel,
    matrix: FloatMatrix,
    *,
    eigvals: NDArray[np.complex128] | None = None,
    indices: IntVector | None = None,
) -> FloatVector:
    """Return per-date mean Gaussian quasi-log score across firms."""
    selected = _selection(panel, indices)
    scores = est.sar_factor_daily_logscore(
        fit,
        panel.excess_returns[selected],
        matrix,
        panel.factors[selected],
        eigvals_w=_eigenvalues(matrix) if eigvals is None else eigvals,
    )
    return cast(
        "FloatVector",
        np.asarray(scores, dtype=np.float64) / panel.excess_returns.shape[1],
    )


def _ledoit_wolf_covariance(
    panel: _AlignedFactorPanel,
    factor_fit: est.FactorSarFit,
    *,
    indices: IntVector | None = None,
) -> FloatMatrix:
    """Estimate a direct covariance from calibration factor residuals."""
    selected = _selection(panel, indices)
    excess = panel.excess_returns[selected]
    design = np.column_stack(
        (np.ones(excess.shape[0], dtype=np.float64), panel.factors[selected])
    )
    residuals = excess - design @ factor_fit.coefficients
    covariance = np.asarray(
        Estimand(Covariance(), estimator=LedoitWolf()).estimate(residuals).sigma,
        dtype=np.float64,
    )
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(covariance)) or float(np.min(eigenvalues)) <= 0.0:
        _raise("Ledoit-Wolf residual covariance is not positive definite", RuntimeError)
    return covariance


def _ledoit_wolf_daily_score(
    factor_fit: est.FactorSarFit,
    panel: _AlignedFactorPanel,
    covariance: FloatMatrix,
    *,
    indices: IntVector | None = None,
) -> FloatVector:
    """Score held-out factor residuals under a fixed direct covariance."""
    selected = _selection(panel, indices)
    excess = panel.excess_returns[selected]
    design = np.column_stack(
        (np.ones(excess.shape[0], dtype=np.float64), panel.factors[selected])
    )
    residuals = excess - design @ factor_fit.coefficients
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        _raise(
            "Ledoit-Wolf residual covariance has invalid log determinant",
            RuntimeError,
        )
    quadratic = np.sum(
        residuals * np.linalg.solve(covariance, residuals.T).T,
        axis=1,
    )
    scores = -0.5 * (
        panel.excess_returns.shape[1] * np.log(2.0 * np.pi) + logdet + quadratic
    )
    return cast(
        "FloatVector",
        np.asarray(scores, dtype=np.float64) / panel.excess_returns.shape[1],
    )


def _tune_rbf(
    panel: _AlignedFactorPanel,
    squared_distances: FloatMatrix,
    config: Paper5EconomicValueConfig,
) -> tuple[FloatMatrix, dict[str, object], list[dict[str, object]]]:
    """Choose an RBF multiplier on the registered pre-2021 inner split."""
    train_mask = pd.to_datetime(panel.dates) <= pd.Timestamp(config.inner_training_end)
    validation_mask = pd.to_datetime(panel.dates) >= pd.Timestamp(
        config.inner_validation_start
    )
    if (
        int(np.sum(train_mask)) < _MINIMUM_DATES
        or int(np.sum(validation_mask)) < _MINIMUM_DATES
    ):
        _raise("RBF tuning split leaves too few factor-complete dates")
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train = type(panel)(
        dates=panel.dates[train_indices],
        raw_returns=panel.raw_returns[train_indices],
        excess_returns=panel.excess_returns[train_indices],
        factors=panel.factors[train_indices],
        dropped_return_dates=0,
    )
    validation = type(panel)(
        dates=panel.dates[validation_indices],
        raw_returns=panel.raw_returns[validation_indices],
        excess_returns=panel.excess_returns[validation_indices],
        factors=panel.factors[validation_indices],
        dropped_return_dates=0,
    )
    median = median_bandwidth(squared_distances)
    factor_fit = _factor_only_fit(train, rbf_weights(squared_distances, median))
    factor_score = float(
        np.mean(
            _daily_score(
                factor_fit,
                validation,
                rbf_weights(squared_distances, median),
            )
        )
    )
    records: list[dict[str, object]] = []
    best_multiplier: float | None = None
    best_score = -np.inf
    for multiplier in config.rbf_multipliers:
        matrix = rbf_weights(squared_distances, median * multiplier)
        fit = _fit(train, matrix)
        score = float(np.mean(_daily_score(fit, validation, matrix)))
        record: dict[str, object] = {
            "bandwidth_multiplier": multiplier,
            "bandwidth": median * multiplier,
            "validation_score": score,
            "validation_gain_vs_factor_only": score - factor_score,
            "converged": fit.converged,
        }
        records.append(record)
        if score > best_score or (
            score == best_score
            and (best_multiplier is None or multiplier < best_multiplier)
        ):
            best_score = score
            best_multiplier = multiplier
    if best_multiplier is None:
        _raise("RBF tuning produced no candidate", RuntimeError)
    return (
        rbf_weights(squared_distances, median * best_multiplier),
        {
            "method": "calibration_inner_split",
            "training_end": config.inner_training_end,
            "validation_start": config.inner_validation_start,
            "median_bandwidth": median,
            "selected_multiplier": best_multiplier,
            "selected_bandwidth": median * best_multiplier,
            "frozen_across_bootstrap_draws": True,
            "candidates": records,
        },
        records,
    )


def _draw_score_gains(
    inputs: _Inputs,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, NDArray[np.complex128]],
    calibration_indices: IntVector,
    score_indices: IntVector,
) -> dict[str, float]:
    """Refit every comparator on one calibration draw and score one evaluation draw."""
    factor_fit = _factor_only_fit(
        inputs.calibration,
        matrices["w2"],
        eigvals=eigenvalues["w2"],
        indices=calibration_indices,
    )
    factor_score = float(
        np.mean(
            _daily_score(
                factor_fit,
                inputs.evaluation,
                matrices["w2"],
                eigvals=eigenvalues["w2"],
                indices=score_indices,
            )
        )
    )
    covariance = _ledoit_wolf_covariance(
        inputs.calibration, factor_fit, indices=calibration_indices
    )
    gains = {
        "ledoit_wolf": float(
            np.mean(
                _ledoit_wolf_daily_score(
                    factor_fit,
                    inputs.evaluation,
                    covariance,
                    indices=score_indices,
                )
            )
        )
        - factor_score
    }
    for model in _FIELD_MODELS:
        fit = _fit(
            inputs.calibration,
            matrices[model],
            eigvals=eigenvalues[model],
            indices=calibration_indices,
        )
        score = float(
            np.mean(
                _daily_score(
                    fit,
                    inputs.evaluation,
                    matrices[model],
                    eigvals=eigenvalues[model],
                    indices=score_indices,
                )
            )
        )
        gains[model] = score - factor_score
    return gains


def _bootstrap_intervals(
    inputs: _Inputs,
    matrices: dict[str, FloatMatrix],
    eigenvalues: dict[str, NDArray[np.complex128]],
    config: Paper5EconomicValueConfig,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], dict[str, object]]:
    """Bootstrap paired held-out score gains under per-draw calibration refits.

    Each draw refits the factor-only restriction, the Ledoit--Wolf residual
    covariance, and every field comparator on a resampled calibration panel,
    then scores them on a resampled evaluation panel. The comparators therefore
    carry the same calibration uncertainty as the primary spatial-value table,
    so a gain reported in both places carries one interval.
    """
    schedule = paired_refit_schedule(
        calibration_observations=len(inputs.calibration.dates),
        evaluation_observations=len(inputs.evaluation.dates),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed,
    )
    records: list[dict[str, object]] = []
    for draw_id in range(config.n_bootstrap):
        gains = _draw_score_gains(
            inputs,
            matrices,
            eigenvalues,
            schedule.calibration[draw_id],
            schedule.evaluation_score[draw_id],
        )
        records.extend(
            {"draw_id": draw_id, "model": model, "score_gain": gains[model]}
            for model in _NON_FACTOR_MODELS
        )
        if (draw_id + 1) % 100 == 0 or draw_id + 1 == config.n_bootstrap:
            logger.info(
                "Paper 5 economic-value refit bootstrap %d/%d",
                draw_id + 1,
                config.n_bootstrap,
            )
    draws = pd.DataFrame.from_records(records)
    tail = (1.0 - config.confidence_level) / 2.0
    intervals = {
        model: (
            float(draws.loc[draws["model"] == model, "score_gain"].quantile(tail)),
            float(
                draws.loc[draws["model"] == model, "score_gain"].quantile(1.0 - tail)
            ),
        )
        for model in _NON_FACTOR_MODELS
    }
    return draws, intervals, paired_refit_provenance(schedule)


def _paired_score_contrast(
    draws: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    point_estimate: float,
    confidence_level: float,
) -> PairedScoreContrast:
    """Summarize a paired held-out score contrast from common bootstrap draws."""
    required = {"draw_id", "model", "score_gain"}
    if not required <= set(draws.columns):
        _raise("bootstrap score draws lack the paired contrast columns")
    selected = draws.loc[
        draws["model"].isin([candidate, reference]),
        ["draw_id", "model", "score_gain"],
    ]
    if selected.duplicated(["draw_id", "model"]).any():
        _raise("paired score draws contain duplicate model/draw rows")
    # The duplicate check above makes (draw_id, model) unique, so the aggregation
    # never combines rows; "mean" is chosen over "first" only because pandas types
    # the latter out of `aggfunc`.
    pivot = selected.pivot_table(
        index="draw_id", columns="model", values="score_gain", aggfunc="mean"
    )
    if candidate not in pivot or reference not in pivot:
        _raise("paired score draws do not contain both contrast models")
    difference = pivot[candidate].to_numpy(dtype=np.float64) - pivot[
        reference
    ].to_numpy(dtype=np.float64)
    difference = difference[np.isfinite(difference)]
    if difference.size == 0:
        _raise("paired score contrast has no finite bootstrap draws")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(difference, [tail, 1.0 - tail])
    return {
        "candidate": candidate,
        "reference": reference,
        "estimate": float(point_estimate),
        "bootstrap_mean": float(np.mean(difference)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "positive_share": float(np.mean(difference > 0.0)),
        "bootstrap_valid_draws": int(difference.size),
        "confidence_level": float(confidence_level),
        "unit": "held-out Gaussian quasi-log score per firm-day",
    }


def _peer_rows(
    tickers: list[str],
    weights: FloatMatrix,
    squared_distances: FloatMatrix,
    co_mentions: FloatMatrix,
    industry: dict[str, str],
    sector: dict[str, str],
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a deterministic top-five peer panel for declared targets."""
    effective = 1.0 / np.sum(np.square(weights), axis=1)
    incoming = weights.sum(axis=0)
    ordered = sorted(zip(effective, tickers, strict=True))
    median_effective = float(np.median(effective))
    roles: dict[str, list[str]] = {"declared_amd": ["AMD"]}
    for role, target in (
        ("minimum_effective_source_count", ordered[0][1]),
        (
            "median_effective_source_count",
            min(
                tickers,
                key=lambda t: (abs(effective[tickers.index(t)] - median_effective), t),
            ),
        ),
        ("maximum_effective_source_count", ordered[-1][1]),
    ):
        roles.setdefault(role, []).append(target)
    selected: dict[str, str] = {}
    for role, targets in roles.items():
        for target in targets:
            selected.setdefault(target, role)
    rows: list[dict[str, object]] = []
    for target in sorted(selected):
        target_index = tickers.index(target)
        candidates = [index for index in range(len(tickers)) if index != target_index]
        ranked_distance = sorted(
            candidates,
            key=lambda index: (squared_distances[target_index, index], tickers[index]),
        )
        distance_rank = {index: rank + 1 for rank, index in enumerate(ranked_distance)}
        top = sorted(
            candidates,
            key=lambda index: (-weights[target_index, index], tickers[index]),
        )[:_TOP_PEERS]
        for candidate_index in top:
            candidate = tickers[candidate_index]
            rows.append(
                {
                    "target": target,
                    "target_role": selected[target],
                    "candidate": candidate,
                    "weight": float(weights[target_index, candidate_index]),
                    "distance_rank": distance_rank[candidate_index],
                    "same_sector": sector[target] == sector[candidate],
                    "same_industry": industry[target] == industry[candidate],
                    "co_mention": bool(
                        co_mentions[target_index, candidate_index] > tolerance
                    ),
                    "effective_source_count": float(effective[target_index]),
                    "incoming_mass": float(incoming[target_index]),
                }
            )
    summary: dict[str, object] = {
        "target_roles": selected,
        "effective_source_count": {
            ticker: float(effective[index]) for index, ticker in enumerate(tickers)
        },
        "incoming_mass": {
            ticker: float(incoming[index]) for index, ticker in enumerate(tickers)
        },
        "construction": "top-five actual W2 weights; distance rank is pairwise W2 rank",
        "descriptive_only": True,
    }
    return pd.DataFrame.from_records(rows), summary


@dataclass(frozen=True, slots=True)
class _ModelFitSummary:
    """Fitted-operator diagnostics for one held-out economic-value comparator."""

    operator_family: str
    rho_hat: float | None
    loglik: float | None
    parameter_count: int | None
    converged: bool


def _model_fit_summary(
    model: str, evaluation_fits: dict[str, est.FactorSarFit]
) -> _ModelFitSummary:
    """Summarize one comparator's fitted operator, or its absence for Ledoit-Wolf.

    The Ledoit-Wolf comparator is a shrinkage covariance rather than a fitted
    spatial operator, so it reports no rho, likelihood, or parameter count and is
    convergent by construction.
    """
    if model == "ledoit_wolf":
        return _ModelFitSummary(
            operator_family="generic_residual_covariance",
            rho_hat=None,
            loglik=None,
            parameter_count=None,
            converged=True,
        )
    spatial = model != "factor_only"
    fit = evaluation_fits[model]
    return _ModelFitSummary(
        operator_family="field_comparator" if spatial else "factor_only",
        rho_hat=float(fit.rho_hat),
        loglik=float(fit.loglik),
        parameter_count=est.factor_sar_parameter_count(fit, spatial=spatial),
        converged=bool(fit.converged),
    )


def run_paper5_economic_value(
    paths: Paper5EconomicValuePaths,
    config: Paper5EconomicValueConfig,
) -> dict[str, object]:
    """Run factor-conditioned comparator and peer-interpretation diagnostics."""
    config.validate()
    inputs = _load_inputs(paths, config)
    centroids = _load_centroids(paths.embeddings_dir, inputs.tickers)
    centroid_matrix = _centroid_weights(centroids)
    equal_matrix = equal_support_operator(
        inputs.matrices["w2"], active_tolerance=config.active_weight_tolerance
    )
    industry_matrix = validate_operator(
        industry_weights(inputs.tickers, inputs.industry, inputs.sector),
        name="industry-sector weights",
    )
    tuned_matrix, tuning, tuning_rows = _tune_rbf(
        inputs.calibration, inputs.squared_distances, config
    )
    matrices: dict[str, FloatMatrix] = {
        "w1": inputs.matrices["w1"],
        "w2": inputs.matrices["w2"],
        "rbf_median": rbf_median_operator(inputs.squared_distances),
        "rbf_tuned": tuned_matrix,
        "centroid": centroid_matrix,
        "equal_support": equal_matrix,
        "co_mentions": co_mention_operator(inputs.co_mentions),
        "industry_sector": industry_matrix,
    }
    eigenvalues = {model: _eigenvalues(matrix) for model, matrix in matrices.items()}
    fits: dict[str, est.FactorSarFit] = {
        "factor_only": _factor_only_fit(
            inputs.calibration, matrices["w2"], eigvals=eigenvalues["w2"]
        )
    }
    fits.update(
        {
            model: _fit(inputs.calibration, matrix, eigvals=eigenvalues[model])
            for model, matrix in matrices.items()
        }
    )
    ledoit_wolf_covariance = _ledoit_wolf_covariance(
        inputs.calibration, fits["factor_only"]
    )
    evaluation_fits: dict[str, est.FactorSarFit] = {
        "factor_only": _factor_only_fit(
            inputs.evaluation, matrices["w2"], eigvals=eigenvalues["w2"]
        )
    }
    evaluation_fits.update(
        {
            model: _fit(inputs.evaluation, matrix, eigvals=eigenvalues[model])
            for model, matrix in matrices.items()
        }
    )
    daily_scores = {
        "factor_only": _daily_score(
            fits["factor_only"],
            inputs.evaluation,
            matrices["w2"],
            eigvals=eigenvalues["w2"],
        ),
        "ledoit_wolf": _ledoit_wolf_daily_score(
            fits["factor_only"], inputs.evaluation, ledoit_wolf_covariance
        ),
    }
    daily_scores.update(
        {
            model: _daily_score(
                fits[model],
                inputs.evaluation,
                matrices[model],
                eigvals=eigenvalues[model],
            )
            for model in _FIELD_MODELS
        }
    )
    bootstrap, intervals, bootstrap_provenance = _bootstrap_intervals(
        inputs, matrices, eigenvalues, config
    )
    factor_score = float(np.mean(daily_scores["factor_only"]))
    paired_contrasts = {
        "w2_minus_centroid": _paired_score_contrast(
            bootstrap,
            candidate="w2",
            reference="centroid",
            point_estimate=float(
                np.mean(daily_scores["w2"] - daily_scores["centroid"])
            ),
            confidence_level=config.confidence_level,
        ),
        "ledoit_wolf_minus_w2": _paired_score_contrast(
            bootstrap,
            candidate="ledoit_wolf",
            reference="w2",
            point_estimate=float(
                np.mean(daily_scores["ledoit_wolf"] - daily_scores["w2"])
            ),
            confidence_level=config.confidence_level,
        ),
    }
    model_rows: list[dict[str, object]] = []
    labels = {
        "factor_only": "Factor-only",
        "ledoit_wolf": "Direct Ledoit--Wolf residual covariance",
        "w1": "W1 barycentric",
        "w2": "W2 barycentric",
        "rbf_median": "Median-bandwidth W2 RBF",
        "rbf_tuned": "Calibration-tuned W2 RBF",
        "centroid": "Centroid reconstruction",
        "equal_support": "Equal active support",
        "co_mentions": "Persistent co-mentions",
        "industry_sector": "Industry/sector fallback",
    }
    for model in _MODELS:
        summary = _model_fit_summary(model, evaluation_fits)
        score = float(np.mean(daily_scores[model]))
        interval = (0.0, 0.0) if model == "factor_only" else intervals[model]
        model_rows.append(
            {
                "model": model,
                "label": labels[model],
                "operator_family": summary.operator_family,
                "rho_hat": summary.rho_hat,
                "loglik": summary.loglik,
                "held_out_score": score,
                "held_out_score_gain_vs_factor_only": score - factor_score,
                "held_out_gain_ci_lower": interval[0],
                "held_out_gain_ci_upper": interval[1],
                "parameter_count": summary.parameter_count,
                "converged": summary.converged,
            }
        )
    peer_rows, peer_summary = _peer_rows(
        inputs.tickers,
        inputs.matrices["w2"],
        inputs.squared_distances,
        inputs.co_mentions,
        inputs.industry,
        inputs.sector,
        config.active_weight_tolerance,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(model_rows).to_parquet(
        paths.output_dir / "model_rows.parquet", index=False
    )
    peer_rows.to_parquet(paths.output_dir / "peer_rows.parquet", index=False)
    bootstrap.to_parquet(paths.output_dir / "bootstrap_draws.parquet", index=False)
    payload: dict[str, object] = {
        "schema_version": 2,
        "analysis_id": "paper5-economic-value-v2",
        "scope": {
            "claim": (
                "factor-conditioned peer-field and covariance benchmark comparison "
                "with descriptive peer interpretation"
            ),
            "portfolio_optimization": False,
            "risk_forecasting": False,
            "nonnested_qlr": False,
        },
        "sample": {
            "n_assets": len(inputs.tickers),
            "calibration_dates": len(inputs.calibration.dates),
            "evaluation_dates": len(inputs.evaluation.dates),
            "calibration_start": str(
                pd.Timestamp(inputs.calibration.dates.min()).date()
            ),
            "calibration_end": str(pd.Timestamp(inputs.calibration.dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(inputs.evaluation.dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(inputs.evaluation.dates.max()).date()),
        },
        "model_order": list(_MODELS),
        "model_rows": model_rows,
        "paired_contrasts": paired_contrasts,
        "rbf_tuning": tuning,
        "rbf_tuning_rows": tuning_rows,
        "bootstrap": {
            **bootstrap_provenance,
            "confidence_level": config.confidence_level,
            "unit": "paired held-out daily score gain per firm-day",
            "refit_scope": (
                "factor-only restriction, Ledoit-Wolf residual covariance, and "
                "every field coefficient are refitted on each calibration draw; "
                "the tuned RBF bandwidth is frozen at its full-calibration "
                "selection so the operator is held fixed across draws"
            ),
        },
        "residual_covariance_benchmark": {
            "estimator": "statadic.estimation.covariance.LedoitWolf",
            "input": "2018-2022 factor-only residuals",
            "lookahead": False,
            "positive_definite": True,
            "refit_per_bootstrap_draw": True,
        },
        "peer_panel": peer_summary,
        "peer_panel_rows": peer_rows.to_dict(orient="records"),
        "interpretation_status": (
            "descriptive_governed_panel_not_external_business_validation"
        ),
    }
    (paths.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


__all__ = [
    "Paper5EconomicValueConfig",
    "Paper5EconomicValuePaths",
    "run_paper5_economic_value",
]

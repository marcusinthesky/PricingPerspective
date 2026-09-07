"""Artifact driver for the preregistered Paper 5 economic gate."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd

from pipeline.io.barycentre_artifacts import read_target_projection_artifact
from pipeline.stages.papers.paper5._gate.bootstrap import (
    covariance_bootstrap_pvalue,
    diebold_mariano_agreement,
    paired_rmse_bootstrap_pvalue,
    randomization_pvalue,
    stationary_bootstrap_indices,
)
from pipeline.stages.papers.paper5._gate.contracts import (
    CALIBRATION_START,
    GATE_FOLDS,
    raise_gate_error,
)
from pipeline.stages.papers.paper5._gate.metrics import (
    FloatMatrix,
    correlation_knn_weights,
    covariance_row_tv,
    equal_support_weights,
    evaluate_weights,
    fit_market_betas,
    industry_weights,
    random_matched_sparsity_weights,
    validate_weight_matrix,
)
from pipeline.stages.papers.paper5._run.robustness import _load_shared_w_flat
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline.stages.papers.paper5._gate.contracts import (
        Paper5GateConfig,
        Paper5GatePaths,
    )

logger = logging.getLogger(__name__)
_MIN_EVALUATION_DATES = 2
type WeightEvaluation = dict[str, FloatMatrix | float]


class _OperatorIdentity(Protocol):
    """Structural identity required to load the frozen W-flat artifact.

    Declared read-only: every stage config that supplies it is a frozen
    dataclass, which cannot satisfy a protocol demanding settable attributes.
    """

    @property
    def provider_id(self) -> str:
        """Return the embedding provider the artifact must have been built with."""

    @property
    def representation_id(self) -> str:
        """Return the representation the artifact must have been built with."""

    @property
    def barycentre_arm_id(self) -> str:
        """Return the barycentre arm the artifact must have been built with."""

    @property
    def geometry_id(self) -> str:
        """Return the ground geometry the artifact must have been built with."""


def _load_gate_operator(
    artifact_dir: Path,
    config: _OperatorIdentity,
) -> tuple[list[str], NDArray[np.float64], dict[str, object]]:
    """Load W-flat by ticker identity; candidate indices are row-local."""
    weights, _diagnostics, summary = read_target_projection_artifact(
        artifact_dir,
        expected_identity={
            "arm_id": config.barycentre_arm_id,
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "geometry": config.geometry_id,
            "feasible_set": "simplex_nonnegative",
        },
    )
    targets = weights[["target_index", "target"]].drop_duplicates()
    targets = targets.sort_values("target_index", kind="mergesort")
    tickers = cast("list[str]", targets["target"].tolist())
    if len(tickers) != int(weights["target_index"].nunique()):
        raise_gate_error("target indices do not define a unique ticker order")
    matrix, _ = _load_shared_w_flat(
        artifact_dir,
        tickers,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        arm_id=config.barycentre_arm_id,
        geometry_id=config.geometry_id,
    )
    validate_weight_matrix(matrix)
    return tickers, matrix, summary


def _load_classifications(
    universe_csv: Path,
    tickers: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Load industry and sector labels for exactly the priced universe."""
    universe = pd.read_csv(universe_csv)
    required = {"Symbol", "Industry", "Sector"}
    if not required.issubset(universe.columns):
        raise_gate_error(f"universe is missing columns {sorted(required)}")
    indexed = universe.set_index("Symbol")
    missing = sorted(set(tickers) - set(indexed.index))
    if missing:
        raise_gate_error(f"universe classifications missing tickers {missing}")
    industry = {ticker: str(indexed.loc[ticker, "Industry"]) for ticker in tickers}
    sector = {ticker: str(indexed.loc[ticker, "Sector"]) for ticker in tickers}
    return industry, sector


def _random_metrics(
    training_returns: NDArray[np.float64],
    evaluation_returns: NDArray[np.float64],
    random_weights: NDArray[np.float64],
    *,
    active_tolerance: float,
    chunk_size: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate random systematic and covariance errors without a dense 4-D tensor."""
    betas = fit_market_betas(training_returns)
    market = evaluation_returns.mean(axis=1)
    target_residuals = evaluation_returns - market[:, None] * betas[None, :]
    systematic = np.empty(len(random_weights), dtype=np.float64)
    covariance = np.empty(len(random_weights), dtype=np.float64)
    for start in range(0, len(random_weights), chunk_size):
        stop = min(start + chunk_size, len(random_weights))
        weights = random_weights[start:stop]
        basket_returns = np.einsum(
            "tn,din->dti", evaluation_returns, weights, optimize=True
        )
        basket_betas = np.einsum("dij,j->di", weights, betas, optimize=True)
        basket_residuals = (
            basket_returns - market[None, :, None] * basket_betas[:, None, :]
        )
        errors = target_residuals[None, :, :] - basket_residuals
        systematic[start:stop] = np.sqrt(np.mean(np.square(errors), axis=(1, 2)))
        for local_index, matrix in enumerate(weights):
            covariance[start + local_index], _ = covariance_row_tv(
                evaluation_returns,
                matrix,
                active_tolerance=active_tolerance,
            )
    return systematic, covariance


def classify_gate(
    *,
    decisive_pass: bool,
    systematic_random_pass: bool,
    covariance_random_pass: bool,
    covariance_equal_pass: bool,
) -> tuple[bool, str]:
    """Apply the registered all-or-nothing gate and name the failure branch."""
    passed = all(
        (
            decisive_pass,
            systematic_random_pass,
            covariance_random_pass,
            covariance_equal_pass,
        )
    )
    if passed:
        return True, "pass"
    if not systematic_random_pass or not covariance_random_pass:
        return False, "not_distinguishable_from_random"
    return False, "support_informative_magnitudes_not"


def _plain(value: object) -> object:
    """Convert NumPy containers and scalars to deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _evaluation_scalar(result: WeightEvaluation, metric: str) -> float:
    """Read an aggregate metric while rejecting an unexpected array payload."""
    value = result[metric]
    if not isinstance(value, float):
        raise_gate_error(f"{metric} must be an aggregate float")
    return value


def _write_results(
    output_dir: Path,
    payload: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    """Persist the gate's machine-readable decision and per-firm evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(records).sort_values(
        ["fold", "scheme", "ticker"], kind="mergesort"
    ).to_parquet(output_dir / "results.parquet", index=False)


def run_paper5_economic_gate(  # noqa: PLR0915 - auditable preregistered protocol
    paths: Paper5GatePaths,
    config: Paper5GateConfig,
) -> dict[str, object]:
    """Run the frozen-design 2023--2026 economic-validation gate."""
    config.validate()
    tickers, w_flat, operator_summary = _load_gate_operator(
        paths.shared_barycentre_dir, config
    )
    w_equal = equal_support_weights(
        w_flat,
        active_tolerance=config.active_weight_tolerance,
    )
    industry_of, sector_of = _load_classifications(paths.universe_csv, tickers)
    w_industry = industry_weights(tickers, industry_of, sector_of)
    random_weights = random_matched_sparsity_weights(
        w_flat,
        draws=config.n_random,
        seed=config.seed,
        active_tolerance=config.active_weight_tolerance,
    )

    folds: dict[str, object] = {}
    records: list[dict[str, object]] = []
    pooled_returns: list[NDArray[np.float64]] = []
    pooled_flat_mse: list[NDArray[np.float64]] = []
    pooled_equal_mse: list[NDArray[np.float64]] = []
    systematic_random_passes = 0
    covariance_random_passes = 0

    for fold_index, (fold, (fold_start, fold_end)) in enumerate(GATE_FOLDS.items()):
        training_end = f"{int(fold_start[:4]) - 1}-12-31"
        _, training_returns = load_aligned_return_panel(
            paths.returns_dir,
            tickers,
            CALIBRATION_START,
            training_end,
            oos_returns_dir=paths.oos_returns_dir,
        )
        dates, evaluation_returns = load_aligned_return_panel(
            paths.returns_dir,
            tickers,
            fold_start,
            fold_end,
            oos_returns_dir=paths.oos_returns_dir,
        )
        if len(evaluation_returns) < _MIN_EVALUATION_DATES:
            raise_gate_error(f"{fold} has fewer than two evaluation dates")
        w_correlation = correlation_knn_weights(training_returns, k=config.knn_k)
        schemes = {
            "w_flat": w_flat,
            "equal_support": w_equal,
            "industry": w_industry,
            "correlation_knn": w_correlation,
        }
        scheme_results: dict[str, WeightEvaluation] = {}
        for scheme, matrix in schemes.items():
            result = evaluate_weights(
                training_returns,
                evaluation_returns,
                matrix,
                active_tolerance=config.active_weight_tolerance,
            )
            scheme_results[scheme] = result
            raw_asset = cast("NDArray[np.float64]", result["raw_asset_rmse"])
            systematic_asset = cast(
                "NDArray[np.float64]", result["systematic_asset_rmse"]
            )
            covariance_asset = cast(
                "NDArray[np.float64]", result["covariance_asset_tv"]
            )
            records.extend(
                {
                    "fold": fold,
                    "scheme": scheme,
                    "ticker": ticker,
                    "raw_rmse": float(raw_asset[index]),
                    "systematic_rmse": float(systematic_asset[index]),
                    "covariance_tv": float(covariance_asset[index]),
                }
                for index, ticker in enumerate(tickers)
            )

        random_systematic, random_covariance = _random_metrics(
            training_returns,
            evaluation_returns,
            random_weights,
            active_tolerance=config.active_weight_tolerance,
        )
        focal_systematic = _evaluation_scalar(
            scheme_results["w_flat"], "systematic_rmse"
        )
        focal_covariance = _evaluation_scalar(scheme_results["w_flat"], "covariance_tv")
        systematic_p = randomization_pvalue(focal_systematic, random_systematic)
        covariance_p = randomization_pvalue(focal_covariance, random_covariance)
        systematic_pass = bool(
            focal_systematic < np.quantile(random_systematic, config.alpha)
        )
        covariance_pass = bool(
            focal_covariance < np.quantile(random_covariance, config.alpha)
        )
        systematic_random_passes += int(systematic_pass)
        covariance_random_passes += int(covariance_pass)

        flat_date_mse = cast(
            "NDArray[np.float64]", scheme_results["w_flat"]["systematic_date_mse"]
        )
        equal_date_mse = cast(
            "NDArray[np.float64]",
            scheme_results["equal_support"]["systematic_date_mse"],
        )
        pooled_returns.append(evaluation_returns)
        pooled_flat_mse.append(flat_date_mse)
        pooled_equal_mse.append(equal_date_mse)
        folds[fold] = {
            "start": str(pd.Timestamp(dates.min()).date()),
            "end": str(pd.Timestamp(dates.max()).date()),
            "trading_days": len(evaluation_returns),
            "training_end": training_end,
            "schemes": {
                name: {
                    "raw_rmse": result["raw_rmse"],
                    "systematic_rmse": result["systematic_rmse"],
                    "covariance_tv": result["covariance_tv"],
                }
                for name, result in scheme_results.items()
            },
            "random_floor": {
                "systematic_q05": float(np.quantile(random_systematic, config.alpha)),
                "systematic_p_value": systematic_p,
                "systematic_pass": systematic_pass,
                "covariance_q05": float(np.quantile(random_covariance, config.alpha)),
                "covariance_p_value": covariance_p,
                "covariance_pass": covariance_pass,
            },
        }
        logger.info("paper5 gate completed %s (%d/4)", fold, fold_index + 1)

    pooled_flat = np.concatenate(pooled_flat_mse)
    pooled_equal = np.concatenate(pooled_equal_mse)
    pooled_panel = np.concatenate(pooled_returns, axis=0)
    bootstrap_indices = stationary_bootstrap_indices(
        len(pooled_panel),
        draws=config.n_bootstrap,
        expected_block_length=config.block_length,
        seed=config.seed + 10_000,
    )
    tracking_p, tracking_differences = paired_rmse_bootstrap_pvalue(
        pooled_flat,
        pooled_equal,
        bootstrap_indices,
    )
    flat_rmse = float(np.sqrt(np.mean(pooled_flat)))
    equal_rmse = float(np.sqrt(np.mean(pooled_equal)))
    reduction = (equal_rmse - flat_rmse) / equal_rmse
    covariance_p, covariance_differences = covariance_bootstrap_pvalue(
        pooled_panel,
        w_flat,
        w_equal,
        bootstrap_indices,
        active_tolerance=config.active_weight_tolerance,
    )
    pooled_flat_covariance, _ = covariance_row_tv(
        pooled_panel,
        w_flat,
        active_tolerance=config.active_weight_tolerance,
    )
    pooled_equal_covariance, _ = covariance_row_tv(
        pooled_panel,
        w_equal,
        active_tolerance=config.active_weight_tolerance,
    )

    decisive_pass = bool(
        flat_rmse < equal_rmse
        and tracking_p < config.alpha
        and reduction >= config.minimum_rmse_reduction
    )
    systematic_random_pass = bool(
        systematic_random_passes >= config.minimum_passing_folds
    )
    covariance_random_pass = bool(
        covariance_random_passes >= config.minimum_passing_folds
    )
    covariance_equal_pass = bool(
        pooled_flat_covariance < pooled_equal_covariance and covariance_p < config.alpha
    )
    passed, outcome = classify_gate(
        decisive_pass=decisive_pass,
        systematic_random_pass=systematic_random_pass,
        covariance_random_pass=covariance_random_pass,
        covariance_equal_pass=covariance_equal_pass,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "outcome": outcome,
        "pre_registration_date": "2026-07-28",
        "operator": {
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "arm_id": config.barycentre_arm_id,
            "geometry": config.geometry_id,
            "construction_window": "2018-01-01/2022-12-31",
            "ticker_count": len(tickers),
            "artifact_summary": operator_summary,
        },
        "design": {
            "folds": GATE_FOLDS,
            "n_bootstrap": config.n_bootstrap,
            "n_random": config.n_random,
            "expected_block_length": config.block_length,
            "knn_k": config.knn_k,
            "seed": config.seed,
            "market_proxy": "equal_weight_cross_sectional_return",
            "exposure_test": "not_answerable_on_available_data",
        },
        "thresholds": {
            "alpha": config.alpha,
            "minimum_rmse_reduction": config.minimum_rmse_reduction,
            "minimum_passing_folds": config.minimum_passing_folds,
        },
        "folds": folds,
        "pooled": {
            "trading_days": len(pooled_panel),
            "w_flat_systematic_rmse": flat_rmse,
            "equal_support_systematic_rmse": equal_rmse,
            "systematic_rmse_reduction": reduction,
            "systematic_bootstrap_p_value": tracking_p,
            "systematic_bootstrap_difference_q025": float(
                np.quantile(tracking_differences, 0.025)
            ),
            "systematic_bootstrap_difference_q975": float(
                np.quantile(tracking_differences, 0.975)
            ),
            "diebold_mariano": diebold_mariano_agreement(
                pooled_flat,
                pooled_equal,
                max_lag=config.block_length,
            ),
            "w_flat_covariance_tv": pooled_flat_covariance,
            "equal_support_covariance_tv": pooled_equal_covariance,
            "covariance_bootstrap_p_value": covariance_p,
            "covariance_bootstrap_difference_q025": float(
                np.quantile(covariance_differences, 0.025)
            ),
            "covariance_bootstrap_difference_q975": float(
                np.quantile(covariance_differences, 0.975)
            ),
        },
        "decision": {
            "decisive_systematic_pass": decisive_pass,
            "systematic_random_floor_folds": systematic_random_passes,
            "systematic_random_floor_pass": systematic_random_pass,
            "covariance_random_floor_folds": covariance_random_passes,
            "covariance_random_floor_pass": covariance_random_pass,
            "covariance_equal_support_pass": covariance_equal_pass,
            "all_conditions_pass": passed,
            "consequence": (
                "economically_informative_weighting_operator"
                if passed
                else "text_derived_weighting_scheme_only"
            ),
        },
    }
    _write_results(paths.output_dir, payload, records)
    return payload


__all__ = ["classify_gate", "run_paper5_economic_gate"]

"""Orchestrate Paper 3's full-sample certificate-feasibility surface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never, cast

import numpy as np
import pandas as pd

from pipeline._kernels.paper3_certificate import (
    certificate_surface,
    equal_weights,
    inverse_volatility_weights,
)
from pipeline._kernels.paper3_validation import solve_certificate
from pipeline.io.paper3_certificate import sha256_path, write_certificate_artifacts
from pipeline.io.params import load_params
from pipeline.stages.papers.paper3.certificate_validation import _gmv
from pipeline.stages.substrate.panel import (
    load_aligned_return_panel,
    load_priced_distance_universe,
)

_MIN_STANDARDIZATION_OBSERVATIONS = 2
_MATRIX_NDIM = 2
_SUM_TOLERANCE = 1e-8
_MAX_CAP_PASSES = 32
_MIN_REFERENCE_DRAWS = 1000


@dataclass(frozen=True, slots=True)
class Paper3CertificatePaths:
    """Declared inputs and outputs for the feasibility stage."""

    params_file: Path
    returns_dir: Path
    distance_artifact_dir: Path
    output_dir: Path


def _value_error(message: str) -> Never:
    raise ValueError(message)


def _type_error(message: str) -> Never:
    raise TypeError(message)


def _number_grid(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        _value_error(f"{field} must be a nonempty list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        _type_error(f"{field} must contain only numbers")
    return tuple(float(item) for item in value)


def _code_identity() -> str:
    sources = (
        Path(__file__),
        Path(__file__).parents[3] / "_kernels" / "paper3_certificate.py",
        Path(__file__).parents[3] / "_kernels" / "paper3_validation.py",
        Path(__file__).parents[3] / "io" / "paper3_certificate.py",
    )
    payload = [(path.name, sha256_path(path)) for path in sources]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _standardize_returns(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each asset by its complete-panel sample standard deviation."""
    panel = np.asarray(returns, dtype=np.float64)
    if (
        panel.ndim != _MATRIX_NDIM
        or panel.shape[0] < _MIN_STANDARDIZATION_OBSERVATIONS
        or panel.shape[1] == 0
    ):
        _value_error("returns must contain at least two observations and one asset")
    if not np.isfinite(panel).all():
        _value_error("returns must contain only finite values")
    volatility = np.std(panel, axis=0, ddof=1)
    if not np.isfinite(volatility).all() or np.any(volatility <= 0.0):
        _value_error("every asset must have finite positive sample volatility")
    standardized = panel / volatility
    if not np.isfinite(standardized).all():
        _value_error("standardized returns must contain only finite values")
    return standardized, volatility


def _portfolio_variance(returns: np.ndarray, weights: np.ndarray) -> float:
    """Return a finite sample variance for a normalized long-only allocation."""
    panel = np.asarray(returns, dtype=np.float64)
    q = np.asarray(weights, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] != panel.shape[1]:
        _value_error("portfolio weights must match the return-panel roster")
    if not np.isfinite(q).all() or np.any(q < 0.0):
        _value_error("portfolio weights must be finite and nonnegative")
    if abs(float(q.sum()) - 1.0) > _SUM_TOLERANCE:
        _value_error("portfolio weights must sum to one")
    variance = float(np.var(panel @ q, ddof=1))
    if not np.isfinite(variance) or variance < 0.0:
        _value_error("portfolio variance must be finite and nonnegative")
    return variance


def _effective_number_assets(weights: np.ndarray) -> float:
    """Return inverse Herfindahl concentration for a simplex allocation."""
    q = np.asarray(weights, dtype=np.float64)
    concentration = float(q @ q)
    if not np.isfinite(concentration) or concentration <= 0.0:
        _value_error("portfolio concentration must be finite and positive")
    return 1.0 / concentration


@dataclass(frozen=True, slots=True)
class _ReferenceDesign:
    """One declared sampling law for the feasible-allocation reference set."""

    label: str
    concentration: float
    maximum_weight: float


def _reference_designs(value: object) -> tuple[_ReferenceDesign, ...]:
    """Parse the declared reference-population designs from params.yaml."""
    field = "paper3.certificate_feasibility.news_reference.designs"
    if not isinstance(value, list) or not value:
        _value_error(f"{field} must be a nonempty list")
    designs: list[_ReferenceDesign] = []
    for entry in value:
        if not isinstance(entry, dict):
            _type_error(f"{field} entries must be mappings")
        label = entry.get("label")
        concentration = entry.get("concentration")
        maximum_weight = entry.get("maximum_weight")
        if not isinstance(label, str) or not label:
            _value_error(f"{field} entries must declare a nonempty label")
        if (
            isinstance(concentration, bool)
            or not isinstance(concentration, (int, float))
            or float(concentration) <= 0.0
        ):
            _value_error(f"{field}.{label} concentration must be positive")
        if (
            isinstance(maximum_weight, bool)
            or not isinstance(maximum_weight, (int, float))
            or not 0.0 < float(maximum_weight) <= 1.0
        ):
            _value_error(f"{field}.{label} maximum_weight must lie in (0, 1]")
        designs.append(
            _ReferenceDesign(label, float(concentration), float(maximum_weight))
        )
    labels = [design.label for design in designs]
    if len(set(labels)) != len(labels):
        _value_error(f"{field} labels must be unique")
    return tuple(designs)


def _reference_draws(
    design: _ReferenceDesign, n_assets: int, draws: int, seed: int
) -> np.ndarray:
    """Draw capped long-only allocations from one declared reference design."""
    if n_assets * design.maximum_weight < 1.0:
        _value_error(
            f"reference design {design.label} cannot fill the simplex: "
            f"{n_assets} assets capped at {design.maximum_weight}"
        )
    rng = np.random.default_rng(seed)
    sample = rng.dirichlet(np.full(n_assets, design.concentration), size=draws)
    capped = np.minimum(sample, design.maximum_weight)
    normalized = capped / capped.sum(axis=1, keepdims=True)
    # Renormalizing after the cap can lift an entry back above it; iterate to a
    # fixed point so every retained draw is genuinely feasible.
    for _ in range(_MAX_CAP_PASSES):
        if normalized.max() <= design.maximum_weight + _SUM_TOLERANCE:
            break
        capped = np.minimum(normalized, design.maximum_weight)
        normalized = capped / capped.sum(axis=1, keepdims=True)
    feasible = normalized[
        normalized.max(axis=1) <= design.maximum_weight + _SUM_TOLERANCE
    ]
    if len(feasible) < _MIN_REFERENCE_DRAWS:
        _value_error(
            f"reference design {design.label} retained {len(feasible)} feasible draws"
        )
    if not np.isfinite(feasible).all() or np.any(feasible < 0.0):
        _value_error(f"reference design {design.label} produced invalid draws")
    return feasible


def _reference_percentiles(
    covariance: np.ndarray,
    candidates: dict[str, np.ndarray],
    designs: tuple[_ReferenceDesign, ...],
    draws: int,
    seed: int,
    *,
    cap_policy: Literal["error", "annotate"] = "error",
) -> list[dict[str, object]]:
    """Rank each candidate allocation inside every declared reference law."""
    n_assets = covariance.shape[0]
    rows: list[dict[str, object]] = []
    for index, design in enumerate(designs):
        infeasible = {
            name
            for name, weights in candidates.items()
            if weights.max() > design.maximum_weight + _SUM_TOLERANCE
        }
        if infeasible and cap_policy == "error":
            for name in sorted(infeasible):
                _value_error(
                    f"candidate {name} exceeds reference design {design.label} "
                    f"cap {design.maximum_weight}"
                )
        sample = _reference_draws(design, n_assets, draws, seed + index)
        variances = np.einsum("ij,jk,ik->i", sample, covariance, sample)
        if not np.isfinite(variances).all():
            _value_error(f"reference design {design.label} produced invalid variances")
        concentrations = np.einsum("ij,ij->i", sample, sample)
        rows.append(
            {
                "label": design.label,
                "concentration": design.concentration,
                "maximum_weight": design.maximum_weight,
                "retained_draws": len(sample),
                "mean_effective_number_assets": float(
                    np.mean(np.reciprocal(concentrations))
                ),
                "variance_quantiles": {
                    str(level): float(np.quantile(variances, level))
                    for level in (0.01, 0.05, 0.50)
                },
                "percentile": {
                    name: (
                        None
                        if name in infeasible
                        else float(
                            np.mean(variances <= float(weights @ covariance @ weights))
                        )
                    )
                    for name, weights in candidates.items()
                },
                "candidate_status": {
                    name: "infeasible_cap" if name in infeasible else "computed"
                    for name in candidates
                },
            }
        )
    return rows


def _news_only_benchmark(
    tickers: list[str],
    returns: np.ndarray,
    squared_w2: np.ndarray,
    designs: tuple[_ReferenceDesign, ...],
    reference_draws: int,
    reference_seed: int,
    *,
    cap_policy: Literal["error", "annotate"] = "error",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Construct and evaluate the descriptive zero-slack news allocation."""
    standardized, volatility = _standardize_returns(returns)
    news_solution = solve_certificate(squared_w2, cap=1.0)
    news_weights = np.asarray(news_solution.weights, dtype=np.float64)
    equal = equal_weights(len(tickers))
    equal_credit = 0.5 * float(equal @ squared_w2 @ equal)
    if news_solution.credit + 1e-10 < equal_credit:
        _value_error("news-only certificate objective is below equal-weight credit")
    sample_covariance = np.cov(standardized, rowvar=False, ddof=1)
    sample_gmv = _gmv(sample_covariance)
    variances = {
        "news_only": _portfolio_variance(standardized, news_weights),
        "equal_weight": _portfolio_variance(standardized, equal),
        "sample_gmv": _portfolio_variance(standardized, sample_gmv),
    }
    gmv_variance = variances["sample_gmv"]
    if gmv_variance <= 0.0:
        _value_error("sample-GMV standardized variance must be positive")
    weights = pd.DataFrame({"ticker": tickers, "weight": news_weights})
    if (
        len(weights) != len(tickers)
        or weights["ticker"].duplicated().any()
        or not np.isfinite(news_weights).all()
        or np.any(news_weights < 0.0)
        or abs(float(news_weights.sum()) - 1.0) > _SUM_TOLERANCE
    ):
        _value_error("news-only weights must be finite, nonnegative, and sum to one")
    benchmark: dict[str, object] = {
        "status": "full_sample_descriptive",
        "objective": {
            "name": "zero_slack_w2_certificate_credit",
            "cap": 1.0,
            "certificate_credit": news_solution.credit,
            "equal_weight_certificate_credit": equal_credit,
            "iterations": news_solution.iterations,
            "kkt_residual": news_solution.kkt_residual,
            "minimum_centered_kernel_eigenvalue": (
                news_solution.minimum_centered_kernel_eigenvalue
            ),
        },
        "standardization": {
            "method": "per_asset_sample_standard_deviation",
            "ddof": 1,
            "n_observations": int(standardized.shape[0]),
            "minimum_sample_volatility": float(volatility.min()),
            "maximum_sample_volatility": float(volatility.max()),
        },
        "portfolios": {
            "news_only": {
                "variance": variances["news_only"],
                "gmv_relative_variance": variances["news_only"] / gmv_variance,
                "certificate_credit": news_solution.credit,
                "maximum_weight": float(news_weights.max()),
                "effective_number_assets": _effective_number_assets(news_weights),
            },
            "equal_weight": {
                "variance": variances["equal_weight"],
                "gmv_relative_variance": variances["equal_weight"] / gmv_variance,
                "certificate_credit": equal_credit,
            },
            "sample_gmv": {
                "variance": gmv_variance,
                "gmv_relative_variance": 1.0,
            },
        },
        "reference_population": {
            "interpretation": (
                "In-sample rank of a return-free construction inside declared "
                "feasible-allocation laws; not coverage, calibration, or a "
                "chronological result."
            ),
            "requested_draws": reference_draws,
            "seed": reference_seed,
            "designs": _reference_percentiles(
                sample_covariance,
                {"news_only": news_weights, "equal_weight": equal},
                designs,
                reference_draws,
                reference_seed,
                cap_policy=cap_policy,
            ),
        },
    }
    return weights, benchmark


def run_paper3_certificate(paths: Paper3CertificatePaths) -> None:
    """Compute descriptive non-vacuity evidence from governed existing assets."""
    params = load_params(paths.params_file)
    paper3 = params.get("paper3")
    if not isinstance(paper3, dict):
        _type_error("params.yaml paper3 section must be a mapping")
    config = paper3.get("certificate_feasibility")
    if not isinstance(config, dict):
        _type_error("paper3.certificate_feasibility must be a mapping")
    scales = _number_grid(config.get("L"), "paper3.certificate_feasibility.L")
    slacks = _number_grid(config.get("tau"), "paper3.certificate_feasibility.tau")
    requested = config.get("portfolios")
    if requested != ["equal_weight", "inverse_volatility"]:
        _value_error(
            "paper3.certificate_feasibility.portfolios must declare "
            "equal_weight then inverse_volatility"
        )
    reference = config.get("news_reference")
    if not isinstance(reference, dict):
        _type_error("paper3.certificate_feasibility.news_reference must be a mapping")
    designs = _reference_designs(reference.get("designs"))
    reference_draws = reference.get("draws")
    reference_seed = reference.get("seed")
    if (
        isinstance(reference_draws, bool)
        or not isinstance(reference_draws, int)
        or reference_draws < _MIN_REFERENCE_DRAWS
    ):
        _value_error(
            "paper3.certificate_feasibility.news_reference.draws must be an "
            f"integer of at least {_MIN_REFERENCE_DRAWS}"
        )
    if isinstance(reference_seed, bool) or not isinstance(reference_seed, int):
        _value_error(
            "paper3.certificate_feasibility.news_reference.seed must be an integer"
        )

    tickers, distance, squared_distance = load_priced_distance_universe(
        paths.distance_artifact_dir, paths.returns_dir
    )
    dates, returns = load_aligned_return_panel(paths.returns_dir, tickers)
    portfolios = {
        "equal_weight": equal_weights(len(tickers)),
        "inverse_volatility": inverse_volatility_weights(returns),
    }
    cells = certificate_surface(distance, portfolios, scales, slacks)
    credit_values = [cell.certificate_credit for cell in cells]
    portfolio_credit = {
        portfolio: {
            "least_conservative": next(
                cell.certificate_credit
                for cell in cells
                if cell.portfolio == portfolio
                and cell.lipschitz_inverse_scale == scales[0]
                and cell.common_slack == slacks[0]
            ),
            "most_conservative": next(
                cell.certificate_credit
                for cell in cells
                if cell.portfolio == portfolio
                and cell.lipschitz_inverse_scale == scales[-1]
                and cell.common_slack == slacks[-1]
            ),
        }
        for portfolio in portfolios
    }
    scoped_params = {
        "L": list(scales),
        "tau": list(slacks),
        "portfolios": cast("list[str]", requested),
        "news_reference": {
            "draws": reference_draws,
            "seed": reference_seed,
            "designs": [
                {
                    "label": design.label,
                    "concentration": design.concentration,
                    "maximum_weight": design.maximum_weight,
                }
                for design in designs
            ],
        },
    }
    parameter_identity = hashlib.sha256(
        json.dumps(scoped_params, sort_keys=True).encode("utf-8")
    ).hexdigest()
    news_weights, news_benchmark = _news_only_benchmark(
        tickers,
        returns,
        squared_distance,
        designs,
        reference_draws,
        reference_seed,
    )
    summary: dict[str, object] = {
        "schema_version": "paper3_certificate.v2",
        "evidence_status": "full_sample_feasibility",
        "interpretation": (
            "Descriptive non-vacuity and sensitivity only; not historical calibration, "
            "sampling uncertainty, ex-ante validation, or portfolio performance."
        ),
        "universe": {
            "n_tickers": len(tickers),
            "n_return_observations": int(returns.shape[0]),
            "return_start": str(dates[0])[:10],
            "return_end": str(dates[-1])[:10],
            "tickers": tickers,
        },
        "scenario_grid": scoped_params,
        "certificate_credit": {
            "minimum": float(min(credit_values)),
            "maximum": float(max(credit_values)),
        },
        "portfolio_credit": portfolio_credit,
        "weight_construction": {
            "equal_weight": "one over the priced universe",
            "inverse_volatility": (
                "inverse sample standard deviation over the complete aligned return "
                "panel"
            ),
        },
        "distance_artifact": str(paths.distance_artifact_dir),
        "news_only_weights_artifact": str(
            paths.output_dir / "news_only_weights.parquet"
        ),
        "news_only_benchmark": news_benchmark,
    }
    write_certificate_artifacts(
        paths.output_dir,
        cells,
        summary,
        news_weights=news_weights,
        code_identity=_code_identity(),
        parameter_identity=parameter_identity,
        upstream_paths=(paths.distance_artifact_dir, paths.returns_dir),
    )

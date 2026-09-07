"""Frozen chronological validation for Paper 3's portfolio certificate."""

from __future__ import annotations

import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Never, cast

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.covariance import LedoitWolf

from pipeline._kernels.paper3_validation import (
    capital_to_budget_weights,
    coverage_lower,
    select_stationary_block_length,
    solve_certificate,
    stationary_block_indices,
)
from pipeline.io.contracts import write_manifest
from pipeline.io.typed_analysis import StatisticalDistanceSpec, load_typed_analysis
from pipeline.io.typed_providers import load_dated_provider_observations

PROVIDER_ID = "qwen3-embedding-8b"
REPRESENTATION_ID = "qwen3-embedding-8b-unit"
OUTPUT_NAMES = (
    "formation_results.parquet",
    "portfolio_weights.parquet",
    "random_portfolios.parquet",
    "placebos.parquet",
    "summary.yaml",
)
EXPECTED_ROSTER_SIZE = 100
WEIGHT_SUM_TOLERANCE = 1e-8


class CertificateValidationStageError(ValueError):
    """Raised when the governed chronological design cannot be reproduced."""


def _invalid(message: str) -> Never:
    raise CertificateValidationStageError(message)


@dataclass(frozen=True, slots=True)
class Paper3CertificateValidationPaths:
    """Declared inputs and outputs for the validation stage."""

    params_file: Path
    returns_dir: Path
    holdout_returns_dir: Path
    pit_w2_path: Path
    sample_descriptives_path: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Validated scalar protocol configuration."""

    lookback: int
    horizon: int
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    frozen_vintage: str
    scales: tuple[float, ...]
    slacks: tuple[float, ...]
    coverage_target: float
    confidence: float
    article_reps: int
    return_reps: int
    seed: int
    cap: float
    random_portfolios: int
    label_permutations: int
    residual_budgets: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def _load_config(path: Path) -> ValidationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["paper3"][
        "certificate_validation"
    ]
    if raw.get("formation_frequency") != "monthly":
        _invalid("paper3.certificate_validation must use monthly formation")
    return ValidationConfig(
        lookback=int(raw["volatility_lookback_days"]),
        horizon=int(raw["evaluation_horizon_days"]),
        calibration_start=pd.Timestamp(raw["calibration_start"]),
        calibration_end=pd.Timestamp(raw["calibration_end"]),
        evaluation_start=pd.Timestamp(raw["evaluation_start"]),
        evaluation_end=pd.Timestamp(raw["evaluation_end"]),
        frozen_vintage=str(raw["frozen_vintage"]),
        scales=tuple(float(value) for value in raw["L"]),
        slacks=tuple(float(value) for value in raw["tau"]),
        coverage_target=float(raw["coverage_target"]),
        confidence=float(raw["confidence_level"]),
        article_reps=int(raw["article_block_replications"]),
        return_reps=int(raw["return_block_replications"]),
        seed=int(raw["seed"]),
        cap=float(raw["maximum_weight"]),
        random_portfolios=int(raw["random_portfolios"]),
        label_permutations=int(raw["label_permutations"]),
        residual_budgets=tuple(float(value) for value in raw["residual_budgets"]),
    )


def _load_roster(path: Path) -> tuple[str, ...]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items = manifest.get("metadata", {}).get("item_ids", [])
    roster = tuple(str(item) for item in items)
    if len(roster) != EXPECTED_ROSTER_SIZE or len(set(roster)) != EXPECTED_ROSTER_SIZE:
        _invalid(
            f"PIT W2 manifest must declare the {EXPECTED_ROSTER_SIZE}-firm roster, "
            f"got {len(roster)}"
        )
    return roster


def _load_return_panel(
    calibration_dir: Path, holdout_dir: Path, tickers: tuple[str, ...]
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for ticker in tickers:
        pieces: list[pd.DataFrame] = []
        for root in (calibration_dir, holdout_dir):
            frame = pd.read_parquet(
                root / f"{ticker}.parquet", columns=["Date", "return"]
            )
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
            pieces.append(frame)
        joined = pd.concat(pieces, ignore_index=True).drop_duplicates(
            "Date", keep="last"
        )
        columns[ticker] = joined.set_index("Date")["return"]
    panel = pd.DataFrame(columns).sort_index().dropna(how="any")
    if panel.empty or not np.isfinite(panel.to_numpy()).all():
        _invalid("return panel is empty or nonfinite after complete-case alignment")
    return panel


def _monthly_formations(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int,
    horizon: int,
    horizon_end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, ...]:
    eligible = dates[(dates >= start) & (dates <= end)]
    first_by_month = (
        pd.Series(eligible, index=eligible.to_period("M")).groupby(level=0).min()
    )
    locations = {date: index for index, date in enumerate(dates)}
    formations: list[pd.Timestamp] = []
    for date in first_by_month:
        location = locations[date]
        if location < lookback or location + horizon > len(dates):
            continue
        if horizon_end is not None and dates[location + horizon - 1] > horizon_end:
            continue
        formations.append(date)
    return tuple(formations)


def _windows(
    panel: pd.DataFrame, formation: pd.Timestamp, config: ValidationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    location = panel.index.get_loc(formation)
    if not isinstance(location, int):
        _invalid("formation date is not unique in the aligned return panel")
    trailing = panel.iloc[location - config.lookback : location]
    future = panel.iloc[location : location + config.horizon]
    if trailing.index.max() >= formation or future.index.min() < formation:
        _invalid("strict timing contract was violated")
    return trailing, future


def _latest_vintage(vintages: tuple[str, ...], formation: pd.Timestamp) -> str:
    available = [value for value in vintages if pd.Timestamp(value) < formation]
    if not available:
        _invalid(f"no W2 vintage is strictly prior to {formation.date()}")
    return max(available)


def _floor_squared(squared_w2: np.ndarray, scale: float, slack: float) -> np.ndarray:
    floor = np.maximum(0.0, np.sqrt(squared_w2) / scale - 2.0 * slack)
    result = np.square(floor)
    np.fill_diagonal(result, 0.0)
    return result


def _credit(squared_floor: np.ndarray, budget_weights: np.ndarray) -> float:
    return 0.5 * float(budget_weights @ squared_floor @ budget_weights)


def _capital_from_budget(q: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    unscaled = q / volatility
    return unscaled / unscaled.sum()


def _gmv(covariance: np.ndarray) -> np.ndarray:
    n = len(covariance)
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.full(n, 1.0 / n),
        jac=lambda weights: 2.0 * covariance @ weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success or abs(result.x.sum() - 1.0) > WEIGHT_SUM_TOLERANCE:
        _invalid(f"GMV solver failed: {result.message}")
    return np.asarray(result.x)


def _portfolio_weights(
    trailing: np.ndarray, squared_floor: np.ndarray | None, cap: float
) -> tuple[dict[str, np.ndarray], dict[str, float | int | bool]]:
    volatility = np.std(trailing, axis=0, ddof=1)
    inverse = np.reciprocal(volatility)
    inverse /= inverse.sum()
    covariance = np.cov(trailing, rowvar=False, ddof=1)
    weights = {
        "equal_weight": np.full(trailing.shape[1], 1.0 / trailing.shape[1]),
        "inverse_volatility": inverse,
        "sample_gmv": _gmv(covariance),
        "ledoit_wolf_gmv": _gmv(LedoitWolf().fit(trailing).covariance_),
    }
    diagnostics: dict[str, float | int | bool] = {
        "success": True,
        "iterations": 0,
        "kkt_residual": 0.0,
        "minimum_centered_kernel_eigenvalue": float("nan"),
    }
    if squared_floor is None:
        weights["certificate"] = inverse.copy()
    else:
        solution = solve_certificate(squared_floor, cap)
        weights["certificate"] = _capital_from_budget(solution.weights, volatility)
        diagnostics = {
            "success": solution.success,
            "iterations": solution.iterations,
            "kkt_residual": solution.kkt_residual,
            "minimum_centered_kernel_eigenvalue": (
                solution.minimum_centered_kernel_eigenvalue
            ),
        }
    return weights, diagnostics


def _article_block_draws(
    params_file: Path,
    tickers: tuple[str, ...],
    vintages: tuple[str, ...],
    replications: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    registry = load_typed_analysis(params_file)
    provider = registry.providers[PROVIDER_ID]
    representation = registry.representations[REPRESENTATION_ID]
    draws_by_vintage: dict[str, np.ndarray] = {}
    lengths_by_vintage: dict[str, dict[str, object]] = {}
    for vintage in vintages:
        distance_id = f"wasserstein_w2_pit_{vintage[:4]}"
        distance = registry.distances.get(distance_id)
        if not isinstance(distance, StatisticalDistanceSpec):
            _invalid(f"missing registered distance {distance_id}")
        window = registry.window_for(distance)
        observations = load_dated_provider_observations(
            provider,
            representation,
            item_ids=tickers,
            sample_size=distance.sample_size,
            window=window,
        )
        arrays: list[np.ndarray] = []
        index_draws: list[np.ndarray] = []
        block_lengths: dict[str, object] = {}
        for observation in observations:
            order = np.lexsort((np.asarray(observation.url_hashes), observation.dates))
            values = observation.values[order]  # noqa: PD011 - dataclass array field
            centroid_distance = np.linalg.norm(values - values.mean(axis=0), axis=1)
            block_length = select_stationary_block_length(centroid_distance)
            arrays.append(values)
            block_lengths[observation.item_id] = {
                "block_length": block_length,
                "n_articles": len(values),
                "sample_identity": hashlib.sha256(
                    (
                        observation.source_hash
                        + "|"
                        + "|".join(observation.url_hashes[index] for index in order)
                    ).encode()
                ).hexdigest(),
            }
            index_draws.append(
                stationary_block_indices(
                    len(values),
                    block_length,
                    replications,
                    _seed(seed, vintage, observation.item_id),
                )
            )
        matrices = np.zeros(
            (replications, len(tickers), len(tickers)), dtype=np.float64
        )

        def pair_values(
            pair: tuple[int, int],
            arrays_current: list[np.ndarray] = arrays,
            draws_current: list[np.ndarray] = index_draws,
        ) -> tuple[int, int, np.ndarray]:
            left, right = pair
            costs = np.maximum(
                0.0,
                2.0 * (1.0 - arrays_current[left] @ arrays_current[right].T),
            )
            values = np.empty(replications, dtype=np.float64)
            for draw in range(replications):
                sampled = costs[
                    np.ix_(draws_current[left][draw], draws_current[right][draw])
                ]
                rows, columns = linear_sum_assignment(sampled)
                values[draw] = float(np.mean(sampled[rows, columns]))
            return left, right, values

        pairs = [
            (left, right)
            for left in range(len(tickers))
            for right in range(left + 1, len(tickers))
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            for left, right, values in executor.map(pair_values, pairs):
                matrices[:, left, right] = values
                matrices[:, right, left] = values
        draws_by_vintage[vintage] = matrices
        lengths_by_vintage[vintage] = block_lengths
    return draws_by_vintage, lengths_by_vintage


def _credit_lower(
    matrices: np.ndarray,
    q: np.ndarray,
    scale: float,
    slack: float,
    confidence: float,
    point_credit: float | None = None,
) -> float:
    floors = np.square(np.maximum(0.0, np.sqrt(matrices) / scale - 2.0 * slack))
    credit_draws = 0.5 * np.einsum("i,bij,j->b", q, floors, q, optimize=True)
    endpoint = float(np.quantile(credit_draws, 1.0 - confidence, method="linear"))
    return endpoint if point_credit is None else min(point_credit, endpoint)


def _return_bootstrap_variances(
    future: np.ndarray, weights: np.ndarray, replications: int, seed: int
) -> np.ndarray:
    portfolio = future @ weights
    block = select_stationary_block_length(portfolio)
    indices = stationary_block_indices(len(portfolio), block, replications, seed)
    return np.var(portfolio[indices], axis=1, ddof=1)


def _select_cell(
    formations: tuple[pd.Timestamp, ...],
    panel: pd.DataFrame,
    vintage_matrices: dict[str, np.ndarray],
    article_draws: dict[str, np.ndarray],
    config: ValidationConfig,
) -> tuple[float, float, list[dict[str, object]]]:
    vintages = tuple(sorted(vintage_matrices))
    evidence: list[dict[str, object]] = []
    for scale in config.scales:
        for slack in config.slacks:
            bounds: list[float] = []
            point_credits: list[float] = []
            lower_credits: list[float] = []
            boot_variances: list[np.ndarray] = []
            realized: list[float] = []
            for formation in formations:
                trailing, future = _windows(panel, formation, config)
                trailing_values = trailing.to_numpy()
                future_values = future.to_numpy()
                volatility = np.std(trailing_values, axis=0, ddof=1)
                capital = np.reciprocal(volatility)
                capital /= capital.sum()
                q = capital_to_budget_weights(capital, volatility)
                vintage = _latest_vintage(vintages, formation)
                floor = _floor_squared(vintage_matrices[vintage], scale, slack)
                point = _credit(floor, q)
                lower = _credit_lower(
                    article_draws[vintage],
                    q,
                    scale,
                    slack,
                    config.confidence,
                    point,
                )
                amplitude = float(capital @ volatility)
                bounds.append(amplitude**2 * (1.0 - lower))
                point_credits.append(point)
                lower_credits.append(lower)
                realized.append(float(np.var(future_values @ capital, ddof=1)))
                boot_variances.append(
                    _return_bootstrap_variances(
                        future_values,
                        capital,
                        config.return_reps,
                        _seed(config.seed, "calibration-return", formation),
                    )
                )
            bounds_array = np.asarray(bounds)
            realized_array = np.asarray(realized)
            bootstrap = np.column_stack(boot_variances)
            margin_draws = np.mean(bounds_array[None, :] - bootstrap, axis=1)
            coverage_draws = np.mean(bootstrap <= bounds_array[None, :], axis=1)
            margin_lower = float(np.quantile(margin_draws, 1.0 - config.confidence))
            coverage_interval_lower = float(
                np.quantile(coverage_draws, 1.0 - config.confidence)
            )
            evidence.append(
                {
                    "L": scale,
                    "tau": slack,
                    "mean_margin": float(np.mean(bounds_array - realized_array)),
                    "mean_margin_lower": margin_lower,
                    "coverage": float(np.mean(realized_array <= bounds_array)),
                    "coverage_lower": coverage_interval_lower,
                    "mean_point_credit": float(np.mean(point_credits)),
                    "mean_lower_credit": float(np.mean(lower_credits)),
                    "passes": bool(
                        margin_lower >= 0.0
                        and coverage_interval_lower >= config.coverage_target
                    ),
                }
            )
    passing = [row for row in evidence if row["passes"]]
    if not passing:
        return float("inf"), float("inf"), evidence
    passing.sort(
        key=lambda row: (
            -float(row["mean_lower_credit"]),
            float(row["L"]),
            float(row["tau"]),
        )
    )
    first = passing[0]
    return cast("float", first["L"]), cast("float", first["tau"]), evidence


def _sector_labels(path: Path, tickers: tuple[str, ...]) -> np.ndarray:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = {row["symbol"]: row["sector"] for row in raw["universe"]}
    if any(ticker not in mapping for ticker in tickers):
        _invalid("sample descriptives do not cover the validation roster")
    return np.asarray([mapping[ticker] for ticker in tickers])


def _permuted_matrix(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(permutation, permutation)]


def _random_capped_weights(n: int, count: int, cap: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    while len(accepted) < count:
        candidates = rng.dirichlet(np.ones(n), size=max(128, count - len(accepted)))
        accepted.extend(candidate for candidate in candidates if candidate.max() <= cap)
    return np.asarray(accepted[:count])


def _weight_stability(
    point_floor: np.ndarray | None,
    article_matrices: np.ndarray,
    selected: tuple[float, float],
    cap: float,
) -> dict[str, object]:
    """Re-optimize article draws without claiming portfolio-uniform coverage."""
    if point_floor is None:
        return {
            "successful_draws": 0,
            "failed_kernel_or_solver_draws": 0,
            "mean_l1_distance": 0.0,
            "mean_cosine_similarity": 1.0,
            "interpretation": "not run under the zero-credit fallback",
        }
    point = solve_certificate(point_floor, cap).weights
    l1: list[float] = []
    cosine: list[float] = []
    failures = 0
    for matrix in article_matrices:
        try:
            draw = solve_certificate(_floor_squared(matrix, *selected), cap).weights
        except ValueError:
            failures += 1
            continue
        l1.append(float(np.abs(draw - point).sum()))
        cosine.append(
            float(draw @ point / (np.linalg.norm(draw) * np.linalg.norm(point)))
        )
    return {
        "successful_draws": len(l1),
        "failed_kernel_or_solver_draws": failures,
        "mean_l1_distance": float(np.mean(l1)) if l1 else float("nan"),
        "mean_cosine_similarity": float(np.mean(cosine)) if cosine else float("nan"),
        "interpretation": (
            "conditional weight-stability diagnostic; not a portfolio-uniform "
            "finite-sample certificate"
        ),
    }


def _evaluate(  # noqa: C901, PLR0912, PLR0915
    formations: tuple[pd.Timestamp, ...],
    panel: pd.DataFrame,
    tickers: tuple[str, ...],
    point_matrix: np.ndarray,
    stale_matrix: np.ndarray,
    article_matrices: np.ndarray,
    sectors: np.ndarray,
    selected: tuple[float, float],
    config: ValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fallback = not np.isfinite(selected[0])
    squared_floor = None if fallback else _floor_squared(point_matrix, *selected)
    rng = np.random.default_rng(config.seed)
    ticker_permutation = rng.permutation(len(tickers))
    sector_permutation = np.arange(len(tickers))
    for sector in sorted(set(sectors)):
        members = np.flatnonzero(sectors == sector)
        sector_permutation[members] = rng.permutation(members)
    random_capital = _random_capped_weights(
        len(tickers), config.random_portfolios, config.cap, _seed(config.seed, "random")
    )
    roster_identity = hashlib.sha256("|".join(tickers).encode()).hexdigest()
    formation_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    previous: dict[str, np.ndarray] = {}
    evaluation_cache: list[tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]] = []
    for formation in formations:
        trailing, future = _windows(panel, formation, config)
        trailing_values = trailing.to_numpy()
        future_values = future.to_numpy()
        trailing_volatility = np.std(trailing_values, axis=0, ddof=1)
        future_volatility = np.std(future_values, axis=0, ddof=1)
        portfolios, solver = _portfolio_weights(
            trailing_values, squared_floor, config.cap
        )
        evaluation_cache.append(
            (formation, trailing_volatility, future_volatility, future_values)
        )
        for name, capital in portfolios.items():
            realized_variance = float(np.var(future_values @ capital, ddof=1))
            operational_q = capital_to_budget_weights(capital, trailing_volatility)
            geometry_q = capital_to_budget_weights(capital, future_volatility)
            if fallback:
                point_operational = lower_operational = 0.0
                point_geometry = lower_geometry = 0.0
            else:
                if squared_floor is None:
                    _invalid("certificate floor missing outside fallback")
                point_operational = _credit(squared_floor, operational_q)
                point_geometry = _credit(squared_floor, geometry_q)
                if name in {"equal_weight", "inverse_volatility", "certificate"}:
                    lower_operational = _credit_lower(
                        article_matrices,
                        operational_q,
                        *selected,
                        config.confidence,
                        point_operational,
                    )
                    lower_geometry = _credit_lower(
                        article_matrices,
                        geometry_q,
                        *selected,
                        config.confidence,
                        point_geometry,
                    )
                else:
                    lower_operational = point_operational
                    lower_geometry = point_geometry
            operational_amplitude = float(capital @ trailing_volatility)
            geometry_amplitude = float(capital @ future_volatility)
            operational_bound = operational_amplitude**2 * (1.0 - lower_operational)
            geometry_bound = geometry_amplitude**2 * (1.0 - lower_geometry)
            required_delta = max(
                0.0,
                realized_variance / operational_amplitude**2
                - (1.0 - lower_operational),
            )
            turnover = (
                float("nan")
                if name not in previous
                else 0.5 * float(np.abs(capital - previous[name]).sum())
            )
            previous[name] = capital
            formation_rows.append(
                {
                    "epoch": "evaluation",
                    "formation_date": formation,
                    "lookback_start": trailing.index[0],
                    "lookback_end": trailing.index[-1],
                    "evaluation_start": future.index[0],
                    "evaluation_end": future.index[-1],
                    "w2_vintage": config.frozen_vintage,
                    "portfolio": name,
                    "selected_L": selected[0],
                    "selected_tau": selected[1],
                    "zero_credit_fallback": fallback,
                    "point_credit_operational": point_operational,
                    "lower_credit_operational": lower_operational,
                    "point_credit_geometry": point_geometry,
                    "lower_credit_geometry": lower_geometry,
                    "operational_bound": operational_bound,
                    "geometry_only_bound": geometry_bound,
                    "realized_variance": realized_variance,
                    "operational_covered": realized_variance <= operational_bound,
                    "geometry_only_covered": realized_variance <= geometry_bound,
                    "required_delta": required_delta,
                    **{
                        f"covered_delta_{delta:g}": (
                            realized_variance
                            <= operational_amplitude**2
                            * (1.0 - lower_operational + delta)
                        )
                        for delta in config.residual_budgets
                    },
                    "turnover": turnover,
                    "concentration": float(capital @ capital),
                    "solver_success": solver["success"]
                    if name == "certificate"
                    else True,
                    "solver_iterations": solver["iterations"]
                    if name == "certificate"
                    else 0,
                    "solver_kkt_residual": solver["kkt_residual"]
                    if name == "certificate"
                    else 0.0,
                    "kernel_minimum_eigenvalue": solver[
                        "minimum_centered_kernel_eigenvalue"
                    ]
                    if name == "certificate"
                    else float("nan"),
                    "sample_identity": hashlib.sha256(
                        (
                            f"{roster_identity}|{formation.date()}|"
                            f"{trailing.index[0].date()}|{trailing.index[-1].date()}|"
                            f"{future.index[0].date()}|{future.index[-1].date()}"
                        ).encode()
                    ).hexdigest(),
                }
            )
            for ticker, capital_weight, budget_weight in zip(
                tickers, capital, operational_q, strict=True
            ):
                weight_rows.append(
                    {
                        "formation_date": formation,
                        "portfolio": name,
                        "ticker": ticker,
                        "capital_weight": capital_weight,
                        "budget_weight": budget_weight,
                    }
                )
        for portfolio_id, capital in enumerate(random_capital):
            q = capital_to_budget_weights(capital, trailing_volatility)
            if fallback:
                credit = 0.0
            else:
                if squared_floor is None:
                    _invalid("certificate floor missing outside fallback")
                credit = _credit(squared_floor, q)
            amplitude = float(capital @ trailing_volatility)
            random_rows.append(
                {
                    "formation_date": formation,
                    "portfolio_id": portfolio_id,
                    "certificate_credit": credit,
                    "operational_bound": amplitude**2 * (1.0 - credit),
                    "realized_variance": float(np.var(future_values @ capital, ddof=1)),
                    "maximum_weight": float(capital.max()),
                }
            )
    formation_frame = pd.DataFrame(formation_rows)
    weight_frame = pd.DataFrame(weight_rows)
    random_frame = pd.DataFrame(random_rows)

    placebo_rows: list[dict[str, object]] = []
    placebo_matrices = {
        "ticker": _permuted_matrix(point_matrix, ticker_permutation),
        "within_sector": _permuted_matrix(point_matrix, sector_permutation),
        "stale_2018": stale_matrix,
    }
    for name, matrix in placebo_matrices.items():
        floor = np.zeros_like(matrix) if fallback else _floor_squared(matrix, *selected)
        for (
            formation,
            trailing_volatility,
            _future_volatility,
            future_values,
        ) in evaluation_cache:
            capital = np.reciprocal(trailing_volatility)
            capital /= capital.sum()
            q = capital_to_budget_weights(capital, trailing_volatility)
            placebo_rows.append(
                {
                    "placebo": name,
                    "replication": 0,
                    "formation_date": formation,
                    "certificate_credit": _credit(floor, q),
                    "realized_variance": float(np.var(future_values @ capital, ddof=1)),
                }
            )
    for replication in range(config.label_permutations):
        permutation = rng.permutation(len(tickers))
        floor = (
            np.zeros_like(point_matrix)
            if fallback
            else _floor_squared(_permuted_matrix(point_matrix, permutation), *selected)
        )
        permutation_credits = []
        for (
            _formation,
            trailing_volatility,
            _future_volatility,
            _future_values,
        ) in evaluation_cache:
            capital = np.reciprocal(trailing_volatility)
            capital /= capital.sum()
            permutation_credits.append(
                _credit(floor, capital_to_budget_weights(capital, trailing_volatility))
            )
        placebo_rows.append(
            {
                "placebo": "label_permutation",
                "replication": replication,
                "formation_date": pd.NaT,
                "certificate_credit": float(np.mean(permutation_credits)),
                "realized_variance": float("nan"),
            }
        )
    placebo_frame = pd.DataFrame(placebo_rows)

    primary = formation_frame[
        formation_frame["portfolio"].isin(["certificate", "inverse_volatility"])
    ]
    validity: dict[str, object] = {}
    for mode, column in (
        ("operational", "operational_covered"),
        ("geometry_only", "geometry_only_covered"),
    ):
        subset = formation_frame[formation_frame["portfolio"] == "inverse_volatility"]
        successes = int(subset[column].sum())
        validity[mode] = {
            "coverage": successes / len(subset),
            "coverage_lower": coverage_lower(successes, len(subset), config.confidence),
            "formations": len(subset),
        }
    means = primary.groupby("portfolio", sort=True)["realized_variance"].mean()
    summary: dict[str, object] = {
        "validity": validity,
        "informativeness": {
            "mean_lower_credit_inverse_volatility": float(
                formation_frame.loc[
                    formation_frame["portfolio"] == "inverse_volatility",
                    "lower_credit_operational",
                ].mean()
            ),
            "floor_activation_rate": float(np.mean(squared_floor > 0.0))
            if squared_floor is not None
            else 0.0,
            "positive_floor_quantiles": (
                {
                    str(q): float(np.quantile(squared_floor[squared_floor > 0.0], q))
                    for q in (0.1, 0.5, 0.9)
                }
                if squared_floor is not None and np.any(squared_floor > 0.0)
                else {"0.1": 0.0, "0.5": 0.0, "0.9": 0.0}
            ),
        },
        "decision_value": {
            "mean_future_variance": {key: float(value) for key, value in means.items()},
            "certificate_minus_inverse_volatility": float(
                means.get("certificate", np.nan)
                - means.get("inverse_volatility", np.nan)
            ),
            "mean_required_delta": float(
                formation_frame.loc[
                    formation_frame["portfolio"] == "inverse_volatility",
                    "required_delta",
                ].mean()
            ),
        },
        "diagnostics": {
            "weight_stability": _weight_stability(
                squared_floor, article_matrices, selected, config.cap
            ),
            "mean_turnover": {
                str(key): float(value)
                for key, value in formation_frame.groupby("portfolio")["turnover"]
                .mean()
                .items()
            },
            "mean_concentration": {
                str(key): float(value)
                for key, value in formation_frame.groupby("portfolio")["concentration"]
                .mean()
                .items()
            },
        },
    }
    return formation_frame, weight_frame, random_frame, placebo_frame, summary


def _write_outputs(
    paths: Paper3CertificateValidationPaths,
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    summary: dict[str, object],
    config: ValidationConfig,
    tickers: tuple[str, ...],
    selected: tuple[float, float],
    calibration_evidence: list[dict[str, object]],
    block_lengths: dict[str, dict[str, object]],
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    names = OUTPUT_NAMES[:4]
    for name, frame in zip(names, frames, strict=True):
        ordered = (
            frame.sort_values(
                list(frame.columns[: min(3, len(frame.columns))]), kind="mergesort"
            )
            if not frame.empty
            else frame
        )
        ordered.to_parquet(paths.output_dir / name, index=False)
    document = {
        "schema_version": "paper3_certificate_validation.v1",
        "evidence_status": "frozen_chronological_validation",
        "selected_cell": {
            "L": None if not np.isfinite(selected[0]) else selected[0],
            "tau": None if not np.isfinite(selected[1]) else selected[1],
            "zero_credit_fallback": not np.isfinite(selected[0]),
        },
        "protocol": {
            "calibration": [
                str(config.calibration_start.date()),
                str(config.calibration_end.date()),
            ],
            "evaluation": [
                str(config.evaluation_start.date()),
                str(config.evaluation_end.date()),
            ],
            "lookback_days": config.lookback,
            "horizon_days": config.horizon,
            "article_replications": config.article_reps,
            "return_replications": config.return_reps,
            "seed": config.seed,
            "maximum_weight": config.cap,
            "random_portfolios": config.random_portfolios,
            "label_permutations": config.label_permutations,
            "residual_budgets": list(config.residual_budgets),
        },
        "universe": {"n_tickers": len(tickers), "tickers": list(tickers)},
        "calibration_cells": calibration_evidence,
        "article_block_lengths": block_lengths,
        **summary,
    }
    summary_path = paths.output_dir / "summary.yaml"
    summary_path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    outputs = [paths.output_dir / name for name in OUTPUT_NAMES]
    source_paths = (
        Path(__file__),
        Path(__file__).parents[3] / "_kernels" / "paper3_validation.py",
    )
    code_identity = hashlib.sha256(
        "".join(_sha256(path) for path in source_paths).encode()
    ).hexdigest()
    write_manifest(
        paths.output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "p3_certificate_validation",
            "lane": "p3",
            "phase": "evaluate",
            "protocol_id": "paper3.certificate-validation.v1",
            "code_identity": {"kind": "sha256", "value": code_identity},
            "parameter_identity": {
                "kind": "sha256",
                "value": _sha256(paths.params_file),
            },
            "seed_policy": f"numpy-pcg64-derived-from-{config.seed}",
            "upstream_artifacts": [
                {"kind": "path-sha256", "value": f"{path}:{_sha256(path)}"}
                for path in (
                    paths.params_file,
                    paths.pit_w2_path,
                    paths.sample_descriptives_path,
                )
            ]
            + [
                {"kind": "path", "value": str(path)}
                for path in (
                    paths.returns_dir,
                    paths.holdout_returns_dir,
                    Path("data/shared/embeddings/qwen3-embedding-8b"),
                )
            ],
            "environment": {
                "python": platform.python_version(),
                "compute_dtype": "float64",
            },
            "outputs": [
                {
                    "path": str(path),
                    "kind": path.suffix.lstrip("."),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in outputs
            ],
            "risk_class": "high",
            "metadata": {
                "formation_result_rows": len(frames[0]),
                "weight_rows": len(frames[1]),
                "random_rows": len(frames[2]),
                "placebo_rows": len(frames[3]),
                "holdout_isolation": "selected_cell computed before evaluation rows",
                "ticker_roster_sha256": hashlib.sha256(
                    "|".join(tickers).encode()
                ).hexdigest(),
            },
        },
    )


def run_paper3_certificate_validation(paths: Paper3CertificateValidationPaths) -> None:
    """Run calibration, freeze the cell, and evaluate the untouched holdout."""
    config = _load_config(paths.params_file)
    tickers = _load_roster(paths.pit_w2_path.with_name("provenance.manifest.json"))
    panel = _load_return_panel(paths.returns_dir, paths.holdout_returns_dir, tickers)
    calibration = _monthly_formations(
        pd.DatetimeIndex(panel.index),
        config.calibration_start,
        config.calibration_end,
        config.lookback,
        config.horizon,
        config.calibration_end,
    )
    evaluation = _monthly_formations(
        pd.DatetimeIndex(panel.index),
        config.evaluation_start,
        config.evaluation_end,
        config.lookback,
        config.horizon,
        config.evaluation_end,
    )
    if not calibration or not evaluation or evaluation[0] != config.evaluation_start:
        _invalid("declared calibration/evaluation formation schedule is unavailable")
    with np.load(paths.pit_w2_path, allow_pickle=False) as archive:
        vintage_matrices = {
            key: np.asarray(archive[key], dtype=np.float64) for key in archive.files
        }
    if config.frozen_vintage not in vintage_matrices:
        _invalid("frozen 2022 W2 vintage is missing")
    article_draws, block_lengths = _article_block_draws(
        paths.params_file,
        tickers,
        tuple(sorted(vintage_matrices)),
        config.article_reps,
        config.seed,
    )
    selected_l, selected_tau, calibration_evidence = _select_cell(
        calibration, panel, vintage_matrices, article_draws, config
    )
    selected = (selected_l, selected_tau)
    sectors = _sector_labels(paths.sample_descriptives_path, tickers)
    formation, weights, random, placebos, summary = _evaluate(
        evaluation,
        panel,
        tickers,
        vintage_matrices[config.frozen_vintage],
        vintage_matrices[min(vintage_matrices)],
        article_draws[config.frozen_vintage],
        sectors,
        selected,
        config,
    )
    _write_outputs(
        paths,
        (formation, weights, random, placebos),
        summary,
        config,
        tickers,
        selected,
        calibration_evidence,
        block_lengths,
    )


__all__ = [
    "CertificateValidationStageError",
    "Paper3CertificateValidationPaths",
    "run_paper3_certificate_validation",
]

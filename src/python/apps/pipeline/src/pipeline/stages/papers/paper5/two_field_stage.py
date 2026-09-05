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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.figures.paper5.figures import TwoFieldRegion, render_two_field_region
from pipeline.stages.papers.paper5._gate.bootstrap import stationary_bootstrap_indices
from pipeline.stages.papers.paper5._gate.contracts import (
    GATE_FOLDS,
    POST_CORPUS_END_DATE,
    POST_CORPUS_START_DATE,
)
from pipeline.stages.papers.paper5._gate.driver import _load_gate_operator
from pipeline.stages.papers.paper5._run.robustness import _load_co_mentions_w
from pipeline.stages.papers.paper5.field_overlap import (
    induced_field_similarity,
    matched_random_induced_null,
    matched_random_support_null,
    support_overlap,
    weight_similarity,
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

    from pipeline.stages.papers.paper5.two_field import (
        NestedQLRBootstrap,
        TwoFieldProblem,
    )

_POOLED_PERIOD = "pooled_2023_2026"
_OPERATOR_WINDOW = "2018-01-01/2022-12-31"
_LOWER_QUANTILE = 0.025
_UPPER_QUANTILE = 0.975
# Coarse profile resolution retained for the joint-region exhibit only; the
# reported estimates always come from the full-resolution profile above.
_SURFACE_THETA_POINTS = 81
_SURFACE_RHO_POINTS = 81
_QLR_METHOD = "null-imposed stationary residual-block bootstrap"


@dataclass(frozen=True)
class Paper5TwoFieldPaths:
    """Input and output boundaries for the two-field stage."""

    returns_dir: Path
    oos_returns_dir: Path
    shared_barycentre_dir: Path
    co_mentions_adjacency: Path
    output_dir: Path
    figures_dir: Path | None = None


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


def _bootstrap_summary(draws: NDArray[np.float64]) -> dict[str, float]:
    """Summarize the joint-date refits as marginal intervals and a joint shape."""
    rho_b, rho_n, theta = draws[:, 0], draws[:, 1], draws[:, 2]
    total = rho_b + rho_n
    summary = {
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
        news_test = nested_qlr_bootstrap(
            returns,
            joint_problem,
            boundary_problems["w_flat"],
            indices,
        )
        distributional_test = nested_qlr_bootstrap(
            returns,
            joint_problem,
            boundary_problems["w_co_mentions"],
            indices,
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


def _write_exhibit(
    exhibit: _ExhibitInputs,
    result: dict[str, object],
    figures_dir: Path | None,
) -> None:
    """Emit the joint-region exhibit's profile surface, cloud, and figure."""
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
    if figures_dir is None:
        return
    joint = result["joint"]
    boundaries = result["boundaries"]
    if not isinstance(joint, dict) or not isinstance(boundaries, dict):
        message = "the exhibit needs a joint fit and both nested boundaries"
        raise TypeError(message)
    figures_dir.mkdir(parents=True, exist_ok=True)
    render_two_field_region(
        TwoFieldRegion(
            surface_rho_b=rho_b,
            surface_rho_n=rho_n,
            surface_loglik=loglik,
            bootstrap_rho_b=draws[:, 0],
            bootstrap_rho_n=draws[:, 1],
            joint_rho_b=joint["rho_b"],
            joint_rho_n=joint["rho_n"],
            field_b_boundary=boundaries["w_flat"]["rho_b"],
            field_n_boundary=boundaries["w_co_mentions"]["rho_n"],
        ),
        figures_dir / "two_field_region.pgf",
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
    tickers, w_flat, _summary = _load_gate_operator(paths.shared_barycentre_dir, config)
    w_news, _diagnostics = _load_co_mentions_w(paths.co_mentions_adjacency, tickers)

    periods = {
        _POOLED_PERIOD: (POST_CORPUS_START_DATE, POST_CORPUS_END_DATE),
        **GATE_FOLDS,
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
                result,
                paths.figures_dir,
            )
        result["start"] = str(pd.Timestamp(dates.min()).date())
        result["end"] = str(pd.Timestamp(dates.max()).date())
        result["trading_days"] = len(returns)
        results[period] = result
        rows.append(_period_row(period, result))

    payload: dict[str, object] = {
        "schema_version": 2,
        "primary_period": _POOLED_PERIOD,
        "annual_periods": list(GATE_FOLDS),
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
            },
        },
        "overlap": overlap,
        "results": results,
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "results.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(paths.output_dir / "results.parquet", index=False)
    return payload


__all__ = [
    "Paper5TwoFieldConfig",
    "Paper5TwoFieldPaths",
    "run_paper5_two_field_qmle",
]

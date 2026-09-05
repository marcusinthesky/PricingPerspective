"""Pooled and non-overlapping annual post-corpus SAR QMLE estimates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5 import weights as wgt
from pipeline.stages.papers.paper5._gate.bootstrap import stationary_bootstrap_indices
from pipeline.stages.papers.paper5._gate.contracts import (
    GATE_FOLDS,
    POST_CORPUS_END_DATE,
    POST_CORPUS_START_DATE,
)
from pipeline.stages.papers.paper5._gate.driver import _load_gate_operator
from pipeline.stages.papers.paper5._gate.metrics import equal_support_weights
from pipeline.stages.papers.paper5._run.robustness import _load_co_mentions_w
from pipeline.stages.substrate.panel import (
    load_aligned_return_panel,
    load_priced_distance_universe,
)

if TYPE_CHECKING:
    from pathlib import Path

_RHO_BOUNDARY_DIAGNOSTIC = 0.998


@dataclass(frozen=True)
class Paper5PostCorpusPaths:
    """Input and output boundaries for the post-corpus QMLE stage."""

    returns_dir: Path
    oos_returns_dir: Path
    shared_barycentre_dir: Path
    distance_artifact_dir: Path
    co_mentions_adjacency: Path
    output_dir: Path


@dataclass(frozen=True)
class Paper5PostCorpusConfig:
    """Frozen operator identity and dependent-date bootstrap controls."""

    n_bootstrap: int = 2000
    block_length: int = 21
    seed: int = 0
    active_weight_tolerance: float = 1e-8
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    barycentre_arm_id: str = "wasserstein_w2_loo"
    geometry_id: str = "wasserstein_w2"
    pooled_only: bool = False
    focal_matrices_only: bool = False
    output_bootstrap: Path | None = None


def _bootstrap_rho(
    returns: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Refit point-only QMLE on every joint-date bootstrap draw."""
    draws = np.empty(len(indices), dtype=np.float64)
    for draw, date_indices in enumerate(indices):
        fit = est.sar_qmle(returns[date_indices], weights, compute_se=False)
        draws[draw] = fit["rho_hat"]
    return draws


def _plain(value: object) -> object:
    """Convert NumPy values for deterministic JSON serialization."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run_paper5_post_corpus_qmle(
    paths: Paper5PostCorpusPaths,
    config: Paper5PostCorpusConfig,
) -> dict[str, object]:
    """Estimate pooled 2023--2026 QMLE and four annual stability cells."""
    if config.n_bootstrap < 1 or config.block_length < 1:
        message = "bootstrap draws and block length must be positive"
        raise ValueError(message)
    tickers, w_flat, _summary = _load_gate_operator(
        paths.shared_barycentre_dir,
        config,
    )
    distance_tickers, _distance, squared_distance = load_priced_distance_universe(
        paths.distance_artifact_dir, paths.returns_dir
    )
    if distance_tickers != tickers:
        message = (
            "post-corpus barycentre and W2 distance artifacts use different tickers"
        )
        raise ValueError(message)
    w_h, bandwidth_h = wgt.build_w_kernel(squared_distance, tickers)
    matrices: dict[str, np.ndarray] = {
        "w_flat": w_flat,
        "w_h": w_h,
    }
    if not config.focal_matrices_only:
        matrices.update(
            {
                "equal_support": equal_support_weights(
                    w_flat,
                    active_tolerance=config.active_weight_tolerance,
                ),
                "w_co_mentions": _load_co_mentions_w(
                    paths.co_mentions_adjacency,
                    tickers,
                )[0],
            }
        )
    periods = {
        "pooled_2023_2026": (POST_CORPUS_START_DATE, POST_CORPUS_END_DATE),
    }
    if not config.pooled_only:
        periods.update(GATE_FOLDS)
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    results: dict[str, object] = {}
    for period_index, (period, (start, end)) in enumerate(periods.items()):
        dates, returns = load_aligned_return_panel(
            paths.returns_dir,
            tickers,
            start,
            end,
            oos_returns_dir=paths.oos_returns_dir,
        )
        indices = stationary_bootstrap_indices(
            len(returns),
            draws=config.n_bootstrap,
            expected_block_length=config.block_length,
            seed=config.seed + period_index,
        )
        period_results: dict[str, object] = {}
        for name, weights in matrices.items():
            fit = est.sar_qmle(returns, weights)
            bootstrap = _bootstrap_rho(returns, weights, indices)
            bootstrap_rows.extend(
                {
                    "period": period,
                    "w_variant": name,
                    "draw_id": draw_id,
                    "rho_draw": float(value),
                    "solve_ok": bool(np.isfinite(value)),
                }
                for draw_id, value in enumerate(bootstrap)
            )
            record: dict[str, object] = {
                "period": period,
                "start": str(pd.Timestamp(dates.min()).date()),
                "end": str(pd.Timestamp(dates.max()).date()),
                "trading_days": len(returns),
                "w_variant": name,
                "rho_hat": fit["rho_hat"],
                "rho_ci_lower": float(np.quantile(bootstrap, 0.025)),
                "rho_ci_upper": float(np.quantile(bootstrap, 0.975)),
                "rho_hessian_se": fit["rho_se"],
                "alpha_hat": fit["alpha_hat"],
                "sigma2_hat": fit["sigma2_hat"],
                "loglik": fit["loglik"],
                "converged": bool(fit["converged"]),
                "bootstrap_boundary_share": float(
                    np.mean(np.abs(bootstrap) >= _RHO_BOUNDARY_DIAGNOSTIC)
                ),
            }
            rows.append(record)
            period_results[name] = record
        results[period] = period_results

    payload: dict[str, object] = {
        "schema_version": 3,
        "primary_period": "pooled_2023_2026",
        "annual_periods": [] if config.pooled_only else list(GATE_FOLDS),
        "operator_construction_window": "2018-01-01/2022-12-31",
        "identity": {
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "barycentre_arm_id": config.barycentre_arm_id,
            "geometry_id": config.geometry_id,
        },
        "operators": {
            "matrix_ids": list(matrices),
            "w_h": {
                "bandwidth_h": bandwidth_h,
                "bandwidth_rule": "median off-diagonal squared W2 distance",
                "distance_artifact": str(paths.distance_artifact_dir),
            },
        },
        "inference": {
            "method": "joint-date stationary bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "expected_block_length": config.block_length,
            "seed": config.seed,
            "hessian_se_role": "diagnostic_only",
            "bootstrap_persistence": (
                "common schedule persisted for paired contrasts"
                if config.output_bootstrap is not None
                else "not requested"
            ),
        },
        "results": results,
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "results.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(paths.output_dir / "results.parquet", index=False)
    if config.output_bootstrap is not None:
        config.output_bootstrap.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bootstrap_rows).to_parquet(config.output_bootstrap, index=False)
    return payload


__all__ = [
    "Paper5PostCorpusConfig",
    "Paper5PostCorpusPaths",
    "run_paper5_post_corpus_qmle",
]

"""Spatial-weight construction and sensitivity variants for Paper 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

from pipeline.io.barycentre_artifacts import read_target_projection_artifact
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5 import hypotheses as hyp
from pipeline.stages.papers.paper5 import weights as wgt
from pipeline.stages.papers.paper5._run.contracts import Paper5EmpiricalError
from pipeline.stages.papers.paper5.energy_radius import stat_radius_matrix

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(frozen=True)
class _FlatDiagnostic:
    """Minimal diagnostics adapter required by the perturbation bound."""

    tangent_curvature_min: float
    converged: bool
    assignment_gap_min: float
    assignment_support_size: int


class _CoMentionDiagnostics(TypedDict):
    """Validated metadata emitted with the co-mention comparator matrix."""

    construction: str
    retained_undirected_pairs: int
    retained_pair_weight: float
    row_stochastic_max_dev: float
    zero_diagonal_max_abs: float
    certified: bool


def _load_co_mentions_w(
    adjacency_path: Path, tickers: list[str]
) -> tuple[np.ndarray, _CoMentionDiagnostics]:
    """Load persistent co-mention counts as a row-stochastic comparator.

    The shared artifact stores each undirected pair once. Paper 5 intersects
    those edges with its priced universe, mirrors the weights, and normalizes
    each row. Edges to the one shared-substrate ticker outside the priced
    universe are dropped rather than folded into row mass.
    """
    edges = pd.read_parquet(adjacency_path)
    required = {"firm_i", "firm_j", "weight"}
    missing = required.difference(edges.columns)
    if missing:
        message = f"co-mention adjacency missing columns {sorted(missing)}"
        raise Paper5EmpiricalError(message)

    selected = edges.loc[:, ["firm_i", "firm_j", "weight"]].copy()
    if selected[["firm_i", "firm_j"]].isna().any().any():
        message = "co-mention adjacency contains missing firm identifiers"
        raise Paper5EmpiricalError(message)
    selected["firm_i"] = selected["firm_i"].astype(str).str.upper()
    selected["firm_j"] = selected["firm_j"].astype(str).str.upper()
    edge_weights = selected["weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(edge_weights)) or np.any(edge_weights <= 0.0):
        message = "co-mention adjacency weights must be finite and strictly positive"
        raise Paper5EmpiricalError(message)
    if bool((selected["firm_i"] == selected["firm_j"]).any()):
        message = "co-mention adjacency must not contain self-edges"
        raise Paper5EmpiricalError(message)

    pair_keys = selected.apply(
        lambda row: tuple(sorted((row["firm_i"], row["firm_j"]))), axis=1
    )
    if bool(pair_keys.duplicated().any()):
        message = "co-mention adjacency contains duplicate undirected pairs"
        raise Paper5EmpiricalError(message)

    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    matrix = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    retained_pairs = 0
    retained_weight = 0.0
    for left_name, right_name, edge_weight in zip(
        selected["firm_i"].astype(str),
        selected["firm_j"].astype(str),
        edge_weights,
        strict=True,
    ):
        left = ticker_index.get(left_name)
        right = ticker_index.get(right_name)
        if left is None or right is None:
            continue
        weight = float(edge_weight)
        matrix[left, right] = weight
        matrix[right, left] = weight
        retained_pairs += 1
        retained_weight += weight

    row_sums = matrix.sum(axis=1)
    isolated = [
        ticker for ticker, total in zip(tickers, row_sums, strict=True) if total <= 0
    ]
    if isolated:
        message = f"co-mention adjacency isolates priced tickers {isolated}"
        raise Paper5EmpiricalError(message)
    matrix /= row_sums[:, None]
    diagnostics: _CoMentionDiagnostics = {
        "construction": "persistent_weighted_news_co_mentions",
        "retained_undirected_pairs": retained_pairs,
        "retained_pair_weight": retained_weight,
        "row_stochastic_max_dev": float(np.max(np.abs(matrix.sum(axis=1) - 1.0))),
        "zero_diagonal_max_abs": float(np.max(np.abs(np.diag(matrix)))),
        "certified": False,
    }
    return matrix, diagnostics


def _load_shared_w_flat(
    artifact_dir: Path,
    tickers: list[str],
    *,
    provider_id: str,
    representation_id: str,
    arm_id: str,
    geometry_id: str,
) -> tuple[np.ndarray, list[_FlatDiagnostic]]:
    """Load row-stochastic Paper 5 weights from the shared artifact."""
    weights, diagnostics, _ = read_target_projection_artifact(
        artifact_dir,
        expected_identity={
            "arm_id": arm_id,
            "provider_id": provider_id,
            "representation_id": representation_id,
            "geometry": geometry_id,
            "feasible_set": "simplex_nonnegative",
        },
    )
    w_flat = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    flat_diagnostics: list[_FlatDiagnostic] = []
    for target_index, ticker in enumerate(tickers):
        target_rows = weights.loc[weights["target"] == ticker].sort_values(
            "candidate_index", kind="mergesort"
        )
        expected_candidates = [
            candidate for candidate in tickers if candidate != ticker
        ]
        if target_rows["candidate"].tolist() != expected_candidates:
            message = f"shared barycentre candidate ordering disagrees for {ticker}"
            raise Paper5EmpiricalError(message)
        candidate_indices = [
            candidate_index
            for candidate_index in range(len(tickers))
            if candidate_index != target_index
        ]
        if int(target_rows["target_index"].iloc[0]) != target_index:
            message = f"shared barycentre target ordering disagrees for {ticker}"
            raise Paper5EmpiricalError(message)
        w_flat[target_index, candidate_indices] = target_rows["weight"].to_numpy(
            dtype=np.float64
        )
        target_diagnostic = diagnostics.loc[diagnostics["target_id"] == ticker]
        if len(target_diagnostic) != 1:
            message = f"shared barycentre diagnostics missing target {ticker}"
            raise Paper5EmpiricalError(message)
        curvature = target_diagnostic.iloc[0].get("tangent_curvature_min")
        if curvature is None or not np.isfinite(float(curvature)):
            message = f"shared barycentre curvature missing target {ticker}"
            raise Paper5EmpiricalError(message)
        flat_diagnostics.append(
            _FlatDiagnostic(
                tangent_curvature_min=float(curvature),
                converged=bool(target_diagnostic.iloc[0]["converged"]),
                assignment_gap_min=float(
                    target_diagnostic.iloc[0].get("assignment_gap_min", 0.0)
                ),
                assignment_support_size=int(
                    target_diagnostic.iloc[0].get("assignment_support_size", 0)
                ),
            )
        )
    return w_flat, flat_diagnostics


def _robustness_cuts(
    train_returns: np.ndarray, d2: np.ndarray, bandwidth: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate neighbor-count and kernel-bandwidth sensitivity cuts."""
    knn_rows = []
    for neighbor_count in (3, 5, 10, 15):
        weights = est.correlation_knn_w(train_returns, k=neighbor_count)
        qmle = est.sar_qmle(train_returns, weights)
        knn_rows.append(
            hyp.robustness_row(
                "knn_k",
                float(neighbor_count),
                qmle["rho_hat"],
                est.moran_i(train_returns, weights),
            )
        )
    bandwidth_rows = []
    for multiplier in (0.5, 1.0, 2.0):
        adjusted_bandwidth = bandwidth * multiplier
        kernel = np.exp(-d2 / adjusted_bandwidth)
        np.fill_diagonal(kernel, 0.0)
        row_sums = kernel.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        weights = kernel / row_sums
        qmle = est.sar_qmle(train_returns, weights)
        bandwidth_rows.append(
            hyp.robustness_row(
                "bandwidth_h",
                float(adjusted_bandwidth),
                qmle["rho_hat"],
                est.moran_i(train_returns, weights),
            )
        )
    return knn_rows, bandwidth_rows


def _build_weight_variants(
    tickers: list[str],
    d2: np.ndarray,
    eps_stat: dict[str, float],
    sector_of: dict[str, str],
    shared_barycentre_dir: Path,
    wasserstein_w1_barycentre_dir: Path,
    co_mentions_adjacency: Path,
    *,
    provider_id: str,
    representation_id: str,
    barycentre_arm_id: str,
    wasserstein_w1_arm_id: str,
    distance_id: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Mapping[str, object]],
    dict[str, float | bool | str],
    dict[str, float],
    float,
    np.ndarray,
    np.ndarray,
]:
    """Construct primary W2 weights and the external co-mention comparator."""
    w_flat, flat_diagnostics = _load_shared_w_flat(
        shared_barycentre_dir,
        tickers,
        provider_id=provider_id,
        representation_id=representation_id,
        arm_id=barycentre_arm_id,
        geometry_id=distance_id,
    )
    w_w1, w1_diagnostics = _load_shared_w_flat(
        wasserstein_w1_barycentre_dir,
        tickers,
        provider_id=provider_id,
        representation_id=representation_id,
        arm_id=wasserstein_w1_arm_id,
        geometry_id="wasserstein_w1",
    )
    w_h, bandwidth = wgt.build_w_kernel(d2, tickers)
    w_co_mentions, co_mentions_diagnostics = _load_co_mentions_w(
        co_mentions_adjacency, tickers
    )
    eps_stat_matrix = stat_radius_matrix(eps_stat, tickers)
    perturbed_w_h = wgt.perturb_w_kernel(d2, eps_stat_matrix, bandwidth)
    delta_w_h = wgt.delta_w_kernel(w_h, perturbed_w_h)
    flat_bound = wgt.delta_w_flat_bound(flat_diagnostics, eps_stat, tickers)
    diagnostics: dict[str, Mapping[str, object]] = {
        "w_flat": {
            "row_stochastic_max_dev": float(np.max(np.abs(w_flat.sum(axis=1) - 1.0))),
            "zero_diagonal_max_abs": float(np.max(np.abs(np.diag(w_flat)))),
            "all_converged": all(item.converged for item in flat_diagnostics),
            "delta_w_bound_row_median": flat_bound["bound_row_median"],
            "delta_w_bound_row_max": flat_bound["bound_row_max"],
            "s_min": flat_bound["s_min"],
            "assignment_gap_min": flat_bound["assignment_gap_min"],
            "certified": flat_bound["certified"],
            "certification_status": flat_bound["certification_status"],
        },
        "w_h": {
            "bandwidth_h": bandwidth,
            "bandwidth_rule": "median off-diagonal squared W2 distance",
            "row_sum_max_dev": float(np.max(np.abs(w_h.sum(axis=1) - 1.0))),
            "zero_diagonal_max_abs": float(np.max(np.abs(np.diag(w_h)))),
            "delta_w_frobenius": delta_w_h["frobenius_norm"],
            "delta_w_row_norm_median": delta_w_h["row_norm_median"],
            "delta_w_row_norm_max": delta_w_h["row_norm_max"],
        },
        "w_w1": {
            "construction": "target_anchored_wasserstein_w1_barycentre",
            "row_stochastic_max_dev": float(np.max(np.abs(w_w1.sum(axis=1) - 1.0))),
            "zero_diagonal_max_abs": float(np.max(np.abs(np.diag(w_w1)))),
            "all_converged": all(item.converged for item in w1_diagnostics),
        },
        "w_co_mentions": co_mentions_diagnostics,
    }
    variants = {
        "w_flat": w_flat,
        "w_h": w_h,
        "w_w1": w_w1,
        "w_co_mentions": w_co_mentions,
        "industry_adjacency": est.industry_adjacency_w(tickers, sector_of),
        "rho_zero": est.zero_w(len(tickers)),
    }
    return variants, diagnostics, flat_bound, delta_w_h, bandwidth, w_flat, w_h

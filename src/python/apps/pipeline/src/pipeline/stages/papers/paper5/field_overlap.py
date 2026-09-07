"""Geometry and induced-field overlap between two frozen interaction fields.

A likelihood ordering says which field organizes returns better; it does not say
whether the two fields measure the same relationships. These diagnostics answer
that question in the three stages the comparison naturally decomposes into: do
the constructions choose the same peers, do they weight the chosen peers the
same way, and do they generate the same spatial signal on the evaluation panel?

The induced-field correlation is also the statistic that governs whether the
joint two-field fit is identified, so it is reported next to the joint estimates
rather than in a separate corner: two fields that generate nearly the same
peer-return series leave the likelihood flat along a ridge, however different
they look edge by edge.

Support overlap is reported three ways because the two constructions do not
select comparably many peers. Raw Jaccard on such a pair moves with the size gap
rather than with agreement, so a directed share and a symmetric rank-matched
share are reported alongside it, calibrated against a matched-sparsity random
null so "the constructions agree on peers" is a statement about more than
chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

from pipeline.stages.papers.paper5._gate.metrics import (
    random_matched_sparsity_weights,
    validate_weight_matrix,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_PANEL_NDIM = 2


class FieldOverlapError(ValueError):
    """Report an input that cannot support a field-overlap comparison."""


@dataclass(frozen=True)
class SupportOverlap:
    """Row-level agreement between the active peer sets of two fields."""

    jaccard: NDArray[np.float64]
    directed_share: NDArray[np.float64]
    rank_matched_share: NDArray[np.float64]
    focal_support_sizes: NDArray[np.int64]
    comparator_support_sizes: NDArray[np.int64]


@dataclass(frozen=True)
class WeightSimilarity:
    """Row-level agreement between the cardinal weights of two fields."""

    union_pearson: NDArray[np.float64]
    union_cosine: NDArray[np.float64]
    common_edge_spearman: NDArray[np.float64]
    common_edge_counts: NDArray[np.int64]
    offdiagonal_pearson: float


@dataclass(frozen=True)
class InducedFieldSimilarity:
    """Agreement between the peer-return series the two fields generate."""

    pooled_pearson: float
    pooled_r_squared: float
    per_asset_pearson: NDArray[np.float64]


def _active_support(
    weights: NDArray[np.float64], *, tolerance: float
) -> NDArray[np.bool_]:
    """Return the strictly active, self-excluding peer set of every row."""
    support = weights > tolerance
    support[np.diag_indices_from(support)] = False
    return support


def _top_k_support(
    weights: NDArray[np.float64],
    sizes: NDArray[np.int64],
    *,
    tolerance: float,
) -> NDArray[np.bool_]:
    """Keep each row's ``k`` heaviest strictly active peers.

    Padding a short row with inactive entries would manufacture agreement, so
    only strictly active peers are eligible however large ``k`` is.
    """
    matched = np.zeros(weights.shape, dtype=np.bool_)
    order = np.argsort(-weights, axis=1, kind="stable")
    for row, size in enumerate(sizes):
        keep = [
            index
            for index in order[row]
            if index != row and weights[row, index] > tolerance
        ][: int(size)]
        matched[row, keep] = True
    return matched


def support_overlap(
    focal: NDArray[np.float64],
    comparator: NDArray[np.float64],
    *,
    tolerance: float,
) -> SupportOverlap:
    """Compare the active peer sets of two interaction fields row by row.

    Agreement is reported three ways because the two constructions need not
    select comparably many peers. Jaccard is symmetric but falls purely from a
    size gap; the directed share asks what fraction of the focal row's peers the
    comparator also names; the rank-matched share compares only the strongest
    peers of each, at the smaller of the two row sizes.

    Args:
        focal: Focal interaction field, shape ``(n, n)``.
        comparator: Comparator interaction field, shape ``(n, n)``.
        tolerance: Strict activity threshold on a weight.

    Returns:
        Row-level Jaccard, directed and rank-matched shares, and support sizes.

    Raises:
        FieldOverlapError: If either field leaves a row without an active peer.

    """
    _validate_pair(focal, comparator)
    focal_support = _active_support(focal, tolerance=tolerance)
    comparator_support = _active_support(comparator, tolerance=tolerance)
    focal_sizes = focal_support.sum(axis=1).astype(np.int64)
    comparator_sizes = comparator_support.sum(axis=1).astype(np.int64)
    if np.any(focal_sizes == 0) or np.any(comparator_sizes == 0):
        message = "both fields must select at least one peer in every row"
        raise FieldOverlapError(message)

    # Size-matched agreement: of the k strongest peers each construction names,
    # with k the smaller of the two row sizes, what share do they share?
    matched_sizes = np.minimum(focal_sizes, comparator_sizes)
    focal_top = _top_k_support(focal, matched_sizes, tolerance=tolerance)
    comparator_top = _top_k_support(comparator, matched_sizes, tolerance=tolerance)

    intersection = np.logical_and(focal_support, comparator_support).sum(axis=1)
    union = np.logical_or(focal_support, comparator_support).sum(axis=1)
    matched_intersection = np.logical_and(focal_top, comparator_top).sum(axis=1)
    return SupportOverlap(
        jaccard=intersection / np.maximum(union, 1),
        directed_share=intersection / focal_sizes,
        rank_matched_share=matched_intersection / matched_sizes,
        focal_support_sizes=focal_sizes,
        comparator_support_sizes=comparator_sizes,
    )


def weight_similarity(
    focal: NDArray[np.float64],
    comparator: NDArray[np.float64],
    *,
    tolerance: float,
) -> WeightSimilarity:
    """Compare the cardinal weights of two interaction fields row by row.

    Similarity is measured twice: over the union of the two supports, where a
    peer chosen by only one construction counts as a disagreement, and over the
    common edges alone, where it measures whether shared peers are ranked alike.

    Args:
        focal: Focal interaction field, shape ``(n, n)``.
        comparator: Comparator interaction field, shape ``(n, n)``.
        tolerance: Strict activity threshold on a weight.

    Returns:
        Row-level union and common-edge similarities plus the pooled
        off-diagonal correlation.

    """
    _validate_pair(focal, comparator)
    focal_support = _active_support(focal, tolerance=tolerance)
    comparator_support = _active_support(comparator, tolerance=tolerance)
    n_assets = focal.shape[0]
    union_pearson = np.full(n_assets, np.nan, dtype=np.float64)
    union_cosine = np.full(n_assets, np.nan, dtype=np.float64)
    common_spearman = np.full(n_assets, np.nan, dtype=np.float64)
    common_counts = np.zeros(n_assets, dtype=np.int64)

    for row in range(n_assets):
        union = np.logical_or(focal_support[row], comparator_support[row])
        left, right = focal[row, union], comparator[row, union]
        union_pearson[row] = _safe_pearson(left, right)
        union_cosine[row] = _safe_cosine(left, right)
        common = np.logical_and(focal_support[row], comparator_support[row])
        common_counts[row] = int(np.count_nonzero(common))
        common_spearman[row] = _safe_spearman(
            focal[row, common], comparator[row, common]
        )

    offdiagonal = ~np.eye(n_assets, dtype=np.bool_)
    return WeightSimilarity(
        union_pearson=union_pearson,
        union_cosine=union_cosine,
        common_edge_spearman=common_spearman,
        common_edge_counts=common_counts,
        offdiagonal_pearson=_safe_pearson(focal[offdiagonal], comparator[offdiagonal]),
    )


def induced_field_similarity(
    returns: NDArray[np.float64],
    focal: NDArray[np.float64],
    comparator: NDArray[np.float64],
) -> InducedFieldSimilarity:
    """Compare the peer-return series the two fields generate on one panel.

    Two fields can differ edge by edge and still produce nearly identical
    spatial lags; conversely, largely overlapping supports can carry different
    aggregate signals when the weights differ.

    Args:
        returns: Evaluation return panel, shape ``(T, n)``.
        focal: Focal interaction field, shape ``(n, n)``.
        comparator: Comparator interaction field, shape ``(n, n)``.

    Returns:
        Pooled correlation and coefficient of determination between the two
        spatial lags, and the per-asset correlation across dates.

    """
    _validate_pair(focal, comparator)
    if returns.ndim != _PANEL_NDIM or returns.shape[1] != focal.shape[0]:
        message = "the return panel must be a date-by-asset matrix matching the fields"
        raise FieldOverlapError(message)
    focal_lag = returns @ focal.T
    comparator_lag = returns @ comparator.T
    pooled = _safe_pearson(focal_lag.reshape(-1), comparator_lag.reshape(-1))
    per_asset = np.array(
        [
            _safe_pearson(focal_lag[:, asset], comparator_lag[:, asset])
            for asset in range(focal.shape[0])
        ],
        dtype=np.float64,
    )
    return InducedFieldSimilarity(
        pooled_pearson=pooled,
        pooled_r_squared=pooled**2,
        per_asset_pearson=per_asset,
    )


def matched_random_support_null(
    focal: NDArray[np.float64],
    comparator: NDArray[np.float64],
    *,
    draws: int,
    seed: int,
    tolerance: float,
) -> NDArray[np.float64]:
    """Calibrate support agreement against fixed-seed sparsity-matched baskets.

    Reuses the gate's matched-sparsity draws so "the two constructions agree on
    peers" is stated against the agreement a random basket of the same row
    sparsity would reach by chance.

    Args:
        focal: Focal interaction field, shape ``(n, n)``.
        comparator: Comparator interaction field, shape ``(n, n)``.
        draws: Number of random baskets.
        seed: Fixed random-support seed.
        tolerance: Strict activity threshold on a weight.

    Returns:
        Mean directed support share of each random draw, shape ``(draws,)``.

    """
    _validate_pair(focal, comparator)
    return np.array(
        [
            float(
                np.mean(
                    support_overlap(
                        draw, comparator, tolerance=tolerance
                    ).directed_share
                )
            )
            for draw in _matched_random_baskets(
                focal, draws=draws, seed=seed, tolerance=tolerance
            )
        ],
        dtype=np.float64,
    )


def matched_random_induced_null(
    returns: NDArray[np.float64],
    focal: NDArray[np.float64],
    comparator: NDArray[np.float64],
    *,
    draws: int,
    seed: int,
    tolerance: float,
) -> NDArray[np.float64]:
    """Calibrate induced-field agreement against sparsity-matched random baskets.

    Any two row-stochastic averages of a co-moving cross-section produce
    correlated peer-return series, so a high correlation between the two fitted
    spatial lags is not by itself evidence that the constructions agree. This
    null asks what correlation a random basket of the focal field's own row
    sparsity would reach against the same comparator.

    Args:
        returns: Evaluation return panel, shape ``(T, n)``.
        focal: Focal interaction field, shape ``(n, n)``.
        comparator: Comparator interaction field, shape ``(n, n)``.
        draws: Number of random baskets.
        seed: Fixed random-support seed.
        tolerance: Strict activity threshold on a weight.

    Returns:
        Pooled induced-field correlation of each random draw, shape ``(draws,)``.

    """
    _validate_pair(focal, comparator)
    return np.array(
        [
            induced_field_similarity(returns, draw, comparator).pooled_pearson
            for draw in _matched_random_baskets(
                focal, draws=draws, seed=seed, tolerance=tolerance
            )
        ],
        dtype=np.float64,
    )


def _matched_random_baskets(
    focal: NDArray[np.float64],
    *,
    draws: int,
    seed: int,
    tolerance: float,
) -> NDArray[np.float64]:
    """Draw fixed-seed baskets sharing the focal field's row sparsities."""
    return random_matched_sparsity_weights(
        focal,
        draws=draws,
        seed=seed,
        active_tolerance=tolerance,
    )


def _validate_pair(focal: NDArray[np.float64], comparator: NDArray[np.float64]) -> None:
    """Reject a pair that does not satisfy the interaction-matrix contract."""
    if focal.shape != comparator.shape:
        message = "both interaction fields must have the same shape"
        raise FieldOverlapError(message)
    validate_weight_matrix(focal)
    validate_weight_matrix(comparator)


def _safe_pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Return a Pearson correlation, or NaN when either side is degenerate."""
    if left.size < _PANEL_NDIM:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_energy = float(left_centered @ left_centered)
    right_energy = float(right_centered @ right_centered)
    denominator = float(np.sqrt(left_energy * right_energy))
    if denominator <= 0.0:
        return float("nan")
    return float(left_centered @ right_centered) / denominator


def _safe_cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Return a cosine similarity, or NaN when either side has no mass."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return float("nan")
    return float(left @ right) / denominator


def _safe_spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Return a Spearman correlation, or NaN when either side carries no ranking."""
    if left.size < _PANEL_NDIM or _is_constant(left) or _is_constant(right):
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _is_constant(values: NDArray[np.float64]) -> bool:
    """Report whether every entry is identical, so ranks carry no information."""
    return bool(np.ptp(values) <= 0.0)


__all__ = [
    "FieldOverlapError",
    "InducedFieldSimilarity",
    "SupportOverlap",
    "WeightSimilarity",
    "induced_field_similarity",
    "matched_random_induced_null",
    "matched_random_support_null",
    "support_overlap",
    "weight_similarity",
]

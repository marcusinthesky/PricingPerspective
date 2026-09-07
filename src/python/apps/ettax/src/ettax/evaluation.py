# SPDX-License-Identifier: Apache-2.0
"""Host-side intrinsic diagnostics for normalized document embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from numpy.typing import NDArray

type FloatMatrix = NDArray[np.float64]
type InputMatrix = NDArray[np.float32] | NDArray[np.float64]
type ScalarMetrics = dict[str, float | int]

_RETRIEVAL_MIN_ROWS = 2
_GEOMETRY_MIN_ROWS = 3
_MATRIX_DIMENSIONS = 2
_RECALL_AT_5 = 5
_RECALL_AT_10 = 10
_MANTEL_DEFAULT_PERMUTATIONS = 999


def retrieval_metrics(
    queries: InputMatrix,
    candidates: InputMatrix,
    *,
    shuffle_seed: int,
) -> ScalarMetrics:
    """Measure paired query-to-candidate retrieval against chance and a derangement."""
    left = _normalized(queries)
    right = _normalized(candidates)
    if left.shape != right.shape:
        message = "paired retrieval matrices must have identical shapes"
        raise ValueError(message)
    rows = left.shape[0]
    if rows < _RETRIEVAL_MIN_ROWS:
        message = "paired retrieval requires at least two rows"
        raise ValueError(message)

    similarities = left @ right.T
    order = np.argsort(-similarities, axis=1, kind="stable")
    targets = np.arange(rows)[:, None]
    ranks = np.argmax(order == targets, axis=1) + 1
    matched = similarities[np.arange(rows), np.arange(rows)]
    shift = 1 + shuffle_seed % (rows - 1)
    shuffled = similarities[np.arange(rows), np.roll(np.arange(rows), shift)]
    harmonic = float(np.sum(1.0 / np.arange(1, rows + 1)) / rows)

    return {
        "chance_mrr": harmonic,
        "chance_recall_at_1": 1.0 / rows,
        "chance_recall_at_5": min(_RECALL_AT_5, rows) / rows,
        "chance_recall_at_10": min(_RECALL_AT_10, rows) / rows,
        "matched_cosine_mean": float(np.mean(matched)),
        "matched_minus_shuffled_cosine": float(np.mean(matched) - np.mean(shuffled)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= _RECALL_AT_5)),
        "recall_at_10": float(np.mean(ranks <= _RECALL_AT_10)),
        "rows": rows,
        "shuffled_cosine_mean": float(np.mean(shuffled)),
    }


def embedding_diagnostics(vectors: InputMatrix) -> ScalarMetrics:
    """Measure collapse, anisotropy, and occupied dimension in float64."""
    values = _normalized(vectors)
    rows, dimensions = values.shape
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectrum = np.square(singular_values)
    total = float(np.sum(spectrum))
    if total > 0.0:
        probabilities = spectrum[spectrum > 0.0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        stable_rank = float(total / np.max(spectrum))
    else:
        effective_rank = 0.0
        stable_rank = 0.0

    centroid = np.mean(values, axis=0)
    cosine_sum = float(np.sum(np.square(np.sum(values, axis=0))) - rows)
    pair_count = rows * (rows - 1)
    unique_rows = int(np.unique(values, axis=0).shape[0])
    return {
        "centroid_norm": float(np.linalg.norm(centroid)),
        "dimensions": dimensions,
        "duplicate_rows": rows - unique_rows,
        "effective_rank": effective_rank,
        "mean_off_diagonal_cosine": cosine_sum / pair_count,
        "rows": rows,
        "stable_rank": stable_rank,
    }


def compare_geometries(
    candidate: InputMatrix,
    reference: InputMatrix,
    *,
    neighbours: int = 10,
) -> ScalarMetrics:
    """Compare cosine-distance ranks and local neighborhoods across two models."""
    left = _normalized(candidate)
    right = _normalized(reference)
    if left.shape[0] != right.shape[0]:
        message = "geometry comparison requires identical row counts"
        raise ValueError(message)
    rows = left.shape[0]
    if rows < _GEOMETRY_MIN_ROWS:
        message = "geometry comparison requires at least three rows"
        raise ValueError(message)
    k = min(neighbours, rows - 1)

    left_similarity = left @ left.T
    right_similarity = right @ right.T
    upper = np.triu_indices(rows, k=1)
    left_distances = 1.0 - left_similarity[upper]
    right_distances = 1.0 - right_similarity[upper]
    correlation = spearmanr(left_distances, right_distances).statistic

    np.fill_diagonal(left_similarity, -np.inf)
    np.fill_diagonal(right_similarity, -np.inf)
    left_neighbours = np.argpartition(left_similarity, -k, axis=1)[:, -k:]
    right_neighbours = np.argpartition(right_similarity, -k, axis=1)[:, -k:]
    overlap = np.fromiter(
        (
            np.intersect1d(left_row, right_row, assume_unique=True).size / k
            for left_row, right_row in zip(
                left_neighbours, right_neighbours, strict=True
            )
        ),
        dtype=np.float64,
        count=rows,
    )
    return {
        "distance_pair_count": int(left_distances.size),
        "distance_spearman": float(correlation),
        f"nearest_neighbour_overlap_at_{k}": float(np.mean(overlap)),
        "rows": rows,
    }


def mantel_tests(
    candidate: InputMatrix,
    reference: InputMatrix,
    *,
    permutations: int = _MANTEL_DEFAULT_PERMUTATIONS,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Compare chord and angular geometries with a row-label Mantel permutation test."""
    left = _normalized(candidate)
    right = _normalized(reference)
    if left.shape[0] != right.shape[0]:
        message = "Mantel comparison requires identical row counts"
        raise ValueError(message)
    rows = left.shape[0]
    if rows < _GEOMETRY_MIN_ROWS:
        message = "Mantel comparison requires at least three rows"
        raise ValueError(message)
    if permutations < 1:
        raise ValueError("Mantel permutations must be positive")

    upper = np.triu_indices(rows, k=1)
    left_cosine = np.clip(left @ left.T, -1.0, 1.0)
    right_cosine = np.clip(right @ right.T, -1.0, 1.0)
    distances = {
        "chord": (
            np.sqrt(np.maximum(0.0, 2.0 - 2.0 * left_cosine)),
            np.sqrt(np.maximum(0.0, 2.0 - 2.0 * right_cosine)),
        ),
        "angular": (
            np.arccos(left_cosine),
            np.arccos(right_cosine),
        ),
    }
    rng = np.random.default_rng(seed)
    permutations_array = [rng.permutation(rows) for _ in range(permutations)]
    results: dict[str, dict[str, float | int]] = {}
    for metric, (left_distance, right_distance) in distances.items():
        candidate_values = left_distance[upper]
        reference_values = right_distance[upper]
        observed = _pearson(candidate_values, reference_values)
        null = np.empty(permutations, dtype=np.float64)
        for index, permutation in enumerate(permutations_array):
            permuted = left_distance[np.ix_(permutation, permutation)][upper]
            null[index] = _pearson(permuted, reference_values)
        results[metric] = {
            "pair_count": int(candidate_values.size),
            "pearson_r": observed,
            "permutations": permutations,
            "p_value_right_tail": float(
                (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
            ),
            "spearman_r": float(
                spearmanr(candidate_values, reference_values).statistic
            ),
        }
    return results


def _pearson(left: FloatMatrix, right: FloatMatrix) -> float:
    """Compute Pearson correlation without float32 accumulation."""
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("Mantel distance vector has zero variance")
    return float(np.dot(left_centered, right_centered) / denominator)


def _normalized(vectors: InputMatrix) -> FloatMatrix:
    """Validate and normalize one finite matrix for cosine diagnostics."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != _MATRIX_DIMENSIONS or min(values.shape) < 1:
        message = "embeddings must be a non-empty two-dimensional matrix"
        raise ValueError(message)
    if not np.all(np.isfinite(values)):
        message = "embeddings contain non-finite values"
        raise ValueError(message)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).tiny):
        message = "embeddings contain zero vectors"
        raise ValueError(message)
    return values / norms

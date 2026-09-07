"""Construct the diffusion interaction matrix and Paper 5 bounds.

``W^h`` is the pre-registered diffusion kernel ``∝ exp(-D^2/h)``, row
normalized and zero-diagonal. Bandwidth ``h`` is fixed by the median
off-diagonal squared W2 distance and recorded alongside the matrix.

The replication operator ``W♭`` is loaded from the shared typed barycentre
artifact by the Paper 5 runner. This module only computes the diffusion
variant and perturbation diagnostics; it does not solve barycentre programs.

For ``W^h`` perturbation is computed exactly by perturbing ``D^2`` at the
radius-widened distance and reforming the kernel. For ``W♭`` the S5a'
Lipschitz propagation is used as a disclosed first-order bound based on the
shared solver diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


class _TangentCurvatureDiagnostic(Protocol):
    """Diagnostics interface shared by typed barycentre artifacts."""

    @property
    def tangent_curvature_min(self) -> float: ...

    @property
    def assignment_gap_min(self) -> float: ...

    @property
    def assignment_support_size(self) -> int: ...


Float = NDArray[np.float64]


def build_w_kernel(squared_distances: Float, tickers: list[str]) -> tuple[Float, float]:
    """Build ``W^h`` — the row-normalized diffusion kernel ``∝ exp(-D^2/h)``.

    Bandwidth ``h`` is the pre-registered rule: the median of the
    off-diagonal squared statistical distances (the standard diffusion-maps
    default bandwidth).

    Args:
        squared_distances: Squared statistical-distance matrix with zero diagonal.
        tickers: Ticker order (only used for shape/logging).

    Returns:
        Tuple ``(W_h, h)`` — ``W_h`` row-normalized, zero-diagonal.

    """
    expected_shape = (len(tickers), len(tickers))
    if squared_distances.shape != expected_shape:
        message = (
            f"distance matrix shape {squared_distances.shape} does not match "
            f"{len(tickers)} tickers"
        )
        raise ValueError(message)
    n = squared_distances.shape[0]
    iu = np.triu_indices(n, k=1)
    h = float(np.median(squared_distances[iu]))
    if h <= 0:
        message = "Degenerate bandwidth h<=0: statistical-distance matrix collapsed"
        raise ValueError(message)
    k = np.exp(-squared_distances / h)
    np.fill_diagonal(k, 0.0)
    row_sums = k.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w_h: Float = k / row_sums
    return w_h, h


def perturb_w_kernel(
    squared_distances: Float, eps_stat_matrix: Float, h: float
) -> Float:
    """``W^h`` re-formed at the radius-widened distance ``D + eps^stat``.

    Exact re-solve for the kernel construction (unlike ``W♭``, see module
    docstring): ``E_hi = sqrt(D2) + eps^stat`` (T1 triangle-inequality
    composition), then ``D2_hi = E_hi^2``, kernel + row-normalize.

    Args:
        squared_distances: Squared statistical-distance matrix, shape ``(n, n)``.
        eps_stat_matrix: Pairwise ``eps^stat`` radius matrix (see
            :func:`pipeline.stages.papers.paper5.energy_radius.stat_radius_matrix`).
        h: Bandwidth (fixed, from the point estimate; not re-selected).

    Returns:
        Perturbed ``(n, n)`` row-normalized kernel matrix.

    """
    e = np.sqrt(np.clip(squared_distances, 0.0, None))
    e_hi = e + eps_stat_matrix
    d2_hi = e_hi**2
    np.fill_diagonal(d2_hi, 0.0)
    k = np.exp(-d2_hi / h)
    np.fill_diagonal(k, 0.0)
    row_sums = k.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return k / row_sums


def delta_w_kernel(w_h: Float, w_h_perturbed: Float) -> dict[str, float]:
    """Summary of ``ΔW^h = W^h_perturbed - W^h`` (row/Frobenius norms)."""
    diff = w_h_perturbed - w_h
    row_norms = np.linalg.norm(diff, axis=1)
    return {
        "frobenius_norm": float(np.linalg.norm(diff)),
        "row_norm_median": float(np.median(row_norms)),
        "row_norm_max": float(np.max(row_norms)),
    }


def delta_w_flat_bound(
    diags: Sequence[_TangentCurvatureDiagnostic],
    eps_stat_per_firm: dict[str, float],
    tickers: list[str],
) -> dict[str, float | bool | str]:
    """Refuse certification until aligned-cost and gradient radii are available.

    The historical ``eps_i / curvature_i`` shortcut mixed a radius for the
    bootstrap mean embedding with the perturbation of an aligned-support W2
    quadratic program.  Those are different objects.  We retain the measured
    curvature and assignment margins as diagnostics, but emit no certified
    weight radius until the producer supplies cost-matrix and objective-gradient
    radii satisfying the local matching-gap theorem.

    Args:
        diags: Per-row shared barycentre diagnostics.
        eps_stat_per_firm: Dict ``{ticker: eps_i}`` (energy-robustness radii).
        tickers: Ticker order (index-aligned to ``diags``).

    Returns:
        A fail-closed certificate record. Numerical bounds are ``NaN`` while
        certification is unavailable.

    """
    curvatures = np.array(
        [float(d.tangent_curvature_min) for d in diags], dtype=np.float64
    )
    gaps = np.array([float(d.assignment_gap_min) for d in diags], dtype=np.float64)
    support_sizes = np.array(
        [int(d.assignment_support_size) for d in diags], dtype=np.int64
    )
    legacy_radii = np.array(
        [eps_stat_per_firm[ticker] for ticker in tickers], dtype=np.float64
    )
    minimum_gap = float(np.min(gaps))
    certification_status = (
        "zero_assignment_gap_and_missing_aligned_radii"
        if minimum_gap <= 0.0
        else "missing_aligned_cost_and_gradient_radii"
    )
    return {
        "certified": False,
        "certification_status": certification_status,
        "s_min": float(np.min(curvatures)),
        "assignment_gap_min": minimum_gap,
        "assignment_support_size_max": float(np.max(support_sizes)),
        "legacy_embedding_radius_max": float(np.max(legacy_radii)),
        "bound_row_median": float("nan"),
        "bound_row_max": float("nan"),
    }

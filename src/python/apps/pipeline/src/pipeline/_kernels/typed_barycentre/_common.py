"""Vocabulary the barycentre solver families share.

The bottom of this package's dependency graph: the typed error surface, the
registry constants, and the small numerical/validation helpers every geometry
family needs.  It imports nothing else from ``typed_barycentre`` -- the family
modules and ``_diagnostics`` import *from* here, never the reverse -- so the
graph stays acyclic without a lint rule having to say so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NoReturn

import numpy as np

from pipeline._kernels.typed_geometry import TypedDistanceError

if TYPE_CHECKING:
    from pipeline._kernels.typed_geometry import (
        EmbeddingCloud,
        KernelMeanMatrix,
        MeanGroundDistanceMatrix,
    )

GroundMetric = Literal["euclidean", "cosine", "angular"]
MINIMUM_CLOUDS = 2
MINIMUM_LEAVE_ONE_OUT_CLOUDS = 3
RANK_TWO = 2
TWO_SOURCE_COUNT = 2
ACTIVE_WEIGHT_EPS_FACTOR = 32.0


def _barycentre_error(message: str) -> NoReturn:
    raise TypedDistanceError(message)


def _reraise_as_barycentre_error(error: Exception) -> NoReturn:
    """Restate a ``jcor`` array-boundary rejection in the typed vocabulary.

    The solvers here own a typed error surface; ``jcor`` raises bare
    ``ValueError`` at its own eager boundary.  Chaining preserves the original
    for a traceback while keeping ``TypedDistanceError`` the single exception
    callers of this package have to catch.
    """
    raise TypedDistanceError(str(error)) from error


def _stacked_sources(items: tuple[EmbeddingCloud, ...]) -> np.ndarray:
    """Stack equal-shaped clouds into the dense ``(k, n, d)`` array jcor takes.

    Shape agreement is checked here rather than left to ``stack`` so that a
    ragged roster fails with this package's vocabulary instead of a NumPy
    message about array dimensions.
    """
    reference = np.asarray(items[0].values)
    if reference.ndim != RANK_TWO or reference.shape[0] == 0:
        _barycentre_error("Wasserstein sources must have equal non-empty shapes")
    if any(np.asarray(item.values).shape != reference.shape for item in items):
        _barycentre_error("Wasserstein sources must have equal non-empty shapes")
    dtype = np.result_type(*(np.asarray(item.values).dtype for item in items))
    return np.stack([np.asarray(item.values, dtype=dtype) for item in items])


def _constraint_violation(weights: np.ndarray, feasible_set: str) -> float:
    if feasible_set == "simplex_nonnegative":
        return float(
            max(
                abs(float(np.sum(weights)) - 1.0),
                float(max(0.0, -float(np.min(weights)))),
            )
        )
    if feasible_set == "affine_signed":
        return float(abs(float(np.sum(weights)) - 1.0))
    return 0.0


def _component_values(
    component: MeanGroundDistanceMatrix | KernelMeanMatrix,
    items: tuple[EmbeddingCloud, ...],
    expected_kind: str,
) -> np.ndarray:
    """Validate a precomputed component against the current cloud roster."""
    if component.metadata.component_kind != expected_kind:
        _barycentre_error(
            "expected "
            f"{expected_kind} component, got {component.metadata.component_kind}"
        )
    item_ids = tuple(item.item_id for item in items)
    if component.item_ids != item_ids:
        _barycentre_error("precomputed geometry component item ordering disagrees")
    return np.asarray(component.values)


__all__ = [
    "ACTIVE_WEIGHT_EPS_FACTOR",
    "MINIMUM_CLOUDS",
    "MINIMUM_LEAVE_ONE_OUT_CLOUDS",
    "RANK_TWO",
    "TWO_SOURCE_COUNT",
    "GroundMetric",
    "TypedDistanceError",
]

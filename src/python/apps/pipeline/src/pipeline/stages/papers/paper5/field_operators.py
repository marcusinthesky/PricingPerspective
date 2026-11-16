"""Fixed target-to-peer interaction-field operators shared by Paper 5 stages.

``p5_economic_value`` scores these operators in the factor-conditioned field
comparison; ``p5_peer_definition_rows`` reads single rows of the same matrices
for the peer-basket exhibits.  Keeping one builder per operator guarantees that
an exhibit row is the row-normalized matrix the comparison actually scored.
No return information enters any builder here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from pipeline.stages.papers.paper5.baskets import equal_support_weights
from pipeline.stages.papers.paper5.co_mentions import load_co_mentions_w
from pipeline.stages.papers.paper5.operators import (
    load_classifications,
    load_operator,
)
from pipeline.stages.substrate.panel import load_priced_distance_universe

if TYPE_CHECKING:
    from pathlib import Path

FloatMatrix = NDArray[np.float64]

_MATRIX_DIMENSIONS = 2
_NONNEGATIVE_TOLERANCE = 1e-10


def _raise(message: str, exception: type[Exception] = ValueError) -> NoReturn:
    """Raise a validation error while keeping messages at the call site small."""
    raise exception(message)


@dataclass(frozen=True, slots=True)
class FieldOperatorPaths:
    """Governed artifacts from which every fixed field is constructed."""

    returns_dir: Path
    w1_barycentre_dir: Path
    w2_barycentre_dir: Path
    distance_artifact_dir: Path
    co_mentions_adjacency: Path
    universe_csv: Path


@dataclass(frozen=True, slots=True)
class FieldOperatorIdentity:
    """Provider, representation, and arm identity of the transport fields."""

    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    w1_barycentre_arm_id: str = "wasserstein_w1_loo"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w1_geometry_id: str = "wasserstein_w1"
    w2_geometry_id: str = "wasserstein_w2"


@dataclass(frozen=True, slots=True)
class FieldOperatorInputs:
    """Validated transport fields plus the raw ingredients of the comparators."""

    tickers: list[str]
    matrices: dict[str, FloatMatrix]
    squared_distances: FloatMatrix
    co_mentions: FloatMatrix
    industry: dict[str, str]
    sector: dict[str, str]


@dataclass(frozen=True, slots=True)
class _ArmIdentity:
    provider_id: str
    representation_id: str
    barycentre_arm_id: str
    geometry_id: str


def validate_operator(matrix: FloatMatrix, *, name: str) -> FloatMatrix:
    """Validate a nonnegative, zero-diagonal row-stochastic operator."""
    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim != _MATRIX_DIMENSIONS or result.shape[0] != result.shape[1]:
        _raise(f"{name} must be square")
    if not np.all(np.isfinite(result)) or np.min(result) < -_NONNEGATIVE_TOLERANCE:
        _raise(f"{name} must be finite and nonnegative")
    if not np.allclose(np.diag(result), 0.0, atol=1e-10, rtol=0.0):
        _raise(f"{name} must have a zero diagonal")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        _raise(f"{name} must be row stochastic")
    return result


def rbf_weights(squared_distances: FloatMatrix, bandwidth: float) -> FloatMatrix:
    """Construct a zero-diagonal Wasserstein-distance RBF field."""
    distances = np.asarray(squared_distances, dtype=np.float64)
    if distances.ndim != _MATRIX_DIMENSIONS or distances.shape[0] != distances.shape[1]:
        _raise("RBF distances must be square")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        _raise("RBF bandwidth must be finite and positive")
    kernel = np.exp(-distances / bandwidth)
    np.fill_diagonal(kernel, 0.0)
    row_sums = kernel.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        _raise("RBF kernel has an empty or nonfinite row")
    return cast("FloatMatrix", kernel / row_sums)


def median_bandwidth(squared_distances: FloatMatrix) -> float:
    """Return the median off-diagonal squared distance (the default bandwidth)."""
    off_diagonal = squared_distances[~np.eye(len(squared_distances), dtype=bool)]
    return float(np.median(off_diagonal))


def rbf_median_operator(squared_distances: FloatMatrix) -> FloatMatrix:
    """Build the median-bandwidth W2 RBF comparator."""
    return rbf_weights(squared_distances, median_bandwidth(squared_distances))


def equal_support_operator(
    w_flat: FloatMatrix, *, active_tolerance: float
) -> FloatMatrix:
    """Spread each row equally over its active barycentric support."""
    return validate_operator(
        equal_support_weights(w_flat, active_tolerance=active_tolerance),
        name="equal-support weights",
    )


def co_mention_operator(co_mentions: FloatMatrix) -> FloatMatrix:
    """Validate the row-normalized persistent co-mention comparator."""
    return validate_operator(co_mentions, name="co-mention weights")


def _load_transport_fields(
    paths: FieldOperatorPaths, identity: FieldOperatorIdentity
) -> tuple[list[str], dict[str, FloatMatrix]]:
    """Load target-anchored W1 and W2 fields with an exact shared ticker order."""
    arms = {
        "w1": (
            paths.w1_barycentre_dir,
            identity.w1_barycentre_arm_id,
            identity.w1_geometry_id,
        ),
        "w2": (
            paths.w2_barycentre_dir,
            identity.w2_barycentre_arm_id,
            identity.w2_geometry_id,
        ),
    }
    tickers: list[str] | None = None
    matrices: dict[str, FloatMatrix] = {}
    for order, (directory, arm_id, geometry_id) in arms.items():
        order_tickers, matrix, _summary = load_operator(
            directory,
            _ArmIdentity(
                identity.provider_id, identity.representation_id, arm_id, geometry_id
            ),
        )
        if tickers is None:
            tickers = order_tickers
        elif tickers != order_tickers:
            _raise("target-anchored W1 and W2 artifacts use different ticker orders")
        matrices[order] = np.asarray(matrix, dtype=np.float64)
    if tickers is None:
        _raise("field-operator loading produced no transport operator")
    return tickers, matrices


def load_field_operator_inputs(
    paths: FieldOperatorPaths, identity: FieldOperatorIdentity
) -> FieldOperatorInputs:
    """Load the transport fields and align distances, co-mentions, and labels."""
    tickers, transport = _load_transport_fields(paths, identity)
    distance_tickers, _distance, squared = load_priced_distance_universe(
        paths.distance_artifact_dir, paths.returns_dir
    )
    if set(distance_tickers) != set(tickers):
        _raise("distance and barycentre universes do not match")
    distance_index = {ticker: index for index, ticker in enumerate(distance_tickers)}
    order = [distance_index[ticker] for ticker in tickers]
    co_mentions, _ = load_co_mentions_w(paths.co_mentions_adjacency, tickers)
    industry, sector = load_classifications(paths.universe_csv, tickers)
    return FieldOperatorInputs(
        tickers=tickers,
        matrices={
            name: validate_operator(matrix, name=name)
            for name, matrix in transport.items()
        },
        squared_distances=cast("FloatMatrix", squared[np.ix_(order, order)]),
        co_mentions=cast("FloatMatrix", co_mentions),
        industry=industry,
        sector=sector,
    )


__all__ = [
    "FieldOperatorIdentity",
    "FieldOperatorInputs",
    "FieldOperatorPaths",
    "co_mention_operator",
    "equal_support_operator",
    "load_field_operator_inputs",
    "median_bandwidth",
    "rbf_median_operator",
    "rbf_weights",
    "validate_operator",
]

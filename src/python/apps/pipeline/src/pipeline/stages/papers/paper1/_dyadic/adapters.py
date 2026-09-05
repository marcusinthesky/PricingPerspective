"""Paper-1 adapters for historical node-count matrices and artifact metadata."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, cast

import numpy as np
from jcor.model.dyadic import (
    DyadicInputError,
    NodeCountSchedule,
    NodeIndex,
    draw_node_counts,
)

if TYPE_CHECKING:
    from jcor.model.dyadic import DyadicSample


def draw_legacy_node_counts(n_nodes: int, n_boot: int, seed: int) -> np.ndarray:
    """Return the historical bare count matrix while jcor keeps typed schedules."""
    try:
        node_index = NodeIndex[int](tuple(range(n_nodes)))
    except (TypeError, ValueError) as error:
        raise DyadicInputError(str(error)) from error
    # jcor returns an immutable jax.Array; this legacy adapter is the
    # host boundary for the callers that still expect a NumPy matrix.
    return np.asarray(draw_node_counts(node_index, n_boot, seed).counts)


def legacy_node_count_schedule[NodeIdT: Hashable](
    sample: DyadicSample[NodeIdT],
    node_counts: np.ndarray,
) -> NodeCountSchedule[NodeIdT]:
    """Tie one historical matrix to the sample's canonical entity mapping."""
    try:
        node_index = cast(
            "NodeIndex[NodeIdT]",
            NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j),
        )
    except ValueError as error:
        raise DyadicInputError(str(error)) from error
    return NodeCountSchedule(node_index=node_index, counts=node_counts)

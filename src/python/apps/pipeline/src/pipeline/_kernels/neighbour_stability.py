"""Label adapter over ``jcor``'s entity-neutral neighbour-stability core.

t46.9 promoted the method to :mod:`jcor.geometry.neighbour` and left this file
a pure re-export. t48.3 re-signed that core on **integer object ids**, because
its determinism device — a ``lexsort`` tie-break — was reaching for
``dtype=object`` string arrays and so made jcor know what a ticker is. This
module is where that knowledge now lives: it owns the ticker↔index mapping, the
comparator-model names, and the two label-bearing row schemas the Paper 1 stage
writes to CSV and parquet. The public signature is unchanged, so
``stages/papers/paper1/encoder_stability.py`` and the DVC stage are untouched.

Why the mapping is pinned to *sorted* tickers
---------------------------------------------
The tie-break used to sort equidistant neighbours by ticker string; it now
sorts by integer id. Those agree only if the ids are order-isomorphic to the
labels, so :func:`ticker_ids` assigns each ticker its **rank in the sorted
ticker list** — never its position in a ``dict``, a DuckDB result, or a parquet
row order, any of which would trade a reproducible ranking for a flaky one. The
rank is computed from the labels themselves, so the caller may present rows in
any order and still get the same neighbour ordering;
``tests/kernels/test_neighbour_stability_mapping.py`` builds the mapping two
ways and pins that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

# numpy kind: host boundary — id vectors handed to the jcor NumPy core.
import numpy as np
from jcor.geometry.neighbour import (
    NEIGHBOUR_INPUT_AXIOMS,
    FloatMatrix,
    neighbour_order,
)
from jcor.geometry.neighbour import (
    cross_encoder_neighbour_stability as _neighbour_stability_by_id,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jax.typing import ArrayLike
    from numpy.typing import NDArray

__all__ = [
    "NEIGHBOUR_INPUT_AXIOMS",
    "FloatMatrix",
    "NeighbourStabilityEdge",
    "NeighbourStabilitySummary",
    "cross_encoder_neighbour_stability",
    "neighbour_order",
    "ticker_ids",
]


class NeighbourStabilitySummary(TypedDict):
    """One comparator's aggregate neighbour-persistence diagnostics.

    The label-bearing counterpart of
    :class:`jcor.geometry.neighbour.NeighbourStabilitySummary`; the field order
    is the column order of ``encoder_neighbour_stability/summary.csv``.
    """

    baseline_model: str
    comparator_model: str
    n_tickers: int
    top_k: int
    mean_top_k_overlap: float
    exact_nearest_agreement: int
    exact_nearest_agreement_share: float


class NeighbourStabilityEdge(TypedDict):
    """One directed anchor-neighbour comparison with complete ranks.

    The field order is the column order of
    ``encoder_neighbour_stability/edges.parquet``.
    """

    baseline_model: str
    comparator_model: str
    anchor: str
    neighbour: str
    baseline_distance: float
    comparator_distance: float
    baseline_rank: int
    comparator_rank: int
    in_baseline_top_k: bool
    in_comparator_top_k: bool


def ticker_ids(tickers: Sequence[str]) -> NDArray[np.int64]:
    """Return the pinned integer id of each ticker, in the given row order.

    The id is the ticker's rank in the sorted ticker list, so the mapping is a
    function of the label set alone and never of how the caller happened to
    enumerate it.

    Args:
        tickers: Unique object labels, in distance-matrix row order.

    Returns:
        One ``int64`` id per input position.

    Raises:
        ValueError: If the labels are not unique.

    """
    ordered = sorted(tickers)
    if len(set(ordered)) != len(ordered):
        message = "ticker labels must be unique"
        raise ValueError(message)
    rank = {ticker: index for index, ticker in enumerate(ordered)}
    return np.asarray([rank[ticker] for ticker in tickers], dtype=np.int64)


def cross_encoder_neighbour_stability(
    *,
    baseline_model: str,
    tickers: Sequence[str],
    baseline_distances: ArrayLike,
    comparison_distances: Mapping[str, ArrayLike],
    top_k: int = 5,
) -> tuple[list[NeighbourStabilitySummary], list[NeighbourStabilityEdge]]:
    """Name the objects and models around the integer-id jcor core.

    Args:
        baseline_model: Name of the reference model.
        tickers: Unique object labels, in matrix row order.
        baseline_distances: The reference ``(n, n)`` distance matrix.
        comparison_distances: Comparator name to its ``(n, n)`` distance
            matrix; comparators are evaluated in sorted name order.
        top_k: Neighbourhood size, in ``[1, n - 1]``.

    Returns:
        The per-comparator summaries and the complete directed edge list, both
        ordered by comparator name and then by anchor.

    Raises:
        ValueError: If the labels are not unique, ``top_k`` is out of range, or
            no comparator was supplied.

    """
    ticker_list = list(tickers)
    object_ids = ticker_ids(ticker_list)
    comparator_models = sorted(comparison_distances)
    if not comparator_models:
        message = "at least one comparison encoder is required"
        raise ValueError(message)

    summaries, edges = _neighbour_stability_by_id(
        object_ids=object_ids,
        baseline_distances=baseline_distances,
        comparison_distances=[
            comparison_distances[model] for model in comparator_models
        ],
        top_k=top_k,
    )
    named_summaries: list[NeighbourStabilitySummary] = [
        {
            "baseline_model": baseline_model,
            "comparator_model": comparator_models[summary["comparator_index"]],
            "n_tickers": summary["n_objects"],
            "top_k": summary["top_k"],
            "mean_top_k_overlap": summary["mean_top_k_overlap"],
            "exact_nearest_agreement": summary["exact_nearest_agreement"],
            "exact_nearest_agreement_share": summary["exact_nearest_agreement_share"],
        }
        for summary in summaries
    ]
    named_edges: list[NeighbourStabilityEdge] = [
        {
            "baseline_model": baseline_model,
            "comparator_model": comparator_models[edge["comparator_index"]],
            "anchor": ticker_list[edge["anchor_index"]],
            "neighbour": ticker_list[edge["neighbour_index"]],
            "baseline_distance": edge["baseline_distance"],
            "comparator_distance": edge["comparator_distance"],
            "baseline_rank": edge["baseline_rank"],
            "comparator_rank": edge["comparator_rank"],
            "in_baseline_top_k": edge["in_baseline_top_k"],
            "in_comparator_top_k": edge["in_comparator_top_k"],
        }
        for edge in edges
    ]
    return named_summaries, named_edges

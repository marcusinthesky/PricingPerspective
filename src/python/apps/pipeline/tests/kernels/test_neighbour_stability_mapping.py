"""Presentation-independence of the t48.3 ticker↔index mapping.

t48.3 moved the neighbour-stability tie-break from a ticker string to a
caller-assigned integer id, which makes the *caller* responsible for
determinism: ids read off a ``dict``, a DuckDB result or a parquet row order
would make the ranking depend on how the rows happened to arrive. The adapter
pins the id to the ticker's rank in the sorted ticker list instead, and these
tests are the evidence that this holds — they build the mapping from three
genuinely different presentations of the same label set (sorted, an explicit
non-identity permutation, and ``dict`` iteration order), permute the distance
matrices to match, and require the ids *and* the complete neighbour output to
be identical.

The fixtures are deliberately tie-heavy: with all-distinct distances the
ranking is fixed by the distances alone and a broken mapping would go
unnoticed. The last test is the negative control — positional ids over the same
presented matrix give a different ranking, so the pinning is load-bearing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pipeline._kernels.neighbour_stability import (
    cross_encoder_neighbour_stability,
    neighbour_order,
    ticker_ids,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pipeline._kernels.neighbour_stability import (
        NeighbourStabilityEdge,
        NeighbourStabilitySummary,
    )

_LABELS = ("MSFT", "AAPL", "TSLA", "GOOG", "NVDA")
_SORTED = tuple(sorted(_LABELS))
#: An explicit non-identity presentation; a seeded shuffle could return sorted.
_PRESENTATION = ("TSLA", "MSFT", "NVDA", "AAPL", "GOOG")
_TOP_K = 2

#: Every off-diagonal distance equal: the ranking is the tie-break alone.
_TIED = np.ones((len(_SORTED), len(_SORTED)), dtype=np.float64) - np.eye(
    len(_SORTED), dtype=np.float64
)

#: Two distance levels over the sorted labels, so most anchors see a tie.
_MIXED = np.array(
    [
        [0.0, 1.0, 1.0, 2.0, 2.0],
        [1.0, 0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 0.0, 2.0, 1.0],
        [2.0, 1.0, 2.0, 0.0, 1.0],
        [2.0, 2.0, 1.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _present(matrix: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Return a sorted-order matrix re-indexed onto the given row order.

    Args:
        matrix: A distance matrix whose rows follow :data:`_SORTED`.
        labels: The presentation to re-index both axes onto.

    Returns:
        The same distances, permuted on rows and columns together.

    """
    order = [_SORTED.index(label) for label in labels]
    return np.asarray(matrix)[np.ix_(order, order)]


def _run(
    labels: Sequence[str],
) -> tuple[list[NeighbourStabilitySummary], list[NeighbourStabilityEdge]]:
    """Return the full neighbour output for one presentation of the labels.

    Args:
        labels: The label set in the row order to present it in.

    Returns:
        The adapter's summaries and edges.

    """
    return cross_encoder_neighbour_stability(
        baseline_model="baseline",
        tickers=list(labels),
        baseline_distances=_present(_TIED, labels),
        comparison_distances={
            "tied": _present(_TIED, labels),
            "mixed": _present(_MIXED, labels),
        },
        top_k=_TOP_K,
    )


def _by_edge_key(
    edges: list[NeighbourStabilityEdge],
) -> dict[tuple[str, str, str], NeighbourStabilityEdge]:
    """Key the edge list by its label identity rather than by row position.

    Args:
        edges: The adapter's directed edge list.

    Returns:
        Each edge under ``(comparator_model, anchor, neighbour)``.

    """
    keyed = {
        (edge["comparator_model"], edge["anchor"], edge["neighbour"]): edge
        for edge in edges
    }
    assert len(keyed) == len(edges), "edge identity keys must not collide"
    return keyed


def test_the_ids_are_a_function_of_the_label_set_not_its_presentation() -> None:
    """Assign the same id to a ticker however the caller enumerated the rows."""
    shuffled = list(_PRESENTATION)
    assert shuffled != sorted(shuffled), "the permutation must not be the identity"
    dict_order = list(dict.fromkeys(_LABELS))
    assert dict_order != sorted(dict_order)

    from_sorted = dict(zip(_SORTED, ticker_ids(_SORTED).tolist(), strict=True))
    from_shuffled = dict(zip(shuffled, ticker_ids(shuffled).tolist(), strict=True))
    from_dict = dict(zip(dict_order, ticker_ids(dict_order).tolist(), strict=True))

    assert from_sorted == {label: rank for rank, label in enumerate(_SORTED)}
    assert from_shuffled == from_sorted
    assert from_dict == from_sorted


def test_the_neighbour_output_is_identical_across_presentations() -> None:
    """Return the same labelled summaries and edges from a permuted matrix."""
    sorted_summaries, sorted_edges = _run(_SORTED)
    shuffled_summaries, shuffled_edges = _run(_PRESENTATION)
    dict_summaries, dict_edges = _run(list(dict.fromkeys(_LABELS)))

    assert sorted_summaries == shuffled_summaries
    assert sorted_summaries == dict_summaries
    assert _by_edge_key(sorted_edges) == _by_edge_key(shuffled_edges)
    assert _by_edge_key(sorted_edges) == _by_edge_key(dict_edges)


def test_positional_ids_would_change_the_ranking() -> None:
    """Show the pinning is load-bearing: row position is not the sorted rank."""
    labels = list(_PRESENTATION)
    presented = _present(_TIED, labels)
    pinned = neighbour_order(ticker_ids(labels), presented)
    positional = neighbour_order(range(len(labels)), presented)

    assert pinned.tolist() != positional.tolist()
    ranked = [labels[index] for index in pinned[0].tolist()]
    assert ranked == [label for label in _SORTED if label != labels[0]]

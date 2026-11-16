"""Object-identity gate for the shared numerical kernels.

``pipeline._kernels.neighbour_stability`` re-exports its entity-neutral parts
from ``statadic``. Import succeeding is not enough: a redefinition rather than a
re-export would silently diverge from the promoted kernel, so these tests
assert identity, not importability.

The Statadic core accepts integer object ids, so the
ticker↔index mapping, the comparator-model names and the two label-bearing row
schemas live in the shim, which is the natural home for project-entity
knowledge. Those label-bearing parts are pinned to the *shape* that makes them
adapters rather than copies.
"""

from __future__ import annotations

import inspect

from pipeline._kernels import neighbour_stability as shim_neighbour
from statadic.geometry.graph import neighbour as statadic_neighbour


def test_the_entity_neutral_neighbour_surface_is_re_exported() -> None:
    """Share every part of the neighbour surface that names no project entity."""
    assert shim_neighbour.neighbour_order is statadic_neighbour.neighbour_order
    assert (
        shim_neighbour.NEIGHBOUR_INPUT_AXIOMS
        is statadic_neighbour.NEIGHBOUR_INPUT_AXIOMS
    )
    assert shim_neighbour.FloatMatrix is statadic_neighbour.FloatMatrix


def test_the_label_bearing_neighbour_surface_adapts_rather_than_re_exports() -> None:
    """Own the ticker↔index mapping in the pipeline adapter."""
    assert (
        shim_neighbour.cross_encoder_neighbour_stability
        is not statadic_neighbour.cross_encoder_neighbour_stability
    )
    parameters = inspect.signature(
        shim_neighbour.cross_encoder_neighbour_stability
    ).parameters
    assert "tickers" in parameters
    assert "baseline_model" in parameters
    assert "object_ids" not in parameters
    assert set(shim_neighbour.NeighbourStabilitySummary.__annotations__) >= {
        "baseline_model",
        "comparator_model",
        "n_tickers",
    }
    assert set(shim_neighbour.NeighbourStabilityEdge.__annotations__) >= {
        "anchor",
        "neighbour",
        "comparator_model",
    }
    for name in ("NeighbourStabilitySummary", "NeighbourStabilityEdge"):
        assert getattr(shim_neighbour, name) is not getattr(statadic_neighbour, name)

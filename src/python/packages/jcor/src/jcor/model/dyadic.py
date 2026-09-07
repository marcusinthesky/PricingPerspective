"""Public facade for entity-neutral undirected dyadic estimation."""

from jcor.model._dyadic.contracts import (
    DyadicBootstrapDesign,
    DyadicBootstrapResult,
    DyadicComputationError,
    DyadicFit,
    DyadicFitResult,
    DyadicInference,
    DyadicInputError,
    DyadicModel,
    DyadicSample,
    NodeCountSchedule,
)
from jcor.model._dyadic.fit import build_dyadic_design, fit_dyadic_model
from jcor.model._dyadic.inference import (
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    draw_node_counts,
    node_bootstrap_coefficient,
)
from jcor.operators.design import NodeIndex

__all__ = [
    "DyadicBootstrapDesign",
    "DyadicBootstrapResult",
    "DyadicComputationError",
    "DyadicFit",
    "DyadicFitResult",
    "DyadicInference",
    "DyadicInputError",
    "DyadicModel",
    "DyadicSample",
    "NodeCountSchedule",
    "NodeIndex",
    "bootstrap_coefficient_inference",
    "bootstrap_dyadic",
    "build_dyadic_design",
    "draw_node_counts",
    "fit_dyadic_model",
    "node_bootstrap_coefficient",
]

# Keep persisted qualified names stable while private modules own implementation.
for _public_object in (
    DyadicBootstrapDesign,
    DyadicBootstrapResult,
    DyadicComputationError,
    DyadicFit,
    DyadicFitResult,
    DyadicInference,
    DyadicInputError,
    DyadicModel,
    DyadicSample,
    NodeCountSchedule,
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    build_dyadic_design,
    draw_node_counts,
    fit_dyadic_model,
    node_bootstrap_coefficient,
):
    _public_object.__module__ = __name__

del _public_object

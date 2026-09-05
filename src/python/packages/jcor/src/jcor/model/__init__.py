"""S6 — estimators over an operator plus data.

Position
--------
rank 6

Consumes
--------
an operator (waist 2) and data

Produces
--------
an ``EstimateResult``

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

This package is a t46 skeleton: modules land here via the t46.4-t46.9
migrations. Re-exports are appended per-owner, in the region marked below.
"""

from __future__ import annotations

__all__: list[str] = []

# --- t46 migration re-exports; each task appends only to its own region ---

# --- dyadic estimation ---
from jcor.model.dyadic import (
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
    NodeIndex,
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    build_dyadic_design,
    draw_node_counts,
    fit_dyadic_model,
    node_bootstrap_coefficient,
)

__all__ += [
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

# --- covariance estimation and shrinkage ---
from jcor.model.covariance import (  # noqa: E402
    LedoitWolfResult,
    ShrinkageResult,
    cv_lambda_grid,
    ledoit_wolf_sample,
    optimal_lambda_dist,
    plug_in_lambda_dist,
    sample_cov_entry_variance,
    sample_covariance,
    shrink_to_dist,
    target_weight_dist,
)

__all__ += [
    "LedoitWolfResult",
    "ShrinkageResult",
    "cv_lambda_grid",
    "ledoit_wolf_sample",
    "optimal_lambda_dist",
    "plug_in_lambda_dist",
    "sample_cov_entry_variance",
    "sample_covariance",
    "shrink_to_dist",
    "target_weight_dist",
]

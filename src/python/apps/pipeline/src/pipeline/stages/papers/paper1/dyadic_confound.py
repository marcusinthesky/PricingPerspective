"""Paper-1 dyadic association stage.

The publication specification regresses return-correlation chord distance on
the square root of the sample energy functional with symmetric additive firm
effects and multinomial node-bootstrap inference.  Its ladder, input schemas,
diagnostics, and artifact rendering are Paper-specific and live under
``paper1._dyadic``.  Array-only design and estimation contracts live in
``jcor.operators.design`` and ``jcor.model.dyadic``.

This module intentionally remains a stable import facade for the CLI and for
downstream Paper-1 stages.  Every name below is a direct re-export; no numeric
implementation is duplicated here.
"""

from __future__ import annotations

from jcor.model.dyadic import (
    DyadicBootstrapDesign,
    DyadicComputationError,
    DyadicFit,
    DyadicInference,
    DyadicInputError,
    DyadicModel,
    DyadicSample,
    bootstrap_coefficient_inference,
    build_dyadic_design,
)
from jcor.operators.design import (
    drop_collinear_columns,
    fwl_coefficient,
    symmetric_dyadic_effects,
)

from pipeline.stages.papers.paper1._dyadic.adapters import (
    draw_legacy_node_counts,
    legacy_node_count_schedule,
)
from pipeline.stages.papers.paper1._dyadic.contracts import (
    SQUARED_DISTANCE_ARBITRATION as _SQUARED_DISTANCE_ARBITRATION,
)
from pipeline.stages.papers.paper1._dyadic.contracts import (
    DyadicConfoundConfig,
)
from pipeline.stages.papers.paper1._dyadic.diagnostics import (
    _design_diagnostic_row,
    _influence_diagnostics,
    _shared_endpoint_scale_statistic,
)
from pipeline.stages.papers.paper1._dyadic.driver import run_dyadic_confound
from pipeline.stages.papers.paper1._dyadic.estimation import (
    _bootstrap_spec_draws,
    _BootstrapDesign,
    _fit_spec,
    _fwl_display,
    _node_bootstrap_energy,
    _partial_r2_block,
    _quadratic_functional_form_sensitivity,
    _return_chord_translation,
)
from pipeline.stages.papers.paper1._dyadic.inputs import (
    embedding_dispersions as _embedding_dispersions,
)
from pipeline.stages.papers.paper1._dyadic.inputs import (
    load_covariance_long as _load_covariance_long,
)
from pipeline.stages.papers.paper1._dyadic.inputs import (
    load_energy_long as _load_energy_long,
)
from pipeline.stages.papers.paper1._dyadic.inputs import load_w2_long as _load_w2_long

_bootstrap_inference = bootstrap_coefficient_inference
_build_design = build_dyadic_design
draw_node_counts = draw_legacy_node_counts
_draw_node_counts = draw_legacy_node_counts
_drop_collinear_columns = drop_collinear_columns
_fwl_energy_coef = fwl_coefficient
_symmetric_firm_effects = symmetric_dyadic_effects

__all__ = [
    "_SQUARED_DISTANCE_ARBITRATION",
    "DyadicBootstrapDesign",
    "DyadicComputationError",
    "DyadicConfoundConfig",
    "DyadicFit",
    "DyadicInference",
    "DyadicInputError",
    "DyadicModel",
    "DyadicSample",
    "_BootstrapDesign",
    "_bootstrap_inference",
    "_bootstrap_spec_draws",
    "_build_design",
    "_design_diagnostic_row",
    "_draw_node_counts",
    "_drop_collinear_columns",
    "_embedding_dispersions",
    "_fit_spec",
    "_fwl_display",
    "_fwl_energy_coef",
    "_influence_diagnostics",
    "_load_covariance_long",
    "_load_energy_long",
    "_load_w2_long",
    "_node_bootstrap_energy",
    "_partial_r2_block",
    "_quadratic_functional_form_sensitivity",
    "_return_chord_translation",
    "_shared_endpoint_scale_statistic",
    "_symmetric_firm_effects",
    "bootstrap_coefficient_inference",
    "build_dyadic_design",
    "draw_node_counts",
    "drop_collinear_columns",
    "fwl_coefficient",
    "legacy_node_count_schedule",
    "run_dyadic_confound",
    "symmetric_dyadic_effects",
]

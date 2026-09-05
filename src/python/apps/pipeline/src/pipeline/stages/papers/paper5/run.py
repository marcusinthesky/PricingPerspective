"""Stable façade for the end-to-end Paper 5 empirical orchestrator.

Reuses the shared 100-ticker universe/radius machinery, loads the shared typed
``W♭`` barycentre artifact, constructs ``W^h``, fits SAR QMLE/GMM per fold and
baseline, and evaluates H2/H3/H4. The cohesive implementation lives under
:mod:`pipeline.stages.papers.paper5._run`; this module remains the permanent
CLI and caller import surface.

Folds use the disjoint-epoch protocol: train on
``[calib_start, fold_start)`` and test on each fold's calendar interval.
"""

from __future__ import annotations

from pipeline.stages.papers.paper5._run.contracts import (
    _CALIB_START,
    _IN_SAMPLE_FOLDS,
    _OOS_FOLDS,
    _RHO_SE_FALLBACK,
    ACTIVE_WEIGHT_TOLERANCE,
    EXPECTED_TICKERS,
    Paper5EmpiricalConfig,
    Paper5EmpiricalError,
    Paper5EmpiricalPaths,
    _H3EvaluationInputs,
    _RhoFitInputs,
)
from pipeline.stages.papers.paper5._run.driver import (
    _returns_seam_disclosure,
    _to_plain,
    run_paper5_empirical,
)
from pipeline.stages.papers.paper5._run.evaluation import (
    _evaluate_h3,
    _evaluate_h3_per_fold,
    _evaluate_h4,
)
from pipeline.stages.papers.paper5._run.rho import (
    _fit_rho_models,
    _resolve_rho_se,
    _rho_stability_records,
)
from pipeline.stages.papers.paper5._run.robustness import (
    _build_weight_variants,
    _robustness_cuts,
)

# Private names were addressable on the former single-file module. Keep them
# bound here for direct tests and notebooks, while the public ``__all__`` stays
# intentionally small. Referencing the tuple also makes the compatibility
# surface explicit to static unused-import checks.
_COMPATIBILITY_EXPORTS = (
    _CALIB_START,
    _H3EvaluationInputs,
    _IN_SAMPLE_FOLDS,
    _OOS_FOLDS,
    _RHO_SE_FALLBACK,
    _RhoFitInputs,
    _build_weight_variants,
    _evaluate_h3,
    _evaluate_h3_per_fold,
    _evaluate_h4,
    _fit_rho_models,
    _resolve_rho_se,
    _returns_seam_disclosure,
    _rho_stability_records,
    _robustness_cuts,
    _to_plain,
)

__all__ = [
    "ACTIVE_WEIGHT_TOLERANCE",
    "EXPECTED_TICKERS",
    "Paper5EmpiricalConfig",
    "Paper5EmpiricalError",
    "Paper5EmpiricalPaths",
    "run_paper5_empirical",
]

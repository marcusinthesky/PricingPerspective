"""S8 — what to conclude from p-values.

Position
--------
rank 8

Consumes
--------
``TestResult`` objects

Produces
--------
equivalence/relevance verdicts, multiplicity-controlled selections

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

# --- t46.8 ---
from jcor.decision.sharpe import (
    DeflatedSharpeResult,
    MultipleSharpeResult,
    deflated_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    multiple_sharpe_test,
    probabilistic_sharpe_ratio,
    sharpe_difference_test,
    sharpe_ratio,
    sharpe_ratio_ci,
)

__all__ += [
    "DeflatedSharpeResult",
    "MultipleSharpeResult",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe_ratio",
    "multiple_sharpe_test",
    "probabilistic_sharpe_ratio",
    "sharpe_difference_test",
    "sharpe_ratio",
    "sharpe_ratio_ci",
]
# --- end t46.8 ---

# --- t46.7 ---
from jcor.decision.equivalence import (  # noqa: E402
    equivalence_pvalue_from_bootstrap,
    equivalence_test,
    poor_hedge_margin,
)
from jcor.decision.multiplicity import (  # noqa: E402
    equivalence_multiplicity_control,
)

__all__ += [
    "equivalence_multiplicity_control",
    "equivalence_pvalue_from_bootstrap",
    "equivalence_test",
    "poor_hedge_margin",
]
# --- end t46.7 ---

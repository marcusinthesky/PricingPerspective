"""Shared contracts — the axiom lattice, branded containers, result types.

Position
--------
rank 0 · importable by every stage

Consumes
--------
nothing (leaf of the import graph)

Produces
--------
type aliases, ``Axioms``, ``DMat``, ``TestResult``, ``EstimateResult``

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

Unlike the other stage packages, ``core`` is populated in wave A (t46.1) rather
than by a migration: every later task imports these names, so they are final
before wave B starts. Re-exports are appended per-owner, in the region marked
below.
"""

from __future__ import annotations

__all__ = [
    "DIVERGENCE",
    "METRIC",
    "NEGATIVE_TYPE",
    "PSEUDOMETRIC",
    "QUASIMETRIC",
    "SEMIMETRIC",
    "STATIC",
    "STRONG_NEGATIVE_TYPE",
    "Array",
    "Axioms",
    "Bool",
    "DMat",
    "DivergenceLaw",
    "EstimateResult",
    "Float",
    "Int",
    "Law",
    "LawProperties",
    "LawT",
    "LawfulCallable",
    "LegacyDMat",
    "Metric",
    "MetricLaw",
    "NegativeType",
    "NegativeTypeLaw",
    "Nonnegative",
    "Num",
    "PRNGKey",
    "Params",
    "Premetric",
    "PremetricLaw",
    "PseudometricLaw",
    "QuasimetricLaw",
    "Real",
    "ResultT",
    "Scalar",
    "ScalarLike",
    "Semimetric",
    "SemimetricLaw",
    "Separating",
    "Static",
    "StrongNegativeTypeLaw",
    "Symmetric",
    "TestResult",
    "TriangleInequality",
    "ZeroDiagonal",
    "cell_key",
    "typed_jit",
    "typed_vmap",
]

# --- t46 migration re-exports; each task appends only to its own region ---

# t46.1 — typing foundation.
from jcor.core.axioms import (
    DIVERGENCE,
    METRIC,
    NEGATIVE_TYPE,
    PSEUDOMETRIC,
    QUASIMETRIC,
    SEMIMETRIC,
    STRONG_NEGATIVE_TYPE,
    Axioms,
    DivergenceLaw,
    Law,
    LawProperties,
    Metric,
    MetricLaw,
    NegativeType,
    NegativeTypeLaw,
    Nonnegative,
    Premetric,
    PremetricLaw,
    PseudometricLaw,
    QuasimetricLaw,
    Semimetric,
    SemimetricLaw,
    Separating,
    StrongNegativeTypeLaw,
    Symmetric,
    TriangleInequality,
    ZeroDiagonal,
)

# --- estimating-equation contracts (waist 2) ---
from jcor.core.equations import (
    EquationContract,
    EquationDescriptor,
    EquationJacobian,
    EstimatingEquationKind,
    EstimatingFunction,
    MeanCenteredLongRun,
    MomentCondition,
    PlainLongRun,
    SandwichCovariance,
    VariabilityMatrix,
    VariabilityScheme,
    sandwich_covariance,
)
from jcor.core.matrices import DMat
from jcor.core.random import cell_key
from jcor.core.results import EstimateResult, TestResult
from jcor.core.transforms import typed_jit, typed_vmap
from jcor.core.typing import (
    STATIC,
    Array,
    Bool,
    Float,
    Int,
    LawfulCallable,
    LawT,
    Num,
    Params,
    PRNGKey,
    Real,
    ResultT,
    Scalar,
    ScalarLike,
    Static,
)
from jcor.core.typing import (
    DMat as LegacyDMat,
)

__all__ += [
    "EquationContract",
    "EquationDescriptor",
    "EquationJacobian",
    "EstimatingEquationKind",
    "EstimatingFunction",
    "MeanCenteredLongRun",
    "MomentCondition",
    "PlainLongRun",
    "SandwichCovariance",
    "VariabilityMatrix",
    "VariabilityScheme",
    "sandwich_covariance",
]

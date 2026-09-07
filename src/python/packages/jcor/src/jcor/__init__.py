"""jcor — energy statistics and distance-based inference in JAX.

The package is organised by **position in the inference chain**, not by
mathematical provenance (t46). Import from the stage that owns the symbol::

    sample -> ground -> discrepancy -> geometry/association -> operator
           -> model -> inference -> decision
                  ^                        ^
              WAIST 1                  WAIST 2
        "a distance matrix"     "a structured operator"

Below waist 1 anything producing a distance matrix is interchangeable; above
it, nothing knows which produced it. The same holds at waist 2. Organising by
position is what makes novel combinations — energy divergences standing in for
covariance, spatial operators built from energy geometry — free rather than
boundary violations.

Stages
------
:mod:`jcor.core`
    Contracts importable by every stage: jaxtyping aliases, the ``DMat[Ax]``
    axiom lattice, ``TestResult``/``EstimateResult``, deterministic PRNG keys.
:mod:`jcor.optimize`
    Cross-cutting numerical substrate: simplex-constrained PGD, PSD repair.
:mod:`jcor.ground`
    S2 — ``d: X x X -> R`` on points (euclidean, angular, cosine, cdist/pdist).
:mod:`jcor.discrepancy`
    S3, **waist 1** — ``d: P x P -> R`` on measures: energy, transport, exact
    Wasserstein, metrization, energy kernels.
:mod:`jcor.association`
    S4 — distance matrix to a coefficient: Mantel, DISCO.
:mod:`jcor.geometry`
    S4 — distance matrix to structure: PCoA, Weiszfeld medians, energy
    barycentres, neighbour stability.
:mod:`jcor.operators`
    S5, **waist 2** — structured operators: long-run/HAC covariance.
:mod:`jcor.model`
    S6 — estimators over an operator plus data.
:mod:`jcor.inference`
    S7 — permutation, bootstrap, studentized and mixture tests; block length.
:mod:`jcor.decision`
    S8 — equivalence/TOST, multiplicity control, Sharpe.
:mod:`jcor.experimental`
    Staging area. **No semver promise.**

Naming grammar
--------------
``*_distance`` / ``*_statistic`` return a scalar; ``*_test`` returns a
``TestResult``; ``*_fit`` / ``*_estimate`` return an ``EstimateResult``;
``*_kernel`` is a jit-ready inner function; a leading ``_`` promises nothing.
Namespace-redundant prefixes are kept deliberately
(``discrepancy.energy_distance``), as scipy does.

Root surface
------------
The names below are the *headline* surface — the entry points a first-time
reader reaches for. They are re-exports; the owning stage module is the
documented import path and the one that keeps working when this list changes.
Everything else lives on its stage. t46.10 shrank this from 49 flat exports and
deleted the pre-t46 module paths (``jcor.metrics``, ``jcor.statistics``,
``jcor.wasserstein``, ``jcor.kernels``, ``jcor.permutation``,
``jcor.hypothesis``, ``jcor.equivalence``, ``jcor.block_length``,
``jcor.mantel``, ``jcor.seeds``, ``jcor.pit_energy``,
``jcor._energy_components``).
"""

from __future__ import annotations

from jcor import (
    association,
    core,
    decision,
    discrepancy,
    experimental,
    geometry,
    ground,
    inference,
    model,
    operators,
    optimize,
    sample,
)
from jcor.association.mantel import mantel_test
from jcor.core.random import cell_key
from jcor.core.results import EstimateResult, TestResult
from jcor.decision import equivalence_test
from jcor.discrepancy.energy import energy_distance, energy_test_statistic
from jcor.discrepancy.transport import w1_1d, w2_1d
from jcor.geometry import energy_barycentre_weights
from jcor.ground.metrics import cdist, euclidean_distance, pdist
from jcor.inference import permutation_test

__version__ = "0.1.0"


__all__ = [
    "EstimateResult",
    "TestResult",
    "association",
    "cdist",
    "cell_key",
    "core",
    "decision",
    "discrepancy",
    "energy_barycentre_weights",
    "energy_distance",
    "energy_test_statistic",
    "equivalence_test",
    "euclidean_distance",
    "experimental",
    "geometry",
    "ground",
    "inference",
    "mantel_test",
    "model",
    "operators",
    "optimize",
    "pdist",
    "permutation_test",
    "sample",
    "w1_1d",
    "w2_1d",
]

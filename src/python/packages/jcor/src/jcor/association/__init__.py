"""S4 — functionals of a distance matrix that return a coefficient.

Position
--------
rank 4

Consumes
--------
one or two distance matrices (waist 1)

Produces
--------
a scalar or a decomposition: Mantel, distance correlation, DISCO.
These are *not* distances, which is why they are a stage above one

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

# --- t46.6 (mantel, dispersion) ---
from jcor.association.decomposition import (
    DiscoDecomposition,
    DiscoOrigin,
    DispersionDecomposableLaw,
    disco_components,
    group_design,
)
from jcor.association.decomposition import (
    disco_components as disco_decomposition,
)
from jcor.association.dispersion import (
    disco_decomposition as disco_decomposition_values,
)
from jcor.association.mantel import (
    MantelMethod,
    MantelTestResult,
    PairedMantelBootstrapResult,
    mantel_bootstrap_ci,
    mantel_test,
    paired_mantel_bootstrap,
)

__all__ += [
    "DiscoDecomposition",
    "DiscoOrigin",
    "DispersionDecomposableLaw",
    "MantelMethod",
    "MantelTestResult",
    "PairedMantelBootstrapResult",
    "disco_components",
    "disco_decomposition",
    "disco_decomposition_values",
    "group_design",
    "mantel_bootstrap_ci",
    "mantel_test",
    "paired_mantel_bootstrap",
]

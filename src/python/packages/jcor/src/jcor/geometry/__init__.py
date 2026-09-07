"""S4 — geometric structure derived from a distance matrix.

Position
--------
rank 4

Consumes
--------
a distance matrix (waist 1)

Produces
--------
embeddings, barycentres, medians, neighbour graphs

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

This package is a t46 skeleton: modules land here via the t46.4-t46.9
migrations. Re-exports are appended per-owner, in the region marked below.
"""

from __future__ import annotations

from jcor.geometry._barycentre import (
    EnergyBarycentreDiagnostics,
    KernelBarycentreDiagnostics,
    WassersteinBarycentreResult,
    energy_barycentre_weights,
    energy_barycentre_weights_batched,
    energy_barycentre_weights_pairwise,
    kernel_barycentre_weights,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)
from jcor.geometry.embedding import (
    EIGENVALUE_TOLERANCE,
    NonEuclideanEmbeddingError,
    pcoa,
    pcoa_matrix,
)
from jcor.geometry.median import (
    WEISZFELD_ITERATIONS,
    dispersion_f_ratio,
    dispersion_permutation_null,
    weiszfeld_medians,
)
from jcor.geometry.neighbour import (
    NEIGHBOUR_INPUT_AXIOMS,
    NeighbourStabilityEdge,
    NeighbourStabilitySummary,
    cross_encoder_neighbour_stability,
    neighbour_order,
)

__all__: list[str] = []

# --- t46 migration re-exports; each task appends only to its own region ---

# --- t22 (typed kernel barycentre) ---
__all__ += [
    "EnergyBarycentreDiagnostics",
    "KernelBarycentreDiagnostics",
    "WassersteinBarycentreResult",
    "energy_barycentre_weights",
    "energy_barycentre_weights_batched",
    "energy_barycentre_weights_pairwise",
    "kernel_barycentre_weights",
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
]

# --- t46.6 (embedding, median) ---
__all__ += [
    "EIGENVALUE_TOLERANCE",
    "WEISZFELD_ITERATIONS",
    "NonEuclideanEmbeddingError",
    "dispersion_f_ratio",
    "dispersion_permutation_null",
    "pcoa",
    "pcoa_matrix",
    "weiszfeld_medians",
]

# --- t46.9 (neighbour) ---
__all__ += [
    "NEIGHBOUR_INPUT_AXIOMS",
    "NeighbourStabilityEdge",
    "NeighbourStabilitySummary",
    "cross_encoder_neighbour_stability",
    "neighbour_order",
]

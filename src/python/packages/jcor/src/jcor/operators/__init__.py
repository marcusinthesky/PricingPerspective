"""S5 — structured operators a model consumes.  WAIST 2.

Position
--------
rank 5

Consumes
--------
geometry and/or raw panels

Produces
--------
a square operator with declared properties: PSD covariance, row-stochastic
spatial weights, a regression design.  ``sigma_dist`` and ``build_w_flat``
are the same move applied twice, which is why they live side by side

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
from jcor.operators.longrun import (
    bartlett_weights,
    circular_block_bootstrap_indices,
    hall_centered_hac,
    newey_west,
    optimal_bandwidth,
)

__all__ += [
    "bartlett_weights",
    "circular_block_bootstrap_indices",
    "hall_centered_hac",
    "newey_west",
    "optimal_bandwidth",
]
# --- end t46.8 ---

# --- t56.2 shared JAX primitives ---
from jcor.operators.longrun import (  # noqa: E402
    bartlett_weights_kernel,
    circular_block_bootstrap_indices_kernel,
    hall_centered_hac_kernel,
    newey_west_kernel,
)

__all__ += [
    "bartlett_weights_kernel",
    "circular_block_bootstrap_indices_kernel",
    "hall_centered_hac_kernel",
    "newey_west_kernel",
]
# --- end t56.2 ---

# --- dyadic design operators ---
from jcor.operators.design import (  # noqa: E402
    NodeIndex,
    drop_collinear_columns,
    fwl_coefficient,
    symmetric_dyadic_effects,
)

__all__ += [
    "NodeIndex",
    "drop_collinear_columns",
    "fwl_coefficient",
    "symmetric_dyadic_effects",
]

# --- covariance operators ---
from jcor.operators.covariance import (  # noqa: E402
    sigma_dist,
    sigma_dist_hetero_kappa,
    sigma_dist_hetero_kappa_kernel,
    sigma_dist_kernel,
)

__all__ += [
    "sigma_dist",
    "sigma_dist_hetero_kappa",
    "sigma_dist_hetero_kappa_kernel",
    "sigma_dist_kernel",
]

# --- typed meat doors (waist 2) ---
from jcor.operators.longrun import (  # noqa: E402
    hall_centered_variability,
    newey_west_variability,
)

__all__ += [
    "hall_centered_variability",
    "newey_west_variability",
]

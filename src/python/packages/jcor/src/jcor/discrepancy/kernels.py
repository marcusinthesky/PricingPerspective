"""Deprecated compatibility path for :mod:`jcor.discrepancy.balancing`.

The energy-distance simplex solvers exported here are balancing procedures, not
Mercer kernels.  Import from :mod:`jcor.discrepancy.balancing`; this path remains
only for the 0.3 compatibility window and is removed by the next namespace-
breaking jcor release.
"""

from __future__ import annotations

import warnings

from jcor.discrepancy.balancing import (
    energy_distance_kernel,
    energy_distance_kernel_mixture,
    energy_distance_kernel_pairwise,
    energy_test_kernel,
    energy_test_kernel_pairwise_max,
    inverse_variance_m_eff,
    kernel_balancing_weights,
)

warnings.warn(
    "jcor.discrepancy.kernels is deprecated; import jcor.discrepancy.balancing instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "energy_distance_kernel",
    "energy_distance_kernel_mixture",
    "energy_distance_kernel_pairwise",
    "energy_test_kernel",
    "energy_test_kernel_pairwise_max",
    "inverse_variance_m_eff",
    "kernel_balancing_weights",
]

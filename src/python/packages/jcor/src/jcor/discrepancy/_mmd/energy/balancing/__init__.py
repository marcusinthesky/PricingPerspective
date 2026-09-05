"""Energy-induced MMD balancing procedures."""

from __future__ import annotations

from jcor.discrepancy._mmd.energy.balancing.effective_size import (
    inverse_variance_m_eff,
)
from jcor.discrepancy._mmd.energy.balancing.mixture import (
    energy_distance_kernel_mixture,
)
from jcor.discrepancy._mmd.energy.balancing.testing import (
    energy_test_kernel,
    energy_test_kernel_pairwise_max,
)
from jcor.discrepancy._mmd.energy.balancing.weights import (
    energy_distance_kernel,
    energy_distance_kernel_pairwise,
    kernel_balancing_weights,
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

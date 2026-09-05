"""S3 — energy kernels: simplex weights over candidate *distributions*.

Position
--------
rank 3 · discrepancy. Consumes sample clouds and a ground metric; produces a
weight vector on the simplex of candidate distributions, plus the energy
statistic evaluated at those weights.

Why these are not "kernel methods"
----------------------------------
The former ``jcor.kernels`` package exported ``energy_distance_kernel*``,
``energy_test_kernel*``, ``kernel_balancing_weights`` and
``inverse_variance_m_eff`` — every one of them energy-distance machinery that
merely *uses* a solver. An ML reader reads "kernel" as Mercer/RBF, which is not
what any of this is. t46.6 dissolved that package: the *approximate,
surrogate-objective* weights stay here at the discrepancy stage, while the
*exact* energy barycentre — which needs the conditionally-negative-definite
geometry of the candidate Gram matrix — moved up to
:mod:`jcor.geometry._barycentre`.

This module is the canonical public path; the bodies live in the private
:mod:`jcor.discrepancy._mmd.energy.balancing` package.
"""

from __future__ import annotations

from jcor.discrepancy._mmd.energy.balancing.effective_size import inverse_variance_m_eff
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

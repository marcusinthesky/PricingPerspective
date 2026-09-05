"""Barycentre solvers — private sub-package.

Created private from the outset: the migrated bodies total ~710 lines
(``kernels/weights.py`` 423 + ``kernels/_barycentre.py`` 288), well over the
250-line view knee, so this is split once rather than twice.

Nothing outside ``jcor.geometry`` may import this package.
"""

from __future__ import annotations

from jcor.geometry._barycentre.energy import (
    EnergyBarycentreDiagnostics,
    _energy_barycentre_weights_batched_core,
    energy_barycentre_weights,
    energy_barycentre_weights_batched,
    energy_barycentre_weights_pairwise,
)
from jcor.geometry._barycentre.mmd import (
    KernelBarycentreDiagnostics,
    kernel_barycentre_weights,
)
from jcor.geometry._barycentre.wasserstein import (
    WassersteinBarycentreResult,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)

__all__ = [
    "EnergyBarycentreDiagnostics",
    "KernelBarycentreDiagnostics",
    "WassersteinBarycentreResult",
    "_energy_barycentre_weights_batched_core",
    "energy_barycentre_weights",
    "energy_barycentre_weights_batched",
    "energy_barycentre_weights_pairwise",
    "kernel_barycentre_weights",
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
]

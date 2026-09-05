"""Energy-MMD barycentre specialization.

Energy is a distance-induced MMD kernel, so the numerical implementation shares
the generic MMD simplex machinery.  This facade gives the concrete energy
specialization its discoverable ``_barycentre/energy`` home without duplicating
the solver or creating a second objective definition.
"""

from jcor.geometry._barycentre.mmd.solve import EnergyBarycentreDiagnostics
from jcor.geometry._barycentre.mmd.weights import (
    _energy_barycentre_weights_batched_core,
    energy_barycentre_weights,
    energy_barycentre_weights_batched,
    energy_barycentre_weights_pairwise,
)

__all__ = [
    "EnergyBarycentreDiagnostics",
    "_energy_barycentre_weights_batched_core",
    "energy_barycentre_weights",
    "energy_barycentre_weights_batched",
    "energy_barycentre_weights_pairwise",
]

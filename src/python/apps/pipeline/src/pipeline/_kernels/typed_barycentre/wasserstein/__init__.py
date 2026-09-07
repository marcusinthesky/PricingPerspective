"""Wasserstein barycentres, split by the ``problem`` the arm declares.

``projection`` solves ``target_projection`` (leave-one-out, simplex weights over
candidate clouds); ``measure`` solves ``source_barycentre`` (an explicit support
at prescribed source weights).  Same geometry family, different optimisation --
which is why they are siblings rather than one module.
"""

from __future__ import annotations

from pipeline._kernels.typed_barycentre.wasserstein.measure import (
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)
from pipeline._kernels.typed_barycentre.wasserstein.projection import (
    solve_wasserstein_target_projections,
)

__all__ = [
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
    "solve_wasserstein_target_projections",
]

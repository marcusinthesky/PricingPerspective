"""Pure-JAX Wasserstein barycentre solvers."""

from jcor.geometry._barycentre.wasserstein.measure import (
    WassersteinBarycentreResult,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)

__all__ = [
    "WassersteinBarycentreResult",
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
]

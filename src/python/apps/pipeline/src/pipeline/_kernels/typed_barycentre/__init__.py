"""Typed barycentre solvers used by the shared artifact stage.

One module per geometry family, mirroring ``jcor.geometry._barycentre`` so both
halves of a computation carry the same name: what ``jcor`` implements as pure
array geometry, this package wraps in the typed registry's vocabulary --
declared-arm validation, cloud IDs, certificates, artifact semantics.

Dependency direction inside the package runs one way, ``energy``/``mmd``/
``wasserstein`` -> :mod:`._diagnostics` -> :mod:`._common` -> nothing, so the
graph is acyclic by construction.

This facade is the supported import path.  It replaced a single 969-line module
of the same name, so every existing ``from pipeline._kernels.typed_barycentre
import ...`` keeps resolving unchanged.
"""

from __future__ import annotations

from pipeline._kernels.typed_barycentre._common import TypedDistanceError
from pipeline._kernels.typed_barycentre._diagnostics import (
    BarycentreDiagnostics,
    BarycentreResult,
    MeasureBarycentreKernelResult,
)
from pipeline._kernels.typed_barycentre.energy import solve_energy_target_projections
from pipeline._kernels.typed_barycentre.mmd import solve_mmd_target_projections
from pipeline._kernels.typed_barycentre.wasserstein import (
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
    solve_wasserstein_target_projections,
)

__all__ = [
    "BarycentreDiagnostics",
    "BarycentreResult",
    "MeasureBarycentreKernelResult",
    "TypedDistanceError",
    "solve_energy_target_projections",
    "solve_mmd_target_projections",
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
    "solve_wasserstein_target_projections",
]

"""Result and certificate shapes crossing the artifact boundary.

These are the pipeline's own vocabulary, not geometry: ``jcor`` deliberately
knows nothing about cloud IDs, arm IDs, or artifact schemas, so the solver
families here translate its array-level certificates into these.  Sits above
:mod:`._common` and below every family module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from jcor.geometry import EnergyBarycentreDiagnostics


@dataclass(frozen=True)
class BarycentreDiagnostics:
    """One target-projection solver certificate."""

    target_id: str
    objective: float
    initial_objective: float
    optimality_gap: float
    constraint_violation: float
    active_support_size: int
    converged: bool
    iterations: int
    solver: str
    feasible_set: str
    projected_gradient_norm: float = 0.0
    cnd_tangent_max_eigenvalue: float = 0.0
    tangent_curvature_min: float = 0.0
    simplex_sum_error: float = 0.0
    min_weight: float = 0.0
    psd_tangent_min_eigenvalue: float = 0.0
    assignment_gap_min: float = 0.0
    assignment_support_size: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialize solver diagnostics for the artifact boundary."""
        return {
            "target_id": self.target_id,
            "objective": self.objective,
            "initial_objective": self.initial_objective,
            "optimality_gap": self.optimality_gap,
            "constraint_violation": self.constraint_violation,
            "active_support_size": self.active_support_size,
            "converged": self.converged,
            "iterations": self.iterations,
            "solver": self.solver,
            "feasible_set": self.feasible_set,
            "projected_gradient_norm": self.projected_gradient_norm,
            "cnd_tangent_max_eigenvalue": self.cnd_tangent_max_eigenvalue,
            "tangent_curvature_min": self.tangent_curvature_min,
            "simplex_sum_error": self.simplex_sum_error,
            "min_weight": self.min_weight,
            "psd_tangent_min_eigenvalue": self.psd_tangent_min_eigenvalue,
            "assignment_gap_min": self.assignment_gap_min,
            "assignment_support_size": self.assignment_support_size,
        }


@dataclass(frozen=True)
class MeasureBarycentreKernelResult:
    """Source-measure empirical Wasserstein barycentre result."""

    support_ids: tuple[str, ...]
    support_values: np.ndarray
    source_ids: tuple[str, ...]
    source_weights: tuple[float, ...]
    objective: float
    initial_objective: float
    optimality_gap: float
    constraint_violation: float
    converged: bool
    iterations: int
    solver: str
    feasible_set: str
    objective_semantics: str
    #: Distance from the returned support to the alternating map's fixed point;
    #: ``0.0`` once the assignments repeat, and always ``0.0`` for the
    #: closed-form two-source solver.  Unlike ``optimality_gap`` -- which is only
    #: the improvement over the starting support -- this is a residual a reader
    #: can act on.
    support_shift: float = 0.0


@dataclass(frozen=True)
class BarycentreResult:
    """Leave-one-out weights and diagnostics in stable cloud order."""

    target_ids: tuple[str, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    weights: tuple[np.ndarray, ...]
    diagnostics: tuple[BarycentreDiagnostics, ...]
    arm_id: str
    geometry_id: str


def _certificate_scalars(
    certificate: EnergyBarycentreDiagnostics,
) -> tuple[float, float, float, float, float]:
    """Extract scalar solver diagnostics for one target projection."""
    return (
        float(certificate.projected_gradient_norm),
        float(certificate.cnd_tangent_max_eigenvalue),
        float(certificate.tangent_curvature_min),
        float(certificate.simplex_sum_error),
        float(certificate.min_weight),
    )


__all__ = [
    "BarycentreDiagnostics",
    "BarycentreResult",
    "MeasureBarycentreKernelResult",
]

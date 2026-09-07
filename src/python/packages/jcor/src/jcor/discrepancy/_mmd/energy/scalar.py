"""Scalar energy-distance estimators and their shared exponent contract.

Position
--------
rank 3 (WAIST 1) · private implementation behind
:mod:`jcor.discrepancy.energy`

The theorem/configuration root is :mod:`jcor.discrepancy._mmd.energy.geometry`.
This scalar layer depends on it and owns the shared eager resolver consumed by
matrix, test, and mixture siblings, keeping the private implementation graph
acyclic.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import jax.numpy as jnp
from jax import jit

from jcor.core.axioms import (
    SEMIMETRIC,
    Law,
    Premetric,
    Semimetric,
)

if TYPE_CHECKING:
    from jcor.core.axioms import Axioms
from jcor.core.typing import (  # noqa: TC001  # runtime; see jcor.core.typing
    Array,
    Float,
)
from jcor.discrepancy._mmd.energy.geometry import (
    UNIT_EXPONENT,
    DistributionExponent,
    EnergyGeometry,  # noqa: TC001  # runtime contract
    _DeclaredEnergyGeometry,
    _energy_result_law,
    parse_distribution_exponent,
)
from jcor.ground.config import (  # noqa: TC001  # runtime contract
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.ground.metrics import (  # noqa: TC001  # runtime contract
    ANGULAR,
    EUCLIDEAN,
    GroundDistance,
    cdist,
)

__all__ = [
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "IncompatibleEnergyGroundExponentError",
    "energy_distance",
    "energy_distance_with_geometry",
]

_DECLARED_ENERGY_GEOMETRIES: Final = {
    "energy_distance[euclidean]": (EUCLIDEAN, UNIT_EXPONENT),
    "energy_distance[angular]": (ANGULAR, UNIT_EXPONENT),
}

DECLARED_AXIOMS: Final[dict[str, Axioms]] = {
    name: _energy_result_law(ground.law, ground.name, exponent).axioms
    for name, (ground, exponent) in _DECLARED_ENERGY_GEOMETRIES.items()
}
"""Derived property-battery view; energy geometry laws are authoritative."""

DECLARED_BRANDS: Final[dict[str, type[Premetric]]] = {
    name: Semimetric if axioms & SEMIMETRIC == SEMIMETRIC else Premetric
    for name, axioms in DECLARED_AXIOMS.items()
}
"""Derived migration-only marker view; numerical producers never consume it."""


class IncompatibleEnergyGroundExponentError(ValueError):
    """Raised when a valid exponent lacks the selected ground-law theorem."""

    def __init__(self, metric: str, exponent: object) -> None:
        """Name both parts of the incompatible eager configuration."""
        super().__init__(
            f"ground distance {metric!r} is incompatible with "
            f"energy exponent {exponent!r}"
        )


def _resolve_distribution_inputs(
    exponent: float,
    metric: GroundDistanceSelection,
) -> tuple[DistributionExponent, GroundDistance[Law]]:
    """Parse an eager compatibility selection and validate its theorem branch."""
    validated_exponent = parse_distribution_exponent(exponent)
    resolved_metric = resolve_ground_distance(metric)
    try:
        _energy_result_law(
            resolved_metric.law,
            resolved_metric.name,
            validated_exponent,
        )
    except TypeError:
        raise IncompatibleEnergyGroundExponentError(
            resolved_metric.name,
            exponent,
        ) from None
    return validated_exponent, resolved_metric


@partial(jit, static_argnums=(2, 3))
def _energy_distance_core(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    exponent: float,
    metric: GroundDistance[Law],
) -> Float[Array, ""]:
    """Fused scalar energy-distance core (JIT-compiled).

    Computes the three ``cdist`` calls (``xy``, ``xx``, ``yy``), applies the
    exponent, and reduces to the scalar energy-distance statistic inside a
    single jitted trace so XLA can fuse the shared norm/matmul work rather
    than materializing three eager intermediates.

    Args:
        x: First sample, shape (n, d).
        y: Second sample, shape (m, d).
        exponent: Distance exponent (static).
        metric: Declared ground-distance strategy (static).

    Returns:
        Energy distance value.

    """
    n, m = x.shape[0], y.shape[0]
    dxy = cdist(x, y, metric=metric) ** exponent
    dxx = cdist(x, x, metric=metric) ** exponent
    dyy = cdist(y, y, metric=metric) ** exponent
    term1 = 2.0 * jnp.sum(dxy) / (n * m)
    term2 = jnp.sum(dxx) / (n * n)
    term3 = jnp.sum(dyy) / (m * m)
    return term1 - term2 - term3


def energy_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "angular",
) -> Float[Array, ""]:
    """Compute energy distance between two samples.

    Energy distance is a statistical distance between probability distributions
    based on expected distances between random samples. For samples X and Y:

    E(X,Y) = 2·E[||X-Y||^α] - E[||X-X'||^α] - E[||Y-Y'||^α]

    where X' and Y' are independent copies and α is the distance exponent.

    Axioms: see :data:`DECLARED_AXIOMS` — ``SEMIMETRIC`` under
    ``metric="euclidean"``, only ``NONNEGATIVE | SYMMETRY`` under the
    ``metric="angular"`` **default**, which cannot see scale. Typed metrization
    is available only through :func:`jcor.core.matrices.metrize_energy`, whose
    signature requires the missing ground and moment hypotheses.

    Args:
        x: First sample, array of shape (n, d).
        y: Second sample, array of shape (m, d).
        exponent: Distance exponent α. Must be in (0, 2) for valid metric.
            The default is 1.0; the distinct α=2 ANOVA regime is represented by
            :class:`AnovaExponent`, not accepted by this distribution distance.
        metric: Declared strategy or closed serialized built-in selection.

    Returns:
        Zero-dimensional JAX array containing the non-negative energy distance.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    Examples:
        >>> x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> y = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        >>> energy_distance(x, y, exponent=1.0)
        9.345  # Large distance between separated clusters

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    return _energy_distance_core(
        x,
        y,
        validated_exponent.value,
        resolved_metric,
    )


def energy_distance_with_geometry[
    GroundLawT: Law,
    ExponentT: DistributionExponent,
    EnergyLawT: Law,
](
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT],
) -> Float[Array, ""]:
    """Evaluate a prevalidated distribution-sensitive energy geometry.

    Args:
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``.
        geometry: Compatible typed ground/exponent/result-law construction.

    Returns:
        Zero-dimensional JAX energy functional.

    """
    if not isinstance(geometry, _DeclaredEnergyGeometry):
        message = (
            "geometry must be constructed by energy_geometry or unsafe_energy_geometry"
        )
        raise TypeError(message)
    return _energy_distance_core(
        x,
        y,
        geometry.exponent.value,
        geometry.ground,
    )

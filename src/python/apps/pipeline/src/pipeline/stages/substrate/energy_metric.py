"""Convert the stored energy functional to its host-side metric scale."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from statadic.geometry.discrepancy.metrize import (
    NegativeEnergyFunctionalError,
    NonfiniteEnergyFunctionalError,
    sqrt_energy_functional,
)

if TYPE_CHECKING:
    from jax.typing import ArrayLike

__all__ = [
    "ENERGY_FUNCTIONAL_SCALE",
    "ENERGY_METRIC_SCALE",
    "NegativeEnergyFunctionalError",
    "NonfiniteEnergyFunctionalError",
    "_energy_metric_from_functional",
]

ENERGY_FUNCTIONAL_SCALE = "energy_functional_S=2A-B_i-B_j"
ENERGY_METRIC_SCALE = "sqrt_energy_functional"


def _energy_metric_from_functional(functional: ArrayLike) -> np.ndarray:
    """Project the stored functional to the metric and land it back on the host.

    Args:
        functional: Stored energy functional ``S``, any shape.

    Returns:
        ``sqrt(S)`` as a host ``float64`` array.

    Raises:
        NonfiniteEnergyFunctionalError: If any value is NaN or infinite.
        NegativeEnergyFunctionalError: If any value is materially negative.

    """
    return np.asarray(sqrt_energy_functional(functional), dtype=np.float64)

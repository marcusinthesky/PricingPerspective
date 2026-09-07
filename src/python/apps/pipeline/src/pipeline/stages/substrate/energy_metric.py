"""Paper 1 artifact loading for the immutable shared energy functional.

What is left here is the **artifact** half: the two provenance strings that
label the stored scale, and the loader that reads the parquet the substrate
stage writes. The *method* half — ``sqrt(S)``, the semimetric-to-metric
conversion — moved to :mod:`jcor.discrepancy.metrize` (t46.5). It never was a
Paper 1 projection: this module's entire body was ``sqrt(S)`` plus two
validations, and a general conversion filed under a paper's directory is
invisible to every other consumer of it.

The error classes below are re-exported verbatim (same objects, so ``except
NegativeEnergyFunctionalError`` and ``is`` checks still hold) because the
consumers — ``substrate/{mantel,dimensionality,ablation}.py``,
``figures/paper1/*`` and their tests — are not this task's to edit.

``_energy_metric_from_functional`` is no longer a bare alias. Since t65 jcor
returns ``jax.Array``, and this app is where the conversion to the reporting
stack belongs: every consumer assigns the result into a pandas column or feeds
it to SciPy. Converting once here also closes the x64 scope-exit hazard — a
float64-labelled JAX array leaving ``jax.enable_x64`` silently computes its
*next* op in float32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jcor.discrepancy.metrize import (
    NegativeEnergyFunctionalError,
    NonfiniteEnergyFunctionalError,
    sqrt_energy_functional,
)

from pipeline.stages.substrate.energy_shared import _load_energy_distances

if TYPE_CHECKING:
    from pathlib import Path

    from jax.typing import ArrayLike

__all__ = [
    "ENERGY_FUNCTIONAL_SCALE",
    "ENERGY_METRIC_SCALE",
    "NegativeEnergyFunctionalError",
    "NonfiniteEnergyFunctionalError",
    "_energy_metric_from_functional",
    "_load_energy_metric",
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


def _load_energy_metric(
    parquet_path: Path,
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> tuple[list[str], np.ndarray]:
    """Load the raw shared functional and explicitly project it to the metric.

    Args:
        parquet_path: Path to the stored pairwise energy-functional parquet.
        provider_id: Expected provider identity in the typed artifact.
        representation_id: Expected representation identity in the typed artifact.
        distance_id: Expected statistical-distance identity in the typed artifact.

    Returns:
        Tuple of (tickers, ``sqrt(S)`` matrix).

    """
    tickers, functional = _load_energy_distances(
        parquet_path,
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id=distance_id,
    )
    return tickers, _energy_metric_from_functional(functional)

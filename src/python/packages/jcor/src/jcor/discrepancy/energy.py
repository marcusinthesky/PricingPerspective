"""Stable façade for S3 energy discrepancies on empirical measures.

Position
--------
rank 3 (WAIST 1) · consumes sample clouds and a ground metric · produces a
scalar discrepancy or a distance matrix between distributions

This high-fan-in module is the permanent public import surface.  Implementations
are split by cohesive responsibility under the private MMD family at
``_mmd.energy``; callers should continue importing public functions from
:mod:`jcor.discrepancy.energy`.

Axioms are a declared property, not a name (t46): see :data:`DECLARED_AXIOMS`.
The declaration is **inherited from the ground metric** — over a separating
ground metric (``metric="euclidean"``) energy distance is a semimetric, but over
its own *default* (``metric="angular"``) it is not, because angular distance
cannot see scale and therefore ``E(X, 3X) == 0`` for any all-positive ``X``.
That is measured, not assumed (t46.1 Finding 2).

The ground/exponent theorem matrix
----------------------------------
``exponent`` is parsed into subunit, unit, or superunit regimes at every eager
compatibility door, then checked against the resolved ground law. Euclidean and
normalized Euclidean support the full open interval ``(0, 2)``; angular and
cosine stop at one because explicit four-point CND witnesses rule out larger
powers. Compiled cores receive only the validated numeric value and declared
strategy, so no string dispatch or tracer-dependent validation enters them.

Third-party theorem extensions should use :func:`energy_geometry` whenever
their structural ground law proves a supported branch. The conspicuous
:func:`unsafe_energy_geometry` escape hatch bypasses only that result-law
implication; it still validates the JAX backend, nominal exponent regime,
runtime law descriptor, and hashability, and it does not waive the ground's
conditional value-domain requirements.
"""

from __future__ import annotations

from jcor.discrepancy._mmd.energy.geometry import (
    ANOVA_EXPONENT,
    DEFAULT_DISTRIBUTION_EXPONENT,
    UNIT_EXPONENT,
    AnovaEnergyLaw,
    AnovaExponent,
    AnovaGroundLaw,
    DistributionEnergyGroundLaw,
    DistributionEnergyLaw,
    DistributionExponent,
    EnergyExponent,
    EnergyGeometry,
    FullFractionalPowerGroundLaw,
    MeanEnergyLaw,
    SeparatingDistributionEnergyGroundLaw,
    SeparatingDistributionEnergyLaw,
    SeparatingStrongNegativeTypeGroundLaw,
    SeparatingSubunitPowerGroundLaw,
    StrongFullFractionalPowerGroundLaw,
    SubunitExponent,
    SubunitPowerGroundLaw,
    SuperunitExponent,
    UnitExponent,
    UnitPowerMeanGroundLaw,
    energy_geometry,
    parse_distribution_exponent,
    parse_energy_exponent,
    unsafe_energy_geometry,
)
from jcor.discrepancy._mmd.energy.matrix import (
    energy_distance_matrix as energy_distance_matrix_values,
)
from jcor.discrepancy._mmd.energy.matrix import (
    energy_functional_matrix,
    mean_distance_matrix,
)
from jcor.discrepancy._mmd.energy.mixture import (
    # Re-exported for `jcor.inference.{mixture,studentized}`, which import it
    # from this facade rather than reaching into the private module. Unused
    # *here* and deliberately absent from `__all__`, so ruff reads it as F401
    # and `--fix` deletes it — which breaks both importers and the seam
    # regression at `tests/discrepancy/test_energy.py:55-58`. The suppression is
    # what makes `prek run --all-files` idempotent instead of self-breaking.
    _pooled_from_candidates,  # noqa: F401
    mixture_energy_distance,
)
from jcor.discrepancy._mmd.energy.scalar import (
    DECLARED_AXIOMS,
    DECLARED_BRANDS,
    IncompatibleEnergyGroundExponentError,
    energy_distance,
    energy_distance_with_geometry,
)
from jcor.discrepancy._mmd.energy.testing import (
    energy_test_statistic,
    energy_test_statistic_from_blocks,
    energy_test_statistic_from_distances,
)

# Canonical public matrix producer: domain, law, scheme, and origin propagate.
energy_distance_matrix = energy_functional_matrix

# Stable internal seam used by rank-7 inference.  It remains outside public
# ``__all__`` so callers do not learn the private MMD implementation topology.

__all__ = [
    "ANOVA_EXPONENT",
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "DEFAULT_DISTRIBUTION_EXPONENT",
    "UNIT_EXPONENT",
    "AnovaEnergyLaw",
    "AnovaExponent",
    "AnovaGroundLaw",
    "DistributionEnergyGroundLaw",
    "DistributionEnergyLaw",
    "DistributionExponent",
    "EnergyExponent",
    "EnergyGeometry",
    "FullFractionalPowerGroundLaw",
    "IncompatibleEnergyGroundExponentError",
    "MeanEnergyLaw",
    "SeparatingDistributionEnergyGroundLaw",
    "SeparatingDistributionEnergyLaw",
    "SeparatingStrongNegativeTypeGroundLaw",
    "SeparatingSubunitPowerGroundLaw",
    "StrongFullFractionalPowerGroundLaw",
    "SubunitExponent",
    "SubunitPowerGroundLaw",
    "SuperunitExponent",
    "UnitExponent",
    "UnitPowerMeanGroundLaw",
    "energy_distance",
    "energy_distance_matrix",
    "energy_distance_matrix_values",
    "energy_distance_with_geometry",
    "energy_geometry",
    "energy_test_statistic",
    "energy_test_statistic_from_blocks",
    "energy_test_statistic_from_distances",
    "mean_distance_matrix",
    "mixture_energy_distance",
    "parse_distribution_exponent",
    "parse_energy_exponent",
    "unsafe_energy_geometry",
]

"""S3 — discrepancies: ``d: P x P -> R`` on measures/samples.  WAIST 1.

Position
--------
rank 3

Consumes
--------
sample clouds and a ground metric

Produces
--------
a distance matrix between distributions.  Axiom-neutral by name: it holds
true metrics (Wasserstein), semimetrics (energy distance) and, later,
divergences (Sinkhorn, KL) side by side

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

This package is a t46 skeleton: modules land here via the t46.4-t46.9
migrations. Re-exports are appended per-owner, in the region marked below.
"""

from __future__ import annotations

__all__: list[str] = []

# --- t46 migration re-exports; each task appends only to its own region ---

# --- t50 (mmd) ---
from jcor.discrepancy.mmd import (
    DECLARED_AXIOMS as MMD_DECLARED_AXIOMS,
)
from jcor.discrepancy.mmd import (
    DECLARED_BRANDS as MMD_DECLARED_BRANDS,
)
from jcor.discrepancy.mmd import (
    InsufficientUStatisticSampleError,
    NonfiniteMeanEmbeddingError,
    UnbiasedMmdDistanceError,
    ZeroNormMeanEmbeddingError,
    cosine_mean_mmd,
    cosine_mean_mmd_functional,
    mmd,
    mmd_matrix,
    mmd_squared,
    mmd_squared_matrix,
)

__all__ += [
    "MMD_DECLARED_AXIOMS",
    "MMD_DECLARED_BRANDS",
    "InsufficientUStatisticSampleError",
    "NonfiniteMeanEmbeddingError",
    "UnbiasedMmdDistanceError",
    "ZeroNormMeanEmbeddingError",
    "cosine_mean_mmd",
    "cosine_mean_mmd_functional",
    "mmd",
    "mmd_matrix",
    "mmd_squared",
    "mmd_squared_matrix",
]

# --- t46.9 (exact) ---
from jcor.discrepancy.exact import (  # noqa: E402
    EXACT_BALANCED_W1_AXIOMS,
    EXACT_BALANCED_W1_BRAND,
    EXACT_BALANCED_WASSERSTEIN_AXIOMS,
    EXACT_BALANCED_WASSERSTEIN_BRAND,
    EmptyOrNonMatrixSampleError,
    InvalidCostMatrixError,
    InvalidWassersteinOrderError,
    NegativeCostError,
    NonfiniteCostError,
    NonfiniteSampleError,
    OffUnitSphereSampleError,
    UnbalancedSampleShapeError,
    ZeroNormSampleError,
    balanced_wasserstein1_from_cost,
    balanced_wasserstein1_from_cost_jax,
    balanced_wasserstein_from_cost,
    balanced_wasserstein_from_cost_jax,
    chord_cost_matrix,
    chord_cost_matrix_jax,
    exact_balanced_wasserstein,
    exact_balanced_wasserstein1,
    exact_balanced_wasserstein1_jax,
    exact_balanced_wasserstein_jax,
)

__all__ += [
    "EXACT_BALANCED_W1_AXIOMS",
    "EXACT_BALANCED_W1_BRAND",
    "EXACT_BALANCED_WASSERSTEIN_AXIOMS",
    "EXACT_BALANCED_WASSERSTEIN_BRAND",
    "EmptyOrNonMatrixSampleError",
    "InvalidCostMatrixError",
    "InvalidWassersteinOrderError",
    "NegativeCostError",
    "NonfiniteCostError",
    "NonfiniteSampleError",
    "OffUnitSphereSampleError",
    "UnbalancedSampleShapeError",
    "ZeroNormSampleError",
    "balanced_wasserstein1_from_cost",
    "balanced_wasserstein1_from_cost_jax",
    "balanced_wasserstein_from_cost",
    "balanced_wasserstein_from_cost_jax",
    "chord_cost_matrix",
    "chord_cost_matrix_jax",
    "exact_balanced_wasserstein",
    "exact_balanced_wasserstein1",
    "exact_balanced_wasserstein1_jax",
    "exact_balanced_wasserstein_jax",
]

# --- t46.5 (energy, transport, metrize) ---
from jcor.discrepancy.energy import (  # noqa: E402
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
    IncompatibleEnergyGroundExponentError,
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
    energy_distance,
    energy_distance_matrix,
    energy_distance_with_geometry,
    energy_geometry,
    energy_test_statistic,
    energy_test_statistic_from_distances,
    mean_distance_matrix,
    mixture_energy_distance,
    parse_distribution_exponent,
    parse_energy_exponent,
    unsafe_energy_geometry,
)
from jcor.discrepancy.energy import (  # noqa: E402
    DECLARED_AXIOMS as ENERGY_DECLARED_AXIOMS,
)
from jcor.discrepancy.energy import (  # noqa: E402
    DECLARED_BRANDS as ENERGY_DECLARED_BRANDS,
)
from jcor.discrepancy.metrize import (  # noqa: E402
    NegativeEnergyFunctionalError,
    NonfiniteEnergyFunctionalError,
    sqrt_energy_functional,
)
from jcor.discrepancy.transport import (  # noqa: E402
    DECLARED_AXIOMS as TRANSPORT_DECLARED_AXIOMS,
)
from jcor.discrepancy.transport import (  # noqa: E402
    DECLARED_BRANDS as TRANSPORT_DECLARED_BRANDS,
)
from jcor.discrepancy.transport import (  # noqa: E402
    sliced_wasserstein,
    w1_1d,
    w2_1d,
)

__all__ += [
    "ANOVA_EXPONENT",
    "DEFAULT_DISTRIBUTION_EXPONENT",
    "ENERGY_DECLARED_AXIOMS",
    "ENERGY_DECLARED_BRANDS",
    "TRANSPORT_DECLARED_AXIOMS",
    "TRANSPORT_DECLARED_BRANDS",
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
    "NegativeEnergyFunctionalError",
    "NonfiniteEnergyFunctionalError",
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
    "energy_distance_with_geometry",
    "energy_geometry",
    "energy_test_statistic",
    "energy_test_statistic_from_distances",
    "mean_distance_matrix",
    "mixture_energy_distance",
    "parse_distribution_exponent",
    "parse_energy_exponent",
    "sliced_wasserstein",
    "sqrt_energy_functional",
    "unsafe_energy_geometry",
    "w1_1d",
    "w2_1d",
]

# --- t88 (exact w2) ---
from jcor.discrepancy.exact import (  # noqa: E402
    EXACT_BALANCED_W2_AXIOMS,
    EXACT_BALANCED_W2_BRAND,
    balanced_wasserstein2_from_cost,
    balanced_wasserstein2_from_cost_jax,
    exact_balanced_wasserstein2,
    exact_balanced_wasserstein2_jax,
)

__all__ += [
    "EXACT_BALANCED_W2_AXIOMS",
    "EXACT_BALANCED_W2_BRAND",
    "balanced_wasserstein2_from_cost",
    "balanced_wasserstein2_from_cost_jax",
    "exact_balanced_wasserstein2",
    "exact_balanced_wasserstein2_jax",
]

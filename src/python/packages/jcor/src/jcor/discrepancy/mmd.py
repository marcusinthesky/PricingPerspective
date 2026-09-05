"""S3 maximum mean discrepancy on empirical measures.

Position
--------
rank 3 (WAIST 1) · consumes sample clouds and a ground kernel · produces raw
MMD² estimates or V-statistic distance matrices between distributions

This stable facade is split behind cohesive private owners: ``_mmd.strategy``
(law-bearing kernel construction), ``_mmd.scalar`` (scalar estimates),
``_mmd.matrix`` (packed numerics), ``_mmd.branded`` (typed matrices), and
``_mmd.functional`` (the checked prototype artifact door).

The identity, on the right estimator
------------------------------------
Székely & Rizzo define squared energy distance
``D² = 2 E d(X,Y) - E d(X,X') - E d(Y,Y')`` and energy distance ``D`` as its
square root. JCOR's :func:`jcor.discrepancy.energy.energy_distance` is ``D²``;
typed rooting retains its hypotheses through
:func:`jcor.core.matrices.metrize_energy`.

:func:`jcor.ground.similarities.distance_induced_gram` implements the unhalved
Lean kernel ``d(x,z0) + d(y,z0) - d(x,y)``.  Consequently
``mmd_squared(..., unbiased=False) == energy_distance(...)`` at constant one,
as proved by ``MeanEmbedding.lean:525`` (``norm_sub_energyEmbed_sq``).  Under a
half-normalized kernel, ``D² = 2 MMD²`` and ``D = sqrt(2) MMD`` instead.

The estimator convention is part of the contract.  The energy implementation
is a V-statistic, so only V-statistic MMD distances are rootable. An unbiased
finite-sample MMD² U-statistic may be
negative.  :func:`mmd_squared` and :func:`mmd_squared_matrix` return that signed,
unbranded estimate; :func:`mmd` and :func:`mmd_matrix` reject ``unbiased=True``
instead of clamping, rooting, and falsely branding it as a distance.

Validated parameter domains
---------------------------
RBF bandwidths are concrete, finite and strictly positive. Distance-induced
exponents are concrete and finite, and the selected ground law must justify the
regime: Euclidean and normalized Euclidean support ``0 < alpha < 2`` while the
angular and cosine CND guarantees stop at ``alpha = 1``. At ``alpha = 2`` the
Euclidean energy functional collapses to a comparison of means and loses
distributional separation, so every MMD distance door excludes it. Cosine at
``alpha = 1`` is already mean-embedding-only; angular at one and normalized
Euclidean on raw vectors are nonseparating.

U-statistic within-sample blocks require at least two observations. Nonfinite
samples are outside every declared domain. Angular, cosine, and normalized
Euclidean grounds—and the cosine kernel—divide by a row norm, so they also
require every sample row to have a norm that is nonzero *and* finite. Finiteness
of the row is not enough: ``np.linalg.norm`` forms a sum of squares, so a finite
row of order ``1e200`` normalizes by ``inf`` and would silently become the zero
vector. ``cosine_mean_mmd`` is different after the normalization boundary: its
inputs are finite, already-unit prototypes supplied by the pipeline. On that
domain it is Euclidean chord distance, hence a separating Hilbertian metric of
negative type. Transformable JAX paths assume that domain; the NumPy functional
door checks it before artifact construction.

The split across kernels is deliberate. Only the normalizing sample paths raise,
because only they turn an overflow into a plausible finite number: the collapsed
Gram measured ``0.19512147 -> 0.0``. The linear kernel and the Euclidean
ground divide by nothing, so their overflow stays visible as ``inf``/``nan`` and
is deliberately left unchecked rather than converted into a typed error.
Checked carrier evidence and raw-vector versus direction-quotient propagation
belong to T55.3.

The static overloads branch on mathematical capability, never built-in names.
A characteristic strategy yields ``HilbertianMetricLaw``; any declared PSD
strategy yields ``HilbertianPseudometricLaw``. Bare names and callables are not
accepted. Distance-induced strategies close a validated ``EnergyGeometry``, so
an incompatible exponent cannot survive into MMD construction.

A Mercer Gram is not a distance Gram
------------------------------------
The PSD Mercer form used here is the centred, sign-flipped counterpart of the
conditionally-negative-definite distance Gram consumed by
``jcor.geometry._barycentre``.  Do not feed ``ground.similarities.gram`` directly
to that solver.  Build distribution distances through :func:`mmd_matrix`.

The float64 doors are separate because jcor defaults to float32. On
Paper-1-shaped prototype inputs the float32 route measured a ``4.9e-04``
self-distance, while :func:`cosine_mean_mmd_functional` preserves the locked
float64 comparator and rejects nonfinite or off-sphere prototypes.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from jcor.core.axioms import (
    METRIC,
    PSEUDOMETRIC,
    Axioms,
    NegativeType,
    Premetric,
    Semimetric,
)
from jcor.discrepancy._mmd.branded import (
    mmd_squared_estimate_matrix,
    rooted_mmd_matrix,
)
from jcor.discrepancy._mmd.contracts import (
    InsufficientUStatisticSampleError,
    NonfiniteMeanEmbeddingError,
    UnbiasedMmdDistanceError,
    ZeroNormMeanEmbeddingError,
)
from jcor.discrepancy._mmd.energy.geometry import UNIT_EXPONENT, energy_geometry
from jcor.discrepancy._mmd.functional import (
    cosine_mean_mmd_functional,
)
from jcor.discrepancy._mmd.matrix import (
    mmd_matrix_values,
    mmd_squared_matrix,
)
from jcor.discrepancy._mmd.scalar import cosine_mean_mmd, mmd, mmd_squared
from jcor.discrepancy._mmd.strategy import distance_induced_kernel
from jcor.ground.metrics import ANGULAR, EUCLIDEAN
from jcor.ground.similarities import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    CharacteristicKernelLaw,
    CharacteristicKernelProperties,
    PositiveSemidefiniteKernelLaw,
    PositiveSemidefiniteKernelProperties,
    SimilarityStrategy,
    declare_similarity,
    rbf_kernel,
)

# Preserve the producer's overload set under the concise public name.
mmd_matrix = rooted_mmd_matrix


def rooted_mmd_axioms(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
) -> Axioms:
    """Derive the rooted MMD law from the kernel capability object."""
    if isinstance(kernel.law, CharacteristicKernelLaw):
        return METRIC
    return PSEUDOMETRIC


def squared_mmd_axioms(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
) -> Axioms:
    """Derive the squared-functional law without assigning triangle inequality."""
    result = Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY
    if isinstance(kernel.law, CharacteristicKernelLaw):
        result |= Axioms.SEPARATION
    return result


_RBF_KERNEL = rbf_kernel(1.0)
_EUCLIDEAN_ENERGY_KERNEL = distance_induced_kernel(
    energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
)
_ANGULAR_ENERGY_KERNEL = distance_induced_kernel(
    energy_geometry(ANGULAR, UNIT_EXPONENT)
)

BUILTIN_MMD_KERNELS: Final = MappingProxyType(
    {
        "rbf": _RBF_KERNEL,
        "distance_induced[euclidean]": _EUCLIDEAN_ENERGY_KERNEL,
        "distance_induced[angular]": _ANGULAR_ENERGY_KERNEL,
        "linear": LINEAR_KERNEL,
        "cosine": COSINE_KERNEL,
    }
)
"""Named documentation registry; values are the sole executable declarations."""

DECLARED_AXIOMS: Final = {
    **{
        f"mmd[{name}]": rooted_mmd_axioms(kernel)
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    **{
        f"mmd_squared[{name}]": squared_mmd_axioms(kernel)
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    "cosine_mean_mmd": METRIC,
}
"""Derived compatibility view used by the axiom battery; never hand-authored."""

DECLARED_BRANDS: Final = {
    **{
        f"mmd[{name}]": (
            NegativeType
            if isinstance(kernel.law, CharacteristicKernelLaw)
            else Premetric
        )
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    **{
        f"mmd_squared[{name}]": (
            Semimetric if isinstance(kernel.law, CharacteristicKernelLaw) else Premetric
        )
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    "cosine_mean_mmd": NegativeType,
}
"""Derived legacy marker view; typed producers use structural law protocols."""

__all__ = [
    "BUILTIN_MMD_KERNELS",
    "COSINE_KERNEL",
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "LINEAR_KERNEL",
    "CharacteristicKernelLaw",
    "CharacteristicKernelProperties",
    "InsufficientUStatisticSampleError",
    "NonfiniteMeanEmbeddingError",
    "PositiveSemidefiniteKernelLaw",
    "PositiveSemidefiniteKernelProperties",
    "SimilarityStrategy",
    "UnbiasedMmdDistanceError",
    "ZeroNormMeanEmbeddingError",
    "cosine_mean_mmd",
    "cosine_mean_mmd_functional",
    "declare_similarity",
    "distance_induced_kernel",
    "mmd",
    "mmd_matrix",
    "mmd_matrix_values",
    "mmd_squared",
    "mmd_squared_estimate_matrix",
    "mmd_squared_matrix",
    "rbf_kernel",
    "rooted_mmd_axioms",
    "squared_mmd_axioms",
]

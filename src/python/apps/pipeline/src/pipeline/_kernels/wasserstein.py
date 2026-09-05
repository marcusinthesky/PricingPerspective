"""Re-export shim — the exact balanced Wasserstein kernel lives in ``jcor``.

Promoted by t46.9 to :mod:`jcor.discrepancy.exact`. The names below are the
*same objects*, not copies, so every ``except`` clause in this package still
catches the exception types it always did.

t50 added the cosine-mean comparator alongside them, promoted to
:mod:`jcor.discrepancy.mmd`. It arrives through the same shim for the same
reason: ``dvc.yaml`` lists this path as a dependency of
``p1_distributional_comparator``, so a stage-visible import belongs here rather
than scattered through the stage module.

This path stays because ``dvc.yaml`` lists it as a dependency of
``p1_distributional_comparator`` and ``tests/papers/paper1/test_wasserstein_dvc.py``
asserts that. t46.10 owns retiring the shim together with the ``dvc.yaml`` edit.
"""

from __future__ import annotations

from jcor.discrepancy.exact import (
    EXPECTED_ARRAY_DIMENSIONS,
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
    balanced_wasserstein2_from_cost,
    balanced_wasserstein2_from_cost_jax,
    balanced_wasserstein_from_cost,
    balanced_wasserstein_from_cost_jax,
    chord_cost_matrix,
    chord_cost_matrix_jax,
    exact_balanced_wasserstein,
    exact_balanced_wasserstein1,
    exact_balanced_wasserstein1_jax,
    exact_balanced_wasserstein2,
    exact_balanced_wasserstein2_jax,
    exact_balanced_wasserstein_jax,
)
from jcor.discrepancy.mmd import (
    NonfiniteMeanEmbeddingError,
    ZeroNormMeanEmbeddingError,
    cosine_mean_mmd_functional,
)

__all__ = [
    "EXPECTED_ARRAY_DIMENSIONS",
    "EmptyOrNonMatrixSampleError",
    "InvalidCostMatrixError",
    "InvalidWassersteinOrderError",
    "NegativeCostError",
    "NonfiniteCostError",
    "NonfiniteMeanEmbeddingError",
    "NonfiniteSampleError",
    "OffUnitSphereSampleError",
    "UnbalancedSampleShapeError",
    "ZeroNormMeanEmbeddingError",
    "ZeroNormSampleError",
    "balanced_wasserstein1_from_cost",
    "balanced_wasserstein1_from_cost_jax",
    "balanced_wasserstein2_from_cost",
    "balanced_wasserstein2_from_cost_jax",
    "balanced_wasserstein_from_cost",
    "balanced_wasserstein_from_cost_jax",
    "chord_cost_matrix",
    "chord_cost_matrix_jax",
    "cosine_mean_mmd_functional",
    "exact_balanced_wasserstein",
    "exact_balanced_wasserstein1",
    "exact_balanced_wasserstein1_jax",
    "exact_balanced_wasserstein2",
    "exact_balanced_wasserstein2_jax",
    "exact_balanced_wasserstein_jax",
]

"""Solver constants and candidate-cloud validation for the energy barycentre.

The sink of :mod:`jcor.geometry._barycentre`: it imports no sibling, so every
other module in the sub-package may depend on it without creating a cycle.

``_MAX_DISTANCE_EXPONENT`` is deliberately duplicated from
:mod:`jcor.geometry._barycentre.mmd._common` — see that module's docstring for why
one shared float is not worth a cross-stage import edge.
"""

from __future__ import annotations

from typing import Final

import jax.numpy as jnp

# Reduced-QP solver constants for the exact energy barycentre.
_QP_RIDGE_REL: Final = 1e-12  # relative ridge on the reduced Hessian for convexity
_QP_MAX_ITER: Final = (
    500  # ill-conditioned 51-candidate projections need a larger IP budget
)
# The interior-point solver targets a tolerance this much tighter than the
# caller's acceptance gap, so that even an early KKT-residual stop clears the
# Frank-Wolfe-gap gate with margin (the two metrics differ by an O(1) factor).
_QP_TOL_FACTOR: Final = 1e-2
_MAX_DISTANCE_EXPONENT: Final = 2.0
_DEGENERATE_CURVATURE: Final = 1e-14
_MATRIX_NDIM: Final = 2
_STACKED_CLOUD_NDIM: Final = 3
_MIN_CANDIDATES: Final = 2


def _candidate_list(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
) -> list[jnp.ndarray]:
    """Normalize and validate stacked/list candidate inputs.

    Returns:
        Validated list of two-dimensional candidate arrays.

    """
    target = jnp.asarray(target)
    if target.ndim != _MATRIX_NDIM or target.shape[0] == 0:
        message = "target must be a non-empty array of shape (n, d)"
        raise ValueError(message)

    if isinstance(candidates, jnp.ndarray):
        if candidates.ndim != _STACKED_CLOUD_NDIM:
            message = "stacked candidates must have shape (K, m, d)"
            raise ValueError(message)
        candidates_list = [candidates[i] for i in range(candidates.shape[0])]
    else:
        candidates_list = [jnp.asarray(candidate) for candidate in candidates]

    if not candidates_list:
        message = "at least one candidate distribution is required"
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(target))):
        message = "target contains non-finite values"
        raise ValueError(message)
    for candidate in candidates_list:
        if (
            candidate.ndim != _MATRIX_NDIM
            or candidate.shape[0] == 0
            or candidate.shape[1] != target.shape[1]
        ):
            message = (
                "each candidate must be non-empty with the target feature dimension"
            )
            raise ValueError(message)
        if not bool(jnp.all(jnp.isfinite(candidate))):
            message = "candidates contain non-finite values"
            raise ValueError(message)
    return candidates_list

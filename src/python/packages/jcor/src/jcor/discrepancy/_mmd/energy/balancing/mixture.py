"""Energy distance from a target sample to a weighted mixture of candidates.

The evaluation half of the kernel family: given weights from any source
(:mod:`jcor.discrepancy._mmd.energy.balancing.weights`, the exact barycentre in
:mod:`jcor.geometry._barycentre`, or a caller's own procedure), score them.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcor.discrepancy._mmd.components import _compute_energy_components
from jcor.discrepancy._mmd.energy.balancing._common import _MAX_DISTANCE_EXPONENT
from jcor.ground.config import (  # noqa: TC001  # runtime annotations
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.ground.metrics import cdist


def energy_distance_kernel_mixture(
    target: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    weights: jnp.ndarray,
    metric: GroundDistanceSelection = "euclidean",
    exponent: float = 1.0,
    *,
    precomputed_components: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    precomputed_within: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Compute energy distance from target to weighted mixture of candidates.

    Evaluates the energy distance E(target, mixture) where mixture is defined
    by the convex combination: mixture = sum_k weights[k] * candidates[k].

    This is useful for evaluating the quality of mixture weights obtained from
    energy_distance_kernel or other optimization procedures.

    Args:
        target: Target distribution array of shape (n, d).
        candidates: Either:
            - Array of shape (K, m, d) with uniform sizes (backward compatible)
            - List of K arrays with shapes (m_k, d) for irregular sizes (recommended)
        weights: Mixture weights array of shape (K,), should sum to 1.
        metric: Distance metric (see cdist for options).
        exponent: Distance exponent α in (0, 2).
        precomputed_components: Optional ``(XX, YX)`` pair to reuse instead
            of recomputing the Gram/cross terms; see
            :func:`energy_barycentre_weights` for the bit-identical-input
            requirement.
        precomputed_within: Optional precomputed target within-term
            ``mean(cdist(target, target, metric=metric) ** exponent)`` (the
            ``YY`` term) to reuse instead of recomputing it.

    Returns:
        Energy distance between target and weighted mixture (non-negative scalar).

    Examples:
        >>> target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # 2 samples
        >>> cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])  # 3 samples
        >>> candidates = [cand1, cand2]
        >>> weights = jnp.array([1.0, 0.0])
        >>> energy_distance_kernel_mixture(target, candidates, weights)
        0.0

    """
    if not 0.0 < exponent < _MAX_DISTANCE_EXPONENT:
        message = f"exponent must be in (0, 2), got {exponent}"
        raise ValueError(message)
    resolved_metric = resolve_ground_distance(metric)

    # Convert uniform array to list if needed
    if isinstance(candidates, jnp.ndarray):
        candidates_list = [candidates[i] for i in range(candidates.shape[0])]
    else:
        candidates_list = candidates

    # Compute energy components
    if precomputed_components is not None:
        candidate_gram, target_cross = precomputed_components
    else:
        candidate_gram, target_cross = _compute_energy_components(
            target, candidates_list, resolved_metric, exponent
        )

    # Also need YY (mean distance within target)
    if precomputed_within is not None:
        target_self = precomputed_within
    else:
        dyy = cdist(target, target, metric=resolved_metric) ** exponent
        target_self = jnp.mean(dyy)

    # Energy distance for mixture:
    # E(Y, mixture) = 2*E[||Y - mixture||] - E[||Y - Y'||] - E[||mixture - mixture'||]
    #               = 2*w^T*YX - YY - w^T*XX*w
    term1 = 2 * jnp.dot(weights, target_cross)
    term2 = target_self
    term3 = jnp.dot(weights, jnp.dot(candidate_gram, weights))

    return term1 - term2 - term3

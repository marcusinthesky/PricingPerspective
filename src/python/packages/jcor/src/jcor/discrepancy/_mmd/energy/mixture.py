"""Weighted-mixture and barycentric helpers for energy distance.

Position
--------
rank 3 (WAIST 1) · private implementation behind
:mod:`jcor.discrepancy.energy`
"""

from __future__ import annotations

import jax.numpy as jnp

from jcor.core.axioms import Law  # noqa: TC001  # runtime contract
from jcor.core.typing import (  # noqa: TC001  # runtime; see jcor.core.typing
    Array,
    Float,
)
from jcor.discrepancy._mmd.components import _compute_energy_components
from jcor.discrepancy._mmd.energy.scalar import _resolve_distribution_inputs
from jcor.ground.config import (  # noqa: TC001  # runtime contract
    GroundDistanceSelection,
)
from jcor.ground.metrics import GroundDistance, cdist  # noqa: TC001  # runtime contract

__all__ = ["mixture_energy_distance"]


def _pooled_from_candidates(
    candidates: Array | list[Array],
) -> tuple[list[Array], Float[Array, " k"]]:
    """Normalize candidates to a list and return (list, m_sizes).

    Args:
        candidates: K candidate samples, stacked or as a list.

    Returns:
        Tuple of (candidate list, per-group sizes array).

    """
    if isinstance(candidates, jnp.ndarray):
        clist = [candidates[i] for i in range(candidates.shape[0])]
    else:
        clist = list(candidates)
    m_sizes = jnp.array([len(c) for c in clist], dtype=jnp.float32)
    return clist, m_sizes


def _energy_terms(
    x: Float[Array, "n d"],
    candidates: list[Array],
    exponent: float,
    metric: GroundDistance[Law],
) -> tuple[Float[Array, ""], Float[Array, " k"], Float[Array, "k k"]]:
    """Return target and candidate energy V-statistic components.

    Args:
        x: Target sample, shape (n, d).
        candidates: K candidate samples.
        exponent: Distance exponent α.
        metric: Declared ground-distance strategy.

    Returns:
        Tuple (term_xx scalar, target_cross shape (K,), candidate_gram shape (K, K)).

    """
    dxx = cdist(x, x, metric=metric) ** exponent
    term_xx = jnp.mean(dxx)
    candidate_gram, target_cross = _compute_energy_components(
        x, candidates, metric, exponent
    )
    return term_xx, target_cross, candidate_gram


def mixture_energy_distance(
    x: Float[Array, "n d"],
    candidates: Array | list[Array],
    weights: Float[Array, " k"],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
) -> Float[Array, ""]:
    """Energy distance ``E(X, Y_w)`` between target and weighted mixture.

    ``E = 2 wᵀ·XY - E[d(x,x')] - wᵀ·candidate_gram·w`` (nonnegative by negative type for
    ``metric``-induced ground distance with ``exponent ∈ (0,2)``).

    Args:
        x: Target sample, shape (n, d).
        candidates: K candidate samples (list or (K, m, d) array).
        weights: Mixture weights, shape (K,).
        exponent: Distance exponent α in (0, 2).
        metric: Distance metric.

    Returns:
        Scalar energy distance ``E(X, Y_w)``.

    Raises:
        ValueError: If exponent is not in (0, 2).
        IncompatibleEnergyGroundExponentError: If the selected ground law does
            not support the exponent regime.

    """
    validated_exponent, resolved_metric = _resolve_distribution_inputs(
        exponent,
        metric,
    )

    clist, _ = _pooled_from_candidates(candidates)
    term_xx, target_cross, candidate_gram = _energy_terms(
        x,
        clist,
        validated_exponent.value,
        resolved_metric,
    )
    return (
        2.0 * jnp.dot(weights, target_cross)
        - term_xx
        - jnp.dot(weights, jnp.dot(candidate_gram, weights))
    )

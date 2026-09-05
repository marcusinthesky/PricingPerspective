"""Dimension-reduction maps for high-dimensional two-sample tests.

High-dimensional energy/MMD tests lose power polynomially in ``d`` at fixed ``n``
(Ramdas et al. 2015). Reducing to ``d' ≈ 10-50`` before a two-sample test restores
power. :func:`fit_projection` is intended to be fit on a SELECTION-SPLIT-A ONLY (no
look-ahead into the test split).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import random


@dataclass(frozen=True)
class Projection:
    """A fitted linear dimension-reduction map (PCA or random projection).

    ``fit_projection`` returns this; apply with :meth:`transform`. The projection
    is fit on SELECTION-SPLIT-A ONLY to avoid look-ahead / double-dipping into the
    test split.
    """

    mean: jnp.ndarray
    components: jnp.ndarray  # (d, d') columns are directions
    kind: str

    def transform(self, z: jnp.ndarray) -> jnp.ndarray:
        """Project observations using the fitted centering and components."""
        return (z - self.mean) @ self.components


def fit_projection(
    data: jnp.ndarray,
    d_out: int = 32,
    kind: str = "pca",
    seed: int | None = None,
) -> Projection:
    """Fit a dimension-reduction map on ``data`` (SELECTION-SPLIT-A ONLY).

    High-dimensional energy/MMD tests lose power polynomially in ``d`` at fixed
    ``n`` (Ramdas et al. 2015). Reducing to ``d' ≈ 10-50`` before the two-sample
    test restores power. Fitting on split A only keeps the test split
    independent of the reduction (no look-ahead).

    Args:
        data: Selection-split-A data, shape (n_a, d).
        d_out: Target dimension d' (clamped to min(d, n_a) as needed).
        kind: ``"pca"`` (top principal directions) or ``"random"`` (Gaussian
            random projection; JL).
        seed: RNG seed for ``kind="random"``.

    Returns:
        A fitted :class:`Projection`.

    """
    mean = jnp.mean(data, axis=0)
    d = data.shape[1]
    d_eff = int(min(d_out, d, max(1, data.shape[0] - 1)))
    if kind == "pca":
        centered = data - mean
        # Right singular vectors are the principal directions.
        _, _, vt = jnp.linalg.svd(centered, full_matrices=False)
        components = vt[:d_eff].T  # (d, d')
    elif kind == "random":
        key = random.PRNGKey(seed if seed is not None else 0)
        proj = random.normal(key, (d, d_eff)) / jnp.sqrt(d_eff)
        components = proj
    else:
        message = f"kind must be 'pca' or 'random', got {kind!r}"
        raise ValueError(message)
    return Projection(mean=mean, components=components, kind=kind)

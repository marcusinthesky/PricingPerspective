"""Intensity-times-capacity reporting and finite-support article geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from statadic.core.typing import Array, ArrayLike, Float, PRNGKey  # noqa: TC002

from hotelling.precision import require_float64

PANEL_AXES = 3
REPORTING_REGIMES = ("common_fixed", "heterogeneous_fixed", "reporting_only", "joint")


def reporting_weights(
    q: ArrayLike, intensity: ArrayLike | float
) -> Float[Array, "*batch activity"]:
    """Normalize lambda*q across activities without hiding undefined firm laws."""
    require_float64()
    q = jnp.asarray(q, dtype=jnp.float64)
    intensity = jnp.asarray(intensity, dtype=jnp.float64)
    traced = any(isinstance(value, jax.core.Tracer) for value in (q, intensity))
    if (
        not q.ndim
        or not q.shape[-1]
        or (
            not traced
            and (
                not jnp.isfinite(q).all()
                or not jnp.isfinite(intensity).all()
                or jnp.any(q < 0)
                or jnp.any(intensity < 0)
            )
        )
    ):
        msg = "finite nonnegative capacity and reporting intensity required"
        raise ValueError(msg)
    try:
        weighted = q * intensity
    except ValueError as error:
        msg = "reporting intensity must broadcast to capacity"
        raise ValueError(msg) from error
    total = weighted.sum(axis=-1, keepdims=True)
    if not traced and (jnp.any(total <= 0) or not jnp.isfinite(total).all()):
        msg = "nonpositive or nonfinite reporting total: firm law is undefined"
        raise ValueError(msg)
    valid = jnp.all(
        jnp.isfinite(weighted) & (weighted >= 0), axis=-1, keepdims=True
    ) & (total > 0)
    return jnp.where(valid, weighted / total, jnp.nan)


def component_support(
    activities: int,
    dimension: int | None = None,
    kind: str = "axis",
    seed: int = 0,
) -> Float[Array, "atom dimension"]:
    """Create adjacent +/- component supports in an independent article space.

    Axis components use orthonormal unit vectors (and require dimension>=A).
    Nonorthogonal components are seeded unit directions and permit dimension<A.
    Neither constructor changes opportunity-space geometry.
    """
    dimension = activities if dimension is None else dimension
    if activities < 1 or dimension < 1:
        msg = "positive activity count and article dimension required"
        raise ValueError(msg)
    require_float64()
    key = jax.random.fold_in(jax.random.key(seed, impl="threefry2x32"), 301)
    if kind == "axis":
        if dimension < activities:
            msg = "axis article dimension must be at least the activity count"
            raise ValueError(msg)
        vectors = jnp.eye(dimension, activities)
        if dimension > activities:
            vectors, _ = jnp.linalg.qr(
                jax.random.normal(key, (dimension, activities), dtype=jnp.float64),
                mode="reduced",
            )
        vectors = vectors.T
    elif kind == "nonorthogonal":
        vectors = jax.random.normal(key, (activities, dimension), dtype=jnp.float64)
        vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)
    else:
        msg = f"unknown component support: {kind}"
        raise ValueError(msg)
    return jnp.stack((vectors, -vectors), axis=1).reshape(2 * activities, dimension)


def population_law(shares: ArrayLike) -> Float[Array, "*batch atom"]:
    """Put half each component probability on its two antipodal support points."""
    require_float64()
    shares = jnp.asarray(shares, dtype=jnp.float64)
    if (
        not shares.ndim
        or not shares.shape[-1]
        or (
            not isinstance(shares, jax.core.Tracer)
            and (
                jnp.any(shares < 0)
                or not jnp.isfinite(shares).all()
                or not jnp.allclose(shares.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)
            )
        )
    ):
        msg = "component shares must define probability laws"
        raise ValueError(msg)
    return jnp.repeat(shares / 2, 2, axis=-1)


def reporting_intensity(
    shape: tuple[int, int, int], regime: str, key: PRNGKey
) -> Float[Array, "date firm activity"]:
    """Draw reporting treatments; reporting-only capacity freezing is the caller's job.

    Heterogeneity has fixed log scale .35; changing reporting adds independent
    tick-specific log scale .25. Support geometry is a separate experiment.
    """
    if len(shape) != PANEL_AXES or min(shape) < 1 or regime not in REPORTING_REGIMES:
        msg = "positive (ticks,firms,activities) shape and known regime required"
        raise ValueError(msg)
    require_float64()
    if regime == "common_fixed":
        return jnp.ones(shape)
    baseline = 0.35 * jax.random.normal(
        jax.random.fold_in(key, 0), (1, shape[1], shape[2]), dtype=jnp.float64
    )
    if regime in {"reporting_only", "joint"}:
        baseline = baseline + 0.25 * jax.random.normal(
            jax.random.fold_in(key, 1), shape, dtype=jnp.float64
        )
    return jnp.broadcast_to(jnp.exp(baseline), shape).copy()

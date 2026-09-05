"""Alternative-hypothesis samplers for two-sample-test size/power studies.

All generators are pure functions of a :class:`jax.random.PRNGKey` and static
config args, producing **one replication** of shape ``(n, d)``.  Batch via
:func:`jax.vmap`.

Design note — Student-t construction
--------------------------------------
Draws ``z / sqrt(chi2/df)`` where ``chi2 = sum of df independent
N(0,1)² values``.  In JAX we use the identity ``chi2(df) = 2 · Gamma(df/2)``
so that ``jax.random.gamma`` replaces a variable-shape reduction.  This is a
standard textbook identity and produces the same marginal distribution.

Design note — mixture contamination count
-------------------------------------------
``n_contam = int(n * frac)`` is deterministic, not random, so callers can
reason about the exact contamination count as ``int(n * spec.param)`` without
inspecting the returned array.  The clean block is drawn first (rows
``0 … n_clean-1``) and the contaminated block second (rows
``n_clean … n-1``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jcor.core.typing import Array, Float  # noqa: TC001  # runtime; see docstring

# ---------------------------------------------------------------------------
# AltSpec — specification of an alternative distribution
# ---------------------------------------------------------------------------


class AltSpec(NamedTuple):
    """Specification of an alternative-hypothesis distribution.

    Attributes:
        kind: Distribution type.  One of:
            ``"normal"``, ``"student_t"``, ``"mean_shift"``,
            ``"scale"``, ``"mixture"``.
        param: Distribution-specific scalar parameter:
            * ``"normal"``       — ignored (set to ``None`` or any value).
            * ``"student_t"``    — degrees of freedom ``df > 2`` (float).
            * ``"mean_shift"``   — mean shift ``δ`` (float).
            * ``"scale"``        — scale factor ``s > 0`` (float).
            * ``"mixture"``      — contamination fraction ``frac ∈ (0, 1)``
              (float).
        shift: Shift applied to the contaminating component in ``"mixture"``
            distributions.  Default ``2.0``.

    """

    kind: str
    param: float | None
    shift: float = 2.0


# ---------------------------------------------------------------------------
# Individual draw functions (jit/vmap-safe, pure)
# ---------------------------------------------------------------------------


def _draw_normal(
    key: jax.Array,
    n: int,
    d: int,
) -> Float[Array, "n d"]:
    return jax.random.normal(key, shape=(n, d))


def _draw_student_t(
    key: jax.Array,
    n: int,
    d: int,
    df: float,
) -> Float[Array, "n d"]:
    """Multivariate t with independent marginals via chi-square scaling.

    ``chi2(df) = 2 · Gamma(df/2)`` identity used to avoid a variable-size
    reduction.  Each row is scaled by the *same* chi-square draw, giving
    independent (but not i.i.d. across dimensions) marginals.

    Returns:
        Array of shape ``(n, d)``.

    """
    key_z, key_g = jax.random.split(key)
    z = jax.random.normal(key_z, shape=(n, d))
    # chi2(df) per observation (n, 1)
    g = jax.random.gamma(key_g, a=df / 2.0, shape=(n, 1))
    chi2 = 2.0 * g
    return z / jnp.sqrt(chi2 / df)


def _draw_mean_shift(
    key: jax.Array,
    n: int,
    d: int,
    delta: float,
) -> Float[Array, "n d"]:
    """Draw a normal sample with mean shift ``δ`` in every dimension.

    Returns:
        Array of shape ``(n, d)``.

    """
    return jax.random.normal(key, shape=(n, d)) + delta


def _draw_scale(
    key: jax.Array,
    n: int,
    d: int,
    scale: float,
) -> Float[Array, "n d"]:
    """Draw a normal sample scaled by ``scale``.

    Returns:
        Array of shape ``(n, d)``.

    """
    return jax.random.normal(key, shape=(n, d)) * scale


def _draw_mixture(
    key: jax.Array,
    n: int,
    d: int,
    frac: float,
    shift: float,
) -> Float[Array, "n d"]:
    """Draw a normal sample with deterministic shifted-row contamination.

    Contamination count is ``int(n * frac)`` (deterministic).  Clean
    observations are drawn first, contaminated last.

    Returns:
        Array of shape ``(n, d)``.

    """
    key_clean, key_contam = jax.random.split(key)
    n_contam = int(n * frac)
    n_clean = n - n_contam
    clean = jax.random.normal(key_clean, shape=(n_clean, d))
    contam = jax.random.normal(key_contam, shape=(n_contam, d)) + shift
    return jnp.concatenate([clean, contam], axis=0)


# ---------------------------------------------------------------------------
# Unified draw function
# ---------------------------------------------------------------------------


def draw(
    key: jax.Array,
    spec: AltSpec,
    n: int,
    d: int,
) -> Float[Array, "n d"]:
    """Draw one ``(n, d)`` sample from the distribution specified by ``spec``.

    Args:
        key: JAX PRNG key.
        spec: :class:`AltSpec` describing the target distribution.
        n: Number of observations.
        d: Number of dimensions.

    Returns:
        Array of shape ``(n, d)``.

    Raises:
        ValueError: If ``spec.kind`` is not recognised.

    """
    kind = spec.kind
    param = spec.param
    shift = spec.shift

    if kind == "normal":
        return _draw_normal(key, n, d)
    if kind == "student_t":
        if param is None:
            message = "AltSpec.param must be df for student_t"
            raise ValueError(message)
        return _draw_student_t(key, n, d, df=float(param))
    if kind == "mean_shift":
        if param is None:
            message = "AltSpec.param must be delta for mean_shift"
            raise ValueError(message)
        return _draw_mean_shift(key, n, d, delta=float(param))
    if kind == "scale":
        if param is None:
            message = "AltSpec.param must be scale for scale"
            raise ValueError(message)
        return _draw_scale(key, n, d, scale=float(param))
    if kind == "mixture":
        if param is None:
            message = "AltSpec.param must be frac for mixture"
            raise ValueError(message)
        return _draw_mixture(key, n, d, frac=float(param), shift=shift)
    message = (
        f"Unknown AltSpec kind '{kind}'. "
        "Choose from: normal, student_t, mean_shift, scale, mixture."
    )
    raise ValueError(message)

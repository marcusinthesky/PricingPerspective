"""Scalar MMD estimates and the firm-level cosine-mean comparator."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal  # runtime contract

import jax.numpy as jnp
from jax import jit

from jcor.core.typing import Array, Float  # noqa: TC001  # runtime contract
from jcor.discrepancy._mmd.contracts import (
    _require_u_sample_size,
    _require_v_distance,
)
from jcor.ground.similarities import (
    LINEAR_KERNEL,
    SimilarityStrategy,
    gram,
)

if TYPE_CHECKING:
    from jcor.ground.similarities import PositiveSemidefiniteKernelLaw

__all__ = ["cosine_mean_mmd", "mmd", "mmd_squared"]


def _shared_reference(
    anchor: Float[Array, "n d"],
    reference: Float[Array, " d"] | None,
) -> Float[Array, " d"]:
    """Resolve the single reference shared by every Gram block.

    Distance-induced reference terms cancel from MMD only when ``Kxx``, ``Kyy``
    and ``Kxy`` use the same point.  A sample point is the safe fallback because
    the origin is not in the angular ground space.
    """
    if reference is not None:
        return reference
    return anchor[0]


@partial(
    jit,
    static_argnames=("kernel", "unbiased"),
)
def _mmd_squared_core(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: Float[Array, " d"] | None,
    *,
    unbiased: bool,
) -> Float[Array, ""]:
    """Evaluate the V- or U-statistic MMD² estimate in one compiled core."""
    shared_reference = _shared_reference(x, reference)
    n, m = x.shape[0], y.shape[0]
    kxx = gram(x, x, kernel, reference=shared_reference)
    kyy = gram(y, y, kernel, reference=shared_reference)
    kxy = gram(x, y, kernel, reference=shared_reference)
    if unbiased:
        xx = (jnp.sum(kxx) - jnp.trace(kxx)) / (n * (n - 1))
        yy = (jnp.sum(kyy) - jnp.trace(kyy)) / (m * (m - 1))
    else:
        xx = jnp.sum(kxx) / (n * n)
        yy = jnp.sum(kyy) / (m * m)
    # A PSD kernel is symmetric. Average both floating-point orientations so
    # every declared strategy realizes that law without name-based dispatch.
    kyx = gram(y, x, kernel, reference=shared_reference)
    cross = 0.5 * (jnp.sum(kxy) + jnp.sum(kyx)) / (n * m)
    return xx + yy - 2.0 * cross


def mmd_squared(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: bool = False,
) -> Float[Array, ""]:
    """Compute a raw squared maximum-mean-discrepancy estimate.

    ``unbiased=False`` is the V-statistic and is the only convention represented
    in ``DECLARED_AXIOMS``/``DECLARED_BRANDS``.  ``unbiased=True`` returns the
    unbranded U-statistic estimate unchanged: it may be negative for distinct
    finite samples and is never clamped.

    Args:
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``.
        kernel: Declared PSD strategy. Its parameters and mathematical law are
            already closed by construction.
        reference: Shared distance-induced reference point.
        unbiased: Drop within-sample diagonals to form the U-statistic.

    Returns:
        Scalar MMD² estimate.  A U-statistic result may be negative.

    Raises:
        InsufficientUStatisticSampleError: If either U-statistic block has fewer
            than two observations.
        TypeError: If a bare string or callable bypasses declaration.

    Notes:
        Normalizing kernels and grounds require the evidence named by their
        strategy. This transformable array door leaves an out-of-domain value
        visible as ``NaN``; checked carrier doors reject eagerly.

    """
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    if unbiased:
        _require_u_sample_size(x.shape[0])
        _require_u_sample_size(y.shape[0])
    return _mmd_squared_core(
        x,
        y,
        kernel,
        reference,
        unbiased=unbiased,
    )


def mmd(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: Literal[False] = False,
) -> Float[Array, ""]:
    """Compute the V-statistic MMD distance between two empirical measures.

    The U-statistic estimates MMD² and can be negative, so ``unbiased=True`` is
    rejected here both statically and at runtime.  Use :func:`mmd_squared` for
    that unbranded estimate.

    Args:
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``.
        kernel: Declared PSD strategy with its parameters and law closed.
        reference: Shared distance-induced reference point.
        unbiased: Must remain ``False``; retained to reject old calls clearly.

    Returns:
        Nonnegative square root of the V-statistic MMD² value.

    Raises:
        UnbiasedMmdDistanceError: If ``unbiased=True`` is supplied at runtime.

    """
    _require_v_distance(unbiased=unbiased)
    squared = mmd_squared(
        x,
        y,
        kernel=kernel,
        reference=reference,
        unbiased=False,
    )
    return jnp.sqrt(jnp.maximum(squared, 0.0))


@jit
def cosine_mean_mmd(
    psi_x: Float[Array, " d"],
    psi_y: Float[Array, " d"],
) -> Float[Array, ""]:
    """Compare two already-normalized prototype directions by chord distance.

    This is an MMD between point masses on the space of empirical measures, not
    a ``kernel=`` choice for article-cloud :func:`mmd`. Pooling and
    normalization are pipeline transformations; this core does neither.

    Args:
        psi_x: First pooled mean embedding, shape ``(d,)``.
        psi_y: Second pooled mean embedding, shape ``(d,)``.

    Returns:
        Chord distance between the supplied unit directions.

    """
    return jnp.linalg.norm(psi_x - psi_y)

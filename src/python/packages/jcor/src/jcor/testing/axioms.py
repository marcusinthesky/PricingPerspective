"""Measure — and assert — which metric axioms a dissimilarity actually satisfies.

A declared axiom that nobody checks is a comment. This module turns
:class:`jcor.core.axioms.Axioms` into an executable contract:
:func:`measure_axioms` reports what a callable *does*, :func:`assert_axioms`
gates on what it *claims*, and the gap between the two is the finding.

The uniform adapter
-------------------
Every check runs over a finite set of **objects** and the matrix
``D[i, j] = fn(objects[i], objects[j])``. What an "object" is depends on the
stage, and that is the point of the taxonomy: for a ``ground`` metric it is a
point (shape ``(d,)``), for a ``discrepancy`` it is a whole sample (shape
``(n, d)``). Callers wrap the stage's signature into
``fn(a, b) -> scalar`` — fixing the PRNG key, the exponent, the ground metric —
and the battery is then identical for both.

Tolerances
----------
Every comparison carries ``atol`` **and** ``rtol``: the triangle inequality on
distances of order 10³ cannot be checked to 1e-9 in float32, and a battery that
pretends otherwise reports numerical noise as a violated axiom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import (  # noqa: TC002  # runtime import is load-bearing; see jcor.core.typing
    Array,
    Float,
)

from jcor.core.axioms import Axioms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "assert_axioms",
    "assert_negative_type",
    "dissimilarity_matrix",
    "measure_axioms",
    "negative_type_eigmin",
]

_MIN_OBJECTS = 3


def dissimilarity_matrix(
    fn: Callable[[Array, Array], Array],
    objects: Sequence[Array],
) -> Float[Array, "k k"]:  # noqa: F722  # jaxtyping shape
    """Build the full ``k × k`` matrix of pairwise dissimilarities.

    Args:
        fn: Dissimilarity between two objects, returning a scalar.
        objects: The objects to compare. A point for a ground metric, a whole
            sample for a discrepancy.

    Returns:
        Matrix ``D`` with ``D[i, j] = fn(objects[i], objects[j])``.

    """
    return jnp.asarray(
        [[jnp.asarray(fn(a, b)).reshape(()) for b in objects] for a in objects]
    )


def measure_axioms(
    fn: Callable[[Array, Array], Array],
    objects: Sequence[Array],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> Axioms:
    """Report which axioms ``fn`` satisfies on ``objects`` — the reporting instrument.

    This is deliberately *not* an assertion: the t46.1 battery exists to compare
    measured behaviour against declared behaviour, so the measurement has to be
    obtainable without failing.

    Args:
        fn: Dissimilarity between two objects, returning a scalar.
        objects: Distinct objects to compare; at least three, so the triangle
            inequality has a middle point to fail on.
        atol: Absolute tolerance.
        rtol: Relative tolerance, scaled by the magnitude of the compared terms.

    Returns:
        The flags that hold. ``ZERO_DIAGONAL`` and ``SEPARATION`` are measured
        independently; the compatibility member ``IDENTITY`` is present only
        when both bits hold. Separation needs ``objects`` to be pairwise
        distinct — the caller guarantees that.

    Raises:
        ValueError: If fewer than three objects are supplied.

    """
    if len(objects) < _MIN_OBJECTS:
        message = f"need at least {_MIN_OBJECTS} objects, got {len(objects)}"
        raise ValueError(message)

    d = dissimilarity_matrix(fn, objects)
    scale = float(jnp.max(jnp.abs(d)))
    tol = atol + rtol * scale
    off_diagonal = ~jnp.eye(d.shape[0], dtype=bool)

    held = Axioms(0)
    if bool(jnp.all(d >= -tol)):
        held |= Axioms.NONNEGATIVE
    diagonal_zero = bool(jnp.all(jnp.abs(jnp.diagonal(d)) <= tol))
    distinct_nonzero = bool(jnp.all(jnp.abs(d[off_diagonal]) > tol))
    if diagonal_zero:
        held |= Axioms.ZERO_DIAGONAL
    if distinct_nonzero:
        held |= Axioms.SEPARATION
    if bool(jnp.allclose(d, d.T, atol=atol, rtol=rtol)):
        held |= Axioms.SYMMETRY
    # d[i, k] <= d[i, j] + d[j, k] for every middle point j.
    detours = d[:, :, None] + d[None, :, :]
    if bool(jnp.all(d[:, None, :] <= detours + tol)):
        held |= Axioms.TRIANGLE
    return held


def assert_axioms(
    fn: Callable[[Array, Array], Array],
    holds: Axioms,
    objects: Sequence[Array],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> None:
    """Assert that ``fn`` satisfies at least the axioms it declares.

    Args:
        fn: Dissimilarity between two objects, returning a scalar.
        holds: The declared axioms. Pass **flags**, never an alias name — see
            :mod:`jcor.core.axioms`.
        objects: Distinct objects to compare; at least three.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Raises:
        AssertionError: If any declared axiom does not hold. The message names
            the missing flags and the flags that were measured, so a wrong
            *declaration* is as visible as a wrong implementation.

    """
    measured = measure_axioms(fn, objects, atol=atol, rtol=rtol)
    missing = holds & ~measured
    if missing:
        message = (
            f"{getattr(fn, '__name__', fn)!s} declares {holds!s} but "
            f"{missing!s} does not hold; measured {measured!s}"
        )
        raise AssertionError(message)


def negative_type_eigmin(
    fn: Callable[[Array, Array], Array],
    objects: Sequence[Array],
) -> float:
    """Smallest eigenvalue of the double-centred Gram matrix ``-½ J D J``.

    ``d`` is of **negative type** iff ``Σ_i Σ_j c_i c_j d(x_i, x_j) <= 0``
    whenever ``Σ_i c_i == 0``, which is exactly ``-½ J D J ⪰ 0`` for
    ``J = I - 11ᵀ/k``. That Gram matrix is what classical MDS / PCoA embeds and
    what the energy statistics need to be non-negative, so this is the check
    behind the :class:`jcor.core.axioms.NegativeType` brand.

    Note the convention: ``D`` holds ``d`` itself, **not** ``d²``. Squared
    Euclidean distance is of negative type; Euclidean distance is too, but they
    are different matrices and the caller must pass the one it is branding.

    Args:
        fn: Dissimilarity between two objects, returning a scalar.
        objects: At least three objects.

    Returns:
        The minimum eigenvalue. Non-negative (up to tolerance) iff ``fn`` is of
        negative type on this object set.

    Raises:
        ValueError: If fewer than three objects are supplied.

    """
    if len(objects) < _MIN_OBJECTS:
        message = f"need at least {_MIN_OBJECTS} objects, got {len(objects)}"
        raise ValueError(message)
    d = dissimilarity_matrix(fn, objects)
    k = d.shape[0]
    centring = jnp.eye(k) - jnp.ones((k, k)) / k
    gram = -0.5 * centring @ d @ centring
    # Symmetrize away the O(eps) asymmetry that the two matmuls introduce.
    return float(jnp.min(jnp.linalg.eigvalsh(0.5 * (gram + gram.T))))


def assert_negative_type(
    fn: Callable[[Array, Array], Array],
    objects: Sequence[Array],
    *,
    atol: float = 1e-8,
) -> None:
    """Assert that ``fn`` is of negative type on ``objects``.

    Args:
        fn: Dissimilarity between two objects, returning a scalar.
        objects: At least three objects.
        atol: How negative the smallest eigenvalue may be before the double-
            centred Gram matrix counts as indefinite rather than merely noisy.

    Raises:
        AssertionError: If the double-centred Gram matrix is not PSD.

    """
    eigmin = negative_type_eigmin(fn, objects)
    if eigmin < -atol:
        message = (
            f"{getattr(fn, '__name__', fn)!s} is not of negative type: "
            f"min eigenvalue of -0.5·J·D·J is {eigmin:.3e} < {-atol:.3e}"
        )
        raise AssertionError(message)

"""S4 — classical (Torgerson/Gower) scaling: a distance matrix to coordinates.

Position
--------
rank 4 · geometry. Consumes a distance matrix (waist 1); produces a Euclidean
embedding. Not a distance, hence a stage above ``discrepancy``.

Precondition — the reason this module exists
--------------------------------------------
PCoA is only defined when the double-centred Gram matrix ``-½ J D² J`` is
positive semi-definite, i.e. when ``D`` is of **negative type** (Deza & Deza
§15.2). This arrived in the repository as ``paper1/disco.py::_pcoa``, whose
final line was

.. code-block:: python

    coords = eigvecs[:, keep] * np.sqrt(np.clip(eigvals[keep], 0.0, None))

— a ``clip`` that would have silently swallowed a genuinely non-embeddable
input. The typed producer now carries the exact Hilbertian law and energy-root
origin instead of relying on a detached compatibility brand. The dependency is
stated twice:

1. statically — :func:`pcoa` takes a three-axis ``DMat`` whose law explicitly
   declares squared-distance negative type, so an ordinary negative-type law
   is rejected;
2. dynamically — the transformable kernel returns a ``valid`` diagnostic and
   both eager doors **raise** :class:`NonEuclideanEmbeddingError` when the
   smallest eigenvalue falls below ``-tolerance * λ_max`` instead of clipping.
   The residual clip is gone: ``active`` already selects
   ``eigvals > tolerance * λ_max`` with ``λ_max >= 1``, so every retained
   eigenvalue is strictly positive and the clip was provably a no-op.

Three callables at the array/artifact boundary
----------------------------------------------
``DMat.values`` is ``Float[Array, "*batch n n"]`` — a JAX array — and, measured
under beartype, ``Float[Array, …]`` rejects an ``np.ndarray`` while
``Float[np.ndarray, …]`` rejects a ``jax.Array``. So the branded form and the
artifact-boundary form cannot be unified. :func:`pcoa_kernel` is the single
fixed-width numerical implementation: it is ``jit``/``vmap`` compatible and
returns activity and validity as array leaves. :func:`pcoa` is the branded eager
presentation door; :func:`pcoa_matrix` is the eager artifact door. Both return
JAX arrays, but only :func:`pcoa_matrix` opens x64, so the process default
cannot truncate a stored float64 matrix read off an artifact; :func:`pcoa` runs
at whatever precision its caller owns, because a branded ``DMat`` was already
built under that caller's policy. Both eager doors delegate to the kernel,
concretise its validity diagnostic, and only then select active axes; tests pin
the three surfaces to the same spectrum and coordinates.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from typing import Final, NamedTuple

import jax.numpy as jnp

# Runtime imports, not TYPE_CHECKING-guarded: an unresolvable name makes beartype
# silently SKIP the annotation rather than fail it, so a guarded `DMat` or
# `NegativeType` would erase the `DMat` check too. See `jcor.core.typing`.
from jcor.core.axioms import Axioms, requires
from jcor.core.domains import ConstructionOrigin, ObjectDomain
from jcor.core.matrices import (  # noqa: TC001  # runtime type contract
    DMat,
    SquaredDistanceNegativeTypeLaw,
)
from jcor.core.typing import Array, ArrayLike, Bool, Float  # noqa: TC001

__all__ = [
    "EIGENVALUE_TOLERANCE",
    "NonEuclideanEmbeddingError",
    "PCoAKernelResult",
    "pcoa",
    "pcoa_kernel",
    "pcoa_matrix",
]

#: Relative slack on the non-negativity of the Gram spectrum. A *genuinely*
#: non-negative-type input produces a negative eigenvalue comparable in
#: magnitude to the leading positive one; anything smaller than this multiple of
#: ``max(|λ|, 1)`` is finite-sample estimation noise in an *empirical* distance
#: whose population counterpart is exactly of negative type. The same constant
#: sets the rank cut, so the axes that are kept are strictly positive.
EIGENVALUE_TOLERANCE: Final = 1e-8


class NonEuclideanEmbeddingError(ValueError):
    """Raised when the PCoA Gram spectrum is materially negative.

    A ``ValueError`` subclass so the pre-t46 ``raise ValueError`` contract at
    Paper 1's DISCO call site still holds for any caller catching the base
    class.
    """

    def __init__(self, min_eigenvalue: float, max_eigenvalue: float) -> None:
        """Record the offending spectrum bounds in the message.

        Args:
            min_eigenvalue: Smallest Gram eigenvalue observed.
            max_eigenvalue: Scale the tolerance was taken relative to —
                ``max(|λ_max|, 1)``.

        """
        super().__init__(
            "PCoA produced a negative eigenvalue beyond numerical tolerance "
            f"({min_eigenvalue:.3e}) relative to the leading eigenvalue "
            f"({max_eigenvalue:.3e}); expected approximately non-negative "
            "(negative-type distance)."
        )


class PCoAKernelResult(NamedTuple):
    """Fixed-structure result of :func:`pcoa_kernel`.

    Inactive coordinate columns are zero. Consumers inside a transform retain
    the full fixed-width carrier and use ``active``; eager presentation doors
    may select those columns after leaving the trace.
    """

    coordinates: Float[Array, "*batch n n"]
    eigenvalues: Float[Array, "*batch n"]
    active: Bool[Array, "*batch n"]
    minimum_eigenvalue: Float[Array, "*batch"]
    spectral_scale: Float[Array, "*batch"]
    valid: Bool[Array, "*batch"]


def pcoa_kernel(
    distances: Float[Array, "*batch n n"],
    tolerance: float = EIGENVALUE_TOLERANCE,
) -> PCoAKernelResult:
    """Return a fixed-width classical-scaling embedding on the JAX graph.

    The kernel never raises or changes output shape. ``valid`` records whether
    the Gram spectrum satisfies the negative-type precondition within the
    requested tolerance; eager doors turn a false diagnostic into
    :class:`NonEuclideanEmbeddingError`.
    """
    values = jnp.asarray(distances)
    n = values.shape[-1]
    centering = (
        jnp.eye(n, dtype=values.dtype) - jnp.ones((n, n), dtype=values.dtype) / n
    )
    gram = -0.5 * centering @ (values**2) @ centering
    gram = (gram + jnp.swapaxes(gram, -1, -2)) / 2.0
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = eigenvalues[..., ::-1]
    eigenvectors = eigenvectors[..., :, ::-1]
    spectral_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues), axis=-1), 1.0)
    minimum = jnp.min(eigenvalues, axis=-1)
    active = eigenvalues > tolerance * spectral_scale[..., None]
    coordinates = (
        eigenvectors
        * jnp.sqrt(
            jnp.where(active, eigenvalues, jnp.zeros((), dtype=eigenvalues.dtype))
        )[..., None, :]
    )
    return PCoAKernelResult(
        coordinates=coordinates,
        eigenvalues=eigenvalues,
        active=active,
        minimum_eigenvalue=minimum,
        spectral_scale=spectral_scale,
        valid=minimum >= -tolerance * spectral_scale,
    )


def _raise_if_non_euclidean(result: PCoAKernelResult) -> None:
    """Raise at an eager boundary when a kernel diagnostic is false."""
    if not bool(result.valid):
        raise NonEuclideanEmbeddingError(
            float(result.minimum_eigenvalue),
            float(result.spectral_scale),
        )


@requires(Axioms.NONNEGATIVE | Axioms.IDENTITY | Axioms.SYMMETRY | Axioms.TRIANGLE)
def pcoa[DomainA: ObjectDomain, OriginA: ConstructionOrigin](
    distances: DMat[DomainA, SquaredDistanceNegativeTypeLaw, OriginA],
    *,
    tolerance: float = EIGENVALUE_TOLERANCE,
) -> tuple[Float[Array, "n k"], Float[Array, " n"]]:
    """Embed a negative-type distance matrix by classical scaling.

    The branded, in-package form. Double-centres the *squared* distances (the
    standard Gower construction), eigendecomposes, and keeps the
    strictly-positive axes.

    Args:
        distances: Square distance matrix branded ``NegativeType`` — the
            precondition classical scaling needs, carried by the type rather
            than by a comment.
        tolerance: Relative spectral slack; see :data:`EIGENVALUE_TOLERANCE`.

    Returns:
        ``(coords, eigenvalues)``. ``coords`` has one column per retained
        positive axis, scaled by ``sqrt(λ)``; ``eigenvalues`` is the full
        descending spectrum, including any near-zero negative tail, so callers
        can report the minimum as a diagnostic.

    Raises:
        NonEuclideanEmbeddingError: If the smallest eigenvalue is below
            ``-tolerance * max(|λ_max|, 1)``.

    """
    result = pcoa_kernel(distances.values, tolerance)
    _raise_if_non_euclidean(result)
    return result.coordinates[:, result.active], result.eigenvalues


def pcoa_matrix(
    distances: ArrayLike,
    *,
    tolerance: float = EIGENVALUE_TOLERANCE,
) -> tuple[Float[Array, "n k"], Float[Array, " n"]]:
    """Embed a stored float64 distance matrix by classical scaling.

    The eager artifact-boundary form: it requests ``float64`` for a matrix read
    off disk and returns JAX arrays, keeping every eager door on the same
    surface. The request is honoured only under a caller-owned
    ``jax_enable_x64``; this module opens no scope of its own. Use this when the
    matrix was read off an artifact rather than produced in-package: it cannot
    carry the ``NegativeType`` brand (see the module docstring), so the
    precondition travels as the raise below rather than as a type.

    Args:
        distances: Square distance matrix; converted to ``float64``.
        tolerance: Relative spectral slack; see :data:`EIGENVALUE_TOLERANCE`.

    Returns:
        ``(coords, eigenvalues)``, as :func:`pcoa`.

    Raises:
        NonEuclideanEmbeddingError: If the smallest eigenvalue is below
            ``-tolerance * max(|λ_max|, 1)``.

    """
    values = jnp.asarray(distances, dtype=jnp.float64)
    result = pcoa_kernel(values, tolerance)
    _raise_if_non_euclidean(result)
    return result.coordinates[:, result.active], result.eigenvalues

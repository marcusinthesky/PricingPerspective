"""Shared eager validation for covariance estimators and shrinkage.

Every routine here is an **eager API door**: each closes a device sync with
``bool()``/``float()`` to reach a Python ``raise``, so none is jittable and none
is intended to be. t65.2 removed the ``Float = NDArray[np.float64]`` alias and
the ``np.asarray(x, dtype=np.float64)`` coercions that went with it — those were
silent upcasts forcing float64 regardless of caller configuration. Validation
now evaluates at the caller's dtype; a stage that needs float64 opens the scope
itself (``pipeline.precision.own_float64``).
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import jax.numpy as jnp

from jcor.core.typing import Array, ArrayLike, Float, as_index  # noqa: TC001

MATRIX_NDIM = 2
MIN_COVARIANCE_OBSERVATIONS = 2
MIN_CV_FOLD_OBSERVATIONS = 2
PSD_RELATIVE_TOLERANCE = 1e-10

#: Multiples of ``n * eps * scale`` below which a negative eigenvalue is read as
#: ``eigh`` roundoff rather than indefiniteness. Measured 2026-08-03, not chosen:
#: over 200k draws of ``factor.T @ factor`` from the property suite's own
#: generator range (entries in [-4, 4], up to 6x5 factors) the largest realized
#: ``-min(eigvalsh) / (eps * scale)`` was 2.85, so 8 leaves better than eightfold
#: headroom while staying four decades under ``PSD_RELATIVE_TOLERANCE``.
PSD_NOISE_FLOOR_FACTOR = 8


def validate_observation_panel(
    returns: ArrayLike,
    *,
    ddof: int,
    min_observations: int = 1,
) -> Float[Array, "t n"]:
    """Validate a finite rectangular panel and its covariance divisor."""
    panel = jnp.asarray(returns)
    if panel.ndim != MATRIX_NDIM or panel.shape[1] < 1:
        message = "returns must have shape (n_observations, n_variables)."
        raise ValueError(message)
    if as_index(ddof) is None or ddof < 0:
        message = "ddof must be a nonnegative integer."
        raise ValueError(message)
    if panel.shape[0] < min_observations or panel.shape[0] <= ddof:
        message = (
            "returns has too few observations for the requested covariance divisor."
        )
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(panel))):
        message = "returns must contain only finite values."
        raise ValueError(message)
    return panel


def validate_symmetric_target(
    target: ArrayLike,
    n_variables: int,
) -> Float[Array, "n n"]:
    """Validate a finite real symmetric target without claiming covariance PSD."""
    raw = jnp.asarray(target)
    if not jnp.issubdtype(raw.dtype, jnp.number) or jnp.issubdtype(
        raw.dtype,
        jnp.bool_,
    ):
        message = "target must contain real numeric values."
        raise ValueError(message)
    if jnp.iscomplexobj(raw):
        message = "target must contain real numeric values."
        raise ValueError(message)
    matrix = raw
    if matrix.shape != (n_variables, n_variables):
        message = (
            "target must have shape "
            f"({n_variables}, {n_variables}); observed {matrix.shape}."
        )
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        message = "target must contain only finite values."
        raise ValueError(message)
    if not bool(jnp.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)):
        message = "target must be symmetric."
        raise ValueError(message)
    symmetric = 0.5 * matrix + 0.5 * matrix.T
    if not bool(jnp.all(jnp.isfinite(symmetric))):
        message = "target symmetrization must remain finite."
        raise ValueError(message)
    if bool(jnp.any(jnp.diag(symmetric) < 0.0)):
        message = "target diagonal entries must be nonnegative variances."
        raise ValueError(message)
    return symmetric


def validate_covariance_target(
    target: ArrayLike,
    n_variables: int,
) -> Float[Array, "n n"]:
    """Validate a finite symmetric positive-semidefinite covariance target.

    A symmetric matrix with nonnegative diagonal is not necessarily a covariance
    matrix. Materially negative eigenvalues therefore fail here instead of
    flowing through ``shrink_to_dist(..., lam=0)`` under a false covariance law.

    Between rejection and acceptance sits a band whose width is set by ``eigh``'s
    own error, and the boundary between "repair" and "pass through" is placed at
    that width rather than at zero. Testing ``minimum < 0.0`` would make this
    function discontinuous on the boundary of the cone: a rank-deficient PSD
    target — ``factor.T @ factor`` with fewer factor rows than columns — has
    exact zero eigenvalues that ``eigh`` returns as small negatives, which fired
    the repair on 24% of draws from the property suite's generator range
    (measured 2026-08-03). Each such firing perturbs an already-admissible target
    by O(eps) and so breaks the ``lam=0`` endpoint identity, and it does not buy
    positive semidefiniteness in exchange: rebuilding from clipped eigenpairs
    leaves a negative eigenvalue of the same order 75% of the time, reducing it
    by a median factor of only 2.5.

    Inside the noise floor the target is therefore returned unchanged, being PSD
    to the precision any downstream consumer can measure. Only indefiniteness
    that clears the floor — real, but under the rejection threshold — has its
    spectrum clipped, and that reduction remains approximate for the same reason.

    **Non-convergence is detected, not caught.** This was the one site in the
    package holding ``try/except np.linalg.LinAlgError``. ``jnp.linalg.eigh``
    never raises: LAPACK failure and nonfinite input both surface as ``NaN``
    eigenvalues, so the same contract is expressed by testing for them and
    raising the identical ``ValueError``. The three-way branch below stays eager
    and host-side because it returns *different values* — raise, clip, or pass
    through — rather than merely a verdict.
    """
    symmetric = validate_symmetric_target(target, n_variables)
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric)
    if not bool(jnp.all(jnp.isfinite(eigenvalues))):
        message = "covariance target eigendecomposition did not converge."
        raise ValueError(message)
    scale = max(float(jnp.max(jnp.abs(eigenvalues), initial=0.0)), 1.0)
    minimum = float(jnp.min(eigenvalues, initial=0.0))
    if minimum < -PSD_RELATIVE_TOLERANCE * scale:
        message = "covariance target must be positive semidefinite."
        raise ValueError(message)
    unit_roundoff = float(jnp.finfo(eigenvalues.dtype).eps)
    noise_floor = PSD_NOISE_FLOOR_FACTOR * n_variables * unit_roundoff * scale
    if minimum < -noise_floor:
        clipped = (
            eigenvectors @ jnp.diag(jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        )
        projected = 0.5 * clipped + 0.5 * clipped.T
        if not bool(jnp.all(jnp.isfinite(projected))):
            message = "covariance target PSD projection must remain finite."
            raise ValueError(message)
        return projected
    return symmetric

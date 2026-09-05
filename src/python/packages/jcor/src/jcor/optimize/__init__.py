"""Cross-cutting numerical substrate — importable by every stage.

Position
--------
rank 0 (with :mod:`jcor.core`)

Consumes
--------
raw matrices and vectors; nothing stage-shaped

Produces
--------
simplex-constrained optima and PSD-repaired matrices

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Because this package is rank 0 it may
import ``jcor.core`` and nothing else from ``jcor``.

Modules
-------
:mod:`jcor.optimize.simplex`
    Two deliberately different projected-gradient solvers on the probability
    simplex: the pure quadratic ``min/max wᵀAw`` pair with a fixed iteration
    count, and the affine ``min ½wᵀQw + cᵀw`` solver with early stopping.
:mod:`jcor.optimize.psd`
    Traceable PSD repairs: Qi-Sun semismooth-Newton nearest correlation and
    covariance, ridge, eigenvalue clipping, diagonal shrink, and diagnostics.
    All JAX; there is no NumPy backend.
"""

from __future__ import annotations

from jcor.optimize.backends import least_squares, minimise_bfgs
from jcor.optimize.psd import (
    DEFAULT_RIDGE_EPS,
    NearestCorrelationResult,
    RepairDiagnostics,
    ShrinkResult,
    eig_clip,
    gerber_shrink_psd,
    nearest_correlation,
    nearest_covariance,
    repair_diagnostics,
    ridge_psd,
)
from jcor.optimize.simplex import (
    pgd_maximize_quadratic_form,
    pgd_minimize_quadratic_form,
    pgd_simplex_affine,
)

__all__ = [
    "DEFAULT_RIDGE_EPS",
    "NearestCorrelationResult",
    "RepairDiagnostics",
    "ShrinkResult",
    "eig_clip",
    "gerber_shrink_psd",
    "least_squares",
    "minimise_bfgs",
    "nearest_correlation",
    "nearest_covariance",
    "pgd_maximize_quadratic_form",
    "pgd_minimize_quadratic_form",
    "pgd_simplex_affine",
    "repair_diagnostics",
    "ridge_psd",
]

"""Compatibility facade for the PSD repairs now owned by :mod:`jcor`.

All public objects are direct re-exports, so historical
``simulation.psd_repair`` imports retain object identity. New production code
should import from :mod:`jcor.optimize.psd`.

The surface changed in t65.4: the NumPy Higham/Dykstra backend was deleted and
``nearest_correlation`` / ``nearest_covariance`` (Qi–Sun semismooth Newton)
replace ``higham_correlation_repair``, ``higham_psd_repair`` and their
``higham_nearest_*`` aliases. ``ridge_repair`` was a byte-identical NumPy port
of :func:`jcor.optimize.psd.ridge_psd` and was deleted rather than re-exported.
Both replacements return a :class:`~jcor.optimize.psd.NearestCorrelationResult`
rather than a bare array — take ``.matrix``.
"""

from jcor.optimize.psd import (
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

__all__ = [
    "NearestCorrelationResult",
    "RepairDiagnostics",
    "ShrinkResult",
    "eig_clip",
    "gerber_shrink_psd",
    "nearest_correlation",
    "nearest_covariance",
    "repair_diagnostics",
    "ridge_psd",
]

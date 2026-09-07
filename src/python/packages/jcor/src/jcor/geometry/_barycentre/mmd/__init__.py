"""Generic kernel-MMD barycentre solvers."""

from __future__ import annotations

from jcor.geometry._barycentre.mmd.solve import KernelBarycentreDiagnostics
from jcor.geometry._barycentre.mmd.weights import (
    kernel_barycentre_weights,
)

__all__ = [
    "KernelBarycentreDiagnostics",
    "kernel_barycentre_weights",
]

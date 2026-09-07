"""Compatibility facade for :mod:`jcor.inference.gmm`."""

from jcor.inference.gmm import (
    HansenJResult,
    LinearHansenJResult,
    WaldResult,
    gmm_wald,
    hansen_j,
    linear_hansen_j,
)

__all__ = [
    "HansenJResult",
    "LinearHansenJResult",
    "WaldResult",
    "gmm_wald",
    "hansen_j",
    "linear_hansen_j",
]

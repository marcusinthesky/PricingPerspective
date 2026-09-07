"""Estimator and eager prototype errors shared by typed MMD producers."""

from __future__ import annotations

_MINIMUM_U_SAMPLE_SIZE = 2

__all__ = [
    "InsufficientUStatisticSampleError",
    "NonfiniteMeanEmbeddingError",
    "UnbiasedMmdDistanceError",
    "ZeroNormMeanEmbeddingError",
]


class ZeroNormMeanEmbeddingError(ValueError):
    """Raised when a pooled mean embedding has zero norm."""

    def __init__(self) -> None:
        super().__init__("mean embedding has zero norm")


class NonfiniteMeanEmbeddingError(ValueError):
    """Raised when a prototype or its norm is nonfinite."""

    def __init__(self) -> None:
        super().__init__(
            "sample is nonfinite, or a 2-norm required to normalize it "
            "overflows to nonfinite"
        )


class InsufficientUStatisticSampleError(ValueError):
    """Raised when a U-statistic block has fewer than two rows."""

    def __init__(self, size: int) -> None:
        super().__init__(f"U-statistic requires at least 2 observations, got {size}")


class UnbiasedMmdDistanceError(ValueError):
    """Raised when a signed U-statistic is requested as a rooted distance."""

    def __init__(self) -> None:
        super().__init__(
            "unbiased MMD squared estimates may be negative and cannot be rooted "
            "or branded as distances; use mmd_squared or mmd_squared_matrix"
        )


def _require_u_sample_size(size: int) -> None:
    if size < _MINIMUM_U_SAMPLE_SIZE:
        raise InsufficientUStatisticSampleError(size)


def _require_v_distance(*, unbiased: bool) -> None:
    if unbiased:
        raise UnbiasedMmdDistanceError

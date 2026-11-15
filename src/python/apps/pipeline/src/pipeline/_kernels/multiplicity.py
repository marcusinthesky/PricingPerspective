"""Host-side multiplicity adjustments for saved-result comparisons."""

from __future__ import annotations

from statsmodels.stats.multitest import multipletests


def holm_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Adjust available p-values within one declared family, preserving positions."""
    indices = [index for index, value in enumerate(pvalues) if value is not None]
    result: list[float | None] = [None] * len(pvalues)
    if indices:
        available = [pvalues[index] for index in indices]
        adjusted = multipletests(available, method="holm")[1]
        for index, value in zip(indices, adjusted, strict=True):
            result[index] = float(value)
    return result

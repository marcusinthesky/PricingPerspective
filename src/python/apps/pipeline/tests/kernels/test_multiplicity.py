"""Holm adjustment across partial saved-result families."""

from __future__ import annotations

import pytest
from pipeline._kernels.multiplicity import holm_adjust


@pytest.mark.parametrize("values", [[], [None], [None, None]])
def test_empty_available_family_preserves_missing_positions(
    values: list[float | None],
) -> None:
    """Unavailable comparisons do not create an artificial test family."""
    assert holm_adjust(values) == values


def test_holm_preserves_order_ties_and_missing_comparisons() -> None:
    """Four available hypotheses use one Holm family, independent of order."""
    actual = holm_adjust([0.04, None, 0.001, 0.2, 0.04])
    assert actual[1] is None
    assert [actual[index] for index in (0, 2, 3, 4)] == pytest.approx(
        [0.12, 0.004, 0.2, 0.12]
    )

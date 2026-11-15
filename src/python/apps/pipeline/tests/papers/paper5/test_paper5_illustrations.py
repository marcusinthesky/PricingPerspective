"""Tests for Paper 5's numerically honest conceptual illustrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pipeline.figures.paper5.illustrations import (
    render_paper5_illustrations,
    simulate_w2_operator_construction,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_w2_operator_construction_emits_an_admissible_row() -> None:
    """Keep target assignments fixed before producing a simplex operator row."""
    data = simulate_w2_operator_construction()

    assert data.aligned_candidates.shape == data.candidates.shape
    np.testing.assert_array_equal(data.row[data.self_index], 0.0)
    assert np.all(data.row >= 0.0)
    np.testing.assert_allclose(data.row.sum(), 1.0, atol=1e-12)
    unweighted = np.mean(data.aligned_candidates, axis=0)
    fitted_error = np.mean((data.target - data.reconstruction) ** 2)
    unweighted_error = np.mean((data.target - unweighted) ** 2)
    assert fitted_error < unweighted_error


def test_render_paper5_illustrations(tmp_path: Path) -> None:
    """Render exactly the two illustrations the chapter includes."""
    render_paper5_illustrations(tmp_path)

    assert not (tmp_path / "rank_one_collapse.pgf").exists()
    assert (tmp_path / "spatial_exposure_arc.pgf").exists()
    assert (tmp_path / "w2_operator_construction.pgf").exists()

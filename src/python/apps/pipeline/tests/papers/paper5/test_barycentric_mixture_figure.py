"""Unit contracts for the Paper 5 barycentric-mixture display."""

from pathlib import Path

import numpy as np
import yaml
from pipeline.figures.paper5.figures import (
    _BARYCENTRIC_TOP_COUNT,
    _five_largest_weights,
)
from pipeline.figures.paper5.figures import (
    _render_barycentric_mixture_map as _render_shared_mixture_map,
)


def test_barycentric_display_preserves_the_five_largest_actual_weights() -> None:
    """The display truncates visibility without renormalising a target row."""
    matrix = np.zeros((7, 7), dtype=np.float64)
    coefficients = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    for target in range(matrix.shape[0]):
        candidates = [source for source in range(matrix.shape[1]) if source != target]
        matrix[target, candidates] = coefficients

    displayed = _five_largest_weights(matrix)

    assert np.all(np.count_nonzero(displayed, axis=1) == _BARYCENTRIC_TOP_COUNT)
    assert np.allclose(displayed.sum(axis=1), 0.96)
    assert np.all(displayed <= matrix)
    assert np.allclose(np.diag(displayed), 0.0)


def test_barycentric_mixture_figure_is_a_paper5_dvc_leaf() -> None:
    """The figure depends only on the frozen W2 target-projection inputs."""
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dvc.yaml").exists()
    )
    stages = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    stage = stages["p5_barycentric_mixture_render"]

    assert "render-paper5-barycentric-mixture" in stage["cmd"]
    assert set(stage["deps"]) == {
        ".semflow/hashes/entrypoints/pipeline/render-paper5-barycentric-mixture.json",
        "${shared}/barycentres/${paper5.provider_id}/${paper5.representation_id}/${paper5.barycentre_arm_id}",
        "data/universe.csv",
    }
    outputs = "\n".join(str(item) for item in stage["outs"])
    assert "w2_barycentric_mixture_weights.pgf" in outputs
    assert "-jfec" not in outputs


def test_barycentric_mixture_figure_uses_barycentric_display_term(
    tmp_path: Path,
) -> None:
    """The reader-facing colorbar names the barycentric field coefficient."""
    matrix = np.full((3, 3), 0.5, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    output = tmp_path / "weights.pgf"

    _render_shared_mixture_map(output, matrix, ["AAA", "BBB", "CCC"])

    rendered = output.read_text(encoding="utf-8")
    assert r"Actual barycentric weight $W^\flat_{ij}$" in rendered

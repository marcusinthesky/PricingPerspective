"""Corporate replay protects weighted geometry, paired dates and saved evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hotelling.dynamics.artifacts import file_hash
from pipeline._kernels.paper6.corporate_analysis import (
    YEARS,
    corporate_summary,
    exact_transport,
    normalized,
    point_masses,
    redraw_masses,
    return_readout,
)
from pipeline.stages.papers.paper6.artifacts import write
from pipeline.stages.papers.paper6.corporate_analysis.campaign import (
    CELLS,
    CorporateConfig,
    _article_summaries,
    _completed,
    _manifest,
    atomic_npz,
    validate_checkpoint,
)
from pipeline.stages.papers.paper6.corporate_analysis.geometry import CostCache, pairs
from scipy.optimize import linprog

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline._kernels.paper6.corporate_analysis import Cloud


def _clouds() -> dict[tuple[int, int], Cloud]:
    return {
        (firm, year): {
            "points": np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            "dates": np.asarray(
                [f"{year}-01-01", f"{year}-01-03", f"{year}-01-04"],
                dtype="datetime64[D]",
            ),
            "core": np.asarray([True, True, False]),
        }
        for firm in range(3)
        for year in YEARS
    }


def test_weighted_unequal_transport_matches_independent_lp() -> None:
    """Squared costs and zero bootstrap masses agree with an independent LP."""
    left = np.asarray([[0.0], [0.7], [2.0]])
    right = np.asarray([[0.1], [1.5]])
    cost = (left - right.T) ** 2
    a, b = np.asarray([0.25, 0.0, 0.75]), np.asarray([0.6, 0.4])
    constraints = np.concatenate(
        (np.kron(np.eye(3), np.ones((1, 2))), np.kron(np.ones((1, 3)), np.eye(2))),
        axis=0,
    )
    solved = linprog(
        cost.ravel(),
        A_eq=constraints,
        b_eq=np.r_[a, b],
        bounds=(0, None),
        method="highs",
    )
    assert solved.success
    assert exact_transport(cost, a, b, certify=True) == pytest.approx(
        solved.fun, abs=1e-12
    )
    assert exact_transport(9 * cost, a, b) == pytest.approx(9 * solved.fun)


def test_prefix_normalization_precedes_squared_geometry() -> None:
    """Prefix representation discards the suffix before unit normalization."""
    points = np.asarray([[3.0, 4.0, 100.0], [-3.0, -4.0, 100.0]])
    transformed = normalized(points, 2)
    np.testing.assert_allclose(transformed, [[0.6, 0.8], [-0.6, -0.8]])
    assert float(np.sum((transformed[0] - transformed[1]) ** 2)) == pytest.approx(4)


def test_redraw_strata_keep_matched_and_expanded_samples_paired() -> None:
    """The canonical draw is shared across supports and every origin role."""
    clouds = _clouds()
    masses = redraw_masses(clouds, 20261026, 4)
    repeated = redraw_masses(clouds, 20261026, 4)
    for key in clouds:
        np.testing.assert_array_equal(masses["matched"][key], repeated["matched"][key])
        np.testing.assert_array_equal(
            masses["expanded"][key][:2] * 3, masses["matched"][key][:2] * 2
        )
        assert masses["matched"][key][2] == 0
        assert masses["expanded"][key].sum() == pytest.approx(1)
    point = point_masses(clouds, "matched")
    np.testing.assert_array_equal(point[0, 2018], [0.5, 0.5, 0])


def test_redraw_dates_and_strict_return_tails_use_the_same_masses() -> None:
    """Same-day news is excluded and dates follow selected article multiplicity."""
    clouds = _clouds()
    masses = {key: np.asarray([0.25, 0.75, 0.0]) for key in clouds}
    panels = {
        year: (
            np.asarray(
                [f"{year}-01-01", f"{year}-01-02", f"{year}-01-03"],
                dtype="datetime64[D]",
            ),
            np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]),
        )
        for year in YEARS[1:]
    }
    readout = return_readout(clouds, panels, masses, 3)
    np.testing.assert_allclose(readout[:, 0, 1], 0.75 * (2 + 20))
    np.testing.assert_allclose(readout[:, 1, 2], 0.75 * (3 + 30))
    assert np.isnan(readout[:, np.arange(3), np.arange(3)]).all()


def test_shared_origin_cache_reuses_opposite_orientation(tmp_path: Path) -> None:
    """The origin transport input is transposed rather than independently redrawn."""
    counts = np.asarray([[2, 3, 1]] * len(YEARS))
    cache = CostCache(tmp_path, counts)
    values = np.arange(cache.offsets[-1], dtype=float)
    np.save(tmp_path / "d64.npy", values)
    np.testing.assert_array_equal(cache.cost(64, 0, 0, 1), cache.cost(64, 0, 1, 0).T)
    assert (0, 1, 0) not in pairs(3)
    assert (1, 1, 0) in pairs(3)


def test_normalized_alignment_and_annual_components_use_exact_pooled_arrays() -> None:
    """A known within-array slope has unit fit and the expected scaling behavior."""
    rng = np.random.default_rng(104)
    returns = rng.normal(size=(4, 4, 4))
    distance = np.concatenate(
        (np.full((1, 4, 4), 100.0), 100 - np.cumsum(0.3 * returns, axis=0))
    )
    distance[:, np.arange(4), np.arange(4)] = np.nan
    returns[:, np.arange(4), np.arange(4)] = np.nan
    original = corporate_summary(distance, returns, list("ABCD"))
    rescaled = corporate_summary(distance, 2 * returns, list("ABCD"))
    assert original["beta"] == pytest.approx(0.3)
    assert original["normalized_alignment"] == pytest.approx(1)
    assert original["within_projection_fit"] == pytest.approx(1)
    assert rescaled["beta"] == pytest.approx(0.15)
    assert rescaled["normalized_alignment"] == pytest.approx(
        original["normalized_alignment"]
    )
    assert sum(
        row["share"] for row in original["year_denominator"]["shares"]
    ) == pytest.approx(1)
    assert sum(
        row["theta_contribution"] for row in original["annual_components"]
    ) == pytest.approx(original["theta"])


def test_checkpoint_corruption_and_design_mismatch_are_rejected(tmp_path: Path) -> None:
    """Replay cannot silently adopt truncated bytes or a different scenario."""
    path = tmp_path / "draw.npz"
    arrays = {"identity": np.asarray("identity"), "draw": np.asarray(2)}
    for name in CELLS:
        arrays.update(
            {
                f"{name}_beta": np.asarray(0.5),
                f"{name}_theta": np.asarray(1.0),
                f"{name}_second_moment": np.asarray(2.0),
                f"{name}_defined": np.ones((), dtype=bool),
            }
        )
    atomic_npz(path, arrays)
    digest = file_hash(path)
    validate_checkpoint(path, "identity", 2, digest)
    with pytest.raises(ValueError, match="design mismatch"):
        validate_checkpoint(path, "another", 2, digest)
    path.write_bytes(path.read_bytes()[:10])
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_checkpoint(path, "identity", 2, digest)


def test_article_summary_counts_undefined_attempts_and_uses_real_q05() -> None:
    """Unavailable draws stay visible and the fifth percentile is recomputed."""
    cells: dict[str, dict[str, Any]] = {name: {} for name in CELLS}
    arrays: dict[str, Any] = {}
    for name in CELLS:
        arrays[f"draw_{name}_beta"] = np.asarray([-2.0, 1.0, 2.0, np.nan])
        arrays[f"draw_{name}_defined"] = np.asarray([True, True, True, False])
    attempts, available = 4, 3
    _article_summaries(cells, arrays, attempts)
    for cell in cells.values():
        assert cell["article_draw_attempts"] == attempts
        assert cell["article_draw_defined"] == available
        assert cell["article_draw_undefined"] == 1
        assert cell["article_sensitivity_q05"] == pytest.approx(
            np.quantile([-2.0, 1.0, 2.0], 0.05)
        )
        assert cell["article_sensitivity_q05"] != cell["article_sensitivity_q025"]


def test_manifest_rejects_changed_design_without_touching_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different draw budgets require a separate analysis identity."""
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.corporate_analysis.campaign.entrypoint_code_identity",
        lambda _name: "frozen-code",
    )
    initial = _manifest(tmp_path, CorporateConfig(draws=4), {})
    assert _manifest(tmp_path, CorporateConfig(draws=4), {}) == initial
    with pytest.raises(ValueError, match="manifest incompatible"):
        _manifest(tmp_path, CorporateConfig(draws=199), {})


def test_completed_record_requires_hashes_and_every_declared_draw(
    tmp_path: Path,
) -> None:
    """A missing replicate or edited result cannot become publication data."""
    atomic_npz(tmp_path / "measurement.npz", {"measurement": np.asarray(1)})
    write(tmp_path / "reconciliation.json", {"status": "reconciled"})
    result = {
        "manifest_identity": "identity",
        "measurement_sha256": file_hash(tmp_path / "measurement.npz"),
        "status": "complete",
        "draws": 1,
    }
    write(tmp_path / "results.json", result)
    manifest: dict[str, Any] = {
        "identity": "identity",
        "design": {"configuration": {"draws": 1}},
        "checkpoints": {},
        "artifacts": {
            name: file_hash(tmp_path / name)
            for name in ("results.json", "reconciliation.json", "measurement.npz")
        },
    }
    with pytest.raises(ValueError, match="integrity mismatch"):
        _completed(tmp_path, manifest)
    write(tmp_path / "results.json", {**result, "draws": 99})
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _completed(tmp_path, manifest)

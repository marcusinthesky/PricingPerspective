"""Checks for Paper 1 Statadic W2 PERMANOVA and residual PERMDISP."""

from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pytest
from pipeline.stages.papers.paper1.w2_permanova import _analyze as run_w2_permanova
from scipy.stats import f_oneway


def _distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def test_scalar_point_mass_w2_matches_independent_coordinate_anova() -> None:
    """Squaring rooted W2 once recovers coordinate-based sums of squares."""
    x = np.array([0.0, 1.0, 4.0, 5.0])
    summary, audit = run_w2_permanova(
        np.abs(x[:, None] - x), ["A", "A", "B", "B"], n_permutations=31
    )
    main = summary["permanova"]
    assert isinstance(main, dict)
    assert main["ss_total"] == pytest.approx(np.sum((x - x.mean()) ** 2))
    assert main["ss_within"] == pytest.approx(1)
    assert main["ss_between"] == pytest.approx(16)
    assert main["r2"] == pytest.approx(16 / 17)
    assert main["pseudo_f"] == pytest.approx(float(f_oneway(x[:2], x[2:]).statistic))
    np.testing.assert_array_equal(
        np.sort(audit["permanova_permutation_indices"], axis=1),
        np.tile(np.arange(4), (31, 1)),
    )


def test_exact_fixed_size_relabeling_has_analytic_r2_expectation() -> None:
    """The expectation includes singleton groups and depends on K and N."""
    d = _distances(np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 5.0]]))
    schedule = np.array(list(permutations(range(4))))
    summary, audit = run_w2_permanova(
        d,
        ["A", "A", "B", "C"],
        n_permutations=len(schedule),
        permutation_indices=schedule,
    )
    main = summary["permanova"]
    assert isinstance(main, dict)
    assert main["null_r2_expectation"] == pytest.approx(2 / 3)
    assert audit["permanova_null_r2"].mean() == pytest.approx(2 / 3)
    assert main["excess_r2"] == pytest.approx(
        main["r2"] - audit["permanova_null_r2"].mean()
    )


def test_equal_distances_are_ties_with_p_one() -> None:
    """An intrinsically positive null R2 does not imply sector structure."""
    summary, audit = run_w2_permanova(
        1 - np.eye(8), ["A"] * 3 + ["B"] * 3 + ["C", "D"], n_permutations=41
    )
    main = summary["permanova"]
    assert isinstance(main, dict)
    assert main["r2"] == pytest.approx(3 / 7)
    assert main["pseudo_f"] == pytest.approx(1)
    assert main["pvalue"] == 1
    np.testing.assert_allclose(audit["permanova_null_pseudo_f"], 1)
    json.dumps(summary, allow_nan=False)


def test_scale_invariance_and_float32_admission() -> None:
    """Distance units and harmless source rounding preserve calibrated results."""
    points = np.array([[0.0], [1.0], [4.0], [8.0], [10.0], [13.0]])
    d = _distances(points)
    expected, audit = run_w2_permanova(d, ["A"] * 3 + ["B"] * 3, n_permutations=37)
    for scaled in (d * 1e-150, d * 1e150, d.astype(np.float32)):
        actual, observed_audit = run_w2_permanova(
            scaled, ["A"] * 3 + ["B"] * 3, n_permutations=37
        )
        expected_main, actual_main = expected["permanova"], actual["permanova"]
        assert isinstance(expected_main, dict)
        assert isinstance(actual_main, dict)
        assert actual_main["r2"] == pytest.approx(expected_main["r2"])
        assert actual_main["pseudo_f"] == pytest.approx(expected_main["pseudo_f"])
        assert actual_main["pvalue"] == expected_main["pvalue"]
        np.testing.assert_allclose(
            observed_audit["permanova_null_r2"], audit["permanova_null_r2"], atol=1e-14
        )
        json.dumps(actual, allow_nan=False)


def test_matching_schedules_preserve_row_order_invariance() -> None:
    """Reindexing whole firms and their schedules cannot alter either test."""
    d = _distances(np.array([[0.0], [1.0], [4.0], [8.0], [10.0], [13.0]]))
    labels = np.array(["A"] * 3 + ["B"] * 3)
    summary, audit = run_w2_permanova(d, labels.tolist(), n_permutations=43)
    order = np.array([2, 5, 0, 4, 1, 3])
    inverse = np.argsort(order)
    reordered, other = run_w2_permanova(
        d[np.ix_(order, order)],
        labels[order].tolist(),
        n_permutations=43,
        permutation_indices=inverse[audit["permanova_permutation_indices"][:, order]],
        dispersion_permutation_indices=inverse[
            audit["dispersion_permutation_indices"][:, order]
        ],
    )
    for section, statistic in (("permanova", "pseudo_f"), ("dispersion", "statistic")):
        expected, actual = summary[section], reordered[section]
        assert isinstance(expected, dict)
        assert isinstance(actual, dict)
        assert actual[statistic] == pytest.approx(expected[statistic])
        assert actual["pvalue"] == expected["pvalue"]
    np.testing.assert_allclose(
        audit["dispersion_null_f"], other["dispersion_null_f"], atol=1e-12
    )


def test_singletons_are_retained_only_in_main_test() -> None:
    """Removing singleton spread contributions cannot shrink the main cohort."""
    d = _distances(
        np.array([[0.0], [1.0], [4.0], [8.0], [10.0], [13.0], [20.0], [25.0]])
    )
    summary, audit = run_w2_permanova(
        d, ["A"] * 3 + ["B"] * 3 + ["C", "D"], n_permutations=11
    )
    assert summary["n_firms"] == len(d)
    dispersion = summary["dispersion"]
    assert isinstance(dispersion, dict)
    assert dispersion["n_firms"] == len(d) - len(["C", "D"])
    assert dispersion["n_groups"] == len(["A", "B"])
    assert dispersion["excluded_singleton_sectors"] == ["C", "D"]
    assert audit["permanova_permutation_indices"].shape == (11, 8)
    assert audit["dispersion_permutation_indices"].shape == (11, 6)


def test_companion_matches_full_model_residual_permutation_reference() -> None:
    """Refit permuted full-model residuals, without adding fitted group means."""
    x = np.array([0.0, 1.0, 4.0, 8.0, 10.0, 13.0])
    labels = ["A"] * 3 + ["B"] * 3
    schedule = np.array([[0, 1, 2, 3, 4, 5], [1, 3, 4, 0, 5, 2], [5, 4, 3, 2, 1, 0]])
    summary, audit = run_w2_permanova(
        np.abs(x[:, None] - x),
        labels,
        n_permutations=3,
        dispersion_permutation_indices=schedule,
    )
    radii = np.array([1.0, 0.0, 3.0, 2.0, 0.0, 3.0]) * np.sqrt(3 / 2)
    residuals = np.concatenate(
        [group - group.mean() for group in (radii[:3], radii[3:])]
    )
    expected = np.array(
        [
            float(f_oneway(residuals[p[:3]], residuals[p[3:]]).statistic)
            for p in schedule
        ]
    )
    dispersion = summary["dispersion"]
    assert isinstance(dispersion, dict)
    assert dispersion["status"] == "computed"
    np.testing.assert_allclose(audit["dispersion_radii"], radii / x.max(), atol=1e-12)
    np.testing.assert_allclose(audit["dispersion_null_f"], expected, atol=1e-12)
    assert dispersion["statistic"] == pytest.approx(
        float(f_oneway(radii[:3], radii[3:]).statistic)
    )


def test_zero_distances_export_explicit_undefined_values() -> None:
    """A zero denominator is an explicit missing statistic, never JSON NaN."""
    summary, _ = run_w2_permanova(
        np.zeros((6, 6)), ["A"] * 3 + ["B"] * 3, n_permutations=7
    )
    main = summary["permanova"]
    assert isinstance(main, dict)
    assert main["status"] == "not_computed"
    assert main["reason"] == "all_distances_zero"
    assert main["r2"] is None
    assert main["pvalue"] is None
    json.dumps(summary, allow_nan=False)


def test_infinite_f_ties_remain_in_permutation_denominator() -> None:
    """Perfect separation is calibrated with all finite and infinite draws."""
    x = np.array([0.0, 0.0, 1.0, 1.0])
    schedule = np.array(list(permutations(range(4))))
    summary, audit = run_w2_permanova(
        np.abs(x[:, None] - x),
        ["A", "A", "B", "B"],
        n_permutations=24,
        permutation_indices=schedule,
    )
    main = summary["permanova"]
    assert isinstance(main, dict)
    assert main["pseudo_f"] is None
    assert main["pseudo_f_status"] == "positive_infinity"
    assert main["pvalue"] == pytest.approx(
        (1 + np.isposinf(audit["permanova_null_pseudo_f"]).sum()) / 25
    )
    json.dumps(summary, allow_nan=False)


def test_material_negative_inertia_withholds_only_companion() -> None:
    """A non-Euclidean cycle retains its original-distance main statistic."""
    d = np.array([[0.0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    summary, _ = run_w2_permanova(d, ["A", "A", "B", "B"], n_permutations=9)
    main, dispersion = summary["permanova"], summary["dispersion"]
    assert isinstance(main, dict)
    assert isinstance(dispersion, dict)
    assert main["status"] == "computed"
    assert main["ss_total"] == pytest.approx(np.sum(np.triu(d**2, 1)) / 4)
    assert dispersion["reason"] == "material_negative_inertia"
    assert dispersion["pvalue"] is None


def test_companion_uses_every_positive_axis_and_reports_degenerate_radius_model() -> (
    None
):
    """Tetrahedra have known three-dimensional radii that two axes cannot retain."""
    tetrahedron = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    points = np.vstack([tetrahedron, 3 * tetrahedron + [10, 0, 0]])
    d = _distances(points)
    summary, audit = run_w2_permanova(d, ["A"] * 4 + ["B"] * 4, n_permutations=5)
    expected = np.array([2.0] * 4 + [6.0] * 4) / d.max()
    np.testing.assert_allclose(audit["dispersion_radii"], expected, atol=1e-12)
    dispersion = summary["dispersion"]
    assert isinstance(dispersion, dict)
    assert dispersion["reason"] == "zero_within_group_radius_variation"
    assert dispersion["pvalue"] is None


def test_unrepresentable_squared_units_preserve_scale_free_inference() -> None:
    """Overflow in reporting units never becomes JSON infinity or invalid inference."""
    d = _distances(np.array([[0.0], [1.0], [4.0], [8.0], [10.0], [13.0]]))
    summary, _ = run_w2_permanova(d * 1e200, ["A"] * 3 + ["B"] * 3, n_permutations=5)
    main = summary["permanova"]
    assert isinstance(main, dict)
    assert main["sum_of_squares_units_status"] == "outside_float64_range"
    assert main["ss_total"] is None
    assert main["pseudo_f_status"] == "finite"
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    "defect",
    ["negative", "asymmetric", "nonzero_diagonal", "nonfinite", "malformed_schedule"],
)
def test_malformed_scientific_inputs_are_rejected(defect: str) -> None:
    """Reject malformed matrices and schedules before statistical computation."""
    d = 1 - np.eye(4)
    schedule = np.tile(np.arange(4), (3, 1))
    if defect == "negative":
        d[0, 1] = -1
    elif defect == "asymmetric":
        d[0, 1] = 2
    elif defect == "nonzero_diagonal":
        d[0, 0] = 1
    elif defect == "nonfinite":
        d[0, 0] = np.nan
    else:
        schedule[0, 0] = 1
    with pytest.raises(ValueError, match=r"distances|diagonal|permutation"):
        run_w2_permanova(
            d, ["A", "A", "B", "B"], n_permutations=3, permutation_indices=schedule
        )

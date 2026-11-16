"""Timing, isolation, uncertainty, and placebo contracts for Paper 3 validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pipeline._kernels.paper3.validation import solve_certificate
from pipeline.stages.papers.paper3.certificate_validation import (
    ValidationConfig,
    _credit,
    _credit_lower,
    _latest_vintage,
    _monthly_formations,
    _permuted_matrix,
    _select_cell,
    _windows,
)

LOOKBACK = 5
HORIZON = 3
LATEST_FIRST_TRADING_DAY = 3
KKT_TOLERANCE = 1e-6
REPO_ROOT = Path(__file__).resolve().parents[7]
FROZEN_MATRICES = REPO_ROOT / "data/papers/paper3/pit_w2/matrices.npz"
GOVERNED_OUTPUT_COUNT = 5
ARTICLE_REPLICATIONS = 499
RETURN_REPLICATIONS = 2_000


def _config() -> ValidationConfig:
    return ValidationConfig(
        lookback=LOOKBACK,
        horizon=HORIZON,
        calibration_start=pd.Timestamp("2020-01-01"),
        calibration_end=pd.Timestamp("2020-03-31"),
        evaluation_start=pd.Timestamp("2023-01-04"),
        evaluation_end=pd.Timestamp("2023-03-31"),
        frozen_vintage="2022-12-31",
        scales=(1.0,),
        slacks=(0.0,),
        coverage_target=0.0,
        confidence=0.95,
        article_reps=3,
        return_reps=20,
        seed=42,
        cap=0.6,
        label_permutations=3,
        residual_budgets=(0.0, 0.025, 0.05),
    )


def test_windows_use_strictly_prior_and_subsequent_returns() -> None:
    """Formation is excluded from the trailing window and starts evaluation."""
    dates = pd.bdate_range("2019-12-20", periods=20)
    panel = pd.DataFrame({"A": np.arange(20.0), "B": np.arange(20.0)}, index=dates)
    formation = dates[8]
    trailing, future = _windows(panel, formation, _config())
    assert trailing.index.max() < formation
    assert future.index.min() == formation
    assert len(trailing) == LOOKBACK
    assert len(future) == HORIZON


def test_vintage_selection_is_strict_and_holdout_is_frozen() -> None:
    """Calibration uses the latest strictly prior vintage; holdout sees 2022."""
    vintages = ("2019-12-31", "2020-12-31", "2022-12-31")
    assert _latest_vintage(vintages, pd.Timestamp("2020-12-31")) == "2019-12-31"
    assert _latest_vintage(vintages, pd.Timestamp("2023-01-04")) == "2022-12-31"


def test_monthly_formations_respect_available_horizon() -> None:
    """Only the first eligible date of a month with a full future window forms."""
    dates = pd.bdate_range("2019-12-20", "2020-03-31")
    formations = _monthly_formations(
        dates,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-03-31"),
        lookback=5,
        horizon=20,
        horizon_end=pd.Timestamp("2020-03-31"),
    )
    assert formations[0] == pd.Timestamp("2020-01-01")
    assert all(date.day <= LATEST_FIRST_TRADING_DAY for date in formations)


def test_article_lower_endpoint_does_not_exceed_point_credit() -> None:
    """The designed lower-tail draws remain below their point geometry."""
    point = np.array([[0.0, 1.0], [1.0, 0.0]])
    draws = np.stack([0.5 * point, 0.8 * point, point])
    q = np.array([0.5, 0.5])
    lower = _credit_lower(draws, q, 1.0, 0.0, 0.95, _credit(point, q))
    assert lower <= _credit(point, q)


def test_holdout_perturbation_cannot_change_calibration_selection() -> None:
    """Changing post-calibration returns leaves the selected cell unchanged."""
    dates = pd.bdate_range("2019-12-20", "2023-02-28")
    values = np.column_stack(
        [np.sin(np.arange(len(dates)) / 7.0), np.cos(np.arange(len(dates)) / 9.0)]
    )
    panel = pd.DataFrame(values, index=dates, columns=["A", "B"])
    formations = (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-03"))
    matrix = np.array([[0.0, 0.25], [0.25, 0.0]])
    article = np.stack([matrix, matrix, matrix])
    config = _config()
    first = _select_cell(
        formations,
        panel,
        {"2019-12-31": matrix},
        {"2019-12-31": article},
        config,
    )[:2]
    perturbed = panel.copy()
    perturbed.loc["2023-01-01":] *= 1000.0
    second = _select_cell(
        formations,
        perturbed,
        {"2019-12-31": matrix},
        {"2019-12-31": article},
        config,
    )[:2]
    assert first == second


def test_label_permutation_preserves_matrix_identity() -> None:
    """A placebo relabels both axes with a bijection and changes no entries."""
    matrix = np.arange(16.0).reshape(4, 4)
    matrix = matrix + matrix.T
    permutation = np.array([2, 0, 3, 1])
    permuted = _permuted_matrix(matrix, permutation)
    assert np.sort(permuted.ravel()) == pytest.approx(np.sort(matrix.ravel()))
    assert np.array_equal(permuted, permuted.T)


@pytest.mark.skipif(
    not FROZEN_MATRICES.is_file(),
    reason=f"{FROZEN_MATRICES} is DVC-tracked and absent from CI checkouts",
)
def test_registered_frozen_certificate_kernels_pass_fail_closed_gate() -> None:
    """Every predeclared 2022 cell has the required positive centered kernel."""
    with np.load(FROZEN_MATRICES) as archive:
        squared_w2 = archive["2022-12-31"]
    for scale in (1.0, 1.1, 1.25, 1.5):
        for slack in (0.0, 0.025, 0.05, 0.1):
            floor = np.square(
                np.maximum(0.0, np.sqrt(squared_w2) / scale - 2.0 * slack)
            )
            solution = solve_certificate(floor, 0.1)
            assert solution.minimum_centered_kernel_eigenvalue > 0.0
            assert solution.kkt_residual <= KKT_TOLERANCE


def test_dvc_stage_declares_frozen_validation_interface() -> None:
    """The governed stage exposes the preregistered inputs and six artifacts."""
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    stage = dvc["stages"]["p3_certificate_validation"]
    command = stage["cmd"]
    assert "paper3-certificate-validation" in command
    assert "--holdout-returns-dir ${shared}/oos/returns" in command
    assert stage["params"] == [
        "paper3.certificate_validation",
        "paper3.pit_vintage_grid",
        "typed_analysis.providers.qwen3-embedding-8b",
        "typed_analysis.representations.qwen3-embedding-8b-unit",
        "typed_analysis.distances.wasserstein_w2_pit_2018",
        "typed_analysis.distances.wasserstein_w2_pit_2019",
        "typed_analysis.distances.wasserstein_w2_pit_2020",
        "typed_analysis.distances.wasserstein_w2_pit_2021",
        "typed_analysis.distances.wasserstein_w2_pit_2022",
        "typed_analysis.windows.pit-2018-12-31",
        "typed_analysis.windows.pit-2019-12-31",
        "typed_analysis.windows.pit-2020-12-31",
        "typed_analysis.windows.pit-2021-12-31",
        "typed_analysis.windows.pit-2022-12-31",
    ]
    assert len(stage["outs"]) == GOVERNED_OUTPUT_COUNT
    validation = params["paper3"]["certificate_validation"]
    assert validation["evaluation_start"] == "2023-01-04"
    assert validation["evaluation_end"] == "2026-04-06"
    assert validation["article_block_replications"] == ARTICLE_REPLICATIONS
    assert validation["return_block_replications"] == RETURN_REPLICATIONS

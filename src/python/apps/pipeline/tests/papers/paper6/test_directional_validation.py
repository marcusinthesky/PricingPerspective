"""Zero targets, attempted-run accounting and immutable directional evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pipeline._kernels.paper6.directional_validation import (
    HISTORIES,
    SCENARIOS,
    DirectionalValidationConfig,
    _population_target,
    legacy_directional,
    legacy_rows,
    measure_validation_history,
    panel_diagnostics,
    performance,
    summarize_rows,
)
from pipeline._kernels.paper6.recovery import (
    HistoryMeasurement,
    article_panels,
    article_redraws,
)
from pipeline._kernels.paper6.reference_alignment import recovery_summary
from pipeline.stages.papers.paper6.directional_validation import (
    _check_membership,
    _failures,
    _manifest,
    _read_frame,
    _validate_design,
    _write_frame,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _history() -> HistoryMeasurement:
    return HistoryMeasurement(
        1, 1, np.ones(5), np.ones(5), np.ones((2, 3, 128)), 0, 1, 1, 1, 0
    )


def test_zero_oracle_retains_absolute_metrics() -> None:
    """Zero targets retain errors without inventing normalized or sign recovery."""
    estimates = np.array([[0.0, 1.0, -1.0, 0.0]])
    summary = recovery_summary(estimates, np.zeros(1))
    assert summary.rmse == pytest.approx(np.sqrt(0.5))
    assert summary.mae == pytest.approx(0.5)
    assert summary.bias == 0
    assert summary.normalized_rmse is None
    assert summary.normalization_reason == "zero_oracle_scale"
    assert not summary.recovered
    directional = performance(estimates, np.zeros(1))
    assert directional["correct_sign"] is None
    assert directional["wrong_sign"] is None
    assert directional["sign_ineligible_attempts"] == estimates.size
    assert directional["above_zero"]["rate"] == pytest.approx(0.25)
    assert directional["legacy_recovered"] is None
    assert directional["legacy_magnitude_status"] == "unavailable"
    assert directional["legacy_magnitude_reason"] == "zero_oracle_scale"


def test_signs_and_undefined_keep_attempt_denominators() -> None:
    """Undefined attempts are retained in rates and suppress complete errors."""
    estimates = np.array([[1.0, -1.0, 0.0, np.nan], [-1.0, 1.0, 0.0, np.inf]])
    summary = performance(estimates, np.array([1.0, -1.0]))
    assert summary["correct_sign"]["rate"] == pytest.approx(0.25)
    assert summary["wrong_sign"]["rate"] == pytest.approx(0.25)
    assert summary["correct_sign"]["denominator"] == estimates.size
    assert summary["undefined"] == len(estimates)
    assert summary["unresolved"]["rate"] == pytest.approx(0.25)
    assert summary["rmse"] is None
    assert summary["legacy_recovered"] is None
    assert summary["legacy_magnitude_status"] == "unavailable"
    assert summary["legacy_magnitude_reason"] == "undefined_estimates"


@pytest.mark.parametrize(("estimate", "recovered"), [(1.0, True), (2.0, False)])
def test_eligible_legacy_magnitude_retains_pass_fail_decisions(
    estimate: float, *, recovered: bool
) -> None:
    """Positive scales preserve the original normalized-error pass/fail rule."""
    summary = performance(np.full((1, 2), estimate), np.ones(1))
    assert summary["normalized_rmse"] == pytest.approx(abs(estimate - 1))
    assert summary["legacy_recovered"] is recovered
    assert summary["legacy_magnitude_status"] == ("pass" if recovered else "fail")
    assert summary["legacy_magnitude_reason"] is None


def test_undefined_target_cannot_be_a_legacy_magnitude_failure() -> None:
    """An unavailable target makes the benchmark unavailable with its reason."""
    summary = performance(np.ones((1, 2)), np.array([np.nan]))
    assert summary["legacy_recovered"] is None
    assert summary["legacy_magnitude_status"] == "unavailable"
    assert summary["legacy_magnitude_reason"] == "undefined_target"


def test_mcse_uses_within_history_not_between_history_dispersion() -> None:
    """Exhaustively enumerated strata do not create Monte Carlo uncertainty."""
    summary = performance(np.array([[1.0, 1.0], [-1.0, -1.0]]), np.ones(2))
    assert summary["above_zero"]["rate"] == pytest.approx(0.5)
    assert summary["above_zero"]["mcse"] == 0
    assert summary["bias_mcse"] == 0


def test_legacy_literal_positive_diagnostic_has_unchanged_denominator() -> None:
    """Historical literal positivity remains separate from new sign resolution."""
    history = _history()
    sample = history.estimates.copy()
    sample[:, :, :4] = [0, 1e-15, np.nan, np.inf]
    records = [replace(history, estimates=sample) for _ in range(HISTORIES)]
    summary = legacy_directional(records)
    first = summary["cells"][0][0]
    assert first["total"] == HISTORIES * 128
    assert first["positive"] == HISTORIES * 125
    assert first["undefined"] == HISTORIES * 2
    p = 125 / 128
    assert first["mcse"] == pytest.approx(np.sqrt(p * (1 - p) / (127 * HISTORIES)))
    rows = legacy_rows(records)
    assert (
        rows[0]["moment_unavailability_reason"] == "not_saved_in_historical_checkpoint"
    )
    assert rows[0]["q"] is None
    with pytest.raises(ValueError, match="sixteen"):
        legacy_rows(records[:-1])


def test_stationary_population_control_is_identified_with_zero_approach() -> None:
    """A constant reported law has zero population approach and positive d."""
    pooled = np.broadcast_to(
        np.array(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.2, 0.2, 0.3, 0.3],
                [0.3, 0.3, 0.2, 0.2],
                [0.4, 0.4, 0.1, 0.1],
            ]
        ),
        (5, 4, 4),
    )
    counts = np.ones((5, 12, 4), dtype=np.int64)
    news = np.zeros((60, 4))
    news[[18, 30, 42, 54]] = [-1.0, -0.2, 0.3, 1.0]
    summary = panel_diagnostics(pooled, counts, news)
    assert summary["q"] == 0
    assert summary["beta"] == 0
    assert summary["d"] > 0
    assert summary["sign"] == 0
    assert summary["candidate_se"] == 0
    assert summary["normalized_alignment"] is None


def test_shared_sampler_reproduces_legacy_stream_without_retaining_tensors() -> None:
    """Scalar diagnostics reuse legacy panels; pilots use separate random keys."""
    counts = np.ones((5, 12, 4), dtype=np.int64)
    laws = np.full((2, 5, 12, 4, 4), 0.25)
    news = np.zeros((60, 4))
    news[[18, 30, 42, 54]] = [-1.0, -0.2, 0.3, 1.0]
    expected = article_redraws(laws, counts, news, 3, redraws=2)
    actual = np.empty_like(expected)
    for regime, budget, replicate, pooled, dates in article_panels(
        laws, counts, 3, redraws=2
    ):
        actual[regime, budget, replicate] = panel_diagnostics(pooled, dates, news)[
            "beta"
        ]
    np.testing.assert_array_equal(expected, actual)
    original = next(article_panels(laws, counts, 3, redraws=2))[3]
    pilot = next(article_panels(laws, counts, 3, redraws=2, stream=(99,)))[3]
    assert not np.array_equal(original, pilot)


def test_complete_attempt_accounting_and_paired_budget_uncertainty() -> None:
    """A failed economy contributes every declared attempted panel to summaries."""
    rows = [
        _failures(history, 2, RuntimeError("failed")) for history in range(HISTORIES)
    ]
    flattened = [item for group in rows for item in group]
    _check_membership(flattened, 2)
    summary = summarize_rows(flattened, redraws=2)
    assert all(item["undefined"] == HISTORIES * 2 for item in summary["aggregate"])
    assert all(item["correct_sign"] is None for item in summary["aggregate"])
    positive = [
        item
        for item in summary["paired_budget_contrasts"]
        if item["metric"] == "positive"
    ]
    assert all(item["mcse"] == 0 for item in positive)
    with pytest.raises(ValueError, match="membership"):
        _check_membership(flattened[:-1], 2)
    with pytest.raises(ValueError, match="strata"):
        summarize_rows(flattened[:-1], redraws=2)


def test_parquet_nullable_checkpoint_detects_corruption(tmp_path: Path) -> None:
    """Durable scalar records retain missing quantities and reject changed bytes."""
    path = tmp_path / "rows.parquet"
    digest = _write_frame(
        path,
        [
            {"beta": 1.0, "q": 1.0, "defined": True},
            {"beta": None, "q": None, "defined": False},
        ],
    )
    restored = _read_frame(path, digest)
    assert restored[1]["beta"] is None
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        _read_frame(path, digest)


def test_design_cannot_change_replication_count_or_precision() -> None:
    """The fixed budget and numerical resolution cannot drift after outcomes."""
    with pytest.raises(ValueError, match="160"):
        DirectionalValidationConfig.model_validate({"redraws": 128})
    with pytest.raises(ValueError, match="fixed conditional"):
        _validate_design(
            {"frozen_before_new_results": True}, DirectionalValidationConfig()
        )


def test_path_failure_retains_attempts_for_each_declared_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed path leaves explicit unavailability instead of shrinking results."""

    def fail(*_args: object, **_kwargs: object) -> None:
        message = "domain departure"
        raise ArithmeticError(message)

    monkeypatch.setattr(
        "pipeline._kernels.paper6.directional_validation.solve_history", fail
    )
    rows = measure_validation_history(
        0, np.ones((5, 12, 100), dtype=np.int64), redraws=2
    )
    assert len(rows) == sum(len(s.regimes) * len(s.budgets) * 2 for s in SCENARIOS)
    assert all(not row["execution_succeeded"] for row in rows)
    assert all(row["oracle_beta"] is None for row in rows)


def test_resume_requires_identical_design_and_checkpoint_bytes(tmp_path: Path) -> None:
    """New manifests verify design copies and reject incompatible resumption."""
    design = tmp_path / "source-design.json"
    design.write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    identity: dict[str, Any] = {"code_identity": "fixed", "design": "fixed"}
    first = _manifest(output, design, identity)
    assert _manifest(output, design, identity) == first
    with pytest.raises(ValueError, match="incompatible"):
        _manifest(output, design, {**identity, "code_identity": "changed"})
    (output / "design.json").write_text("changed")
    with pytest.raises(ValueError, match="output hash mismatch"):
        _manifest(output, design, identity)


def test_structural_zero_retains_numerical_residual_without_normalizing_by_it() -> None:
    """Only known stationary-law targets are exact zero; small alternatives remain."""
    raw = {"beta": 1.5e-15, "q": 3.0e-16, "d": 0.2, "sign": 0, "approach_rms": 2.0e-16}
    zero = _population_target(raw, constructed_zero=True)
    assert zero["beta"] == 0
    assert zero["q"] == 0
    assert zero["arithmetic_beta"] == raw["beta"]
    assert zero["arithmetic_q"] == raw["q"]
    assert _population_target(raw, constructed_zero=False) == raw
    summary = performance(np.array([[0.1, -0.1]]), np.array([zero["beta"]]))
    assert summary["normalized_rmse"] is None
    assert summary["normalization_reason"] == "zero_oracle_scale"


def test_locality_summaries_keep_tangent_target_and_oracle_sign_changes() -> None:
    """A finite positive oracle cannot hide a negative tangent comparison."""
    rows = [
        row
        for h in range(HISTORIES)
        for row in _failures(h, 2, ArithmeticError("fixture"))
        if row["scenario_id"] == "D_half"
    ]
    for row in rows:
        row.update(
            beta=1.0,
            sign=1,
            oracle_beta=1.0,
            oracle_q=1.0,
            oracle_d=1.0,
            oracle_sign=1,
            operating_beta=1.0,
            operating_q=1.0,
            operating_d=1.0,
            operating_sign=1,
            tangent_beta=-1.0,
            tangent_q=-1.0,
            tangent_d=1.0,
            tangent_sign=-1,
            execution_succeeded=True,
            premises_held=True,
        )
    summary = summarize_rows(rows, redraws=2)
    tangent = next(
        item for item in summary["aggregate"] if item["target"] == "tangent_beta0_h"
    )
    assert tangent["wrong_sign"]["rate"] == 1
    assert tangent["correct_sign_denominator"] == len(rows)
    assert summary["locality"][0]["finite_tangent_sign_disagreements"] == HISTORIES
    assert summary["locality"][0]["tangent_oracle_sign_counts"]["-1"] == HISTORIES


def test_equal_history_ratios_and_moments_remain_separate_without_dropping_strata() -> (
    None
):
    """The average coefficient differs from the ratio of mean q and mean d."""
    rows = [
        row
        for h in range(HISTORIES)
        for row in _failures(h, 2, ArithmeticError("fixture"))
        if row["scenario_id"] == "A"
        and row["regime"] == "common"
        and row["budget"] == 1
    ]
    q = np.arange(1.0, HISTORIES + 1)
    d = 1.0 + np.arange(HISTORIES) % 2
    for row in rows:
        h = row["history"]
        row.update(
            beta=q[h] / d[h],
            sign=1,
            oracle_beta=q[h] / d[h],
            oracle_q=q[h],
            oracle_d=d[h],
            oracle_sign=1,
            operating_beta=q[h] / d[h],
            operating_q=q[h],
            operating_d=d[h],
            operating_sign=1,
            execution_succeeded=True,
            premises_held=True,
        )
    summary = summarize_rows(rows, redraws=2)["aggregate"][0]
    assert summary["equal_history_mean_ratio"] == pytest.approx(np.mean(q / d))
    assert summary["ratio_of_equal_history_moments"] == pytest.approx(
        np.mean(q) / np.mean(d)
    )
    assert summary["equal_history_mean_q"] == pytest.approx(np.mean(q))
    assert summary["equal_history_mean_d"] == pytest.approx(np.mean(d))
    assert (
        summary["equal_history_mean_ratio"] != summary["ratio_of_equal_history_moments"]
    )
    rows[0]["oracle_q"] = None
    missing = summarize_rows(rows, redraws=2)["aggregate"][0]
    assert missing["ratio_of_equal_history_moments"] is None
    assert (
        missing["ratio_of_equal_history_moments_reason"]
        == "missing_history_oracle_moments"
    )

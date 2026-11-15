"""Publish saved calibration and inference summaries without numerical inputs."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from pipeline.figures.paper6.article_redraw_validation import render as render_redraw
from pipeline.figures.paper6.inference_audit import render as render_inference

if TYPE_CHECKING:
    from pathlib import Path


def _redraw_record() -> dict[str, Any]:
    """Supply adverse saved rates and distinct conditioning denominators."""
    rate: dict[str, Any] = {
        "equal_history_rate": 0.4,
        "mcse": 0.009,
        "count": 1024,
        "attempts": 2560,
        "available_attempts": 2048,
        "unavailable_attempts": 512,
        "available_rate": 0.5,
        "unavailable_histories": [],
        "unavailable_reason": None,
    }
    return {
        "schema_version": 1,
        "histories": 16,
        "outer_replications": 160,
        "inner_redraws": 199,
        "signal_scale": {
            "status": "complete",
            "units": "fractions",
            "first_response_abs_share_change_min": 0.0002,
            "first_response_abs_share_change_max": 0.0004,
            "maximum_abs_share_change_from_origin": 0.001,
            "news_source_tick": 18,
            "first_response_tick": 19,
        },
        "scenarios": [
            {
                "scenario_id": scenario,
                "budget": 1,
                "support": "expanded",
                "target": "beta_lambda_h",
                "outer_attempts": 2560,
                "available_outer_attempts": 2048,
                "sampling_target": "Fixed-history reported-law projection.",
                "lower_bound_method": "article_sensitivity_q05",
                "quantile_method": "linear",
                "status": "simulation_calibration_only",
                "reason": "Corporate coverage is not established.",
                "positive_q05_interpretation": (
                    "null rejection" if scenario == "B" else "positive-bound frequency"
                ),
                "inner_undefined": 7,
                "outer_failed_execution": 3,
                "failed_premises": 2,
                **{
                    measure: copy.deepcopy(rate)
                    for measure in (
                        "central_containment",
                        "q05_containment",
                        "positive_q05",
                    )
                },
            }
            for scenario in ("C", "B", "A")
        ],
    }


def _inference_record(extension: dict[str, Any]) -> dict[str, Any]:
    """Reuse the saved N=16 fixture shape for independently named sizes."""
    baseline = copy.deepcopy(extension["inference_status"]["iid_fixture"])
    for cell in baseline["scenarios"]:
        cell.update(
            attempts=2500,
            undefined_estimates=3,
            unavailable_candidate_se=5,
            degenerate_or_negative_candidate_se=2,
            usable_bounds=2490,
            one_sided_z=1.6448536269514715,
        )
    current = copy.deepcopy(baseline)
    current["firms"] = 100
    current["scenarios"][0]["target_containment"].update(rate=0.7, mcse=0.009)
    return {
        "schema_version": 1,
        "iid_fixture": current,
        "baseline_iid_fixture": baseline,
        "empirical_inference": copy.deepcopy(
            extension["inference_status"]["empirical_inference"]
        ),
    }


def _render(
    tmp_path: Path, record: dict[str, Any], *, inference: bool = False
) -> tuple[str, str]:
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(record))
    project = tmp_path / "manuscript"
    generated = project / "src/generated"
    generated.mkdir(parents=True)
    retained = generated / "information_recovery_numbers.tex"
    retained.write_text("retained legacy bindings")
    render = render_inference if inference else render_redraw
    render(source, project)
    assert retained.read_text() == "retained legacy bindings"
    prefix = "inference_audit" if inference else "article_redraw_validation"
    return (
        (generated / f"{prefix}_numbers.tex").read_text(),
        (generated / f"{prefix}_table.tex").read_text(),
    )


def _number(catalog: str, key: str) -> float:
    match = re.search(
        r"\\ppDeclareValue\{" + re.escape(key) + r"\}\{([^}]+)\}", catalog
    )
    assert match is not None
    return float(match[1])


def test_redraw_publication_preserves_adverse_rates_and_sampling_scope(
    tmp_path: Path,
) -> None:
    """The three controls retain all-attempt rates and separate availability."""
    numbers, table = _render(tmp_path, _redraw_record())
    assert "{ir-redraw-mechanism-central-coverage-pct}{40}" in numbers
    assert "{ir-redraw-mechanism-central-coverage-available-only-pct}{50}" in numbers
    assert "{ir-redraw-stationary-positive-q05-available}{2048}" in numbers
    assert "{ir-redraw-stationary-positive-q05-attempts}{2560}" in numbers
    assert _number(numbers, "ir-redraw-signal-first-response-min") == pytest.approx(
        0.0002
    )
    assert _number(numbers, "ir-redraw-signal-first-response-min-pct") == pytest.approx(
        0.02
    )
    assert _number(numbers, "ir-redraw-signal-first-response-max-pct") == pytest.approx(
        0.04
    )
    assert _number(
        numbers, "ir-redraw-signal-maximum-origin-change-pct"
    ) == pytest.approx(0.1)
    assert "{ir-redraw-signal-first-response-tick}{19}" in numbers
    assert table.index("Common reporting &") < table.index("Stationary reporting &")
    assert table.index("Stationary reporting &") < table.index("Reporting only &")
    assert "zero-null rejection frequency only for the stationary" in table
    assert "under simulated reporting laws" in table
    assert "matching history-specific reported-law target" in table
    assert "ir-redraw-reporting-failed-premises" in table
    assert "T83" not in table
    body = table.split(r"\midrule", 1)[1]
    assert "central-coverage-available" not in body
    assert "central-coverage-attempts" not in body
    assert "ir-redraw-mechanism-central-coverage-attempts" in table
    assert "ir-redraw-mechanism-central-coverage-available" in table
    assert "Unavailable bounds remain non-events" in table
    clean = _redraw_record()
    for cell in clean["scenarios"]:
        cell.update(
            inner_undefined=0,
            outer_failed_execution=0,
            failed_premises=0,
            available_outer_attempts=2560,
        )
        for measure in ("central_containment", "q05_containment", "positive_q05"):
            cell[measure].update(
                available_attempts=2560, unavailable_attempts=0, available_rate=0.4
            )
    clean_path = tmp_path / "clean"
    clean_path.mkdir()
    _, clean_table = _render(clean_path, clean)
    assert "Unavailable bounds remain non-events" not in clean_table
    assert "missing history target" not in clean_table
    assert "saved reason" not in clean_table
    assert "Exceptions" not in clean_table
    assert "ir-redraw-mechanism-central-coverage-attempts" in clean_table


def test_redraw_unequal_availability_remains_visible_beside_saved_rates(
    tmp_path: Path,
) -> None:
    """One cell with different availability cannot inherit common count wording."""
    record = _redraw_record()
    record["scenarios"][0]["central_containment"].update(
        available_attempts=2000, unavailable_attempts=560
    )
    numbers, table = _render(tmp_path, record)
    assert "{ir-redraw-reporting-central-coverage-available}{2000}" in numbers
    assert (
        "ir-redraw-reporting-central-coverage-available"
        in table.split(r"\midrule", 1)[1]
    )
    assert "Each percentage is followed by available/attempted outer panels." in table


def test_redraw_missing_history_oracle_is_unavailable_with_reason(
    tmp_path: Path,
) -> None:
    """Missing containment targets do not become zero coverage or hidden failures."""
    record = _redraw_record()
    rate = record["scenarios"][2]["central_containment"]
    rate.update(
        equal_history_rate=None,
        mcse=None,
        available_rate=None,
        unavailable_histories=[3],
        unavailable_reason="missing_declared_history_oracle",
    )
    numbers, table = _render(tmp_path, record)
    assert r"{ir-redraw-mechanism-central-coverage-pct}{\ppmissing}" in numbers
    assert r"{ir-redraw-mechanism-central-coverage-se-pct}{\ppmissing}" in numbers
    assert (
        r"{ir-redraw-mechanism-central-coverage-available-only-pct}{\ppmissing}"
        in numbers
    )
    assert "missing\\_declared\\_history\\_oracle" in numbers
    assert "{ir-redraw-mechanism-central-coverage-unavailable-histories}{3}" in numbers
    assert "ir-redraw-mechanism-central-coverage-unavailable-reason" in table
    assert "Containment with a missing history target is unavailable" in table
    assert "ir-redraw-mechanism-central-coverage-available" in table
    assert "{ir-redraw-mechanism-positive-q05-pct}{40}" in numbers


def test_redraw_missing_operating_history_leaves_signal_scale_unavailable(
    tmp_path: Path,
) -> None:
    """Missing economic histories preserve the saved unavailable scale decision."""
    record = _redraw_record()
    record["signal_scale"] = {
        "status": "unavailable",
        "reason": "missing_economic_history",
        "unavailable_histories": [4],
    }
    numbers, table = _render(tmp_path, record)
    assert r"{ir-redraw-signal-first-response-min}{\ppmissing}" in numbers
    assert r"{ir-redraw-signal-first-response-min-pct}{\ppmissing}" in numbers
    assert "missing\\_economic\\_history" in numbers
    assert "{ir-redraw-signal-unavailable-histories}{4}" in numbers
    assert "ir-redraw-signal-reason" in table


@pytest.mark.parametrize("duplicate", [False, True])
def test_redraw_publication_rejects_changed_control_identity(
    tmp_path: Path, *, duplicate: bool
) -> None:
    """The publisher cannot substitute a favorable coverage or scenario."""
    record = _redraw_record()
    if duplicate:
        record["scenarios"].append(record["scenarios"][0])
    else:
        record["scenarios"][0]["budget"] = 4
    with pytest.raises(ValueError, match="calibration"):
        _render(tmp_path, record)


def test_inference_publication_preserves_both_sizes_and_ineligibility(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Adverse coverage and rejection remain distinct from positive sign rates."""
    numbers, table = _render(
        tmp_path, _inference_record(validation_extension), inference=True
    )
    assert "{ir-inference-audit-n16-coverage-pct}{80}" in numbers
    assert "{ir-inference-n100-coverage-pct}{70}" in numbers
    assert "{ir-inference-n100-null-rejection-pct}{20}" in numbers
    assert "{ir-inference-n100-alternative-rejection-pct}{60}" in numbers
    assert "{ir-inference-n100-null-available}{2490}" in numbers
    assert "{ir-inference-n100-null-attempts}{2500}" in numbers
    assert r"{ir-inference-n100-corporate-lower-bound}{\ppmissing}" in numbers
    assert "No matching corporate sampling design is established." in numbers
    assert "{ir-inference-firms}" not in numbers
    assert "ir-inference-audit-n16-null-attempts" in table
    assert "ir-inference-audit-n16-null-available" in table
    body = table.split(r"\midrule", 1)[1]
    assert "-attempts" not in body
    assert "-available" not in body
    assert " & Avail." not in table
    assert "ir-inference-n100-null-unavailable-se" in table
    assert "unavailable bounds remain in the performance denominators" in table
    assert r"Nominal 95\% lower bounds" in table
    assert "performance under independent firm-history sampling" in table
    assert "Both panel sizes retain four dependent years" in table
    assert "ir-inference-n100-corporate-reason" not in table


def test_inference_unequal_availability_remains_visible_per_row(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Differing usable bounds remain explicit without changing saved rates."""
    record = _inference_record(validation_extension)
    record["iid_fixture"]["scenarios"][0]["usable_bounds"] = 2400
    numbers, table = _render(tmp_path, record, inference=True)
    assert "{ir-inference-n100-null-available}{2400}" in numbers
    body = table.split(r"\midrule", 1)[1]
    assert "ir-inference-n100-null-available" in body
    assert "ir-inference-n100-null-rejection-attempts" in body
    assert " & Avail." in table


def test_inference_unavailable_standard_error_is_missing_not_zero(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Nullable saved scale summaries stay unavailable and counts stay visible."""
    record = _inference_record(validation_extension)
    record["iid_fixture"]["scenarios"][0].update(
        average_candidate_se=None,
        empirical_dispersion=None,
        candidate_se_to_dispersion=None,
        unavailable_candidate_se=2500,
        usable_bounds=0,
    )
    numbers, table = _render(tmp_path, record, inference=True)
    assert r"{ir-inference-n100-mean-se}{\ppmissing}" in numbers
    assert r"{ir-inference-n100-se-ratio}{\ppmissing}" in numbers
    assert "{ir-inference-n100-null-available}{0}" in numbers
    assert "ir-inference-n100-null-unavailable-se" in table


@pytest.mark.parametrize("duplicate", [False, True])
def test_inference_publication_rejects_changed_fixture_identity(
    tmp_path: Path, validation_extension: dict[str, Any], *, duplicate: bool
) -> None:
    """Saved size and target identities control the public labels."""
    record = _inference_record(validation_extension)
    if duplicate:
        record["iid_fixture"]["scenarios"].append(record["iid_fixture"]["scenarios"][0])
    else:
        record["iid_fixture"]["firms"] = 99
    with pytest.raises(ValueError, match="iid audit publication"):
        _render(tmp_path, record, inference=True)

"""Directional rendering reads saved metrics and preserves target distinctions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pipeline.figures.paper6._publication import write_numbers
from pipeline.figures.paper6.information_recovery.directional import load_directional
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path


def saved_metrics(extension: dict[str, Any]) -> dict[str, Any]:
    """Supply a zero target with positive estimates and no correct-positive sign."""
    return {
        "schema_version": 1,
        **extension,
        "legacy_directional": {
            "cells": [
                [
                    {
                        "positive": 1079,
                        "total": 2048,
                        "frequency": 1079 / 2048,
                        "mcse": 0.011044379757565,
                        "undefined": 0,
                    }
                ]
            ],
            "reporting_only_positive_histories": 16,
        },
        "publication_scenarios": [
            {
                "publication_label": "B, observed",
                "target_label": "zero",
                "correct_sign_rate": None,
                "correct_sign_mcse": None,
                "correct_sign_denominator": None,
                "positive_rate": 0.5,
                "positive_mcse": 0.009,
                "positive_denominator": 2560,
                "rmse": 0.2,
                "undefined": 2,
                "total": 2560,
                "failed_premises": 160,
                "domain_departures": 1,
            }
        ],
    }


def test_saved_metrics_preserve_full_precision_and_zero_target(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """The reader neither recomputes signs nor invents success at a zero oracle."""
    source = tmp_path / "results.json"
    source.write_text(json.dumps(saved_metrics(validation_extension)))
    result = load_directional(source)
    assert result.legacy_directional.cells[0][0].frequency == pytest.approx(1079 / 2048)
    values = result.publication_values()
    assert values["validation-0-correct-pct"] is None
    expected = {
        "validation-0-positive-pct": 50,
        "validation-0-undefined": 2,
        "validation-0-total": 2560,
        "validation-0-positive-denominator": 2560,
        "validation-0-failed-premises": 160,
        "validation-0-domain-departures": 1,
    }
    assert {key: values[key] for key in expected} == expected
    assert "zero-alignment measurement control" in result.publication_table()
    assert "not test power" in result.details_table()


def test_publication_preserves_distinct_rate_denominators_and_failures(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Keep unresolved oracle strata out of the displayed correct-sign denominator."""
    record = saved_metrics(validation_extension)
    cell = record["publication_scenarios"][0]
    cell.update(
        correct_sign_rate=0.6,
        correct_sign_mcse=0.02,
        correct_sign_denominator=1280,
    )
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    result = load_directional(source)
    values = result.publication_values()
    expected = {
        "validation-0-correct-denominator": 1280,
        "validation-0-positive-denominator": 2560,
        "validation-0-failed-premises": 160,
        "validation-0-domain-departures": 1,
    }
    assert {key: values[key] for key in expected} == expected
    table = result.details_table()
    assert "ir-validation-0-correct-denominator" in table
    assert "ir-validation-0-positive-denominator" in table
    assert "ir-validation-0-failed-premises" in table
    assert "ir-validation-0-domain-departures" in table
    assert "resolved nonzero oracles" in table
    assert "failed-premise attempts" in table


def test_primary_table_has_fixed_five_rows_and_preserves_details(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Keep the main exhibit compact without discarding adverse diagnostics."""
    record = saved_metrics(validation_extension)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    result = load_directional(source)
    main = result.publication_table()
    details = result.details_table()
    assert r"Common reporting & $1\times$" in main
    assert r"Common reporting & $4\times$" in main
    assert r"Common reporting & $16\times$" in main
    assert r"Stationary reporting & $1\times$" in main
    assert r"Reporting only & $1\times$" in main
    assert "Correct" not in main
    assert "Absolute RMSE" in main
    assert r"\shortstack{Article-count\\multiplier}" in main
    assert r"\shortstack{Positive estimates\\(\%)}" in main
    assert "numerically resolved positive estimates" in main
    assert "arithmetic mean" in main
    assert "not a ratio of expected moments" in details
    assert "Exceptions" in main
    assert "No undefined estimates" not in main
    assert r"\label{tab:directional-validation-controls}" in main
    assert r"\label{tab:directional-validation-details}" in details
    assert "Correct" in details


def test_main_table_states_common_attempts_once_but_retains_unequal_counts(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Common accounting moves to the caption; unequal counts stay beside rates."""
    record = saved_metrics(validation_extension)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    common = load_directional(source).publication_table()
    body = common.split(r"\midrule", 1)[1]
    assert "positive-denominator" not in body
    assert "ir-validation-common-1-positive-denominator" in common
    assert "Exceptions" in common
    cell = next(
        cell
        for cell in record["scenarios"]
        if (cell["scenario_id"], cell["regime"], cell["budget"]) == ("A", "common", 4)
    )
    cell.update(total=1280, positive_denominator=1280)
    source.write_text(json.dumps(record))
    unequal = load_directional(source).publication_table()
    assert (
        "ir-validation-common-4-positive-denominator"
        in unequal.split(r"\midrule", 1)[1]
    )
    assert "Each percentage is followed by its attempted denominator." in unequal


def test_incomplete_saved_metric_contract_is_rejected(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """A source missing declared publication scenarios cannot inherit old labels."""
    record = saved_metrics(validation_extension)
    del record["publication_scenarios"]
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    with pytest.raises(ValidationError):
        load_directional(source)


def test_routine_counts_and_clean_status_remain_in_appendix(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Keep complete accounting available without lengthening the clean main caption."""
    record = saved_metrics(validation_extension)
    for cell in (*record["scenarios"], *record["publication_scenarios"]):
        cell.update(undefined=0, failed_premises=0, domain_departures=0)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    result = load_directional(source)
    main = result.publication_table()
    details = result.details_table()
    assert "Each row retains" not in main
    assert "No undefined estimates" not in main
    assert "Each row retains" in details
    assert "ir-validation-histories" in details
    assert "ir-validation-redraws" in details
    assert "No undefined estimates, failed premises or domain departures." in details
    body = details.split(r"\midrule", 1)[1]
    assert "positive-denominator" not in body
    assert "U/P/D" not in details
    assert "failed-premises}" not in body
    assert r"Appendix~\ref{app:directional-diagnostics}" in details


def test_named_bindings_use_saved_scenario_identity_and_adverse_metrics(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Keep semantic bindings stable when saved rows change order or perform poorly."""
    record = saved_metrics(validation_extension)
    common = record["scenarios"][0]
    common.update(correct_sign_rate=0.35, normalized_rmse=7.5)
    common["wrong_sign"]["rate"] = 0.65
    record["scenarios"].reverse()
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    result = load_directional(source)
    values = result.publication_values()
    expected = {
        "validation-histories": 16,
        "validation-redraws": 160,
        "validation-common-1-correct-pct": 35,
        "validation-common-1-wrong-pct": 65,
        "validation-common-1-normalized-rmse": 7.5,
        "validation-common-4-normalized-rmse": 9,
        "validation-common-16-normalized-rmse": 9,
        "validation-changing-16-correct-pct": 40,
        "validation-half-correct-pct": 40,
        "validation-double-correct-pct": 40,
        "validation-common-1-failed-premises": 160,
        "validation-common-1-domain-departures": 1,
    }
    assert {key: values[key] for key in expected} == pytest.approx(expected)
    assert result.legacy_directional.cells[0][0].frequency == pytest.approx(1079 / 2048)


def test_named_zero_control_and_corporate_inference_remain_unavailable(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Emit genuine null metrics as missing values with their saved reasons."""
    source = tmp_path / "results.json"
    source.write_text(json.dumps(saved_metrics(validation_extension)))
    values = load_directional(source).publication_values()
    for budget in (1, 4, 16):
        prefix = f"validation-stationary-{budget}"
        assert values[f"{prefix}-correct-pct"] is None
        assert values[f"{prefix}-normalized-rmse"] is None
        assert values[f"{prefix}-normalization-reason"] == "zero_oracle_scale"
    assert values["inference-corporate-eligible"] is False
    assert values["inference-corporate-status"] == "unavailable"
    assert values["inference-corporate-lower-bound"] is None
    assert values["inference-corporate-p-value"] is None
    catalog = tmp_path / "numbers.tex"
    write_numbers(catalog, values)
    emitted = catalog.read_text()
    assert r"{ir-validation-stationary-1-normalized-rmse}{\ppmissing}" in emitted
    assert r"{ir-inference-corporate-lower-bound}{\ppmissing}" in emitted
    assert "No matching corporate sampling design is established." in emitted


def test_inference_bindings_preserve_saved_coverage_and_rejection_separately(
    tmp_path: Path, validation_extension: dict[str, Any]
) -> None:
    """Preserve undercoverage and excessive null rejection without relabeling signs."""
    source = tmp_path / "results.json"
    source.write_text(json.dumps(saved_metrics(validation_extension)))
    values = load_directional(source).publication_values()
    expected = {
        "inference-replications": 2500,
        "inference-firms": 16,
        "inference-years": 4,
        "inference-z": 1.6448536269514715,
        "inference-null-rejection-pct": 20,
        "inference-null-rejection-se-pct": 0.8,
        "inference-coverage-pct": 80,
        "inference-coverage-se-pct": 0.8,
        "inference-alternative-rejection-pct": 60,
        "inference-alternative-rejection-se-pct": 0.8,
        "inference-alternative-target": 0.5,
        "inference-mean-se": 0.15,
        "inference-empirical-sd": 0.3,
        "inference-se-ratio": 0.5,
    }
    assert {key: values[key] for key in expected} == pytest.approx(expected)


@pytest.mark.parametrize("duplicate", [False, True])
def test_named_bindings_reject_missing_or_duplicate_saved_scenarios(
    tmp_path: Path, validation_extension: dict[str, Any], *, duplicate: bool
) -> None:
    """Reject ambiguous saved identities instead of silently changing a binding."""
    record = saved_metrics(validation_extension)
    if duplicate:
        record["scenarios"].append(record["scenarios"][0])
    else:
        record["scenarios"].pop(0)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    result = load_directional(source)
    with pytest.raises(ValueError, match="saved validation scenario"):
        result.publication_values()

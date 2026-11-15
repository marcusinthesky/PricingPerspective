"""Saved-result publication keeps primary, simulation and audit evidence separate."""

import json
import re
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from pipeline.figures.paper6 import corporate_analysis, information_audit
from pipeline.figures.paper6.information_recovery import render as simulation

_ROOT = Path(__file__).resolve().parents[7]
_EVIDENCE = _ROOT / "data/papers/paper6/review-artifacts"
_CAMPAIGN = _ROOT / "data/papers/paper6/information_recovery/campaign/summary.json"


def corporate_fixture() -> dict[str, Any]:
    """Emulate the new saved contract using unchanged historical evidence."""
    result = json.loads((_EVIDENCE / "t81/results.json").read_bytes())
    result["schema_version"] = 1
    result["inventory"] = json.loads((_EVIDENCE / "t81/inventory.json").read_bytes())
    frontier = json.loads((_EVIDENCE / "t82/frontier-results.json").read_bytes())
    for key, cell in result["cells"].items():
        low, high = cell["article_measurement_percentile95"]
        cell.update(
            article_sensitivity_q025=low,
            article_sensitivity_q05=cell["article_measurement_lower95"],
            article_sensitivity_q975=high,
            normalized_alignment=frontier["cells"][key]["normalized_alignment"],
            within_projection_fit=frontier["cells"][key]["normalized_alignment"] ** 2,
            annual_beta=[
                {
                    "label": numerator["label"],
                    "beta": numerator["contribution"] / denominator["contribution"],
                }
                for numerator, denominator in zip(
                    cell["annual_theta_contributions"],
                    cell["annual_second_moment_contributions"],
                    strict=True,
                )
            ],
            year_denominator={
                "shares": [
                    {"share": row["contribution"] / cell["second_moment"]}
                    for row in cell["annual_second_moment_contributions"]
                ]
            },
            inference={
                "status": "unavailable",
                "reason": "No calibrated corporate sampling model.",
            },
        )
    return result


def directional_fixture(extension: dict[str, Any]) -> dict[str, object]:
    """Supply saved historical frequencies and one explicit zero-target control."""
    cell = {
        "positive": 1079,
        "total": 2048,
        "frequency": 1079 / 2048,
        "mcse": 0.011,
        "undefined": 0,
    }
    return {
        "schema_version": 1,
        **extension,
        "legacy_directional": {
            "cells": [[cell] * 3] * 2,
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
                "failed_premises": 0,
                "domain_departures": 0,
            }
        ],
    }


def test_primary_publication_requires_only_saved_corporate_analysis(
    tmp_path: Path,
) -> None:
    """No T82, T83, campaign or context tree is available in this release."""
    source = tmp_path / "results.json"
    source.write_text(json.dumps(corporate_fixture()))
    project = tmp_path / "project"
    corporate_analysis.publish(source, project)
    generated = project / "src/generated"
    values = (generated / "corporate_analysis_numbers.tex").read_text()
    expected = {
        "primary-beta": 0.004768643278367574,
        "d2560-matched-beta": 0.005552226228293822,
        "d64-expanded-beta": 0.011319143868042641,
        "d64-matched-beta": 0.011975116963378142,
        "primary-low": 0.0017519782567947508,
    }
    for key, value in expected.items():
        assert rf"\ppDeclareValue{{ir-{key}}}{{{value:.17g}}}" in values
    table = (generated / "information_recovery_empirical_table.tex").read_text()
    assert r"\shortstack{5th\\percentile}" in table
    first_row = table.split(r"\midrule", 1)[1].split(r"\\", 1)[0]
    assert first_row.index("d2560-expanded-low") < first_row.index("d2560-expanded-q05")
    assert first_row.index("d2560-expanded-q05") < first_row.index(
        "d2560-expanded-high"
    )
    assert "Alignment" in table
    assert "Skill" not in table
    assert r"Section~\ref{sec:data}" in table
    assert "fixed 2018 reference profiles" in table
    assert "ir-primary-inference-reason" not in table
    assert "Coefficient inference:" not in table
    assert r"ir-primary-inference-lower95}{\ppmissing}" in values
    assert "No calibrated corporate sampling model." in values
    assert "heldout" not in values
    assert "observation" not in values
    assert not (generated / "information_recovery_numbers.tex").exists()


def test_dimension_table_compresses_only_common_success_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_extension: dict[str, Any],
) -> None:
    """Routine zero columns disappear, but saved failures reappear by dimension."""
    project = tmp_path / "project"
    summary = tmp_path / "summary.json"
    directional = tmp_path / "directional.json"
    directional.write_text(json.dumps(directional_fixture(validation_extension)))
    monkeypatch.setattr(
        simulation,
        "audit_campaign",
        lambda _path: {"status": "complete", "integrity": True},
    )
    monkeypatch.setattr(simulation, "render", Mock())
    campaign = simulation.Campaign.model_validate_json(_CAMPAIGN.read_bytes())
    for dimension in campaign.appendix:
        dimension.failures.clear()
        for cell in dimension.recovery:
            cell.undefined = 0
    summary.write_text(campaign.model_dump_json())
    simulation.publish(summary, project, directional)
    table_path = project / "src/generated/information_recovery_dimensions_table.tex"
    complete = table_path.read_text()
    header = complete.split(r"\toprule", 1)[1].split(r"\midrule", 1)[0]
    body = complete.split(r"\midrule", 1)[1]
    assert "Undefined" not in header
    assert "Status" not in header
    assert "undefined}" not in body
    assert "Complete" not in body
    configurations = sum(len(dimension.budgets) for dimension in campaign.appendix)
    assert f"All {configurations} configurations completed" in complete
    campaign.appendix[0].failures.append({"reason": "failed_scientific_premise"})
    campaign.appendix[0].recovery[0].undefined = 2
    summary.write_text(campaign.model_dump_json())
    simulation.publish(summary, project, directional)
    failed = table_path.read_text()
    header = failed.split(r"\toprule", 1)[1].split(r"\midrule", 1)[0]
    body = failed.split(r"\midrule", 1)[1]
    assert "Undefined" in header
    assert "Status" in header
    assert "undefined}" in body
    assert "Failed" in body
    assert "Complete" in body
    assert f"All {configurations} configurations completed" not in failed


def test_missing_inference_displays_an_explicit_unavailable_reason(
    tmp_path: Path,
) -> None:
    """A missing inferential result cannot be displayed as zero uncertainty."""
    record = corporate_fixture()
    for cell in record["cells"].values():
        del cell["inference"]
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    corporate_analysis.publish(source, tmp_path / "project")
    values = (
        tmp_path / "project/src/generated/corporate_analysis_numbers.tex"
    ).read_text()
    assert "ir-primary-inference-status}{unavailable}" in values
    assert "No calibrated coefficient inference was supplied." in values
    assert r"ir-primary-inference-lower95}{\ppmissing}" in values


def test_primary_publication_rejects_an_omitted_cell(tmp_path: Path) -> None:
    """The display cannot quietly retain a reduced favorable sample."""
    record = corporate_fixture()
    del record["cells"]["d64_matched"]
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="all four retained cells"):
        corporate_analysis.publish(source, tmp_path / "project")
    assert not (tmp_path / "project").exists()


def test_unavailable_inference_cannot_supply_zero_uncertainty(tmp_path: Path) -> None:
    """Reject a numeric bound when the saved inference status is unavailable."""
    record = corporate_fixture()
    record["cells"]["d2560_expanded"]["inference"]["lower95"] = 0.0
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unavailable inference cannot carry"):
        corporate_analysis.publish(source, tmp_path / "project")
    assert not (tmp_path / "project").exists()


def test_primary_table_uses_an_unfavorable_saved_coefficient(tmp_path: Path) -> None:
    """Display a saved negative coefficient without estimating or selecting it."""
    record = corporate_fixture()
    record["cells"]["d2560_expanded"]["beta"] = -0.125
    source = tmp_path / "results.json"
    source.write_text(json.dumps(record))
    corporate_analysis.publish(source, tmp_path / "project")
    values = (
        tmp_path / "project/src/generated/corporate_analysis_numbers.tex"
    ).read_text()
    assert r"\ppDeclareValue{ir-primary-beta}{-0.125}" in values
    assert r"\ppDeclareValue{ir-d2560-expanded-beta}{-0.125}" in values


def test_secondary_audit_retains_historical_negative_and_locality_results(
    tmp_path: Path,
) -> None:
    """T81 and campaign inputs are absent; T82/T83 alone reproduce the audit."""
    evidence = tmp_path / "evidence"
    for relative in (
        "t82/gmm-results.json",
        "t82/frontier-results.json",
        "t83/results.json",
    ):
        target = evidence / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_EVIDENCE / relative, target)
    information_audit.publish(evidence, tmp_path / "project")
    values = (
        tmp_path / "project/src/generated/information_audit_numbers.tex"
    ).read_text()
    assert "ir-observation-locality-count}{663}" in values
    assert "ir-observation-locality-defined}{784}" in values
    expected = -0.004839040645791659
    assert (
        rf"\ppDeclareValue{{ir-observation-heldout-2020}}{{{expected:.17g}}}" in values
    )
    assert "ir-observation-d64-matched-heldout-pct}{-2.079" in values
    table = (
        tmp_path / "project/src/generated/information_recovery_observation_table.tex"
    ).read_text()
    assert "Skill" in table
    assert "does not validate the financial coefficient" in table


def test_incomplete_campaign_cannot_reach_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Development previews cannot silently become numerical publication evidence."""
    monkeypatch.setattr(
        simulation,
        "audit_campaign",
        lambda _path: {"status": "running", "integrity": True},
    )
    with pytest.raises(ValueError, match="complete campaign"):
        simulation.publish(
            tmp_path / "summary.json", tmp_path / "project", tmp_path / "results.json"
        )
    assert not (tmp_path / "project").exists()


def test_simulation_publication_reads_saved_metrics_without_corporate_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_extension: dict[str, Any],
) -> None:
    """Numerical directional work is already saved; rendering projects its values."""
    metrics = tmp_path / "results.json"
    metrics.write_text(json.dumps(directional_fixture(validation_extension)))
    monkeypatch.setattr(
        simulation,
        "audit_campaign",
        lambda _path: {"status": "complete", "integrity": True},
    )
    figure = Mock()
    monkeypatch.setattr(simulation, "render", figure)
    project = tmp_path / "project"
    simulation.publish(_CAMPAIGN, project, metrics)
    generated = project / "src/generated"
    values = (generated / "information_recovery_numbers.tex").read_text()
    assert "ir-common-1-positive-count}{1079}" in values
    assert r"ir-validation-0-correct-pct}{\ppmissing}" in values
    assert "ir-validation-0-positive-pct}{50}" in values
    assert "ir-primary-beta" not in values
    assert not (generated / "information_recovery_empirical_table.tex").exists()
    assert (generated / "information_recovery_validation_details_table.tex").exists()
    figure.assert_called_once()


def test_publication_producers_have_exclusive_scalar_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_extension: dict[str, Any],
) -> None:
    """Loading all number catalogs cannot redefine a primary or audit binding."""
    project = tmp_path / "project"
    source = tmp_path / "results.json"
    source.write_text(json.dumps(corporate_fixture()))
    corporate_analysis.publish(source, project)
    information_audit.publish(_EVIDENCE, project)
    source.write_text(json.dumps(directional_fixture(validation_extension)))
    monkeypatch.setattr(
        simulation,
        "audit_campaign",
        lambda _path: {"status": "complete", "integrity": True},
    )
    monkeypatch.setattr(simulation, "render", Mock())
    simulation.publish(_CAMPAIGN, project, source)
    catalogs = list((project / "src/generated").glob("*numbers.tex"))
    keys = [
        key
        for path in catalogs
        for key in re.findall(r"\\ppDeclareValue\{([^}]+)\}", path.read_text())
    ]
    assert keys
    assert len(keys) == len(set(keys))

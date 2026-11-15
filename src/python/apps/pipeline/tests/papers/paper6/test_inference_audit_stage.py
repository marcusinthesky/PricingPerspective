"""Resumable additive iid calibration and its command boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hotelling.dynamics.artifacts import file_hash, read_arrays
from pipeline._kernels.paper6.inference_audit import (
    iid_fixture,
    iid_replications,
    pooled_candidate,
)
from pipeline._kernels.paper6.reference_alignment import analyze
from pipeline.cli.commands.p6 import inference_audit as commands
from pipeline.stages.papers.paper6 import inference_audit_extension as stage
from pipeline.stages.papers.paper6.artifacts import read, write
from pipeline.stages.papers.paper6.inference_audit import run_audit
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _inputs(tmp_path: Path) -> tuple[stage.InferenceAuditConfig, Path, Path]:
    config = stage.InferenceAuditConfig(
        firms=100, replications=4, pilot_replications=2, seed=620260100
    )
    design = tmp_path / "design.json"
    write(
        design,
        {
            "schema_version": 1,
            "frozen_before_new_results": True,
            **config.model_dump(),
            "targets": [0.0, 0.5],
            "years": 4,
        },
    )
    baseline = tmp_path / "baseline"
    run_audit(baseline, conditional_rows=[], fixture_replications=2)
    return config, design, baseline


def test_n100_candidate_matches_full_triples_and_joint_year_covariance() -> None:
    """The production-size contraction retains every target/reference role."""
    rng = np.random.default_rng(620260100)
    x, error = rng.normal(size=(2, 100))
    financial, approach = iid_fixture(x, error, rng.choice([-1, 1], size=100), beta=0.5)
    candidate = pooled_candidate(financial, approach)
    summary, arrays = analyze(financial, approach)
    assert candidate["beta"] == pytest.approx(summary["original_pooled_beta"])
    expected_se = np.std(arrays["pooled_beta_if"], ddof=1) / np.sqrt(100)
    assert candidate["candidate_se"] == pytest.approx(expected_se, rel=1e-12)


def test_pilot_spawn_production_and_resume_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed campaign preserves streams, completed outputs and N16 evidence."""
    config, design, baseline = _inputs(tmp_path)
    monkeypatch.setattr(stage, "entrypoint_code_identity", lambda _: "a" * 64)
    baseline_hashes = {
        name: file_hash(baseline / name)
        for name in ("inference-status.json", "fixture-replicates.npz")
    }
    output = tmp_path / "audit"
    pilot = stage.run_inference_audit(
        output, config, design=design, baseline=baseline, pilot_only=True
    )
    assert pilot["status"] == "pilot_complete"
    assert pilot["execution"]["pilot"]["serial_spawn_agreement"]
    assert pilot["execution"]["pilot"]["seed"] == config.seed + 1
    pilot_arrays = read_arrays(output / "pilot.npz")
    expected_pilot = iid_replications(
        replications=config.pilot_replications, firms=config.firms, seed=config.seed + 1
    )
    np.testing.assert_array_equal(
        pilot_arrays["estimates"], expected_pilot["estimates"]
    )
    result = stage.run_inference_audit(output, config, design=design, baseline=baseline)
    assert result["status"] == "complete"
    assert not result["empirical_inference"]["eligible"]
    assert result["baseline_input_hashes"] == baseline_hashes
    actual = read_arrays(output / "fixture-replicates.npz")
    expected = iid_replications(
        replications=config.replications, firms=config.firms, seed=config.seed
    )
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    assert not np.array_equal(
        pilot_arrays["estimates"], actual["estimates"][:, : config.pilot_replications]
    )

    def no_new_executor(**_kwargs: object) -> None:
        msg = "completed inference audit must not launch another task"
        raise AssertionError(msg)

    monkeypatch.setattr(stage, "ProcessPoolExecutor", no_new_executor)
    replay = stage.run_inference_audit(output, config, design=design, baseline=baseline)
    assert replay == result
    assert baseline_hashes == {
        name: file_hash(baseline / name) for name in baseline_hashes
    }
    (output / "fixture-replicates.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        stage.run_inference_audit(output, config, design=design, baseline=baseline)


def test_resume_rejects_changed_code_design_and_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject resumption under altered scientific inputs or source identity."""
    config, design, baseline = _inputs(tmp_path)
    monkeypatch.setattr(stage, "entrypoint_code_identity", lambda _: "a" * 64)
    output = tmp_path / "audit"
    stage.run_inference_audit(output, config, design=design, baseline=baseline)
    monkeypatch.setattr(stage, "entrypoint_code_identity", lambda _: "b" * 64)
    with pytest.raises(ValueError, match="incompatible code, design or inputs"):
        stage.run_inference_audit(output, config, design=design, baseline=baseline)
    monkeypatch.setattr(stage, "entrypoint_code_identity", lambda _: "a" * 64)
    design_record = read(design)
    design_record["annotation"] = "changed design evidence"
    write(design, design_record)
    with pytest.raises(ValueError, match="incompatible code, design or inputs"):
        stage.run_inference_audit(output, config, design=design, baseline=baseline)
    (baseline / "fixture-replicates.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="comparison checkpoint hash mismatch"):
        stage.run_inference_audit(output, config, design=design, baseline=baseline)


def test_inference_cli_reads_cohesive_config_and_passes_explicit_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI validates a config block and defers all scientific computation."""
    config, design, baseline = _inputs(tmp_path)
    params = tmp_path / "params.yaml"
    params.write_text("paper6:\n  inference_audit: {}\n", encoding="utf-8")
    received: dict[str, object] = {}

    def run(
        output: Path, current: stage.InferenceAuditConfig, **kwargs: object
    ) -> dict[str, str]:
        received.update(output=output, config=current, **kwargs)
        return {"status": "pilot_complete"}

    monkeypatch.setattr(stage, "run_inference_audit", run)
    output = tmp_path / "output"
    result = CliRunner().invoke(
        commands.commands,
        [
            "paper6-inference-audit",
            "--design",
            str(design),
            "--params-file",
            str(params),
            "--baseline",
            str(baseline),
            "--output",
            str(output),
            "--threads",
            "1",
            "--pilot-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received["config"] == stage.InferenceAuditConfig()
    assert received["output"] == output
    assert received["design"] == design
    assert received["baseline"] == baseline
    assert received["pilot_only"]
    received_config = received["config"]
    assert isinstance(received_config, stage.InferenceAuditConfig)
    assert config.firms == received_config.firms

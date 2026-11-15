"""Frozen input checks, durable undefined outcomes and honest recovery resumptions."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from hotelling.dynamics.artifacts import file_hash, value_hash
from pipeline._kernels.paper6.recovery import HistoryMeasurement, summarize_measurements
from pipeline.io.semantic_identity import standalone_entrypoint_code_identity
from pipeline.stages.papers.paper6.artifacts import write
from pipeline.stages.papers.paper6.recovery import campaign
from pipeline.stages.papers.paper6.recovery.artifacts import (
    COMMAND,
    FROZEN_SUPPORT_KEY,
    INVENTORY,
    SUPPORT,
    capture_provenance,
    load_measurement,
    load_month_counts,
    open_campaign,
    save_measurement,
    verify_provenance,
)
from pipeline.stages.papers.paper6.recovery.campaign import (
    _measure,
    _record_result,
    _run_pending,
)

if TYPE_CHECKING:
    from typing import Any

EXPANDED_ARTICLES = 43720
RESUME_HISTORIES = 3


def _measurement() -> HistoryMeasurement:
    estimates = np.full((2, 3, 2), 1.0)
    estimates[0, 0, 0] = np.nan
    return HistoryMeasurement(
        1.0, 1.0, np.ones(5), np.zeros(5), estimates, 1e-11, 0.01, 10.0, 0.9, 0
    )


def test_frozen_monthly_counts_reconcile_without_private_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exported data retain coverage and its immutable historical hash contract."""
    root = Path(__file__).resolve().parents[7]
    for relative in (SUPPORT, INVENTORY):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.repository_root",
        lambda: tmp_path,
    )
    counts, provenance = load_month_counts()
    assert counts.shape == (5, 12, 100)
    assert counts.sum() == EXPANDED_ARTICLES
    assert np.any(counts == 0)
    assert np.all(counts.sum(axis=1) > 0)
    assert provenance["articles"] == EXPANDED_ARTICLES
    assert provenance["hashes"] == {
        relative: file_hash(root / relative) for relative in (SUPPORT, INVENTORY)
    }
    assert not (tmp_path / ".context").exists()


def test_modified_support_fails_before_count_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different support file cannot silently become a coverage calibration."""
    inventory = tmp_path / INVENTORY
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps({"input_sha256": {FROZEN_SUPPORT_KEY: "frozen"}}))
    support = tmp_path / SUPPORT
    support.parent.mkdir(parents=True)
    support.write_bytes(b"changed")
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.repository_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError, match="frozen T81 inventory"):
        load_month_counts()


def test_undefined_draws_roundtrip_and_tampering_is_rejected(tmp_path: Path) -> None:
    """Finite checkpoint storage retains an explicit undefined-estimate mask."""
    path = tmp_path / "history.npz"
    result = _measurement()
    digest = save_measurement(path, result)
    restored = load_measurement(path, digest)
    np.testing.assert_array_equal(restored.estimates, result.estimates)
    with np.load(path, allow_pickle=False) as saved:
        assert all(np.isfinite(saved[name]).all() for name in saved.files)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        load_measurement(path, digest)


def test_failed_history_cannot_disappear_from_balanced_recovery() -> None:
    """One failed history leaves the overall recovery result undefined."""
    result = summarize_measurements([_measurement(), None])
    assert result["recovery"] is None
    assert result["diagnostics"]["failed_histories"] == 1
    assert result["smallest_recovered_budget"] == [None, None]


def test_undefined_population_oracle_becomes_explicit_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only article estimates may be undefined in a persisted successful history."""
    result = replace(_measurement(), beta0=np.nan)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.campaign.measure_history",
        lambda *_a, **_k: result,
    )
    with pytest.raises(ArithmeticError, match="undefined population oracle"):
        _measure(0, np.ones((5, 12, 100), dtype=np.int64), 2, None, 1)


def test_scientific_premise_failure_keeps_checkpoint_and_failure_record(
    tmp_path: Path,
) -> None:
    """Failed domain premises are recorded alongside their scientific arrays."""
    manifest: dict[str, Any] = {"completed": {}, "failures": []}
    result = replace(_measurement(), active_domain_violations=1)
    _record_result(tmp_path, manifest, "canonical/history-00.npz", result)
    assert manifest["failures"][0]["stage"] == "scientific-premises"
    restored = load_measurement(
        tmp_path / "canonical/history-00.npz",
        manifest["completed"]["canonical/history-00.npz"],
    )
    assert restored.active_domain_violations == 1
    assert (tmp_path / "manifest.json").is_file()


def test_archived_sources_are_evidence_not_numerical_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof/tooling edits leave compatible checkpoints reusable and auditable."""
    root = tmp_path / "checkout"
    root.mkdir()
    output = tmp_path / "saved"
    source = output / "sources/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("original")
    identities = {"module.py": file_hash(source)}
    provenance: dict[str, Any] = {
        "code_identity": "numerical-code",
        "files": identities,
        "source_hash": value_hash(identities),
        "inputs": {"hashes": {}},
        "dependencies": {},
    }
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.repository_root", lambda: root
    )
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.standalone_entrypoint_code_identity",
        lambda _: "numerical-code",
    )
    (root / "module.py").write_text("changed")
    verify_provenance(output, provenance, current=False)
    verify_provenance(output, provenance, current=True)
    (root / "module.py").unlink()
    verify_provenance(output, provenance, current=True)
    source.write_text("tampered archive")
    with pytest.raises(ValueError, match="source snapshot mismatch"):
        verify_provenance(output, provenance, current=True)


def test_new_provenance_uses_semflow_and_checks_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing numerical identity replaces live proof and environment checks."""
    root = tmp_path / "checkout"
    output = tmp_path / "saved"
    hashes = {}
    for name in (SUPPORT, INVENTORY):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen")
        hashes[name] = file_hash(path)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.repository_root", lambda: root
    )
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.standalone_entrypoint_code_identity",
        lambda _: "numerical-code",
    )
    provenance = capture_provenance(output, {"hashes": hashes})
    verify_provenance(output, provenance, current=True)
    (root / SUPPORT).write_bytes(b"changed input")
    verify_provenance(output, provenance, current=False)
    with pytest.raises(ValueError, match="frozen input mismatch"):
        verify_provenance(output, provenance, current=True)
    (output / "inputs" / Path(SUPPORT).name).write_bytes(b"tampered archive")
    with pytest.raises(ValueError, match="frozen input mismatch"):
        verify_provenance(output, provenance, current=False)


def test_legacy_input_snapshots_remain_auditable_without_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relocation preserves integrity auditing of saved historical input names."""
    root = tmp_path / "checkout"
    root.mkdir()
    output = tmp_path / "saved"
    hashes = {}
    for relative in (
        FROZEN_SUPPORT_KEY,
        ("evidence/inventory.json"),
    ):
        snapshot = output / "inputs" / Path(relative).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"frozen historical input")
        hashes[relative] = file_hash(snapshot)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.repository_root", lambda: root
    )
    provenance = {"inputs": {"hashes": hashes}}
    verify_provenance(output, provenance, current=False)
    assert not (root / ".context").exists()
    (output / "inputs/inventory.json").write_bytes(b"changed historical input")
    with pytest.raises(ValueError, match="frozen input mismatch"):
        verify_provenance(output, provenance, current=False)


@pytest.mark.integration
def test_fresh_standalone_provenance_preserves_historical_audit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh outputs record live Semflow identity; frozen audits never replace it."""
    root = tmp_path / "checkout"
    output = tmp_path / "saved"
    hashes = {}
    for name in (SUPPORT, INVENTORY):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen")
        hashes[name] = file_hash(path)
    monkeypatch.chdir(Path(__file__).resolve().parents[7])
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.repository_root", lambda: root
    )
    identity = standalone_entrypoint_code_identity(COMMAND)
    provenance = capture_provenance(output, {"hashes": hashes})
    assert provenance["code_identity"] == identity
    verify_provenance(output, provenance, current=True)

    historical = {**provenance, "code_identity": "historical-code"}
    historical_hash = value_hash(historical)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.standalone_entrypoint_code_identity",
        lambda _: pytest.fail("a frozen audit must not resolve live identity"),
    )
    verify_provenance(output, historical, current=False)
    assert value_hash(historical) == historical_hash


@pytest.mark.parametrize("identity", [None, "changed-numerical-code"])
def test_completed_campaign_cannot_bypass_numerical_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str | None
) -> None:
    """Even a complete legacy or incompatible run needs reviewed reuse."""
    _completed_campaign(tmp_path, identity)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.standalone_entrypoint_code_identity",
        lambda _: "numerical-code",
    )
    monkeypatch.setattr(campaign, "load_month_counts", lambda: (None, {}))
    with pytest.raises(ValueError, match="checkpoint code identity differs"):
        campaign.run_campaign(tmp_path)


def test_completed_compatible_campaign_returns_without_starting_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DVC rerun can reuse compatible completed results without recomputation."""
    _completed_campaign(tmp_path)
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.artifacts.standalone_entrypoint_code_identity",
        lambda _: "numerical-code",
    )
    monkeypatch.setattr(campaign, "load_month_counts", lambda: (None, {}))
    workers = Mock(side_effect=AssertionError("completed histories recomputed"))
    monkeypatch.setattr(campaign, "ProcessPoolExecutor", workers)
    assert campaign.run_campaign(tmp_path) == {"status": "complete"}
    workers.assert_not_called()


def _completed_campaign(output: Path, identity: str | None = "numerical-code") -> None:
    """Write a real summary and measurement for terminal-run compatibility checks."""
    summary = output / "summary.json"
    write(summary, {"status": "complete"})
    provenance: dict[str, Any] = {
        "code_identity": "numerical-code",
        "inputs": {"hashes": {}},
    }
    if identity is None:
        del provenance["code_identity"]
    else:
        provenance["code_identity"] = identity
    measurement = output / "history.npz"
    digest = save_measurement(measurement, _measurement())
    write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "design": campaign.DESIGN,
            "provenance": provenance,
            "provenance_hash": value_hash(provenance),
            "completed": {measurement.name: digest},
            "summary_hash": file_hash(summary),
            "status": "complete",
        },
    )


def test_unrecognized_nonempty_partition_is_never_overwritten(tmp_path: Path) -> None:
    """Only a declared recovery manifest authorizes reopening an existing output."""
    (tmp_path / "historical.json").write_text("{}")
    with pytest.raises(FileExistsError, match="resumable manifest"):
        open_campaign(tmp_path)


def test_interrupted_campaign_resumes_only_missing_histories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable history is reused across partial and completed resumptions."""
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.campaign.DIMENSIONS", ()
    )
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.recovery.campaign.HISTORIES", RESUME_HISTORIES
    )
    result = _measurement()
    manifest: dict[str, Any] = {"completed": {}, "failures": []}
    _record_result(tmp_path, manifest, "canonical/history-00.npz", result)
    first_digest = manifest["completed"]["canonical/history-00.npz"]
    future: Future[HistoryMeasurement] = Future()
    future.set_result(result)
    executor = Mock(spec=ProcessPoolExecutor, submit=Mock(return_value=future))
    counts = np.ones((5, 12, 100), dtype=np.int64)
    _run_pending(tmp_path, manifest, executor, counts, 2, 1)
    assert [call.args[1] for call in executor.submit.call_args_list] == [1, 2]
    assert manifest["completed"]["canonical/history-00.npz"] == first_digest
    executor.reset_mock()
    _run_pending(tmp_path, manifest, executor, counts, 2, 1)
    executor.submit.assert_not_called()
    assert len(manifest["completed"]) == RESUME_HISTORIES

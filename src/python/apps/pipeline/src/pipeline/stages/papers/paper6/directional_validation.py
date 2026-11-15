"""Bounded, resumable known-target analysis consuming immutable legacy evidence."""

from __future__ import annotations

import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hotelling.artifacts import reject_symlinks
from hotelling.dynamics.artifacts import file_hash, repository_root, value_hash
from hotelling.dynamics.resources import (
    ExecutionSettings,
    choose_resources,
    detect_limits,
)
from threadpoolctl import threadpool_limits

from pipeline._kernels.paper6.directional_validation import (
    HISTORIES,
    SCENARIOS,
    DirectionalValidationConfig,
    failed_scenario,
    legacy_directional,
    legacy_rows,
    measure_validation_history,
    summarize_rows,
)
from pipeline._kernels.paper6.recovery import REDRAWS
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.jax_cache import configure_persistent_cache
from pipeline.precision import own_float64
from pipeline.stages.papers.paper6.artifacts import read, write
from pipeline.stages.papers.paper6.inference_audit import run_audit
from pipeline.stages.papers.paper6.recovery.artifacts import (
    load_measurement,
    load_month_counts,
)
from pipeline.stages.papers.paper6.recovery.campaign import audit_campaign

COMMAND = "paper6-directional-validation"

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pipeline._kernels.paper6.recovery import HistoryMeasurement


def _write_frame(path: Path, records: list[dict[str, Any]]) -> str:
    """Commit a scalar-record table atomically, retaining nullable quantities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    os.close(descriptor)
    try:
        pd.DataFrame.from_records(records).to_parquet(name, index=False)
        with Path(name).open("rb") as stream:
            os.fsync(stream.fileno())
        Path(name).replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)
    return file_hash(path)


def _read_frame(path: Path, digest: str) -> list[dict[str, Any]]:
    if file_hash(path) != digest:
        msg = f"directional checkpoint hash mismatch: {path}"
        raise ValueError(msg)
    frame = pd.read_parquet(path)
    return [
        {
            str(key): None if isinstance(value, float) and np.isnan(value) else value
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _measure(
    index: int,
    counts: NDArray[np.int64],
    redraws: int,
    *,
    pilot: bool,
    resolution: float,
    threads: int,
) -> list[dict[str, Any]]:
    own_float64()
    configure_persistent_cache()
    with threadpool_limits(limits=threads):
        return measure_validation_history(
            index, counts, redraws=redraws, pilot=pilot, resolution=resolution
        )


def _failures(index: int, redraws: int, error: Exception) -> list[dict[str, Any]]:
    """Retain all attempted panels when a history cannot execute numerically."""
    return [
        row
        for scenario in SCENARIOS
        for row in failed_scenario(scenario, index, redraws, error)
    ]


def _validate_design(
    record: dict[str, Any], config: DirectionalValidationConfig
) -> None:
    expected = {
        "schema_version": 1,
        "histories": HISTORIES,
        "redraws": config.redraws,
        "pilot_redraws": config.pilot_redraws,
        "seed": 20261024,
        "budgets": [1, 4, 16],
        "reporting_amplitudes": [0, 0.001, 0.0025, 0.005, 0.01],
        "shock_multipliers": [0.5, 2],
        "numerical_resolution": config.numerical_resolution,
    }
    if not record.get("frozen_before_new_results") or any(
        record.get(key) != value for key, value in expected.items()
    ):
        msg = "analysis design differs from the fixed conditional validation design"
        raise ValueError(msg)


def _open(output: Path) -> Path:
    reject_symlinks(output.absolute())
    output = output.resolve()
    allowed = repository_root() / "data/papers/paper6/directional_validation"
    temporary = Path(tempfile.gettempdir()).resolve()
    if output == temporary or not (
        output.is_relative_to(allowed) or output.is_relative_to(temporary)
    ):
        msg = "directional validation output requires its governed namespace or /tmp"
        raise ValueError(msg)
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "provenance.manifest.json").is_file()
    ):
        msg = "nonempty directional output has no resumable manifest"
        raise FileExistsError(msg)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _check_membership(rows: list[dict[str, Any]], redraws: int) -> None:
    expected = {
        (scenario.scenario_id, regime, budget, history, replicate)
        for scenario in SCENARIOS
        for regime in scenario.regimes
        for budget in scenario.budgets
        for history in range(HISTORIES)
        for replicate in range(redraws)
    }
    actual = [
        (
            row["scenario_id"],
            row["regime"],
            row["budget"],
            row["history"],
            row["replicate"],
        )
        for row in rows
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        msg = "directional scenario membership is missing, duplicated or undeclared"
        raise ValueError(msg)


def _prefix(
    rows: list[dict[str, Any]], measurement: HistoryMeasurement
) -> float | None:
    """Require A to reproduce old scientific estimates before using its extension."""
    selected = [
        row for row in rows if row["scenario_id"] == "A" and row["replicate"] < REDRAWS
    ]
    if any(not row["execution_succeeded"] for row in selected):
        return None
    errors: list[float] = []
    for row in selected:
        regime = ("common", "changing").index(row["regime"])
        budget = (1, 4, 16).index(row["budget"])
        old = measurement.estimates[regime, budget, row["replicate"]]
        new = np.nan if row["beta"] is None else row["beta"]
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12, equal_nan=True)
        if np.isfinite(new) and np.isfinite(old):
            errors.append(abs(new - old))
    return max(errors, default=0.0)


def _publication(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the predeclared compact exhibit before rendering."""
    selected = []
    for row in rows:
        scenario, target = row["scenario_id"], row["target"]
        include = (
            (scenario == "A" and target == "beta0_h")
            or (scenario == "B" and target == "beta_lambda_h")
            or (
                scenario == "C"
                and row["regime"] == "reporting_0.005"
                and row["budget"] == 1
            )
            or (scenario in ("D_half", "D_double") and target == "beta0_h")
        )
        if include:
            selected.append(
                {
                    **row,
                    "publication_label": {
                        "A": "Mechanism",
                        "B": "Stationary law",
                        "C": "Reporting only",
                        "D_half": "Half shock",
                        "D_double": "Double shock",
                    }[scenario]
                    + (
                        f" {row['regime']} {row['budget']}"
                        if scenario in ("A", "B")
                        else ""
                    ),
                    "target_label": "Operating" if target == "beta0_h" else "Reported",
                }
            )
    return selected


def _legacy(
    legacy_campaign: Path,
) -> tuple[list[HistoryMeasurement], dict[str, object]]:
    """Admit only the entire hash-verified historical canonical design."""
    audit = audit_campaign(legacy_campaign)
    legacy_manifest = read(legacy_campaign / "manifest.json")
    names = [f"canonical/history-{index:02}.npz" for index in range(HISTORIES)]
    actual = sorted(
        name for name in legacy_manifest["completed"] if name.startswith("canonical/")
    )
    if actual != names or any(not legacy_manifest["completed"][name] for name in names):
        msg = "legacy analysis requires sixteen complete canonical checkpoints"
        raise ValueError(msg)
    measurements = [
        load_measurement(legacy_campaign / name, legacy_manifest["completed"][name])
        for name in names
    ]
    return measurements, audit


def _manifest(output: Path, design: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Verify every durable result before resuming scientifically identical work."""
    path = output / "provenance.manifest.json"
    manifest: dict[str, Any]
    if path.exists():
        manifest = read(path)
        if manifest.get("identity") != identity or manifest.get(
            "identity_hash"
        ) != value_hash(identity):
            msg = "directional manifest has incompatible code, design or inputs"
            raise ValueError(msg)
        for key, digest in manifest["completed"].items():
            _read_frame(output / key, digest)
        for key, digest in manifest.get("outputs", {}).items():
            if file_hash(output / key) != digest:
                msg = f"directional output hash mismatch: {key}"
                raise ValueError(msg)
    else:
        shutil.copyfile(design, output / "design.json")
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "identity_hash": value_hash(identity),
            "status": "running",
            "completed": {},
            "outputs": {"design.json": file_hash(output / "design.json")},
            "execution": {},
            "failures": [],
            "stream_policy": {
                "A": "[20261024,history,replicate,role,regime?]; historical prefix",
                "B_C_D": "scenario keys appended",
                "pilot": "99 appended; separate from every production stream",
            },
        }
        write(path, manifest)
    return manifest


def _save_legacy(
    output: Path,
    manifest: dict[str, Any],
    measurements: list[HistoryMeasurement],
    audit: dict[str, object],
) -> dict[str, Any]:
    """Extract cheap evidence first, without running or relabeling old simulation."""
    original_rows = legacy_rows(measurements)
    legacy_summary = summarize_rows(original_rows, redraws=REDRAWS)
    legacy_summary.update(
        {
            "audit": audit,
            "historical_128": True,
            "moment_unavailability_reason": "Replicate q/d were not saved; "
            "do not reconstruct them from rounded ratios.",
        }
    )
    manifest["outputs"]["legacy-replicates.parquet"] = _write_frame(
        output / "legacy-replicates.parquet", original_rows
    )
    write(
        output / "legacy-summary.json",
        {**legacy_summary, "legacy_directional": legacy_directional(measurements)},
    )
    manifest["outputs"]["legacy-summary.json"] = file_hash(
        output / "legacy-summary.json"
    )
    write(output / "provenance.manifest.json", manifest)
    return legacy_summary


def _pilot(
    output: Path,
    manifest: dict[str, Any],
    executor: ProcessPoolExecutor,
    counts: NDArray[np.int64],
    config: DirectionalValidationConfig,
    threads: int,
) -> None:
    """Measure runtime and serial/spawn agreement on separate pilot streams."""
    pilot_started = perf_counter()
    serial = _measure(
        0,
        counts,
        config.pilot_redraws,
        pilot=True,
        resolution=config.numerical_resolution,
        threads=threads,
    )
    serial_seconds = perf_counter() - pilot_started
    pilot_started = perf_counter()
    parallel = executor.submit(
        _measure,
        0,
        counts,
        config.pilot_redraws,
        pilot=True,
        resolution=config.numerical_resolution,
        threads=threads,
    ).result()
    if value_hash(serial) != value_hash(parallel):
        for left, right in zip(serial, parallel, strict=True):
            for key in ("beta", "q", "d", "oracle_beta", "candidate_se"):
                if left[key] is None or right[key] is None:
                    if left[key] != right[key]:
                        msg = "serial and spawned pilot defined statuses disagree"
                        raise ArithmeticError(msg)
                else:
                    np.testing.assert_allclose(
                        left[key], right[key], rtol=1e-12, atol=1e-12
                    )
    manifest["execution"]["pilot"] = {
        "history": 0,
        "redraws": config.pilot_redraws,
        "serial_seconds": serial_seconds,
        "spawn_seconds": perf_counter() - pilot_started,
        "serial_parallel_agreement": True,
        "production_redraws_per_history": config.redraws,
        "purpose": "Resource and integrity only; fixed production budget.",
    }
    manifest["outputs"]["pilot-replicates.parquet"] = _write_frame(
        output / "pilot-replicates.parquet", parallel
    )
    write(output / "provenance.manifest.json", manifest)


def _pending(
    output: Path,
    manifest: dict[str, Any],
    executor: ProcessPoolExecutor,
    counts: NDArray[np.int64],
    config: DirectionalValidationConfig,
    workers: int,
    threads: int,
    measurements: list[HistoryMeasurement],
) -> None:
    """Bound in-flight histories and durably retain every attempt after failure."""
    pending = [
        index
        for index in range(HISTORIES)
        if f"checkpoints/history-{index:02}.parquet" not in manifest["completed"]
    ]
    for offset in range(0, len(pending), workers):
        batch = pending[offset : offset + workers]
        futures = [
            executor.submit(
                _measure,
                index,
                counts,
                config.redraws,
                pilot=False,
                resolution=config.numerical_resolution,
                threads=threads,
            )
            for index in batch
        ]
        for index, future in zip(batch, futures, strict=True):
            try:
                rows = future.result()
                prefix = _prefix(rows, measurements[index])
            except (
                ValueError,
                ArithmeticError,
                TimeoutError,
                RuntimeError,
                AssertionError,
            ) as error:
                rows = _failures(index, config.redraws, error)
                prefix = None
                manifest["failures"].append(
                    {
                        "history": index,
                        "stage": "execution_or_prefix",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            key = f"checkpoints/history-{index:02}.parquet"
            manifest["completed"][key] = _write_frame(output / key, rows)
            manifest["execution"].setdefault("A_prefix_maximum_absolute_error", {})[
                str(index)
            ] = prefix
            write(output / "provenance.manifest.json", manifest)


def run_validation(
    output: Path,
    *,
    legacy_campaign: Path,
    design: Path,
    config: DirectionalValidationConfig | None = None,
    workers: str = "auto",
    threads: int = 1,
    pilot_only: bool = False,
) -> dict[str, Any]:
    """Derive legacy diagnostics, then run only the locked bounded control design."""
    own_float64()
    configure_persistent_cache()
    config = config or DirectionalValidationConfig()
    design_record = read(design)
    _validate_design(design_record, config)
    started = perf_counter()
    output = _open(output)
    measurements, audit = _legacy(legacy_campaign)
    counts, inputs = load_month_counts()
    identity = {
        "schema_version": 1,
        "code_identity": entrypoint_code_identity(COMMAND),
        "design_sha256": file_hash(design),
        "config": config.model_dump(),
        "inputs": inputs,
        "legacy_manifest_sha256": file_hash(legacy_campaign / "manifest.json"),
        "legacy_summary_sha256": file_hash(legacy_campaign / "summary.json"),
    }
    manifest = _manifest(output, design, identity)
    if manifest["status"] in ("complete", "scientific_failures", "execution_failed"):
        return read(output / "results.json")
    legacy_summary = _save_legacy(output, manifest, measurements, audit)

    requested = min(3, detect_limits().cpus) if workers == "auto" else int(workers)
    choice = choose_resources(
        ExecutionSettings(workers=requested, threads=threads),
        deadline=perf_counter() + 3600,
        run_count=HISTORIES,
    )
    manifest["execution"]["resources"] = choice.to_dict()
    with ProcessPoolExecutor(
        max_workers=choice.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        if "pilot" not in manifest["execution"]:
            _pilot(output, manifest, executor, counts, config, choice.threads)
        if pilot_only:
            return {
                "status": "pilot_complete",
                "execution": manifest["execution"],
                "legacy": legacy_summary,
                "failures": manifest["failures"],
            }
        _pending(
            output,
            manifest,
            executor,
            counts,
            config,
            choice.workers,
            choice.threads,
            measurements,
        )
    rows = [
        row
        for key, digest in sorted(manifest["completed"].items())
        for row in _read_frame(output / key, digest)
    ]
    _check_membership(rows, config.redraws)
    summarized = summarize_rows(rows, redraws=config.redraws)
    manifest["outputs"]["replicates.parquet"] = _write_frame(
        output / "replicates.parquet", rows
    )
    inference = run_audit(
        output / "inference",
        conditional_rows=rows,
        fixture_replications=design_record["iid_replications"],
    )
    failures = manifest["failures"]
    for summary in summarized["aggregate"]:
        if summary["target"] != "beta_lambda_h":
            continue
        if summary["failed_premises"]:
            failures.append(
                {
                    "scenario_id": summary["scenario_id"],
                    "regime": summary["regime"],
                    "budget": summary["budget"],
                    "stage": "scientific_premises",
                    "attempts": summary["failed_premises"],
                }
            )
        if summary["failed_execution"]:
            failures.append(
                {
                    "scenario_id": summary["scenario_id"],
                    "regime": summary["regime"],
                    "budget": summary["budget"],
                    "stage": "execution_failed",
                    "attempts": summary["failed_execution"],
                }
            )
    execution_failed = any(item["failed_execution"] for item in summarized["aggregate"])
    result = {
        "schema_version": 1,
        "status": "execution_failed"
        if execution_failed
        else "scientific_failures"
        if failures
        else "complete",
        "sampling_scope": "Fixed economy and history; article selection sensitivity "
        "and known reported-law measurement.",
        "histories": HISTORIES,
        "redraws": config.redraws,
        "legacy": legacy_summary,
        "legacy_directional": legacy_directional(measurements),
        "scenarios": summarized["aggregate"],
        "per_history": summarized["per_history"],
        "paired_budget_contrasts": summarized["paired_budget_contrasts"],
        "locality": summarized["locality"],
        "publication_scenarios": _publication(summarized["aggregate"]),
        "inference_status": inference,
        "execution": manifest["execution"],
        "failures": failures,
        "legacy_magnitude_convention": "Normalized RMSE + 2 MCSE <= .25; "
        "unavailable for zero oracle scale.",
    }
    manifest["execution"]["elapsed_seconds"] = perf_counter() - started
    for name in ("results.json", "summary.json"):
        write(output / name, result)
        manifest["outputs"][name] = file_hash(output / name)
    for name in ("inference/fixture-replicates.npz", "inference/inference-status.json"):
        manifest["outputs"][name] = file_hash(output / name)
    manifest["status"] = result["status"]
    write(output / "provenance.manifest.json", manifest)
    return result

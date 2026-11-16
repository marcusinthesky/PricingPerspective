"""One resumable parent-owned production study with a reused bounded executor."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from hotelling.dynamics.artifacts import file_hash, value_hash
from hotelling.dynamics.recovery import DIMENSIONS, REPORTING_AMPLITUDES
from hotelling.dynamics.resources import (
    ExecutionSettings,
    choose_resources,
    detect_limits,
)
from hotelling.orthant import SOLVE_TARGET
from threadpoolctl import threadpool_limits

from pipeline._kernels.paper6.recovery import (
    BUDGETS,
    REDRAWS,
    SEED,
    measure_history,
    summarize_measurements,
)
from pipeline.jax_cache import configure_persistent_cache
from pipeline.precision import own_float64
from pipeline.stages.papers.paper6.artifacts import read, write
from pipeline.stages.papers.paper6.recovery.artifacts import (
    capture_provenance,
    checkpoint,
    load_measurement,
    load_month_counts,
    open_campaign,
    save_measurement,
    verify_provenance,
)

PILOT_REDRAWS = 4
HISTORIES = 16
AFFINE_LIMIT = 0.01
DESIGN = {
    "seed": SEED,
    "redraws": REDRAWS,
    "histories": HISTORIES,
    "budgets": list(BUDGETS),
    "dimensions": list(DIMENSIONS),
}

if TYPE_CHECKING:
    from pipeline._kernels.paper6.recovery import HistoryMeasurement


def _measure(
    index: int,
    counts: np.ndarray[Any, np.dtype[np.int64]],
    redraws: int,
    dimension: int | None,
    threads: int,
) -> HistoryMeasurement:
    """Prepare this process before numerical work, then cap native task threads."""
    own_float64()
    configure_persistent_cache()
    with threadpool_limits(limits=threads):
        result = measure_history(index, counts, redraws=redraws, dimension=dimension)
    if any(
        not np.isfinite(getattr(result, field.name)).all()
        for field in fields(result)
        if field.name != "estimates"
    ):
        msg = "undefined population oracle or nonfinite scientific diagnostic"
        raise ArithmeticError(msg)
    return result


def _same(first: HistoryMeasurement, second: HistoryMeasurement) -> bool:
    return all(
        np.allclose(
            getattr(first, field.name),
            getattr(second, field.name),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        for field in fields(first)
    )


def _key(dimension: int | None, index: int) -> str:
    prefix = "canonical" if dimension is None else f"appendix/dimension-{dimension}"
    return f"{prefix}/history-{index:02}.npz"


def _pilot(
    output: Path,
    manifest: dict[str, Any],
    executor: ProcessPoolExecutor,
    counts: np.ndarray[Any, np.dtype[np.int64]],
    threads: int,
) -> None:
    """Measure fixed-pilot throughput and direct/spawn scientific agreement."""
    started = perf_counter()
    serial = _measure(0, counts, PILOT_REDRAWS, None, threads)
    serial_seconds = perf_counter() - started
    started = perf_counter()
    parallel = executor.submit(
        _measure, 0, counts, PILOT_REDRAWS, None, threads
    ).result()
    elapsed = perf_counter() - started
    agreement = _same(serial, parallel)
    manifest["execution"]["pilot"] = {
        "history": 0,
        "redraws": PILOT_REDRAWS,
        "seconds": elapsed,
        "serial_seconds": serial_seconds,
        "panels_per_second": PILOT_REDRAWS / elapsed,
    }
    manifest["execution"]["serial_parallel_agreement"] = agreement
    manifest["pilot_hash"] = save_measurement(output / "pilot.npz", parallel)
    checkpoint(output, manifest)
    if not agreement:
        msg = "serial and spawned pilot scientific statistics disagree at 1e-12"
        raise ArithmeticError(msg)


def _record_result(
    output: Path,
    manifest: dict[str, Any],
    key: str,
    result: HistoryMeasurement,
) -> None:
    manifest["completed"][key] = save_measurement(output / key, result)
    if (
        result.active_domain_violations
        or result.minimum_opportunity <= 0
        or result.minimum_baseline_price <= 0
        or result.maximum_solver_bound > SOLVE_TARGET
    ):
        manifest["failures"].append(
            {
                "cell": key.rsplit("/", 1)[0],
                "history": int(Path(key).stem[-2:]),
                "stage": "scientific-premises",
                "error": ("opportunity, price, solver or active-domain premise failed"),
            }
        )
    checkpoint(output, manifest)


def _run_pending(
    output: Path,
    manifest: dict[str, Any],
    executor: ProcessPoolExecutor,
    counts: np.ndarray[Any, np.dtype[np.int64]],
    workers: int,
    threads: int,
) -> None:
    """Bound in-flight work and checkpoint every completed or failed history."""
    jobs = [
        (dimension, index)
        for dimension in (None, *DIMENSIONS)
        for index in range(HISTORIES)
        if _key(dimension, index) not in manifest["completed"]
    ]
    for offset in range(0, len(jobs), workers):
        batch = jobs[offset : offset + workers]
        futures = [
            executor.submit(_measure, index, counts, REDRAWS, dimension, threads)
            for dimension, index in batch
        ]
        for (dimension, index), future in zip(batch, futures, strict=True):
            key = _key(dimension, index)
            try:
                result = future.result()
            except (ValueError, ArithmeticError, TimeoutError, RuntimeError) as error:
                manifest["failures"].append(
                    {
                        "cell": key.rsplit("/", 1)[0],
                        "history": index,
                        "stage": "measurement",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                manifest["completed"][key] = None
                checkpoint(output, manifest)
            else:
                _record_result(output, manifest, key, result)


def _cell(
    output: Path, manifest: dict[str, Any], dimension: int | None
) -> dict[str, Any]:
    results = [
        load_measurement(output / key, manifest["completed"][key])
        if manifest["completed"].get(key)
        else None
        for key in (_key(dimension, index) for index in range(HISTORIES))
    ]
    return summarize_measurements(results, appendix=dimension is not None)


def _summary(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = _cell(output, manifest, None)
    departure = summary["finite_affine_departure"]
    if departure is not None and departure >= AFFINE_LIMIT:
        manifest["failures"].append(
            {
                "cell": "canonical",
                "history": "all",
                "stage": "affine-oracle",
                "error": f"normalized finite/affine departure {departure} exceeds 0.01",
            }
        )
    appendix = [
        {
            "dimension": dimension,
            "budgets": [1, 16],
            **_cell(output, manifest, dimension),
            "failures": [
                item
                for item in manifest["failures"]
                if item["cell"] == f"appendix/dimension-{dimension}"
            ],
        }
        for dimension in DIMENSIONS
    ]
    return {
        "schema_version": 1,
        "status": "failed" if manifest["failures"] else "complete",
        "histories": HISTORIES,
        "redraws": REDRAWS,
        "budgets": list(BUDGETS),
        "regimes": ["common", "changing"],
        "reporting_amplitudes": list(REPORTING_AMPLITUDES),
        **summary,
        "appendix": appendix,
        "failures": manifest["failures"],
        "execution": manifest["execution"],
    }


def run_campaign(
    output: Path, *, workers: str = "auto", threads: int = 1
) -> dict[str, Any]:
    """Run the fixed design, or resume matching Semflow code and frozen inputs."""
    started = perf_counter()
    output = open_campaign(output)
    counts, inputs = load_month_counts()
    path = output / "manifest.json"
    manifest: dict[str, Any]
    if path.exists():
        manifest = read(path)
        _validate_manifest(output, manifest)
        verify_provenance(output, manifest["provenance"], current=True)
        for name, digest in manifest["completed"].items():
            if digest:
                load_measurement(output / name, digest)
        if manifest["status"] in ("complete", "failed"):
            return read(output / "summary.json")
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "design": DESIGN,
            "completed": {},
            "failures": [],
            "execution": {},
            "provenance": capture_provenance(output, inputs),
        }
        manifest["provenance_hash"] = value_hash(manifest["provenance"])
        checkpoint(output, manifest)
    requested = min(3, detect_limits().cpus) if workers == "auto" else int(workers)
    choice = choose_resources(
        ExecutionSettings(workers=requested, threads=threads),
        deadline=perf_counter() + 3600,
        run_count=HISTORIES * (1 + len(DIMENSIONS)),
    )
    manifest["execution"]["resources"] = choice.to_dict()
    checkpoint(output, manifest)
    with ProcessPoolExecutor(
        max_workers=choice.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        if "pilot" not in manifest["execution"]:
            _pilot(output, manifest, executor, counts, choice.threads)
        _run_pending(output, manifest, executor, counts, choice.workers, choice.threads)
    verify_provenance(output, manifest["provenance"], current=True)
    manifest["execution"]["elapsed_seconds"] = (
        manifest["execution"].get("elapsed_seconds", 0) + perf_counter() - started
    )
    summary = _summary(output, manifest)
    write(output / "summary.json", summary)
    manifest["summary_hash"] = file_hash(output / "summary.json")
    manifest["status"] = summary["status"]
    checkpoint(output, manifest)
    return summary


def audit_campaign(output: Path) -> dict[str, object]:
    """Read-only integrity audit remains available after source retirement."""
    manifest = read(output / "manifest.json")
    _validate_manifest(output, manifest)
    verify_provenance(output, manifest["provenance"], current=False)
    for name, digest in manifest["completed"].items():
        if digest:
            load_measurement(output / name, digest)
    return {
        "status": manifest["status"],
        "integrity": True,
        "completed": len(manifest["completed"]),
        "failures": manifest["failures"],
    }


def _validate_manifest(output: Path, manifest: dict[str, Any]) -> None:
    if manifest["schema_version"] != 1 or manifest["design"] != DESIGN:
        msg = "recovery manifest differs from the fixed version-one design"
        raise ValueError(msg)
    if value_hash(manifest["provenance"]) != manifest["provenance_hash"]:
        msg = "recovery provenance metadata hash mismatch"
        raise ValueError(msg)
    if "pilot_hash" in manifest:
        load_measurement(output / "pilot.npz", manifest["pilot_hash"])
    if (
        "summary_hash" in manifest
        and file_hash(output / "summary.json") != manifest["summary_hash"]
    ):
        msg = "recovery summary hash mismatch"
        raise ValueError(msg)

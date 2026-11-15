"""Bounded nested article-redraw calibration with per-archive durable checkpoints."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from hotelling.artifacts import reject_symlinks
from hotelling.dynamics.artifacts import (
    file_hash,
    read_arrays,
    repository_root,
    value_hash,
    write_arrays,
)
from hotelling.dynamics.resources import (
    ExecutionSettings,
    choose_resources,
    detect_limits,
)
from threadpoolctl import threadpool_limits

from pipeline._kernels.paper6.article_redraw_validation import (
    HISTORIES,
    SCENARIOS,
    ArticleRedrawValidationConfig,
    economic_design,
    measure_outer,
    summarize_outer,
)
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.jax_cache import configure_persistent_cache
from pipeline.precision import own_float64
from pipeline.stages.papers.paper6.artifacts import read, write
from pipeline.stages.papers.paper6.recovery.artifacts import (
    load_stratified_month_counts,
)

COMMAND = "paper6-article-redraw-validation"
MAX_WORKERS = 3
WORKER_BYTES = 512 * 1024**2
_THREAD_LIMIT: object | None = None


def _initialize(threads: int) -> None:
    global _THREAD_LIMIT  # noqa: PLW0603
    own_float64()
    configure_persistent_cache()
    _THREAD_LIMIT = threadpool_limits(limits=threads)


def _open(output: Path) -> Path:
    reject_symlinks(output.absolute())
    output = output.resolve()
    allowed = repository_root() / "data/papers/paper6/article_redraw_validation"
    temporary = Path(tempfile.gettempdir()).resolve()
    if output == temporary or not (
        output.is_relative_to(allowed) or output.is_relative_to(temporary)
    ):
        msg = "article-redraw output requires its namespace or a /tmp subdirectory"
        raise ValueError(msg)
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "provenance.manifest.json").is_file()
    ):
        msg = "nonempty nested output has no resumable manifest"
        raise FileExistsError(msg)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_design(
    record: dict[str, Any], config: ArticleRedrawValidationConfig
) -> None:
    expected = {
        "schema_version": 1,
        "histories": HISTORIES,
        "firms": 100,
        "support": "expanded",
        "budgets": [1],
        "scenarios": list(SCENARIOS),
        **config.model_dump(),
    }
    if not record.get("frozen_before_new_results") or any(
        record.get(key) != value for key, value in expected.items()
    ):
        msg = "article-redraw design differs from the declared fixed design"
        raise ValueError(msg)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    os.close(descriptor)
    try:
        pd.DataFrame.from_records(rows).to_parquet(name, index=False)
        with Path(name).open("rb") as stream:
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)
    return file_hash(path)


def _read_rows(path: Path, digest: str) -> list[dict[str, Any]]:
    if file_hash(path) != digest:
        msg = f"nested outer record hash mismatch: {path}"
        raise ValueError(msg)
    return [
        {
            str(key): None if isinstance(value, float) and np.isnan(value) else value
            for key, value in row.items()
        }
        for row in pd.read_parquet(path).to_dict(orient="records")
    ]


def _manifest(
    output: Path,
    design: Path,
    config: ArticleRedrawValidationConfig,
    inputs: dict[str, object],
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "code_identity": entrypoint_code_identity(COMMAND),
        "config": config.model_dump(),
        "design_sha256": file_hash(design),
        "inputs": inputs,
    }
    path = output / "provenance.manifest.json"
    if path.exists():
        saved = read(path)
        if saved["identity"] != identity or saved["identity_hash"] != value_hash(
            identity
        ):
            msg = (
                "nested article-redraw manifest has incompatible design, code or inputs"
            )
            raise ValueError(msg)
        for name, digest in saved["outputs"].items():
            if file_hash(output / name) != digest:
                msg = f"nested saved output hash mismatch: {name}"
                raise ValueError(msg)
        return saved
    write(output / "design.json", read(design))
    saved: dict[str, Any] = {
        "schema_version": 1,
        "identity": identity,
        "identity_hash": value_hash(identity),
        "status": "running",
        "histories": {},
        "completed": {},
        "outputs": {"design.json": file_hash(output / "design.json")},
        "failures": [],
        "execution": {},
    }
    write(path, saved)
    return saved


def _save_manifest(output: Path, manifest: dict[str, Any]) -> None:
    write(output / "provenance.manifest.json", manifest)


def _prepare_history(
    index: int, counts: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return economic_design(index, counts)


def _history_files(output: Path, index: int) -> tuple[Path, Path]:
    path = output / "histories" / f"history-{index:02}.npz"
    return path, path.with_suffix(".json")


def _admit_histories(
    output: Path,
    manifest: dict[str, Any],
    pool: ProcessPoolExecutor,
    counts: np.ndarray,
    histories: list[int],
) -> None:
    pending = []
    for index in histories:
        if str(index) in manifest["histories"]:
            saved = manifest["histories"][str(index)]
            for name, digest in saved["files"].items():
                if file_hash(output / name) != digest:
                    msg = f"nested economic-history hash mismatch: {name}"
                    raise ValueError(msg)
        else:
            pending.append(index)
    futures = {pool.submit(_prepare_history, index, counts): index for index in pending}
    for future in as_completed(futures):
        index = futures[future]
        array_path, meta_path = _history_files(output, index)
        try:
            arrays, metadata = future.result()
            write_arrays(array_path, arrays)
            write(meta_path, metadata)
            saved = {
                "available": True,
                "files": {
                    str(path.relative_to(output)): file_hash(path)
                    for path in (array_path, meta_path)
                },
            }
        except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as error:
            metadata = {
                "history": index,
                "execution_error": f"{type(error).__name__}: {error}",
                "premises_held": False,
            }
            write(meta_path, metadata)
            saved = {
                "available": False,
                "files": {str(meta_path.relative_to(output)): file_hash(meta_path)},
            }
            manifest["failures"].append(metadata)
        manifest["histories"][str(index)] = saved
        _save_manifest(output, manifest)


def _failed_outer(
    history: int,
    outer: int,
    inner_redraws: int,
    error: str,
    *,
    design: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays = {
        key: np.zeros((len(SCENARIOS), inner_redraws))
        for key in ("beta", "q", "d", "defined", "execution_succeeded")
    }
    rows: list[dict[str, Any]] = [
        {
            "scenario_id": scenario,
            "history": history,
            "outer_replicate": outer,
            "budget": 1,
            "support": "expanded",
            "target": "beta_lambda_h",
            "execution_succeeded": False,
            "execution_error": error,
            "premises_held": False,
            "oracle_beta": None,
            "oracle_q": None,
            "oracle_d": None,
            "oracle_defined": False,
            "article_sensitivity_q025": None,
            "article_sensitivity_q05": None,
            "article_sensitivity_q975": None,
            "inner_attempts": inner_redraws,
            "inner_defined": 0,
            "inner_undefined": inner_redraws,
            "inner_execution_attempts": 0,
            "inner_execution_failed": inner_redraws,
            "quantile_method": "linear",
            "quantile_status": "unavailable",
            "quantile_reason": "execution_failed",
        }
        for scenario in SCENARIOS
    ]
    if design is not None:
        for regime, row in enumerate(rows):
            row.update(
                {
                    f"oracle_{key}": float(design[f"target_{key}"][regime])
                    for key in ("beta", "q", "d", "defined")
                }
            )
    return arrays, rows


def _outer_task(
    output: Path,
    history: int,
    outer: int,
    counts: np.ndarray,
    inner_redraws: int,
    *,
    pilot: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    started = perf_counter()
    path, meta_path = _history_files(output, history)
    metadata = read(meta_path)
    if "execution_error" in metadata:
        arrays, rows = _failed_outer(
            history, outer, inner_redraws, metadata["execution_error"]
        )
    else:
        design_values = None
        try:
            design_values = read_arrays(path)
            arrays, rows = measure_outer(
                design_values,
                counts,
                history,
                outer,
                inner_redraws=inner_redraws,
                pilot=pilot,
            )
            rows = [{**row, **metadata, "execution_error": None} for row in rows]
        except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as error:
            arrays, rows = _failed_outer(
                history,
                outer,
                inner_redraws,
                f"{type(error).__name__}: {error}",
                design=design_values,
            )
    arrays.update(
        {
            "history": np.asarray(float(history)),
            "outer": np.asarray(float(outer)),
            "pilot": np.asarray(float(pilot)),
        }
    )
    return arrays, rows, perf_counter() - started


def _checkpoint(
    output: Path,
    manifest: dict[str, Any],
    history: int,
    outer: int,
    result: tuple[dict[str, np.ndarray], list[dict[str, Any]], float],
    *,
    pilot: bool,
) -> None:
    arrays, rows, seconds = result
    directory = output / ("pilot" if pilot else "checkpoints") / f"history-{history:02}"
    path = directory / f"outer-{outer:04}.npz"
    write_arrays(path, arrays)
    row_path = path.with_suffix(".parquet")
    _write_rows(row_path, rows)
    key = str(path.relative_to(output))
    manifest["completed"][key] = {
        "arrays_sha256": file_hash(path),
        "rows_sha256": file_hash(row_path),
        "seconds": seconds,
    }
    failures = [
        row
        for row in rows
        if not row["execution_succeeded"] or row.get("inner_execution_failed", 0)
    ]
    if failures:
        manifest.setdefault("failures", []).append(
            {
                "history": history,
                "outer_replicate": outer,
                "pilot": pilot,
                "scenarios": [row["scenario_id"] for row in failures],
                "reason": "outer_or_inner_execution_failure",
            }
        )
    _save_manifest(output, manifest)


def _load_checkpoint(
    output: Path, key: str, saved: dict[str, Any], inner_redraws: int
) -> list[dict[str, Any]]:
    path = Path(key)
    if (
        path.is_absolute()
        or ".." in path.parts
        or file_hash(output / path) != saved["arrays_sha256"]
    ):
        msg = f"nested inner-replicate integrity mismatch: {key}"
        raise ValueError(msg)
    arrays = read_arrays(output / path)
    for name in ("beta", "q", "d", "defined", "execution_succeeded"):
        if arrays[name].shape != (len(SCENARIOS), inner_redraws):
            msg = f"nested inner-replicate shape mismatch: {key}/{name}"
            raise ValueError(msg)
    if not np.isin(arrays["defined"], [0, 1]).all() or np.any(arrays["d"] < 0):
        msg = f"nested inner-replicate numerical contract mismatch: {key}"
        raise ValueError(msg)
    rows = _read_rows(output / path.with_suffix(".parquet"), saved["rows_sha256"])
    history, outer = int(arrays["history"]), int(arrays["outer"])
    if (
        len(rows) != len(SCENARIOS)
        or {row["scenario_id"] for row in rows} != set(SCENARIOS)
        or any(
            row["history"] != history or row["outer_replicate"] != outer for row in rows
        )
    ):
        msg = f"nested outer-scalar membership mismatch: {key}"
        raise ValueError(msg)
    return rows


def _run_outers(
    output: Path,
    manifest: dict[str, Any],
    pool: ProcessPoolExecutor,
    counts: np.ndarray,
    histories: list[int],
    outer_replications: int,
    inner_redraws: int,
    workers: int,
    *,
    pilot: bool,
) -> None:
    prefix = "pilot" if pilot else "checkpoints"
    pending = []
    for history in histories:
        for outer in range(outer_replications):
            key = f"{prefix}/history-{history:02}/outer-{outer:04}.npz"
            if key in manifest["completed"]:
                _load_checkpoint(output, key, manifest["completed"][key], inner_redraws)
            else:
                pending.append((history, outer))
    for offset in range(0, len(pending), workers):
        batch = pending[offset : offset + workers]
        futures = [
            pool.submit(
                _outer_task, output, history, outer, counts, inner_redraws, pilot=pilot
            )
            for history, outer in batch
        ]
        for (history, outer), future in zip(batch, futures, strict=True):
            _checkpoint(output, manifest, history, outer, future.result(), pilot=pilot)


def _pilot(
    output: Path,
    manifest: dict[str, Any],
    pool: ProcessPoolExecutor,
    counts: np.ndarray,
    config: ArticleRedrawValidationConfig,
    workers: int,
) -> None:
    if "pilot" in manifest["execution"]:
        return
    _admit_histories(output, manifest, pool, counts, [0])
    _run_outers(
        output,
        manifest,
        pool,
        counts,
        [0],
        config.pilot_replications,
        config.inner_redraws,
        workers,
        pilot=True,
    )
    first = "pilot/history-00/outer-0000.npz"
    parallel = _load_checkpoint(
        output, first, manifest["completed"][first], config.inner_redraws
    )
    serial_arrays, serial_rows, serial_seconds = _outer_task(
        output, 0, 0, counts, config.inner_redraws, pilot=True
    )
    path = output / first
    stored = read_arrays(path)
    for key, value in serial_arrays.items():
        np.testing.assert_allclose(value, stored[key], rtol=1e-12, atol=1e-12)
    for left, right in zip(serial_rows, parallel, strict=True):
        for key, value in left.items():
            if isinstance(value, (int, float)) and isinstance(right[key], (int, float)):
                np.testing.assert_allclose(value, right[key], rtol=1e-12, atol=1e-12)
            elif value != right[key]:
                msg = f"nested serial/spawned pilot scalar disagrees: {key}"
                raise ArithmeticError(msg)
    times = [
        saved["seconds"]
        for key, saved in manifest["completed"].items()
        if key.startswith("pilot/")
    ]
    manifest["execution"]["pilot"] = {
        "history": 0,
        "outer_replications": config.pilot_replications,
        "inner_redraws": config.inner_redraws,
        "seconds_per_outer": float(np.mean(times)),
        "serial_outer_seconds": serial_seconds,
        "serial_parallel_agreement": True,
        "estimated_production_seconds": float(
            np.mean(times) * HISTORIES * config.outer_replications / workers
        ),
        "streams": "99 appended; disjoint from production",
    }
    _save_manifest(output, manifest)


def _finalize(
    output: Path, manifest: dict[str, Any], config: ArticleRedrawValidationConfig
) -> dict[str, Any]:
    expected = {
        f"checkpoints/history-{h:02}/outer-{outer:04}.npz"
        for h in range(HISTORIES)
        for outer in range(config.outer_replications)
    }
    actual = {key for key in manifest["completed"] if key.startswith("checkpoints/")}
    if actual != expected:
        msg = "nested production checkpoint membership is incomplete or undeclared"
        raise ValueError(msg)
    rows = [
        row
        for key in sorted(expected)
        for row in _load_checkpoint(
            output, key, manifest["completed"][key], config.inner_redraws
        )
    ]
    summarized = summarize_outer(rows, config.outer_replications)
    manifest["outputs"]["outer-replicates.parquet"] = _write_rows(
        output / "outer-replicates.parquet", rows
    )
    result = {
        "schema_version": 1,
        "status": "execution_failed"
        if any(
            not row["execution_succeeded"] or row.get("inner_execution_failed", 0)
            for row in rows
        )
        else "scientific_failures"
        if any(not row["premises_held"] for row in rows)
        else "complete",
        "histories": HISTORIES,
        "outer_replications": config.outer_replications,
        "inner_redraws": config.inner_redraws,
        "firms": 100,
        "support": "expanded",
        "budget": 1,
        "sampling_target": (
            "Fixed competitive economy and enumerated history; "
            "reported population law with prescribed core/additional "
            "monthly margins."
        ),
        "method": (
            "Nested paired-stratum article resampling; "
            "conditional-on-defined percentile sensitivity."
        ),
        "quantile_method": "linear",
        "scope": (
            "Calibration under generated monthly laws and timing; "
            "does not establish unconditional corporate inference."
        ),
        "execution": manifest["execution"],
        "failures": manifest["failures"],
        "signal_scale": _signal_scale(output),
        **summarized,
    }
    write(output / "results.json", result)
    manifest["outputs"]["results.json"] = file_hash(output / "results.json")
    manifest["status"] = result["status"]
    _save_manifest(output, manifest)
    return result


def _signal_scale(output: Path) -> dict[str, Any]:
    records = [read(_history_files(output, index)[1]) for index in range(HISTORIES)]
    missing = [row["history"] for row in records if "signal_scale" not in row]
    if missing:
        return {
            "status": "unavailable",
            "unavailable_histories": missing,
            "reason": "missing_economic_history",
        }
    scales = [row["signal_scale"] for row in records]
    return {
        "status": "complete",
        "units": "fractions",
        "first_response_abs_share_change_min": min(
            row["first_response_abs_share_change_min"] for row in scales
        ),
        "first_response_abs_share_change_max": max(
            row["first_response_abs_share_change_max"] for row in scales
        ),
        "maximum_abs_share_change_from_origin": max(
            row["maximum_abs_share_change_from_origin"] for row in scales
        ),
        "news_source_tick": scales[0]["news_source_tick"],
        "first_response_tick": scales[0]["first_response_tick"],
    }


def run_validation(
    output: Path,
    *,
    design: Path,
    config: ArticleRedrawValidationConfig | None = None,
    workers: str = "auto",
    threads: int = 1,
    pilot_only: bool = False,
) -> dict[str, Any]:
    """Run the declared full nested design, preserving every completed outer draw."""
    own_float64()
    configure_persistent_cache()
    config = config or ArticleRedrawValidationConfig()
    _validate_design(read(design), config)
    output = _open(output)
    counts, inputs = load_stratified_month_counts()
    manifest = _manifest(output, design, config, inputs)
    if manifest["status"] != "running":
        # Finalization rechecks the complete checkpoint roster and every input hash.
        return _finalize(output, manifest, config)
    requested = (
        min(MAX_WORKERS, detect_limits().cpus) if workers == "auto" else int(workers)
    )
    if requested < 1 or requested > MAX_WORKERS:
        msg = "nested validation requires one to three bounded workers"
        raise ValueError(msg)
    choice = choose_resources(
        ExecutionSettings(workers=requested, threads=threads),
        deadline=perf_counter() + 3600,
        run_count=HISTORIES * config.outer_replications,
        estimated_worker_bytes=WORKER_BYTES,
    )
    manifest["execution"]["resources"] = choice.to_dict()
    started = perf_counter()
    with ProcessPoolExecutor(
        max_workers=choice.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(choice.threads,),
    ) as pool:
        _pilot(output, manifest, pool, counts, config, choice.workers)
        if pilot_only:
            return {
                "status": "pilot_complete",
                "execution": manifest["execution"],
                "failures": manifest["failures"],
            }
        _admit_histories(output, manifest, pool, counts, list(range(HISTORIES)))
        _run_outers(
            output,
            manifest,
            pool,
            counts,
            list(range(HISTORIES)),
            config.outer_replications,
            config.inner_redraws,
            choice.workers,
            pilot=False,
        )
    manifest["execution"]["elapsed_seconds_this_run"] = perf_counter() - started
    return _finalize(output, manifest, config)

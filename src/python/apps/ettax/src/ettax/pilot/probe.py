# SPDX-License-Identifier: Apache-2.0
"""Fresh-process GPU capacity controller; deliberately never imports JAX."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from ettax.pilot.config import PilotConfig

type AttemptStatus = Literal["pass", "cuda_oom", "failure", "timeout"]

_CUDA_OOM_MARKERS = (
    "can't reduce memory use below",
    "cuda_error_out_of_memory",
    "cuda out of memory",
    "out of memory while trying to allocate",
    "resource_exhausted",
    "failed to allocate",
)


@dataclass(frozen=True, slots=True)
class RawProcessResult:
    """Minimal process result retained for deterministic subprocess tests."""

    returncode: int
    stdout: str
    stderr: str
    pid: int


@dataclass(frozen=True, slots=True)
class ProbeAttempt:
    """One isolated batch/length compilation and optimizer-step attempt."""

    length: int
    batch_size: int
    steps: int
    status: AttemptStatus
    pid: int
    elapsed_seconds: float
    stderr: str = ""
    result: dict[str, object] | None = None

    @property
    def passed(self) -> bool:
        """Return whether compilation and every requested step succeeded."""
        return self.status == "pass"


class ProbeError(RuntimeError):
    """An unrelated subprocess failure that must not be classified as capacity."""


type ProcessRunner = Callable[[Sequence[str], Mapping[str, str], int], RawProcessResult]


def run_capacity_probe(
    config: PilotConfig,
    config_path: str | Path,
    *,
    runner: ProcessRunner | None = None,
) -> dict[str, object]:
    """Resolve every configured length and write one downstream manifest."""
    if config.probe_manifest_path.exists():
        message = (
            f"resolved probe already exists at {config.probe_manifest_path}; "
            "use --reuse to retain it"
        )
        raise FileExistsError(message)
    execute = runner or _run_process
    config_sha256 = _sha256(Path(config_path))
    progress = _load_progress(config, config_sha256=config_sha256)
    all_attempts = list(
        cast("Sequence[Mapping[str, object]]", progress.get("attempts", ()))
    )
    resolutions = dict(cast("Mapping[str, object]", progress.get("resolutions", {})))
    for length in config.sequence_buckets:
        if str(length) in resolutions:
            continue
        maximum, attempts = search_maximum_batch(
            config,
            config_path,
            length=length,
            runner=execute,
        )
        all_attempts.extend(asdict(attempt) for attempt in attempts)
        physical, accumulation = resolve_physical_batch(
            maximum_batch=maximum,
            sequence_length=length,
            target_slots=config.target_slots_per_update,
        )
        validation = _validated_attempt(attempts, maximum)
        resolutions[str(length)] = {
            "gradient_accumulation": accumulation,
            "maximum_batch": maximum,
            "physical_batch": physical,
            "target_effective_batch": config.target_slots_per_update // length,
            "target_slots_per_update": config.target_slots_per_update,
            "validation": _compact_validation(validation),
        }
        snapshot = _probe_manifest(
            config,
            config_sha256=config_sha256,
            attempts=all_attempts,
            resolutions=resolutions,
            complete=False,
        )
        _write_json_atomic(config.probe_progress_path, snapshot)

    manifest = _probe_manifest(
        config,
        config_sha256=config_sha256,
        attempts=all_attempts,
        resolutions=resolutions,
        complete=True,
    )
    _write_json_atomic(config.probe_progress_path, manifest)
    _write_json_atomic(config.probe_manifest_path, manifest)
    write_compact_probe_attachment(config, manifest)
    return manifest


def _load_progress(config: PilotConfig, *, config_sha256: str) -> dict[str, object]:
    if not config.probe_progress_path.is_file():
        return {}
    progress = json.loads(config.probe_progress_path.read_text(encoding="utf-8"))
    if (
        progress.get("pilot_id") != config.pilot_id
        or progress.get("config_sha256") != config_sha256
    ):
        raise ProbeError(
            "probe progress belongs to a different pilot configuration; "
            f"move it aside before retrying: {config.probe_progress_path}"
        )
    return cast("dict[str, object]", progress)


def _probe_manifest(
    config: PilotConfig,
    *,
    config_sha256: str,
    attempts: Sequence[Mapping[str, object]],
    resolutions: Mapping[str, object],
    complete: bool,
) -> dict[str, object]:
    successful_payloads = [
        cast("Mapping[str, object]", attempt["result"])
        for attempt in attempts
        if attempt.get("result") is not None
    ]
    return {
        "allocator": {
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{config.allocator_fraction:.2f}",
        },
        "attempts": list(attempts),
        "complete": complete,
        "config_sha256": config_sha256,
        "device": successful_payloads[0].get("device") if successful_payloads else None,
        "environment": successful_payloads[0].get("environment")
        if successful_payloads
        else None,
        "pilot_id": config.pilot_id,
        "resolutions": dict(resolutions),
    }


def write_compact_probe_attachment(
    config: PilotConfig, manifest: Mapping[str, object]
) -> Path:
    """Persist the governed probe fields without raw attempts or stderr."""
    compact = {
        key: manifest.get(key)
        for key in ("allocator", "device", "environment", "pilot_id")
    }
    compact["resolutions"] = _compact_resolutions(manifest)
    destination = config.plan_attachment_dir / "probe-manifest.json"
    _write_json_atomic(destination, compact)
    return destination


def _compact_resolutions(manifest: Mapping[str, object]) -> dict[str, object]:
    raw_resolutions = cast(
        "Mapping[str, Mapping[str, object]]", manifest["resolutions"]
    )
    raw_attempts = cast("Sequence[Mapping[str, object]]", manifest.get("attempts", ()))
    compact: dict[str, object] = {}
    for raw_length, raw_resolution in raw_resolutions.items():
        resolution = dict(raw_resolution)
        if "validation" not in resolution:
            maximum = int(cast("int", resolution["maximum_batch"]))
            candidates = [
                attempt
                for attempt in raw_attempts
                if int(cast("int", attempt["length"])) == int(raw_length)
                and int(cast("int", attempt["batch_size"])) == maximum
                and str(attempt["status"]) == "pass"
            ]
            if candidates:
                chosen = max(candidates, key=lambda attempt: int(attempt["steps"]))
                resolution["validation"] = _compact_serialized_attempt(chosen)
        compact[raw_length] = resolution
    return compact


def _validated_attempt(
    attempts: Sequence[ProbeAttempt], maximum_batch: int
) -> ProbeAttempt:
    candidates = [
        attempt
        for attempt in attempts
        if attempt.batch_size == maximum_batch and attempt.passed
    ]
    if not candidates:
        raise ProbeError(f"maximum batch {maximum_batch} lacks a passing attempt")
    return max(candidates, key=lambda attempt: attempt.steps)


def _compact_validation(attempt: ProbeAttempt) -> dict[str, object]:
    values: dict[str, object] = {
        "elapsed_seconds": attempt.elapsed_seconds,
        "pid": attempt.pid,
        "steps": attempt.steps,
    }
    result = attempt.result or {}
    for key in (
        "compiler_memory_analysis",
        "loss",
        "token_slots_per_second",
    ):
        values[key] = result.get(key)
    return values


def _compact_serialized_attempt(attempt: Mapping[str, object]) -> dict[str, object]:
    values: dict[str, object] = {
        "elapsed_seconds": attempt.get("elapsed_seconds"),
        "pid": attempt.get("pid"),
        "steps": attempt.get("steps"),
    }
    result = cast("Mapping[str, object]", attempt.get("result") or {})
    for key in (
        "compiler_memory_analysis",
        "loss",
        "token_slots_per_second",
    ):
        values[key] = result.get(key)
    return values


def search_maximum_batch(
    config: PilotConfig,
    config_path: str | Path,
    *,
    length: int,
    runner: ProcessRunner,
) -> tuple[int, list[ProbeAttempt]]:
    """Search powers of two, close the integer gap, then validate ten steps."""
    attempts: list[ProbeAttempt] = []
    highest_pass = 0
    first_failure: int | None = None
    candidate = 1
    while candidate <= config.probe_max_batch:
        attempt = run_probe_attempt(
            config,
            config_path,
            length=length,
            batch_size=candidate,
            steps=1,
            runner=runner,
        )
        attempts.append(attempt)
        _raise_unrelated(attempt)
        if not attempt.passed:
            first_failure = candidate
            break
        highest_pass = candidate
        candidate *= 2
    if highest_pass == 0:
        raise ProbeError(f"length {length} cannot train at batch one")
    if first_failure is None:
        raise ProbeError(
            f"length {length} still passes at configured probe_max_batch="
            f"{config.probe_max_batch}; raise the cap to observe a failure boundary"
        )
    if first_failure is not None and first_failure - highest_pass > 1:
        highest_pass = _binary_search(
            config,
            config_path,
            length=length,
            low=highest_pass,
            high=first_failure - 1,
            steps=1,
            runner=runner,
            attempts=attempts,
        )

    validation = run_probe_attempt(
        config,
        config_path,
        length=length,
        batch_size=highest_pass,
        steps=config.probe_validation_steps,
        runner=runner,
    )
    attempts.append(validation)
    _raise_unrelated(validation)
    if validation.passed:
        return highest_pass, attempts
    validated = _binary_search(
        config,
        config_path,
        length=length,
        low=1,
        high=highest_pass - 1,
        steps=config.probe_validation_steps,
        runner=runner,
        attempts=attempts,
    )
    return validated, attempts


def run_probe_attempt(
    config: PilotConfig,
    config_path: str | Path,
    *,
    length: int,
    batch_size: int,
    steps: int,
    runner: ProcessRunner,
) -> ProbeAttempt:
    """Launch exactly one candidate in a fresh configured child process."""
    command = (
        sys.executable,
        "-m",
        "ettax.pilot.worker",
        "--config",
        str(Path(config_path).absolute()),
        "--length",
        str(length),
        "--batch-size",
        str(batch_size),
        "--steps",
        str(steps),
    )
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{config.allocator_fraction:.2f}",
    }
    started = time.perf_counter()
    try:
        process = runner(command, environment, config.training_ceiling_seconds)
    except subprocess.TimeoutExpired as error:
        return ProbeAttempt(
            length=length,
            batch_size=batch_size,
            steps=steps,
            status="timeout",
            pid=int(error.output or -1) if str(error.output or "").isdigit() else -1,
            elapsed_seconds=time.perf_counter() - started,
            stderr=str(error.stderr or ""),
        )
    elapsed = time.perf_counter() - started
    payload = _last_json_object(process.stdout)
    if process.returncode == 0 and payload is not None:
        return ProbeAttempt(
            length=length,
            batch_size=batch_size,
            steps=steps,
            status="pass",
            pid=int(cast("int", payload.get("pid", process.pid))),
            elapsed_seconds=elapsed,
            result=payload,
        )
    status: AttemptStatus = (
        "cuda_oom" if is_cuda_oom(process.stderr, process.stdout) else "failure"
    )
    return ProbeAttempt(
        length=length,
        batch_size=batch_size,
        steps=steps,
        status=status,
        pid=process.pid,
        elapsed_seconds=elapsed,
        stderr=process.stderr,
        result=payload,
    )


def resolve_physical_batch(
    *, maximum_batch: int, sequence_length: int, target_slots: int
) -> tuple[int, int]:
    """Prefer the desired batch; otherwise use its largest passing divisor."""
    if min(maximum_batch, sequence_length, target_slots) < 1:
        raise ValueError("batch, length, and target slots must be positive")
    if target_slots % sequence_length:
        raise ValueError("target slots must be divisible by sequence length")
    desired = target_slots // sequence_length
    physical = max(
        candidate
        for candidate in range(1, min(desired, maximum_batch) + 1)
        if desired % candidate == 0
    )
    return physical, desired // physical


def is_cuda_oom(*messages: str) -> bool:
    """Classify only allocator/resource-exhaustion failures as CUDA OOM."""
    combined = "\n".join(messages).lower()
    return any(marker in combined for marker in _CUDA_OOM_MARKERS)


def _binary_search(
    config: PilotConfig,
    config_path: str | Path,
    *,
    length: int,
    low: int,
    high: int,
    steps: int,
    runner: ProcessRunner,
    attempts: list[ProbeAttempt],
) -> int:
    maximum = 0
    while low <= high:
        middle = (low + high) // 2
        attempt = run_probe_attempt(
            config,
            config_path,
            length=length,
            batch_size=middle,
            steps=steps,
            runner=runner,
        )
        attempts.append(attempt)
        _raise_unrelated(attempt)
        if attempt.passed:
            maximum = middle
            low = middle + 1
        else:
            high = middle - 1
    if maximum == 0:
        raise ProbeError(f"length {length} has no validated passing batch")
    return maximum


def _raise_unrelated(attempt: ProbeAttempt) -> None:
    if attempt.status in {"failure", "timeout"}:
        raise ProbeError(
            f"probe failed independently of capacity for length={attempt.length} "
            f"batch={attempt.batch_size}: {attempt.stderr}"
        )


def _run_process(
    command: Sequence[str], environment: Mapping[str, str], timeout: int
) -> RawProcessResult:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout, output=str(process.pid), stderr=stderr
        ) from None
    return RawProcessResult(process.returncode, stdout, stderr, process.pid)


def _last_json_object(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

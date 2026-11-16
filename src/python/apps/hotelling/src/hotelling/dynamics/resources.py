"""Bounded execution calibration without changing the scientific design."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_PREFLIGHT_SECONDS = 60.0
WORKER_CANDIDATES = (1, 3, 6, 12)
AUTO_CANDIDATES = ((1, 1), (1, 4), (3, 1), (3, 2), (6, 1), (12, 1))
DEFAULT_WORKER_BYTES = 256 * 1024**2
MEMORY_FRACTION = 0.8
THROUGHPUT_TIE_FRACTION = 0.05
MIN_PARALLEL_TASKS = 3


@dataclass(frozen=True)
class ExecutionSettings:
    """Execution-only requests; these controls never enter scientific identities."""

    workers: int | Literal["auto"] = "auto"
    threads: int | None = None
    preflight_seconds: float = MAX_PREFLIGHT_SECONDS

    def __post_init__(self) -> None:
        """Reject ambiguous integers and unbounded calibration requests."""
        if self.workers != "auto" and (
            type(self.workers) is not int or self.workers <= 0
        ):
            msg = "workers must be auto or a positive integer"
            raise ValueError(msg)
        if self.threads is not None and (
            type(self.threads) is not int or self.threads <= 0
        ):
            msg = "threads must be a positive integer"
            raise ValueError(msg)
        if (
            type(self.preflight_seconds) not in (int, float)
            or not math.isfinite(self.preflight_seconds)
            or not 0 < self.preflight_seconds <= MAX_PREFLIGHT_SECONDS
        ):
            msg = "preflight_seconds must be in (0, 60]"
            raise ValueError(msg)


@dataclass(frozen=True)
class ResourceLimits:
    """Effective host limits, including affinity and ancestor cgroup quotas."""

    cpus: int
    available_memory_bytes: int
    affinity_cpus: int
    quota_cpus: int | None


@dataclass(frozen=True)
class ResourceMeasurement:
    """One bounded throughput measurement from a representative fixed workload."""

    workers: int
    threads: int
    tasks_per_second: float | None
    elapsed_seconds: float
    status: str


@dataclass(frozen=True)
class ResourceChoice:
    """Selected execution controls and the evidence used to choose them."""

    workers: int
    threads: int
    preflight_seconds: float
    cpu_limit: int
    memory_limit_bytes: int
    estimated_worker_bytes: int
    candidates: tuple[ResourceMeasurement, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return finite JSON-compatible execution metadata."""
        return asdict(self)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def _cgroup_directories() -> tuple[Path, ...]:
    root = Path("/sys/fs/cgroup")
    membership = _read(Path("/proc/self/cgroup")) or ""
    relative = next(
        (line[3:] for line in membership.splitlines() if line.startswith("0::")), "/"
    )
    leaf = root / relative.lstrip("/")
    return tuple(path for path in (leaf, *leaf.parents) if path.is_relative_to(root))


def detect_limits() -> ResourceLimits:
    """Read effective CPU and currently available memory without external tools."""
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    quotas: list[int] = []
    memory = [
        int(line.split()[1]) * 1024
        for line in (_read(Path("/proc/meminfo")) or "").splitlines()
        if line.startswith("MemAvailable:")
    ]
    for directory in _cgroup_directories():
        quota = _read(directory / "cpu.max")
        if quota is not None:
            amount, period = quota.split()
            if amount != "max":
                quotas.append(max(1, int(amount) // int(period)))
        maximum = _read(directory / "memory.max")
        current = _read(directory / "memory.current")
        if maximum is not None and maximum != "max":
            memory.append(max(0, int(maximum) - int(current or "0")))
    quota_cpus = min(quotas) if quotas else None
    cpus = max(1, min(affinity, quota_cpus or affinity))
    available = min(memory) if memory else 2 * DEFAULT_WORKER_BYTES
    return ResourceLimits(cpus, available, affinity, quota_cpus)


def _worker_cap(
    limits: ResourceLimits, threads: int, run_count: int, worker_bytes: int
) -> int:
    memory_workers = (
        int(limits.available_memory_bytes * MEMORY_FRACTION) // worker_bytes
    )
    if threads > limits.cpus or memory_workers < 1:
        msg = "requested execution exceeds the effective CPU or memory budget"
        raise ValueError(msg)
    return min(limits.cpus // threads, memory_workers, run_count)


def choose_resources(
    settings: ExecutionSettings,
    *,
    deadline: float,
    run_count: int,
    estimated_worker_bytes: int = DEFAULT_WORKER_BYTES,
    probe: Callable[[int, int, float], float] | None = None,
) -> ResourceChoice:
    """Select worker/thread pairs within 60 seconds of the campaign deadline.

    The optional probe returns completed representative tasks per second and
    must honor its absolute deadline. It must use a fixed workload for every
    candidate. Auto compares (1,1), (1,4), (3,1), (3,2), (6,1), (12,1)
    after resource clipping. A gain must exceed 5% to replace the earlier,
    usually smaller pool. Explicit threads hold that axis fixed. Only
    execution settings adapt; no scientific cell is removed.
    """
    started = perf_counter()
    if not math.isfinite(deadline) or deadline <= started:
        msg = "resource selection requires time remaining on the campaign deadline"
        raise TimeoutError(msg)
    if (
        type(run_count) is not int
        or run_count < 1
        or type(estimated_worker_bytes) is not int
        or estimated_worker_bytes < 1
    ):
        msg = "run_count and estimated_worker_bytes must be positive integers"
        raise ValueError(msg)
    limits = detect_limits()
    threads = settings.threads or 1
    cap = _worker_cap(limits, threads, run_count, estimated_worker_bytes)
    measurements: list[ResourceMeasurement] = []
    if isinstance(settings.workers, int):
        if settings.workers > cap:
            msg = f"requested {settings.workers} workers exceeds effective cap {cap}"
            raise ValueError(msg)
        selected, reason = settings.workers, "explicit_execution_override"
    elif run_count < MIN_PARALLEL_TASKS or cap == 1:
        selected, reason = 1, "serial_small_campaign_or_resource_cap"
    elif probe is None:
        selected, reason = 1, "serial_without_measured_parallel_evidence"
    else:
        selected, threads, measurements = _calibrate(
            settings, limits, run_count, estimated_worker_bytes, deadline, probe
        )
        reason = (
            "bounded_measured_throughput"
            if measurements
            else "serial_preflight_budget_exhausted"
        )
    return ResourceChoice(
        selected,
        threads,
        perf_counter() - started,
        limits.cpus,
        limits.available_memory_bytes,
        estimated_worker_bytes,
        tuple(measurements),
        reason,
    )


def _calibrate(
    settings: ExecutionSettings,
    limits: ResourceLimits,
    run_count: int,
    worker_bytes: int,
    deadline: float,
    probe: Callable[[int, int, float], float],
) -> tuple[int, int, list[ResourceMeasurement]]:
    preflight_deadline = min(deadline, perf_counter() + settings.preflight_seconds)
    requested = (
        tuple((count, settings.threads) for count in WORKER_CANDIDATES)
        if settings.threads is not None
        else AUTO_CANDIDATES
    )
    candidates = list(
        dict.fromkeys(
            (
                min(
                    workers,
                    _worker_cap(
                        limits, min(threads, limits.cpus), run_count, worker_bytes
                    ),
                ),
                min(threads, limits.cpus),
            )
            for workers, threads in requested
        )
    )
    measurements: list[ResourceMeasurement] = []
    selected, selected_threads, best = 1, settings.threads or 1, 0.0
    for workers, threads in candidates:
        started = perf_counter()
        if started >= preflight_deadline:
            break
        try:
            throughput = probe(workers, threads, preflight_deadline)
        except TimeoutError:
            measurements.append(
                ResourceMeasurement(
                    workers, threads, None, perf_counter() - started, "timeout"
                )
            )
            break
        if not math.isfinite(throughput) or throughput <= 0:
            msg = "resource probe must return finite positive tasks per second"
            raise ValueError(msg)
        measurements.append(
            ResourceMeasurement(
                workers, threads, throughput, perf_counter() - started, "measured"
            )
        )
        if throughput > best * (1 + THROUGHPUT_TIE_FRACTION):
            selected, selected_threads, best = workers, threads, throughput
    return selected, selected_threads, measurements

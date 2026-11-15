"""Resource choices respect effective limits and retain calibration evidence."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import pytest
from hotelling.dynamics import resources
from hotelling.dynamics.resources import (
    ExecutionSettings,
    ResourceLimits,
    choose_resources,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def roomy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic execution capacity independent of the test host."""
    monkeypatch.setattr(
        resources, "detect_limits", lambda: ResourceLimits(12, 8 * 1024**3, 12, None)
    )


@pytest.mark.usefixtures("roomy")
def test_measured_candidates_preserve_evidence_and_ties() -> None:
    """Choose measured throughput while preferring fewer workers within 5%."""
    calls: list[tuple[int, int, float]] = []
    started = perf_counter()

    def probe(workers: int, threads: int, deadline: float) -> float:
        calls.append((workers, threads, deadline))
        return {
            (1, 1): 1.0,
            (1, 4): 1.01,
            (3, 1): 2.5,
            (3, 2): 2.6,
            (6, 1): 2.6,
            (12, 1): 2.4,
        }[workers, threads]

    choice = choose_resources(
        ExecutionSettings(), deadline=started + 100, run_count=252, probe=probe
    )
    assert (choice.workers, choice.threads) == (3, 1)
    assert [(workers, threads) for workers, threads, _ in calls] == list(
        resources.AUTO_CANDIDATES
    )
    assert all(deadline <= started + 61 for _, _, deadline in calls)
    assert len(choice.candidates) == len(calls)
    json.dumps(choice.to_dict(), allow_nan=False)


def test_affinity_quota_and_memory_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ancestor quotas and remaining cgroup memory further restrict affinity."""
    texts = {
        "/proc/meminfo": "MemAvailable: 9000000 kB",
        "/cg/leaf/cpu.max": "350000 100000",
        "/cg/cpu.max": "200000 100000",
        "/cg/leaf/memory.max": "1000000000",
        "/cg/leaf/memory.current": "200000000",
    }

    def read(path: Path) -> str | None:
        return texts.get(str(path))

    monkeypatch.setattr(resources, "_read", read)
    monkeypatch.setattr(
        resources, "_cgroup_directories", lambda: (Path("/cg/leaf"), Path("/cg"))
    )
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(8)))
    limits = resources.detect_limits()
    assert (limits.cpus, limits.quota_cpus, limits.available_memory_bytes) == (
        2,
        2,
        800000000,
    )


def test_memory_clips_auto_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration never launches more workers than its conservative memory cap."""
    monkeypatch.setattr(
        resources, "detect_limits", lambda: ResourceLimits(12, 1000, 12, None)
    )
    seen: list[int] = []

    def probe(workers: int, threads: int, deadline: float) -> float:
        assert threads > 0
        assert deadline > perf_counter()
        seen.append(workers)
        return float(workers)

    choice = choose_resources(
        ExecutionSettings(),
        deadline=perf_counter() + 60,
        run_count=20,
        estimated_worker_bytes=250,
        probe=probe,
    )
    assert seen == [1, 1, 3, 3]
    assert choice.workers == seen[-1]


@pytest.mark.usefixtures("roomy")
def test_small_campaign_skips_preflight() -> None:
    """A two-task campaign avoids pool startup and calibration overhead."""
    choice = choose_resources(
        ExecutionSettings(), deadline=perf_counter() + 60, run_count=2
    )
    assert choice.workers == 1
    assert not choice.candidates


@pytest.mark.usefixtures("roomy")
def test_auto_selects_a_measured_thread_count() -> None:
    """Thread calibration can win independently of the number of processes."""

    def probe(workers: int, threads: int, deadline: float) -> float:
        assert deadline > perf_counter()
        return 10.0 if (workers, threads) == (1, 4) else 1.0

    choice = choose_resources(
        ExecutionSettings(), deadline=perf_counter() + 60, run_count=20, probe=probe
    )
    assert (choice.workers, choice.threads) == (1, 4)


@pytest.mark.usefixtures("roomy")
def test_explicit_threads_keep_one_axis_fixed() -> None:
    """An explicit BLAS thread count prevents hidden thread-choice changes."""
    seen: list[tuple[int, int]] = []

    def probe(workers: int, threads: int, deadline: float) -> float:
        assert deadline > perf_counter()
        seen.append((workers, threads))
        return float(workers)

    choice = choose_resources(
        ExecutionSettings(threads=2),
        deadline=perf_counter() + 60,
        run_count=20,
        probe=probe,
    )
    assert seen == [(1, 2), (3, 2), (6, 2)]
    assert (choice.workers, choice.threads) == (6, 2)


@pytest.mark.usefixtures("roomy")
def test_explicit_override_and_oversubscription() -> None:
    """Explicit workers and threads are recorded and checked as a product."""
    choice = choose_resources(
        ExecutionSettings(workers=3, threads=4),
        deadline=perf_counter() + 60,
        run_count=20,
    )
    assert (choice.workers, choice.threads) == (3, 4)
    with pytest.raises(ValueError, match="effective cap"):
        choose_resources(
            ExecutionSettings(workers=6, threads=4),
            deadline=perf_counter() + 60,
            run_count=20,
        )


@pytest.mark.usefixtures("roomy")
def test_preflight_timeout_retains_successful_candidate() -> None:
    """A slower candidate cannot erase an earlier usable calibration result."""

    def probe(workers: int, threads: int, deadline: float) -> float:
        assert threads > 0
        assert deadline > perf_counter()
        if workers > 1:
            raise TimeoutError
        return 1.0

    choice = choose_resources(
        ExecutionSettings(), deadline=perf_counter() + 60, run_count=20, probe=probe
    )
    assert choice.workers == 1
    assert [entry.status for entry in choice.candidates] == [
        "measured",
        "measured",
        "timeout",
    ]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ExecutionSettings(workers=0),
        lambda: ExecutionSettings(workers=True),
        lambda: ExecutionSettings(threads=0),
        lambda: ExecutionSettings(preflight_seconds=61),
    ],
)
def test_invalid_execution_controls(
    constructor: Callable[[], ExecutionSettings],
) -> None:
    """Execution overrides reject zero, booleans and oversized calibration."""
    with pytest.raises(ValueError, match="must"):
        constructor()


def test_expired_campaign_deadline() -> None:
    """Preflight cannot run outside the campaign's existing budget."""
    with pytest.raises(TimeoutError):
        choose_resources(ExecutionSettings(), deadline=perf_counter() - 1, run_count=4)

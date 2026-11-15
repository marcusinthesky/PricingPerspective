"""Unit tests for the telemetry package."""

from __future__ import annotations

import json
import pstats
import selectors
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from dotell import begin, jax_diagnostics
from dotell._runtime import _Sampler

_APPEND_RUN_COUNT = 3
_JAX_EVENT_COUNT = 5
_FAST_INTERVAL = 0.05
_CHILD_TIMEOUT = 5.0


def test_begin_writes_one_row(tmp_path: Path) -> None:
    """A begin/finalize cycle appends exactly one well-formed JSON row."""
    sink = tmp_path / "runs.jsonl"
    finalize = begin("unit-stage", app="pytest", sink=sink, interval=0.01)
    # simulate a tiny amount of work so the sampler takes a reading
    sum(range(100_000))
    finalize()

    rows = sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["stage"] == "unit-stage"
    assert row["app"] == "pytest"
    assert row["wall_s"] >= 0.0
    assert row["peak_rss_gb"] > 0.0  # the test process has non-zero RSS
    for key in ("ts", "mean_cpu_pct", "peak_cpu_pct", "n_cpu", "host", "git_sha"):
        assert key in row


def test_begin_appends_across_runs(tmp_path: Path) -> None:
    """Successive runs append rather than overwrite (across-run history)."""
    sink = tmp_path / "runs.jsonl"
    for _ in range(_APPEND_RUN_COUNT):
        begin("s", app="pytest", sink=sink, interval=0.01)()
    assert (
        len(sink.read_text(encoding="utf-8").strip().splitlines()) == _APPEND_RUN_COUNT
    )


def test_begin_profiles_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PP_TELEMETRY_PROFILE=1 dumps a loadable .pstats file and adds profile_path."""
    monkeypatch.setenv("PP_TELEMETRY_PROFILE", "1")
    sink = tmp_path / "runs.jsonl"
    finalize = begin("prof-stage", app="pytest", sink=sink, interval=0.01)
    sum(range(100_000))
    finalize()

    row = json.loads(sink.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "profile_path" in row
    profile_path = Path(row["profile_path"])
    assert profile_path.is_file()
    assert profile_path.parent == sink.parent / "profiles"
    # Must load cleanly as a pstats file.
    pstats.Stats(str(profile_path))


def test_begin_no_profile_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the row schema unchanged when profiling is disabled."""
    monkeypatch.delenv("PP_TELEMETRY_PROFILE", raising=False)
    sink = tmp_path / "runs.jsonl"
    finalize = begin("no-prof-stage", app="pytest", sink=sink, interval=0.01)
    sum(range(100_000))
    finalize()

    assert not (sink.parent / "profiles").exists()
    row = json.loads(sink.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "profile_path" not in row
    assert set(row.keys()) == {
        "ts",
        "app",
        "stage",
        "wall_s",
        "peak_rss_gb",
        "mean_cpu_pct",
        "peak_cpu_pct",
        "n_cpu",
        "host",
        "git_sha",
        # Observed XLA compilation counting is default-on (see `_xla_compiles`);
        # unlike the `PP_TELEMETRY_JAX` fields it is part of every row.
        "xla_compiles",
        "xla_compile_s",
        "xla_compiles_per_wall_s",
        # Spawned workers report separately: a stage that fans all its work out
        # has its churn in these, and zero in the three above.
        "xla_compiles_workers",
        "xla_compile_s_workers",
        "xla_worker_reports",
    }


class _ArrayLike:
    """JAX-free stand-in proving signatures use only public shape/dtype metadata."""

    shape = (8, 3)
    dtype = "float32"


def test_jax_diagnostics_distinguish_variation_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit hooks retain shape/static variants and fallback separately."""
    monkeypatch.setenv("PP_TELEMETRY_JAX", "1")
    sink = tmp_path / "runs.jsonl"
    finalize = begin("jax-stage", app="pytest", sink=sink, interval=0.01)
    diagnostics = jax_diagnostics()
    with diagnostics.compile(
        "statadic.energy", _ArrayLike(), static={"n_steps": 8}, cache="miss"
    ):
        pass
    with diagnostics.execution("statadic.energy", _ArrayLike(), static={"n_steps": 16}):
        pass
    with diagnostics.sync("statadic.energy", _ArrayLike()):
        pass
    diagnostics.fallback("statadic.energy.irregular", _ArrayLike())
    diagnostics.worker_failure("statadic.energy.worker", _ArrayLike())
    finalize()

    row = json.loads(sink.read_text(encoding="utf-8").strip())
    assert row["jax_compile_attempts"] == 1
    assert row["jax_compile_failures"] == 0
    assert row["jax_cache_misses"] == 1
    assert row["jax_cache_hits"] == 0
    assert row["jax_fallback_count"] == 1
    assert row["jax_worker_failures"] == 1
    assert row["jax_execution_s"] >= 0.0
    assert row["jax_sync_s"] >= 0.0
    assert len(row["jax_events"]) == _JAX_EVENT_COUNT
    signatures = [event["signature"] for event in row["jax_events"]]
    assert all("float32[8x3]" in signature for signature in signatures)
    assert signatures[0] != signatures[1]  # known static variation remains visible
    assert "n_steps:int#" in signatures[0]
    assert "n_steps:int#" in signatures[1]


def test_jax_diagnostics_are_absent_when_not_opted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional JAX schema does not alter legacy telemetry rows by default."""
    monkeypatch.delenv("PP_TELEMETRY_JAX", raising=False)
    sink = tmp_path / "runs.jsonl"
    finalize = begin("jax-disabled", app="pytest", sink=sink, interval=0.01)
    with jax_diagnostics().compile("ignored", _ArrayLike(), cache="miss"):
        pass
    finalize()

    row = json.loads(sink.read_text(encoding="utf-8").strip())
    assert not any(key.startswith("jax_") for key in row)


@dataclass
class _Process:
    """Controllable psutil readings; each new object starts with no CPU baseline."""

    pid: int
    created: float = 1.0
    cpu: float = 0.0
    rss: int = 100
    descendants: list[_Process] = field(default_factory=list)
    cpu_reads: int = 0
    cpu_error: psutil.Error | None = None
    rss_error: psutil.Error | None = None

    def children(self, *, recursive: bool) -> list[_Process]:
        """Expose the current recursive process inventory."""
        assert recursive
        return self.descendants

    def create_time(self) -> float:
        """Distinguish recycled PIDs from the previous process identity."""
        return self.created

    def cpu_percent(self, *, interval: float | None) -> float:
        """Prime once, then return the configured nonblocking CPU reading."""
        assert interval is None
        if self.cpu_error is not None:
            raise self.cpu_error
        self.cpu_reads += 1
        return self.cpu if self.cpu_reads > 1 else 0.0

    def memory_info(self) -> SimpleNamespace:
        """Read current RSS without caching it."""
        if self.rss_error is not None:
            raise self.rss_error
        return SimpleNamespace(rss=self.rss)


@pytest.fixture
def process(monkeypatch: pytest.MonkeyPatch) -> _Process:
    """Give the sampler a parent whose inventory and readings tests control."""
    parent = _Process(1)
    monkeypatch.setattr("dotell._runtime.psutil.Process", lambda: parent)
    return parent


def test_sampler_reuses_primed_processes_and_sums_cpu(process: _Process) -> None:
    """Fresh enumeration wrappers must not reset a live child's CPU baseline."""
    child = _Process(2)
    process.descendants = [child]
    sampler = _Sampler(_FAST_INTERVAL)
    sampler.sample()
    assert child.cpu_reads > 1  # discovery primes before the first measurement
    assert sampler.cpu_samples == [0.0]

    process.cpu = 10.0
    child.cpu = 60.0
    process.descendants = [_Process(2)]
    sampler.sample()
    assert sampler.cpu_samples == [0.0, 70.0]


def test_sampler_restarts_recycled_pids_and_drops_exited_children(
    process: _Process,
) -> None:
    """Recycled PIDs restart their baseline; exited children stop contributing."""
    child = _Process(2, cpu=90.0)
    process.descendants = [child]
    sampler = _Sampler(_FAST_INTERVAL)
    sampler.sample()
    replacement = _Process(2, created=2.0)
    process.descendants = [replacement]
    sampler.sample()
    assert replacement.cpu_reads > 1

    replacement.cpu = 40.0
    sampler.sample()
    process.descendants = []
    sampler.sample()
    assert sampler.cpu_samples == [90.0, 0.0, 40.0, 0.0]
    assert sampler._tree() == [process]  # noqa: SLF001


def test_sampler_reads_fresh_rss(process: _Process) -> None:
    """Caching CPU identities must not cache memory values."""
    child = _Process(2, rss=200)
    process.descendants = [child]
    sampler = _Sampler(_FAST_INTERVAL)
    sampler.sample()
    assert sampler.peak_rss == process.rss + child.rss
    child.rss = 400
    sampler.sample()
    assert sampler.peak_rss == process.rss + child.rss


@pytest.mark.parametrize("error", [psutil.NoSuchProcess(2), psutil.AccessDenied(2)])
@pytest.mark.parametrize("reading", ["rss", "cpu"])
def test_sampler_recovers_from_process_read_errors(
    process: _Process, error: psutil.Error, reading: str
) -> None:
    """Failed reads after successful enumeration must not stop later samples."""
    child = _Process(2)
    process.descendants = [child]
    sampler = _Sampler(_FAST_INTERVAL)
    sampler.sample()
    setattr(child, f"{reading}_error", error)
    sampler.sample()  # error occurs inside a read, after successful enumeration
    setattr(child, f"{reading}_error", None)
    child.cpu = 40.0
    child.rss = 300
    sampler.sample()
    assert sampler.cpu_samples == [0.0, 0.0, 40.0]
    assert sampler.peak_rss == process.rss + child.rss


def test_sampler_observes_a_real_child_and_stops() -> None:
    """A pipe handshake keeps one real child alive without sleeps or CPU thresholds."""
    script = 'import sys; print("ready", flush=True); sys.stdin.read()'
    with subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert child.stdout is not None
            with selectors.DefaultSelector() as ready:
                ready.register(child.stdout, selectors.EVENT_READ)
                assert ready.select(timeout=_CHILD_TIMEOUT), (
                    "child did not become ready"
                )
            assert child.stdout.readline() == "ready\n"
            sampler = _Sampler(_FAST_INTERVAL)
            sampler.start()
            try:
                assert child.pid in {p.pid for p in sampler._tree()}  # noqa: SLF001
            finally:
                sampler.stop()
            assert not sampler.is_alive()
            assert sampler.cpu_samples
            assert sampler.peak_rss > 0
        finally:
            try:
                child.communicate(input="\n", timeout=_CHILD_TIMEOUT)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=_CHILD_TIMEOUT)

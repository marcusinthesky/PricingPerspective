"""Process resource sampling and per-stage telemetry persistence."""

from __future__ import annotations

import cProfile
import json
import os
import pstats
import socket
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, override

import psutil
from loguru import logger

from . import _xla_compiles
from ._jax import _activate, _deactivate
from ._logging import (
    _end_stage_span,
    _force_flush_otel,
    _set_stage_span_attributes,
    _start_stage_span,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.context import Token
    from opentelemetry.trace import Span

    _StageSpan = tuple[Span, Token[object]]

DEFAULT_SINK = Path("data/telemetry/runs.jsonl")
_DEFAULT_INTERVAL = 2.0
_TRUTHY_ENV_VALUES = frozenset({"1", "true"})


class _Sampler(threading.Thread):
    """Poll peak RSS and CPU use of the current process tree."""

    def __init__(self, interval: float) -> None:
        """Create a daemon sampler with the requested interval."""
        super().__init__(daemon=True, name="telemetry-sampler")
        self._interval = interval
        self._stop_event = threading.Event()
        self._process = psutil.Process()
        # ``cpu_percent(interval=None)`` measures against the previous call *on
        # that same object*, so a freshly constructed Process always reports
        # 0.0.  Children must therefore persist across samples or every
        # spawn-pool worker contributes nothing for its whole lifetime --
        # ``simulation.harness.parallel.parallel_map`` runs the Monte-Carlo
        # stages entirely in spawned children, which read as ~2% of one core
        # instead of the ~200-400% they actually burn.  Keyed by identity
        # (pid, create_time) rather than pid alone so a recycled pid starts a
        # fresh baseline instead of inheriting the dead process's.
        self._tracked: dict[tuple[int, float], psutil.Process] = {}
        self.peak_rss = 0
        self.cpu_samples: list[float] = []

    def _tree(self) -> list[psutil.Process]:
        try:
            found = [self._process, *self._process.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []
        live: dict[tuple[int, float], psutil.Process] = {}
        for process in found:
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                key = (process.pid, process.create_time())
                tracked = self._tracked.get(key)
                if tracked is None:
                    tracked = process
                    # Prime the baseline; this sample still reads 0.0 for it.
                    with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        tracked.cpu_percent(interval=None)
                live[key] = tracked
        # Drop exited workers so a long stage's pool churn cannot grow this
        # without bound.
        self._tracked = live
        return list(live.values())

    def _tree_rss(self) -> int:
        total = 0
        for process in self._tree():
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                total += process.memory_info().rss
        return total

    def _tree_cpu(self) -> float:
        total = 0.0
        for process in self._tree():
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                total += process.cpu_percent(interval=None)
        return total

    def sample(self) -> None:
        """Record one RSS peak and CPU reading."""
        self.peak_rss = max(self.peak_rss, self._tree_rss())
        self.cpu_samples.append(self._tree_cpu())

    @override
    def run(self) -> None:
        """Sample until the owning stage requests shutdown."""
        self._tree_cpu()
        self.peak_rss = self._tree_rss()
        while not self._stop_event.wait(self._interval):
            self.sample()

    def stop(self) -> None:
        """Stop sampling after capturing one final measurement."""
        self._stop_event.set()
        self.join(timeout=self._interval + 1.0)
        self.sample()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def begin(
    stage: str,
    *,
    app: str,
    sink: Path = DEFAULT_SINK,
    interval: float | None = None,
) -> Callable[[], None]:
    """Start resource sampling and return an idempotent-style finalizer."""
    if os.environ.get("TELEMETRY_DISABLE") == "1":
        return lambda: None

    resolved_interval = (
        interval
        if interval is not None
        else float(os.environ.get("TELEMETRY_INTERVAL", _DEFAULT_INTERVAL))
    )
    started_at = time.perf_counter()
    sampler = _Sampler(resolved_interval)
    sampler.start()
    diagnostics, diagnostics_token = _activate(enabled=_env_enabled("PP_TELEMETRY_JAX"))
    # Observed (not caller-asserted) XLA compilations, on by default: a stage
    # whose shapes churn recompiles per iteration, and opt-in observability
    # nobody opts into is no observability. No-op unless JAX is already
    # imported; `configure_jax_compilation_cache` attaches it otherwise.
    _xla_compiles.attach()
    compiles_at_start = _xla_compiles.snapshot()
    # Advertise a sidecar directory before any pool is built, so spawned workers
    # inherit it in their environment and can report their own compile counts
    # back. Without it a stage that fans all its work out reports ~0 compiles.
    _xla_compiles.open_worker_sink()
    profiler = cProfile.Profile() if _env_enabled("PP_TELEMETRY_PROFILE") else None
    if profiler is not None:
        profiler.enable()
    stage_span, span_token = _start_stage_span(app, stage)
    finished = False

    def finalize() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        try:
            sampler.stop()
            wall_seconds = time.perf_counter() - started_at
            cpu_samples = sampler.cpu_samples
            row: dict[str, object] = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "app": app,
                "stage": stage,
                "wall_s": round(wall_seconds, 3),
                "peak_rss_gb": round(sampler.peak_rss / 2**30, 4),
                "mean_cpu_pct": (
                    round(sum(cpu_samples) / len(cpu_samples), 1)
                    if cpu_samples
                    else 0.0
                ),
                "peak_cpu_pct": round(max(cpu_samples), 1) if cpu_samples else 0.0,
                "n_cpu": os.cpu_count(),
                "host": socket.gethostname(),
                "git_sha": _git_sha(),
            }
            row.update(
                _xla_compiles.summary(compiles_at_start, wall_seconds=wall_seconds)
            )
            if profiler is not None:
                row["profile_path"] = _persist_profile(profiler, sink, app, stage)
            if diagnostics.enabled:
                row.update(diagnostics.summary())
            _append_jsonl(sink, row)
            _set_stage_span_attributes(stage_span, row)
            logger.bind(app=app).info(
                "stage '{}' finished in {:.1f}s (peak {:.2f} GiB, mean CPU {:.0f}%)",
                stage,
                wall_seconds,
                row["peak_rss_gb"],
                row["mean_cpu_pct"],
            )
        finally:
            _deactivate(diagnostics_token)
            _end_stage_span(stage_span, span_token)
            _force_flush_otel()

    return finalize


def _persist_profile(
    profiler: cProfile.Profile, sink: Path, app: str, stage: str
) -> str:
    profiler.disable()
    profiles_dir = sink.parent / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    profile_path = profiles_dir / f"{app}__{stage}__{stamp}.pstats"
    pstats.Stats(profiler).dump_stats(profile_path)
    return str(profile_path)


def _append_jsonl(sink: Path, row: dict[str, object]) -> None:
    """Atomically append one short JSON record."""
    sink.parent.mkdir(parents=True, exist_ok=True)
    with sink.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


@cache
def _git_sha() -> str:
    git_dir = _find_git_dir(Path.cwd())
    if git_dir is None:
        return "unknown"
    try:
        return _read_git_sha(git_dir)
    except OSError:
        return "unknown"


def _read_git_sha(git_dir: Path) -> str:
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head[:7]
    reference = head.removeprefix("ref: ")
    common_dir = _common_git_dir(git_dir)
    loose_ref = common_dir / reference
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()[:7]
    packed_refs = (common_dir / "packed-refs").read_text(encoding="utf-8")
    for line in packed_refs.splitlines():
        sha, _, name = line.partition(" ")
        if name == reference:
            return sha[:7]
    return "unknown"


def _common_git_dir(git_dir: Path) -> Path:
    common_marker = git_dir / "commondir"
    if not common_marker.is_file():
        return git_dir
    common_dir = Path(common_marker.read_text(encoding="utf-8").strip())
    return common_dir if common_dir.is_absolute() else (git_dir / common_dir).resolve()


def _find_git_dir(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        marker = directory / ".git"
        if marker.is_dir():
            return marker
        if marker.is_file():
            try:
                declaration = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            prefix = "gitdir: "
            if declaration.startswith(prefix):
                git_dir = Path(declaration.removeprefix(prefix))
                return git_dir if git_dir.is_absolute() else directory / git_dir
    return None

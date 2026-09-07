"""Observed XLA compilation counting (default-on, parent plus spawned workers).

``_jax.JaxDiagnostics`` records what a *caller asserts* about a JAX operation --
including its ``cache="hit"|"miss"`` label, which nothing verifies -- and only
when ``PP_TELEMETRY_JAX`` is set. This module records what JAX itself reports.

JAX wraps every backend compilation in ``jax/_src/interpreters/pxla.py``'s
``_cached_compilation`` with the ``/jax/core/compile/backend_compile_duration``
event, published through the public ``jax.monitoring`` listener API. That
function is ``weakref_lru_cache``d on the jaxpr and its avals, so the event
fires exactly once per *in-process jit-cache miss*: a stage whose array shapes
churn (the ``design[active]`` slice in ``jcor.model._dyadic.inference``, which
made every bootstrap draw recompile) shows a compile count that scales with
iterations rather than with kernels. A persistent-cache hit still fires it --
correctly, since the executable still had to be fetched and loaded.

Counted precisely by event name, never by substring: the sibling
``jaxpr_trace_duration`` / ``jaxpr_to_mlir_module_duration`` events and the
``/jax/compilation_cache/*`` events also contain "compile", and the latter fire
only while the persistent cache is enabled.

Counts are reported per *process role*, not merged into one number:

* **This process** -- ``xla_compiles`` / ``xla_compile_s``.
* **Its spawned workers** -- ``xla_compiles_workers`` / ``xla_compile_s_workers``,
  summed from exit-time sidecars, with ``xla_worker_reports`` saying how many
  workers actually reported. The numerical stages fan out to ``spawn`` workers
  that cold-import their stage module (see ``_jax_cache``), and a worker's
  listener lives in its own interpreter; before this, a stage that ran
  *everything* in workers reported its compile churn as ~0.

Two limits remain, both visible in the emitted row rather than papered over:

* **Attached, not omniscient.** The listener attaches at the first ``begin()``
  with JAX already imported, or at ``configure_jax_compilation_cache()``, which
  every stage calls before its first jit. If JAX was imported during a stage but
  the listener never attached, the row reports ``null`` -- not a misleading ``0``.
* **Exit-time export.** A worker that is killed rather than shut down never runs
  its hook. ``xla_worker_reports`` is the count that arrived, so a truncated
  fan-out reads as a truncated fan-out and not as cheap compilation.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

# jax/_src/dispatch.py::BACKEND_COMPILE_EVENT. Private to JAX only in the sense
# that the *constant* lives under `_src`; the event name is what the public
# `jax.monitoring` listeners receive, and is asserted by this package's tests.
_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


@dataclass(frozen=True)
class CompileTotals:
    """Process-cumulative observed compilation totals at one instant."""

    compiles: int
    seconds: float


class _CompileCounter:
    """Accumulate JAX-reported backend compilations for this process."""

    def __init__(self) -> None:
        """Create an empty, unattached counter."""
        self._lock = threading.Lock()
        self._compiles = 0
        self._seconds = 0.0

    def observe(self, event: str, duration_secs: float, **metadata: str | int) -> None:
        """Count one JAX monitoring event (the registered listener callback).

        Matches ``jax.monitoring.EventDurationListenerWithMetadata``: JAX calls
        this on the thread that compiled, so the work is one comparison and a
        short locked increment.
        """
        del metadata  # JAX passes `fun_name`; the kernel identity is not needed.
        if event != _BACKEND_COMPILE_EVENT:
            return
        with self._lock:
            self._compiles += 1
            self._seconds += duration_secs

    def totals(self) -> CompileTotals:
        """Snapshot the running totals."""
        with self._lock:
            return CompileTotals(self._compiles, self._seconds)


_COUNTER = _CompileCounter()
_ATTACH_LOCK = threading.Lock()
_attached = False


def _jax_imported() -> bool:
    """Whether JAX is already imported here -- this module never imports it.

    ``find_spec`` distinguishes "not installed" from "not imported yet" without
    executing ``jax/__init__.py``; either way the answer is the same no-op, and
    the constraint documented in ``_jax_cache`` (importing ``dotell`` for
    logging alone must not pull in JAX) is preserved.
    """
    return "jax" in sys.modules and find_spec("jax") is not None


def attach() -> bool:
    """Register the compilation listener once per process; return whether it is on.

    A no-op returning ``False`` when JAX is absent or not yet imported. Callers
    that have just imported JAX (``configure_jax_compilation_cache``) should
    call this so the listener is live before the process's first jit.
    """
    global _attached  # noqa: PLW0603 - one process-global JAX listener registration.
    with _ATTACH_LOCK:
        if _attached:
            return True
        if not _jax_imported():
            return False
        # Deferred import: reached only once JAX is in `sys.modules` anyway.
        import jax.monitoring  # noqa: PLC0415

        jax.monitoring.register_event_duration_secs_listener(_COUNTER.observe)
        _attached = True
        return True


def attached() -> bool:
    """Whether the listener is registered in this process."""
    return _attached


# --- worker fan-out -------------------------------------------------------
#
# A spawned worker's listener counts only that worker's compiles, so a fanned-out
# stage under-reported by however much its pool did -- for the Monte-Carlo
# stages, which run *everything* in workers, that was the entire figure. Workers
# export their cumulative totals to a per-stage directory at interpreter exit and
# the parent sums them into its row.
#
# The directory travels in the environment rather than through a pool's
# ``initargs`` because spawned children inherit the parent's environment
# wholesale: that reaches both ``simulation.harness.parallel.parallel_map`` and
# the ``pipeline`` hedge pool (which passes no initializer at all) without either
# needing to know telemetry exists.
_WORKER_SINK_ENV = "PP_TELEMETRY_WORKER_COMPILE_DIR"
_owner_pid: int | None = None
_worker_export_registered = False


@dataclass(frozen=True)
class WorkerTotals:
    """Compilation totals summed across the workers that reported."""

    compiles: int
    seconds: float
    reports: int


def open_worker_sink() -> Path | None:
    """Parent-side: publish a directory for spawned workers to export totals to.

    Returns:
        The directory now advertised to children, or ``None`` if one could not
        be created (in which case worker counts are simply unavailable, exactly
        as before this existed).

    """
    global _owner_pid  # noqa: PLW0603 - one sink per process, mirroring `_attached`.
    try:
        directory = Path(tempfile.mkdtemp(prefix="dotell-workers-"))
    except OSError:
        return None
    os.environ[_WORKER_SINK_ENV] = str(directory)
    _owner_pid = os.getpid()
    return directory


def close_worker_sink() -> WorkerTotals:
    """Parent-side: sum every sidecar, then drop the sink.

    A worker killed outright (rather than shut down) never runs its exit hook,
    so ``reports`` is the count that actually arrived -- deliberately reported
    alongside the totals so a partial fan-out is visible as a partial fan-out
    rather than as a small number of compiles.
    """
    directory = os.environ.pop(_WORKER_SINK_ENV, None)
    if directory is None:
        return WorkerTotals(0, 0.0, 0)
    root = Path(directory)
    compiles = 0
    seconds = 0.0
    reports = 0
    if root.is_dir():
        for sidecar in sorted(root.glob("*.json")):
            with suppress(OSError, ValueError, TypeError):
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                compiles += int(payload["compiles"])
                seconds += float(payload["seconds"])
                reports += 1
    with suppress(OSError):
        shutil.rmtree(root, ignore_errors=True)
    return WorkerTotals(compiles, seconds, reports)


def _write_worker_totals(target: Path) -> None:
    """Exit hook: publish this worker's cumulative totals for the parent."""
    totals = _COUNTER.totals()
    with suppress(OSError):
        target.write_text(
            json.dumps({"compiles": totals.compiles, "seconds": totals.seconds}),
            encoding="utf-8",
        )


def register_worker_export() -> bool:
    """Worker-side: arrange for this process's totals to reach the parent.

    A no-op in the process that opened the sink -- it reports its own counts
    directly -- and when no stage published one. Comparing against the recorded
    owner pid rather than a "was I spawned" flag keeps this correct under
    ``fork`` too, where the child inherits this module's globals.

    Returns:
        Whether an exit hook is now registered for this process.

    """
    global _worker_export_registered  # noqa: PLW0603 - one hook per interpreter.
    directory = os.environ.get(_WORKER_SINK_ENV)
    if directory is None or _owner_pid == os.getpid():
        return False
    with _ATTACH_LOCK:
        if _worker_export_registered:
            return True
        atexit.register(_write_worker_totals, Path(directory) / f"{os.getpid()}.json")
        _worker_export_registered = True
        return True


def snapshot() -> CompileTotals:
    """Read the process-cumulative totals, for differencing over a stage."""
    return _COUNTER.totals()


def summary(start: CompileTotals, *, wall_seconds: float) -> dict[str, object]:
    """Summarize compilations observed since ``start`` for a telemetry row.

    Emitted by default (unlike the opt-in ``PP_TELEMETRY_JAX`` fields): a stage
    that recompiles per iteration is otherwise invisible. Values are ``None``
    when JAX ran in this process without the listener attached, so an
    un-instrumented stage cannot be misread as a clean zero.

    The ``*_workers`` fields carry what spawned workers reported, kept separate
    from this process's own counts rather than folded in: a fanned-out stage and
    a single-process one with the same total are different problems, and
    ``xla_worker_reports`` is what distinguishes "the pool compiled nothing"
    from "no worker reported".
    """
    workers = close_worker_sink()
    worker_fields: dict[str, object] = {
        "xla_compiles_workers": workers.compiles,
        "xla_compile_s_workers": round(workers.seconds, 6),
        "xla_worker_reports": workers.reports,
    }
    if not _attached:
        if _jax_imported():
            return {
                "xla_compiles": None,
                "xla_compile_s": None,
                "xla_compiles_per_wall_s": None,
                **worker_fields,
            }
        # JAX was never imported here, so nothing in *this* process can have
        # compiled -- but a worker still can have, so the pool's counts stand.
        return {
            "xla_compiles": 0,
            "xla_compile_s": 0.0,
            "xla_compiles_per_wall_s": 0.0,
            **worker_fields,
        }
    totals = _COUNTER.totals()
    compiles = totals.compiles - start.compiles
    return {
        "xla_compiles": compiles,
        "xla_compile_s": round(totals.seconds - start.seconds, 6),
        # Compiles per second of *stage wall time* -- the denominator is the
        # `wall_s` field of the same row, so the ratio is checkable. A healthy
        # stage compiles a fixed set of kernels once and trends to ~0 as it
        # runs; a shape-churning one holds a high rate for the whole stage.
        # Workers are excluded here: they compile concurrently, so folding them
        # into a per-wall-second rate would mix a rate with a parallel sum.
        "xla_compiles_per_wall_s": (
            round(compiles / wall_seconds, 3) if wall_seconds > 0 else 0.0
        ),
        **worker_fields,
    }

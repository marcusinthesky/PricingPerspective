"""Caller-owned JAX timing and outcome diagnostics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from contextvars import Token


class _JaxEvent(TypedDict):
    """Structured record for one explicitly observed JAX operation."""

    kind: str
    kernel: str
    signature: str
    failed: bool
    cache: NotRequired[Literal["hit", "miss"]]
    duration_s: NotRequired[float]


@dataclass(frozen=True)
class _JaxOperation:
    """Inputs that identify one caller-observed JAX operation."""

    kind: str
    kernel: str
    inputs: tuple[object, ...]
    static: Mapping[str, object] | None
    cache: Literal["hit", "miss"] | None


def _static_token(value: object) -> str:
    """Describe common static arguments without retaining their values."""
    match value:
        case None | bool() | int() | float() | str():
            return repr(value)
        case tuple() | list():
            return "[" + ",".join(_static_token(item) for item in value) + "]"
        case dict():
            items = sorted(value.items(), key=lambda item: repr(item[0]))
            return (
                "{"
                + ",".join(f"{key!r}:{_static_token(item)}" for key, item in items)
                + "}"
            )
        case _:
            return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _redacted_signature(
    inputs: tuple[object, ...], static: Mapping[str, object] | None
) -> str:
    """Return a stable shape, dtype, and static fingerprint without array values."""
    arrays = ",".join(
        f"{getattr(value, 'dtype', type(value).__name__)}["
        f"{'x'.join(map(str, getattr(value, 'shape', ())))}]"
        for value in inputs
    )
    statics = ",".join(
        f"{name}:{type(value).__name__}#"
        f"{sha256(_static_token(value).encode()).hexdigest()[:12]}"
        for name, value in sorted((static or {}).items())
    )
    return f"args[{arrays}];static[{statics}]"


class JaxDiagnostics:
    """Collect diagnostics only around operations explicitly identified by callers."""

    def __init__(self, *, enabled: bool) -> None:
        """Create an active or inert diagnostic collector."""
        self._enabled = enabled
        self.events: list[_JaxEvent] = []

    @property
    def enabled(self) -> bool:
        """Whether this collector will record events."""
        return self._enabled

    @contextmanager
    def compile(
        self,
        kernel: str,
        *inputs: object,
        static: Mapping[str, object] | None = None,
        cache: Literal["hit", "miss"] | None = None,
    ) -> Generator[None]:
        """Time an explicit JAX trace or compilation attempt."""
        with self._timed("compile", kernel, inputs, static, cache):
            yield

    @contextmanager
    def execution(
        self,
        kernel: str,
        *inputs: object,
        static: Mapping[str, object] | None = None,
    ) -> Generator[None]:
        """Time caller-defined dispatch or execution."""
        with self._timed("execution", kernel, inputs, static, None):
            yield

    @contextmanager
    def sync(
        self,
        kernel: str,
        *inputs: object,
        static: Mapping[str, object] | None = None,
    ) -> Generator[None]:
        """Time an explicit host synchronization."""
        with self._timed("sync", kernel, inputs, static, None):
            yield

    @contextmanager
    def _timed(
        self,
        kind: Literal["compile", "execution", "sync"],
        kernel: str,
        inputs: tuple[object, ...],
        static: Mapping[str, object] | None,
        cache: Literal["hit", "miss"] | None,
    ) -> Generator[None]:
        if not self._enabled:
            yield
            return
        start = time.perf_counter()
        operation = _JaxOperation(kind, kernel, inputs, static, cache)
        try:
            yield
        except BaseException:
            self._record(operation, failed=True, start=start)
            raise
        else:
            self._record(operation, failed=False, start=start)

    def fallback(
        self,
        kernel: str,
        *inputs: object,
        static: Mapping[str, object] | None = None,
    ) -> None:
        """Record one caller-confirmed use of a fallback path."""
        operation = _JaxOperation("fallback", kernel, inputs, static, None)
        self._record(operation, failed=False)

    def worker_failure(
        self,
        kernel: str,
        *inputs: object,
        static: Mapping[str, object] | None = None,
    ) -> None:
        """Record a worker failure observed by the parent process."""
        operation = _JaxOperation("worker_failure", kernel, inputs, static, None)
        self._record(operation, failed=True)

    def _record(
        self,
        operation: _JaxOperation,
        *,
        failed: bool,
        start: float | None = None,
    ) -> None:
        if not self._enabled:
            return
        event = _JaxEvent(
            kind=operation.kind,
            kernel=operation.kernel,
            signature=_redacted_signature(operation.inputs, operation.static),
            failed=failed,
        )
        if operation.cache is not None:
            event["cache"] = operation.cache
        if start is not None:
            event["duration_s"] = round(time.perf_counter() - start, 6)
        self.events.append(event)

    def summary(self) -> dict[str, object]:
        """Summarize explicitly recorded events for a telemetry row."""
        events = self.events
        return {
            "jax_compile_trace_s": _duration_sum(events, "compile"),
            "jax_execution_s": _duration_sum(events, "execution"),
            "jax_sync_s": _duration_sum(events, "sync"),
            "jax_compile_attempts": sum(event["kind"] == "compile" for event in events),
            "jax_compile_failures": sum(
                event["kind"] == "compile" and event["failed"] for event in events
            ),
            "jax_cache_hits": sum(event.get("cache") == "hit" for event in events),
            "jax_cache_misses": sum(event.get("cache") == "miss" for event in events),
            "jax_fallback_count": sum(event["kind"] == "fallback" for event in events),
            "jax_worker_failures": sum(
                event["kind"] == "worker_failure" for event in events
            ),
            "jax_events": events,
        }


def _duration_sum(events: list[_JaxEvent], kind: str) -> float:
    return round(
        sum(event.get("duration_s", 0.0) for event in events if event["kind"] == kind),
        6,
    )


_JAX_DIAGNOSTICS: ContextVar[JaxDiagnostics | None] = ContextVar(
    "dotell_jax_diagnostics", default=None
)


def _activate(*, enabled: bool) -> tuple[JaxDiagnostics, Token[JaxDiagnostics | None]]:
    diagnostics = JaxDiagnostics(enabled=enabled)
    return diagnostics, _JAX_DIAGNOSTICS.set(diagnostics)


def _deactivate(token: Token[JaxDiagnostics | None]) -> None:
    _JAX_DIAGNOSTICS.reset(token)


def jax_diagnostics() -> JaxDiagnostics:
    """Return this stage's diagnostic collector, or an inert collector."""
    return _JAX_DIAGNOSTICS.get() or JaxDiagnostics(enabled=False)

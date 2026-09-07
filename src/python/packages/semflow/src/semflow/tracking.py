"""Identity-preserving metadata for explicit function-level tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

_TRACK_ATTRIBUTE = "__semflow_track__"


@dataclass(frozen=True)
class TrackSpec:
    """Opt-in policy attached to a tracked entrypoint without wrapping it."""

    include: tuple[str, ...] = ()
    fallback: Literal["module"] = "module"


@overload
def track[CallableT: Callable[..., object]](function: CallableT, /) -> CallableT: ...


@overload
def track[CallableT: Callable[..., object]](
    function: None = None,
    /,
    *,
    include: tuple[str, ...] = (),
    fallback: Literal["module"] = "module",
) -> Callable[[CallableT], CallableT]: ...


def track[CallableT: Callable[..., object]](
    function: CallableT | None = None,
    /,
    *,
    include: tuple[str, ...] = (),
    fallback: Literal["module"] = "module",
) -> CallableT | Callable[[CallableT], CallableT]:
    """Attach immutable tracking metadata and return the original callable."""
    spec = TrackSpec(include=include, fallback=fallback)

    def decorate(callback: CallableT) -> CallableT:
        setattr(callback, _TRACK_ATTRIBUTE, spec)
        return callback

    return decorate(function) if function is not None else decorate


def tracking_spec(function: Callable[..., object]) -> TrackSpec | None:
    """Return metadata attached by :func:`track`, if present."""
    value = getattr(function, _TRACK_ATTRIBUTE, None)
    return value if isinstance(value, TrackSpec) else None

"""Shared deterministic artifact I/O for Paper 6 stages."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write(path: Path, value: object) -> None:
    """Commit finite JSON atomically with durable contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def read(path: Path) -> dict[str, Any]:
    """Read an object-shaped governed record."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = "JSON object required"
        raise TypeError(msg)
    return value

"""Atomic numerical arrays and exact artifact identities."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hotelling.artifacts import reject_symlinks
from hotelling.contracts import canonical_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


def repository_root() -> Path:
    """Locate the checkout without invoking or modifying Git."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    msg = "Paper 6 execution requires its source checkout"
    raise ValueError(msg)


def file_hash(path: Path) -> str:
    """Hash exact bytes, including uncommitted source edits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_hash(value: object) -> str:
    """Hash canonical finite JSON values."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_arrays(path: Path, arrays: Mapping[str, NDArray[np.float64]]) -> None:
    """Atomically persist finite numerical inputs and outputs without pickles."""
    if any(not np.isfinite(value).all() for value in arrays.values()):
        msg = "scientific array artifacts must be finite"
        raise ArithmeticError(msg)
    reject_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, allow_pickle=False, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_arrays(path: Path) -> dict[str, NDArray[np.float64]]:
    """Load only finite float64 arrays from a saved scientific run."""
    reject_symlinks(path)
    with np.load(path, allow_pickle=False) as saved:
        if any(saved[key].dtype != np.dtype(np.float64) for key in saved.files):
            msg = f"scientific arrays require float64 storage: {path.name}"
            raise ValueError(msg)
        result = {key: np.asarray(saved[key], dtype=np.float64) for key in saved.files}
    if any(not np.isfinite(value).all() for value in result.values()):
        msg = f"nonfinite scientific arrays: {path.name}"
        raise ValueError(msg)
    return result

"""Atomic Parquet and YAML writers with embedded backend provenance.

All writers use a write-to-temp-then-rename pattern so that concurrent
readers never observe a partially-written file.  Backend metadata
(:class:`~simulation.backend.BackendInfo`) is embedded in summary YAML
files so that results can be traced back to the JAX platform and dtype
used during computation.

Source: io boilerplate from all four donor scripts.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

if TYPE_CHECKING:
    import pandas as pd

    from simulation.backend import BackendInfo


def _backend_dict(backend_info: BackendInfo | None) -> dict[str, str | int]:
    """Convert :class:`BackendInfo` to a plain dict for YAML embedding.

    Args:
        backend_info: Backend snapshot or ``None``.

    Returns:
        Dict with ``platform``, ``dtype``, ``jax_version`` keys, or
        ``{"jax_version": jax.__version__, "note": "backend not recorded"}``
        when ``backend_info`` is ``None``.

    """
    if backend_info is None:
        return {"jax_version": jax.__version__, "note": "backend not recorded"}
    return {
        "platform": backend_info.platform,
        "dtype": backend_info.dtype,
        "device_count": backend_info.device_count,
        "jax_version": backend_info.jax_version,
    }


def write_parquet_atomic(
    df: pd.DataFrame,
    path: str | Path,
    backend_info: BackendInfo | None = None,
) -> Path:
    """Write a DataFrame to Parquet atomically (temp-file + rename).

    Parent directory is created if it does not exist.  The backend
    provenance is stored in Parquet metadata if ``backend_info`` is given.

    Args:
        df: DataFrame to write.
        path: Destination path (``*.parquet``).
        backend_info: Optional backend snapshot.  If provided, a
            ``"backend"`` column is **not** added to the DataFrame;
            instead, metadata is embedded in the Parquet schema.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if backend_info is not None:
        meta = {b"simulation.backend": str(_backend_dict(backend_info)).encode()}
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, tmp_path)
        Path(tmp_path).replace(path)
    except Exception:
        with suppress(OSError):
            Path(tmp_path).unlink()
        raise

    return path


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV atomically (temp-file + rename).

    Parent directory is created if it does not exist.  Atomic replacement
    keeps reruns working when a prior DVC checkout left the destination
    read-only.

    Args:
        df: DataFrame to write.
        path: Destination path (``*.csv``).

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        Path(tmp_path).replace(path)
    except Exception:
        with suppress(OSError):
            Path(tmp_path).unlink()
        raise

    return path


def write_yaml_summary(
    summary: dict[str, Any],
    path: str | Path,
    backend_info: BackendInfo | None = None,
) -> Path:
    """Write a summary dict to YAML atomically, embedding backend provenance.

    The ``"backend"`` key is added (or overwritten) in the top-level dict.

    Args:
        summary: Summary dictionary to serialise.
        path: Destination path (``*.yaml``).
        backend_info: Optional backend snapshot to embed.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    full_summary = {**summary, "backend": _backend_dict(backend_info)}

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".yaml")
    os.close(fd)
    try:
        with Path(tmp_path).open("w", encoding="utf-8") as f:
            yaml.dump(full_summary, f, default_flow_style=False, sort_keys=False)
        Path(tmp_path).replace(path)
    except Exception:
        with suppress(OSError):
            Path(tmp_path).unlink()
        raise

    return path

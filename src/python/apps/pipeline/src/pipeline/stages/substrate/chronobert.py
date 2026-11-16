"""Identity checks for published numeric embedding inputs and their cohorts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, NoReturn

import pandas as pd
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

_SAMPLE_ROWS = 128


def _inference_error(message: str) -> NoReturn:
    raise ValueError(message)


def _read_article_sample(path: Path, symbol: str) -> pa.Table:
    """Validate the entire input, then select its first 128 stable URL hashes.

    Dates are checked without replacing the original Arrow column or metadata.
    A malformed unselected row also fails rather than hiding a broken source.
    """
    table = pq.read_table(path)
    context = f"{symbol} ({path})"
    if "url_hash" not in table.column_names:
        _inference_error(f"{context} is missing url_hash")
    hashes = table["url_hash"].to_pylist()
    if any(not isinstance(value, str) or not value for value in hashes):
        _inference_error(f"{context} has missing/null or invalid url_hash values")
    if len(set(hashes)) != len(hashes):
        _inference_error(f"{context} has duplicate url_hash values")
    if "created_date" not in table.column_names:
        _inference_error(f"{context} is missing created_date")
    dates = pd.to_datetime(
        table["created_date"].to_pylist(), format="mixed", errors="coerce", utc=True
    )
    if dates.isna().any():
        _inference_error(f"{context} has null or unparsable created_date values")
    if table.num_rows < _SAMPLE_ROWS:
        _inference_error(f"{context} has {table.num_rows} rows; requires at least 128")
    # Stable Arrow sorting preserves every original column and its metadata.
    return table.sort_by([("url_hash", "ascending")]).slice(0, _SAMPLE_ROWS)


def _file_identity(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path),
        "kind": "parquet",
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _json_identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()

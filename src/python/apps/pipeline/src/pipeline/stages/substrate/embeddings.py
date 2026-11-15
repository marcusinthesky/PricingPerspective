"""Coverage and numerical validation of published, text-free embedding shards."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import typer
import yaml

from pipeline._kernels.arrays import l2_normalize_rows

if TYPE_CHECKING:
    from collections.abc import Mapping
logger = logging.getLogger(__name__)

MIN_COVERAGE_RATIO = 0.9

ENERGY_SANITY_TICKERS = 3

ENERGY_SANITY_ROWS = 64

MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class CoverageSpec:
    """Expected ticker universe, row bounds, and embedding width."""

    symbols: list[str]
    target_rows: int
    minimum_rows: int
    dimensions: int


class CoverageConfigurationError(ValueError):
    """Raised when a model's embedding coverage bounds are malformed."""

    @classmethod
    def invalid_bounds(
        cls, model_key: str, minimum_rows: int, target_rows: int
    ) -> CoverageConfigurationError:
        """Describe inverted or non-positive model-specific row bounds."""
        return cls(
            f"invalid coverage bounds for {model_key}: "
            f"minimum_rows={minimum_rows}, target_rows={target_rows}"
        )


def _resolve_coverage_bounds(
    params: Mapping[str, object], model_key: str
) -> tuple[int, int]:
    """Resolve model-specific row bounds, falling back to the corpus sample size."""
    embedding_cfg = _as_dict(params.get("embedding", {}))
    models = _as_dict(embedding_cfg.get("models", {}))
    model_cfg = _as_dict(models.get(model_key, {}))
    coverage_cfg = _as_dict(model_cfg.get("coverage", {}))

    target_rows = _as_int(coverage_cfg.get("target_rows", params["n_samples"]))
    minimum_rows = _as_int(
        coverage_cfg.get("minimum_rows", target_rows * MIN_COVERAGE_RATIO)
    )
    if minimum_rows <= 0 or target_rows < minimum_rows:
        raise CoverageConfigurationError.invalid_bounds(
            model_key, minimum_rows, target_rows
        )
    return target_rows, minimum_rows


@dataclass(frozen=True)
class CoverageScan:
    """Per-ticker validation rows and loaded arrays for downstream sanity checks."""

    rows: list[dict[str, object]]
    embeddings: dict[str, np.ndarray]
    missing: list[str]
    all_files_valid: bool


def _as_dict(obj: object) -> dict[str, object]:
    """Coerce a params value to a ``dict[str, object]`` (empty for non-mappings)."""
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {}


def _as_int(obj: object) -> int:
    """Coerce a params value (already-validated int/float from YAML) to ``int``."""
    if isinstance(obj, (int, float)):
        return int(obj)
    return int(str(obj))


def _mean_pairwise_dist(left: np.ndarray, right: np.ndarray) -> float:
    """Mean Euclidean distance between all pairs (ai, bj).

    Canonical JAX implementation:
    ``statadic.geometry.discrepancy.mmd.energy.energy_distance``.
    This numpy version is for lightweight validation only.
    """
    left_norms = np.sum(left**2, axis=1, keepdims=True)
    right_norms = np.sum(right**2, axis=1, keepdims=True)
    squared_distances = left_norms + right_norms.T - 2.0 * (left @ right.T)
    squared_distances = np.maximum(squared_distances, 0.0)
    return float(np.mean(np.sqrt(squared_distances)))


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    """E(X, Y) = 2·E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]."""
    exy = _mean_pairwise_dist(left, right)
    exx = _mean_pairwise_dist(left, left)
    eyy = _mean_pairwise_dist(right, right)
    return float(2.0 * exy - exx - eyy)


def _scan_embedding_coverage(embeddings_dir: Path, spec: CoverageSpec) -> CoverageScan:
    """Validate every expected ticker artifact and retain valid arrays."""
    found_files = {
        symbol: embeddings_dir / f"{symbol}.parquet"
        for symbol in spec.symbols
        if (embeddings_dir / f"{symbol}.parquet").exists()
    }
    missing = [symbol for symbol in spec.symbols if symbol not in found_files]
    typer.echo(
        f"Found {len(found_files)}/{len(spec.symbols)} files. "
        f"Missing: {missing or 'none'}"
    )
    validation_rows: list[dict[str, object]] = []
    embeddings: dict[str, np.ndarray] = {}
    all_files_valid = True
    for symbol in spec.symbols:
        row: dict[str, object] = {
            "symbol": symbol,
            "file_present": symbol in found_files,
        }
        if symbol not in found_files:
            row.update(
                {
                    "rows": None,
                    "dim": None,
                    "nan_count": None,
                    "zero_count": None,
                    "ok": False,
                }
            )
            validation_rows.append(row)
            all_files_valid = False
            continue

        table = pq.read_table(found_files[symbol])
        embedding = np.array(table.column("embedding").to_pylist(), dtype=np.float64)
        row_count = len(table)
        dimensions = embedding.shape[1] if embedding.ndim == MATRIX_DIMENSIONS else -1
        nan_count = int(np.sum(np.isnan(embedding)))
        zero_count = int(np.sum(np.count_nonzero(embedding, axis=1) == 0))
        count_ok = spec.minimum_rows <= row_count <= spec.target_rows
        short = count_ok and row_count < spec.target_rows
        valid = (
            count_ok
            and dimensions == spec.dimensions
            and nan_count == 0
            and zero_count == 0
        )
        row.update(
            {
                "rows": row_count,
                "dim": dimensions,
                "nan_count": nan_count,
                "zero_count": zero_count,
                "ok": valid,
                "short": short,
            }
        )
        if not valid:
            all_files_valid = False
            typer.echo(
                f"  FAIL {symbol}: rows={row_count} "
                f"(want {spec.minimum_rows}-{spec.target_rows}), "
                f"dim={dimensions} (want {spec.dimensions}), "
                f"nan={nan_count}, zero_rows={zero_count}"
            )
        elif short:
            typer.echo(
                f"  WARN {symbol}: rows={row_count} < "
                f"n_samples={spec.target_rows} (≥{spec.minimum_rows} floor OK) — "
                "subsample-limited"
            )
        validation_rows.append(row)
        embeddings[symbol] = embedding
    return CoverageScan(validation_rows, embeddings, missing, all_files_valid)


def _run_energy_sanity(
    expected_symbols: list[str], embeddings: dict[str, np.ndarray]
) -> tuple[bool, dict[str, object]]:
    """Compare split-half and cross-ticker energy distances on a small sample."""
    sanity_symbols = [symbol for symbol in expected_symbols if symbol in embeddings][
        :ENERGY_SANITY_TICKERS
    ]
    if len(sanity_symbols) != ENERGY_SANITY_TICKERS:
        typer.echo(
            f"WARNING: only {len(sanity_symbols)} tickers loaded, "
            "skipping energy sanity."
        )
        return False, {}

    typer.echo(f"Running energy-distance sanity on {sanity_symbols}...")
    arrays = {
        symbol: l2_normalize_rows(embeddings[symbol][:ENERGY_SANITY_ROWS])
        for symbol in sanity_symbols
    }
    self_distances: dict[str, float] = {}
    for symbol in sanity_symbols:
        values = arrays[symbol]
        half = len(values) // 2
        distance = _energy_distance(values[:half], values[half:])
        self_distances[symbol] = distance
        typer.echo(f"  d({symbol}_split1, {symbol}_split2) = {distance:.6f}")

    cross_distances: dict[str, float] = {}
    for index, left_symbol in enumerate(sanity_symbols):
        for right_symbol in sanity_symbols[index + 1 :]:
            distance = _energy_distance(arrays[left_symbol], arrays[right_symbol])
            cross_distances[f"{left_symbol}_vs_{right_symbol}"] = distance
            typer.echo(f"  d({left_symbol}, {right_symbol}) = {distance:.6f}")

    max_within = max(self_distances.values())
    min_cross = min(cross_distances.values())
    passed = max_within < min_cross
    typer.echo(
        f"  Max within-split: {max_within:.6f} vs min cross: {min_cross:.6f} "
        f"→ {'PASS' if passed else 'FAIL'}"
    )
    return passed, {
        "sanity_symbols": sanity_symbols,
        "within_split_distances": {
            key: round(value, 8) for key, value in self_distances.items()
        },
        "cross_ticker_distances": {
            key: round(value, 8) for key, value in cross_distances.items()
        },
        "max_within_split": round(max_within, 8),
        "min_cross_ticker": round(min_cross, 8),
    }


def _write_coverage_outputs(
    output_dir: Path,
    summary: dict[str, object],
    coverage_rows: list[dict[str, object]],
) -> None:
    """Write deterministic YAML summary and CSV ticker diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as stream:
        yaml.dump(summary, stream, default_flow_style=False, sort_keys=False)
    typer.echo(f"Written {summary_path}")

    coverage_path = output_dir / "coverage.csv"
    fieldnames = [
        "symbol",
        "file_present",
        "rows",
        "dim",
        "nan_count",
        "zero_count",
        "ok",
        "short",
    ]
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)
    typer.echo(f"Written {coverage_path}")


def validate_coverage(
    model_key: str,
    embeddings_dir: Path,
    output_dir: Path,
    params_file: Path,
) -> None:
    """Validate embedding coverage and run energy-distance sanity checks.

    Checks:
    - 100/100 parquet files present.
    - Per-ticker row count in the model-configured coverage bounds, falling back
      to [MIN_COVERAGE_RATIO * n_samples, n_samples] when unspecified (a bounded
      shortfall is a subsample-limited tail, not a failure).
    - Uniform embedding_dim matches params.
    - No NaN and no all-zero embedding vectors.
    - Energy-distance sanity on 3 tickers × 64 rows (split-half).
    """
    with Path(params_file).open(encoding="utf-8") as f:
        params = yaml.safe_load(f)

    expected_symbols: list[str] = params["symbols"]
    n_samples, min_rows = _resolve_coverage_bounds(params, model_key)
    expected_dim: int = params["embedding"]["models"][model_key]["dims"]
    spec = CoverageSpec(expected_symbols, n_samples, min_rows, expected_dim)

    typer.echo(
        f"Validating {model_key}: expecting {len(expected_symbols)} tickers, "
        f"{min_rows}-{n_samples} rows each, {expected_dim} dims."
    )

    scan = _scan_embedding_coverage(embeddings_dir, spec)
    energy_sanity_pass, energy_details = _run_energy_sanity(
        expected_symbols, scan.embeddings
    )
    coverage_count = len(expected_symbols) - len(scan.missing)
    final_pass = (
        scan.all_files_valid
        and energy_sanity_pass
        and coverage_count == len(expected_symbols)
    )

    summary: dict[str, object] = {
        "pass": final_pass,
        "model_key": model_key,
        "expected_tickers": len(expected_symbols),
        "found_tickers": coverage_count,
        "missing_tickers": scan.missing,
        "expected_rows_per_ticker": n_samples,
        "min_rows_per_ticker": min_rows,
        "expected_embedding_dim": expected_dim,
        "all_row_counts_ok": all(
            min_rows <= _as_int(r["rows"]) <= n_samples
            for r in scan.rows
            if r["rows"] is not None
        ),
        "short_tickers": {r["symbol"]: r["rows"] for r in scan.rows if r.get("short")},
        "all_dims_ok": all(
            r["dim"] == expected_dim for r in scan.rows if r["dim"] is not None
        ),
        "total_nan_vectors": sum(
            0 if r["nan_count"] is None else _as_int(r["nan_count"]) for r in scan.rows
        ),
        "total_zero_vectors": sum(
            0 if r["zero_count"] is None else _as_int(r["zero_count"])
            for r in scan.rows
        ),
        "energy_sanity_pass": energy_sanity_pass,
        "energy_sanity": energy_details,
    }

    _write_coverage_outputs(output_dir, summary, scan.rows)

    if final_pass:
        typer.echo(f"\nVALIDATION PASSED for {model_key}")
    else:
        typer.echo(f"\nVALIDATION FAILED for {model_key}", err=True)
        raise typer.Exit(code=1)


EMBEDDING_COLUMNS = frozenset(
    {
        "embedding",
        "url_hash",
        "symbol",
        "created_date",
        "embedding_dim",
        "text_length",
        "_missing_date",
    }
)
REQUIRED_COLUMNS = frozenset({"embedding", "url_hash", "created_date"})


def validate_embedding(path: Path, expected: dict[str, object]) -> None:
    """Reject undeclared columns, changed bytes, and a changed observation roster."""
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected["sha256"]:
        message = f"published embedding checksum differs: {path}"
        raise ValueError(message)
    metadata = pq.read_metadata(path)
    columns = set(pq.read_schema(path).names)
    if not REQUIRED_COLUMNS.issubset(columns) or not columns.issubset(
        EMBEDDING_COLUMNS
    ):
        message = f"embedding has undeclared or missing columns: {path}"
        raise ValueError(message)
    expected_columns = expected["columns"]
    if not isinstance(expected_columns, list) or any(
        not isinstance(name, str) for name in expected_columns
    ):
        message = "manifest columns must be a list of names"
        raise ValueError(message)
    if metadata.num_rows != expected["rows"] or columns != set(expected_columns):
        message = f"published embedding cohort differs: {path}"
        raise ValueError(message)


def validate_public_inputs(manifest_file: Path, root: Path) -> None:
    """Verify every published shard before executing the numeric pipeline."""
    document = json.loads(manifest_file.read_text(encoding="utf-8"))
    if document["version"] != 1 or not document["embeddings"]:
        message = "unsupported or empty public input manifest"
        raise ValueError(message)
    for relative, expected in document["embeddings"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            message = "unsafe path in public input manifest"
            raise ValueError(message)
        validate_embedding(root / path, expected)

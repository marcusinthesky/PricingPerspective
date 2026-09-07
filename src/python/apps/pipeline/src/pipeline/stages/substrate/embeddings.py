"""Embedding generation (OpenRouter) and coverage validation.

Provides:
- ``generate_embeddings`` — embed a single ticker's subsampled articles via
  OpenRouter (OpenAI-compatible embeddings API) with caching.
- ``validate_coverage`` — validate embedding coverage for a model: 100/100 files,
  row counts, uniform dims, no NaN/zero vectors, energy-distance sanity.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml
from tqdm import tqdm

from pipeline._kernels.arrays import l2_normalize_rows
from pipeline.io.settings import openrouter_settings

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pipeline.io.settings import OpenRouterSettings

logger = logging.getLogger(__name__)

# Coverage floor: ``subsample`` takes ``LIMIT n_samples`` rows, so a ticker with
# fewer eligible articles than ``n_samples`` (after the date-window + min-body
# filters) legitimately yields ``rows < n_samples`` — e.g. ORLY has 425 of 445 at
# n_samples=445. A bounded shortfall down to this fraction of ``n_samples`` is
# accepted (WARN, not FAIL); a gross shortfall, an oversize count, or a
# dim/NaN/zero problem still fails. Downstream energy/Mantel stages handle unequal
# per-ticker n (each pair records its own n_samples1/n_samples2).
MIN_COVERAGE_RATIO = 0.9
ENERGY_SANITY_TICKERS = 3
ENERGY_SANITY_ROWS = 64
MATRIX_DIMENSIONS = 2
HTTP_OK = 200
MAX_ERROR_EXCERPT_CHARS = 500
RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

# Published-shard column policy. The written shards are redistributed (see
# REPLICATION.md); their ``subsampled_headlines`` input is a licensed news corpus
# that is not, so the writer must never copy article text into an output. This is
# an allow-list rather than a deny-list on purpose: ``subsample`` selects ``*``
# from the normalized corpus, so a new upstream column would otherwise be passed
# through silently. ``ettax.semantic``/``ettax.producer`` carry the same policy
# for the vintage shards; keep the three in step. Nothing here is load-bearing
# for results — consumers read only {embedding, created_date, url_hash} via
# ``pipeline.io.typed_providers``.
SHARD_BASE_COLUMNS = frozenset(
    {"url_hash", "symbol", "embedding", "text_length", "embedding_dim"}
)
SHARD_PASSTHROUGH_COLUMNS = frozenset(
    {
        "id",
        "primarysymbol",
        "body_len",
        "author",
        "created_date",
        "url",
        "primarytopic",
        "related_symbols",
        "date_accessed",
        "_missing_date",
    }
)


def shard_passthrough_columns(column_names: Iterable[str]) -> list[str]:
    """Retained passthrough columns for a written shard, in input order.

    Drops the base fields the writer sets itself and every column outside
    ``SHARD_PASSTHROUGH_COLUMNS`` — notably ``title``, ``body`` and
    ``publisher``.
    """
    return [
        column
        for column in column_names
        if column in SHARD_PASSTHROUGH_COLUMNS and column not in SHARD_BASE_COLUMNS
    ]


@dataclass(frozen=True)
class ModelConfig:
    """Validated embedding-model and request settings."""

    model_id: str
    dimensions: int
    base_url: str
    batch_size: int
    concurrency: int
    max_chars: int


@dataclass(frozen=True)
class EmbeddingRequestConfig:
    """Authentication and retry policy for one embedding provider."""

    model: ModelConfig
    api_key: str
    max_retries: int = 6


@dataclass(frozen=True)
class EmbeddingRunConfig:
    """Model, cache, and ticker context for one asynchronous embedding run."""

    symbol: str
    model_key: str
    request: EmbeddingRequestConfig
    cache_dir: Path


@dataclass(frozen=True)
class GenerateEmbeddingOptions:
    """Output and model-selection options for one ticker embedding stage."""

    model_key: str
    output_file: Path
    params_file: Path
    cache_dir: Path
    skip_existing: bool = False


@dataclass(frozen=True)
class CacheMiss:
    """One article that requires a provider request before caching."""

    url_hash: str
    text: str
    content_sha256: str


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


class EmbeddingRequestError(RuntimeError):
    """Raised when the embedding provider cannot satisfy a request."""

    @classmethod
    def exhausted(
        cls, attempts: int, error: httpx.RequestError
    ) -> EmbeddingRequestError:
        """Describe an exhausted transport retry budget."""
        return cls(f"request failed after {attempts} attempts: {error}")

    @classmethod
    def missing_data(cls, model_id: str, detail: object) -> EmbeddingRequestError:
        """Describe a successful envelope without embedding data."""
        excerpt = str(detail)[:MAX_ERROR_EXCERPT_CHARS]
        return cls(
            f"OpenRouter returned 200 without data for model {model_id}: {excerpt}"
        )

    @classmethod
    def api_response(cls, status: int, text: str) -> EmbeddingRequestError:
        """Describe a non-retryable provider response."""
        return cls(f"OpenRouter API error {status}: {text[:MAX_ERROR_EXCERPT_CHARS]}")

    @classmethod
    def retry_loop_exhausted(cls, attempts: int) -> EmbeddingRequestError:
        """Describe an unexpectedly exhausted retry loop."""
        return cls(f"embedding request failed after {attempts} retries")


class EmbeddingDimensionError(RuntimeError):
    """Raised when a provider vector violates the configured model width."""

    def __init__(self, observed: int, expected: int, model_id: str) -> None:
        """Describe the observed and configured dimensions."""
        super().__init__(
            f"dimension mismatch: got {observed}, expected {expected} "
            f"for model {model_id}"
        )


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def resolve_api_key(settings: OpenRouterSettings | None = None) -> str:
    """Return OPENROUTER_API_KEY or exit with a clear message."""
    secret = (settings or openrouter_settings()).api_key
    if secret is None:
        typer.echo(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            "  Set it in your environment or in a .env file at the project root.\n"
            "  Never write the key to any tracked file.",
            err=True,
        )
        raise typer.Exit(code=1)
    return secret.get_secret_value()


# ---------------------------------------------------------------------------
# params.yaml helpers
# ---------------------------------------------------------------------------


def load_params(params_path: Path) -> dict[str, object]:
    """Load the embedding-stage YAML parameters from disk."""
    with Path(params_path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def resolve_model_config(params: dict[str, object], model_key: str) -> ModelConfig:
    """Return validated request settings for ``model_key``."""
    embedding_cfg = _as_dict(params.get("embedding", {}))
    models = _as_dict(embedding_cfg.get("models", {}))
    if model_key not in models:
        available = list(models.keys())
        typer.echo(
            f"ERROR: model key '{model_key}' not in params.yaml embedding.models.\n"
            f"  Available: {available}",
            err=True,
        )
        raise typer.Exit(code=1)
    cfg = _as_dict(models[model_key])
    return ModelConfig(
        model_id=str(cfg["id"]),
        dimensions=_as_int(cfg["dims"]),
        base_url=str(embedding_cfg.get("base_url", "https://openrouter.ai/api/v1")),
        batch_size=_as_int(embedding_cfg.get("batch_size", 32)),
        concurrency=_as_int(embedding_cfg.get("concurrency", 4)),
        # Per-model max_chars overrides the global default: some encoders (e.g.
        # bge-large-en-v1.5) have a 512-token context window (~1600 chars) far
        # below the global 8000-char cap used for long-context Qwen/bge-m3 models.
        max_chars=_as_int(cfg.get("max_chars", embedding_cfg.get("max_chars", 8000))),
    )


# ---------------------------------------------------------------------------
# Content hash (cache key)
# ---------------------------------------------------------------------------


def content_sha256(model_id: str, text: str) -> str:
    """Return a stable cache key scoped to the model and source text."""
    h = hashlib.sha256()
    h.update((model_id + "\x00" + text).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, model_key: str, symbol: str, url_hash: str) -> Path:
    return cache_dir / model_key / symbol / f"{url_hash}.json"


def _load_from_cache(path: Path, expected_sha: str) -> list[float] | None:
    if not path.exists():
        return None
    try:
        with Path(path).open(encoding="utf-8") as f:
            entry = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable embedding cache %s: %s", path, exc)
        return None

    if not isinstance(entry, dict):
        logger.warning("Ignoring embedding cache with invalid object shape: %s", path)
        return None
    if entry.get("content_sha256") != expected_sha:
        return None

    embedding = entry.get("embedding")
    if (
        not isinstance(embedding, list)
        or not embedding
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in embedding
        )
    ):
        logger.warning("Ignoring embedding cache with invalid vector: %s", path)
        return None
    return [float(value) for value in embedding]


def _save_to_cache(path: Path, sha: str, model_id: str, embedding: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(
            {"content_sha256": sha, "model_id": model_id, "embedding": embedding}, f
        )


# ---------------------------------------------------------------------------
# OpenRouter embedding call (with retry)
# ---------------------------------------------------------------------------


async def _embed_batch(
    client: httpx.AsyncClient,
    texts: list[str],
    config: EmbeddingRequestConfig,
) -> list[list[float]]:
    """POST to /embeddings, return list of embedding vectors."""
    url = config.model.base_url.rstrip("/") + "/embeddings"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/pricing-perspective",
        "X-Title": "pricing-perspective",
    }
    payload = {"model": config.model.model_id, "input": texts}

    delay = 1.0
    for attempt in range(config.max_retries):
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
        except httpx.RequestError as exc:
            if attempt == config.max_retries - 1:
                raise EmbeddingRequestError.exhausted(config.max_retries, exc) from exc
            await asyncio.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == HTTP_OK:
            data = resp.json()
            # OpenRouter can wrap an upstream provider failure (e.g. a 400 for
            # exceeding a model's context window) inside a 200 envelope with an
            # 'error' key and no 'data'. Surface it instead of raising KeyError.
            if "data" not in data:
                err = data.get("error", data)
                raise EmbeddingRequestError.missing_data(config.model.model_id, err)
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

        if resp.status_code in RETRYABLE_HTTP_STATUSES:
            retry_after = float(resp.headers.get("Retry-After", delay))
            if attempt < config.max_retries - 1:
                await asyncio.sleep(retry_after)
                delay = max(delay * 2, retry_after)
                continue

        raise EmbeddingRequestError.api_response(resp.status_code, resp.text)

    raise EmbeddingRequestError.retry_loop_exhausted(config.max_retries)


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------


def _prepare_embedding_items(
    df: pa.Table, symbol: str, model: ModelConfig
) -> tuple[list[str], dict[str, list[object]], list[tuple[str, str, str]]]:
    """Build stable text/hash items while retaining passthrough columns."""
    columns = list(df.column_names)
    has_body = "body" in columns
    if not has_body:
        typer.echo(
            f"WARNING: 'body' column absent — embedding title-only for {symbol}.",
            err=True,
        )
    rows: dict[str, list[object]] = df.to_pydict()
    article_count = len(rows["url_hash"])
    titles = rows.get("title", [""] * article_count)
    bodies = (
        rows.get("body", [None] * article_count) if has_body else [None] * article_count
    )
    items: list[tuple[str, str, str]] = []
    for index in range(article_count):
        title = str(titles[index] or "")
        body = str(bodies[index] or "") if has_body else ""
        text = (
            (title + "\n\n" + body)[: model.max_chars]
            if body
            else title[: model.max_chars]
        )
        url_hash = str(rows["url_hash"][index])
        items.append((url_hash, text, content_sha256(model.model_id, text)))
    return shard_passthrough_columns(columns), rows, items


def _partition_cached_items(
    items: list[tuple[str, str, str]], config: EmbeddingRunConfig
) -> tuple[dict[str, list[float]], list[CacheMiss], int]:
    """Load valid cached vectors and return the remaining provider work."""
    embeddings: dict[str, list[float]] = {}
    misses: list[CacheMiss] = []
    for url_hash, text, sha in items:
        cache_path = _cache_path(
            config.cache_dir, config.model_key, config.symbol, url_hash
        )
        cached = _load_from_cache(cache_path, sha)
        if cached is None:
            misses.append(CacheMiss(url_hash, text, sha))
        else:
            embeddings[url_hash] = cached
    return embeddings, misses, len(items) - len(misses)


async def _populate_embedding_misses(
    misses: list[CacheMiss],
    embeddings: dict[str, list[float]],
    config: EmbeddingRunConfig,
) -> int:
    """Request cache misses concurrently and persist every validated vector."""
    model = config.request.model
    batches = [
        misses[index : index + model.batch_size]
        for index in range(0, len(misses), model.batch_size)
    ]
    semaphore = asyncio.Semaphore(model.concurrency)

    async def process_batch(batch: list[CacheMiss], client: httpx.AsyncClient) -> None:
        async with semaphore:
            vectors = await _embed_batch(
                client, [miss.text for miss in batch], config.request
            )
            for miss, vector in zip(batch, vectors, strict=False):
                if len(vector) != model.dimensions:
                    raise EmbeddingDimensionError(
                        len(vector), model.dimensions, model.model_id
                    )
                embeddings[miss.url_hash] = vector
                cache_path = _cache_path(
                    config.cache_dir,
                    config.model_key,
                    config.symbol,
                    miss.url_hash,
                )
                _save_to_cache(cache_path, miss.content_sha256, model.model_id, vector)

    async with httpx.AsyncClient() as client:
        tasks = [process_batch(batch, client) for batch in batches]
        for coroutine in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Batches {config.symbol}",
        ):
            await coroutine
    return len(batches)


async def _run_embeddings(
    df: pa.Table,
    config: EmbeddingRunConfig,
) -> list[dict[str, object]]:
    """Embed all rows in df, using cache where available."""
    model = config.request.model
    symbol = config.symbol

    columns, rows, items = _prepare_embedding_items(df, symbol, model)
    n = len(rows["url_hash"])

    total_chars = sum(len(t) for _, t, _ in items)
    est_tokens = total_chars // 4
    typer.echo(
        f"Cost estimate: ~{est_tokens:,} tokens for {n} articles. "
        f"(price discoverable only at runtime; typically $0.001-$0.01 per M tokens)"
    )

    embeddings, misses, cache_hits = _partition_cached_items(items, config)

    typer.echo(f"Cache: {cache_hits} hits, {len(misses)} misses out of {n} articles.")

    if misses:
        total_requests = await _populate_embedding_misses(misses, embeddings, config)
        typer.echo(
            f"Summary: {total_requests} requests, {cache_hits} cache hits, "
            f"~{len(misses) * (model.batch_size // 2)} tokens sent (estimate)."
        )

    results = []
    for i in range(n):
        url_hash, text, _sha = items[i]
        if url_hash not in embeddings:
            typer.echo(f"WARNING: no embedding for {url_hash}, skipping.", err=True)
            continue
        vec = embeddings[url_hash]
        record: dict[str, object] = {
            "url_hash": url_hash,
            "symbol": symbol,
            "embedding": vec,
            "text_length": len(text),
            "embedding_dim": len(vec),
        }
        for col in columns:
            if col not in record:
                record[col] = rows[col][i]
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def _build_parquet_schema(
    results: list[dict[str, object]], passthrough_cols: list[str]
) -> pa.Schema:
    base_fields = [
        pa.field("url_hash", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("embedding", pa.list_(pa.float64())),
        pa.field("text_length", pa.int64()),
        pa.field("embedding_dim", pa.int64()),
    ]
    skip = SHARD_BASE_COLUMNS
    extra_fields = []
    if results:
        first = results[0]
        for col in passthrough_cols:
            if col in skip:
                continue
            val = first.get(col)
            if isinstance(val, str) or val is None:
                extra_fields.append(pa.field(col, pa.string()))
            elif isinstance(val, (bool, np.bool_)):
                extra_fields.append(pa.field(col, pa.bool_()))
            elif isinstance(val, (int, np.integer)):
                extra_fields.append(pa.field(col, pa.int64()))
            elif isinstance(val, (float, np.floating)):
                extra_fields.append(pa.field(col, pa.float64()))
            else:
                extra_fields.append(pa.field(col, pa.string()))
    return pa.schema(base_fields + extra_fields)


def _write_parquet_atomic(
    results: list[dict[str, object]], output_file: Path, passthrough_cols: list[str]
) -> None:
    schema = _build_parquet_schema(results, passthrough_cols)
    skip = SHARD_BASE_COLUMNS

    arrays: dict[str, list[object]] = {f.name: [] for f in schema}
    for r in results:
        arrays["url_hash"].append(r["url_hash"])
        arrays["symbol"].append(r["symbol"])
        arrays["embedding"].append(r["embedding"])
        arrays["text_length"].append(r["text_length"])
        arrays["embedding_dim"].append(r["embedding_dim"])
        for f in schema:
            if f.name in skip:
                continue
            val = r.get(f.name)
            arrays[f.name].append(
                str(val)
                if val is not None
                and not isinstance(val, (int, float, np.integer, np.floating))
                else val
            )

    pa_arrays = []
    for f in schema:
        col_data = arrays[f.name]
        if f.type == pa.list_(pa.float64()):
            pa_arrays.append(pa.array(col_data, type=pa.list_(pa.float64())))
        else:
            pa_arrays.append(pa.array(col_data, type=f.type))

    table = pa.table(
        dict(zip([f.name for f in schema], pa_arrays, strict=True)), schema=schema
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.rename(output_file)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_embeddings(
    symbol: str,
    parquet_path: Path,
    options: GenerateEmbeddingOptions,
) -> None:
    r"""Generate embeddings via OpenRouter for a single ticker subsample parquet.

    Text = title + '\\n\\n' + body[:max_chars]; falls back to title-only if body absent.
    Embeddings are NOT L2-normalised (energy_tests normalises at runtime).
    """
    if options.skip_existing and options.output_file.exists():
        typer.echo(f"Output already exists, skipping: {options.output_file}")
        return

    api_key = resolve_api_key()

    params = load_params(options.params_file)
    model_cfg = resolve_model_config(params, options.model_key)

    typer.echo(
        f"Model: {options.model_key} → {model_cfg.model_id} "
        f"({model_cfg.dimensions} dims)\n"
        f"Symbol: {symbol}  Parquet: {parquet_path}"
    )

    df = pq.read_table(parquet_path)
    typer.echo(f"Loaded {len(df)} rows from {parquet_path}")

    # ``subsample`` writes rows ORDER BY url_hash, so the model's configured
    # coverage target selects a deterministic md5-ordered prefix -- the same
    # prefix the frozen artifacts already hold. Without this the panel width is
    # whatever ``n_samples`` happened to be the first time a ticker was
    # embedded, which is exactly how the m=128 artifacts and the 445-row
    # subsample came to disagree. Truncating here keeps a roster addition on the
    # same geometry as the firms embedded before it.
    target_rows, _minimum_rows = _resolve_coverage_bounds(params, options.model_key)
    if len(df) > target_rows:
        typer.echo(f"Truncating to the configured {target_rows}-row coverage target")
        df = df.slice(0, target_rows)

    passthrough_cols = shard_passthrough_columns(df.column_names)

    results = asyncio.run(
        _run_embeddings(
            df=df,
            config=EmbeddingRunConfig(
                symbol=symbol,
                model_key=options.model_key,
                request=EmbeddingRequestConfig(model=model_cfg, api_key=api_key),
                cache_dir=options.cache_dir,
            ),
        )
    )

    if not results:
        typer.echo("ERROR: No embeddings produced.", err=True)
        raise typer.Exit(code=1)

    _write_parquet_atomic(results, options.output_file, passthrough_cols)
    typer.echo(f"Written {len(results)} rows to {options.output_file}")


# ---------------------------------------------------------------------------
# Energy distance helpers (for validation)
# ---------------------------------------------------------------------------


def _mean_pairwise_dist(left: np.ndarray, right: np.ndarray) -> float:
    """Mean Euclidean distance between all pairs (ai, bj).

    Canonical JAX implementation: ``jcor.discrepancy.energy.energy_distance``.
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


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------


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

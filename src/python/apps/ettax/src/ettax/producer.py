# SPDX-License-Identifier: Apache-2.0
"""Materialize EttaX document embeddings in the shared parquet contract."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from flax import nnx
from tokenizers import Tokenizer

from ettax.config import ExperimentConfig
from ettax.model import Ettax
from ettax.pooling import (
    aggregate_article_chunks,
    encode_and_pool,
    tokenize_article_chunks,
)
from ettax.semantic import load_articles
from ettax.train import restore_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ettax.semantic import Article
    from ettax.vintage import VintageConfig


@dataclass(frozen=True, slots=True)
class EmbeddingRunOptions:
    """Inputs and output root for one vintage embedding materialization."""

    articles_dir: Path
    output_dir: Path
    batch_size: int = 128
    require_gpu: bool = True

    def __post_init__(self) -> None:
        """Reject invalid producer settings."""
        if self.batch_size < 1:
            raise ValueError("embedding batch_size must be positive")


def embed_vintage(
    config: VintageConfig,
    arm_name: str,
    options: EmbeddingRunOptions,
) -> dict[str, object]:
    """Embed every article shard for one frozen vintage."""
    if options.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"refusing embedding on backend={jax.default_backend()!r}; "
            "set JAX_PLATFORMS=cuda"
        )
    experiment = ExperimentConfig.from_toml(config.experiment_config)
    tokenizer_path = config.tokenizer_path
    checkpoint = config.checkpoint_dir(config.arm(arm_name)) / "model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing model-only checkpoint: {checkpoint}")

    articles = load_articles(options.articles_dir)
    by_symbol: dict[str, list[Article]] = defaultdict(list)
    seen: set[str] = set()
    for article in articles:
        if article.url_hash in seen:
            raise ValueError(f"duplicate article url_hash: {article.url_hash}")
        seen.add(article.url_hash)
        by_symbol[article.symbol].append(article)
    if not by_symbol:
        raise ValueError(f"no articles under {options.articles_dir}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = Ettax(experiment.model, rngs=nnx.Rngs(experiment.train.seed))
    restore_checkpoint(checkpoint, model)
    options.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: dict[str, dict[str, object]] = {}
    total_rows = 0
    special_ids = (
        experiment.model.pad_id,
        experiment.model.doc_id,
        experiment.model.eos_id,
        experiment.model.unk_id,
    )
    for symbol, symbol_articles in sorted(by_symbol.items()):
        texts = [f"{article.title}\n\n{article.body}" for article in symbol_articles]
        tokens, offsets, raw_lengths = tokenize_article_chunks(
            tokenizer,
            texts,
            sequence_length=experiment.data.sequence_length,
            pad_id=experiment.model.pad_id,
            doc_id=experiment.model.doc_id,
            eos_id=experiment.model.eos_id,
        )
        pooled = encode_and_pool(
            model,
            tokens,
            batch_size=options.batch_size,
            policies=(config.pooling_policy,),
            special_ids=special_ids,
        )[config.pooling_policy]
        vectors = aggregate_article_chunks(
            pooled, offsets, policy=config.article_policy
        )
        tokenization = _tokenization_diagnostics(
            tokens,
            texts,
            raw_lengths,
            content_limit=experiment.data.sequence_length - 2,
            article_policy=config.article_policy,
            unknown_id=experiment.model.unk_id,
        )
        if vectors.shape[0] != len(symbol_articles) or not np.all(np.isfinite(vectors)):
            raise ValueError(f"invalid embedding matrix for {symbol}")
        rows: dict[str, object] = {
            "url_hash": [article.url_hash for article in symbol_articles],
            "symbol": [symbol] * len(symbol_articles),
            "embedding": vectors.astype(np.float64).tolist(),
            "text_length": [len(text) for text in texts],
            "embedding_dim": [int(vectors.shape[1])] * len(symbol_articles),
        }
        # ``extras`` is already restricted to ``semantic.SHARD_PASSTHROUGH_FIELDS``
        # at its only construction site (``semantic.load_articles``), so article
        # text never reaches a written shard.
        for column in symbol_articles[0].extras:
            rows[column] = [
                _coerce_passthrough(article.extras.get(column))
                for article in symbol_articles
            ]
        _write_parquet_atomic(rows, options.output_dir / f"{symbol}.parquet")
        diagnostics[symbol] = {
            "rows": len(symbol_articles),
            "tokenization": tokenization,
        }
        total_rows += len(symbol_articles)

    summary: dict[str, object] = {
        "arm": arm_name,
        "article_policy": config.article_policy,
        "backend": jax.default_backend(),
        "checkpoint": str(checkpoint),
        "checkpoint_layout": config.checkpoint_layout,
        "embedding_dimensions": int(experiment.model.width),
        "output_dir": str(options.output_dir),
        "pooling_policy": config.pooling_policy,
        "rows": total_rows,
        "run_id": config.run_id,
        "schema_version": 2,
        "sequence_length": int(experiment.data.sequence_length),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": _file_sha256(tokenizer_path),
        "symbols": diagnostics,
    }
    _write_json_atomic(options.output_dir / "_summary.json", summary)
    return summary


def _tokenization_diagnostics(
    tokens: NDArray[np.int32],
    texts: list[str],
    raw_lengths: list[int],
    *,
    content_limit: int,
    article_policy: str,
    unknown_id: int,
) -> dict[str, float | int]:
    """Report chunking, truncation, and unknown-token rates for one symbol.

    Truncation is reported against the ``first`` window because that is the only
    policy under which content is discarded; ``chunks`` reads every window, so
    its coverage is one by construction.
    """
    lengths = np.asarray(raw_lengths, dtype=np.int64)
    total_tokens = int(np.sum(lengths))
    unknown = int(np.sum(tokens == unknown_id))
    visible = (
        total_tokens
        if article_policy == "chunks"
        else int(np.sum(np.minimum(lengths, content_limit)))
    )
    truncated = int(np.sum(lengths > content_limit))
    return {
        "article_coverage": visible / total_tokens if total_tokens else 1.0,
        "chunk_rows": int(tokens.shape[0]),
        "content_token_limit": content_limit,
        "duplicate_input_text_rows": len(texts) - len(set(texts)),
        "max_content_tokens": int(np.max(lengths)) if len(lengths) else 0,
        "mean_chunks_per_article": float(tokens.shape[0] / len(raw_lengths)),
        "mean_content_tokens": float(np.mean(lengths)) if len(lengths) else 0.0,
        "truncated": truncated,
        "truncation_rate": truncated / len(raw_lengths) if raw_lengths else 0.0,
        "unknown_token_count": unknown,
        "unknown_token_rate": unknown / total_tokens if total_tokens else 0.0,
    }


def _coerce_passthrough(value: object) -> object:
    """Narrow one passthrough value to a Parquet-representable scalar.

    Mirrors ``pipeline.stages.substrate.embeddings._write_parquet_atomic``:
    numerics (including ``bool``) survive as-is and everything else is
    stringified, so ``date32`` becomes ``'2021-07-20'``, ``timestamp`` becomes
    ``'2022-12-31 13:48:53'`` and ``list<string>`` becomes its Python repr.
    """
    if value is None or isinstance(value, (int, float, np.integer, np.floating)):
        return value
    return str(value)


def _passthrough_type(values: list[object]) -> pa.DataType:
    """Infer one passthrough column's Arrow type from its first known value.

    Applies the same ladder as the pipeline writer's schema builder. ``bool`` is
    tested before ``int`` because it is a subclass of it, and an all-null column
    degrades to ``string`` exactly as the pipeline does for a ``None`` sample.
    """
    for value in values:
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            return pa.bool_()
        if isinstance(value, (int, np.integer)):
            return pa.int64()
        if isinstance(value, (float, np.floating)):
            return pa.float64()
        return pa.string()
    return pa.string()


def _write_parquet_atomic(rows: dict[str, object], path: Path) -> None:
    """Write one embedding shard atomically using the shared pipeline schema."""
    base = {
        "url_hash": pa.string(),
        "symbol": pa.string(),
        "embedding": pa.list_(pa.float64()),
        "text_length": pa.int64(),
        "embedding_dim": pa.int64(),
    }
    fields = [pa.field(name, dtype) for name, dtype in base.items()]
    fields.extend(
        pa.field(name, _passthrough_type(cast("list[object]", values)))
        for name, values in rows.items()
        if name not in base
    )
    schema = pa.schema(fields)
    table = pa.table(rows, schema=schema)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    """Hash a tokenizer or other immutable input in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, values: object) -> None:
    """Write a deterministic JSON sidecar atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

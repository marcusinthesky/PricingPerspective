# SPDX-License-Identifier: Apache-2.0
"""Pre-adjudication semantic smoke test over existing Nasdaq articles."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from flax import nnx
from tokenizers import Tokenizer

from ettax.config import ExperimentConfig
from ettax.evaluation import (
    compare_geometries,
    embedding_diagnostics,
    mantel_tests,
    retrieval_metrics,
)
from ettax.inference import tokenize_documents
from ettax.model import Ettax
from ettax.pooling import (
    aggregate_article_chunks,
    encode_and_pool,
    tokenize_article_chunks,
)
from ettax.train import restore_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import date, datetime
    from pathlib import Path

    from numpy.typing import NDArray

    from ettax.vintage import VintageConfig


#: Columns the shared embedding-shard contract computes rather than carries
#: through from the article sample.
SHARD_BASE_FIELDS: frozenset[str] = frozenset(
    {"url_hash", "symbol", "embedding", "text_length", "embedding_dim"}
)

#: Columns carried verbatim from the article sample into the shard, keeping
#: ettax output schema-identical to ``pipeline generate-embeddings`` (see
#: ``pipeline.stages.substrate.embeddings.SHARD_PASSTHROUGH_COLUMNS``, which
#: must stay in step with this set). An allow-list, not a deny-list: the source
#: Parquet comes from a ``SELECT *`` over the normalized corpus, so a new
#: upstream column would otherwise be passed through silently. ``title``,
#: ``body`` and ``publisher`` are deliberately absent — the shards are
#: redistributed and the corpus they derive from is not.
SHARD_PASSTHROUGH_FIELDS: frozenset[str] = frozenset(
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


@dataclass(frozen=True, slots=True)
class Article:
    """Minimal article record needed by the intrinsic evaluation."""

    url_hash: str
    symbol: str
    year: int
    title: str
    body: str
    extras: dict[str, object] = field(default_factory=dict)
    """Passthrough source columns, in source order, minus SHARD_BASE_FIELDS."""


@dataclass(frozen=True, slots=True)
class SemanticSmokeOptions:
    """Bounded inputs and runtime settings for the semantic smoke test."""

    articles_dir: Path
    qwen_dir: Path
    output: Path
    sample_size: int = 2_048
    batch_size: int = 128
    sample_seed: str = "t69-ettax-semantic-smoke-v1"
    legacy_tokenizer: Path | None = None
    mantel_sample_size: int = 512
    mantel_permutations: int = 999
    require_gpu: bool = True

    def __post_init__(self) -> None:
        """Reject settings that cannot support retrieval or geometry metrics."""
        if self.sample_size < 3:
            raise ValueError("semantic smoke sample_size must be at least three")
        if self.batch_size < 1:
            raise ValueError("semantic smoke batch_size must be positive")
        if self.mantel_sample_size < 3:
            raise ValueError("semantic smoke mantel_sample_size must be at least three")
        if self.mantel_sample_size > self.sample_size:
            raise ValueError(
                "semantic smoke mantel_sample_size cannot exceed sample_size"
            )
        if self.mantel_permutations < 1:
            raise ValueError("semantic smoke mantel_permutations must be positive")


def run_semantic_smoke(
    config: VintageConfig,
    options: SemanticSmokeOptions,
) -> dict[str, object]:
    """Evaluate every frozen vintage without touching outcome variables."""
    if options.require_gpu and jax.default_backend() != "gpu":
        message = (
            f"refusing checkpoint evaluation on backend={jax.default_backend()!r}; "
            "set JAX_PLATFORMS=cuda"
        )
        raise RuntimeError(message)
    experiment = ExperimentConfig.from_toml(config.experiment_config)
    if not config.tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer: {config.tokenizer_path}")

    articles = select_articles(
        load_articles(options.articles_dir),
        sample_size=options.sample_size,
        seed=options.sample_seed,
    )
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    title_texts = [article.title for article in articles]
    body_texts = [article.body for article in articles]
    document_texts = [f"{article.title}\n\n{article.body}" for article in articles]
    # Tokenizer diagnostics stay on the single-window encoder: they describe the
    # tokenizer's vocabulary coverage, not the pooling policy, and keeping them
    # here is what makes the optional legacy-tokenizer comparison below an
    # apples-to-apples [UNK]-rate check across two tokenizers.
    _, title_truncation = tokenize_documents(tokenizer, title_texts, experiment)
    _, body_truncation = tokenize_documents(tokenizer, body_texts, experiment)
    _, document_truncation = tokenize_documents(tokenizer, document_texts, experiment)

    # Vectors, by contrast, must follow the run's frozen recipe.
    fields = {
        "title": title_texts,
        "body": body_texts,
        "document": document_texts,
    }
    chunked = {
        name: tokenize_article_chunks(
            tokenizer,
            texts,
            sequence_length=experiment.data.sequence_length,
            pad_id=experiment.model.pad_id,
            doc_id=experiment.model.doc_id,
            eos_id=experiment.model.eos_id,
        )
        for name, texts in fields.items()
    }
    coverage = {
        name: _article_coverage(
            raw_lengths,
            content_limit=experiment.data.sequence_length - 2,
            article_policy=config.article_policy,
        )
        for name, (_, _, raw_lengths) in chunked.items()
    }
    special_ids = (
        experiment.model.pad_id,
        experiment.model.doc_id,
        experiment.model.eos_id,
        experiment.model.unk_id,
    )
    legacy_tokenization: dict[str, object] | None = None
    if options.legacy_tokenizer is not None:
        if not options.legacy_tokenizer.is_file():
            raise FileNotFoundError(
                f"missing legacy tokenizer: {options.legacy_tokenizer}"
            )
        legacy = Tokenizer.from_file(str(options.legacy_tokenizer))
        _, legacy_title = tokenize_documents(legacy, title_texts, experiment)
        _, legacy_body = tokenize_documents(legacy, body_texts, experiment)
        _, legacy_document = tokenize_documents(legacy, document_texts, experiment)
        legacy_tokenization = {
            "body": legacy_body,
            "document": legacy_document,
            "title": legacy_title,
        }
    qwen = load_qwen_embeddings(options.qwen_dir, articles)
    mantel_size = options.mantel_sample_size
    qwen_mantel = qwen[:mantel_size]

    arms: dict[str, object] = {}
    for arm in config.arms:
        checkpoint = config.checkpoint_dir(arm) / "model"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"missing model-only checkpoint: {checkpoint}")
        model = Ettax(experiment.model, rngs=nnx.Rngs(experiment.train.seed))
        restore_checkpoint(checkpoint, model)
        pooled = {
            name: aggregate_article_chunks(
                encode_and_pool(
                    model,
                    tokens,
                    batch_size=options.batch_size,
                    policies=(config.pooling_policy,),
                    special_ids=special_ids,
                )[config.pooling_policy],
                offsets,
                policy=config.article_policy,
            )
            for name, (tokens, offsets, _) in chunked.items()
        }
        title_vectors = pooled["title"]
        body_vectors = pooled["body"]
        document_vectors = pooled["document"]
        arms[arm.name] = {
            "article_coverage": coverage,
            "checkpoint": str(checkpoint),
            "checkpoint_dvc_md5": _dvc_md5_or_none(checkpoint.with_suffix(".dvc")),
            "document_geometry": embedding_diagnostics(document_vectors),
            "embedding_dimensions": int(document_vectors.shape[1]),
            "embedding_dtype": str(document_vectors.dtype),
            "identifier": arm.identifier,
            "qwen_geometry": compare_geometries(document_vectors, qwen),
            "qwen_mantel": mantel_tests(
                document_vectors[:mantel_size],
                qwen_mantel,
                permutations=options.mantel_permutations,
                seed=_integer_seed(options.sample_seed, f"{arm.name}:mantel"),
            ),
            "role": arm.role,
            "title_body_retrieval": retrieval_metrics(
                title_vectors,
                body_vectors,
                shuffle_seed=_integer_seed(options.sample_seed, arm.name),
            ),
            "title_geometry": embedding_diagnostics(title_vectors),
            "body_geometry": embedding_diagnostics(body_vectors),
        }

    tokenization: dict[str, object] = {
        "body": body_truncation,
        "document": document_truncation,
        "title": title_truncation,
    }
    if legacy_tokenization is not None:
        tokenization["legacy"] = legacy_tokenization

    summary: dict[str, object] = {
        "arms": arms,
        "environment": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pyarrow": pa.__version__,
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "inputs": {
            "article_policy": config.article_policy,
            "articles_dir": str(options.articles_dir),
            "checkpoint_layout": config.checkpoint_layout,
            "pooling_policy": config.pooling_policy,
            "qwen_dir": str(options.qwen_dir),
            "run_id": config.run_id,
            "qwen_sample_sha256": _matrix_sha256(articles, qwen),
            "sample_seed": options.sample_seed,
            "sample_sha256": _sample_sha256(articles),
            "sample_size": len(articles),
            "sequence_length": experiment.data.sequence_length,
            "tokenizer": str(config.tokenizer_path),
            "tokenizer_sha256": _file_sha256(config.tokenizer_path),
            "legacy_tokenizer": (
                str(options.legacy_tokenizer)
                if options.legacy_tokenizer is not None
                else None
            ),
            "mantel_sample_size": mantel_size,
            "mantel_permutations": options.mantel_permutations,
        },
        "purpose": (
            "pre-adjudication intrinsic transfer and collapse diagnostic; "
            "no price or return outcomes are read"
        ),
        "sample": {
            "symbols": dict(sorted(Counter(row.symbol for row in articles).items())),
            "years": {
                str(year): count
                for year, count in sorted(Counter(row.year for row in articles).items())
            },
        },
        # v3 emitted schema 2, whose vectors were always a single truncated
        # window read at [DOC] regardless of the run's frozen recipe. Schema 3
        # vectors follow `inputs.pooling_policy`/`inputs.article_policy`, so
        # geometry and retrieval figures are comparable across schema-3 runs
        # only when those two fields agree.
        "schema_version": 3,
        "tokenization": tokenization,
    }
    _write_json_atomic(options.output, summary)
    return summary


def _article_coverage(
    raw_lengths: Sequence[int],
    *,
    content_limit: int,
    article_policy: str,
) -> float:
    """Return the fraction of content tokens the article policy actually reads.

    ``chunks`` reads every window, so its coverage is one by construction;
    ``first`` discards everything past the opening window.
    """
    total = sum(raw_lengths)
    if total == 0:
        return 1.0
    if article_policy == "chunks":
        return 1.0
    return sum(min(length, content_limit) for length in raw_lengths) / total


def load_articles(directory: Path) -> list[Article]:
    """Read existing article Parquets without acquisition or rewriting."""
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no article Parquets under {directory}")
    articles: list[Article] = []
    for path in paths:
        # Every column is read, not just the five the smoke test needs: title
        # and body feed the encoder, and the allow-listed remainder are
        # passthrough columns carried verbatim into the shard.
        table = pq.read_table(path)
        passthrough = [
            c
            for c in table.column_names
            if c in SHARD_PASSTHROUGH_FIELDS and c not in SHARD_BASE_FIELDS
        ]
        rows = cast("list[dict[str, object]]", table.to_pylist())
        for row in rows:
            title = str(row.get("title") or "").strip()
            body = str(row.get("body") or "").strip()
            url_hash = str(row.get("url_hash") or "").strip()
            if not title or not body or not url_hash:
                continue
            articles.append(
                Article(
                    url_hash=url_hash,
                    symbol=str(row.get("symbol") or path.stem),
                    year=_year(row.get("created_date")),
                    title=title,
                    body=body,
                    extras={c: row.get(c) for c in passthrough},
                )
            )
    return articles


def select_articles(
    articles: Iterable[Article],
    *,
    sample_size: int,
    seed: str,
) -> list[Article]:
    """Select a deterministic, balanced ticker-year sample."""
    buckets: dict[tuple[str, int], list[Article]] = defaultdict(list)
    seen: set[str] = set()
    for article in articles:
        if article.url_hash in seen:
            raise ValueError(f"duplicate article url_hash: {article.url_hash}")
        seen.add(article.url_hash)
        buckets[(article.symbol, article.year)].append(article)
    if len(seen) < sample_size:
        raise ValueError(f"requested {sample_size} articles from only {len(seen)} rows")
    for values in buckets.values():
        values.sort(key=lambda row: _selection_key(seed, row.url_hash))

    selected: list[Article] = []
    depth = 0
    keys = sorted(buckets)
    while len(selected) < sample_size:
        added = False
        for key in keys:
            values = buckets[key]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            raise RuntimeError("balanced sampler exhausted before reaching sample_size")
        depth += 1
    selected.sort(key=lambda row: _selection_key(f"{seed}:order", row.url_hash))
    return selected


def load_qwen_embeddings(
    directory: Path,
    articles: Sequence[Article],
) -> NDArray[np.float64]:
    """Load existing Qwen vectors for exactly the selected article hashes."""
    required_by_symbol: dict[str, set[str]] = defaultdict(set)
    for article in articles:
        required_by_symbol[article.symbol].add(article.url_hash)
    vectors: dict[str, NDArray[np.float64]] = {}
    for symbol, required in sorted(required_by_symbol.items()):
        path = directory / f"{symbol}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing Qwen embedding shard: {path}")
        table = pq.read_table(path, columns=["url_hash", "embedding"])
        hashes = table.column("url_hash").to_pylist()
        embeddings = table.column("embedding")
        for index, raw_hash in enumerate(hashes):
            url_hash = str(raw_hash)
            if url_hash in required:
                if url_hash in vectors:
                    raise ValueError(f"duplicate Qwen url_hash: {url_hash}")
                vectors[url_hash] = np.asarray(
                    embeddings[index].as_py(), dtype=np.float64
                )
    missing = [
        article.url_hash for article in articles if article.url_hash not in vectors
    ]
    if missing:
        raise ValueError(f"Qwen embeddings missing {len(missing)} selected articles")
    matrix = np.stack([vectors[article.url_hash] for article in articles])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Qwen sample is not a finite embedding matrix")
    return matrix


def _year(value: object) -> int:
    """Resolve an Arrow date scalar represented as a date, datetime, or ISO string."""
    if hasattr(value, "year"):
        return int(cast("date | datetime", value).year)
    text = str(value)
    if len(text) < 4 or not text[:4].isdigit():
        raise ValueError(f"invalid article date: {value!r}")
    return int(text[:4])


def _selection_key(seed: str, url_hash: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{url_hash}".encode()).digest()


def _integer_seed(seed: str, arm: str) -> int:
    return int.from_bytes(_selection_key(seed, arm)[:8], byteorder="big")


def _sample_sha256(articles: Sequence[Article]) -> str:
    digest = hashlib.sha256()
    for article in articles:
        digest.update(article.url_hash.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _matrix_sha256(
    articles: Sequence[Article],
    matrix: NDArray[np.float64],
) -> str:
    digest = hashlib.sha256()
    for article, vector in zip(articles, matrix, strict=True):
        digest.update(article.url_hash.encode())
        digest.update(b"\0")
        digest.update(np.asarray(vector, dtype="<f8").tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dvc_md5(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing DVC pointer: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        marker = "md5:"
        if marker in stripped:
            return stripped.partition(marker)[2].strip()
    raise ValueError(f"DVC pointer has no md5: {path}")


def _dvc_md5_or_none(path: Path) -> str | None:
    """Read a DVC pointer when present, or mark a local pre-add artifact."""
    if not path.is_file():
        return None
    return _dvc_md5(path)


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)

# SPDX-License-Identifier: Apache-2.0
"""Pooling, chunking, and outcome-free intrinsic pilot evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tokenizers import Tokenizer

from ettax.config import ExperimentConfig
from ettax.evaluation import (
    compare_geometries,
    embedding_diagnostics,
    retrieval_metrics,
)
from ettax.model import Ettax
from ettax.pooling import (
    aggregate_article_chunks,
    encode_and_pool,
    pool_hidden_states,
    reconstruct_chunk_content,
    tokenize_article_chunks,
    tokenize_chunks,
)
from ettax.pooling import (
    normalize_rows as _normalize,
)
from ettax.semantic import load_articles, load_qwen_embeddings, select_articles
from ettax.train import restore_checkpoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ettax.pilot.config import ArticlePolicy, PilotArm, PilotConfig

type FloatMatrix = NDArray[np.float32]
type TokenMatrix = NDArray[np.int32]

__all__ = [
    "aggregate_article_chunks",
    "evaluate_trial",
    "paired_bootstrap_interval",
    "pool_hidden_states",
    "reconstruct_chunk_content",
    "tokenize_chunks",
]


def paired_bootstrap_interval(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Return a fixed-seed percentile interval for a paired mean difference."""
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError(
            "paired bootstrap inputs must be equal one-dimensional samples"
        )
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    difference = left - right
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, difference.size, size=(resamples, difference.size))
    means = np.mean(difference[draws], axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def evaluate_trial(
    config: PilotConfig,
    trial_name: str,
    *,
    arm: PilotArm,
    articles_dir: Path,
    qwen_dir: Path,
    batch_size: int,
    context_winner: int | None = None,
    require_gpu: bool = True,
) -> dict[str, object]:
    """Evaluate one checkpoint under all six frozen inference policies."""
    if batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"refusing pilot evaluation on backend={jax.default_backend()!r}; "
            "set JAX_PLATFORMS=cuda"
        )
    from ettax.pilot.training import resolve_trial_context

    trial = config.trial(trial_name)
    maximum_context = resolve_trial_context(trial, context_winner=context_winner)
    base = ExperimentConfig.from_toml(config.experiment_config)
    experiment = replace(
        base,
        model=replace(
            base.model,
            decoder_mask_rate=trial.decoder_mask_rate,
            bow_weight=trial.bow_weight,
        ),
        data=replace(base.data, sequence_length=maximum_context),
    )
    checkpoint = config.trial_root(arm, trial_name) / "checkpoints" / "model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing pilot checkpoint: {checkpoint}")
    articles = select_articles(
        load_articles(articles_dir),
        sample_size=config.evaluation_sample_size,
        seed=config.sample_seed,
    )
    qwen = load_qwen_embeddings(qwen_dir, articles)
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    model = Ettax(experiment.model, rngs=nnx.Rngs(config.training_seed))
    restore_checkpoint(checkpoint, model)
    texts = {
        "body": [article.body for article in articles],
        "document": [f"{article.title}\n\n{article.body}" for article in articles],
        "title": [article.title for article in articles],
    }
    started = time.perf_counter()
    encoded: dict[str, dict[str, object]] = {}
    total_content_tokens = 0
    for field, values in texts.items():
        tokens, offsets, raw_lengths = tokenize_article_chunks(
            tokenizer,
            values,
            sequence_length=experiment.data.sequence_length,
            pad_id=experiment.model.pad_id,
            doc_id=experiment.model.doc_id,
            eos_id=experiment.model.eos_id,
        )
        pooled = encode_and_pool(
            model,
            tokens,
            batch_size=batch_size,
            policies=config.pooling_policies,
            special_ids=(
                experiment.model.pad_id,
                experiment.model.doc_id,
                experiment.model.eos_id,
                experiment.model.unk_id,
            ),
        )
        encoded[field] = {
            "offsets": offsets,
            "pooled": pooled,
            "raw_lengths": raw_lengths,
            "tokens": tokens,
        }
        total_content_tokens += sum(raw_lengths)
    jax.block_until_ready(jnp.asarray(0))
    gpu_seconds = time.perf_counter() - started

    policies: dict[str, object] = {}
    for pooling in config.pooling_policies:
        for handling in config.article_policies:
            vectors: dict[str, FloatMatrix] = {}
            coverage: dict[str, float] = {}
            for field, values in encoded.items():
                chunk_vectors = cast("dict[str, FloatMatrix]", values["pooled"])[
                    pooling
                ]
                offsets = cast("list[tuple[int, int]]", values["offsets"])
                vectors[field] = aggregate_article_chunks(
                    chunk_vectors, offsets, policy=handling
                )
                raw_lengths = cast("list[int]", values["raw_lengths"])
                coverage[field] = _coverage(
                    raw_lengths,
                    maximum_context=maximum_context,
                    handling=handling,
                )
            reciprocal_ranks = _reciprocal_ranks(vectors["title"], vectors["body"])
            name = f"{pooling}-{handling}"
            policies[name] = {
                "article_coverage": coverage,
                "body_geometry": embedding_diagnostics(vectors["body"]),
                "document_geometry": embedding_diagnostics(vectors["document"]),
                "paired_reciprocal_ranks": reciprocal_ranks.tolist(),
                "qwen_geometry": compare_geometries(vectors["document"], qwen),
                "title_body_retrieval": retrieval_metrics(
                    vectors["title"],
                    vectors["body"],
                    shuffle_seed=config.bootstrap_seed,
                ),
                "title_geometry": embedding_diagnostics(vectors["title"]),
            }

    heldout_tokens = cast("TokenMatrix", encoded["document"]["tokens"])[
        : min(16, len(articles))
    ]
    losses = model.loss_components(
        jnp.asarray(heldout_tokens, dtype=jnp.int32),
        jax.random.key(config.training_seed),
    )
    training_summary_path = config.trial_root(arm, trial_name) / "summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "arm": arm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _directory_fingerprint(checkpoint),
        "content_throughput": float(
            training_summary["content_tokens_per_second_this_run"]
        ),
        "evaluation_content_tokens": total_content_tokens,
        "evaluation_gpu_seconds": gpu_seconds,
        "heldout_news_loss": {
            "bow": float(jax.device_get(losses.bow)),
            "reconstruction": float(jax.device_get(losses.reconstruction)),
            "total": float(jax.device_get(losses.total)),
        },
        "maximum_context": maximum_context,
        "pilot_id": config.pilot_id,
        "policies": policies,
        "purpose": "outcome-free intrinsic recipe selection",
        "sample_sha256": _sample_hash([article.url_hash for article in articles]),
        "sample_size": len(articles),
        "trial": trial_name,
    }
    destination = config.evaluation_root / arm / f"{trial_name}.json"
    _write_json_atomic(destination, result)
    return result


def _reciprocal_ranks(
    queries: FloatMatrix, candidates: FloatMatrix
) -> NDArray[np.float64]:
    left = _normalize(queries).astype(np.float64)
    right = _normalize(candidates).astype(np.float64)
    similarities = left @ right.T
    order = np.argsort(-similarities, axis=1, kind="stable")
    targets = np.arange(len(left))[:, None]
    ranks = np.argmax(order == targets, axis=1) + 1
    return 1.0 / ranks


def _coverage(
    lengths: Sequence[int], *, maximum_context: int, handling: ArticlePolicy
) -> float:
    total = sum(lengths)
    if total == 0:
        return 1.0
    if handling == "chunks":
        return 1.0
    visible = sum(min(length, maximum_context - 2) for length in lengths)
    return visible / total


def _sample_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(candidate.relative_to(path)).encode())
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)

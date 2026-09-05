# SPDX-License-Identifier: Apache-2.0
"""Chunking and pooling primitives shared by pilot selection and production.

These are stable EttaX primitives rather than pilot orchestration: the frozen
recipe selects one ``(pooling, article)`` policy pair, and the vintage producer
must apply exactly the pair that was selected. Keeping one implementation here
is what makes "the deployed encoder uses the frozen recipe" checkable instead of
asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from tokenizers import Tokenizer

    from ettax.model import Ettax

type FloatMatrix = NDArray[np.float32]
type TokenMatrix = NDArray[np.int32]
type PoolingPolicy = Literal["doc", "mean", "hybrid"]
type ArticlePolicy = Literal["first", "chunks"]

POOLING_POLICIES: tuple[PoolingPolicy, ...] = ("doc", "mean", "hybrid")
ARTICLE_POLICIES: tuple[ArticlePolicy, ...] = ("first", "chunks")


@nnx.jit
def _encode_step(model: Ettax, tokens: jax.Array) -> jax.Array:
    """Compile one hidden-state inference batch."""
    return model.encode(tokens)


def normalize_rows(values: NDArray[np.generic]) -> FloatMatrix:
    """Return unit-norm FP32 rows, rejecting degenerate zero vectors."""
    matrix = np.atleast_2d(np.asarray(values, dtype=np.float32))
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).tiny):
        raise ValueError("cannot normalize a zero vector")
    return matrix / norms


def tokenize_chunks(
    tokenizer: Tokenizer,
    text: str,
    *,
    sequence_length: int,
    pad_id: int,
    doc_id: int,
    eos_id: int,
) -> tuple[TokenMatrix, int]:
    """Encode every non-overlapping chunk while preserving source order."""
    if sequence_length < 3:
        raise ValueError("sequence_length must leave room for content and boundaries")
    content = tokenizer.encode(text).ids
    width = sequence_length - 2
    pieces: list[list[int]] = [
        content[offset : offset + width] for offset in range(0, len(content), width)
    ]
    if not pieces:
        pieces = [[]]
    matrix = np.full((len(pieces), sequence_length), pad_id, dtype=np.int32)
    for index, piece in enumerate(pieces):
        row = [doc_id, *piece, eos_id]
        matrix[index, : len(row)] = row
    return matrix, len(content)


def reconstruct_chunk_content(
    chunks: TokenMatrix, *, pad_id: int, doc_id: int, eos_id: int
) -> list[int]:
    """Reconstruct ordered content IDs from a chunk matrix for audit tests."""
    content: list[int] = []
    for row in np.asarray(chunks):
        content.extend(
            int(token) for token in row if token not in {pad_id, doc_id, eos_id}
        )
    return content


def pool_hidden_states(
    hidden: NDArray[np.generic],
    tokens: TokenMatrix,
    *,
    policy: PoolingPolicy,
    special_ids: tuple[int, ...],
) -> FloatMatrix:
    """Pool chunk hidden states using one of three predeclared policies."""
    values = np.asarray(hidden, dtype=np.float32)
    token_values = np.asarray(tokens)
    if values.ndim != 3 or token_values.shape != values.shape[:2]:
        raise ValueError("hidden states and tokens have incompatible shapes")
    doc = normalize_rows(values[:, 0])
    content_mask = np.ones_like(token_values, dtype=np.bool_)
    for token_id in special_ids:
        content_mask &= token_values != token_id
    counts = np.sum(content_mask, axis=1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("content-mean pooling received an empty chunk")
    mean = normalize_rows(
        np.sum(values * content_mask[..., None], axis=1) / counts.astype(np.float32)
    )
    if policy == "doc":
        return doc
    if policy == "mean":
        return mean
    if policy == "hybrid":
        return normalize_rows((doc + mean) / 2.0)
    raise ValueError(f"unknown pooling policy: {policy}")


def aggregate_article_chunks(
    chunk_vectors: FloatMatrix,
    offsets: Sequence[tuple[int, int]],
    *,
    policy: ArticlePolicy,
) -> FloatMatrix:
    """Apply first-window truncation or normalized all-chunk mean per article."""
    rows: list[NDArray[np.float32]] = []
    for start, end in offsets:
        if not 0 <= start < end <= len(chunk_vectors):
            raise ValueError("invalid article chunk offsets")
        if policy == "first":
            rows.append(chunk_vectors[start])
        elif policy == "chunks":
            rows.append(normalize_rows(chunk_vectors[start:end].mean(axis=0))[0])
        else:
            raise ValueError(f"unknown article policy: {policy}")
    return normalize_rows(np.stack(rows))


def tokenize_article_chunks(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    sequence_length: int,
    pad_id: int,
    doc_id: int,
    eos_id: int,
) -> tuple[TokenMatrix, list[tuple[int, int]], list[int]]:
    """Chunk many texts into one matrix plus per-article offsets and raw lengths."""
    chunks: list[TokenMatrix] = []
    offsets: list[tuple[int, int]] = []
    raw_lengths: list[int] = []
    cursor = 0
    for text in texts:
        matrix, raw_length = tokenize_chunks(
            tokenizer,
            text,
            sequence_length=sequence_length,
            pad_id=pad_id,
            doc_id=doc_id,
            eos_id=eos_id,
        )
        chunks.append(matrix)
        offsets.append((cursor, cursor + len(matrix)))
        raw_lengths.append(raw_length)
        cursor += len(matrix)
    return np.concatenate(chunks), offsets, raw_lengths


def encode_and_pool(
    model: Ettax,
    tokens: TokenMatrix,
    *,
    batch_size: int,
    policies: Sequence[PoolingPolicy],
    special_ids: tuple[int, ...],
) -> dict[str, FloatMatrix]:
    """Encode bounded batches once and pool them under every requested policy."""
    if batch_size < 1:
        raise ValueError("pooling batch_size must be positive")
    step = nnx.cached_partial(_encode_step, model)
    outputs: dict[str, list[FloatMatrix]] = {policy: [] for policy in policies}
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size]
        hidden = np.asarray(
            jax.device_get(step(jnp.asarray(batch, dtype=jnp.int32))), dtype=np.float32
        )
        for policy in policies:
            outputs[policy].append(
                pool_hidden_states(
                    hidden,
                    batch,
                    policy=policy,
                    special_ids=special_ids,
                )
            )
    return {policy: np.concatenate(parts) for policy, parts in outputs.items()}

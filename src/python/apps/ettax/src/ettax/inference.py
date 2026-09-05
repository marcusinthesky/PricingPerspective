# SPDX-License-Identifier: Apache-2.0
"""Shared tokenizer and accelerator-inference helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from tokenizers import Encoding, Tokenizer

    from ettax.config import ExperimentConfig
    from ettax.model import Ettax
type TokenMatrix = NDArray[np.int32]
type EmbeddingMatrix = NDArray[np.float32]


@nnx.jit
def _embed_step(model: Ettax, tokens: jax.Array) -> jax.Array:
    """Compile one normalized inference batch."""
    return model.embed(tokens)


def tokenize_documents(
    tokenizer: Tokenizer,
    texts: list[str],
    experiment: ExperimentConfig,
) -> tuple[TokenMatrix, dict[str, float | int]]:
    """Encode fixed-length documents and report truncation and unknown tokens."""
    encodings: list[Encoding] = tokenizer.encode_batch(texts)
    sequence_length = experiment.data.sequence_length
    content_limit = sequence_length - 2
    matrix = np.full(
        (len(encodings), sequence_length),
        experiment.model.pad_id,
        dtype=np.int32,
    )
    token_lengths = np.fromiter(
        (len(encoding.ids) for encoding in encodings),
        dtype=np.int64,
        count=len(encodings),
    )
    unknown_id = tokenizer.token_to_id("[UNK]")
    unknown_counts = np.fromiter(
        (
            encoding.ids.count(unknown_id) if unknown_id is not None else 0
            for encoding in encodings
        ),
        dtype=np.int64,
        count=len(encodings),
    )
    visible_unknown_counts = np.fromiter(
        (
            encoding.ids[:content_limit].count(unknown_id)
            if unknown_id is not None
            else 0
            for encoding in encodings
        ),
        dtype=np.int64,
        count=len(encodings),
    )
    for index, encoding in enumerate(encodings):
        content = encoding.ids[:content_limit]
        row = [experiment.model.doc_id, *content, experiment.model.eos_id]
        matrix[index, : len(row)] = row
    truncated = int(np.sum(token_lengths > content_limit))
    total_tokens = int(np.sum(token_lengths))
    visible_tokens = int(np.sum(np.minimum(token_lengths, content_limit)))
    return matrix, {
        "content_token_limit": content_limit,
        "duplicate_input_text_rows": len(texts) - len(set(texts)),
        "duplicate_token_rows": len(matrix) - int(np.unique(matrix, axis=0).shape[0]),
        "input_unknown_token_count": int(np.sum(unknown_counts)),
        "input_unknown_token_rate": float(np.sum(unknown_counts) / total_tokens),
        "max_content_tokens": int(np.max(token_lengths)),
        "mean_content_tokens": float(np.mean(token_lengths)),
        "rows_with_input_unknown_tokens": int(np.sum(unknown_counts > 0)),
        "rows_with_visible_unknown_tokens": int(np.sum(visible_unknown_counts > 0)),
        "truncated": truncated,
        "truncation_rate": truncated / len(encodings),
        "visible_unknown_token_count": int(np.sum(visible_unknown_counts)),
        "visible_unknown_token_rate": float(
            np.sum(visible_unknown_counts) / visible_tokens
        ),
    }


def embed_tokens(
    model: Ettax,
    tokens: TokenMatrix,
    batch_size: int,
) -> EmbeddingMatrix:
    """Run bounded device batches and materialize normalized FP32 vectors once."""
    if batch_size < 1:
        raise ValueError("embedding batch_size must be positive")
    step = nnx.cached_partial(_embed_step, model)
    chunks: list[EmbeddingMatrix] = []
    for start in range(0, len(tokens), batch_size):
        batch = jnp.asarray(tokens[start : start + batch_size], dtype=jnp.int32)
        chunks.append(cast("EmbeddingMatrix", np.asarray(jax.device_get(step(batch)))))
    values = np.concatenate(chunks, axis=0)
    if values.dtype != np.float32:
        raise TypeError(f"EttaX inference returned {values.dtype}, expected float32")
    return values

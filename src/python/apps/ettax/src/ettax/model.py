# SPDX-License-Identifier: Apache-2.0
"""Ettin-style bidirectional encoder with a lightweight TSDAE objective."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from ettax.config import ModelConfig

type Tokens = Int[Array, "batch length"]
type Mask = Bool[Array, "batch length"]
type Hidden = Float[Array, "*batch width"]
type Mode = Literal["global", "local", "causal", "causal-local"]

_DPA_PARAMETERS = inspect.signature(jax.nn.dot_product_attention).parameters
_ACCELERATED_DPA = jax.default_backend() == "gpu"
_HAS_LOCAL_DPA = _ACCELERATED_DPA and "local_window_size" in _DPA_PARAMETERS
_HAS_LENGTH_DPA = _ACCELERATED_DPA and "query_seq_lengths" in _DPA_PARAMETERS


class LossComponents(NamedTuple):
    """Separately report reconstruction, BoW, and weighted total losses."""

    reconstruction: Array
    bow: Array
    total: Array


def _normal(
    key: Array, shape: tuple[int, ...], dtype: DTypeLike = jnp.float32
) -> Array:
    """Initialize BERT-style weights."""
    return jax.random.normal(key, shape, dtype) * 0.02


class LayerNorm(nnx.Module):
    """Small explicit LayerNorm retaining scale and bias for exact parity."""

    def __init__(self, width: int) -> None:
        """Initialize normalization parameters.

        Args:
            width: Residual width.

        """
        self.scale = nnx.Param(jnp.ones((width,), jnp.float32))
        self.bias = nnx.Param(jnp.zeros((width,), jnp.float32))

    def __call__(self, inputs: Hidden) -> Hidden:
        """Normalize the final axis in float32."""
        source_dtype = inputs.dtype
        values = inputs.astype(jnp.float32)
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
        output = (values - mean) * jax.lax.rsqrt(variance + 1.0e-5)
        return (output * self.scale[...] + self.bias[...]).astype(source_dtype)


class Attention(nnx.Module):
    """Bias-free multi-head attention with RoPE and a fused local kernel hint."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize attention projections.

        Args:
            config: Model settings.
            rngs: NNX random streams.

        """
        self.config = config
        dtype = _dtype(config.compute_dtype)
        self.qkv = nnx.Linear(
            config.width,
            3 * config.width,
            use_bias=False,
            kernel_init=_normal,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.width,
            config.width,
            use_bias=False,
            kernel_init=_normal,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, inputs: Hidden, mask: Mask, mode: Mode) -> Hidden:
        """Mix tokens globally, locally, or causally."""
        batch, length, _ = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.config.heads, -1)
        query, key, value = jnp.moveaxis(qkv, 2, 0)
        query, key = _apply_rope(query, key, self.config.rope_theta)
        implementation = (
            self.config.global_attention_implementation
            if _ACCELERATED_DPA and mode == "global"
            else "xla"
        )
        minimum = jnp.finfo(query.dtype).min
        lengths = jnp.sum(mask, axis=1, dtype=jnp.int32)
        bias = None
        if not _HAS_LENGTH_DPA:
            bias = jnp.where(mask[:, None, None, :], 0.0, minimum).astype(query.dtype)
        local = None
        if mode == "local":
            half = self.config.local_window // 2
            local = (half, max(half - 1, 0))
        elif mode == "causal-local":
            local = (self.config.local_window - 1, 0)
        if _HAS_LOCAL_DPA:
            if _HAS_LENGTH_DPA:
                mixed = jax.nn.dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=mode in {"causal", "causal-local"},
                    query_seq_lengths=lengths,
                    key_value_seq_lengths=lengths,
                    local_window_size=local,
                    implementation=implementation,
                )
            else:
                mixed = jax.nn.dot_product_attention(
                    query,
                    key,
                    value,
                    bias=bias,
                    is_causal=mode in {"causal", "causal-local"},
                    local_window_size=local,
                    implementation=implementation,
                )
        else:
            if bias is None:
                bias = jnp.where(mask[:, None, None, :], 0.0, minimum).astype(
                    query.dtype
                )
            if local is not None:
                positions = jnp.arange(length)
                local_mask = (positions[:, None] - positions[None, :] <= local[0]) & (
                    positions[None, :] - positions[:, None] <= local[1]
                )
                bias = jnp.where(local_mask[None, None], bias, minimum)
            mixed = jax.nn.dot_product_attention(
                query,
                key,
                value,
                bias=bias,
                is_causal=mode in {"causal", "causal-local"},
                implementation=implementation,
            )
        mixed = mixed.reshape(batch, length, self.config.width)
        return self.output(mixed) * mask[..., None]


class FeedForward(nnx.Module):
    """Bias-free GeGLU feed-forward network."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize GeGLU projections.

        Args:
            config: Model settings.
            rngs: NNX random streams.

        """
        dtype = _dtype(config.compute_dtype)
        self.input = nnx.Linear(
            config.width,
            2 * config.ff_width,
            use_bias=False,
            kernel_init=_normal,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.ff_width,
            config.width,
            use_bias=False,
            kernel_init=_normal,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, inputs: Hidden) -> Hidden:
        """Apply GeGLU."""
        gate, value = jnp.split(self.input(inputs), 2, axis=-1)
        return self.output(jax.nn.gelu(gate, approximate=True) * value)


class Block(nnx.Module):
    """Pre-normalized Transformer block."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize one block.

        Args:
            config: Model settings.
            rngs: NNX random streams.

        """
        self.attention_norm = LayerNorm(config.width)
        self.attention = Attention(config, rngs=rngs)
        self.ffn_norm = LayerNorm(config.width)
        self.ffn = FeedForward(config, rngs=rngs)

    def __call__(self, inputs: Hidden, mask: Mask, mode: Mode) -> Hidden:
        """Apply attention and GeGLU residual branches."""
        hidden = inputs + self.attention(self.attention_norm(inputs), mask, mode)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden * mask[..., None]


class Ettax(nnx.Module):
    """A 22M-parameter Ettin-style document encoder."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the encoder.

        Args:
            config: Model settings.
            rngs: NNX random streams.

        """
        self.config = config
        self.tokens = nnx.Embed(
            config.vocab_size,
            config.width,
            embedding_init=_normal,
            dtype=_dtype(config.compute_dtype),
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.blocks = nnx.List([Block(config, rngs=rngs) for _ in range(config.layers)])
        self.final_norm = LayerNorm(config.width)
        self.decoder_blocks = nnx.List(
            [Block(config, rngs=rngs) for _ in range(config.decoder_layers)]
        )
        self.decoder_norm = LayerNorm(config.width)

    def encode(self, token_ids: Tokens) -> Hidden:
        """Encode padded, right-aligned documents bidirectionally.

        Args:
            token_ids: Token IDs with ``[DOC]`` at position zero.

        """
        mask = token_ids != self.config.pad_id
        hidden = self.tokens(token_ids)
        for index, block in enumerate(self.blocks):
            mode: Mode = "global" if index % self.config.global_every == 0 else "local"
            hidden = block(hidden, mask, mode)
        return self.final_norm(hidden) * mask[..., None]

    def embed(
        self, token_ids: Tokens, *, normalize: bool = True
    ) -> Float[Array, "batch width"]:
        """Return the final ``[DOC]`` representation.

        Args:
            token_ids: Token IDs with ``[DOC]`` at position zero.
            normalize: Apply L2 normalization for cosine search.

        """
        vectors = self.encode(token_ids)[:, 0].astype(jnp.float32)
        if normalize:
            vectors /= jnp.maximum(
                jnp.linalg.norm(vectors, axis=-1, keepdims=True), 1.0e-12
            )
        return vectors

    def loss(self, clean: Tokens, key: Array) -> Float[Array, ""]:
        """Compute the configured TSDAE-plus-optional-BoW objective.

        Args:
            clean: Clean fixed-length documents.
            key: JAX random key used for deletion noise.

        """
        return self.loss_components(clean, key).total

    def loss_components(self, clean: Tokens, key: Array) -> LossComponents:
        """Compute and retain every scientific loss component separately."""
        if self.config.decoder_mask_rate > 0.0:
            encoder_key, decoder_key = jax.random.split(key)
        else:
            encoder_key, decoder_key = key, key
        noisy = delete_tokens(
            clean, encoder_key, self.config.pad_id, self.config.delete_rate
        )
        document = self.encode(noisy)[:, 0]
        targets = clean[:, 1:]
        mask = targets != self.config.pad_id
        prefix_tokens = targets[:, :-1]
        if self.config.decoder_mask_rate > 0.0:
            prefix_tokens = mask_decoder_prefix(
                prefix_tokens,
                decoder_key,
                rate=self.config.decoder_mask_rate,
                unk_id=self.config.unk_id,
                special_ids=(
                    self.config.pad_id,
                    self.config.doc_id,
                    self.config.eos_id,
                ),
            )
        prefix = self.tokens(prefix_tokens)
        start = jnp.zeros_like(document[:, None])
        decoder = jnp.concatenate((start, prefix), axis=1) + document[:, None]
        for block in self.decoder_blocks:
            decoder = block(decoder, mask, "causal-local")
        decoder = self.decoder_norm(decoder)
        reconstruction = tied_cross_entropy(
            decoder,
            targets,
            mask,
            self.tokens.embedding[...],
            self.config.logit_chunk,
        )
        bow = jnp.asarray(0.0, dtype=jnp.float32)
        if self.config.bow_weight > 0.0:
            bow = tied_bow_cross_entropy(
                document,
                clean,
                self.tokens.embedding[...],
                special_ids=(
                    self.config.pad_id,
                    self.config.doc_id,
                    self.config.eos_id,
                    self.config.unk_id,
                ),
            )
        total = reconstruction + self.config.bow_weight * bow
        return LossComponents(reconstruction, bow, total)


def delete_tokens(tokens: Tokens, key: Array, pad_id: int, rate: float) -> Tokens:
    """Delete and compact random content tokens while preserving both endpoints."""
    batch, length = tokens.shape
    valid = tokens != pad_id
    keep = jax.random.bernoulli(key, 1.0 - rate, tokens.shape) & valid
    keep = keep.at[:, 0].set(True)
    final = jnp.maximum(jnp.sum(valid, axis=1) - 1, 0)
    keep = keep.at[jnp.arange(batch), final].set(True)
    positions = jnp.broadcast_to(jnp.arange(length), tokens.shape)
    order = jnp.argsort(jnp.where(keep, positions, positions + length), axis=1)
    compact = jnp.take_along_axis(tokens, order, axis=1)
    compact_mask = jnp.arange(length)[None, :] < jnp.sum(keep, axis=1, keepdims=True)
    return jnp.where(compact_mask, compact, pad_id)


def mask_decoder_prefix(
    tokens: Tokens,
    key: Array,
    *,
    rate: float,
    unk_id: int,
    special_ids: tuple[int, ...],
) -> Tokens:
    """Replace eligible shifted-prefix tokens with ``[UNK]`` in place."""
    if not 0.0 <= rate < 1.0:
        raise ValueError("decoder prefix mask rate must lie in [0, 1)")
    eligible = jnp.ones_like(tokens, dtype=jnp.bool_)
    for token_id in special_ids:
        eligible &= tokens != token_id
    selected = jax.random.bernoulli(key, rate, tokens.shape) & eligible
    return jnp.where(selected, unk_id, tokens)


def tied_bow_cross_entropy(
    document: Float[Array, "batch width"],
    clean: Tokens,
    embedding: Float[Array, "vocab width"],
    *,
    special_ids: tuple[int, ...],
) -> Float[Array, ""]:
    """Compare one tied-vocabulary ``[DOC]`` distribution with clean token counts."""
    projection = embedding.astype(document.dtype)
    logits = jnp.matmul(
        document,
        projection.T,
        preferred_element_type=jnp.float32,
    )
    log_probabilities = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    selected = jnp.take_along_axis(log_probabilities, clean, axis=-1)
    content = jnp.ones_like(clean, dtype=jnp.bool_)
    for token_id in special_ids:
        content &= clean != token_id
    counts = jnp.sum(content, axis=1)
    per_document = -jnp.sum(selected * content, axis=1) / jnp.maximum(counts, 1)
    valid = counts > 0
    return jnp.sum(per_document * valid) / jnp.maximum(jnp.sum(valid), 1)


def tied_cross_entropy(
    hidden: Hidden,
    targets: Tokens,
    mask: Mask,
    embedding: Float[Array, "vocab width"],
    chunk_size: int,
) -> Float[Array, ""]:
    """Evaluate tied softmax in bounded token chunks."""
    width = hidden.shape[-1]
    flat_hidden = hidden.reshape(-1, width)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    padding = (-flat_hidden.shape[0]) % chunk_size
    flat_hidden = jnp.pad(flat_hidden, ((0, padding), (0, 0)))
    flat_targets = jnp.pad(flat_targets, (0, padding))
    flat_mask = jnp.pad(flat_mask, (0, padding))
    chunks = flat_hidden.shape[0] // chunk_size
    packed = (
        flat_hidden.reshape(chunks, chunk_size, width),
        flat_targets.reshape(chunks, chunk_size),
        flat_mask.reshape(chunks, chunk_size),
    )
    projection = embedding.astype(flat_hidden.dtype)

    def accumulate(
        total: tuple[Array, Array], values: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array], None]:
        states, labels, active = values
        # Match the configured model-compute dtype for tensor-core operands while
        # retaining stable FP32 accumulation, logits, and cross-entropy reduction.
        logits = jnp.matmul(
            states,
            projection.T,
            preferred_element_type=jnp.float32,
        )
        selected = jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
        losses = jax.nn.logsumexp(logits, axis=-1) - selected
        return (total[0] + jnp.sum(losses * active), total[1] + jnp.sum(active)), None

    (loss_sum, count), _ = jax.lax.scan(
        jax.checkpoint(accumulate),
        (jnp.asarray(0.0, jnp.float32), jnp.asarray(0, jnp.int32)),
        packed,
    )
    return loss_sum / jnp.maximum(count, 1)


def actual_parameter_count(model: Ettax) -> int:
    """Count NNX parameter leaves."""
    total = 0
    for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)):
        value = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
        total += int(value.size)
    return total


def _apply_rope(query: Hidden, key: Hidden, theta: float) -> tuple[Hidden, Hidden]:
    """Apply rotary position embeddings to query and key heads."""
    length, dim = query.shape[1], query.shape[-1]
    frequencies = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    angles = jnp.arange(length, dtype=jnp.float32)[:, None] * frequencies[None, :]
    cosine = jnp.cos(angles)[None, :, None, :]
    sine = jnp.sin(angles)[None, :, None, :]

    def rotate(values: Hidden) -> Hidden:
        even, odd = values[..., ::2], values[..., 1::2]
        paired = jnp.stack(
            (even * cosine - odd * sine, odd * cosine + even * sine), axis=-1
        )
        return paired.reshape(values.shape).astype(values.dtype)

    return rotate(query), rotate(key)


def _dtype(name: str) -> DTypeLike:
    """Resolve the deliberately small compute-dtype vocabulary."""
    try:
        return {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[name]
    except KeyError as error:
        raise ValueError(f"unsupported compute dtype: {name}") from error

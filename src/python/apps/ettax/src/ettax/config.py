# SPDX-License-Identifier: Apache-2.0
"""Typed experiment configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from math import ceil
from pathlib import Path
from typing import Literal, Self, cast

_ROPE_PAIR = 2
_SPECIAL_TOKEN_COUNT = 4
_MIN_SEQUENCE_LENGTH = 8
_MIN_LOCAL_WINDOW = 2
type AttentionImplementation = Literal["xla", "cudnn"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Ettin-style encoder and tied TSDAE decoder settings."""

    vocab_size: int = 32_768
    width: int = 320
    layers: int = 12
    heads: int = 5
    ff_width: int = 576
    local_window: int = 128
    global_every: int = 3
    rope_theta: float = 160_000.0
    pad_id: int = 0
    doc_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    delete_rate: float = 0.60
    decoder_mask_rate: float = 0.0
    bow_weight: float = 0.0
    decoder_layers: int = 1
    logit_chunk: int = 256
    compute_dtype: str = "bfloat16"
    global_attention_implementation: AttentionImplementation = "xla"

    def __post_init__(self) -> None:
        """Reject shapes that cannot form the intended architecture."""
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (self.width // self.heads) % _ROPE_PAIR:
            raise ValueError("RoPE requires an even head dimension")
        if self.decoder_layers < 1:
            raise ValueError("decoder_layers must be positive")
        if not 0.0 < self.delete_rate < 1.0:
            raise ValueError("delete_rate must lie strictly between zero and one")
        if self.local_window < _MIN_LOCAL_WINDOW or self.global_every < 1:
            raise ValueError("attention intervals must be positive")
        if max(self.pad_id, self.doc_id, self.eos_id, self.unk_id) >= self.vocab_size:
            raise ValueError("special token IDs must be inside the vocabulary")
        _validate_objective(self)
        if self.global_attention_implementation not in {"xla", "cudnn"}:
            raise ValueError("unsupported global attention implementation")

    @property
    def head_dim(self) -> int:
        """Return the width of one attention head."""
        return self.width // self.heads

    @property
    def parameter_count(self) -> int:
        """Return deployed encoder parameters, including all LayerNorm terms."""
        embedding = self.vocab_size * self.width
        final_norm = 2 * self.width
        return embedding + self.layers * self.block_parameter_count + final_norm

    @property
    def block_parameter_count(self) -> int:
        """Return parameters in one attention-and-GeGLU block."""
        attention = 4 * self.width * self.width
        geglu = 3 * self.width * self.ff_width
        norms = 4 * self.width
        return attention + geglu + norms

    @property
    def decoder_parameter_count(self) -> int:
        """Return parameters used only by the training decoder."""
        final_norm = 2 * self.width
        return self.decoder_layers * self.block_parameter_count + final_norm

    @property
    def training_parameter_count(self) -> int:
        """Return parameters in the complete encoder-decoder training graph."""
        return self.parameter_count + self.decoder_parameter_count


def _validate_objective(config: ModelConfig) -> None:
    special_ids = {config.pad_id, config.doc_id, config.eos_id, config.unk_id}
    if len(special_ids) != _SPECIAL_TOKEN_COUNT:
        raise ValueError("special token IDs must be distinct")
    if not 0.0 <= config.decoder_mask_rate < 1.0:
        raise ValueError("decoder_mask_rate must lie in [0, 1)")
    if not 0.0 <= config.bow_weight <= 1.0:
        raise ValueError("bow_weight must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Prepared corpus settings."""

    paths: tuple[str, ...] = (
        "data/processed/wiki-0.u16",
        "data/processed/wiki-1.u16",
        "data/processed/wiki-2.u16",
    )
    sequence_length: int = 1_024
    batch_size: int = 8
    shuffle_seed: int = 17

    def __post_init__(self) -> None:
        """Reject unusable batch and sequence dimensions."""
        if (
            not self.paths
            or self.sequence_length < _MIN_SEQUENCE_LENGTH
            or self.batch_size < 1
        ):
            raise ValueError(
                "data requires paths, sequence_length >= 8, and batch_size >= 1"
            )


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Single-device optimizer and checkpoint settings."""

    token_budget: int = 300_000_000
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_fraction: float = 0.02
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    log_every: int = 20
    checkpoint_every: int = 2_000
    checkpoint_dir: str = "artifacts/checkpoints"
    live_dir: str = "artifacts/dvclive"
    metrics_path: str = "artifacts/metrics.json"

    def __post_init__(self) -> None:
        """Reject schedules that cannot be evaluated."""
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0, 1)")
        if min(self.log_every, self.checkpoint_every) < 1:
            raise ValueError("logging and checkpoint intervals must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()

    @property
    def steps(self) -> int:
        """Return updates required to meet the fixed token-slot budget."""
        return ceil(self.train.token_budget / self.tokens_per_step)

    @property
    def tokens_per_step(self) -> int:
        """Return fixed-width token slots presented in one update."""
        return self.data.batch_size * self.data.sequence_length

    @property
    def nominal_tokens(self) -> int:
        """Return token slots actually presented after final-batch rounding."""
        return self.steps * self.tokens_per_step

    @property
    def warmup_steps(self) -> int:
        """Return the deterministic warmup length for the resolved run."""
        return int(self.steps * self.train.warmup_fraction)

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load an experiment from TOML.

        Args:
            path: Configuration file.

        """
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        return cls(
            model=_construct(ModelConfig, raw.get("model", {})),
            data=_construct(DataConfig, raw.get("data", {})),
            train=_construct(TrainConfig, raw.get("train", {})),
        )


def _construct[T](kind: type[T], values: dict[str, object]) -> T:
    """Construct a dataclass while rejecting misspelled keys."""
    config_kind = cast(
        "type[ModelConfig | DataConfig | TrainConfig]",
        kind,
    )
    allowed = {field.name for field in fields(config_kind)}
    unknown = values.keys() - allowed
    if unknown:
        message = f"unknown {kind.__name__} keys: {sorted(unknown)}"
        raise ValueError(message)
    if kind is DataConfig and isinstance(values.get("paths"), list):
        values = {**values, "paths": tuple(values["paths"])}
    return kind(**values)  # type: ignore[arg-type]

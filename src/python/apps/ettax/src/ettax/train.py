# SPDX-License-Identifier: Apache-2.0
"""Single-device NNX training, benchmarking, and resumable checkpoints."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from ettax.config import ExperimentConfig
from ettax.data import MemmapCorpus, grain_batches
from ettax.model import Ettax
from ettax.tracking import TrainingTracker

LOGGER = logging.getLogger(__name__)
_TRAINING_PROBE_BATCHES = 8
_TRAINING_PROBE_SEED = 20_260_809


@nnx.jit
def train_step(
    model: Ettax,
    optimizer: nnx.Optimizer[Ettax],
    batch: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compile one parameter update."""

    def objective(candidate: Ettax) -> jax.Array:
        return candidate.loss(batch, key)

    loss, gradients = nnx.value_and_grad(objective)(model)
    gradient_norm = optax.tree.norm(gradients)
    optimizer.update(model, gradients)
    return loss, gradient_norm


@nnx.jit
def evaluate_step(model: Ettax, batch: jax.Array, key: jax.Array) -> jax.Array:
    """Compile one loss-only step."""
    return model.loss(batch, key)


def run(
    config_or_path: ExperimentConfig | str | Path,
    *,
    resume: bool = True,
) -> dict[str, float | int | str | bool]:
    """Train one experiment and return final metrics.

    Args:
        config_or_path: Resolved experiment or TOML path.
        resume: Restore the latest complete model/optimizer/RNG checkpoint.

    """
    config = _resolve_config(config_or_path)
    corpus, probe_batches, probe_keys = _corpus_with_probe(config)
    stream = iter(
        grain_batches(
            corpus,
            config.data.batch_size,
            seed=config.data.shuffle_seed,
        )
    )
    model = Ettax(config.model, rngs=nnx.Rngs(config.train.seed))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.train.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
        end_value=config.train.min_learning_rate,
    )
    transform = optax.chain(
        optax.clip_by_global_norm(config.train.max_grad_norm),
        optax.adamw(schedule, weight_decay=config.train.weight_decay),
    )
    optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
    key = jax.random.key(config.train.seed)
    checkpoint_root = Path(config.train.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    initial_step = 0
    if resume:
        restored = restore_training_checkpoint(checkpoint_root, model, optimizer)
        if restored is not None:
            initial_step, key = restored
            LOGGER.info(
                "resumed checkpoint at step %d of %d", initial_step, config.steps
            )

    # The model and optimizer graph structures are immutable during training.
    # Cache their NNX traversal once while retaining shared Variable references.
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)

    # Grain's seeded shuffle is deterministic, but its iterator is not part of the
    # Orbax graph. Replay the already-consumed positions so a resumed update sees
    # the same batch it would have seen in an uninterrupted run.
    for _ in range(initial_step):
        next(stream)

    started = time.perf_counter()
    last_loss = float("nan")
    tracker = TrainingTracker(
        config.train.live_dir,
        initial_step=initial_step,
        parameters={
            **config_manifest(config),
            "tracking": {
                "ema_decay": 0.98,
                "training_probe_rows": int(probe_batches.shape[0])
                * config.data.batch_size,
                "training_probe_seed": _TRAINING_PROBE_SEED,
            },
        },
    )
    with tracker:
        if tracker.needs_checkpoint_probe(initial_step):
            tracker.record_checkpoint_probe(
                step=initial_step,
                loss=_training_probe_loss(model, probe_batches, probe_keys),
            )
        for step in range(initial_step + 1, config.steps + 1):
            batch = jnp.asarray(next(stream), dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            loss, gradient_norm = cached_train_step(batch, step_key)
            should_log = (
                step % config.train.log_every == 0
                or step == initial_step + 1
                or step == config.steps
            )
            if should_log:
                last_loss, last_gradient_norm = (
                    float(value) for value in jax.device_get((loss, gradient_norm))
                )
                elapsed = time.perf_counter() - started
                run_steps = step - initial_step
                rate = run_steps * config.tokens_per_step / max(elapsed, 1.0e-9)
                learning_rate = float(jax.device_get(schedule(step - 1)))
                observation = {
                    "step": step,
                    "steps": config.steps,
                    "loss": last_loss,
                    "gradient_norm": last_gradient_norm,
                    "learning_rate": learning_rate,
                    "token_slots_per_second": rate,
                }
                LOGGER.info("%s", json.dumps(observation, sort_keys=True))
                tracker.record(
                    step=step,
                    loss=last_loss,
                    gradient_norm=last_gradient_norm,
                    learning_rate=learning_rate,
                    token_slots_per_second=rate,
                )
            if step % config.train.checkpoint_every == 0 or step == config.steps:
                tracker.record_checkpoint_probe(
                    step=step,
                    loss=_training_probe_loss(model, probe_batches, probe_keys),
                )
                save_training_checkpoint(checkpoint_root, model, optimizer, step, key)

        elapsed = time.perf_counter() - started
        completed_steps = max(config.steps - initial_step, 0)
        if config.steps > 0:
            save_checkpoint(checkpoint_root / "model", model)
        metrics: dict[str, float | int | str | bool] = {
            "backend": jax.default_backend(),
            "batch_size": config.data.batch_size,
            "completed_steps_this_run": completed_steps,
            "compute_dtype": config.model.compute_dtype,
            "elapsed_seconds_this_run": elapsed,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "loss": last_loss,
            "nominal_token_slots": config.nominal_tokens,
            "encoder_parameters": config.model.parameter_count,
            "parameter_dtype": "float32",
            "resumed_from_step": initial_step,
            "sequence_length": config.data.sequence_length,
            "steps": config.steps,
            "token_budget": config.train.token_budget,
            "token_slots_per_second_this_run": completed_steps
            * config.tokens_per_step
            / max(elapsed, 1.0e-9),
        }
        metrics_path = Path(config.train.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(metrics_path, metrics)
        tracker.finalize(metrics)
    return metrics


def benchmark(
    config_or_path: ExperimentConfig | str | Path,
    *,
    warmup: int = 1,
    steps: int = 3,
) -> dict[str, object]:
    """Measure full-model training throughput on synthetic, non-padding tokens."""
    if warmup < 1 or steps < 1:
        raise ValueError("warmup and measured steps must be positive")
    config = _resolve_config(config_or_path)
    model = Ettax(config.model, rngs=nnx.Rngs(config.train.seed))
    optimizer = nnx.Optimizer(model, optax.adamw(1.0e-4), wrt=nnx.Param)
    key = jax.random.key(config.train.seed)
    batch_key, key = jax.random.split(key)
    batch = jax.random.randint(
        batch_key,
        (config.data.batch_size, config.data.sequence_length),
        minval=4,
        maxval=config.model.vocab_size,
        dtype=jnp.int32,
    )
    batch = batch.at[:, 0].set(config.model.doc_id)
    batch = batch.at[:, -1].set(config.model.eos_id)
    last_loss = jnp.asarray(float("nan"), dtype=jnp.float32)
    compiled = train_step.lower(model, optimizer, batch, key).compile()
    compiler_memory = _compiler_memory_analysis(compiled.memory_analysis())
    del compiled
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)
    for _ in range(warmup):
        key, step_key = jax.random.split(key)
        last_loss, _ = cached_train_step(batch, step_key)
        jax.block_until_ready(last_loss)
    started = time.perf_counter()
    for _ in range(steps):
        key, step_key = jax.random.split(key)
        last_loss, _ = cached_train_step(batch, step_key)
        jax.block_until_ready(last_loss)
    elapsed = time.perf_counter() - started
    rate = steps * config.tokens_per_step / elapsed
    return {
        "backend": jax.default_backend(),
        "batch_size": config.data.batch_size,
        "compute_dtype": config.model.compute_dtype,
        "compiler_memory_analysis": compiler_memory,
        "estimated_hours_per_arm": config.nominal_tokens / rate / 3600,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "loss": float(jax.device_get(last_loss)),
        "measured_steps": steps,
        "parameter_dtype": "float32",
        "encoder_parameters": config.model.parameter_count,
        "sequence_length": config.data.sequence_length,
        "training_parameters": config.model.training_parameter_count,
        "token_slots_per_second": rate,
        "warmup_steps": warmup,
    }


def _compiler_memory_analysis(value: object) -> dict[str, int | None]:
    """Normalize JAX's backend-specific compiled-memory report for JSON."""
    fields = (
        "generated_code_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {
        field: int(raw) if (raw := getattr(value, field, None)) is not None else None
        for field in fields
    }


def save_training_checkpoint(
    root: str | Path,
    model: Ettax,
    optimizer: nnx.Optimizer[Ettax],
    step: int,
    key: jax.Array,
) -> Path:
    """Persist model, optimizer, step, and PRNG state, then advance latest.json."""
    checkpoint_root = Path(root).absolute()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"step-{step:08d}"
    payload = {
        "key": np.asarray(jax.random.key_data(key)),
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": np.asarray(step, dtype=np.int64),
    }
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(destination, payload)
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    _write_json_atomic(
        checkpoint_root / "latest.json", {"step": step, "path": destination.name}
    )
    _prune_training_checkpoints(checkpoint_root, keep=destination)
    return destination


def restore_training_checkpoint(
    root: str | Path,
    model: Ettax,
    optimizer: nnx.Optimizer[Ettax],
) -> tuple[int, jax.Array] | None:
    """Restore the latest complete training checkpoint, if one exists."""
    checkpoint_root = Path(root).absolute()
    latest_path = checkpoint_root / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    source = checkpoint_root / str(latest["path"])
    target = {
        "key": np.asarray(jax.random.key_data(jax.random.key(0))),
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": np.asarray(0, dtype=np.int64),
    }
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(source, target=target)
    finally:
        checkpointer.close()
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    key = jax.random.wrap_key_data(jnp.asarray(restored["key"], dtype=jnp.uint32))
    return int(restored["step"]), key


def _prune_training_checkpoints(checkpoint_root: Path, *, keep: Path) -> None:
    """Retain only the checkpoint referenced by ``latest.json``."""
    for candidate in checkpoint_root.glob("step-*"):
        if candidate != keep and candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)


def save_checkpoint(path: str | Path, model: Ettax) -> None:
    """Save model-only state for deployment/export compatibility."""
    checkpointer = ocp.StandardCheckpointer()
    destination = Path(path).absolute()
    state = nnx.state(model)
    try:
        checkpointer.save(destination, state, force=True)
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()


def restore_checkpoint(path: str | Path, model: Ettax) -> Ettax:
    """Restore model-only state into an initialized NNX graph."""
    checkpointer = ocp.StandardCheckpointer()
    source = Path(path).absolute()
    target = nnx.state(model)
    try:
        restored = checkpointer.restore(source, target=target)
    finally:
        checkpointer.close()
    nnx.update(model, restored)
    return model


def _resolve_config(config_or_path: ExperimentConfig | str | Path) -> ExperimentConfig:
    if isinstance(config_or_path, ExperimentConfig):
        return config_or_path
    return ExperimentConfig.from_toml(config_or_path)


def _validated_corpus(config: ExperimentConfig) -> MemmapCorpus:
    corpus = MemmapCorpus(config.data.paths)
    if corpus.sequence_length != config.data.sequence_length:
        raise ValueError("prepared and configured sequence lengths differ")
    expected_tokens = (
        config.model.vocab_size,
        config.model.pad_id,
        config.model.doc_id,
        config.model.eos_id,
    )
    if (
        corpus.vocab_size,
        corpus.pad_id,
        corpus.doc_id,
        corpus.eos_id,
    ) != expected_tokens:
        raise ValueError("prepared corpus and model tokenizer settings differ")
    return corpus


def _corpus_with_probe(
    config: ExperimentConfig,
) -> tuple[MemmapCorpus, jax.Array, jax.Array]:
    corpus = _validated_corpus(config)
    batches, keys = _training_probe(corpus, config)
    return corpus, batches, keys


def _training_probe(
    corpus: MemmapCorpus,
    config: ExperimentConfig,
) -> tuple[jax.Array, jax.Array]:
    probe_rows = _TRAINING_PROBE_BATCHES * config.data.batch_size
    indices = np.linspace(0, len(corpus) - 1, probe_rows, dtype=np.int64)
    rows = np.stack([corpus[int(index)] for index in indices])
    batches = rows.reshape(
        -1,
        config.data.batch_size,
        config.data.sequence_length,
    )
    keys = jax.random.split(jax.random.key(_TRAINING_PROBE_SEED), len(batches))
    return jnp.asarray(batches, dtype=jnp.int32), keys


def _training_probe_loss(
    model: Ettax,
    batches: jax.Array,
    keys: jax.Array,
) -> float:
    losses = [
        evaluate_step(model, batch, key)
        for batch, key in zip(batches, keys, strict=True)
    ]
    return float(jax.device_get(jnp.mean(jnp.stack(losses))))


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def config_manifest(config: ExperimentConfig) -> dict[str, object]:
    """Return the resolved immutable training specification for provenance."""
    return {
        "data": asdict(config.data),
        "model": asdict(config.model),
        "nominal_tokens": config.nominal_tokens,
        "steps": config.steps,
        "train": asdict(config.train),
        "warmup_steps": config.warmup_steps,
    }

# SPDX-License-Identifier: Apache-2.0
"""Content-matched pilot training with true optimizer-step accumulation."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ettax.config import ExperimentConfig
from ettax.data import MemmapCorpus, grain_batches
from ettax.model import Ettax, LossComponents
from ettax.pilot.data import content_token_count, resolve_common_content_budget
from ettax.train import (
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ettax.config import ModelConfig
    from ettax.pilot.config import PilotArm, PilotConfig, PilotTrial

_CALIBRATION_CYCLES = 3


@nnx.jit
def accumulated_train_step(
    model: Ettax,
    optimizer: nnx.Optimizer[Ettax],
    batches: jax.Array,
    keys: jax.Array,
) -> tuple[LossComponents, jax.Array]:
    """Average microbatch objectives, then apply exactly one optimizer update."""

    def objective(candidate: Ettax) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        components = [
            candidate.loss_components(batches[index], keys[index])
            for index in range(batches.shape[0])
        ]
        target_counts = jnp.sum(
            batches[:, :, 1:] != candidate.config.pad_id, axis=(1, 2)
        )
        content = jnp.ones_like(batches, dtype=jnp.bool_)
        for token_id in (
            candidate.config.pad_id,
            candidate.config.doc_id,
            candidate.config.eos_id,
            candidate.config.unk_id,
        ):
            content &= batches != token_id
        document_counts = jnp.sum(jnp.any(content, axis=-1), axis=-1)
        reconstruction = _weighted_component(
            components, target_counts, component_index=0
        )
        bow = _weighted_component(components, document_counts, component_index=1)
        total = reconstruction + candidate.config.bow_weight * bow
        return total, (reconstruction, bow)

    (total, (reconstruction, bow)), gradients = nnx.value_and_grad(
        objective, has_aux=True
    )(model)
    gradient_norm = optax.tree.norm(gradients)
    optimizer.update(model, gradients)
    return LossComponents(reconstruction, bow, total), gradient_norm


def _weighted_component(
    components: list[LossComponents],
    counts: jax.Array,
    *,
    component_index: int,
) -> jax.Array:
    values = jnp.stack([component[component_index] for component in components])
    weights = counts.astype(jnp.float32)
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def run_trial(
    config: PilotConfig,
    trial_name: str,
    *,
    arm: PilotArm = "V0",
    content_budget: int,
    context_winner: int | None = None,
    require_gpu: bool = True,
    output_root: Path | None = None,
    prepared_root: Path | None = None,
    maximum_context_only: bool = False,
) -> dict[str, object]:
    """Train or resume one named recipe to a loss-bearing content-token budget."""
    if content_budget < 1:
        raise ValueError("content_budget must be positive")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"refusing pilot training on backend={jax.default_backend()!r}; "
            "set JAX_PLATFORMS=cuda"
        )
    trial = config.trial(trial_name)
    maximum_context = resolve_trial_context(trial, context_winner=context_winner)
    buckets = tuple(
        value for value in trial.training_buckets if value <= maximum_context
    )
    if maximum_context_only:
        buckets = (maximum_context,)
    probe = _load_probe(config.probe_manifest_path)
    base = ExperimentConfig.from_toml(config.experiment_config)
    model_config = replace(
        base.model,
        decoder_mask_rate=trial.decoder_mask_rate,
        bow_weight=trial.bow_weight,
    )
    model = Ettax(model_config, rngs=nnx.Rngs(config.training_seed))
    estimated_updates = estimate_updates(
        config,
        arm=arm,
        buckets=buckets,
        content_budget=content_budget,
        probe=probe,
        prepared_root=prepared_root,
    )
    warmup_steps = int(estimated_updates * 0.02)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base.train.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=estimated_updates,
        end_value=base.train.min_learning_rate,
    )
    transform = optax.chain(
        optax.clip_by_global_norm(base.train.max_grad_norm),
        optax.adamw(schedule, weight_decay=base.train.weight_decay),
    )
    optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
    trial_root = output_root or config.trial_root(arm, trial_name)
    checkpoint_root = trial_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(config.training_seed)
    initial_update = 0
    restored = restore_training_checkpoint(checkpoint_root, model, optimizer)
    if restored is not None:
        initial_update, key = restored
    stream = EffectiveUpdateStream(
        config,
        arm=arm,
        buckets=buckets,
        probe=probe,
        prepared_root=prepared_root,
    )
    content_seen, nominal_seen = _replay_accounting(
        stream, updates=initial_update, model_config=model_config
    )

    metrics_path = trial_root / "metrics.jsonl"
    _prepare_metrics(metrics_path, initial_update=initial_update)
    summary_path = trial_root / "summary.json"
    completed = _completed_summary(
        summary_path, content_seen=content_seen, content_budget=content_budget
    )
    if completed is not None:
        return completed
    content_at_start = content_seen
    started = time.perf_counter()
    update = initial_update
    last = LossComponents(
        jnp.asarray(float("nan"), jnp.float32),
        jnp.asarray(float("nan"), jnp.float32),
        jnp.asarray(float("nan"), jnp.float32),
    )
    last_update_content = 0
    while content_seen < content_budget:
        batch, bucket = next(stream)
        accumulation = batch.shape[0]
        key, update_key = jax.random.split(key)
        step_keys = jax.random.split(update_key, accumulation)
        last, gradient_norm = accumulated_train_step(
            model,
            optimizer,
            jnp.asarray(batch, dtype=jnp.int32),
            step_keys,
        )
        jax.block_until_ready(last.total)
        update += 1
        last_update_content = content_token_count(
            batch,
            pad_id=model_config.pad_id,
            doc_id=model_config.doc_id,
            eos_id=model_config.eos_id,
        )
        content_seen += last_update_content
        nominal_seen += int(batch.size)
        if update == initial_update + 1 or update % base.train.log_every == 0:
            elapsed = time.perf_counter() - started
            observation = {
                "bow_loss": float(jax.device_get(last.bow)),
                "bucket": bucket,
                "content_tokens": content_seen,
                "content_tokens_per_second": (content_seen - content_at_start)
                / max(elapsed, 1.0e-9),
                "gradient_norm": float(jax.device_get(gradient_norm)),
                "reconstruction_loss": float(jax.device_get(last.reconstruction)),
                "total_loss": float(jax.device_get(last.total)),
                "update": update,
            }
            _append_json_line(metrics_path, observation)
        if update % base.train.checkpoint_every == 0:
            save_training_checkpoint(checkpoint_root, model, optimizer, update, key)
    save_training_checkpoint(checkpoint_root, model, optimizer, update, key)
    save_checkpoint(checkpoint_root / "model", model)
    elapsed = time.perf_counter() - started
    overshoot = content_seen - content_budget
    if overshoot >= last_update_content:
        raise RuntimeError("content-token overshoot exceeds one effective update")
    summary: dict[str, object] = {
        "arm": arm,
        "bow_loss": float(jax.device_get(last.bow)),
        "bow_weight": trial.bow_weight,
        "checkpoint": str(checkpoint_root / "model"),
        "content_budget": content_budget,
        "content_tokens": content_seen,
        "content_tokens_per_second_this_run": (content_seen - content_at_start)
        / max(elapsed, 1.0e-9),
        "decoder_mask_rate": trial.decoder_mask_rate,
        "elapsed_seconds_this_run": elapsed,
        "estimated_updates": estimated_updates,
        "gradient_accumulation_by_bucket": {
            str(bucket): _integer(_resolution(probe, bucket), "gradient_accumulation")
            for bucket in buckets
        },
        "maximum_context": maximum_context,
        "nominal_token_slots": nominal_seen,
        "optimizer_updates": update,
        "overshoot_bound_content_tokens": last_update_content,
        "overshoot_content_tokens": overshoot,
        "padding_efficiency": content_seen / nominal_seen,
        "pilot_id": config.pilot_id,
        "probe_manifest": str(config.probe_manifest_path),
        "probe_manifest_sha256": _sha256(config.probe_manifest_path),
        "reconstruction_loss": float(jax.device_get(last.reconstruction)),
        "total_loss": float(jax.device_get(last.total)),
        "training_buckets": list(buckets),
        "trial": trial_name,
        "warmup_steps": warmup_steps,
        "gpu_seconds_this_run": elapsed,
    }
    _write_json_atomic(summary_path, summary)
    return summary


def resolve_round_content_budget(
    config: PilotConfig,
    trial_name: str,
    *,
    context_winner: int | None = None,
) -> int:
    """Benchmark a screening round once and reuse its immutable common budget."""
    selected = config.trial(trial_name)
    if selected.maximum_context is not None:
        round_name = "context-screen"
        trials = tuple(
            trial for trial in config.trials if trial.maximum_context is not None
        )
    else:
        if context_winner is None:
            raise ValueError(f"trial {trial_name} requires --context-winner")
        round_name = f"loss-screen-{context_winner}"
        baseline = config.trial(f"context-{context_winner}")
        trials = (
            baseline,
            *(trial for trial in config.trials if trial.inherits_context_winner),
        )
    manifest_path = config.run_root / "budgets" / f"{round_name}.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(value["content_budget"])
    probe = _load_probe(config.probe_manifest_path)
    observations: dict[str, object] = {}
    rates: list[float] = []
    update_contents: list[int] = []
    for trial in trials:
        observation = benchmark_real_batches(
            config,
            trial,
            context_winner=context_winner,
            probe=probe,
        )
        observations[trial.name] = observation
        rates.append(_floating(observation, "content_tokens_per_second"))
        update_contents.extend(
            int(value)
            for value in cast("list[int]", observation["content_tokens_per_update"])
        )
    rounding_unit = math.gcd(*update_contents)
    budget = resolve_common_content_budget(
        rates,
        seconds=config.budget_seconds,
        content_tokens_per_effective_update=rounding_unit,
    )
    _write_json_atomic(
        manifest_path,
        {
            "budget_seconds": config.budget_seconds,
            "content_budget": budget,
            "observations": observations,
            "pilot_id": config.pilot_id,
            "round": round_name,
            "rounding_content_tokens": rounding_unit,
        },
    )
    return budget


def benchmark_real_batches(
    config: PilotConfig,
    trial: PilotTrial,
    *,
    context_winner: int | None,
    probe: Mapping[str, object],
) -> dict[str, object]:
    """Measure full optimizer steps on deterministic prepared V0 microbatches."""
    maximum_context = resolve_trial_context(trial, context_winner=context_winner)
    buckets = tuple(
        value for value in trial.training_buckets if value <= maximum_context
    )
    base = ExperimentConfig.from_toml(config.experiment_config)
    model = Ettax(
        replace(
            base.model,
            decoder_mask_rate=trial.decoder_mask_rate,
            bow_weight=trial.bow_weight,
        ),
        rngs=nnx.Rngs(config.training_seed),
    )
    optimizer = nnx.Optimizer(model, optax.adamw(1.0e-4), wrt=nnx.Param)
    stream = EffectiveUpdateStream(config, arm="V0", buckets=buckets, probe=probe)
    key = jax.random.key(config.training_seed)
    warmups, measured = _calibration_batches(stream, bucket_count=len(buckets))
    for warmup_batch, _ in warmups:
        key, warmup_key = jax.random.split(key)
        accumulated_train_step(
            model,
            optimizer,
            jnp.asarray(warmup_batch, dtype=jnp.int32),
            jax.random.split(warmup_key, warmup_batch.shape[0]),
        )[0].total.block_until_ready()
    started = time.perf_counter()
    counts: list[int] = []
    for batch, _ in measured:
        key, step_key = jax.random.split(key)
        accumulated_train_step(
            model,
            optimizer,
            jnp.asarray(batch, dtype=jnp.int32),
            jax.random.split(step_key, batch.shape[0]),
        )[0].total.block_until_ready()
        counts.append(
            content_token_count(
                batch,
                pad_id=base.model.pad_id,
                doc_id=base.model.doc_id,
                eos_id=base.model.eos_id,
            )
        )
    elapsed = time.perf_counter() - started
    return {
        "content_tokens": sum(counts),
        "content_tokens_per_second": sum(counts) / max(elapsed, 1.0e-9),
        "content_tokens_per_update": counts,
        "measured_steps": len(measured),
        "trial": trial.name,
    }


def _calibration_batches(
    stream: Iterator[tuple[np.ndarray, int]], *, bucket_count: int
) -> tuple[list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    """Warm every static shape, then measure complete round-robin cycles."""
    if bucket_count < 1:
        raise ValueError("calibration requires at least one bucket")
    warmups = [next(stream) for _ in range(bucket_count)]
    measured = [next(stream) for _ in range(_CALIBRATION_CYCLES * bucket_count)]
    return warmups, measured


class EffectiveUpdateStream(Iterator[tuple[np.ndarray, int]]):
    """Deterministic round-robin stream of complete effective updates."""

    def __init__(
        self,
        config: PilotConfig,
        *,
        arm: PilotArm,
        buckets: tuple[int, ...],
        probe: Mapping[str, object],
        prepared_root: Path | None = None,
    ) -> None:
        """Open one independent Grain iterator per sequence bucket."""
        self._buckets = buckets
        self._index = 0
        self._accumulation: dict[int, int] = {}
        self._streams: dict[int, Iterator[np.ndarray]] = {}
        data_root = prepared_root or config.prepared_root
        for offset, bucket in enumerate(buckets):
            path = data_root / arm / f"{bucket}.u16"
            corpus = MemmapCorpus((path,))
            resolved = _resolution(probe, bucket)
            physical = _integer(resolved, "physical_batch")
            self._accumulation[bucket] = _integer(resolved, "gradient_accumulation")
            self._streams[bucket] = iter(
                grain_batches(
                    corpus,
                    physical,
                    seed=config.training_seed + offset,
                )
            )

    @override
    def __next__(self) -> tuple[np.ndarray, int]:
        """Return ``[accumulation, physical_batch, length]`` and its bucket."""
        bucket = self._buckets[self._index % len(self._buckets)]
        self._index += 1
        stream = self._streams[bucket]
        microbatches = [next(stream) for _ in range(self._accumulation[bucket])]
        return np.stack(microbatches), bucket


def resolve_trial_context(trial: PilotTrial, *, context_winner: int | None) -> int:
    """Resolve a static context trial or the preselected context winner."""
    if trial.maximum_context is not None:
        return trial.maximum_context
    if context_winner is None:
        raise ValueError(f"trial {trial.name} requires --context-winner")
    if context_winner not in trial.training_buckets:
        raise ValueError("context winner is not available to the inherited trial")
    return context_winner


def estimate_updates(
    config: PilotConfig,
    *,
    arm: PilotArm,
    buckets: tuple[int, ...],
    content_budget: int,
    probe: Mapping[str, object],
    prepared_root: Path | None = None,
) -> int:
    """Forecast optimizer updates from prepared padding efficiencies."""
    per_update: list[float] = []
    data_root = prepared_root or config.prepared_root
    for bucket in buckets:
        metadata_path = data_root / arm / f"{bucket}.u16.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = int(metadata["rows"])
        if rows < 1:
            continue
        average_content = float(metadata["content_tokens"]) / rows
        resolved = _resolution(probe, bucket)
        effective_batch = _integer(resolved, "physical_batch") * _integer(
            resolved, "gradient_accumulation"
        )
        per_update.append(average_content * effective_batch)
    if not per_update:
        raise ValueError("selected training buckets contain no rows")
    return max(1, math.ceil(content_budget / (sum(per_update) / len(per_update))))


def _load_probe(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing resolved probe manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("resolutions"), dict):
        raise TypeError(f"invalid resolved probe manifest: {path}")
    return value


def _resolution(probe: Mapping[str, object], bucket: int) -> Mapping[str, object]:
    resolutions = probe["resolutions"]
    if not isinstance(resolutions, dict) or str(bucket) not in resolutions:
        raise ValueError(f"probe manifest has no resolution for bucket {bucket}")
    value = resolutions[str(bucket)]
    if not isinstance(value, dict):
        raise TypeError(f"probe resolution for {bucket} is not an object")
    return value


def _integer(values: Mapping[str, object], key: str) -> int:
    return int(cast("int", values[key]))


def _floating(values: Mapping[str, object], key: str) -> float:
    return float(cast("float", values[key]))


def _reconcile_metrics(path: Path, initial_update: int) -> None:
    """Drop observations newer than the durable optimizer checkpoint."""
    if not path.is_file():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if int(value["update"]) <= initial_update:
            retained.append(line)
    _write_text_atomic(path, "\n".join(retained) + ("\n" if retained else ""))


def _prepare_metrics(path: Path, *, initial_update: int) -> None:
    """Create a new metric stream or reconcile it to a restored checkpoint."""
    if initial_update == 0:
        _write_text_atomic(path, "")
    else:
        _reconcile_metrics(path, initial_update)


def _replay_accounting(
    stream: EffectiveUpdateStream,
    *,
    updates: int,
    model_config: ModelConfig,
) -> tuple[int, int]:
    """Recover deterministic content and nominal counters for a checkpoint."""
    content_seen = 0
    nominal_seen = 0
    for _ in range(updates):
        batch, _ = next(stream)
        content_seen += content_token_count(
            batch,
            pad_id=model_config.pad_id,
            doc_id=model_config.doc_id,
            eos_id=model_config.eos_id,
        )
        nominal_seen += int(batch.size)
    return content_seen, nominal_seen


def _completed_summary(
    path: Path, *, content_seen: int, content_budget: int
) -> dict[str, object] | None:
    """Return a durable completed summary or reject inconsistent resume state."""
    if content_seen < content_budget:
        return None
    if not path.is_file():
        message = (
            "checkpoint already meets the content budget but summary.json is missing"
        )
        raise RuntimeError(message)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary["content_tokens"]) < content_budget:
        raise RuntimeError("checkpoint and summary content accounting disagree")
    return cast("dict[str, object]", summary)


def _append_json_line(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

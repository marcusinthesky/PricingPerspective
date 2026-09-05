# SPDX-License-Identifier: Apache-2.0
"""Typer command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from dotell import begin, setup_logging

from ettax import __version__

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from ettax.config import ExperimentConfig
    from ettax.pilot.config import PilotArm

DEFAULT_CONFIG = Path("src/python/apps/ettax/configs/base.toml")
DEFAULT_VINTAGES = Path("src/python/apps/ettax/configs/vintages.toml")
DEFAULT_PILOT = Path("src/python/apps/ettax/configs/pilots.toml")

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
vintage_app = typer.Typer(
    no_args_is_help=True, help="Run the governed V0/V1/V3 staircase."
)
pilot_app = typer.Typer(
    no_args_is_help=True, help="Run the isolated outcome-free capacity pilot."
)
app.add_typer(vintage_app, name="vintage")
app.add_typer(pilot_app, name="pilot")


@app.callback()
def _bootstrap(ctx: typer.Context) -> None:
    """Configure shared logging and stage telemetry for one CLI invocation."""
    setup_logging("ettax")
    if ctx.invoked_subcommand is not None and ctx.invoked_subcommand not in {
        "pilot",
        "vintage",
    }:
        ctx.call_on_close(begin(ctx.invoked_subcommand, app="ettax"))


@vintage_app.callback()
def _bootstrap_vintage(ctx: typer.Context) -> None:
    """Register one telemetry stage for a nested vintage command."""
    if ctx.invoked_subcommand is not None:
        ctx.call_on_close(begin(f"vintage.{ctx.invoked_subcommand}", app="ettax"))


@pilot_app.callback()
def _bootstrap_pilot(ctx: typer.Context) -> None:
    """Register one telemetry stage for a nested pilot command."""
    if ctx.invoked_subcommand is not None:
        ctx.call_on_close(begin(f"pilot.{ctx.invoked_subcommand}", app="ettax"))


@pilot_app.command("probe")
def pilot_probe_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_PILOT,
    reuse: Annotated[bool, typer.Option()] = False,
) -> None:
    """Search GPU capacity in a fresh JAX process for every candidate."""
    from ettax.pilot.config import PilotConfig
    from ettax.pilot.probe import run_capacity_probe, write_compact_probe_attachment

    pilot = PilotConfig.from_toml(config)
    if reuse:
        if not pilot.probe_manifest_path.is_file():
            message = f"missing resolved probe manifest: {pilot.probe_manifest_path}"
            raise FileNotFoundError(message)
        manifest = json.loads(pilot.probe_manifest_path.read_text(encoding="utf-8"))
        write_compact_probe_attachment(pilot, manifest)
    else:
        manifest = run_capacity_probe(pilot, config)
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@pilot_app.command("prepare")
def pilot_prepare_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_PILOT,
    arm: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    """Prepare exact deterministic V0/V1 samples into length buckets."""
    from ettax.pilot.config import PilotConfig
    from ettax.pilot.data import prepare_pilot_data

    selected = tuple(arm) if arm else None
    if selected is not None and not set(selected) <= {"V0", "V1", "V3"}:
        message = "--arm values must be V0, V1, or V3"
        raise typer.BadParameter(message)
    typer.echo(
        json.dumps(
            prepare_pilot_data(PilotConfig.from_toml(config), arms=selected),
            indent=2,
            sort_keys=True,
        )
    )


@pilot_app.command("run")
def pilot_run_command(
    trial: Annotated[str, typer.Argument()],
    content_budget: Annotated[int | None, typer.Option(min=1)] = None,
    arm: Annotated[str, typer.Option()] = "V0",
    context_winner: Annotated[int | None, typer.Option(min=256)] = None,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_PILOT,
    allow_cpu: Annotated[bool, typer.Option()] = False,
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train or resume one named content-matched pilot trial."""
    from ettax.pilot.config import PilotConfig
    from ettax.pilot.frozen import run_frozen_queue
    from ettax.pilot.training import resolve_round_content_budget, run_trial

    if arm not in {"V0", "V1", "V3"}:
        message = "arm must be V0, V1, or V3"
        raise typer.BadParameter(message)
    pilot = PilotConfig.from_toml(config)
    if trial == "frozen":
        if smoke:
            message = "--smoke is not valid for the frozen queue"
            raise typer.BadParameter(message)
        typer.echo(
            json.dumps(
                run_frozen_queue(pilot, require_gpu=not allow_cpu),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if content_budget is not None:
        matched_budget = content_budget
    elif smoke:
        matched_budget = 1
    else:
        matched_budget = resolve_round_content_budget(
            pilot, trial, context_winner=context_winner
        )
    result = run_trial(
        pilot,
        trial,
        arm=cast("PilotArm", arm),
        content_budget=matched_budget,
        context_winner=context_winner,
        require_gpu=not allow_cpu,
        output_root=(pilot.run_root / "smoke" / arm / trial if smoke else None),
        maximum_context_only=smoke,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@pilot_app.command("evaluate")
def pilot_evaluate_command(
    trial: Annotated[str, typer.Argument()],
    arm: Annotated[str, typer.Option()] = "V0",
    context_winner: Annotated[int | None, typer.Option(min=256)] = None,
    articles_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/shared/subsampled_headlines"
    ),
    qwen_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/shared/embeddings/qwen3-embedding-4b"
    ),
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_PILOT,
    allow_cpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Evaluate one checkpoint under all six frozen inference policies."""
    from ettax.pilot.config import PilotConfig
    from ettax.pilot.evaluation import evaluate_trial

    if arm not in {"V0", "V1", "V3"}:
        message = "arm must be V0, V1, or V3"
        raise typer.BadParameter(message)
    result = evaluate_trial(
        PilotConfig.from_toml(config),
        trial,
        arm=cast("PilotArm", arm),
        articles_dir=articles_dir,
        qwen_dir=qwen_dir,
        batch_size=batch_size,
        context_winner=context_winner,
        require_gpu=not allow_cpu,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@pilot_app.command("summarize")
def pilot_summarize_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_PILOT,
) -> None:
    """Apply eligibility/selection rules and freeze a V1-confirmed winner."""
    from ettax.pilot.config import PilotConfig
    from ettax.pilot.selection import summarize_and_freeze

    typer.echo(
        json.dumps(
            summarize_and_freeze(PilotConfig.from_toml(config)),
            indent=2,
            sort_keys=True,
        )
    )


@app.command()
def architecture(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CONFIG,
) -> None:
    """Print the resolved architecture and exact parameter count."""
    from dataclasses import asdict

    from flax import nnx

    from ettax.config import ExperimentConfig
    from ettax.model import Ettax, actual_parameter_count

    experiment = ExperimentConfig.from_toml(config)
    model = Ettax(experiment.model, rngs=nnx.Rngs(0))
    details = {
        "actual_training_parameters": actual_parameter_count(model),
        "decoder_parameters": experiment.model.decoder_parameter_count,
        "encoder_parameters": experiment.model.parameter_count,
        "global_layers": [
            index
            for index in range(experiment.model.layers)
            if index % experiment.model.global_every == 0
        ],
        "model": asdict(experiment.model),
        "nominal_tokens": experiment.nominal_tokens,
        "training_parameters": experiment.model.training_parameter_count,
        "steps": experiment.steps,
        "tokens_per_step": experiment.tokens_per_step,
        "warmup_steps": experiment.warmup_steps,
    }
    typer.echo(json.dumps(details, indent=2, sort_keys=True))


@app.command("tokenizer")
def tokenizer_command(
    inputs: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("artifacts/tokenizer.json"),
    vocab_size: Annotated[int, typer.Option(min=256, max=65_535)] = 32_768,
    text_field: Annotated[str, typer.Option()] = "text",
    max_documents: Annotated[int | None, typer.Option(min=1)] = None,
    source_documents: Annotated[int | None, typer.Option(min=1)] = None,
    sample_seed: Annotated[str, typer.Option()] = "ettax-tokenizer",
) -> None:
    """Train a byte-level BPE tokenizer."""
    from ettax.data import train_tokenizer

    train_tokenizer(
        inputs,
        output,
        vocab_size=vocab_size,
        text_field=text_field,
        max_documents=max_documents,
        source_documents=source_documents,
        sample_seed=sample_seed,
    )


@app.command()
def prepare(
    inputs: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    tokenizer: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/tokenizer.json"
    ),
    sequence_length: Annotated[int, typer.Option(min=8)] = 1_024,
    text_field: Annotated[str, typer.Option()] = "text",
    max_documents: Annotated[int | None, typer.Option(min=1)] = None,
    source_documents: Annotated[int | None, typer.Option(min=1)] = None,
    sample_seed: Annotated[str, typer.Option()] = "ettax-corpus",
) -> None:
    """Stream raw documents into a memory-mapped token matrix."""
    from dataclasses import asdict

    from ettax.data import prepare_corpus

    metadata = prepare_corpus(
        inputs,
        output,
        tokenizer,
        sequence_length=sequence_length,
        text_field=text_field,
        max_documents=max_documents,
        source_documents=source_documents,
        sample_seed=sample_seed,
    )
    typer.echo(json.dumps(asdict(metadata), indent=2, sort_keys=True))


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    no_resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run or resume the configured training stage."""
    from ettax.train import run

    typer.echo(json.dumps(run(config, resume=not no_resume), indent=2, sort_keys=True))


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    warmup: Annotated[int, typer.Option(min=1)] = 1,
    steps: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    """Benchmark full-model compiled training on the active JAX backend."""
    from ettax.train import benchmark as run_benchmark

    typer.echo(
        json.dumps(
            run_benchmark(config, warmup=warmup, steps=steps), indent=2, sort_keys=True
        )
    )


@vintage_app.command("status")
def vintage_status_command(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_VINTAGES,
) -> None:
    """Report corpus, tokenizer, preparation, and checkpoint state."""
    from ettax.vintage import VintageConfig, vintage_status

    typer.echo(
        json.dumps(
            vintage_status(VintageConfig.from_toml(config)), indent=2, sort_keys=True
        )
    )


@vintage_app.command("prepare")
def vintage_prepare_command(
    arms: Annotated[list[str] | None, typer.Argument()] = None,
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_VINTAGES,
) -> None:
    """Materialize tokenizer/shards from existing corpus files only."""
    from ettax.vintage import VintageConfig, prepare_vintages

    prepare_vintages(VintageConfig.from_toml(config), tuple(arms or ()))


@vintage_app.command("train")
def vintage_train_command(
    arm: Annotated[str, typer.Argument()],
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_VINTAGES,
    allow_cpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train or resume one arm; GPU is mandatory unless explicitly overridden."""
    from ettax.vintage import VintageConfig, train_vintage

    metrics = train_vintage(
        VintageConfig.from_toml(config),
        arm,
        require_gpu=not allow_cpu,
    )
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@vintage_app.command("run")
def vintage_run_command(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_VINTAGES,
    allow_cpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Prepare missing artifacts and sequentially train every arm."""
    from ettax.vintage import VintageConfig, run_vintages

    run_vintages(VintageConfig.from_toml(config), require_gpu=not allow_cpu)


@vintage_app.command("semantic-smoke")
def vintage_semantic_smoke_command(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_VINTAGES,
    articles_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/shared/subsampled_headlines"
    ),
    qwen_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/shared/embeddings/qwen3-embedding-4b"
    ),
    output: Annotated[Path | None, typer.Option()] = None,
    sample_size: Annotated[int, typer.Option(min=3)] = 2_048,
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
    sample_seed: Annotated[str, typer.Option()] = "t69-ettax-semantic-smoke-v1",
    legacy_tokenizer: Annotated[Path | None, typer.Option()] = None,
    mantel_sample_size: Annotated[int, typer.Option(min=3)] = 512,
    mantel_permutations: Annotated[int, typer.Option(min=1)] = 999,
    allow_cpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run outcome-free retrieval, collapse, and Qwen-geometry diagnostics."""
    from ettax.semantic import SemanticSmokeOptions, run_semantic_smoke
    from ettax.vintage import VintageConfig

    vintage = VintageConfig.from_toml(config)
    destination = output or vintage.run_root / "evaluations" / "semantic-smoke.json"
    summary = run_semantic_smoke(
        vintage,
        SemanticSmokeOptions(
            articles_dir=articles_dir,
            qwen_dir=qwen_dir,
            output=destination,
            sample_size=sample_size,
            batch_size=batch_size,
            sample_seed=sample_seed,
            legacy_tokenizer=legacy_tokenizer,
            mantel_sample_size=mantel_sample_size,
            mantel_permutations=mantel_permutations,
            require_gpu=not allow_cpu,
        ),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@vintage_app.command("embed")
def vintage_embed_command(
    arm: Annotated[str, typer.Argument()],
    articles_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/shared/subsampled_headlines"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("data/shared/embeddings/ettax"),
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_VINTAGES,
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
    allow_cpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Materialize one vintage's document embeddings as parquet shards."""
    import json

    from ettax.producer import EmbeddingRunOptions, embed_vintage
    from ettax.vintage import VintageConfig

    summary = embed_vintage(
        VintageConfig.from_toml(config),
        arm,
        EmbeddingRunOptions(
            articles_dir=articles_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            require_gpu=not allow_cpu,
        ),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def embed(
    texts: Annotated[list[str] | None, typer.Argument()] = None,
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "artifacts/checkpoints/model"
    ),
    tokenizer_path: Annotated[
        Path, typer.Option("--tokenizer", exists=True, dir_okay=False)
    ] = Path("artifacts/tokenizer.json"),
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CONFIG,
) -> None:
    """Embed command-line texts, or newline-delimited standard input."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from tokenizers import Tokenizer

    from ettax.config import ExperimentConfig
    from ettax.model import Ettax
    from ettax.train import restore_checkpoint

    experiment = ExperimentConfig.from_toml(config)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    values = texts or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    rows = [_encode(tokenizer, text, experiment) for text in values]
    model = Ettax(experiment.model, rngs=nnx.Rngs(0))
    restore_checkpoint(checkpoint, model)
    vectors = jax.device_get(model.embed(jnp.asarray(rows, dtype=jnp.int32)))
    for text, vector in zip(values, vectors, strict=True):
        typer.echo(json.dumps({"embedding": vector.tolist(), "text": text}))


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


def _encode(tokenizer: Tokenizer, text: str, experiment: ExperimentConfig) -> list[int]:
    """Encode one document with required boundary tokens."""
    model = experiment.model
    content = tokenizer.encode(text).ids[: experiment.data.sequence_length - 2]
    row = [model.doc_id, *content, model.eos_id]
    return row + [model.pad_id] * (experiment.data.sequence_length - len(row))

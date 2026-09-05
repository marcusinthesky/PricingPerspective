"""Paper 3 cross-encoder robustness command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper3RobustnessCliOptions(CliOptions):
    """Inputs for one or all Paper 3 cross-encoder robustness runs."""

    returns_dir: Path = Field(default=Path("data/shared/returns"))
    output_dir: Path = Field(
        default=Path("data/shared/ablations/paper3_model_robustness")
    )
    params_file: Path = Field(default=Path("params.yaml"))
    baseline_model: str = Field(default="qwen3-embedding-8b")
    distance_artifact_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    )
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    distance_id: str = Field(default="wasserstein_w2")
    canonical_summary: Path = Field(
        default=Path("data/papers/paper3/empirical/summary.yaml")
    )
    n_boot: int = Field(default=300)
    n_mc: int = Field(default=2000)
    n_clusters: int = Field(default=15)
    force: bool = Field(default=True)
    model: str | None = Field(default=None)


paper3_robustness_command = model_command(Paper3RobustnessCliOptions)


@commands.command("run-paper3-model-robustness", cls=paper3_robustness_command)
def cli_run_paper3_model_robustness(ctx: typer.Context) -> None:
    """Run Paper 3's W2 empirical design across encoder models."""
    own_float64()
    from pipeline.stages.substrate.paper3_model_robustness import (
        run_paper3_model_robustness,
        run_paper3_model_robustness_model,
    )

    options = options_from_context(ctx, Paper3RobustnessCliOptions)
    with options.params_file.open(encoding="utf-8") as file:
        models = list(yaml.safe_load(file)["ablation"]["grid"].keys())
    kwargs = {
        "baseline_model": options.baseline_model,
        "returns_dir": options.returns_dir,
        "output_dir": options.output_dir,
        "params_file": options.params_file,
        "distance_artifact_dir": options.distance_artifact_dir,
        "provider_id": options.provider_id,
        "representation_id": options.representation_id,
        "distance_id": options.distance_id,
        "canonical_summary": options.canonical_summary,
        "n_boot": options.n_boot,
        "n_mc": options.n_mc,
        "n_clusters": options.n_clusters,
        "force": options.force,
    }
    if options.model is not None:
        run_paper3_model_robustness_model(model=options.model, **kwargs)
        return
    run_paper3_model_robustness(models=models, **kwargs)


@commands.command("merge-paper3-model-robustness")
def cli_merge_paper3_model_robustness(
    output_dir: Path = typer.Option(
        Path("data/shared/ablations/paper3_model_robustness")
    ),
    params_file: Path = typer.Option(Path("params.yaml")),
    baseline_model: str = typer.Option("qwen3-embedding-8b"),
) -> None:
    """Merge per-model Paper 3 robustness summaries."""
    from pipeline.stages.substrate.paper3_model_robustness import (
        merge_paper3_model_robustness_summaries,
    )

    with params_file.open(encoding="utf-8") as file:
        models = list(yaml.safe_load(file)["ablation"]["grid"].keys())
    merge_paper3_model_robustness_summaries(models, baseline_model, output_dir)

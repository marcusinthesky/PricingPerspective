"""Paper 5 post-corpus QMLE command adapter."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper5PostCorpusOptions(CliOptions):
    """Inputs for pooled and annual post-corpus QMLE."""

    returns_dir: Path = Field(default=Path("data/shared/returns"))
    oos_returns_dir: Path = Field(default=Path("data/shared/oos/returns"))
    shared_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        )
    )
    distance_artifact: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    )
    co_mentions_adjacency: Path = Field(
        default=Path("data/shared/co_mentions/adjacency.parquet")
    )
    output_dir: Path = Field(default=Path("data/papers/paper5/post_corpus_qmle"))
    n_bootstrap: int = Field(default=2000)
    block_length: int = Field(default=21)
    seed: int = Field(default=0)
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    barycentre_arm_id: str = Field(default="wasserstein_w2_loo")
    geometry_id: str = Field(default="wasserstein_w2")
    pooled_only: bool = Field(default=False)
    focal_matrices_only: bool = Field(default=False)
    output_bootstrap: Path | None = Field(default=None)


paper5_post_corpus_command = model_command(Paper5PostCorpusOptions)


@commands.command("paper5-post-corpus-qmle", cls=paper5_post_corpus_command)
def cli_paper5_post_corpus_qmle(ctx: typer.Context) -> None:
    """Estimate pooled and annual QMLE on returns after the text corpus."""
    own_float64()
    from pipeline.stages.papers.paper5.post_corpus import (
        Paper5PostCorpusConfig,
        Paper5PostCorpusPaths,
        run_paper5_post_corpus_qmle,
    )

    options = options_from_context(ctx, Paper5PostCorpusOptions)
    run_paper5_post_corpus_qmle(
        Paper5PostCorpusPaths(
            returns_dir=options.returns_dir,
            oos_returns_dir=options.oos_returns_dir,
            shared_barycentre_dir=options.shared_barycentre_dir,
            distance_artifact_dir=options.distance_artifact,
            co_mentions_adjacency=options.co_mentions_adjacency,
            output_dir=options.output_dir,
        ),
        Paper5PostCorpusConfig(
            n_bootstrap=options.n_bootstrap,
            block_length=options.block_length,
            seed=options.seed,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            barycentre_arm_id=options.barycentre_arm_id,
            geometry_id=options.geometry_id,
            pooled_only=options.pooled_only,
            focal_matrices_only=options.focal_matrices_only,
            output_bootstrap=options.output_bootstrap,
        ),
    )


class Paper5UniverseHorizonBridgeOptions(CliOptions):
    """Inputs for the sequential 52-long, 52-short, and 100-short bridge."""

    returns_dir: Path = Field(default=Path("data/shared/returns"))
    current_oos_returns_dir: Path = Field(default=Path("data/shared/oos/returns"))
    current_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        )
    )
    current_results_json: Path = Field(
        default=Path("data/papers/paper5/post_corpus_qmle/results.json")
    )
    reference_oos_returns_dir: Path = Field(
        default=Path(
            "data/papers/paper5/universe_horizon_bridge/reference_52_oos_returns"
        )
    )
    reference_barycentre_dir: Path = Field(
        default=Path("data/papers/paper5/universe_horizon_bridge/reference_52_w_flat")
    )
    reference_results_json: Path = Field(
        default=Path(
            "data/papers/paper5/universe_horizon_bridge/reference_52_qmle_results.json"
        )
    )
    output_dir: Path = Field(default=Path("data/papers/paper5/universe_horizon_bridge"))
    start: str = Field(default="2023-01-01")
    short_end: str = Field(default="2026-04-06")
    long_end: str = Field(default="2026-07-15")
    expected_reference_universe: int = Field(default=52)
    expected_current_universe: int = Field(default=100)
    n_bootstrap: int = Field(default=2000)
    block_length: int = Field(default=21)
    seed: int = Field(default=0)
    shared_return_tolerance: float = Field(default=0.0)
    result_tolerance: float = Field(default=1e-12)
    reference_revision: str = Field(default="7381033e76244c68835bc232a593f9086384f339")
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    barycentre_arm_id: str = Field(default="wasserstein_w2_loo")
    geometry_id: str = Field(default="wasserstein_w2")


paper5_universe_horizon_bridge_command = model_command(
    Paper5UniverseHorizonBridgeOptions
)


@commands.command(
    "paper5-universe-horizon-bridge",
    cls=paper5_universe_horizon_bridge_command,
)
def cli_paper5_universe_horizon_bridge(ctx: typer.Context) -> None:
    """Estimate the governed sequential universe-and-horizon bridge."""
    own_float64()
    from pipeline.stages.papers.paper5.universe_horizon_bridge import (
        Paper5UniverseHorizonBridgeConfig,
        Paper5UniverseHorizonBridgePaths,
        run_paper5_universe_horizon_bridge,
    )

    options = options_from_context(ctx, Paper5UniverseHorizonBridgeOptions)
    run_paper5_universe_horizon_bridge(
        Paper5UniverseHorizonBridgePaths(
            returns_dir=options.returns_dir,
            current_oos_returns_dir=options.current_oos_returns_dir,
            current_barycentre_dir=options.current_barycentre_dir,
            current_results_json=options.current_results_json,
            reference_oos_returns_dir=options.reference_oos_returns_dir,
            reference_barycentre_dir=options.reference_barycentre_dir,
            reference_results_json=options.reference_results_json,
            output_dir=options.output_dir,
        ),
        Paper5UniverseHorizonBridgeConfig(
            start=options.start,
            short_end=options.short_end,
            long_end=options.long_end,
            expected_reference_universe=options.expected_reference_universe,
            expected_current_universe=options.expected_current_universe,
            n_bootstrap=options.n_bootstrap,
            block_length=options.block_length,
            seed=options.seed,
            shared_return_tolerance=options.shared_return_tolerance,
            result_tolerance=options.result_tolerance,
            reference_revision=options.reference_revision,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            barycentre_arm_id=options.barycentre_arm_id,
            geometry_id=options.geometry_id,
        ),
    )


@commands.command("merge-paper5-representation-ablation")
def cli_merge_paper5_representation_ablation(
    params_file: Path = typer.Option(Path("params.yaml")),
    cells_dir: Path = typer.Option(Path("data/papers/paper5/representation_ablation")),
    canonical_results: Path = typer.Option(
        Path("data/papers/paper5/post_corpus_qmle/results.json")
    ),
    output_summary: Path = typer.Option(
        Path("data/papers/paper5/representation_ablation/summary.json")
    ),
    bootstrap_dir: Path | None = typer.Option(None),
) -> None:
    """Merge ten pooled Paper 5 representation cells and paired contrasts."""
    from pipeline.stages.papers.paper5.representation_ablation import (
        Paper5RepresentationMergePaths,
        merge_paper5_representation_ablation,
    )

    merge_paper5_representation_ablation(
        Paper5RepresentationMergePaths(
            params_file=params_file,
            cells_dir=cells_dir,
            canonical_results=canonical_results,
            output_summary=output_summary,
            bootstrap_dir=bootstrap_dir,
        )
    )

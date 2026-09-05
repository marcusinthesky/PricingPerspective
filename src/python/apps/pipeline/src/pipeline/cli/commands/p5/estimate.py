"""P5 Estimate CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper5GateOptions(CliOptions):
    """Inputs for the preregistered post-corpus economic gate."""

    returns_dir: Path = Field(
        default=Path("data/shared/returns"),
        description="Directory of 2018--2022 per-ticker return parquets.",
    )
    oos_returns_dir: Path = Field(
        default=Path("data/shared/oos/returns"),
        description="Directory of post-corpus per-ticker return parquets.",
    )
    shared_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        ),
        description="Frozen target-anchored W2 projection artifact.",
    )
    universe_csv: Path = Field(
        default=Path("data/universe.csv"),
        description="Universe classifications for industry and sector baskets.",
    )
    output_dir: Path = Field(
        default=Path("data/papers/paper5/wflat_gate"),
        description="Output directory for gate results.json and results.parquet.",
    )
    n_bootstrap: int = Field(default=2000, description="Stationary-bootstrap draws.")
    n_random: int = Field(default=1000, description="Matched random baskets.")
    block_length: int = Field(
        default=21,
        description="Expected bootstrap block length.",
    )
    knn_k: int = Field(default=5, description="Training-window correlation neighbours.")
    seed: int = Field(default=0, description="Fixed preregistered random seed.")
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    barycentre_arm_id: str = Field(default="wasserstein_w2_loo")
    geometry_id: str = Field(default="wasserstein_w2")


paper5_gate_command = model_command(Paper5GateOptions)


@commands.command("paper5-economic-gate", cls=paper5_gate_command)
def cli_paper5_economic_gate(ctx: typer.Context) -> None:
    """Run the frozen-design 2023--2026 W-flat economic gate."""
    own_float64()
    from pipeline.stages.papers.paper5.economic_gate import (
        Paper5GateConfig,
        Paper5GatePaths,
        run_paper5_economic_gate,
    )

    options = options_from_context(ctx, Paper5GateOptions)
    run_paper5_economic_gate(
        Paper5GatePaths(
            returns_dir=options.returns_dir,
            oos_returns_dir=options.oos_returns_dir,
            shared_barycentre_dir=options.shared_barycentre_dir,
            universe_csv=options.universe_csv,
            output_dir=options.output_dir,
        ),
        Paper5GateConfig(
            n_bootstrap=options.n_bootstrap,
            n_random=options.n_random,
            block_length=options.block_length,
            knn_k=options.knn_k,
            seed=options.seed,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            barycentre_arm_id=options.barycentre_arm_id,
            geometry_id=options.geometry_id,
        ),
    )


class Paper5EmpiricalOptions(CliOptions):
    """Inputs for Paper 5's Hilbert/W2 spatial empirics."""

    returns_dir: Path = Field(
        default=Path("data/shared/returns"),
        description="Directory of per-ticker return parquets.",
    )
    distance_artifact_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        description="Shared typed statistical-distance artifact directory.",
    )
    embeddings_dir: Path = Field(
        default=Path("data/shared/embeddings/qwen3-embedding-8b"),
        description="Directory of per-ticker embedding parquets.",
    )
    universe_csv: Path = Field(
        default=Path("data/universe.csv"),
        description="Universe metadata CSV.",
    )
    output_dir: Path = Field(
        default=Path("data/papers/paper5"),
        description="Output directory for results.json + results.parquet.",
    )
    n_boot_stat: int = Field(
        default=300,
        description="Bootstrap resamples for test statistics.",
    )
    knn_k: int = Field(
        default=5,
        description="Number of nearest embedding neighbours.",
    )
    seed: int = Field(default=0, description="Random seed.")
    provider_id: str = Field(
        default="qwen3-embedding-8b", description="Expected typed provider identity."
    )
    representation_id: str = Field(
        default="qwen3-embedding-8b-unit",
        description="Expected typed representation identity.",
    )
    distance_id: str = Field(
        default="wasserstein_w2", description="Expected typed distance identity."
    )
    barycentre_arm_id: str = Field(
        default="wasserstein_w2_loo",
        description="Expected typed W2 barycentre arm identity.",
    )
    wasserstein_w1_arm_id: str = Field(
        default="wasserstein_w1_loo",
        description="Expected typed leave-one-out W1 barycentre arm identity.",
    )
    shared_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        ),
        description=(
            "Shared typed barycentre artifact directory containing "
            "weights.parquet + diagnostics.parquet."
        ),
    )
    wasserstein_w1_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w1_loo"
        ),
        description="Shared leave-one-out target-anchored W1 barycentre artifact.",
    )
    co_mentions_adjacency: Path = Field(
        default=Path("data/shared/co_mentions/adjacency.parquet"),
        description="Persistent weighted news co-mention edge artifact.",
    )
    oos_returns_dir: Path | None = Field(
        default=None,
        description=(
            "Optional OOS return-parquet directory (e.g. data/shared/oos/returns) "
            "unioned with returns_dir at the 2022/2023 seam and disclosed in "
            "results.json under returns_seam. Default: None (identical behavior "
            "to today)."
        ),
    )


paper5_empirical_command = model_command(Paper5EmpiricalOptions)


@commands.command("paper5-empirical", cls=paper5_empirical_command)
def cli_paper5_empirical(ctx: typer.Context) -> None:
    """Run Paper 5's Hilbert/W2 spatial barycentre empirics."""
    own_float64()
    from pipeline.stages.papers.paper5.run import (
        Paper5EmpiricalConfig,
        Paper5EmpiricalPaths,
        run_paper5_empirical,
    )

    options = options_from_context(ctx, Paper5EmpiricalOptions)
    run_paper5_empirical(
        Paper5EmpiricalPaths(
            returns_dir=options.returns_dir,
            distance_artifact_dir=options.distance_artifact_dir,
            embeddings_dir=options.embeddings_dir,
            universe_csv=options.universe_csv,
            output_dir=options.output_dir,
            shared_barycentre_dir=options.shared_barycentre_dir,
            wasserstein_w1_barycentre_dir=options.wasserstein_w1_barycentre_dir,
            co_mentions_adjacency=options.co_mentions_adjacency,
            oos_returns_dir=options.oos_returns_dir,
        ),
        Paper5EmpiricalConfig(
            n_boot_stat=options.n_boot_stat,
            knn_k=options.knn_k,
            seed=options.seed,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            distance_id=options.distance_id,
            barycentre_arm_id=options.barycentre_arm_id,
            wasserstein_w1_arm_id=options.wasserstein_w1_arm_id,
        ),
    )

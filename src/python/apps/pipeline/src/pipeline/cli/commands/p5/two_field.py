"""Paper 5 joint two-field QMLE command adapter."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper5TwoFieldOptions(CliOptions):
    """Inputs for the joint two-field estimates and overlap diagnostics."""

    returns_dir: Path = Field(default=Path("data/shared/returns"))
    oos_returns_dir: Path = Field(default=Path("data/shared/oos/returns"))
    shared_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        )
    )
    co_mentions_adjacency: Path = Field(
        default=Path("data/shared/co_mentions/adjacency.parquet")
    )
    output_dir: Path = Field(default=Path("data/papers/paper5/two_field_qmle"))
    figures_dir: Path = Field(
        default=Path("src/latex/projects/05_spatial_pricing/src/images")
    )
    n_bootstrap: int = Field(default=2000)
    block_length: int = Field(default=21)
    seed: int = Field(default=0)
    n_random: int = Field(default=1000)
    theta_points: int = Field(default=401)
    rho_points: int = Field(default=1000)
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    barycentre_arm_id: str = Field(default="wasserstein_w2_loo")
    geometry_id: str = Field(default="wasserstein_w2")


paper5_two_field_command = model_command(Paper5TwoFieldOptions)


@commands.command("paper5-two-field-qmle", cls=paper5_two_field_command)
def cli_paper5_two_field_qmle(ctx: typer.Context) -> None:
    """Estimate the joint two-field model and compare the interaction fields."""
    own_float64()
    from pipeline.stages.papers.paper5.two_field_stage import (
        Paper5TwoFieldConfig,
        Paper5TwoFieldPaths,
        run_paper5_two_field_qmle,
    )

    options = options_from_context(ctx, Paper5TwoFieldOptions)
    run_paper5_two_field_qmle(
        Paper5TwoFieldPaths(
            returns_dir=options.returns_dir,
            oos_returns_dir=options.oos_returns_dir,
            shared_barycentre_dir=options.shared_barycentre_dir,
            co_mentions_adjacency=options.co_mentions_adjacency,
            output_dir=options.output_dir,
            figures_dir=options.figures_dir,
        ),
        Paper5TwoFieldConfig(
            n_bootstrap=options.n_bootstrap,
            block_length=options.block_length,
            seed=options.seed,
            n_random=options.n_random,
            theta_points=options.theta_points,
            rho_points=options.rho_points,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            barycentre_arm_id=options.barycentre_arm_id,
            geometry_id=options.geometry_id,
        ),
    )

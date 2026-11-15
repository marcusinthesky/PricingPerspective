"""Paper 5 joint two-field QMLE: compute and render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper5-two-field-qmle")
def cli_paper5_two_field_qmle(
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    oos_returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2023--2026 return panel."),
    ],
    shared_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen W-flat artifact."),
    ],
    co_mentions_adjacency: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Co-mention adjacency."),
    ],
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="params.yaml; reads paper5.factor_data."
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
    n_bootstrap: Annotated[int, typer.Option(min=1, help="Bootstrap draws.")],
    block_length: Annotated[
        int, typer.Option(min=1, help="Expected stationary-bootstrap block length.")
    ],
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")],
    n_random: Annotated[
        int, typer.Option(min=1, help="Matched-sparsity random operators.")
    ],
    theta_points: Annotated[int, typer.Option(min=1, help="Mixing-angle grid points.")],
    rho_points: Annotated[int, typer.Option(min=1, help="Rho grid points.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    barycentre_arm_id: Annotated[str, typer.Option(help="Barycentre arm identity.")],
    geometry_id: Annotated[str, typer.Option(help="Ground geometry identity.")],
    factor_model: Annotated[str, typer.Option(help="Factor model.")],
) -> None:
    """Estimate the joint two-field model and compare the interaction fields."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.two_field_stage import (
        Paper5TwoFieldConfig,
        Paper5TwoFieldPaths,
        run_paper5_two_field_qmle,
    )

    run_paper5_two_field_qmle(
        Paper5TwoFieldPaths(
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            shared_barycentre_dir=shared_barycentre_dir,
            co_mentions_adjacency=co_mentions_adjacency,
            params_file=params_file,
            output_dir=output_dir,
        ),
        Paper5TwoFieldConfig(
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
            n_random=n_random,
            theta_points=theta_points,
            rho_points=rho_points,
            provider_id=provider_id,
            representation_id=representation_id,
            barycentre_arm_id=barycentre_arm_id,
            geometry_id=geometry_id,
            factor_model=factor_model,
        ),
    )


@commands.command("render-paper5-two-field-qmle")
def cli_render_paper5_two_field_qmle(
    results_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Two-field result folder (results.json, surface/bootstrap Parquet).",
        ),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Two-field region PGF to write.")
    ],
) -> None:
    """Render the joint two-field region from the saved estimation results."""
    from pipeline.figures.paper5.two_field import render_paper5_two_field_region

    render_paper5_two_field_region(results_dir, output_path)

"""Paper 5 factor-conditioned economic-value comparison: compute and render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("paper5-economic-value")
def cli_paper5_economic_value(
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    oos_returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2023--2026 return panel."),
    ],
    w1_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen W1 artifact."),
    ],
    w2_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen W2 artifact."),
    ],
    distance_artifact_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Typed W2 distances."),
    ],
    embeddings_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Provider embedding shards."),
    ],
    co_mentions_adjacency: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Co-mention adjacency."),
    ],
    universe_csv: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Universe classifications."),
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
    calibration_start: Annotated[str, typer.Option(help="Calibration start date.")],
    calibration_end: Annotated[str, typer.Option(help="Calibration end date.")],
    inner_training_end: Annotated[
        str, typer.Option(help="End of the inner bandwidth-training window.")
    ],
    inner_validation_start: Annotated[
        str, typer.Option(help="Start of the inner bandwidth-validation window.")
    ],
    evaluation_start: Annotated[str, typer.Option(help="Evaluation start date.")],
    evaluation_end: Annotated[str, typer.Option(help="Evaluation end date.")],
    confidence_level: Annotated[float, typer.Option(help="Interval coverage.")],
    factor_model: Annotated[Literal["ff5_mom"], typer.Option(help="Factor model.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    w1_barycentre_arm_id: Annotated[str, typer.Option(help="W1 barycentre arm.")],
    w2_barycentre_arm_id: Annotated[str, typer.Option(help="W2 barycentre arm.")],
    w1_geometry_id: Annotated[str, typer.Option(help="W1 geometry identity.")],
    w2_geometry_id: Annotated[str, typer.Option(help="W2 geometry identity.")],
    rbf_multipliers: Annotated[
        str, typer.Option(help="Comma-separated RBF bandwidth multipliers.")
    ],
    active_weight_tolerance: Annotated[
        float, typer.Option(help="Threshold for an active W-flat weight.")
    ],
) -> None:
    """Compare factor-conditioned interaction fields and describe W2 peers."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.economic_value import (
        Paper5EconomicValueConfig,
        Paper5EconomicValuePaths,
        run_paper5_economic_value,
    )

    run_paper5_economic_value(
        Paper5EconomicValuePaths(
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            w1_barycentre_dir=w1_barycentre_dir,
            w2_barycentre_dir=w2_barycentre_dir,
            distance_artifact_dir=distance_artifact_dir,
            embeddings_dir=embeddings_dir,
            co_mentions_adjacency=co_mentions_adjacency,
            universe_csv=universe_csv,
            params_file=params_file,
            output_dir=output_dir,
        ),
        Paper5EconomicValueConfig(
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            inner_training_end=inner_training_end,
            inner_validation_start=inner_validation_start,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            confidence_level=confidence_level,
            factor_model=factor_model,
            provider_id=provider_id,
            representation_id=representation_id,
            w1_barycentre_arm_id=w1_barycentre_arm_id,
            w2_barycentre_arm_id=w2_barycentre_arm_id,
            w1_geometry_id=w1_geometry_id,
            w2_geometry_id=w2_geometry_id,
            rbf_multipliers=tuple(float(value) for value in rbf_multipliers.split(",")),
            active_weight_tolerance=active_weight_tolerance,
        ),
    )


@commands.command("render-paper5-economic-value")
def cli_render_paper5_economic_value(
    results_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Economic-value results.json."),
    ],
    comparator_output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Field-comparison table fragment.")
    ],
    peer_output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Peer-interpretation table fragment.")
    ],
) -> None:
    """Render Paper 5's factor-conditioned comparison and peer panel."""
    from pipeline.figures.paper5.economic_value import (
        render_paper5_economic_value_tables,
    )

    render_paper5_economic_value_tables(
        results_path, comparator_output_path, peer_output_path
    )

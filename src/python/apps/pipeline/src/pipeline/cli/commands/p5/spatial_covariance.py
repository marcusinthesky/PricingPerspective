"""Paper 5 exploratory spatial covariance command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("paper5-spatial-covariance")
def cli_paper5_spatial_covariance(
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
    transport_results_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Transport-order results.json."),
    ],
    spatial_value_results_json: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Spatial-value comparison results.json."
        ),
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
    calibration_start: Annotated[str, typer.Option(help="Calibration start date.")],
    calibration_end: Annotated[str, typer.Option(help="Calibration end date.")],
    evaluation_start: Annotated[str, typer.Option(help="Evaluation start date.")],
    evaluation_end: Annotated[str, typer.Option(help="Evaluation end date.")],
    factor_model: Annotated[Literal["ff5_mom"], typer.Option(help="Factor model.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    w1_barycentre_arm_id: Annotated[str, typer.Option(help="W1 barycentre arm.")],
    w2_barycentre_arm_id: Annotated[str, typer.Option(help="W2 barycentre arm.")],
    w1_geometry_id: Annotated[str, typer.Option(help="W1 geometry identity.")],
    w2_geometry_id: Annotated[str, typer.Option(help="W2 geometry identity.")],
) -> None:
    """Estimate exploratory QMLE and Ledoit--Wolf spatial covariances."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.spatial_covariance import (
        Paper5SpatialCovarianceConfig,
        Paper5SpatialCovariancePaths,
        run_paper5_spatial_covariance,
    )

    run_paper5_spatial_covariance(
        Paper5SpatialCovariancePaths(
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            w1_barycentre_dir=w1_barycentre_dir,
            w2_barycentre_dir=w2_barycentre_dir,
            transport_results_json=transport_results_json,
            spatial_value_results_json=spatial_value_results_json,
            params_file=params_file,
            output_dir=output_dir,
        ),
        Paper5SpatialCovarianceConfig(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            factor_model=factor_model,
            provider_id=provider_id,
            representation_id=representation_id,
            w1_barycentre_arm_id=w1_barycentre_arm_id,
            w2_barycentre_arm_id=w2_barycentre_arm_id,
            w1_geometry_id=w1_geometry_id,
            w2_geometry_id=w2_geometry_id,
        ),
    )

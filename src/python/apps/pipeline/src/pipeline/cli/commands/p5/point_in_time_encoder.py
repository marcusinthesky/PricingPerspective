"""Paper 5 point-in-time encoder sensitivity: compute and render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("paper5-point-in-time-encoder")
def cli_paper5_point_in_time_encoder(
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    oos_returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2023--2026 return panel."),
    ],
    w2_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="ChronoBERT 2017 W2 artifact."),
    ],
    distance_artifact_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="ChronoBERT 2017 W2 distances."
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
    n_bootstrap: Annotated[int, typer.Option(min=1, help="Bootstrap draws.")],
    block_length: Annotated[
        int, typer.Option(min=1, help="Expected stationary-bootstrap block length.")
    ],
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")],
    calibration_start: Annotated[str, typer.Option(help="Calibration start date.")],
    calibration_end: Annotated[str, typer.Option(help="Calibration end date.")],
    evaluation_start: Annotated[str, typer.Option(help="Evaluation start date.")],
    evaluation_end: Annotated[str, typer.Option(help="Evaluation end date.")],
    confidence_level: Annotated[float, typer.Option(help="Interval coverage.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    w2_barycentre_arm_id: Annotated[str, typer.Option(help="W2 barycentre arm.")],
    w2_geometry_id: Annotated[str, typer.Option(help="W2 geometry identity.")],
    factor_model: Annotated[Literal["ff5_mom"], typer.Option(help="Factor model.")],
) -> None:
    """Run the primary factor-conditioned pre-period encoder sensitivity."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.point_in_time import (
        Paper5PointInTimeConfig,
        Paper5PointInTimePaths,
        run_paper5_point_in_time,
    )

    run_paper5_point_in_time(
        Paper5PointInTimePaths(
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            w2_barycentre_dir=w2_barycentre_dir,
            distance_artifact_dir=distance_artifact_dir,
            params_file=params_file,
            output_dir=output_dir,
        ),
        Paper5PointInTimeConfig(
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            confidence_level=confidence_level,
            provider_id=provider_id,
            representation_id=representation_id,
            w2_barycentre_arm_id=w2_barycentre_arm_id,
            w2_geometry_id=w2_geometry_id,
            factor_model=factor_model,
        ),
    )


@commands.command("render-paper5-point-in-time-encoder")
def cli_render_paper5_point_in_time_encoder(
    results_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Encoder results.json."),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated table fragment.")
    ],
) -> None:
    """Render the pre-period ChronoBERT 2017 factor-conditioned sensitivity."""
    from pipeline.figures.paper5.point_in_time import (
        render_paper5_point_in_time_table,
    )

    render_paper5_point_in_time_table(results_path, output_path)

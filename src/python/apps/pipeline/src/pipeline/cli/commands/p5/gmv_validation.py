"""Paper 5 GMV validation (exploratory portfolio comparison) command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("paper5-gmv-validation")
def cli_paper5_gmv_validation(
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads paper3.pit_vintage_grid and "
            "paper3.certificate_feasibility.news_reference.",
        ),
    ],
    returns_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Calibration returns.")
    ],
    oos_returns_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Holdout returns.")
    ],
    certificate_pit_w2_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Paper 3 PIT squared-W2 .npz."),
    ],
    certificate_pit_manifest_path: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Provenance manifest of the PIT archive."
        ),
    ],
    spatial_covariance_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Paper 5 spatial covariance artifact."
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
    calibration_start: Annotated[str, typer.Option(help="Calibration start date.")],
    calibration_end: Annotated[str, typer.Option(help="Calibration end date.")],
    evaluation_start: Annotated[str, typer.Option(help="Evaluation start date.")],
    evaluation_end: Annotated[str, typer.Option(help="Evaluation end date.")],
    maximum_weight: Annotated[float, typer.Option(help="Common long-only cap.")],
    reference_draws: Annotated[int, typer.Option(help="Reference-population draws.")],
    reference_seed: Annotated[int, typer.Option(help="Reference-population seed.")],
    reference_design_label: Annotated[
        str, typer.Option(help="Paper 3 reference design used as the primary law.")
    ],
    formation_vintage: Annotated[
        str, typer.Option(help="PIT certificate vintage; equals calibration end.")
    ],
    risk_unit: Annotated[
        Literal["excess_return"], typer.Option(help="Common risk unit.")
    ],
    factor_model: Annotated[Literal["ff5_mom"], typer.Option(help="Factor model.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    w1_barycentre_arm_id: Annotated[str, typer.Option(help="W1 barycentre arm.")],
    w2_barycentre_arm_id: Annotated[str, typer.Option(help="W2 barycentre arm.")],
    w1_geometry_id: Annotated[str, typer.Option(help="W1 geometry identity.")],
    w2_geometry_id: Annotated[str, typer.Option(help="W2 geometry identity.")],
    residual_shrinkage_target: Annotated[
        Literal["constant_correlation"],
        typer.Option(help="Residual Ledoit--Wolf shrinkage target."),
    ],
) -> None:
    """Compare frozen Paper 3 certificate and Paper 5 spatial GMV portfolios."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.gmv_validation import (
        Paper5GMVValidationConfig,
        Paper5GMVValidationPaths,
        run_paper5_gmv_validation,
    )

    run_paper5_gmv_validation(
        Paper5GMVValidationPaths(
            params_file=params_file,
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            certificate_pit_w2_path=certificate_pit_w2_path,
            certificate_pit_manifest_path=certificate_pit_manifest_path,
            spatial_covariance_dir=spatial_covariance_dir,
            output_dir=output_dir,
        ),
        Paper5GMVValidationConfig(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            maximum_weight=maximum_weight,
            reference_draws=reference_draws,
            reference_seed=reference_seed,
            reference_design_label=reference_design_label,
            formation_vintage=formation_vintage,
            risk_unit=risk_unit,
            factor_model=factor_model,
            provider_id=provider_id,
            representation_id=representation_id,
            w1_barycentre_arm_id=w1_barycentre_arm_id,
            w2_barycentre_arm_id=w2_barycentre_arm_id,
            w1_geometry_id=w1_geometry_id,
            w2_geometry_id=w2_geometry_id,
            residual_shrinkage_target=residual_shrinkage_target,
        ),
    )


@commands.command("render-paper5-gmv-validation")
def cli_render_paper5_gmv_validation(
    results_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="GMV results.parquet."),
    ],
    results_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="GMV scalar results.json."),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated table fragment.")
    ],
) -> None:
    """Render Paper 5's appendix-only covariance allocation diagnostic."""
    from pipeline.figures.paper5.gmv_validation import (
        render_paper5_gmv_validation_table,
    )

    render_paper5_gmv_validation_table(results_path, results_json, output_path)

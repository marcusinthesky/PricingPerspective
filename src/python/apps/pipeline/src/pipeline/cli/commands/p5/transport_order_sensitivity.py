"""Paper 5 factor-conditioned W1/W2 transport-order sensitivity adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("paper5-transport-order-sensitivity")
def cli_paper5_transport_order_sensitivity(
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
    equivalence_alpha: Annotated[float, typer.Option(help="TOST level.")],
    predictive_equivalence_gain_share: Annotated[
        float, typer.Option(help="Predictive-equivalence margin as a gain share.")
    ],
    minimum_signal_correlation: Annotated[
        float, typer.Option(help="Minimum W1/W2 signal correlation.")
    ],
    maximum_relative_signal_difference_sd: Annotated[
        float, typer.Option(help="Maximum relative signal-difference SD.")
    ],
    active_weight_tolerance: Annotated[
        float, typer.Option(help="Threshold for an active operator weight.")
    ],
    incremental_qlr_alpha: Annotated[
        float, typer.Option(help="Incremental QLR level.")
    ],
    joint_theta_points: Annotated[
        int, typer.Option(min=1, help="Joint-fit mixing-angle grid points.")
    ],
    joint_rho_points: Annotated[
        int, typer.Option(min=1, help="Joint-fit rho grid points.")
    ],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    w1_barycentre_arm_id: Annotated[str, typer.Option(help="W1 barycentre arm.")],
    w2_barycentre_arm_id: Annotated[str, typer.Option(help="W2 barycentre arm.")],
    w1_geometry_id: Annotated[str, typer.Option(help="W1 geometry identity.")],
    w2_geometry_id: Annotated[str, typer.Option(help="W2 geometry identity.")],
    factor_model: Annotated[Literal["ff5_mom"], typer.Option(help="Factor model.")],
) -> None:
    """Compare nested factor sets and target-anchored W1/W2 fields."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.transport_order import (
        Paper5TransportOrderConfig,
        Paper5TransportOrderPaths,
        run_paper5_transport_order_sensitivity,
    )

    run_paper5_transport_order_sensitivity(
        Paper5TransportOrderPaths(
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            w1_barycentre_dir=w1_barycentre_dir,
            w2_barycentre_dir=w2_barycentre_dir,
            params_file=params_file,
            output_dir=output_dir,
        ),
        Paper5TransportOrderConfig(
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            confidence_level=confidence_level,
            equivalence_alpha=equivalence_alpha,
            predictive_equivalence_gain_share=predictive_equivalence_gain_share,
            minimum_signal_correlation=minimum_signal_correlation,
            maximum_relative_signal_difference_sd=maximum_relative_signal_difference_sd,
            active_weight_tolerance=active_weight_tolerance,
            incremental_qlr_alpha=incremental_qlr_alpha,
            joint_theta_points=joint_theta_points,
            joint_rho_points=joint_rho_points,
            provider_id=provider_id,
            representation_id=representation_id,
            w1_barycentre_arm_id=w1_barycentre_arm_id,
            w2_barycentre_arm_id=w2_barycentre_arm_id,
            w1_geometry_id=w1_geometry_id,
            w2_geometry_id=w2_geometry_id,
            factor_model=factor_model,
        ),
    )

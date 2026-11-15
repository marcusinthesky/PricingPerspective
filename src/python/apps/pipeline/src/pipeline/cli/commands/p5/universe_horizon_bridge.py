"""Paper 5 sequential universe-and-horizon bridge: compute and render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper5-universe-horizon-bridge")
def cli_paper5_universe_horizon_bridge(
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    current_oos_returns_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Current 100-firm holdout returns."
        ),
    ],
    current_barycentre_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Current frozen W-flat artifact."
        ),
    ],
    current_results_json: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Current post-corpus QMLE results."
        ),
    ],
    reference_oos_returns_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Frozen 52-firm holdout returns."
        ),
    ],
    reference_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen 52-firm W-flat."),
    ],
    reference_results_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Frozen 52-firm QMLE results."),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
    start: Annotated[str, typer.Option(help="Evaluation start date.")],
    short_end: Annotated[str, typer.Option(help="Governed short-horizon end.")],
    long_end: Annotated[str, typer.Option(help="Historical long-horizon end.")],
    expected_reference_universe: Annotated[
        int, typer.Option(min=1, help="Reference universe size.")
    ],
    expected_current_universe: Annotated[
        int, typer.Option(min=1, help="Current universe size.")
    ],
    n_bootstrap: Annotated[int, typer.Option(min=1, help="Bootstrap draws.")],
    block_length: Annotated[
        int, typer.Option(min=1, help="Expected stationary-bootstrap block length.")
    ],
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")],
    shared_return_tolerance: Annotated[
        float, typer.Option(help="Tolerance on returns common to both panels.")
    ],
    result_tolerance: Annotated[
        float, typer.Option(help="Tolerance when reproducing recorded results.")
    ],
    reference_revision: Annotated[
        str, typer.Option(help="Git revision that produced the reference inputs.")
    ],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    barycentre_arm_id: Annotated[str, typer.Option(help="Barycentre arm identity.")],
    geometry_id: Annotated[str, typer.Option(help="Ground geometry identity.")],
) -> None:
    """Estimate the governed sequential universe-and-horizon bridge."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.universe_horizon_bridge import (
        Paper5UniverseHorizonBridgeConfig,
        Paper5UniverseHorizonBridgePaths,
        run_paper5_universe_horizon_bridge,
    )

    run_paper5_universe_horizon_bridge(
        Paper5UniverseHorizonBridgePaths(
            returns_dir=returns_dir,
            current_oos_returns_dir=current_oos_returns_dir,
            current_barycentre_dir=current_barycentre_dir,
            current_results_json=current_results_json,
            reference_oos_returns_dir=reference_oos_returns_dir,
            reference_barycentre_dir=reference_barycentre_dir,
            reference_results_json=reference_results_json,
            output_dir=output_dir,
        ),
        Paper5UniverseHorizonBridgeConfig(
            start=start,
            short_end=short_end,
            long_end=long_end,
            expected_reference_universe=expected_reference_universe,
            expected_current_universe=expected_current_universe,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
            shared_return_tolerance=shared_return_tolerance,
            result_tolerance=result_tolerance,
            reference_revision=reference_revision,
            provider_id=provider_id,
            representation_id=representation_id,
            barycentre_arm_id=barycentre_arm_id,
            geometry_id=geometry_id,
        ),
    )


@commands.command("render-paper5-universe-horizon-bridge")
def cli_render_paper5_universe_horizon_bridge(
    results_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Bridge results.json."),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated table fragment.")
    ],
) -> None:
    """Render Paper 5's sequential universe-and-horizon bridge."""
    from pipeline.figures.paper5.universe_horizon_bridge import (
        render_paper5_universe_horizon_bridge_table,
    )

    render_paper5_universe_horizon_bridge_table(results_path, output_path)

"""Paper 5 Matryoshka W1/W2 mechanism ladder: compute and render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("paper5-transport-dimension-ladder")
def cli_paper5_transport_dimension_ladder(
    embeddings_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Provider embedding folder."),
    ],
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    oos_returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2023--2026 return panel."),
    ],
    barycentres_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Shared barycentres root."),
    ],
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help=(
                "params.yaml; reads the Qwen3-8B ablation cells, paper5.factor_data, "
                "typed_analysis providers/representations/distances and "
                "market_symbols."
            ),
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
) -> None:
    """Trace W1/W2 equivalence through normalized Matryoshka prefixes."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.transport_dimension import (
        Paper5TransportDimensionConfig,
        Paper5TransportDimensionPaths,
        run_paper5_transport_dimension_ladder,
    )

    run_paper5_transport_dimension_ladder(
        Paper5TransportDimensionPaths(
            embeddings_dir=embeddings_dir,
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            barycentres_root=barycentres_root,
            params_file=params_file,
            output_dir=output_dir,
        ),
        Paper5TransportDimensionConfig(
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            factor_model=factor_model,
        ),
    )


@commands.command("render-paper5-transport-dimension-ladder")
def cli_render_paper5_transport_dimension_ladder(
    results_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Ladder results.json."),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated table fragment.")
    ],
) -> None:
    """Render Paper 5's native-scale Matryoshka W1/W2 mechanism table."""
    from pipeline.figures.paper5.transport_dimension import (
        render_paper5_transport_dimension_table,
    )

    render_paper5_transport_dimension_table(results_path, output_path)

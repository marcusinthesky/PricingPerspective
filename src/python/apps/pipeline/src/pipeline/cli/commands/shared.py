"""CLI adapters for provider-agnostic typed shared artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

commands = typer.Typer()


@commands.command("render-sample-universe-table")
def cli_render_sample_universe_table(
    universe_csv: Annotated[
        Path,
        typer.Option(help="Universe metadata CSV with symbol, name, and sector."),
    ],
    returns_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing one adjusted-return parquet per ticker."
        ),
    ],
    distance_artifact_dir: Annotated[
        Path, typer.Option(help="Governed typed W2 distance artifact directory.")
    ],
    provider_id: Annotated[
        str, typer.Option(help="Expected typed-distance provider identity.")
    ],
    representation_id: Annotated[
        str, typer.Option(help="Expected typed-distance representation identity.")
    ],
    distance_id: Annotated[str, typer.Option(help="Expected typed-distance identity.")],
    output_file: Annotated[Path, typer.Option(help="Generated LaTeX table fragment.")],
    layout: Annotated[
        Literal["longtable", "sector-list"],
        typer.Option(help="Per-firm longtable or compact per-sector ticker list."),
    ] = "longtable",
) -> None:
    """Render a canonical priced W2 roster as a LaTeX appendix table."""
    from pipeline.figures.shared_tables import render_sample_universe_table

    render_sample_universe_table(
        universe_csv=universe_csv,
        distance_artifact_dir=distance_artifact_dir,
        returns_dir=returns_dir,
        output_file=output_file,
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id=distance_id,
        layout=layout,
    )


@commands.command("typed-geometry-component")
def cli_typed_geometry_component(
    provider_id: Annotated[str, typer.Option(help="Registered provider identity.")],
    representation_id: Annotated[
        str, typer.Option(help="Registered representation identity.")
    ],
    geometry_id: Annotated[
        str, typer.Option(help="Registered energy or MMD geometry identity.")
    ],
    embeddings_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            readable=True,
            help="Provider Parquet directory supplied by the stage graph.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option(help="Component artifact directory.")],
    params_file: Annotated[
        Path, typer.Option(help="params.yaml carrying the typed registry.")
    ],
) -> None:
    """Materialize one reusable raw energy or MMD geometry component."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.shared.typed_geometry_components import (
        run_typed_geometry_component,
    )

    run_typed_geometry_component(
        provider_id=provider_id,
        representation_id=representation_id,
        geometry_id=geometry_id,
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        params_file=params_file,
    )


@commands.command("typed-distance")
def cli_typed_distance(
    provider_id: Annotated[str, typer.Option(help="Registered provider identity.")],
    representation_id: Annotated[
        str, typer.Option(help="Registered representation identity.")
    ],
    distance_id: Annotated[
        str, typer.Option(help="Registered statistical-distance identity.")
    ],
    output_dir: Annotated[Path, typer.Option(help="Distance artifact directory.")],
    params_file: Annotated[
        Path, typer.Option(help="params.yaml carrying the typed registry.")
    ],
    embeddings_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            readable=True,
            help=(
                "Provider Parquet directory; required unless deriving energy/MMD "
                "from --geometry-component-dir."
            ),
        ),
    ] = None,
    geometry_component_dir: Annotated[
        Path | None,
        typer.Option(
            help="Cached raw geometry component; energy/MMD distances derive from it."
        ),
    ] = None,
) -> None:
    """Materialize one typed statistical-distance matrix."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.shared.typed_artifacts import (
        TypedStageError,
        run_typed_distance,
    )

    try:
        run_typed_distance(
            provider_id=provider_id,
            representation_id=representation_id,
            distance_id=distance_id,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            component_dir=geometry_component_dir,
            params_file=params_file,
        )
    except TypedStageError as error:
        raise typer.BadParameter(str(error)) from error


@commands.command("typed-barycentre")
def cli_typed_barycentre(
    provider_id: Annotated[str, typer.Option(help="Registered provider identity.")],
    representation_id: Annotated[
        str, typer.Option(help="Registered representation identity.")
    ],
    arm_id: Annotated[str, typer.Option(help="Registered barycentre arm identity.")],
    embeddings_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            readable=True,
            help="Provider Parquet directory supplied by the stage graph.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option(help="Barycentre artifact directory.")],
    params_file: Annotated[
        Path, typer.Option(help="params.yaml carrying the typed registry.")
    ],
    geometry_component_dir: Annotated[
        Path | None,
        typer.Option(help="Raw geometry component directory for energy/MMD arms."),
    ] = None,
) -> None:
    """Materialize one typed barycentre artifact."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.shared.typed_artifacts import run_typed_barycentre

    run_typed_barycentre(
        provider_id=provider_id,
        representation_id=representation_id,
        arm_id=arm_id,
        embeddings_dir=embeddings_dir,
        geometry_component_dir=geometry_component_dir,
        output_dir=output_dir,
        params_file=params_file,
    )

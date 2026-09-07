"""CLI adapters for provider-agnostic typed shared artifacts."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context

commands = typer.Typer()


class SampleUniverseTableCliOptions(CliOptions):
    """Inputs for the shared canonical-roster appendix table."""

    universe_csv: Path = Field(
        default=Path("data/universe.csv"),
        description="Universe metadata CSV with symbol, name, and sector.",
    )
    returns_dir: Path = Field(
        default=Path("data/shared/returns"),
        description="Directory containing one adjusted-return parquet per ticker.",
    )
    distance_artifact_dir: Path = Field(
        description="Governed typed W2 distance artifact directory.",
    )
    provider_id: str = Field(description="Expected typed-distance provider identity.")
    representation_id: str = Field(
        description="Expected typed-distance representation identity.",
    )
    distance_id: str = Field(
        default="wasserstein_w2",
        description="Expected typed-distance identity.",
    )
    output_file: Path = Field(
        description="Generated LaTeX longtable fragment.",
    )


sample_universe_table_command = model_command(SampleUniverseTableCliOptions)


@commands.command("render-sample-universe-table", cls=sample_universe_table_command)
def cli_render_sample_universe_table(ctx: typer.Context) -> None:
    """Render a canonical priced W2 roster as a LaTeX appendix longtable."""
    from pipeline.figures.shared_tables import render_sample_universe_table

    options = options_from_context(ctx, SampleUniverseTableCliOptions)
    render_sample_universe_table(
        universe_csv=options.universe_csv,
        distance_artifact_dir=options.distance_artifact_dir,
        returns_dir=options.returns_dir,
        output_file=options.output_file,
        provider_id=options.provider_id,
        representation_id=options.representation_id,
        distance_id=options.distance_id,
    )


@commands.command("typed-geometry-component")
def cli_typed_geometry_component(
    provider_id: str = typer.Option(...),
    representation_id: str = typer.Option(...),
    geometry_id: str = typer.Option(...),
    output_dir: Path = typer.Option(...),
    params_file: Path = typer.Option(Path("params.yaml")),
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
        output_dir=output_dir,
        params_file=params_file,
    )


@commands.command("typed-distance")
def cli_typed_distance(
    provider_id: str = typer.Option(...),
    roster_key: str | None = typer.Option(None),
    representation_id: str | None = typer.Option(None),
    distance_id: str | None = typer.Option(None),
    output_dir: Path | None = typer.Option(None),
    params_file: Path = typer.Option(Path("params.yaml")),
) -> None:
    """Materialize one typed matrix or a provider's distance roster."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.shared.typed_artifacts import (
        run_typed_distance,
        run_typed_distance_roster,
    )

    if roster_key is not None:
        if representation_id is not None or distance_id is not None:
            message = "use either one representation/distance or a roster, not both"
            raise typer.BadParameter(message)
        run_typed_distance_roster(
            provider_id=provider_id,
            roster_key=roster_key,
            output_dir=(
                output_dir
                if output_dir is not None
                else Path("data/shared/typed_distances") / provider_id
            ),
            params_file=params_file,
        )
        return

    if representation_id is None or distance_id is None:
        message = "single-distance mode requires representation-id and distance-id"
        raise typer.BadParameter(message)

    run_typed_distance(
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id=distance_id,
        output_dir=(
            output_dir
            if output_dir is not None
            else Path("data/shared/typed_distances")
            / provider_id
            / representation_id
            / distance_id
        ),
        params_file=params_file,
    )


@commands.command("typed-barycentre")
def cli_typed_barycentre(
    provider_id: str = typer.Option(...),
    representation_id: str = typer.Option(...),
    arm_id: str = typer.Option(...),
    geometry_component_dir: Path | None = typer.Option(
        None,
        help="Explicit raw geometry component directory for energy/MMD arms.",
    ),
    output_dir: Path | None = typer.Option(None),
    params_file: Path = typer.Option(Path("params.yaml")),
) -> None:
    """Materialize one typed energy target-projection artifact."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.shared.typed_artifacts import run_typed_barycentre

    run_typed_barycentre(
        provider_id=provider_id,
        representation_id=representation_id,
        arm_id=arm_id,
        geometry_component_dir=geometry_component_dir,
        output_dir=(
            output_dir
            if output_dir is not None
            else Path("data/shared/barycentres")
            / provider_id
            / representation_id
            / arm_id
        ),
        params_file=params_file,
    )

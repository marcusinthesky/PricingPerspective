"""Paper 1 referee diagnostics command adapters: the agreement and its figures."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-diagnostics")
def cli_paper1_diagnostics(
    full_distance_artifact_dir: Annotated[
        Path, typer.Option(help="Validated full-width W2 artifact directory.")
    ],
    distance_artifact_64_dir: Annotated[
        Path, typer.Option(help="Validated 64-dimensional W2 artifact directory.")
    ],
    distance_artifact_256_dir: Annotated[
        Path, typer.Option(help="Validated 256-dimensional W2 artifact directory.")
    ],
    provider_id: Annotated[
        str, typer.Option(help="Expected typed distance provider identity.")
    ],
    representation_id: Annotated[
        str, typer.Option(help="Expected full-width representation identity.")
    ],
    representation_64_id: Annotated[
        str, typer.Option(help="Expected 64-dimensional representation identity.")
    ],
    representation_256_id: Annotated[
        str, typer.Option(help="Expected 256-dimensional representation identity.")
    ],
    distance_id: Annotated[
        str, typer.Option(help="Expected primary typed statistical-distance identity.")
    ],
    output_file: Annotated[
        Path,
        typer.Option(help="Results JSON of truncated-vs-full W2 agreement statistics."),
    ],
    output_preservation_file: Annotated[
        Path,
        typer.Option(help="Aligned per-dyad truncated and full distances parquet."),
    ],
) -> None:
    """Compute truncated-vs-full W2 distance agreement per Matryoshka width."""
    own_float64()
    from pipeline.stages.papers.paper1.diagnostics import (
        Paper1DiagnosticsConfig,
        run_paper1_diagnostics,
    )

    run_paper1_diagnostics(
        Paper1DiagnosticsConfig(
            full_distance_artifact_dir=full_distance_artifact_dir,
            distance_artifact_64_dir=distance_artifact_64_dir,
            distance_artifact_256_dir=distance_artifact_256_dir,
            provider_id=provider_id,
            representation_id=representation_id,
            representation_64_id=representation_64_id,
            representation_256_id=representation_256_id,
            distance_id=distance_id,
            output_file=output_file,
            output_preservation_file=output_preservation_file,
        )
    )
    typer.echo(f"Saved: {output_file}")
    typer.echo(f"Saved: {output_preservation_file}")


@commands.command("render-paper1-diagnostics")
def cli_render_paper1_diagnostics(
    results_file: Annotated[
        Path, typer.Option(help="paper1-diagnostics results JSON.")
    ],
    preservation_file: Annotated[
        Path, typer.Option(help="paper1-diagnostics per-dyad distances parquet.")
    ],
    dyadic_summary_file: Annotated[
        Path, typer.Option(help="paper1-dyadic-confound results JSON.")
    ],
    w2_primary_fit_file: Annotated[
        Path, typer.Option(help="Primary W2 fitted/residual/leverage parquet.")
    ],
    lofo_file: Annotated[
        Path, typer.Option(help="Leave-one-firm-out influence parquet.")
    ],
    output_dir: Annotated[Path, typer.Option(help="Output directory for the PGFs.")],
) -> None:
    """Draw the Matryoshka preservation and primary W2 post-fit figures."""
    own_float64()
    from pipeline.figures.paper1.diagnostics import (
        Paper1DiagnosticsRenderPaths,
        render_paper1_diagnostics,
    )

    render_paper1_diagnostics(
        Paper1DiagnosticsRenderPaths(
            results_file=results_file,
            preservation_file=preservation_file,
            dyadic_results_file=dyadic_summary_file,
            w2_primary_fit_file=w2_primary_fit_file,
            lofo_file=lofo_file,
            output_dir=output_dir,
        )
    )

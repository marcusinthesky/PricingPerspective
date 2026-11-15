"""CLI validation of published embedding inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("validate-coverage")
def cli_validate_coverage(
    model_key: Annotated[
        str, typer.Option(help="Model key from params.yaml embedding.models.")
    ],
    embeddings_dir: Annotated[
        Path, typer.Option(help="Directory containing per-ticker parquets.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for summary.yaml and coverage.csv.")
    ],
    params_file: Annotated[Path, typer.Option(help="Path to params.yaml.")],
) -> None:
    """Validate embedding coverage and run energy-distance sanity checks."""
    from pipeline.stages.substrate.embeddings import validate_coverage

    validate_coverage(model_key, embeddings_dir, output_dir, params_file)


@commands.command("validate-public-inputs")
def cli_validate_public_inputs(
    manifest_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    repo_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(),
) -> None:
    """Verify the published embedding schemas, cohorts, and checksums."""
    from pipeline.stages.substrate.embeddings import validate_public_inputs

    validate_public_inputs(manifest_file, repo_root)

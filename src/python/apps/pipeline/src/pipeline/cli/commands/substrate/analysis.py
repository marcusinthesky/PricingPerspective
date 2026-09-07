"""Substrate Analysis CLI command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("compute-energy-tests")
def cli_compute_energy_tests(
    embeddings_dir: Path = typer.Option(
        ..., help="Directory containing embedding parquet files."
    ),
    output_file: Path = typer.Option(
        ..., help="Output parquet file to save energy test results."
    ),
    min_samples: int = typer.Option(
        10, help="Minimum number of samples required per ticker."
    ),
    alpha: float = typer.Option(
        1.0, help="Distance exponent (1.0 = Euclidean distance)."
    ),
    matryoshka_dim: int | None = typer.Option(
        None,
        help="Matryoshka truncation dim; leading k coords kept then renormalized. "
        "Omit / null = full embedding width (canonical).",
    ),
) -> None:
    """Compute energy test for all combinations of ticker embeddings (JAX)."""
    from pipeline.stages.substrate.energy import compute_energy_tests

    compute_energy_tests(
        embeddings_dir, output_file, min_samples, alpha, matryoshka_dim
    )


class AblationCliOptions(CliOptions):
    """Inputs for the embedding-model and Matryoshka ablation grid."""

    embeddings_root: Path = Field(
        default=Path("data/shared/embeddings"),
        description="Root dir containing per-model embedding subdirectories.",
    )
    covariance_matrix: Path = Field(
        default=Path("data/shared/covariance_matrix.parquet"),
        description=(
            "Covariance matrix parquet (for Mantel r_M). If absent, Mantel "
            "columns are null and energy tests are still computed."
        ),
    )
    output_dir: Path = Field(
        default=Path("data/shared/ablations"),
        description="Output dir for per-cell energy tests and consolidated summary.",
    )
    typed_distance_root: Path = Field(
        default=Path("data/shared/typed_distances"),
        description="Root containing typed provider/representation/distance artifacts.",
    )
    params_file: Path = Field(
        default=Path("params.yaml"),
        description="Params file with the ablation.grid block.",
    )
    mantel_permutations: int = Field(
        default=10000,
        description="Permutations for the Mantel p-value.",
    )
    bootstrap_iters: int = Field(
        default=1000,
        description="Bootstrap resamples for the Mantel CI.",
    )
    seed: int = Field(default=42, description="Random seed for the bootstrap CI.")


run_ablation_command = model_command(AblationCliOptions)


@commands.command("run-ablation", cls=run_ablation_command)
def cli_run_ablation(ctx: typer.Context) -> None:
    """Run the (embedding model × Matryoshka dim) ablation grid."""
    own_float64()
    from pipeline.stages.substrate.ablation import (
        AblationOptions,
        ResamplingOptions,
        run_ablation,
    )

    options = options_from_context(ctx, AblationCliOptions)
    run_ablation(
        embeddings_root=options.embeddings_root,
        covariance_matrix=options.covariance_matrix,
        output_dir=options.output_dir,
        params_file=options.params_file,
        options=AblationOptions(
            typed_distance_root=options.typed_distance_root,
            resampling=ResamplingOptions(
                permutations=options.mantel_permutations,
                bootstrap_iters=options.bootstrap_iters,
                seed=options.seed,
            ),
        ),
    )


class MantelCliOptions(CliOptions):
    """Inputs for covariance-distance Mantel inference."""

    distance_artifact_dir: Path = Field(description="Typed distance artifact directory")
    covariance_matrix: Path = Field(description="Covariance matrix parquet file")
    output_file: Path = Field(description="Output CSV file for Mantel test results")
    method: Literal["pearson", "spearman"] = Field(
        default="pearson",
        description="Correlation method: pearson or spearman",
    )
    permutations: int = Field(
        default=10000,
        description="Number of permutations for p-value estimation",
    )
    seed: int = Field(default=42, description="Deterministic permutation seed")
    null_output_file: Path | None = Field(
        default=None,
        description="Output parquet for the complete permutation-null correlations",
    )


mantel_command = model_command(MantelCliOptions)


@commands.command("compute-mantel-tests", cls=mantel_command)
def cli_compute_mantel_tests(ctx: typer.Context) -> None:
    """Compute Mantel test between a typed distance and covariance matrix."""
    own_float64()
    from pipeline.stages.substrate.mantel import MantelOptions, run_mantel_tests

    options = options_from_context(ctx, MantelCliOptions)
    run_mantel_tests(
        options.distance_artifact_dir,
        options.covariance_matrix,
        options.output_file,
        MantelOptions(
            method=options.method,
            permutations=options.permutations,
            seed=options.seed,
            null_output_file=options.null_output_file,
        ),
    )


@commands.command("compute-dimensionality")
def cli_compute_dimensionality(
    distance_artifact_dir: Path = typer.Option(
        ..., help="Typed distance artifact directory"
    ),
    covariance_matrix: Path = typer.Option(..., help="Covariance matrix parquet file"),
    universe_csv: Path = typer.Option(
        ..., help="Universe CSV file with ticker-sector mapping"
    ),
    output_file: Path = typer.Option(..., help="Output CSV file for analysis results"),
) -> None:
    """Compute governed metric-MDS projection-fidelity diagnostics."""
    own_float64()
    from pipeline.stages.substrate.dimensionality import run_dimensionality_analysis

    run_dimensionality_analysis(
        distance_artifact_dir, covariance_matrix, universe_csv, output_file
    )

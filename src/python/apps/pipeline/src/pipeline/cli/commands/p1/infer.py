"""P1 Infer CLI command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("compute-dyadic-oos")
def cli_compute_dyadic_oos(
    *,
    distance_artifact_dir: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        help="Frozen Qwen3-Embedding 8B W2 artifact directory (text side).",
    ),
    covariance_matrix: Path = typer.Option(
        Path("data/shared/oos/covariance_matrix.parquet"),
        help="Out-of-sample correlation matrix parquet (return side).",
    ),
    output_file: Path = typer.Option(
        Path("data/papers/paper1/dyadic_oos/summary.yaml"),
        help="Output YAML summary path.",
    ),
) -> None:
    """Run the out-of-sample primary W2 dyadic association.

    The text-side matrix is held fixed at its in-sample estimate; only the
    return-distance outcome is recomputed on the out-of-sample window.
    """
    own_float64()
    from pipeline.stages.papers.paper1.dyadic_oos import DyadicOosConfig, run_dyadic_oos

    result = run_dyadic_oos(
        DyadicOosConfig(
            distance_artifact_dir=distance_artifact_dir,
            covariance_matrix=covariance_matrix,
            output_file=output_file,
        )
    )
    w2_arm = cast("dict[str, float]", result["w2_arm"])
    typer.echo(
        f"wasserstein_w2 [primary]: coef={w2_arm['coef']:.6f}  "
        f"ci=[{w2_arm['ci_low']:.6f}, {w2_arm['ci_high']:.6f}]"
    )
    typer.echo(f"Saved: {output_file}")

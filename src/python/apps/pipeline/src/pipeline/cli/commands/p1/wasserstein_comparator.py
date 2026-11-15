"""Paper 1 cross-encoder distributional comparator command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-wasserstein-comparator")
def cli_paper1_wasserstein_comparator(
    embeddings_root: Annotated[
        Path,
        typer.Option(
            help="Root containing the four locked per-model embedding directories."
        ),
    ],
    typed_distance_root: Annotated[
        Path,
        typer.Option(
            help="Root containing validated typed statistical-distance artifacts."
        ),
    ],
    covariance_matrix: Annotated[
        Path, typer.Option(help="Return covariance/correlation matrix parquet.")
    ],
    output_pairs: Annotated[
        Path, typer.Option(help="Matched dyad-level metric artifact.")
    ],
) -> None:
    """Run Paper 1's cross-encoder distributional comparison diagnostics."""
    own_float64()
    from pipeline.stages.papers.paper1.wasserstein_comparator import (
        WassersteinComparatorConfig,
        run_wasserstein_comparator,
    )

    result = run_wasserstein_comparator(
        WassersteinComparatorConfig(
            embeddings_root=embeddings_root,
            typed_distance_root=typed_distance_root,
            covariance_matrix=covariance_matrix,
            output_pairs=output_pairs,
        )
    )
    typer.echo(
        f"common m={result['common_m']}  priced firms={result['priced_universe']}  "
        f"dyads={result['n_priced_dyads']}"
    )
    typer.echo(f"Saved: {output_pairs}")

"""Compute and publish the original-W2 sector diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper1-w2-permanova")
def cli_paper1_w2_permanova(
    distance_artifact_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True)
    ],
    covariance_matrix: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    universe_csv: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option(file_okay=False)],
    provider_id: Annotated[str, typer.Option()],
    representation_id: Annotated[str, typer.Option()],
    sample_size: Annotated[int, typer.Option(min=1)] = 128,
    n_permutations: Annotated[int, typer.Option(min=1)] = 9_999,
    random_state: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    """Decompose squared W2 distances and test sector association and spread."""
    from pipeline.stages.papers.paper1.w2_permanova import (
        Paper1W2PermanovaPaths,
        run_w2_permanova,
    )

    run_w2_permanova(
        Paper1W2PermanovaPaths(
            distance_artifact_dir=distance_artifact_dir,
            covariance_matrix=covariance_matrix,
            universe_csv=universe_csv,
            output_dir=output_dir,
        ),
        provider_id=provider_id,
        representation_id=representation_id,
        sample_size=sample_size,
        n_permutations=n_permutations,
        random_state=random_state,
    )


@commands.command("render-paper1-w2-permanova")
def cli_render_paper1_w2_permanova(
    results_file: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    """Publish the saved sector decomposition and separate spread diagnostic."""
    from pipeline.figures.paper1.w2_permanova import (
        Paper1W2PermanovaRenderPaths,
        render_paper1_w2_permanova,
    )

    render_paper1_w2_permanova(
        Paper1W2PermanovaRenderPaths(results_file=results_file, output_dir=output_dir)
    )

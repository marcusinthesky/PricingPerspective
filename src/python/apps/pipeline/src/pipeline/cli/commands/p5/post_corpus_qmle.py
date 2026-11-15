"""Paper 5 post-corpus QMLE (also run per representation-ablation cell)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper5-post-corpus-qmle")
def cli_paper5_post_corpus_qmle(
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    oos_returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2023--2026 return panel."),
    ],
    shared_barycentre_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Frozen leave-one-out W2 artifact."
        ),
    ],
    distance_artifact: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Typed W2 distance artifact folder."
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
    n_bootstrap: Annotated[int, typer.Option(min=1, help="Bootstrap draws.")],
    block_length: Annotated[
        int, typer.Option(min=1, help="Expected stationary-bootstrap block length.")
    ],
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    barycentre_arm_id: Annotated[str, typer.Option(help="Barycentre arm identity.")],
    geometry_id: Annotated[str, typer.Option(help="Ground geometry identity.")],
    *,
    co_mentions_adjacency: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Persistent co-mention adjacency; omitted by ablation cells.",
        ),
    ] = None,
    pooled_only: Annotated[
        bool, typer.Option(help="Estimate the pooled period only (ablation cells).")
    ] = False,
    focal_matrices_only: Annotated[
        bool, typer.Option(help="Fit only W-flat and W-h (ablation cells).")
    ] = False,
    output_bootstrap: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Optional bootstrap-draw Parquet to write."),
    ] = None,
) -> None:
    """Estimate pooled and annual QMLE on returns after the text corpus."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.post_corpus import (
        Paper5PostCorpusConfig,
        Paper5PostCorpusPaths,
        run_paper5_post_corpus_qmle,
    )

    run_paper5_post_corpus_qmle(
        Paper5PostCorpusPaths(
            returns_dir=returns_dir,
            oos_returns_dir=oos_returns_dir,
            shared_barycentre_dir=shared_barycentre_dir,
            distance_artifact_dir=distance_artifact,
            co_mentions_adjacency=co_mentions_adjacency,
            output_dir=output_dir,
        ),
        Paper5PostCorpusConfig(
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
            provider_id=provider_id,
            representation_id=representation_id,
            barycentre_arm_id=barycentre_arm_id,
            geometry_id=geometry_id,
            pooled_only=pooled_only,
            focal_matrices_only=focal_matrices_only,
            output_bootstrap=output_bootstrap,
        ),
    )

"""Paper 5 representation ablation: merge the cells and render the table.

Each cell runs ``paper5-post-corpus-qmle --pooled-only --focal-matrices-only``
into ``representation_ablation/cells/<key>/``; this module merges and renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper5-representation-ablation")
def cli_paper5_representation_ablation(
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads the ablation cell roster and contrast specs.",
        ),
    ],
    cells_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Folder holding one subfolder per cell."
        ),
    ],
    canonical_results: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Canonical post-corpus QMLE results."
        ),
    ],
    output_summary: Annotated[
        Path, typer.Option(dir_okay=False, help="Merged results.json to write.")
    ],
    bootstrap_dir: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Folder of per-cell bootstrap draws; defaults to --cells-dir.",
        ),
    ] = None,
) -> None:
    """Merge the pooled Paper 5 representation cells and paired contrasts."""
    from pipeline.stages.papers.paper5.representation_ablation import (
        Paper5RepresentationMergePaths,
        merge_paper5_representation_ablation,
    )

    merge_paper5_representation_ablation(
        Paper5RepresentationMergePaths(
            params_file=params_file,
            cells_dir=cells_dir,
            canonical_results=canonical_results,
            output_summary=output_summary,
            bootstrap_dir=bootstrap_dir,
        )
    )


@commands.command("render-paper5-representation-ablation")
def cli_render_paper5_representation_ablation(
    summary_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Merged ablation results.json."),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated table fragment.")
    ],
) -> None:
    """Render Paper 5's pooled representation/operator horse race."""
    from pipeline.figures.paper5.tables import (
        render_paper5_representation_ablation_table,
    )

    render_paper5_representation_ablation_table(summary_path, output_path)

"""Paper 3 representation-ablation cell, merge, and render command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper3-representation-ablation-cells")
def cli_paper3_representation_ablation_cells(
    cell_id: Annotated[str, typer.Option(help="Registered ablation cell key.")],
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads the cell roster, "
            "paper3.certificate_feasibility.news_reference and "
            "paper3.certificate_validation.{return_block_replications,seed}.",
        ),
    ],
    returns_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Aligned returns.")
    ],
    distance_artifact: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="This cell's typed W2 distance artifact."
        ),
    ],
    canonical_distance_artifact: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Canonical typed W2 artifact that fixes the bootstrap block length.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Cell result folder to write.")
    ],
) -> None:
    """Evaluate one Paper 3 representation under the fixed variance-rank laws."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper3.representation_ablation import (
        Paper3RepresentationAblationPaths,
        run_paper3_representation_ablation,
    )

    run_paper3_representation_ablation(
        Paper3RepresentationAblationPaths(
            params_file=params_file,
            returns_dir=returns_dir,
            distance_artifact_dir=distance_artifact,
            output_dir=output_dir,
            canonical_distance_artifact_dir=canonical_distance_artifact,
        ),
        cell_id,
    )


@commands.command("paper3-representation-ablation")
def cli_paper3_representation_ablation(
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads the ablation rosters and contrast specs.",
        ),
    ],
    cells_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Folder holding <cell>/results.json."
        ),
    ],
    canonical_certificate_summary: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Certificate results.json to reproduce."
        ),
    ],
    output_summary: Annotated[
        Path, typer.Option(dir_okay=False, help="Merged results.json to write.")
    ],
    contrast_draws_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            help="Folder holding <cell>/return_bootstrap.parquet; "
            "defaults to --cells-dir.",
        ),
    ] = None,
) -> None:
    """Merge the complete Paper 3 representation roster."""
    from pipeline.stages.papers.paper3.representation_ablation import (
        Paper3RepresentationMergePaths,
        merge_paper3_representation_ablation,
    )

    merge_paper3_representation_ablation(
        Paper3RepresentationMergePaths(
            params_file=params_file,
            cells_dir=cells_dir,
            canonical_certificate_summary=canonical_certificate_summary,
            output_summary=output_summary,
            contrast_draws_dir=contrast_draws_dir,
        )
    )


@commands.command("render-paper3-representation-ablation")
def cli_render_paper3_representation_ablation(
    summary_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Merged ablation results.json."),
    ],
    output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Appendix table (.tex) to write.")
    ],
    main_output_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Main-text table (.tex) to write.")
    ],
) -> None:
    """Render Paper 3's main-text summary and appendix robustness tables."""
    from pipeline.figures.paper3.tables import (
        render_paper3_representation_ablation_table,
    )

    render_paper3_representation_ablation_table(
        summary_path, output_path, main_output_path
    )

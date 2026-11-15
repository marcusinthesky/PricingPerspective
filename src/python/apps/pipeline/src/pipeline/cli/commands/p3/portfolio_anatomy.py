"""Paper 3 portfolio-anatomy compute and render command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper3-portfolio-anatomy")
def cli_paper3_portfolio_anatomy(
    weights_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Certificate news weights."),
    ],
    distance_artifact: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Canonical typed W2 distance artifact."
        ),
    ],
    pit_w2_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="PIT squared-W2 .npz; its sibling provenance.manifest.json is read.",
        ),
    ],
    universe_csv: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Ticker universe CSV.")
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
) -> None:
    """Decompose Paper 3's canonical allocation and expanding vintages."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper3.portfolio_anatomy import (
        Paper3PortfolioAnatomyPaths,
        run_paper3_portfolio_anatomy,
    )

    run_paper3_portfolio_anatomy(
        Paper3PortfolioAnatomyPaths(
            weights_path=weights_path,
            distance_artifact_dir=distance_artifact,
            pit_w2_path=pit_w2_path,
            pit_manifest_path=pit_w2_path.with_name("provenance.manifest.json"),
            universe_csv=universe_csv,
            output_dir=output_dir,
        )
    )


@commands.command("render-paper3-portfolio-anatomy")
def cli_render_paper3_portfolio_anatomy(
    anatomy_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Anatomy result folder."),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Manuscript images folder.")
    ],
    table_path: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated holdings table (.tex).")
    ],
) -> None:
    """Render Paper 3's allocation anatomy and expanding-vintage exhibits."""
    from pipeline.figures.paper3.portfolio_anatomy import (
        render_paper3_portfolio_anatomy,
    )

    render_paper3_portfolio_anatomy(anatomy_dir, output_dir, table_path)

"""Paper 1 principal coordinates, metric MDS and saved-result publication."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-pcoa")
def cli_paper1_pcoa(
    distance_root: Annotated[
        Path, typer.Option(help="Root of the validated typed distance artifacts.")
    ],
    covariance_matrix: Annotated[
        Path, typer.Option(help="In-sample return covariance parquet.")
    ],
    universe_csv: Annotated[
        Path, typer.Option(help="Universe metadata (Symbol, Sector, Industry).")
    ],
    params_file: Annotated[
        Path, typer.Option(help="Governed rosters and paper1.mds solver settings.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Saved projection analysis directory.")
    ],
) -> None:
    """Compare W2 spectra with return PCA and fit the canonical metric-MDS map."""
    own_float64()
    from pipeline.stages.papers.paper1.pcoa import (
        Paper1PcoaOptions,
        run_paper1_pcoa,
    )

    run_paper1_pcoa(
        Paper1PcoaOptions(
            distance_root=distance_root,
            covariance_matrix=covariance_matrix,
            universe_csv=universe_csv,
            params_file=params_file,
            output_dir=output_dir,
        )
    )


@commands.command("render-paper1-pcoa")
def cli_render_paper1_pcoa(
    analysis_dir: Annotated[
        Path, typer.Option(help="Saved projection results, coordinates and spectra.")
    ],
    output_dir: Annotated[Path, typer.Option(help="Output directory for the PGFs.")],
    spectrum_table: Annotated[
        Path, typer.Option(help="Generated LaTeX spectrum-summary table.")
    ],
) -> None:
    """Render the metric-MDS/PCoA maps, fidelity and comparison-ladder spectra."""
    from pipeline.figures.paper1.pcoa import Paper1PcoaPaths, render_paper1_pcoa

    render_paper1_pcoa(
        Paper1PcoaPaths(
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            spectrum_table=spectrum_table,
        )
    )

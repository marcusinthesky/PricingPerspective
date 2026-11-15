"""Paper 3 certificate-feasibility compute and render command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper3-certificate")
def cli_paper3_certificate(
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads paper3.certificate_feasibility.",
        ),
    ],
    returns_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Aligned returns.")
    ],
    distance_artifact: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Canonical typed W2 distance artifact."
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
) -> None:
    """Compute Paper 3's full-sample certificate sensitivity surface."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper3.certificate import (
        Paper3CertificatePaths,
        run_paper3_certificate,
    )

    run_paper3_certificate(
        Paper3CertificatePaths(
            params_file=params_file,
            returns_dir=returns_dir,
            distance_artifact_dir=distance_artifact,
            output_dir=output_dir,
        )
    )


@commands.command("render-paper3-certificate")
def cli_render_paper3_certificate(
    certificate_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Certificate result folder to read."
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Manuscript images folder.")
    ],
) -> None:
    """Render the news-variance benchmark figure from certificate results."""
    from pipeline.figures.paper3.exhibits import render_paper3_certificate

    render_paper3_certificate(certificate_dir, output_dir)

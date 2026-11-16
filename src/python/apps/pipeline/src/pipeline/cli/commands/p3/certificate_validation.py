"""Paper 3 chronological certificate-validation command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper3-certificate-validation")
def cli_paper3_certificate_validation(
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads paper3.certificate_validation and typed_analysis.",
        ),
    ],
    returns_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Calibration returns.")
    ],
    holdout_returns_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Holdout returns.")
    ],
    pit_w2_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="PIT squared-W2 .npz; its sibling provenance.manifest.json is read.",
        ),
    ],
    sample_descriptives_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Paper 1 sample descriptives results.json (sector labels).",
        ),
    ],
    embeddings_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Provider Parquet directory supplied by the stage graph.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
) -> None:
    """Calibrate then evaluate Paper 3's frozen chronological certificate."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper3.certificate_validation import (
        Paper3CertificateValidationPaths,
        run_paper3_certificate_validation,
    )

    run_paper3_certificate_validation(
        Paper3CertificateValidationPaths(
            params_file=params_file,
            returns_dir=returns_dir,
            holdout_returns_dir=holdout_returns_dir,
            pit_w2_path=pit_w2_path,
            sample_descriptives_path=sample_descriptives_path,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
        )
    )

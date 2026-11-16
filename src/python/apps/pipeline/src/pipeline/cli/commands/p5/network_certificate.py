"""Paper 5 network-theory certificate command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper5-network-certificate")
def cli_paper5_network_certificate(
    shared_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen W-flat artifact."),
    ],
    qmle_results: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Post-corpus QMLE results.json."
        ),
    ],
    factor_conditioned_results: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Spatial-value comparison results.json."
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Folder receiving results.json.")
    ],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    barycentre_arm_id: Annotated[str, typer.Option(help="Barycentre arm identity.")],
    geometry_id: Annotated[str, typer.Option(help="Ground geometry identity.")],
    tolerance: Annotated[float, typer.Option(help="Numerical certificate tolerance.")],
) -> None:
    """Certify contraction and finite-rho error for the frozen W-flat operator."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.network_certificate import (
        Paper5NetworkCertificateConfig,
        Paper5NetworkCertificatePaths,
        run_paper5_network_certificate,
    )

    run_paper5_network_certificate(
        Paper5NetworkCertificatePaths(
            shared_barycentre_dir=shared_barycentre_dir,
            qmle_results=qmle_results,
            output_dir=output_dir,
            factor_conditioned_results=factor_conditioned_results,
        ),
        Paper5NetworkCertificateConfig(
            provider_id=provider_id,
            representation_id=representation_id,
            barycentre_arm_id=barycentre_arm_id,
            geometry_id=geometry_id,
            tolerance=tolerance,
        ),
    )

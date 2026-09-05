"""Paper 5 network-theory certificate command adapter."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper5NetworkCertificateOptions(CliOptions):
    """Inputs for the frozen-operator Perron and finite-rho witnesses."""

    shared_barycentre_dir: Path = Field(
        default=Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        )
    )
    qmle_results: Path = Field(
        default=Path("data/papers/paper5/post_corpus_qmle/results.json")
    )
    output_json: Path = Field(
        default=Path("data/papers/paper5/network_certificate.json")
    )
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    barycentre_arm_id: str = Field(default="wasserstein_w2_loo")
    geometry_id: str = Field(default="wasserstein_w2")
    tolerance: float = Field(default=1e-12)


paper5_network_certificate_command = model_command(Paper5NetworkCertificateOptions)


@commands.command(
    "paper5-network-certificate",
    cls=paper5_network_certificate_command,
)
def cli_paper5_network_certificate(ctx: typer.Context) -> None:
    """Certify contraction and finite-rho error for the frozen W-flat operator."""
    own_float64()
    from pipeline.stages.papers.paper5.network_certificate import (
        Paper5NetworkCertificateConfig,
        Paper5NetworkCertificatePaths,
        run_paper5_network_certificate,
    )

    options = options_from_context(ctx, Paper5NetworkCertificateOptions)
    run_paper5_network_certificate(
        Paper5NetworkCertificatePaths(
            shared_barycentre_dir=options.shared_barycentre_dir,
            qmle_results=options.qmle_results,
            output_json=options.output_json,
        ),
        Paper5NetworkCertificateConfig(
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            barycentre_arm_id=options.barycentre_arm_id,
            geometry_id=options.geometry_id,
            tolerance=options.tolerance,
        ),
    )

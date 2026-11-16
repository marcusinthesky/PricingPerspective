"""Additive iid-history sample-size audit and saved-result publication."""

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper6-inference-audit")
def audit(
    design: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/papers/paper6/inference_audit_design.json"
    ),
    params_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "params.yaml"
    ),
    baseline: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/papers/paper6/directional_validation/inference"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "data/papers/paper6/inference_audit"
    ),
    threads: Annotated[int, typer.Option(min=1)] = 1,
    pilot_only: Annotated[  # noqa: FBT002 - Typer option signature
        bool, typer.Option(help="Only the fixed serial/spawn runtime pilot.")
    ] = False,
) -> None:
    """Compare fixed N100 calibration with the immutable saved N16 fixture."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.io.params import load_params
    from pipeline.stages.papers.paper6.inference_audit_extension import (
        InferenceAuditConfig,
        run_inference_audit,
    )

    config = InferenceAuditConfig.model_validate(
        load_params(params_file)["paper6"]["inference_audit"]
    )
    result = run_inference_audit(
        output,
        config,
        design=design,
        baseline=baseline,
        threads=threads,
        pilot_only=pilot_only,
    )
    typer.echo(f"Paper 6 inference audit: {result['status']}")


@commands.command("render-paper6-inference-audit")
def render(
    results: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/papers/paper6/inference_audit/results.json"
    ),
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "src/latex/projects/06_information_motion"
    ),
) -> None:
    """Render the saved iid calibration comparison without re-estimation."""
    from pipeline.figures.paper6.inference_audit import render as render_results

    render_results(results, project)

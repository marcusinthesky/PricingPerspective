"""Known-target directional validation with immutable historical evidence."""

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper6-directional-validation")
def validate(
    legacy_campaign: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/papers/paper6/information_recovery/campaign"
    ),
    design: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/papers/paper6/directional_design.json"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "data/papers/paper6/directional_validation"
    ),
    params_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "params.yaml"
    ),
    workers: Annotated[
        str, typer.Option(help="Bounded process count, or auto (at most three). ")
    ] = "auto",
    threads: Annotated[int, typer.Option(min=1)] = 1,
    pilot_only: Annotated[  # noqa: FBT002 - Typer option signature
        bool,
        typer.Option(
            help="Only integrity/runtime pilot and cheap historical diagnostics."
        ),
    ] = False,
) -> None:
    """Run the fixed conditional controls, never alter the historical campaign."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline._kernels.paper6.directional_validation import (
        DirectionalValidationConfig,
    )
    from pipeline.io.params import load_params
    from pipeline.stages.papers.paper6.directional_validation import run_validation

    config = DirectionalValidationConfig.model_validate(
        load_params(params_file)["paper6"]["directional_validation"]
    )
    result = run_validation(
        output,
        legacy_campaign=legacy_campaign,
        design=design,
        config=config,
        workers=workers,
        threads=threads,
        pilot_only=pilot_only,
    )
    typer.echo(
        f"Paper 6 directional validation: {result['status']}; "
        f"{len(result['failures'])} explicit failures"
    )
    if result["status"] == "execution_failed":
        raise typer.Exit(code=1)

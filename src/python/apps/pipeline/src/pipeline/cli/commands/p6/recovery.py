"""Production entrypoints for the fixed Paper 6 information-recovery study."""

from pathlib import Path
from typing import Annotated

import typer

from pipeline.io.paths import PAPERS

commands = typer.Typer()


@commands.command("paper6-information-recovery")
def recover(
    output: Annotated[Path, typer.Option(file_okay=False, writable=True)],
    workers: Annotated[
        str, typer.Option(help="Bounded process count, or auto (at most three).")
    ] = "auto",
    threads: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Run the legacy design in an explicitly selected fresh namespace."""
    historical = (PAPERS / "paper6/information_recovery").resolve()
    if output.resolve().is_relative_to(historical):
        msg = "the historical recovery campaign is immutable"
        raise typer.BadParameter(msg)
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper6.recovery.campaign import run_campaign as run

    result = run(output, workers=workers, threads=threads)
    typer.echo(
        f"Paper 6 recovery: {result['status']}; "
        f"{len(result['failures'])} explicit failures"
    )
    if result["status"] != "complete":
        raise typer.Exit(code=1)


@commands.command("paper6-information-recovery-audit")
def audit(
    output: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
) -> None:
    """Check saved evidence integrity independently of today's source layout."""
    from pipeline.stages.papers.paper6.recovery.campaign import audit_campaign as run

    result = run(output)
    typer.echo(f"Paper 6 recovery integrity: {result['integrity']}")


@commands.command("render-paper6-information-recovery")
def exhibits(
    summary: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        PAPERS / "paper6/information_recovery/campaign/summary.json"
    ),
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "src/latex/projects/06_information_motion"
    ),
    directional_summary: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = PAPERS / "paper6/directional_validation/results.json",
) -> None:
    """Project saved numerical evidence into the Paper 6 figure and tables."""
    from pipeline.figures.paper6.information_recovery.render import publish

    publish(summary, project, directional_summary)

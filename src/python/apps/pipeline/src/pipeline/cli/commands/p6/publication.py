"""Independent primary corporate and secondary observation publication."""

from pathlib import Path
from typing import Annotated

import typer

from pipeline.io.paths import PAPERS

commands = typer.Typer()
_PROJECT = Path("src/latex/projects/06_information_motion")


@commands.command("render-paper6-corporate-analysis")
def corporate(
    results: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        PAPERS / "paper6/corporate_analysis/results.json"
    ),
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = _PROJECT,
) -> None:
    """Render the full-sample projection from its independently saved results."""
    from pipeline.figures.paper6.corporate_analysis import publish

    publish(results, project)


@commands.command("render-paper6-information-audit")
def observation(
    evidence: Annotated[Path, typer.Option(exists=True, file_okay=False)] = (
        PAPERS / "paper6/review-artifacts"
    ),
    project: Annotated[Path, typer.Option(exists=True, file_okay=False)] = _PROJECT,
) -> None:
    """Render the historical T82/T83 observation and locality audit."""
    from pipeline.figures.paper6.information_audit import publish

    publish(evidence, project)

"""Compose the pipeline command adapters into one Typer application."""

from __future__ import annotations

import typer
from dotell import begin, setup_logging

from pipeline.cli.commands import shared
from pipeline.cli.commands.audit import backtest, semflow
from pipeline.cli.commands.cross_paper import estimate as cross_paper_estimate
from pipeline.cli.commands.cross_paper import robustness
from pipeline.cli.commands.p1 import infer as p1_infer
from pipeline.cli.commands.p1 import prepare as p1_prepare
from pipeline.cli.commands.p1.estimate import geometry as p1_geometry
from pipeline.cli.commands.p1.estimate import identification as p1_identification
from pipeline.cli.commands.p1.exhibit import figures as p1_figures
from pipeline.cli.commands.p1.exhibit import tables as p1_tables
from pipeline.cli.commands.p3 import estimate as p3_estimate
from pipeline.cli.commands.p3 import exhibit as p3_exhibit
from pipeline.cli.commands.p5 import calibrate as p5_calibrate
from pipeline.cli.commands.p5 import energy_robustness as p5_energy_robustness
from pipeline.cli.commands.p5 import estimate as p5_estimate
from pipeline.cli.commands.p5 import exhibit as p5_exhibit
from pipeline.cli.commands.p5 import network as p5_network
from pipeline.cli.commands.p5 import post_corpus as p5_post_corpus
from pipeline.cli.commands.p5 import two_field as p5_two_field
from pipeline.cli.commands.substrate import analysis, corpus, embeddings, market

app = typer.Typer(
    help="DVC data-pipeline stages for the pricing-perspective project.",
    no_args_is_help=True,
)


@app.callback()
def _bootstrap(ctx: typer.Context) -> None:
    """Configure logging and per-stage resource telemetry for every command."""
    setup_logging("pipeline")
    if ctx.invoked_subcommand is not None:
        ctx.call_on_close(begin(ctx.invoked_subcommand, app="pipeline"))


app.add_typer(corpus.commands)
app.add_typer(embeddings.commands)
app.add_typer(market.commands)
app.add_typer(analysis.commands)

app.add_typer(p1_prepare.commands)
app.add_typer(p1_geometry.commands)
app.add_typer(p1_identification.commands)
app.add_typer(p1_infer.commands)
app.add_typer(p1_figures.commands)
app.add_typer(p1_tables.commands)

app.add_typer(p3_estimate.commands)
app.add_typer(p3_exhibit.commands)
app.add_typer(cross_paper_estimate.commands)
app.add_typer(p5_energy_robustness.commands)
app.add_typer(p5_estimate.commands)
app.add_typer(p5_calibrate.commands)
app.add_typer(p5_exhibit.commands)
app.add_typer(p5_network.commands)
app.add_typer(p5_post_corpus.commands)
app.add_typer(p5_two_field.commands)

app.add_typer(robustness.commands)
app.add_typer(shared.commands)
app.add_typer(backtest.commands)
app.add_typer(semflow.commands, name="semflow")

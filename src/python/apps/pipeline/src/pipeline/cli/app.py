"""Compose the pipeline command adapters into one Typer application."""

from __future__ import annotations

import typer
from dotell import begin, setup_logging

from pipeline.cli.commands import shared
from pipeline.cli.commands.audit import semflow
from pipeline.cli.commands.p1 import diagnostics as p1_diagnostics
from pipeline.cli.commands.p1 import dyadic_confound as p1_dyadic_confound
from pipeline.cli.commands.p1 import dyadic_oos as p1_dyadic_oos
from pipeline.cli.commands.p1 import etnic as p1_etnic
from pipeline.cli.commands.p1 import illustrations as p1_illustrations
from pipeline.cli.commands.p1 import (
    neighbour_outcome_split as p1_neighbour_outcome_split,
)
from pipeline.cli.commands.p1 import numbers as p1_numbers
from pipeline.cli.commands.p1 import pcoa as p1_pcoa
from pipeline.cli.commands.p1 import (
    representation_sensitivity as p1_representation_sensitivity,
)
from pipeline.cli.commands.p1 import sample_descriptives as p1_sample_descriptives
from pipeline.cli.commands.p1 import text_controls as p1_text_controls
from pipeline.cli.commands.p1 import w2_permanova as p1_w2_permanova
from pipeline.cli.commands.p1 import wasserstein_comparator as p1_wasserstein_comparator
from pipeline.cli.commands.p3 import certificate as p3_certificate
from pipeline.cli.commands.p3 import certificate_validation as p3_certificate_validation
from pipeline.cli.commands.p3 import illustrations as p3_illustrations
from pipeline.cli.commands.p3 import numbers as p3_numbers
from pipeline.cli.commands.p3 import pit_w2 as p3_pit_w2
from pipeline.cli.commands.p3 import portfolio_anatomy as p3_portfolio_anatomy
from pipeline.cli.commands.p3 import (
    representation_ablation as p3_representation_ablation,
)
from pipeline.cli.commands.p5 import barycentric_mixture as p5_barycentric_mixture
from pipeline.cli.commands.p5 import economic_value as p5_economic_value
from pipeline.cli.commands.p5 import gmv_validation as p5_gmv_validation
from pipeline.cli.commands.p5 import illustrations as p5_illustrations
from pipeline.cli.commands.p5 import network_certificate as p5_network_certificate
from pipeline.cli.commands.p5 import numbers as p5_numbers
from pipeline.cli.commands.p5 import peer_definitions as p5_peer_definitions
from pipeline.cli.commands.p5 import point_in_time_encoder as p5_point_in_time_encoder
from pipeline.cli.commands.p5 import post_corpus_qmle as p5_post_corpus_qmle
from pipeline.cli.commands.p5 import (
    representation_ablation as p5_representation_ablation,
)
from pipeline.cli.commands.p5 import sample_descriptives as p5_sample_descriptives
from pipeline.cli.commands.p5 import spatial_covariance as p5_spatial_covariance
from pipeline.cli.commands.p5 import (
    spatial_value_comparison as p5_spatial_value_comparison,
)
from pipeline.cli.commands.p5 import (
    transport_dimension_ladder as p5_transport_dimension_ladder,
)
from pipeline.cli.commands.p5 import (
    transport_order_sensitivity as p5_transport_order_sensitivity,
)
from pipeline.cli.commands.p5 import two_field_qmle as p5_two_field_qmle
from pipeline.cli.commands.p5 import (
    universe_horizon_bridge as p5_universe_horizon_bridge,
)
from pipeline.cli.commands.p6 import (
    article_redraw_validation as p6_article_redraw_validation,
)
from pipeline.cli.commands.p6 import corporate_analysis as p6_corporate_analysis
from pipeline.cli.commands.p6 import directional_validation as p6_directional_validation
from pipeline.cli.commands.p6 import inference_audit as p6_inference_audit
from pipeline.cli.commands.p6 import publication as p6_publication
from pipeline.cli.commands.p6 import recovery as p6_recovery
from pipeline.cli.commands.substrate import analysis, embeddings, market

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


app.add_typer(embeddings.commands)
app.add_typer(market.commands)
app.add_typer(analysis.commands)

app.add_typer(p1_diagnostics.commands)
app.add_typer(p1_dyadic_confound.commands)
app.add_typer(p1_dyadic_oos.commands)
app.add_typer(p1_etnic.commands)
app.add_typer(p1_illustrations.commands)
app.add_typer(p1_neighbour_outcome_split.commands)
app.add_typer(p1_numbers.commands)
app.add_typer(p1_pcoa.commands)
app.add_typer(p1_w2_permanova.commands)
app.add_typer(p1_representation_sensitivity.commands)
app.add_typer(p1_sample_descriptives.commands)
app.add_typer(p1_text_controls.commands)
app.add_typer(p1_wasserstein_comparator.commands)

app.add_typer(p3_certificate.commands)
app.add_typer(p3_certificate_validation.commands)
app.add_typer(p3_illustrations.commands)
app.add_typer(p3_numbers.commands)
app.add_typer(p3_pit_w2.commands)
app.add_typer(p3_portfolio_anatomy.commands)
app.add_typer(p3_representation_ablation.commands)

app.add_typer(p5_barycentric_mixture.commands)
app.add_typer(p5_economic_value.commands)
app.add_typer(p5_gmv_validation.commands)
app.add_typer(p5_illustrations.commands)
app.add_typer(p5_network_certificate.commands)
app.add_typer(p5_numbers.commands)
app.add_typer(p5_peer_definitions.commands)
app.add_typer(p5_point_in_time_encoder.commands)
app.add_typer(p5_post_corpus_qmle.commands)
app.add_typer(p5_representation_ablation.commands)
app.add_typer(p5_sample_descriptives.commands)
app.add_typer(p5_spatial_covariance.commands)
app.add_typer(p5_spatial_value_comparison.commands)
app.add_typer(p5_transport_dimension_ladder.commands)
app.add_typer(p5_transport_order_sensitivity.commands)
app.add_typer(p5_two_field_qmle.commands)
app.add_typer(p5_universe_horizon_bridge.commands)


app.add_typer(shared.commands)
app.add_typer(semflow.commands, name="semflow")

app.add_typer(p6_recovery.commands)
app.add_typer(p6_article_redraw_validation.commands)
app.add_typer(p6_inference_audit.commands)
app.add_typer(p6_directional_validation.commands)
app.add_typer(p6_corporate_analysis.commands)
app.add_typer(p6_publication.commands)

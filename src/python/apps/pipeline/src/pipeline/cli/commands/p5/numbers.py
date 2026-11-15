"""Paper 5 manuscript value catalog: one option per saved input."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper5-numbers")
def cli_render_paper5_numbers(
    post_corpus_qmle: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Post-corpus QMLE results."),
    ],
    transport_order_sensitivity: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Transport-order results."),
    ],
    spatial_value_comparison: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Spatial-value results."),
    ],
    economic_value: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Economic-value results."),
    ],
    point_in_time_encoder: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Point-in-time encoder results."
        ),
    ],
    transport_dimension_ladder: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Matryoshka ladder results."),
    ],
    two_field_qmle: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Two-field QMLE results."),
    ],
    network_certificate: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Network certificate results."),
    ],
    geometry_summary: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Typed W2 distance summary with the balanced cloud size.",
        ),
    ],
    universe_horizon_bridge: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Bridge results."),
    ],
    params_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="params.yaml; reads annual_raw_article_floor.",
        ),
    ],
    output_file: Annotated[
        Path, typer.Option(dir_okay=False, help="Generated numbers.tex.")
    ],
) -> None:
    """Generate number bindings for Paper 5."""
    from pipeline.figures.paper5.numbers import Paper5NumbersPaths, generate_numbers

    generate_numbers(
        Paper5NumbersPaths(
            post_corpus_qmle=post_corpus_qmle,
            transport_order_sensitivity=transport_order_sensitivity,
            spatial_value_comparison=spatial_value_comparison,
            economic_value=economic_value,
            point_in_time_encoder=point_in_time_encoder,
            transport_dimension_ladder=transport_dimension_ladder,
            two_field_qmle=two_field_qmle,
            network_certificate=network_certificate,
            geometry_summary=geometry_summary,
            universe_horizon_bridge=universe_horizon_bridge,
            params_file=params_file,
            output_file=output_file,
        )
    )

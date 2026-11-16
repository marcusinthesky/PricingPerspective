"""Paper 5 named peer baskets under three peer definitions: compute and render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("paper5-peer-definitions")
def cli_paper5_peer_definitions(
    returns_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="2018--2022 return panel."),
    ],
    w1_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen W1 artifact."),
    ],
    w2_barycentre_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen W2 artifact."),
    ],
    distance_artifact_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Typed W2 distances."),
    ],
    co_mentions_adjacency: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Co-mention adjacency."),
    ],
    universe_csv: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Universe classifications."),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Result folder to write.")
    ],
    targets: Annotated[str, typer.Option(help="Comma-separated target tickers.")],
    top_k: Annotated[int, typer.Option(min=1, help="Peers named per target.")],
    provider_id: Annotated[str, typer.Option(help="Embedding provider identity.")],
    representation_id: Annotated[str, typer.Option(help="Representation identity.")],
    w1_barycentre_arm_id: Annotated[str, typer.Option(help="W1 barycentre arm.")],
    w2_barycentre_arm_id: Annotated[str, typer.Option(help="W2 barycentre arm.")],
    w1_geometry_id: Annotated[str, typer.Option(help="W1 geometry identity.")],
    w2_geometry_id: Annotated[str, typer.Option(help="W2 geometry identity.")],
) -> None:
    """Write named top-k peer baskets under W2, RBF, and co-mention fields."""
    from pipeline.precision import own_float64

    own_float64()
    from pipeline.stages.papers.paper5.field_operators import (
        FieldOperatorIdentity,
        FieldOperatorPaths,
    )
    from pipeline.stages.papers.paper5.peer_definitions import (
        PeerDefinitionConfig,
        run_paper5_peer_definition_rows,
    )

    run_paper5_peer_definition_rows(
        FieldOperatorPaths(
            returns_dir=returns_dir,
            w1_barycentre_dir=w1_barycentre_dir,
            w2_barycentre_dir=w2_barycentre_dir,
            distance_artifact_dir=distance_artifact_dir,
            co_mentions_adjacency=co_mentions_adjacency,
            universe_csv=universe_csv,
        ),
        PeerDefinitionConfig(
            targets=tuple(
                value.strip() for value in targets.split(",") if value.strip()
            ),
            top_k=top_k,
            identity=FieldOperatorIdentity(
                provider_id=provider_id,
                representation_id=representation_id,
                w1_barycentre_arm_id=w1_barycentre_arm_id,
                w2_barycentre_arm_id=w2_barycentre_arm_id,
                w1_geometry_id=w1_geometry_id,
                w2_geometry_id=w2_geometry_id,
            ),
        ),
        output_dir,
    )


@commands.command("render-paper5-peer-definitions")
def cli_render_paper5_peer_definitions(
    rows_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Peer-definition rows.parquet."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            help="Folder receiving peer_basket.pgf and peer_definitions.pgf.",
        ),
    ],
    target: Annotated[str, typer.Option(help="Target of the W2 peer-basket figure.")],
    definitions_target: Annotated[
        str, typer.Option(help="Target of the three-definition comparison figure.")
    ],
) -> None:
    """Render Paper 5's named peer-basket and peer-definition exhibits."""
    from pipeline.figures.paper5.peer_basket import (
        PeerBasketFigurePaths,
        render_paper5_peer_basket_figures,
    )

    render_paper5_peer_basket_figures(
        PeerBasketFigurePaths(
            rows_path=rows_path,
            output_dir=output_dir,
            target=target,
            definitions_target=definitions_target,
        )
    )

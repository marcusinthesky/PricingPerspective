"""Paper 1 representation-sensitivity command adapters: estimate and tables."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-representation-sensitivity")
def cli_paper1_representation_sensitivity(
    pairs_file: Annotated[
        Path, typer.Option(help="Governed matched distance/return dyad artifact.")
    ],
    node_counts_file: Annotated[
        Path, typer.Option(help="Common node-bootstrap multiplicity schedule.")
    ],
    representation_dir: Annotated[
        list[Path],
        typer.Option(
            help=(
                "Non-native governed W2 representation artifact directory; "
                "repeatable. Provider and representation identities resolve "
                "against --params-file."
            )
        ),
    ],
    params_file: Annotated[
        Path,
        typer.Option(
            help=(
                "params.yaml; reads representation_ablation_specs, "
                "encoder_vintage_ablation_specs, and representation_contrast_specs."
            )
        ),
    ],
    output_summary: Annotated[
        Path, typer.Option(help="Representation-sensitivity results JSON.")
    ],
    output_draws: Annotated[
        Path, typer.Option(help="Standardized-effect bootstrap draws parquet.")
    ],
    vintage_dir: Annotated[
        list[Path] | None,
        typer.Option(
            help=(
                "Encoder-vintage W2 artifact directory; repeatable. Each must carry "
                "a provider_id with a matching encoder_vintage_ablation_specs cell."
            )
        ),
    ] = None,
    vintage_equivalence_bound: Annotated[
        float,
        typer.Option(
            min=0.0,
            help=(
                "Symmetric equivalence bound for paired standardized-effect contrasts."
            ),
        ),
    ] = 0.012,
    vintage_confidence_level: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Two-sided percentile confidence level for paired vintage contrasts.",
        ),
    ] = 0.95,
) -> None:
    """Compare encoders, widths, and distances under the primary estimator."""
    own_float64()
    from pipeline.stages.papers.paper1.representation_sensitivity import (
        RepresentationSensitivityConfig,
        run_representation_sensitivity,
    )

    result = run_representation_sensitivity(
        RepresentationSensitivityConfig(
            pairs_file=pairs_file,
            node_counts_file=node_counts_file,
            output_summary=output_summary,
            output_draws=output_draws,
            params_file=params_file,
            representation_dirs=tuple(representation_dir),
            vintage_dirs=tuple(vintage_dir or ()),
            vintage_equivalence_bound=vintage_equivalence_bound,
            vintage_confidence_level=vintage_confidence_level,
        )
    )
    cells = result["cells"]
    if not isinstance(cells, list):
        message = "representation sensitivity returned a non-list cells payload"
        raise TypeError(message)
    typer.echo(f"cells={len(cells)}")
    typer.echo(f"Saved: {output_summary}")
    typer.echo(f"Saved: {output_draws}")


@commands.command("render-paper1-representation-sensitivity")
def cli_render_paper1_representation_sensitivity(
    results_file: Annotated[
        Path, typer.Option(help="paper1-representation-sensitivity results JSON.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory for the generated table fragments.")
    ],
) -> None:
    """Render the representation and article-sampling sensitivity tables."""
    from pipeline.figures.paper1.representation_sensitivity import (
        Paper1RepresentationSensitivityRenderPaths,
        render_paper1_representation_sensitivity,
    )

    render_paper1_representation_sensitivity(
        Paper1RepresentationSensitivityRenderPaths(
            results_file=results_file, output_dir=output_dir
        )
    )

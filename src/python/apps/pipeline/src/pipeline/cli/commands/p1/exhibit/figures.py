"""P1 Exhibit Figures CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper1FiguresCliOptions(CliOptions):
    """Artifact inputs for Paper 1's data figures."""

    distance_artifact_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        description="Validated exact W2 artifact used by the primary figures.",
    )
    universe_csv: Path = Field(
        default=Path("data/universe.csv"),
        description="Universe metadata (Symbol, Sector, Industry).",
    )
    output_dir: Path = Field(
        default=Path("src/latex/projects/01_continuous_bounds/src/images"),
        description="Output directory for figure PDFs.",
    )


class Paper1DiagnosticsCliOptions(CliOptions):
    """Governed inputs and outputs for Paper 1 referee diagnostics."""

    full_distance_artifact_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        description="Validated full-width Qwen8B W2 artifact directory.",
    )
    distance_artifact_64_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-64-unit/wasserstein_w2"
        ),
        description="Validated 64-dimensional typed distance artifact directory.",
    )
    distance_artifact_256_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-256-unit/wasserstein_w2"
        ),
        description="Validated 256-dimensional typed distance artifact directory.",
    )
    provider_id: str = Field(
        default="qwen3-embedding-8b",
        description="Expected typed distance provider identity.",
    )
    representation_id: str = Field(
        default="qwen3-embedding-8b-unit",
        description="Expected full-width typed distance representation identity.",
    )
    representation_64_id: str = Field(
        default="qwen3-embedding-8b-64-unit",
        description="Expected 64-dimensional representation identity.",
    )
    representation_256_id: str = Field(
        default="qwen3-embedding-8b-256-unit",
        description="Expected 256-dimensional representation identity.",
    )
    distance_id: str = Field(
        default="wasserstein_w2",
        description="Expected primary typed statistical-distance identity.",
    )
    output_dir: Path = Field(
        default=Path("src/latex/projects/01_continuous_bounds/src/images"),
        description="Output directory for figure PDFs.",
    )
    summary_file: Path = Field(
        default=Path("data/papers/paper1/diagnostics/summary.yaml"),
        description="Output YAML of truncated-vs-full distance-agreement statistics.",
    )
    dyadic_summary_file: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/summary.yaml"),
        description="Dyadic primary/nested diagnostic summary.",
    )
    w2_primary_fit_file: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/w2_primary_fit.parquet"),
        description="Primary W2 fitted/residual/leverage array.",
    )
    lofo_file: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/leave_one_firm_out.parquet"),
        description="Leave-one-firm-out influence array.",
    )


class NeighbourOutcomeSplitRenderCliOptions(CliOptions):
    """Artifacts for the Paper 1 neighbour outcome-split publication leaf."""

    summary_file: Path = Path("data/papers/paper1/neighbour_outcome_split/summary.json")
    provenance_file: Path = Path(
        "data/papers/paper1/neighbour_outcome_split/provenance.json"
    )
    selection_file: Path = Path(
        "data/papers/paper1/neighbour_outcome_split/selection.json"
    )
    pair_paths_file: Path = Path(
        "data/papers/paper1/neighbour_outcome_split/pair_paths.parquet"
    )
    numbers_path: Path = Path(
        "src/latex/projects/01_continuous_bounds/src/generated/numbers.tex"
    )
    figure_path: Path = Path(
        "src/latex/projects/01_continuous_bounds/src/images/neighbour_holdout_paths.pgf"
    )
    contract_path: Path = Path(
        "src/latex/projects/01_continuous_bounds/src/generated/"
        "paper1_neighbour_outcome_split.interpretation.json"
    )


paper1_figures_command = model_command(Paper1FiguresCliOptions)
paper1_diagnostics_command = model_command(Paper1DiagnosticsCliOptions)
neighbour_outcome_split_render_command = model_command(
    NeighbourOutcomeSplitRenderCliOptions
)


@commands.command("render-paper1-figures", cls=paper1_figures_command)
def cli_render_paper1_figures(ctx: typer.Context) -> None:
    """Render Paper 1's PDF data-figures (numbers: ``render-paper1-numbers``)."""
    own_float64()
    from pipeline.figures.paper1.figures import Paper1FigurePaths, render_paper1_figures

    options = options_from_context(ctx, Paper1FiguresCliOptions)
    render_paper1_figures(
        Paper1FigurePaths(
            distance_artifact_dir=options.distance_artifact_dir,
            universe_csv=options.universe_csv,
            output_dir=options.output_dir,
        )
    )


@commands.command("render-paper1-illustrations")
def cli_render_paper1_illustrations(
    output_dir: Path = typer.Option(
        Path("src/latex/projects/01_continuous_bounds/src/images"),
        help="Output directory for schematic figure PDFs.",
    ),
) -> None:
    """Render Paper 1's analytic schematic figures (risk-factor space).

    Non-data-driven: the figures are closed-form illustrations with no ``data/``
    inputs, so this stage is separate from ``render-paper1-figures``.
    """
    from pipeline.figures.paper1.illustrations import render_paper1_illustrations

    render_paper1_illustrations(output_dir=output_dir)


@commands.command("render-paper1-diagnostics", cls=paper1_diagnostics_command)
def cli_render_paper1_diagnostics(ctx: typer.Context) -> None:
    """Render Paper 1's governed referee-response diagnostic figures."""
    own_float64()
    from pipeline.figures.paper1.diagnostics import (
        Paper1DiagnosticPaths,
        render_paper1_diagnostics,
    )

    options = options_from_context(ctx, Paper1DiagnosticsCliOptions)
    render_paper1_diagnostics(
        Paper1DiagnosticPaths(
            full_distance_artifact_dir=options.full_distance_artifact_dir,
            distance_artifact_64_dir=options.distance_artifact_64_dir,
            distance_artifact_256_dir=options.distance_artifact_256_dir,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            representation_64_id=options.representation_64_id,
            representation_256_id=options.representation_256_id,
            distance_id=options.distance_id,
            output_dir=options.output_dir,
            summary_file=options.summary_file,
            dyadic_summary_file=options.dyadic_summary_file,
            w2_primary_fit_file=options.w2_primary_fit_file,
            lofo_file=options.lofo_file,
        )
    )


@commands.command(
    "render-paper1-neighbour-outcome-split", cls=neighbour_outcome_split_render_command
)
def cli_render_paper1_neighbour_outcome_split(ctx: typer.Context) -> None:
    """Render the Paper 1 neighbour holdout and registered semantic contract."""
    from pipeline.figures.paper1.neighbour_outcome_split import (
        NeighbourOutcomeSplitRenderPaths,
        render_paper1_neighbour_outcome_split,
    )

    options = options_from_context(ctx, NeighbourOutcomeSplitRenderCliOptions)
    render_paper1_neighbour_outcome_split(
        NeighbourOutcomeSplitRenderPaths(
            summary_file=options.summary_file,
            provenance_file=options.provenance_file,
            selection_file=options.selection_file,
            pair_paths_file=options.pair_paths_file,
            numbers_path=options.numbers_path,
            figure_path=options.figure_path,
            contract_path=options.contract_path,
        )
    )

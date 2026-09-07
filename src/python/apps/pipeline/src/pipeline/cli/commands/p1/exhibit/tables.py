"""P1 Exhibit Tables CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context

commands = typer.Typer()


class Paper1NumbersCliOptions(CliOptions):
    """Governed source artifacts for Paper 1 number bindings."""

    numbers_path: Path = Field(
        default=Path(
            "src/latex/projects/01_continuous_bounds/src/generated/numbers.tex"
        ),
        description="Generated number bindings.",
    )
    dyadic_confound_summary: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/summary.yaml"),
        description="Dyadic confound-aware robustness summary YAML.",
    )
    disco_summary: Path = Field(
        default=Path("data/papers/paper1/disco/summary.yaml"),
        description=(
            "DISCO (energy-distance ANOVA) global sector-separation summary YAML (t19)."
        ),
    )
    sample_descriptives_summary: Path = Field(
        default=Path("data/papers/paper1/sample_descriptives/summary.yaml"),
        description=(
            "Sample descriptives (sector panel, article counts, volatility) "
            "summary YAML."
        ),
    )
    diagnostics_summary: Path = Field(
        default=Path("data/papers/paper1/diagnostics/summary.yaml"),
        description=(
            "Truncated-vs-full W2-distance agreement statistics summary YAML."
        ),
    )
    dimensionality_csv: Path = Field(
        default=Path(
            "data/shared/dimensionality_analysis/qwen3-embedding-8b/"
            "dimensionality_analysis.csv"
        ),
        description="MDS projection-fidelity summary CSV.",
    )
    neighbour_outcome_split_summary: Path = Field(
        default=Path("data/papers/paper1/neighbour_outcome_split/summary.json"),
        description="Preregistered original-space neighbour outcome-split summary.",
    )
    covariance_matrix: Path = Field(
        default=Path("data/shared/covariance_matrix.parquet"),
        description=(
            "Governed covariance artifact carrying exact return-panel metadata."
        ),
    )
    oos_covariance_matrix: Path | None = Field(
        default=Path("data/shared/oos/covariance_matrix.parquet"),
        description=(
            "OOS covariance artifact carrying the surviving-panel metadata. "
            "Skipped when absent."
        ),
    )
    oos_dyadic_summary: Path | None = Field(
        default=Path("data/papers/paper1/dyadic_oos/summary.yaml"),
        description=(
            "t37 out-of-sample W2 dyadic summary on the frozen text matrix. "
            "Skipped when absent."
        ),
    )


class Paper1TablesCliOptions(CliOptions):
    """Source artifacts and outputs for Paper 1 LaTeX tables."""

    summary_file: Path = Field(
        default=Path("data/papers/paper1/sample_descriptives/summary.yaml"),
        description="Sample-descriptives summary YAML.",
    )
    output_file: Path = Field(
        default=Path(
            "src/latex/projects/01_continuous_bounds/src/generated/"
            "sample_descriptives.tex"
        ),
        description="Generated LaTeX table fragment.",
    )
    dyadic_summary: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/summary.yaml"),
        description="Node-bootstrap dyadic ladder summary.",
    )
    disco_summary: Path = Field(
        default=Path("data/papers/paper1/disco/summary.yaml"),
        description="DISCO decomposition summary.",
    )
    representation_summary: Path = Field(
        default=Path("data/papers/paper1/representation_sensitivity/summary.yaml"),
        description="Symmetric-FE W2 representation sensitivity.",
    )
    oos_dyadic_summary: Path = Field(
        default=Path("data/papers/paper1/dyadic_oos/summary.yaml"),
        description="Timing-separated primary W2 association.",
    )
    dyadic_design_diagnostics: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/design_diagnostics.parquet"),
        description="Pre-fit dyadic design audit parquet.",
    )
    output_dir: Path = Field(
        default=Path("src/latex/projects/01_continuous_bounds/src/generated"),
        description="Directory for empirical-result table fragments.",
    )


paper1_numbers_command = model_command(Paper1NumbersCliOptions)
paper1_tables_command = model_command(Paper1TablesCliOptions)


@commands.command("render-paper1-numbers", cls=paper1_numbers_command)
def cli_render_paper1_numbers(ctx: typer.Context) -> None:
    """Write Paper 1's number bindings (t15 R3 split from figures)."""
    from pipeline.figures.paper1.numbers import Paper1NumberPaths, render_paper1_numbers

    options = options_from_context(ctx, Paper1NumbersCliOptions)
    render_paper1_numbers(
        Paper1NumberPaths(
            numbers_path=options.numbers_path,
            params_path=Path("params.yaml"),
            corpus_summary=Path("data/shared/corpus/summary.yaml"),
            dyadic_confound_summary=options.dyadic_confound_summary,
            disco_summary=options.disco_summary,
            diagnostics_summary=options.diagnostics_summary,
            sample_descriptives_summary=options.sample_descriptives_summary,
            dimensionality_csv=options.dimensionality_csv,
            neighbour_outcome_split_summary=options.neighbour_outcome_split_summary,
            covariance_matrix=options.covariance_matrix,
            oos_covariance_matrix=options.oos_covariance_matrix,
            oos_dyadic_summary=options.oos_dyadic_summary,
        )
    )


@commands.command("render-paper1-tables", cls=paper1_tables_command)
def cli_render_paper1_tables(ctx: typer.Context) -> None:
    """Render Paper 1's sample-descriptives booktabs tables.

    Emit the per-sector and full-universe listings as LaTeX fragments.
    """
    from pipeline.figures.paper1.tables import (
        Paper1EmpiricalTablePaths,
        render_paper1_empirical_tables,
        render_sample_descriptives_table,
    )

    options = options_from_context(ctx, Paper1TablesCliOptions)
    render_sample_descriptives_table(
        summary_file=options.summary_file,
        output_file=options.output_file,
    )
    render_paper1_empirical_tables(
        Paper1EmpiricalTablePaths(
            dyadic_summary=options.dyadic_summary,
            disco_summary=options.disco_summary,
            representation_summary=options.representation_summary,
            oos_dyadic_summary=options.oos_dyadic_summary,
            dyadic_design_diagnostics=options.dyadic_design_diagnostics,
            output_dir=options.output_dir,
        )
    )

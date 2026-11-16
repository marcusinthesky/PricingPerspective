"""Paper 1 number-binding command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

commands = typer.Typer()


@commands.command("render-paper1-numbers")
def cli_render_paper1_numbers(
    universe_csv: Annotated[Path, typer.Option(help="Industry classification roster.")],
    source_articles_dir: Annotated[
        Path,
        typer.Option(help="Retained capped article pool before estimation selection."),
    ],
    numbers_path: Annotated[Path, typer.Option(help="Generated number bindings.")],
    params_file: Annotated[
        Path,
        typer.Option(
            help=(
                "params.yaml; reads symbols, matryoshka_range, start_date, "
                "end_date, n_samples, annual_raw_article_floor, and "
                "embedding.models.qwen3-embedding-8b.dims, and "
                "paper1_text_controls.max_features."
            )
        ),
    ],
    corpus_summary: Annotated[
        Path, typer.Option(help="Shared news-corpus summary YAML.")
    ],
    dyadic_confound_summary: Annotated[
        Path, typer.Option(help="paper1-dyadic-confound results JSON.")
    ],
    w2_permanova_summary: Annotated[
        Path, typer.Option(help="paper1-w2-permanova results JSON.")
    ],
    sample_descriptives_summary: Annotated[
        Path, typer.Option(help="compute-sample-descriptives results JSON.")
    ],
    diagnostics_summary: Annotated[
        Path, typer.Option(help="paper1-diagnostics results JSON.")
    ],
    pcoa_summary: Annotated[
        Path, typer.Option(help="Saved PCoA projection and spectrum results JSON.")
    ],
    neighbour_outcome_split_summary: Annotated[
        Path, typer.Option(help="paper1-neighbour-outcome-split results JSON.")
    ],
    covariance_matrix: Annotated[
        Path,
        typer.Option(help="Covariance artifact carrying exact return-panel metadata."),
    ],
    oos_covariance_matrix: Annotated[
        Path | None,
        typer.Option(
            help="OOS covariance artifact carrying the surviving-panel metadata."
        ),
    ] = None,
    oos_dyadic_summary: Annotated[
        Path | None, typer.Option(help="paper1-dyadic-oos results JSON.")
    ] = None,
    text_controls_summary: Annotated[
        Path | None, typer.Option(help="paper1-text-controls results JSON.")
    ] = None,
) -> None:
    """Write Paper 1's number bindings from saved results."""
    from pipeline.figures.paper1.numbers import Paper1NumberPaths, render_paper1_numbers

    render_paper1_numbers(
        Paper1NumberPaths(
            universe_csv=universe_csv,
            source_articles_dir=source_articles_dir,
            numbers_path=numbers_path,
            params_path=params_file,
            corpus_summary=corpus_summary,
            dyadic_confound_results=dyadic_confound_summary,
            w2_permanova_results=w2_permanova_summary,
            diagnostics_results=diagnostics_summary,
            sample_descriptives_results=sample_descriptives_summary,
            pcoa_summary=pcoa_summary,
            neighbour_outcome_split_results=neighbour_outcome_split_summary,
            covariance_matrix=covariance_matrix,
            oos_covariance_matrix=oos_covariance_matrix,
            oos_dyadic_results=oos_dyadic_summary,
            text_controls_results=text_controls_summary,
        )
    )

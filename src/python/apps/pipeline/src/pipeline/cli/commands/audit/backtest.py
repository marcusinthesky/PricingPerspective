"""Audit Backtest CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer

commands = typer.Typer()


@commands.command("backtest-feasibility")
def cli_backtest_feasibility(
    returns_dir: Path = typer.Option(
        Path("data/shared/returns"),
        help="Directory of per-ticker return parquets (the traded universe).",
    ),
    corpus_path: Path = typer.Option(
        Path("data/shared/corpus/articles.parquet"),
        help="Raw un-embedded news corpus (symbol, created_date).",
    ),
    params_file: Path = typer.Option(
        Path("params.yaml"),
        help="Params file with the backtest_feasibility block (T-grid, "
        "thresholds, window-types).",
    ),
    output_dir: Path = typer.Option(
        Path("data/shared/backtest_feasibility"),
        help="Output directory for the constructibility table + figure + summary.",
    ),
) -> None:
    """Assess the minimal feasible backtest window per channel (returns/news)."""
    from pipeline.stages.analysis.backtest_feasibility import run_backtest_feasibility

    run_backtest_feasibility(returns_dir, corpus_path, params_file, output_dir)

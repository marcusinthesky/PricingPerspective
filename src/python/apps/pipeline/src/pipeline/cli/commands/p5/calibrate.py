"""P5 Calibrate CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer

commands = typer.Typer()


@commands.command("paper5-mc")
def cli_paper5_mc(
    output_dir: Path = typer.Option(
        Path("data/papers/paper5"),
        help="Output directory for mc_results.json.",
    ),
    figures_dir: Path = typer.Option(
        Path("src/latex/projects/05_spatial_pricing/src/images"),
        help=(
            "Output directory for the mc_w_misspecification/mc_size_power figure PDFs."
        ),
    ),
    n_reps: int = typer.Option(500, help="Monte-Carlo replications."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Run Paper 5's spatial-weight misspecification / size-power Monte Carlo."""
    from pipeline.stages.papers.paper5.mc import run_paper5_mc

    run_paper5_mc(output_dir, n_reps=n_reps, seed=seed, figures_dir=figures_dir)

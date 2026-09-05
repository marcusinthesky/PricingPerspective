"""P3 Exhibit CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer

commands = typer.Typer()


@commands.command("render-paper3-portfolio-anatomy")
def cli_render_paper3_portfolio_anatomy(
    anatomy_dir: Path = typer.Option(Path("data/papers/paper3/portfolio_anatomy")),
    output_dir: Path = typer.Option(
        Path("src/latex/projects/03_distance_implied_mpt/src/images")
    ),
    table_path: Path = typer.Option(
        Path(
            "src/latex/projects/03_distance_implied_mpt/src/generated/"
            "portfolio_holdings.tex"
        )
    ),
    manifest_path: Path = typer.Option(
        Path(
            "src/latex/projects/03_distance_implied_mpt/src/images/"
            "portfolio_anatomy.manifest.json"
        )
    ),
) -> None:
    """Render Paper 3's allocation anatomy and expanding-vintage exhibits."""
    from pipeline.figures.paper3.portfolio_anatomy import (
        render_paper3_portfolio_anatomy,
    )

    render_paper3_portfolio_anatomy(
        anatomy_dir,
        output_dir,
        table_path,
        manifest_path,
    )


@commands.command("render-paper3-representation-ablation-table")
def cli_render_paper3_representation_ablation_table(
    summary_path: Path = typer.Option(
        Path("data/papers/paper3/representation_ablation/summary.yaml")
    ),
    output_path: Path = typer.Option(
        Path(
            "src/latex/projects/03_distance_implied_mpt/src/generated/"
            "representation_ablation.tex"
        )
    ),
    main_output_path: Path = typer.Option(
        Path(
            "src/latex/projects/03_distance_implied_mpt/src/generated/"
            "representation_ablation_main.tex"
        )
    ),
) -> None:
    """Render Paper 3's main-text summary and appendix robustness tables."""
    from pipeline.figures.paper3.tables import (
        render_paper3_representation_ablation_table,
    )

    render_paper3_representation_ablation_table(
        summary_path,
        output_path,
        main_output_path,
    )


@commands.command("render-paper3-figures")
def cli_render_paper3_figures(
    paper3_dir: Path = typer.Option(
        Path("data/papers/paper3/empirical"),
        help="Paper 3 empirical results directory.",
    ),
    pricefree_dir: Path = typer.Option(
        Path("data/mc/pricefree_comparison"),
        help="Price-free regret MC directory.",
    ),
    output_dir: Path = typer.Option(
        Path("src/latex/projects/03_distance_implied_mpt/src/images"),
        help="Output directory for figure PDFs.",
    ),
    numbers_path: Path = typer.Option(
        Path("src/latex/projects/03_distance_implied_mpt/src/generated/numbers.tex"),
        help="Generated number bindings.",
    ),
    size_power_path: Path = typer.Option(
        Path("data/mc/gmm_wolak_size_power/summary.yaml"),
        help="GMM/Wolak size-power calibration summary.",
    ),
    certificate_dir: Path = typer.Option(Path("data/papers/paper3/certificate")),
) -> None:
    """Render figures and number bindings for Paper 3."""
    from pipeline.figures.paper3.figures import Paper3FigurePaths, render_paper3_figures

    render_paper3_figures(
        Paper3FigurePaths(
            paper3_dir=paper3_dir,
            pricefree_dir=pricefree_dir,
            output_dir=output_dir,
            numbers_path=numbers_path,
            params_path=Path("params.yaml"),
            mc_validation_dir=Path("data/mc/monte_carlo_validation"),
            hedging_validation_dir=Path("data/mc/hedging_error_validation"),
            cross_model_summary=Path(
                "data/shared/ablations/paper3_model_robustness/summary.yaml"
            ),
            size_power_path=size_power_path,
            certificate_dir=certificate_dir,
        )
    )


@commands.command("paper3-numbers")
def cli_paper3_numbers(
    paper3_dir: Path = typer.Option(Path("data/papers/paper3/empirical")),
    numbers_path: Path = typer.Option(
        Path("src/latex/projects/03_distance_implied_mpt/src/generated/numbers.tex")
    ),
    pricefree_dir: Path = typer.Option(Path("data/mc/pricefree_comparison")),
    certificate_dir: Path = typer.Option(Path("data/papers/paper3/certificate")),
    validation_dir: Path = typer.Option(
        Path("data/papers/paper3/certificate_validation")
    ),
    manifest_path: Path = typer.Option(
        Path(
            "src/latex/projects/03_distance_implied_mpt/src/generated/numbers.manifest.json"
        )
    ),
) -> None:
    """Emit Paper 3 generated numbers and their provenance manifest."""
    from pipeline.figures.paper3.manifests import write_paper3_manifest
    from pipeline.figures.paper3.numbers import Paper3NumberPaths, render_paper3_numbers

    anatomy_dir = Path("data/papers/paper3/portfolio_anatomy")
    render_paper3_numbers(
        Paper3NumberPaths(
            paper3_dir=paper3_dir,
            numbers_path=numbers_path,
            pricefree_dir=pricefree_dir,
            certificate_dir=certificate_dir,
            validation_dir=validation_dir,
            anatomy_dir=anatomy_dir,
        )
    )
    write_paper3_manifest(
        manifest_path,
        stage_key="p3_numbers",
        phase="exhibit",
        outputs=(numbers_path,),
        upstreams=(
            certificate_dir / "summary.yaml",
            validation_dir / "summary.yaml",
            anatomy_dir / "summary.yaml",
        ),
    )


@commands.command("paper3-exhibits")
def cli_paper3_exhibits(
    paper3_dir: Path = typer.Option(Path("data/papers/paper3/empirical")),
    pricefree_dir: Path = typer.Option(Path("data/mc/pricefree_comparison")),
    certificate_dir: Path = typer.Option(Path("data/papers/paper3/certificate")),
    validation_dir: Path = typer.Option(
        Path("data/papers/paper3/certificate_validation")
    ),
    output_dir: Path = typer.Option(
        Path("src/latex/projects/03_distance_implied_mpt/src/images")
    ),
    manifest_path: Path = typer.Option(
        Path(
            "src/latex/projects/03_distance_implied_mpt/src/images/exhibits.manifest.json"
        )
    ),
) -> None:
    """Emit Paper 3 PDFs and their provenance manifest."""
    from pipeline.figures.paper3.exhibits import render_paper3_exhibits

    render_paper3_exhibits(
        paper3_dir,
        pricefree_dir,
        certificate_dir,
        validation_dir,
        output_dir,
        manifest_path,
    )

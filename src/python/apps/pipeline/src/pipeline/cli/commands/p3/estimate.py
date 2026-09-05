"""P3 Estimate CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper3-portfolio-anatomy")
def cli_paper3_portfolio_anatomy(
    weights_path: Path = typer.Option(
        Path("data/papers/paper3/certificate/news_only_weights.parquet")
    ),
    distance_artifact: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    ),
    pit_w2_path: Path = typer.Option(Path("data/papers/paper3/pit_w2/matrices.npz")),
    universe_csv: Path = typer.Option(Path("data/universe.csv")),
    output_dir: Path = typer.Option(Path("data/papers/paper3/portfolio_anatomy")),
) -> None:
    """Decompose Paper 3's canonical allocation and expanding vintages."""
    own_float64()
    from pipeline.stages.papers.paper3.portfolio_anatomy import (
        Paper3PortfolioAnatomyPaths,
        run_paper3_portfolio_anatomy,
    )

    run_paper3_portfolio_anatomy(
        Paper3PortfolioAnatomyPaths(
            weights_path=weights_path,
            distance_artifact_dir=distance_artifact,
            pit_w2_path=pit_w2_path,
            pit_manifest_path=pit_w2_path.with_name("provenance.manifest.json"),
            universe_csv=universe_csv,
            output_dir=output_dir,
        )
    )


@commands.command("paper3-certificate-validation")
def cli_paper3_certificate_validation(
    params_file: Path = typer.Option(Path("params.yaml")),
    returns_dir: Path = typer.Option(Path("data/shared/returns")),
    holdout_returns_dir: Path = typer.Option(Path("data/shared/oos/returns")),
    pit_w2_path: Path = typer.Option(Path("data/papers/paper3/pit_w2/matrices.npz")),
    sample_descriptives_path: Path = typer.Option(
        Path("data/papers/paper1/sample_descriptives/summary.yaml")
    ),
    output_dir: Path = typer.Option(Path("data/papers/paper3/certificate_validation")),
) -> None:
    """Calibrate then evaluate Paper 3's frozen chronological certificate."""
    own_float64()
    from pipeline.stages.papers.paper3.certificate_validation import (
        Paper3CertificateValidationPaths,
        run_paper3_certificate_validation,
    )

    run_paper3_certificate_validation(
        Paper3CertificateValidationPaths(
            params_file=params_file,
            returns_dir=returns_dir,
            holdout_returns_dir=holdout_returns_dir,
            pit_w2_path=pit_w2_path,
            sample_descriptives_path=sample_descriptives_path,
            output_dir=output_dir,
        )
    )


@commands.command("paper3-certificate")
def cli_paper3_certificate(
    params_file: Path = typer.Option(Path("params.yaml")),
    returns_dir: Path = typer.Option(Path("data/shared/returns")),
    distance_artifact: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    ),
    output_dir: Path = typer.Option(Path("data/papers/paper3/certificate")),
) -> None:
    """Compute Paper 3's governed full-sample certificate sensitivity surface."""
    own_float64()
    from pipeline.stages.papers.paper3.certificate import (
        Paper3CertificatePaths,
        run_paper3_certificate,
    )

    run_paper3_certificate(
        Paper3CertificatePaths(
            params_file=params_file,
            returns_dir=returns_dir,
            distance_artifact_dir=distance_artifact,
            output_dir=output_dir,
        )
    )


@commands.command("paper3-representation-ablation")
def cli_paper3_representation_ablation(
    cell_id: str = typer.Option(...),
    params_file: Path = typer.Option(Path("params.yaml")),
    returns_dir: Path = typer.Option(Path("data/shared/returns")),
    distance_artifact: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
    canonical_distance_artifact: Path | None = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    ),
) -> None:
    """Evaluate one Paper 3 representation under the fixed variance-rank laws."""
    own_float64()
    from pipeline.stages.papers.paper3.representation_ablation import (
        Paper3RepresentationAblationPaths,
        run_paper3_representation_ablation,
    )

    run_paper3_representation_ablation(
        Paper3RepresentationAblationPaths(
            params_file=params_file,
            returns_dir=returns_dir,
            distance_artifact_dir=distance_artifact,
            output_dir=output_dir,
            canonical_distance_artifact_dir=canonical_distance_artifact,
        ),
        cell_id,
    )


@commands.command("merge-paper3-representation-ablation")
def cli_merge_paper3_representation_ablation(
    params_file: Path = typer.Option(Path("params.yaml")),
    cells_dir: Path = typer.Option(Path("data/papers/paper3/representation_ablation")),
    canonical_certificate_summary: Path = typer.Option(
        Path("data/papers/paper3/certificate/summary.yaml")
    ),
    output_summary: Path = typer.Option(
        Path("data/papers/paper3/representation_ablation/summary.yaml")
    ),
    returns_dir: Path | None = typer.Option(None),
    contrast_draws_dir: Path | None = typer.Option(None),
) -> None:
    """Merge the complete Paper 3 representation roster."""
    from pipeline.stages.papers.paper3.representation_ablation import (
        Paper3RepresentationMergePaths,
        merge_paper3_representation_ablation,
    )

    merge_paper3_representation_ablation(
        Paper3RepresentationMergePaths(
            params_file=params_file,
            cells_dir=cells_dir,
            canonical_certificate_summary=canonical_certificate_summary,
            output_summary=output_summary,
            returns_dir=returns_dir,
            contrast_draws_dir=contrast_draws_dir,
        )
    )


@commands.command("paper3-pit-w2")
def cli_paper3_pit_w2(
    params_file: Path = typer.Option(
        Path("params.yaml"),
        help="Path to params.yaml (reads paper3.pit_vintage_grid).",
    ),
    out_path: Path = typer.Option(
        Path("data/papers/paper3/pit_w2/matrices.npz"),
        help="Output .npz path for PIT squared-W2 matrices.",
    ),
    distance_root: Path = typer.Option(
        Path("data/shared/typed_distances_windowed"),
        help="Root of the window-scoped typed statistical-distance tree.",
    ),
) -> None:
    """Assemble point-in-time squared-W2 matrices from typed artifacts."""
    from pipeline.stages.papers.paper3.pit_distance import compute_paper3_pit_distance

    compute_paper3_pit_distance(params_file, out_path, distance_root)


class Paper3EmpiricalOptions(CliOptions):
    """Inputs for Paper 3's covariance empirics."""

    returns_dir: Path = Field(
        default=Path("data/shared/returns"),
        description="Directory of per-ticker return parquets.",
    )
    distance_artifact: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        description="Typed statistical-distance artifact directory.",
    )
    output_dir: Path = Field(
        default=Path("data/papers/paper3/empirical"),
        description="Output directory for results.parquet + summary.yaml.",
    )
    n_boot: int = Field(default=500, description="Bootstrap resamples for Sharpe CIs.")
    n_mc: int = Field(
        default=3000,
        description="Monte-Carlo draws for chi-bar-squared.",
    )
    n_clusters: int = Field(
        default=15,
        description="Pair->portfolio clusters (10-20).",
    )
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    distance_id: str = Field(default="wasserstein_w2")


paper3_empirical_command = model_command(Paper3EmpiricalOptions)


@commands.command("paper3-empirical", cls=paper3_empirical_command)
def cli_paper3_empirical(ctx: typer.Context) -> None:
    """Run Paper 3's distance-implied covariance → minimum-variance empirics."""
    own_float64()
    from pipeline.stages.papers.paper3.empirical import (
        Paper3EmpiricalConfig,
        Paper3EmpiricalPaths,
        run_paper3_empirical,
    )

    options = options_from_context(ctx, Paper3EmpiricalOptions)
    run_paper3_empirical(
        Paper3EmpiricalPaths(
            returns_dir=options.returns_dir,
            distance_artifact_dir=options.distance_artifact,
            output_dir=options.output_dir,
        ),
        Paper3EmpiricalConfig(
            n_clusters=options.n_clusters,
            n_boot=options.n_boot,
            n_mc=options.n_mc,
            provider_id=options.provider_id,
            representation_id=options.representation_id,
            distance_id=options.distance_id,
        ),
    )

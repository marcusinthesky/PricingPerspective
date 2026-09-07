"""P1 Estimate Identification CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class DyadicConfoundCliOptions(CliOptions):
    """Artifact inputs for the confound-aware dyadic regression."""

    distance_artifact_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        ),
        description="Primary Qwen3-Embedding 8B W2 artifact.",
    )
    energy_comparator_artifact_dir: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/energy_v"
        ),
        description="Matched Qwen3-Embedding 8B energy-distance comparator artifact.",
    )
    covariance_matrix: Path = Field(
        default=Path("data/shared/covariance_matrix.parquet"),
        description="Sample covariance/correlation matrix parquet.",
    )
    universe_csv: Path = Field(
        default=Path("data/universe.csv"),
        description="Universe metadata (Symbol, Sector, Industry).",
    )
    market_data_dir: Path = Field(
        default=Path("data/shared/market_data"),
        description="Directory of per-ticker OHLCV parquets.",
    )
    returns_dir: Path = Field(
        default=Path("data/shared/returns"),
        description="Directory of per-ticker return parquets.",
    )
    output_file: Path = Field(
        default=Path("data/papers/paper1/dyadic_confound/summary.yaml"),
        description="Output YAML summary path.",
    )
    embeddings_dir: Path = Field(
        default=Path("data/shared/embeddings/qwen3-embedding-8b"),
        description=(
            "Directory of per-ticker embedding parquets (for the cosine_dist "
            "rung of the model ladder)."
        ),
    )


dyadic_confound_command = model_command(DyadicConfoundCliOptions)


@commands.command("compute-dyadic-confound", cls=dyadic_confound_command)
def cli_compute_dyadic_confound(ctx: typer.Context) -> None:
    """Run confound-aware dyadic regression.

    Run the fixed-W2 random-exposure diagnostic and its dyadic regression,
    with the Energy-distance ladder retained as a corollary and multinomial
    node-bootstrap inference.
    """
    own_float64()
    from pipeline.stages.papers.paper1.dyadic_confound import (
        DyadicConfoundConfig,
        run_dyadic_confound,
    )

    options = options_from_context(ctx, DyadicConfoundCliOptions)
    result = run_dyadic_confound(
        DyadicConfoundConfig(
            distance_artifact_dir=options.distance_artifact_dir,
            energy_comparator_artifact_dir=options.energy_comparator_artifact_dir,
            covariance_matrix=options.covariance_matrix,
            universe_csv=options.universe_csv,
            market_data_dir=options.market_data_dir,
            returns_dir=options.returns_dir,
            output_file=options.output_file,
            embeddings_dir=options.embeddings_dir,
        )
    )
    typer.echo(f"n_dyads={result['n_dyads']}  n_firms={result['n_firms']}")
    typer.echo(
        f"wasserstein_w2 [primary]: coef={result['coef_w2_distance']:.6f}  "
        f"se={result['se_w2_distance']:.6f}  "
        f"p={result['pvalue_w2_distance']:.6f}"
    )
    energy_fit = result.get("energy_dyadic_comparator", {})
    if isinstance(energy_fit, dict):
        typer.echo(
            f"energy_v [comparator]: coef={energy_fit['coefficient']:.6f}  "
            f"se={energy_fit['se']:.6f}  p={energy_fit['pvalue']:.6f}"
        )
    w2_fit = result.get("w2_primary_regression", {})
    if isinstance(w2_fit, dict) and w2_fit.get("status") == "computed":
        typer.echo(
            f"fixed_w2: coef={w2_fit['coefficient']:.6f}  "
            f"se={w2_fit['se']:.6f}  p={w2_fit['pvalue']:.6f}"
        )
        diagnostic = result.get("random_exposure_diagnostic", {})
        empty: list[dict[str, object]] = []
        scenarios = (
            diagnostic.get("scenarios", empty)
            if isinstance(diagnostic, dict)
            else empty
        )
        if scenarios:
            baseline = scenarios[0]
            typer.echo(
                f"fixed_w2_envelope(L={baseline['L']}, tau={baseline['tau']}): "
                f"coverage={baseline['coverage_share']:.6f}  "
                f"violation={baseline['violation_share']:.6f}"
            )
    typer.echo(f"Saved: {options.output_file}")


@commands.command("compute-disco")
def cli_compute_disco(
    distance_artifact_dir: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/qwen3-embedding-8b-unit/energy_v"
        ),
        help="Validated typed distance artifact directory.",
    ),
    universe_csv: Path = typer.Option(
        Path("data/universe.csv"), help="Universe metadata (Symbol, Sector, Industry)."
    ),
    output_file: Path = typer.Option(
        Path("data/papers/paper1/disco/summary.yaml"),
        help="Output YAML summary path.",
    ),
    provider_id: str = typer.Option(
        "qwen3-embedding-8b", help="Expected typed-distance provider identity."
    ),
    representation_id: str = typer.Option(
        "qwen3-embedding-8b-unit",
        help="Expected typed-distance representation identity.",
    ),
    distance_id: str = typer.Option(
        "energy_v", help="Expected energy-distance identity."
    ),
    n_permutations: int = typer.Option(
        9_999, help="Number of label permutations for the null."
    ),
    random_state: int = typer.Option(42, help="RNG seed for the permutation null."),
) -> None:
    """Compute the DISCO global sector-separation statistic.

    Use the raw energy-distance matrix and report a distance-to-median
    dispersion companion under a permutation null (t19).
    """
    own_float64()
    from pipeline.stages.papers.paper1.disco import run_disco

    result = run_disco(
        distance_artifact_dir=distance_artifact_dir,
        universe_csv=universe_csv,
        output_file=output_file,
        n_permutations=n_permutations,
        random_state=random_state,
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id=distance_id,
    )
    typer.echo(
        f"R2_E={result['r2_e']:.4f} F={result['f_statistic']:.4f} "
        f"(perm p={result['disco_perm_p']:.4f}); "
        f"dispersion F={result['dispersion_f_statistic']:.4f} "
        f"(perm p={result['dispersion_perm_p']:.4f}); reading={result['reading']}"
    )
    typer.echo(f"Saved: {output_file}")

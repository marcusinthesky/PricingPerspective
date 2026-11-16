"""Paper 1 dyadic confound command adapters: the estimate and its tables."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("paper1-dyadic-confound")
def cli_paper1_dyadic_confound(
    distance_artifact_dir: Annotated[
        Path, typer.Option(help="Primary Qwen3-Embedding 8B W2 artifact.")
    ],
    energy_comparator_artifact_dir: Annotated[
        Path,
        typer.Option(
            help="Matched Qwen3-Embedding 8B energy-distance comparator artifact."
        ),
    ],
    covariance_matrix: Annotated[
        Path, typer.Option(help="Sample covariance/correlation matrix parquet.")
    ],
    universe_csv: Annotated[
        Path, typer.Option(help="Universe metadata (Symbol, Sector, Industry).")
    ],
    liquidity_proxy_file: Annotated[
        Path, typer.Option(help="Published per-firm numeric liquidity proxies.")
    ],
    returns_dir: Annotated[
        Path, typer.Option(help="Directory of per-ticker return parquets.")
    ],
    embeddings_dir: Annotated[
        Path,
        typer.Option(
            help=(
                "Directory of per-ticker embedding parquets (for the cosine_dist "
                "rung of the model ladder)."
            )
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            help="Results JSON; the auditable parquets are written beside it."
        ),
    ],
    etnic_alignment_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "ETNIC alignment result folder (pairs.parquet, results.json); "
                "adds the ETNIC-roster classification rows."
            )
        ),
    ] = None,
) -> None:
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

    result = run_dyadic_confound(
        DyadicConfoundConfig(
            distance_artifact_dir=distance_artifact_dir,
            energy_comparator_artifact_dir=energy_comparator_artifact_dir,
            covariance_matrix=covariance_matrix,
            universe_csv=universe_csv,
            liquidity_proxy_file=liquidity_proxy_file,
            returns_dir=returns_dir,
            output_file=output_file,
            embeddings_dir=embeddings_dir,
            etnic_alignment_dir=etnic_alignment_dir,
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
    typer.echo(f"Saved: {output_file}")


@commands.command("render-paper1-dyadic-confound")
def cli_render_paper1_dyadic_confound(
    dyadic_results_file: Annotated[
        Path, typer.Option(help="paper1-dyadic-confound results JSON.")
    ],
    oos_dyadic_results_file: Annotated[
        Path, typer.Option(help="paper1-dyadic-oos results JSON.")
    ],
    design_diagnostics_file: Annotated[
        Path, typer.Option(help="Pre-fit dyadic design audit parquet.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory for the generated table fragments.")
    ],
) -> None:
    """Render the primary-result, ladder, control, design, and envelope tables."""
    from pipeline.figures.paper1.dyadic_confound import (
        Paper1DyadicConfoundRenderPaths,
        render_paper1_dyadic_confound,
    )

    render_paper1_dyadic_confound(
        Paper1DyadicConfoundRenderPaths(
            dyadic_results_file=dyadic_results_file,
            oos_dyadic_results_file=oos_dyadic_results_file,
            design_diagnostics_file=design_diagnostics_file,
            output_dir=output_dir,
        )
    )

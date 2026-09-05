"""P1 Estimate Geometry CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("compute-wasserstein-comparator")
def cli_compute_wasserstein_comparator(
    embeddings_root: Path = typer.Option(
        Path("data/shared/embeddings"),
        help="Root containing the four locked per-model embedding directories.",
    ),
    typed_distance_root: Path = typer.Option(
        Path("data/shared/typed_distances"),
        help="Root containing validated typed statistical-distance artifacts.",
    ),
    covariance_matrix: Path = typer.Option(
        Path("data/shared/covariance_matrix.parquet"),
        help="Return covariance/correlation matrix parquet.",
    ),
    output_summary: Path = typer.Option(
        Path("data/papers/paper1/wasserstein_comparator/summary.yaml"),
        help="Governed comparator summary YAML.",
    ),
    output_pairs: Path = typer.Option(
        Path("data/papers/paper1/wasserstein_comparator/pairs.parquet"),
        help="Matched dyad-level metric artifact.",
    ),
    bootstrap_iters: int = typer.Option(
        999, min=1, help="Paired firm/node bootstrap draws."
    ),
) -> None:
    """Run Paper 1's cross-encoder distributional comparison diagnostics."""
    own_float64()
    from pipeline.stages.papers.paper1.wasserstein_comparator import (
        WassersteinComparatorConfig,
        run_wasserstein_comparator,
    )

    result = run_wasserstein_comparator(
        WassersteinComparatorConfig(
            embeddings_root=embeddings_root,
            typed_distance_root=typed_distance_root,
            covariance_matrix=covariance_matrix,
            output_summary=output_summary,
            output_pairs=output_pairs,
            bootstrap_iters=bootstrap_iters,
        )
    )
    typer.echo(
        f"common m={result['common_m']}  priced firms={result['priced_universe']}  "
        f"dyads={result['n_priced_dyads']}"
    )
    typer.echo(f"Saved: {output_summary}")
    typer.echo(f"Saved: {output_pairs}")


@commands.command("compute-representation-sensitivity")
def cli_compute_representation_sensitivity(
    pairs_file: Path = typer.Option(
        Path("data/papers/paper1/wasserstein_comparator/pairs.parquet"),
        help="Governed matched distance/return dyad artifact.",
    ),
    node_counts_file: Path = typer.Option(
        Path("data/papers/paper1/dyadic_confound/node_bootstrap_counts.parquet"),
        help="Common node-bootstrap multiplicity schedule.",
    ),
    representation_dir: list[Path] = typer.Option(
        [],
        help=(
            "Non-native governed W2 representation artifact directory; repeatable. "
            "Provider and representation identities resolve against params.yaml."
        ),
    ),
    output_summary: Path = typer.Option(
        Path("data/papers/paper1/representation_sensitivity/summary.yaml"),
        help="W2 representation-sensitivity summary YAML.",
    ),
    output_draws: Path = typer.Option(
        Path("data/papers/paper1/representation_sensitivity/bootstrap_draws.parquet"),
        help="Standardized-effect bootstrap draws.",
    ),
    params_file: Path = typer.Option(
        Path("params.yaml"),
        help="Shared representation ablation registry.",
    ),
    vintage_dir: list[Path] = typer.Option(
        [],
        help=(
            "Encoder-vintage W2 artifact directory; repeatable. Each must carry "
            "a provider_id with a matching encoder_vintage_ablation_specs cell."
        ),
    ),
    vintage_equivalence_bound: float = typer.Option(
        0.012,
        min=0.0,
        help="Symmetric equivalence bound for paired standardized-effect contrasts.",
    ),
    vintage_confidence_level: float = typer.Option(
        0.95,
        min=0.0,
        max=1.0,
        help="Two-sided percentile confidence level for paired vintage contrasts.",
    ),
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
            vintage_dirs=tuple(vintage_dir),
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


@commands.command("compute-neighbour-outcome-split")
def cli_compute_neighbour_outcome_split(
    distance_artifact_dir: Path = typer.Option(
        Path(
            "data/shared/typed_distances_windowed/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_pit_2020"
        ),
        help="Shared 2018-2020 windowed rooted W2 distance artifact.",
    ),
    returns_dir: Path = typer.Option(
        Path("data/shared/returns"),
        help="Per-ticker simple-return directory.",
    ),
    universe_csv: Path = typer.Option(
        Path("data/universe.csv"),
        help="Ticker-sector metadata.",
    ),
    params_file: Path = typer.Option(
        Path("params.yaml"),
        help="Typed registry declaring the selection geometry and its window.",
    ),
    output_dir: Path = typer.Option(
        Path("data/papers/paper1/neighbour_outcome_split"),
        help="Governed output directory.",
    ),
) -> None:
    """Run Paper 1's preregistered original-space neighbour outcome split."""
    # The shared W2 artifact is materialized by JAX-backed transport code, so
    # this consumer owns the repository's float64 precision policy as well.
    own_float64()
    from pipeline.stages.papers.paper1.neighbour_outcome_split import (
        run_neighbour_outcome_split,
    )

    run_neighbour_outcome_split(
        distance_artifact_dir=distance_artifact_dir,
        returns_dir=returns_dir,
        universe_csv=universe_csv,
        params_file=params_file,
        output_dir=output_dir,
    )

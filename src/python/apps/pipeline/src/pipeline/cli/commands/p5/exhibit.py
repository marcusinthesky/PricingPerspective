"""P5 Exhibit CLI command adapters."""

from __future__ import annotations

from pathlib import Path

import typer

commands = typer.Typer()


@commands.command("render-paper5-representation-ablation-table")
def cli_render_paper5_representation_ablation_table(
    summary_path: Path = typer.Option(
        Path("data/papers/paper5/representation_ablation/summary.json")
    ),
    output_path: Path = typer.Option(
        Path(
            "src/latex/projects/05_spatial_pricing/src/generated/"
            "representation_ablation.tex"
        )
    ),
) -> None:
    """Render Paper 5's pooled representation/operator horse race."""
    from pipeline.figures.paper5.tables import (
        render_paper5_representation_ablation_table,
    )

    render_paper5_representation_ablation_table(summary_path, output_path)


@commands.command("render-paper5-universe-horizon-bridge-table")
def cli_render_paper5_universe_horizon_bridge_table(
    summary_path: Path = typer.Option(
        Path("data/papers/paper5/universe_horizon_bridge/summary.json")
    ),
    output_path: Path = typer.Option(
        Path(
            "src/latex/projects/05_spatial_pricing/src/generated/"
            "universe_horizon_bridge.tex"
        )
    ),
) -> None:
    """Render Paper 5's sequential universe-and-horizon bridge."""
    from pipeline.figures.paper5.universe_horizon_bridge import (
        render_paper5_universe_horizon_bridge_table,
    )

    render_paper5_universe_horizon_bridge_table(summary_path, output_path)


@commands.command("paper5-numbers")
def cli_paper5_numbers(
    results_dir: Path = typer.Option(
        Path("data/papers/paper5"),
        help="Paper 5 results directory (results.json + mc_results.json).",
    ),
    geometry_summary_path: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2/summary.json"
        ),
        help="Typed Wasserstein geometry summary with the balanced cloud size.",
    ),
    params_path: Path = typer.Option(
        Path("params.yaml"),
        help="Governed shared configuration, including the annual article floor.",
    ),
    universe_horizon_summary_path: Path = typer.Option(
        Path("data/papers/paper5/universe_horizon_bridge/summary.json"),
        help="Validated universe-by-horizon bridge summary.",
    ),
    out_path: Path = typer.Option(
        Path("src/latex/projects/05_spatial_pricing/src/generated/numbers.tex"),
        help="Generated number bindings.",
    ),
) -> None:
    """Generate number bindings for Paper 5."""
    from pipeline.figures.paper5.numbers import generate_numbers

    generate_numbers(
        results_dir,
        geometry_summary_path,
        params_path,
        universe_horizon_summary_path,
        out_path,
    )


@commands.command("render-paper5-illustrations")
def cli_render_paper5_illustrations(
    output_dir: Path = typer.Option(
        Path("src/latex/projects/05_spatial_pricing/src/images"),
        help="Output directory for Paper 5 conceptual illustration PGFs.",
    ),
) -> None:
    """Render Paper 5's numerically honest conceptual illustrations."""
    from pipeline.figures.paper5.illustrations import render_paper5_illustrations

    render_paper5_illustrations(output_dir)


@commands.command("render-paper5-barycentric-mixtures")
def cli_render_paper5_barycentric_mixtures(
    barycentre_dir: Path = typer.Option(
        Path(
            "data/shared/barycentres/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2_loo"
        ),
        help="Validated leave-one-out W2 target-projection artifact.",
    ),
    universe_csv: Path = typer.Option(
        Path("data/universe.csv"),
        help="Universe metadata used to validate the plotted ticker roster.",
    ),
    output_file: Path = typer.Option(
        Path(
            "src/latex/projects/05_spatial_pricing/src/images/"
            "w2_barycentric_mixture_weights.pgf"
        ),
        help="Output PGF map of the actual target-anchored W2 mixture weights.",
    ),
) -> None:
    """Render Paper 5's empirical target-anchored W2 barycentric mixtures."""
    from pipeline.figures.paper5.figures import (
        BarycentricMixturePaths,
        render_barycentric_mixture_map,
    )

    render_barycentric_mixture_map(
        BarycentricMixturePaths(
            barycentre_dir=barycentre_dir,
            universe_csv=universe_csv,
            output_file=output_file,
        )
    )


_P5_DESCRIPTIVES_CAPTION = (
    r"Per-sector composition of the news frame and its balanced-cloud sample: "
    r"firm count, mean article count per firm, mean daily "
    r"return volatility, and mean within- versus cross-sector rooted $W_2$ "
    r"distance under the frozen Qwen3-Embedding-8B geometry. Firm counts, "
    r"volatility, and distance summaries use the same governed firm set. "
    r"The Articles column reports the constant balanced-cloud "
    r"size after per-firm truncation, not raw frame coverage, and therefore "
    r"carries no information about a firm's true news volume. Volatility is "
    r"daily and unannualised. A dash marks a statistic "
    r"undefined for that sector, such as within-sector distance for a "
    r"singleton. Authors' calculations."
)


@commands.command("render-paper5-descriptives")
def cli_render_paper5_descriptives(
    summary_file: Path = typer.Option(
        Path("data/papers/paper5/sample_descriptives/summary.yaml"),
        help="Sample-descriptives summary computed under Paper 5's encoder.",
    ),
    output_file: Path = typer.Option(
        Path(
            "src/latex/projects/05_spatial_pricing/src/generated/"
            "sample_descriptives.tex"
        ),
        help="Generated per-sector descriptives table fragment.",
    ),
) -> None:
    """Render Paper 5's per-sector sample-descriptives table.

    Shares Paper 1's tabulation and renderer; only the encoder identity, the
    caption and the float label differ, so the two papers cannot collide on a
    label while remaining independently self-contained.
    """
    from pipeline.figures.paper1.tables import render_sample_descriptives_table

    render_sample_descriptives_table(
        summary_file=summary_file,
        output_file=output_file,
        caption=_P5_DESCRIPTIVES_CAPTION,
        label="tab:p5-sample-descriptives",
        generator="pipeline render-paper5-descriptives",
    )

"""Paper 5 energy-robustness command adapter."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import Field

from pipeline.cli.options import CliOptions, model_command, options_from_context
from pipeline.precision import own_float64

commands = typer.Typer()


class Paper5EnergyRobustnessOptions(CliOptions):
    """Inputs for Paper 5's energy-bracket H3 prerequisite."""

    returns_dir: Path = Field(default=Path("data/shared/returns"))
    eval_returns_dir: Path = Field(default=Path("data/shared/oos/returns"))
    distance_artifact: Path = Field(
        default=Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    )
    embeddings_dir: Path = Field(
        default=Path("data/shared/embeddings/qwen3-embedding-8b")
    )
    frontier_metrics: Path = Field(
        default=Path("data/shared/w2_exposure_frontier/frontier_metrics.json")
    )
    output_dir: Path = Field(default=Path("data/papers/paper5/energy_robustness"))
    calib_start: str = Field(default="2018-01-01")
    calib_end: str = Field(default="2022-12-31")
    eval_start: str = Field(default="2023-01-01")
    eval_end: str = Field(default="2026-07-16")
    n_boot_stat: int = Field(default=2000)
    n_boot_ridge: int = Field(default=2000)
    n_boot_sharpe: int = Field(default=2000)
    seed: int = Field(default=0)
    run_folds: bool = Field(default=False)
    provider_id: str = Field(default="qwen3-embedding-8b")
    representation_id: str = Field(default="qwen3-embedding-8b-unit")
    # Must track the p5_energy_robustness stage's --distance-id. The bracket
    # construction is named for the *energy* ambiguity set but is calibrated on
    # the typed rooted-W2 matrix; defaulting to energy_v here made a bare CLI
    # run compute a different object than the DVC graph, silently.
    distance_id: str = Field(default="wasserstein_w2")


paper5_energy_robustness_command = model_command(Paper5EnergyRobustnessOptions)


@commands.command(
    "paper5-energy-robustness",
    cls=paper5_energy_robustness_command,
)
def cli_paper5_energy_robustness(ctx: typer.Context) -> None:
    """Run Paper 5's energy-bracket and robust-portfolio H3 prerequisite."""
    own_float64()
    from pipeline.stages.papers.paper5.energy_robust import (
        Paper5EnergyRobustnessConfig,
        Paper5EnergyRobustnessPaths,
        run_paper5_energy_robustness,
    )

    options = options_from_context(ctx, Paper5EnergyRobustnessOptions)

    def run(
        output_dir: Path,
        *,
        calib_end: str,
        eval_start: str,
        eval_end: str,
        eval_returns_dir: Path = options.eval_returns_dir,
    ) -> None:
        run_paper5_energy_robustness(
            Paper5EnergyRobustnessPaths(
                returns_dir=options.returns_dir,
                eval_returns_dir=eval_returns_dir,
                energy_tests_path=options.distance_artifact,
                embeddings_dir=options.embeddings_dir,
                frontier_metrics_path=options.frontier_metrics,
                output_dir=output_dir,
            ),
            Paper5EnergyRobustnessConfig(
                calib_start=options.calib_start,
                calib_end=calib_end,
                eval_start=eval_start,
                eval_end=eval_end,
                n_boot_stat=options.n_boot_stat,
                n_boot_ridge=options.n_boot_ridge,
                n_boot_sharpe=options.n_boot_sharpe,
                seed=options.seed,
                provider_id=options.provider_id,
                representation_id=options.representation_id,
                distance_id=options.distance_id,
            ),
        )

    run(
        options.output_dir,
        calib_end=options.calib_end,
        eval_start=options.eval_start,
        eval_end=options.eval_end,
    )
    if options.run_folds:
        run(
            options.output_dir / "folds" / "eval2020",
            calib_end="2019-12-31",
            eval_start="2020-01-01",
            eval_end="2020-12-31",
            eval_returns_dir=options.returns_dir,
        )
        run(
            options.output_dir / "folds" / "eval2021",
            calib_end="2020-12-31",
            eval_start="2021-01-01",
            eval_end="2021-12-31",
            eval_returns_dir=options.returns_dir,
        )

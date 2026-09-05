"""Cross-paper calibration command adapters."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from pipeline.precision import own_float64

commands = typer.Typer()


@commands.command("w2-exposure-frontier")
def cli_w2_exposure_frontier(
    returns_dir: Path = typer.Option(Path("data/shared/returns")),
    distance_artifact_dir: Path = typer.Option(
        Path(
            "data/shared/typed_distances/qwen3-embedding-8b/"
            "qwen3-embedding-8b-unit/wasserstein_w2"
        )
    ),
    output_dir: Path = typer.Option(Path("data/shared/w2_exposure_frontier")),
    params_file: Path = typer.Option(Path("params.yaml")),
) -> None:
    """Calibrate the shared W2-to-return-exposure frontier."""
    own_float64()
    from pipeline.stages.substrate.w2_exposure_frontier import (
        H1ArtifactPaths,
        H1PilotConfig,
        run_h1_pilot,
    )

    with params_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)["w2_exposure_frontier"]
    run_h1_pilot(
        H1ArtifactPaths(
            returns_dir=returns_dir,
            distance_artifact_dir=distance_artifact_dir,
            output_dir=output_dir,
        ),
        H1PilotConfig(
            train_start=config["train_start"],
            train_end=config["train_end"],
            ks=tuple(config["ks"]),
            k_target_explained_variance=config["k_target_explained_variance"],
            tau_lo=config["tau_lo"],
            tau_hi=config["tau_hi"],
            n_boot=config["n_boot"],
            n_clusters=config["n_clusters"],
            n_mc=config["n_mc"],
            n_dirichlet=config["n_dirichlet"],
            cap_kill_threshold=config["cap_kill_threshold"],
            cap_marginal_threshold=config["cap_marginal_threshold"],
            wolak_alpha=config["wolak_alpha"],
            wolak_ell0=config["wolak_ell0"],
            seed=config["seed"],
        ),
    )

"""Render the Paper 5 joint two-field region from the saved stage results.

Reads only ``p5_two_field_qmle``'s pooled surface, bootstrap cloud and fitted
boundaries, so a figure edit never reruns the 2,000-draw bootstrap.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from pipeline.figures.paper5.figures import TwoFieldRegion, render_two_field_region

if TYPE_CHECKING:
    from pathlib import Path

POOLED_PERIOD = "pooled_2023_2026"


def render_paper5_two_field_region(results_dir: Path, out: Path) -> None:
    """Draw the pooled-period region exactly as the estimation stage saved it."""
    pooled = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))[
        "results"
    ][POOLED_PERIOD]
    surface = pd.read_parquet(results_dir / "surface.parquet")
    cloud = pd.read_parquet(results_dir / "bootstrap.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_two_field_region(
        TwoFieldRegion(
            surface_rho_b=surface["rho_b"].to_numpy(),
            surface_rho_n=surface["rho_n"].to_numpy(),
            surface_loglik=surface["loglik"].to_numpy(),
            bootstrap_rho_b=cloud["rho_b"].to_numpy(),
            bootstrap_rho_n=cloud["rho_n"].to_numpy(),
            joint_rho_b=pooled["joint"]["rho_b"],
            joint_rho_n=pooled["joint"]["rho_n"],
            field_b_boundary=pooled["boundaries"]["w_flat"]["rho_b"],
            field_n_boundary=pooled["boundaries"]["w_co_mentions"]["rho_n"],
        ),
        out,
    )

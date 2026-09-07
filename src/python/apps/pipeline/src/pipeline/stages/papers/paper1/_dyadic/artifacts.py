"""Writers for the governed Paper-1 dyadic artifact family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from pipeline.stages.papers.paper1._dyadic.contracts import DyadicArtifacts


def write_dyadic_artifacts(artifacts: DyadicArtifacts) -> None:
    """Write the summary and all auditable dyadic-regression artifacts."""
    artifacts.output_file.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_iters = artifacts.node_counts.shape[0]
    count_frame = pd.DataFrame(
        {
            "draw_id": np.repeat(
                np.arange(bootstrap_iters, dtype=np.int32), len(artifacts.tickers)
            ),
            "ticker": np.tile(
                np.asarray(artifacts.tickers, dtype=object), bootstrap_iters
            ),
            "multiplicity": artifacts.node_counts.reshape(-1).astype(np.int16),
        }
    )
    bootstrap_draws = pd.concat(artifacts.bootstrap_frames, ignore_index=True)
    bootstrap_draws["draw_id"] = bootstrap_draws["draw_id"].astype(np.int32)
    bootstrap_draws["spec_index"] = bootstrap_draws["spec_index"].astype(np.int8)
    for name in ("active_nodes", "effective_columns", "effective_rank"):
        bootstrap_draws[name] = bootstrap_draws[name].astype(np.int16)
    bootstrap_draws["active_dyads"] = bootstrap_draws["active_dyads"].astype(np.int32)
    frames = {
        "node_bootstrap_counts.parquet": count_frame,
        "bootstrap_draws.parquet": bootstrap_draws,
        "design_diagnostics.parquet": artifacts.design_diagnostics,
        "w2_primary_fit.parquet": artifacts.primary_fit,
        "leave_one_firm_out.parquet": artifacts.leave_one_out,
        "residual_scale_null.parquet": artifacts.residual_scale_null,
        "random_exposure_diagnostic.parquet": artifacts.random_exposure_diagnostic,
    }
    for filename, frame in frames.items():
        frame.to_parquet(artifacts.output_file.parent / filename, index=False)
    artifacts.output_file.write_text(
        yaml.safe_dump(artifacts.result, sort_keys=False), encoding="utf-8"
    )

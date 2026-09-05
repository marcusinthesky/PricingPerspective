"""Orchestration for the Paper 1 referee-response diagnostic figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from pipeline.figures.paper1._diagnostics.contracts import (
    _TRUNCATION_WIDTHS,
)
from pipeline.figures.paper1._diagnostics.postfit import _fig_dyadic_postfit
from pipeline.figures.paper1._diagnostics.preservation import (
    _distance_preservation,
    _fig_distance_preservation,
)

if TYPE_CHECKING:
    from pipeline.figures.paper1._diagnostics.contracts import (
        Paper1DiagnosticPaths,
    )


def render_paper1_diagnostics(paths: Paper1DiagnosticPaths) -> dict[str, object]:
    """Render the Paper 1 referee-response diagnostic figures.

    The first figure compares truncated W2 matrices with the full-width matrix.
    The second reports FWL, curvature, and leave-one-firm-out diagnostics for
    the primary W2 specification.

    Also writes ``summary_file`` with the per-width agreement statistics, so
    the manuscript consumes them as bound values rather than transcribing
    numbers off a figure.

    Args:
        paths: Governed diagnostic input and output artifact paths.

    """
    output_dir = paths.output_dir
    summary_file = paths.summary_file
    dyadic_summary_file = paths.dyadic_summary_file
    primary_fit_file = paths.w2_primary_fit_file
    lofo_file = paths.lofo_file

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: truncated-vs-full distance preservation ---
    frames, preservation = _distance_preservation(
        full_artifact_dir=paths.full_distance_artifact_dir,
        truncated_artifact_dirs={
            64: paths.distance_artifact_64_dir,
            256: paths.distance_artifact_256_dir,
        },
        provider_id=paths.provider_id,
        full_representation_id=paths.representation_id,
        truncated_representation_ids={
            64: paths.representation_64_id,
            256: paths.representation_256_id,
        },
        distance_id=paths.distance_id,
        widths=_TRUNCATION_WIDTHS,
    )
    _fig_distance_preservation(
        output_dir / "matryoshka_distance_preservation.pgf",
        frames,
        preservation,
    )

    summary: dict[str, object] = {
        "distance_identity": {
            "provider_id": paths.provider_id,
            "representation_id": paths.representation_id,
            "distance_id": paths.distance_id,
        },
        "truncation_widths": list(_TRUNCATION_WIDTHS),
        "distance_preservation": {str(k): v for k, v in preservation.items()},
        "note": (
            "distance_preservation compares the truncate-then-renormalize "
            "W2-distance matrix at each width against the full-width "
            "matrix over the same dyads; it measures whether the geometry "
            "survives truncation, which is what the manuscript's robustness "
            "claim is about"
        ),
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    with Path(dyadic_summary_file).open(encoding="utf-8") as fh:
        dyadic_summary: dict[str, object] = yaml.safe_load(fh)
    primary_fit = pd.read_parquet(primary_fit_file)
    lofo = pd.read_parquet(lofo_file)
    _fig_dyadic_postfit(
        output_dir / "dyadic_postfit_diagnostics.pgf",
        primary_fit,
        lofo,
        dyadic_summary,
    )
    return summary

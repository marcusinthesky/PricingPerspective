r"""Paper 1 full-universe sample-descriptives table emission.

Reads ``data/papers/paper1/sample_descriptives/summary.yaml`` (see
:mod:`pipeline.stages.papers.paper1.sample_descriptives`) and writes a
self-contained booktabs LaTeX fragment to
``src/latex/projects/01_continuous_bounds/src/generated/sample_descriptives.tex``
that the manuscript ``\\input``\\ s.

Unlike the retired energy-robustness backtest table,
this fragment does not reference the ``\\ppvalue``/value-model catalog: binding
these numbers into that catalog is a separate later step this module does not
perform (see the calling task's house rules). Every number is instead
formatted directly, to a fixed stated precision, from the summary YAML, and
undefined/sparse-sector statistics render as an explicit em dash
(``\\textemdash``) rather than a fabricated ``0.00``.

The sample-descriptives fragment emits the compact per-sector table
``tab:sample-descriptives-sectors``. The complete ticker roster is rendered by
the shared ``render-sample-universe-table`` command because it is a separate
canonical-geometry artifact and does not fit a page as a float.

This module owns the ``Paper1EmpiricalTablePaths`` contract, the render
sequence, and the renderer surface downstream code and tests address; the
renderer bodies live in the private
:mod:`pipeline.figures.paper1._tables` package. The private ``_render_*`` names
below are re-exported because ``tests/papers/test_paper1_tables_layout.py`` and
``tests/papers/paper1/test_generated_regressor_uncertainty.py`` import them
directly -- dropping one would break a test no other check covers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from pipeline.figures.paper1._tables.descriptives import (
    _render_sector_table,
    render_sample_descriptives_table,
)
from pipeline.figures.paper1._tables.dyadic import _render_dyadic_ladder
from pipeline.figures.paper1._tables.dyadic_diagnostics import (
    _render_dyadic_design_diagnostics,
)
from pipeline.figures.paper1._tables.hypothesis import (
    _render_disco_table,
)
from pipeline.figures.paper1._tables.random_exposure import (
    _render_random_exposure_diagnostic,
)
from pipeline.figures.paper1._tables.representation import (
    render_w2_primary_results,
    render_w2_representation_sensitivity,
    render_w2_sampling_diagnostics,
)

__all__ = [
    "Paper1EmpiricalTablePaths",
    "_render_disco_table",
    "_render_dyadic_design_diagnostics",
    "_render_dyadic_ladder",
    "_render_random_exposure_diagnostic",
    "_render_sector_table",
    "render_paper1_empirical_tables",
    "render_sample_descriptives_table",
    "render_w2_primary_results",
    "render_w2_representation_sensitivity",
    "render_w2_sampling_diagnostics",
]


@dataclass(frozen=True, slots=True)
class Paper1EmpiricalTablePaths:
    """Governed input and output artifacts for Paper 1 empirical tables."""

    dyadic_summary: Path
    disco_summary: Path
    representation_summary: Path
    oos_dyadic_summary: Path
    dyadic_design_diagnostics: Path
    output_dir: Path


def render_paper1_empirical_tables(paths: Paper1EmpiricalTablePaths) -> None:
    """Render governed empirical-result fragments instead of scalar sprawl."""
    dyadic_summary = paths.dyadic_summary
    disco_summary = paths.disco_summary
    representation_summary = paths.representation_summary
    oos_dyadic_summary = paths.oos_dyadic_summary
    dyadic_design_diagnostics = paths.dyadic_design_diagnostics
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(dyadic_summary).open(encoding="utf-8") as fh:
        dyadic = yaml.safe_load(fh)
    with Path(disco_summary).open(encoding="utf-8") as fh:
        disco = yaml.safe_load(fh)
    with Path(representation_summary).open(encoding="utf-8") as fh:
        representation = yaml.safe_load(fh)
    with Path(oos_dyadic_summary).open(encoding="utf-8") as fh:
        oos_dyadic = yaml.safe_load(fh)
    design_diagnostics = pd.read_parquet(dyadic_design_diagnostics)
    (output_dir / "w2_primary_results.tex").write_text(
        render_w2_primary_results(dyadic, oos_dyadic) + "\n"
    )
    (output_dir / "dyadic_ladder.tex").write_text(_render_dyadic_ladder(dyadic))
    (output_dir / "w2_representation_sensitivity.tex").write_text(
        render_w2_representation_sensitivity(representation) + "\n"
    )
    (output_dir / "w2_sampling_diagnostics.tex").write_text(
        render_w2_sampling_diagnostics(representation) + "\n"
    )
    (output_dir / "dyadic_design_diagnostics.tex").write_text(
        _render_dyadic_design_diagnostics(design_diagnostics, dyadic)
    )
    (output_dir / "random_exposure_diagnostic.tex").write_text(
        _render_random_exposure_diagnostic(dyadic)
    )
    (output_dir / "disco_decomposition.tex").write_text(_render_disco_table(disco))

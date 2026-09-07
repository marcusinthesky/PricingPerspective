"""Paper 1 diagnostics contracts: the artifact paths and the fixed constants.

These are the names ``pipeline.figures.paper1.diagnostics`` re-exports and that
the carved figure modules depend on; keeping them in the DAG root is what makes
the package acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Truncation widths asserted in the manuscript text (Matryoshka ablation).
_TRUNCATION_WIDTHS: tuple[int, ...] = (64, 256)
PRIMARY_SPECIFICATION_INDEX = 6
# Equal-count bins used purely as descriptive overlays; no fit uses them.
_FWL_BIN_COUNT = 20


@dataclass(frozen=True, slots=True)
class Paper1DiagnosticPaths:
    """Governed artifacts consumed and produced by the diagnostic renderer."""

    full_distance_artifact_dir: Path
    distance_artifact_64_dir: Path
    distance_artifact_256_dir: Path
    provider_id: str
    representation_id: str
    representation_64_id: str
    representation_256_id: str
    distance_id: str
    output_dir: Path
    summary_file: Path
    dyadic_summary_file: Path
    w2_primary_fit_file: Path
    lofo_file: Path

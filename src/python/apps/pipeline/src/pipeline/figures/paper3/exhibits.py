"""Paper 3 figure emission, isolated from generated-number code."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pipeline.figures.paper3.figures import (
    _fig_conceptual_pipeline,
    _fig_news_variance_benchmark,
)

if TYPE_CHECKING:
    from pathlib import Path

CERTIFICATE_EXHIBIT = "news_variance_benchmark.pgf"
ILLUSTRATION_EXHIBIT = "conceptual_pipeline.pgf"


def render_paper3_certificate(certificate_dir: Path, output_dir: Path) -> None:
    """Render the news-variance benchmark from the certificate results."""
    summary = json.loads((certificate_dir / "results.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        message = "Paper 3 certificate results must be a mapping"
        raise TypeError(message)
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig_news_variance_benchmark(output_dir / CERTIFICATE_EXHIBIT, summary)


def render_paper3_illustrations(output_dir: Path) -> None:
    """Render the data-free conceptual pipeline illustration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig_conceptual_pipeline(output_dir / ILLUSTRATION_EXHIBIT)

"""Paper 3 PDF exhibit emission, isolated from generated-number code."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.figures._common import _load_yaml
from pipeline.figures.paper3.figures import (
    _fig_calibration_coverage,
    _fig_certificate_surface,
    _fig_certificate_validation,
    _fig_conceptual_pipeline,
    _fig_news_variance_benchmark,
)
from pipeline.figures.paper3.manifests import write_paper3_manifest

EXHIBIT_NAMES = (
    "conceptual_pipeline.pgf",
    "certificate_surface.pgf",
    "certificate_validation.pgf",
    "calibration_coverage.pgf",
    "news_variance_benchmark.pgf",
)


def _coverage_target(params_path: Path) -> float:
    """Read the declared calibration coverage rule the boundary figure draws."""
    params = _load_yaml(params_path)
    paper3 = params.get("paper3")
    validation = (
        paper3.get("certificate_validation") if isinstance(paper3, dict) else None
    )
    target = validation.get("coverage_target") if isinstance(validation, dict) else None
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        message = "paper3.certificate_validation.coverage_target must be numeric"
        raise TypeError(message)
    if not 0.0 < float(target) <= 1.0:
        message = "paper3.certificate_validation.coverage_target must lie in (0, 1]"
        raise ValueError(message)
    return float(target)


def render_paper3_exhibits(
    paper3_dir: Path,
    pricefree_dir: Path,
    certificate_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    params_path: Path = Path("params.yaml"),
) -> None:
    """Render only Paper 3 PDFs and write their output manifest."""
    del paper3_dir, pricefree_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig_conceptual_pipeline(output_dir / "conceptual_pipeline.pgf")
    _fig_certificate_surface(
        output_dir / "certificate_surface.pgf",
        pd.read_parquet(certificate_dir / "surface.parquet"),
    )
    _fig_certificate_validation(
        output_dir / "certificate_validation.pgf",
        pd.read_parquet(validation_dir / "random_portfolios.parquet"),
    )
    _fig_calibration_coverage(
        output_dir / "calibration_coverage.pgf",
        _load_yaml(validation_dir / "summary.yaml"),
        _coverage_target(params_path),
    )
    _fig_news_variance_benchmark(
        output_dir / "news_variance_benchmark.pgf",
        _load_yaml(certificate_dir / "summary.yaml"),
    )
    outputs = [output_dir / name for name in EXHIBIT_NAMES]
    write_paper3_manifest(
        manifest_path,
        stage_key="p3_exhibits",
        phase="exhibit",
        outputs=outputs,
        upstreams=(
            certificate_dir / "surface.parquet",
            certificate_dir / "summary.yaml",
            validation_dir / "random_portfolios.parquet",
            validation_dir / "summary.yaml",
        ),
    )

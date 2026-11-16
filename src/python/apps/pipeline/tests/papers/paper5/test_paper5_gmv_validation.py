"""Tests for the appendix-only Paper 5 GMV exhibit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pipeline.figures.paper5.gmv_validation import (
    Paper5GMVValidationError,
    render_paper5_gmv_validation_table,
)

if TYPE_CHECKING:
    from pathlib import Path


_CANDIDATES = (
    "factor_diagonal_ledoit_wolf",
    "spatial_diagonal_w1_ledoit_wolf",
    "spatial_diagonal_w2_ledoit_wolf",
    "sample_gmv",
    "ledoit_wolf_gmv",
    "equal_weight",
)
_VARIANCES = (0.00005, 0.000051, 0.000052, 0.00006, 0.000055, 0.0001)


def _write_fixture(directory: Path, *, cap: float = 0.15) -> tuple[Path, Path]:
    """Write the minimum governed summary and result rows."""
    summary_path = directory / "results.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "paper3_spatial_portfolio_comparison.v4",
                "analysis_id": "paper3-spatial-portfolio-comparison-v4",
                "sample": {"risk_unit": "excess_return", "factor_model": "ff5_mom"},
                "protocol": {
                    "maximum_weight": cap,
                    "residual_shrinkage_target": "constant_correlation",
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "panel": "oos_primary",
            "candidate": candidate,
            "reference_design": "uniform_cap_15",
            "candidate_status": "computed",
            "portfolio_role": "total_risk_primary",
            "variance": variance,
            "annualized_volatility": 0.1,
            "effective_number_assets": 10.0,
            "maximum_weight": 0.15 if candidate != "equal_weight" else 0.01,
        }
        for candidate, variance in zip(_CANDIDATES, _VARIANCES, strict=True)
    ]
    results_path = directory / "results.parquet"
    pd.DataFrame(rows).to_parquet(results_path, index=False)
    return results_path, summary_path


def test_renders_governed_gmv_table(tmp_path: Path) -> None:
    """The generated table reports all declared total-risk candidates."""
    results_path, summary_path = _write_fixture(tmp_path)
    output_path = tmp_path / "gmv_validation.tex"

    render_paper5_gmv_validation_table(results_path, summary_path, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Target-anchored W2" in rendered
    assert "Realized variance" in rendered
    assert "15\\% single-name cap" in rendered
    assert "\\label{tab:p5-gmv-validation}" in rendered
    assert rendered.count(r"\\") >= len(_CANDIDATES) + 1


def test_rejects_non_governed_cap(tmp_path: Path) -> None:
    """The exhibit cannot silently relabel a different cap contract."""
    results_path, summary_path = _write_fixture(tmp_path, cap=1.0)

    with pytest.raises(Paper5GMVValidationError, match="15% cap"):
        render_paper5_gmv_validation_table(
            results_path, summary_path, tmp_path / "gmv_validation.tex"
        )

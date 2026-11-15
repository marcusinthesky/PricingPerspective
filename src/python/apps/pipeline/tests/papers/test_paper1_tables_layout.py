"""Publication contracts for Paper 1's compact generated tables."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
from pipeline.figures.paper1.dyadic_confound import (
    render_dyadic_design_diagnostics,
    render_dyadic_ladder,
    render_random_exposure_diagnostic,
)
from pipeline.figures.paper1.representation_sensitivity import (
    render_w2_representation_sensitivity,
)
from pipeline.figures.paper1.sample_descriptives import render_sample_descriptives_table

if TYPE_CHECKING:
    from pathlib import Path

_PRIMARY_SPEC_INDEX = 6
_PUBLICATION_NODES = 100
_PUBLICATION_DYADS = _PUBLICATION_NODES * (_PUBLICATION_NODES - 1) // 2
_PUBLICATION_DEGREE = _PUBLICATION_NODES - 1
_PUBLICATION_ENDPOINT_OVERLAP = 2 * (_PUBLICATION_NODES - 2) / (_PUBLICATION_DYADS - 1)
_PUBLICATION_FE_DESIGN_COLUMNS = _PUBLICATION_NODES + 1


def _dyadic_summary() -> dict[str, object]:
    ladder = [
        {
            "spec_index": spec_index,
            "label": f"specification {spec_index}",
            "coef_w2_distance": 1.0 + spec_index / 10,
            "effect_one_sd_w2_distance": 0.06,
            "ci_low_w2_distance": 0.8,
            "ci_high_w2_distance": 1.8,
            "pvalue_w2_distance": 0.001,
            "overall_r2": 0.2,
            "within_r2": 0.378 if spec_index == _PRIMARY_SPEC_INDEX else None,
            "firm_effects": spec_index == _PRIMARY_SPEC_INDEX,
            "regressors": (
                ["w2_distance"] if spec_index == _PRIMARY_SPEC_INDEX else list[str]()
            ),
            "bootstrap_requested_draws": 1999,
        }
        for spec_index in range(1, _PRIMARY_SPEC_INDEX + 1)
    ]
    return {
        "primary_spec_index": _PRIMARY_SPEC_INDEX,
        "se_type": "multinomial node bootstrap (dyad weight w_i*w_j)",
        "w2_scale": "rooted balanced quadratic Wasserstein distance",
        "model_ladder": ladder,
    }


def test_dyadic_ladder_is_contiguous_w2_and_specification_six_is_primary() -> None:
    """Keep exactly six W2 rungs and designate the firm-effect rung primary."""
    rendered = render_dyadic_ladder(_dyadic_summary())
    assert r"$\hat\beta_{W_2}$" in rendered
    assert "energy metric + symmetric firm effects" not in rendered
    assert r"\mathrm{SSE}_{\mathrm{FE}+W_2}" in rendered
    assert all(rendered.count(f"\n{index} &") == 1 for index in range(1, 7))
    assert rendered.index("Panel B") < rendered.index("\n6 &")


def test_design_table_uses_w2_contract() -> None:
    """Describe the W2 design without reviving the retired energy ladder."""
    frame = pd.DataFrame(
        [
            {
                "spec_index": index,
                "n_nodes": _PUBLICATION_NODES,
                "n_dyads": _PUBLICATION_DYADS,
                "degree_min": _PUBLICATION_DEGREE,
                "endpoint_overlap_share": _PUBLICATION_ENDPOINT_OVERLAP,
                "design_rank": _PUBLICATION_FE_DESIGN_COLUMNS,
                "design_columns": _PUBLICATION_FE_DESIGN_COLUMNS,
                "scaled_condition_number": 9.8,
                "focal_vif": 2.1,
                "focal_fwl_sd_ratio": 0.6,
                "dropped_columns": [],
            }
            for index in range(1, 7)
        ]
    )
    rendered = render_dyadic_design_diagnostics(frame, _dyadic_summary())
    assert "Primary design" in rendered
    assert r"\beta_{W_2}" in rendered
    assert "VIF$_{W_2}$" in rendered
    assert "Energy-comparator design" not in rendered


def test_random_exposure_caption_binds_the_governed_dyad_count() -> None:
    """Keep the caption synchronized when the priced universe changes."""
    rendered = render_random_exposure_diagnostic(
        {
            "n_dyads": _PUBLICATION_DYADS,
            "random_exposure_diagnostic": {
                "status": "computed",
                "scenarios": [
                    {
                        "L": 1.0,
                        "tau": 0.0,
                        "coverage_share": 0.5,
                        "coverage_ci_low": 0.4,
                        "coverage_ci_high": 0.6,
                        "unresolved_share": 0.4,
                        "unresolved_ci_low": 0.3,
                        "unresolved_ci_high": 0.5,
                        "violation_share": 0.1,
                        "violation_ci_low": 0.0,
                        "violation_ci_high": 0.2,
                    }
                ],
            },
        }
    )
    assert "shares of the 4,950 dyads" in rendered
    assert "1,326" not in rendered


def test_representation_table_uses_one_estimator_without_mantel() -> None:
    """Compare representations with one estimator and no Mantel surface."""
    cells = []
    for cell_id, dimension in enumerate((64, 256, 4096), start=1):
        cells.append(
            {
                "cell_id": cell_id,
                "group": "encoder_dimension",
                "model": "qwen3-embedding-8b",
                "articles": 128,
                "embedding_dimension": dimension,
                "distance": "wasserstein_w2",
                "status": "computed",
                "geometry_spearman_vs_canonical_w2": 0.9,
                "effect_one_sd": 0.07,
                "effect_one_sd_ci_low": 0.05,
                "effect_one_sd_ci_high": 0.09,
                "partial_r2": 0.37,
            }
        )
    rendered = render_w2_representation_sensitivity({"cells": cells})
    assert "encoder and embedding dimension" in rendered
    assert "Rank agreement" in rendered
    assert "symmetric additive firm effects" in rendered
    assert "Mantel" not in rendered


def test_sample_fragment_excludes_full_ticker_roster(tmp_path: Path) -> None:
    """Keep the appendix sample fragment compact rather than listing all firms."""
    summary = {
        "sectors": {
            "Technology": {
                "n_firms": 2,
                "mean_articles_per_firm": 128.0,
                "mean_daily_return_volatility": 0.02,
                "mean_within_sector_w2_distance": 1.0,
                "mean_cross_sector_w2_distance": 1.1,
            }
        },
        "overall": {
            "n_firms": 2,
            "mean_articles_per_firm": 128.0,
            "mean_daily_return_volatility": 0.02,
            "mean_within_sector_w2_distance": 1.0,
            "mean_cross_sector_w2_distance": 1.1,
        },
        "universe": [{"symbol": "AAA", "name": "A", "sector": "Technology"}],
    }
    summary_file = tmp_path / "results.json"
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    fragment = render_sample_descriptives_table(
        summary_file, tmp_path / "sample_descriptives.tex"
    )
    assert fragment.count(r"\begin{table}") == 1
    assert r"\begin{longtable}" not in fragment
    assert "AAA" not in fragment

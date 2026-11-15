"""Cross-paper merge guards for representation ablation cells."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
from pipeline.figures.paper3.tables import (
    render_paper3_representation_ablation_table,
)
from pipeline.figures.paper5.tables import (
    render_paper5_representation_ablation_table,
)
from pipeline.stages.papers.paper3.representation_ablation import (
    Paper3RepresentationMergePaths,
    merge_paper3_representation_ablation,
)
from pipeline.stages.papers.paper5.representation_ablation import (
    Paper5RepresentationMergePaths,
    merge_paper5_representation_ablation,
)
from pipeline.stages.shared.representation_ablation import (
    load_extended_representation_ablation_specs,
    load_representation_contrast_specs,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
_ANNUAL_YEARS = tuple(range(2017, 2025))
_EXTENDED_CELL_COUNT = 15
_EXTENDED_CONTRAST_COUNT = 35
_VINTAGE_CONTRAST_COUNT = 28
_EXPECTED_CELL_LABELS = {
    "qwen8b_full": "Qwen3-8B full",
    "qwen4b_full": "Qwen3-4B full",
    "qwen8b_1024": "Qwen3-8B@1024",
    "qwen4b_1024": "Qwen3-4B@1024",
    "bge_large_full": "BGE-large-v1.5 full",
    "qwen8b_256": "Qwen3-8B@256",
    "qwen8b_64": "Qwen3-8B@64",
    **{f"chronobert_{year}": f"ChronoBERT {year}" for year in _ANNUAL_YEARS},
}


def _assert_rendered_pairs(rendered: str, *, repetitions: int) -> None:
    """Check each governed pair individually, not merely the aggregate row count."""
    for spec in load_representation_contrast_specs(REPO_ROOT / "params.yaml"):
        candidate = _EXPECTED_CELL_LABELS[spec.candidate_cell_id]
        reference = _EXPECTED_CELL_LABELS[spec.reference_cell_id]
        assert rendered.count(f"{candidate} $-$ {reference} &") == repetitions


def test_paper3_merge_requires_canonical_benchmark_equality(tmp_path: Path) -> None:
    """The merged table cannot drift from Paper 3's headline benchmark."""
    specs = load_extended_representation_ablation_specs(REPO_ROOT / "params.yaml")
    cells_dir = tmp_path / "cells"
    benchmark = {"portfolios": {"news_only": {"variance": 1.0}}}
    universe = {"tickers": ["AAA", "BBB"]}
    for spec in specs:
        cell_dir = cells_dir / spec.cell_id
        cell_dir.mkdir(parents=True)
        (cell_dir / "results.json").write_text(
            json.dumps(
                {
                    "cell": asdict(spec),
                    "universe": universe,
                    "news_only_benchmark": benchmark,
                }
            ),
            encoding="utf-8",
        )
    certificate = tmp_path / "certificate.json"
    certificate.write_text(
        json.dumps({"news_only_benchmark": benchmark}), encoding="utf-8"
    )
    output = tmp_path / "p3-results.json"
    merged = merge_paper3_representation_ablation(
        Paper3RepresentationMergePaths(
            params_file=REPO_ROOT / "params.yaml",
            cells_dir=cells_dir,
            canonical_certificate_summary=certificate,
            output_summary=output,
        )
    )
    assert merged["roster"] == [spec.cell_id for spec in specs]
    assert len(specs) == _EXTENDED_CELL_COUNT
    contrasts = cast("list[dict[str, object]]", merged["representation_contrasts"])
    assert [row["contrast_id"] for row in contrasts] == [
        spec.contrast_id
        for spec in load_representation_contrast_specs(REPO_ROOT / "params.yaml")
    ]
    assert all(row["status"] == "unavailable" for row in contrasts)
    assert merged["schema_version"] == "paper3_representation_ablation_merge.v3"
    assert json.loads(output.read_text(encoding="utf-8"))["roster"] == merged["roster"]
    certificate.write_text(
        json.dumps(
            {"news_only_benchmark": {"portfolios": {"news_only": {"variance": 1.01}}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="headline benchmark"):
        merge_paper3_representation_ablation(
            Paper3RepresentationMergePaths(
                params_file=REPO_ROOT / "params.yaml",
                cells_dir=cells_dir,
                canonical_certificate_summary=certificate,
                output_summary=output,
            )
        )


def test_paper5_merge_pairs_focal_operators_and_fixed_comparators(
    tmp_path: Path,
) -> None:
    """Retain thirty focal fits on one schedule and add fixed matrices once."""
    specs = load_extended_representation_ablation_specs(REPO_ROOT / "params.yaml")
    cells_dir = tmp_path / "cells"
    inference = {"n_bootstrap": 2000, "seed": 0}
    focal = {
        "w_flat": {"start": "2023-01-03", "end": "2026-04-06", "trading_days": 815},
        "w_h": {"start": "2023-01-03", "end": "2026-04-06", "trading_days": 815},
    }
    for spec in specs:
        cell_dir = cells_dir / spec.cell_id
        cell_dir.mkdir(parents=True)
        payload = {
            "identity": {
                "provider_id": spec.provider_id,
                "representation_id": spec.representation_id,
                "barycentre_arm_id": "wasserstein_w2_loo",
                "geometry_id": "wasserstein_w2",
            },
            "inference": inference,
            "results": {"pooled_2023_2026": focal},
        }
        (cell_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "results": {
                    "pooled_2023_2026": {
                        **focal,
                        "w_co_mentions": {"rho_hat": 0.2},
                        "equal_support": {"rho_hat": 0.1},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "representation_ablation" / "results.json"
    merged = merge_paper5_representation_ablation(
        Paper5RepresentationMergePaths(
            params_file=REPO_ROOT / "params.yaml",
            cells_dir=cells_dir,
            canonical_results=canonical,
            output_summary=output,
        )
    )
    assert len(cast("list[object]", merged["cells"])) == len(specs)
    assert len(specs) == _EXTENDED_CELL_COUNT
    assert merged["inference"] == inference
    assert merged["period"] == focal["w_flat"]
    contrasts = cast(
        "dict[str, list[dict[str, object]]]", merged["representation_contrasts"]
    )
    expected_ids = [
        spec.contrast_id
        for spec in load_representation_contrast_specs(REPO_ROOT / "params.yaml")
    ]
    for metric in ("direct_rho", "gap"):
        assert [row["contrast_id"] for row in contrasts[metric]] == expected_ids
        assert all(row["status"] == "unavailable" for row in contrasts[metric])
    assert merged["schema_version"] == "paper5_representation_ablation_merge.v3"
    fixed = cast("dict[str, object]", merged["fixed_comparators"])
    assert set(fixed) == {"w_co_mentions", "equal_support"}
    canonical_payload = json.loads(canonical.read_text(encoding="utf-8"))
    canonical_payload["results"]["pooled_2023_2026"]["w_flat"] = {
        **focal["w_flat"],
        "trading_days": 814,
    }
    canonical.write_text(json.dumps(canonical_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce w_flat"):
        merge_paper5_representation_ablation(
            Paper5RepresentationMergePaths(
                params_file=REPO_ROOT / "params.yaml",
                cells_dir=cells_dir,
                canonical_results=canonical,
                output_summary=output,
            )
        )
    canonical_payload["results"]["pooled_2023_2026"]["w_flat"] = focal["w_flat"]
    canonical.write_text(json.dumps(canonical_payload), encoding="utf-8")
    changed_path = cells_dir / "chronobert_2024" / "results.json"
    changed = json.loads(changed_path.read_text(encoding="utf-8"))
    changed["inference"]["seed"] = 1
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="dates/bootstrap settings"):
        merge_paper5_representation_ablation(
            Paper5RepresentationMergePaths(
                params_file=REPO_ROOT / "params.yaml",
                cells_dir=cells_dir,
                canonical_results=canonical,
                output_summary=output,
            )
        )


def test_paper3_table_renders_all_annual_ladder_rows_and_pairs(tmp_path: Path) -> None:
    """Expose all governed annual cells and paired numerical fields."""
    designs = [
        {
            "label": label,
            "percentile": {"news_only": 0.01, "equal_weight": 0.25},
        }
        for label in (
            "uniform_cap_12p5",
            "uniform_cap_15",
            # Persisted compatibility key; the public design is fixed alpha=0.8.
            "effective_n_matched",
            "concentrated_cap_15",
        )
    ]
    benchmark = {
        "portfolios": {
            "news_only": {"gmv_relative_variance": 1.2},
            "equal_weight": {"gmv_relative_variance": 1.5},
        },
        "reference_population": {"designs": designs},
    }
    summary = {
        "universe": {"n_tickers": 100, "n_return_observations": 1207},
        "cells": [
            {
                "cell": asdict(spec),
                "news_only_benchmark": benchmark,
            }
            for spec in load_extended_representation_ablation_specs(
                REPO_ROOT / "params.yaml"
            )
        ],
        "contrast_analysis": {
            "contrasts": [
                {
                    "contrast_id": spec.contrast_id,
                    "candidate_cell": spec.candidate_cell_id,
                    "reference_cell": spec.reference_cell_id,
                    "family": spec.family,
                    "status": "computed",
                    "estimate": 1.0,
                    "ci_low": -3.0,
                    "ci_high": 2.0,
                    "pvalue": 0.4,
                    "holm_pvalue": 0.8,
                    "equivalence_bound": 2.5,
                    "equivalent": False,
                }
                for spec in load_representation_contrast_specs(
                    REPO_ROOT / "params.yaml"
                )
            ]
        },
    }
    summary_path = tmp_path / "results.json"
    output_path = tmp_path / "table.tex"
    main_output_path = tmp_path / "main-table.tex"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    render_paper3_representation_ablation_table(
        summary_path,
        output_path,
        main_output_path,
    )

    rendered = output_path.read_text(encoding="utf-8")
    main_rendered = main_output_path.read_text(encoding="utf-8")
    for year in _ANNUAL_YEARS:
        assert f"ChronoBERT {year} & 768" in rendered
        assert f"ChronoBERT {year} (primary)" not in rendered
        assert f"ChronoBERT {year} & 768 & 1.00 & 120.0" in main_rendered
    assert r"Uniform 12.5\%" in rendered
    assert r"Fixed $\alpha=0.8$" in rendered
    assert "Equal risk weights & --" in rendered
    assert "100-firm, 1,207-date standardized-return panel" in rendered
    assert "2.500 & No" in rendered
    assert rendered.count("2.500 & No") == _VINTAGE_CONTRAST_COUNT
    _assert_rendered_pairs(rendered, repetitions=1)
    assert "Qwen3-8B full (primary) & 4096 & 1.00 & 120.0" in main_rendered
    assert "Qwen3-8B@64 & 64 & 1.00 & 120.0" in main_rendered
    assert r"\label{tab:p3-representation-ablation}" in rendered
    assert r"\label{tab:p3-representation-main}" in main_rendered


def test_paper5_table_renders_two_operator_rows_per_annual_vintage(
    tmp_path: Path,
) -> None:
    """Apply both Paper 5 operator rows to each of the eight annual arms."""
    result = {
        "rho_hat": 0.7,
        "rho_ci_lower": 0.6,
        "rho_ci_upper": 0.8,
        "loglik": 100.0,
    }
    summary = {
        "cells": [
            {
                "cell": asdict(spec),
                "results": {
                    "results": {"pooled_2023_2026": {"w_flat": result, "w_h": result}}
                },
            }
            for spec in load_extended_representation_ablation_specs(
                REPO_ROOT / "params.yaml"
            )[7:]
        ],
        "fixed_comparators": {
            "w_co_mentions": result,
            "equal_support": result,
        },
    }
    summary_path = tmp_path / "results.json"
    output_path = tmp_path / "table.tex"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    render_paper5_representation_ablation_table(summary_path, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    for year in _ANNUAL_YEARS:
        assert rendered.count(f"ChronoBERT {year}") == 1
        assert f"ChronoBERT {year} & 768 &" in rendered
    assert r"Barycentric $W^\flat$" in rendered
    assert r"RBF--Wasserstein $W^h$" in rendered


def test_paper5_contrast_table_defines_estimands_and_joint_equivalence(
    tmp_path: Path,
) -> None:
    """Make both paired estimands and the joint vintage decision explicit."""
    result = {
        "rho_hat": 0.7,
        "rho_ci_lower": 0.6,
        "rho_ci_upper": 0.8,
        "loglik": 100.0,
    }
    cells = [
        {
            "cell": asdict(spec),
            "results": {
                "results": {"pooled_2023_2026": {"w_flat": result, "w_h": result}}
            },
        }
        for spec in load_extended_representation_ablation_specs(
            REPO_ROOT / "params.yaml"
        )
    ]
    contrast = {
        "family": "vintage",
        "status": "computed",
        "estimate": 0.0008956804790430928,
        "ci_low": 0.000009655759217644392,
        "ci_high": 0.0016425293886213274,
        "pvalue": 0.5,
        "holm_pvalue": 1.0,
        "equivalence_bound": 0.004,
        "equivalent": True,
    }
    contrasts = [
        {
            **contrast,
            "contrast_id": spec.contrast_id,
            "candidate_cell": spec.candidate_cell_id,
            "reference_cell": spec.reference_cell_id,
            "family": spec.family,
        }
        for spec in load_representation_contrast_specs(REPO_ROOT / "params.yaml")
    ]
    summary = {
        "period": {"trading_days": 885},
        "cells": cells,
        "fixed_comparators": {
            "w_co_mentions": result,
            "equal_support": result,
        },
        "contrast_analysis": {
            "direct_rho": contrasts,
            "gap": contrasts,
        },
    }
    summary_path = tmp_path / "results.json"
    output_path = tmp_path / "table.tex"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    render_paper5_representation_ablation_table(summary_path, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Panel A: direct barycentric-" in rendered
    assert "Panel B: barycentric-minus-RBF gap contrasts" in rendered
    assert rendered.count("[0.00001, 0.00164]") == 2 * _EXTENDED_CONTRAST_COUNT
    assert rendered.count("0.00400 & Yes") == 2 * _VINTAGE_CONTRAST_COUNT
    _assert_rendered_pairs(rendered, repetitions=2)
    assert "common 885-date schedule" in rendered

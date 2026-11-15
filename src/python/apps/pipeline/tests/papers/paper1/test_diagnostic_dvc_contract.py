"""DVC ownership for Paper 1 reviewer-legible diagnostic artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _stages() -> dict[str, Any]:
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dvc.yaml").exists()
    )
    return yaml.safe_load((root / "dvc.yaml").read_text())["stages"]


def _outputs(stage: dict[str, Any]) -> set[str]:
    return {
        str(next(iter(item))) if isinstance(item, dict) else str(item)
        for item in [*stage.get("outs", []), *stage.get("metrics", [])]
    }


def test_paper1_diagnostic_artifacts_are_declared_dvc_outputs() -> None:
    """Preserve diagnostic data independently of their publication projections."""
    stages = _stages()
    dyadic_outs = _outputs(stages["p1_dyadic_confound"])
    dyadic_command = str(stages["p1_dyadic_confound"]["cmd"])
    dyadic_deps = set(stages["p1_dyadic_confound"]["deps"])
    assert "--distance-artifact-dir" in dyadic_command
    assert "--energy-comparator-artifact-dir" in dyadic_command
    assert "--w2-distance-artifact-dir" not in dyadic_command
    lead = (
        "${shared}/typed_distances/${scientific_selectors.baseline_model}/"
        "${scientific_selectors.baseline_model}-unit"
    )
    for distance_id in ("wasserstein_w2", "energy_v"):
        assert f"{lead}/{distance_id}" in dyadic_command
        assert f"{lead}/{distance_id}" in dyadic_deps
    for name in (
        "node_bootstrap_counts.parquet",
        "bootstrap_draws.parquet",
        "design_diagnostics.parquet",
        "w2_primary_fit.parquet",
        "leave_one_firm_out.parquet",
        "residual_scale_null.parquet",
    ):
        assert f"${{p1}}/dyadic_confound/{name}" in dyadic_outs

    diagnostic = stages["p1_diagnostics"]
    diagnostic_outs = _outputs(diagnostic)
    assert "${p1}/diagnostics/results.json" in diagnostic_outs
    assert any(path.endswith(".parquet") for path in diagnostic_outs)
    assert not any(path.endswith(".pgf") for path in diagnostic_outs)
    assert "ablation-energy-dir" not in str(diagnostic["cmd"])
    assert not any("ablations/energy_tests" in path for path in diagnostic["deps"])
    assert "${scientific_selectors.baseline_model}" in str(diagnostic["cmd"])
    diagnostic_render = stages["p1_diagnostics_render"]
    assert diagnostic_outs <= set(diagnostic_render["deps"])
    diagnostic_figures = _outputs(diagnostic_render)
    assert any(
        path.endswith("dyadic_postfit_diagnostics.pgf") for path in diagnostic_figures
    )
    assert not any(
        path.endswith("resampling_calibration.pgf") for path in diagnostic_figures
    )
    assert any(
        path.endswith("dyadic_design_diagnostics.tex")
        for path in _outputs(stages["p1_dyadic_confound_render"])
    )
    assert _outputs(stages["p1_representation_sensitivity"]) == {
        "${p1}/representation_sensitivity/results.json",
        "${p1}/representation_sensitivity/bootstrap_draws.parquet",
    }
    pcoa = stages["p1_pcoa"]
    pcoa_outs = _outputs(pcoa)
    assert pcoa_outs == {
        "${p1}/pcoa/results.json",
        "${p1}/pcoa/coords_2d.parquet",
        "${p1}/pcoa/spectrum.parquet",
        "${p1}/pcoa/neighbours.parquet",
        "${p1}/pcoa/metric_mds_coords_2d.parquet",
        "${p1}/pcoa/metric_mds_neighbours.parquet",
    }
    assert "dimensionality_analysis" not in stages
    assert "p1_mds_render" not in stages
    figure_outs = _outputs(stages["p1_pcoa_render"])
    published_names = {
        "pcoa_scree.pgf",
        "pcoa_vintage_scree.pgf",
        "pcoa_spectrum.tex",
        "metric_mds_sector_plot.pgf",
        "metric_mds_shepard_diagnostic.pgf",
    }
    assert {Path(path).name for path in figure_outs} == published_names
    assert not any("w2_vs_return_and_mean_metric.pgf" in path for path in figure_outs)
    assert set(stages["p1_pcoa_render"]["deps"]) == pcoa_outs | {
        ".semflow/hashes/entrypoints/pipeline/render-paper1-pcoa.json"
    }
    illustration_outs = _outputs(stages["p1_illustrations"])
    assert any(
        path.endswith("continuous_bound_construction.pgf") for path in illustration_outs
    )
    assert not any(path.endswith("risk_factor_space.pdf") for path in illustration_outs)


def test_pcoa_dependencies_cover_exactly_the_published_comparison_ladders() -> None:
    """Every advertised model/width tuple invalidates its saved spectrum."""
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "params.yaml").exists()
    )
    params = yaml.safe_load((root / "params.yaml").read_text())
    cell_ids = set(params["representation_ablation_specs"]) | set(
        params["encoder_vintage_ablation_specs"]
    )
    pcoa = _stages()["p1_pcoa"]
    distances = {
        dep for dep in pcoa["deps"] if dep.startswith("${shared}/typed_distances/")
    }
    assert distances == {
        "${shared}/typed_distances/"
        f"${{ablation_cells.{cell_id}.provider_id}}/"
        f"${{ablation_cells.{cell_id}.representation_id}}/wasserstein_w2"
        for cell_id in cell_ids
    }
    assert set(pcoa["params"]) == {
        "representation_ablation_specs",
        "encoder_vintage_ablation_specs",
        "paper1.mds",
    }
    assert "${shared}/covariance_matrix.parquet" in pcoa["deps"]


def test_paper1_distance_stages_use_the_declared_primary_or_energy_exception() -> None:
    """Pin primary stages to W2 and retain energy for its explicit comparison."""
    stages = _stages()
    diagnostic = stages["p1_w2_permanova"]
    command = str(diagnostic["cmd"])
    assert "/wasserstein_w2" in command
    assert "${shared}/covariance_matrix.parquet" in diagnostic["deps"]
    assert _outputs(diagnostic) <= set(stages["p1_w2_permanova_render"]["deps"])
    assert "p1_disco" not in stages

    command = str(stages["sample_descriptives"]["cmd"])
    assert "--distance-id ${scientific_selectors.typed_distance_id}" in command
    assert "${scientific_selectors.baseline_model}" in command
    assert "p1_sample_descriptives" not in stages
    assert "p5_sample_descriptives" not in stages
    assert "${shared}/sample_descriptives/results.json" in set(
        stages["p1_sample_descriptives_render"]["deps"]
    )
    assert "--distance-id wasserstein_w2" in str(stages["p1_diagnostics"]["cmd"])
    assert "p1_generated_regressor_uncertainty" not in stages
    assert "p1_sector_alignment" not in stages
    assert "p1_encoder_neighbour_stability" not in stages

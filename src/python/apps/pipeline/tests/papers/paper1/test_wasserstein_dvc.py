"""DVC and number-surface contracts for Paper 1's W2 comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml
from pipeline.figures.paper1.numbers import Paper1NumberInputs, write_paper1_numbers

_PUBLICATION_FIRMS = 100
_ANNUAL_YEARS = tuple(range(2017, 2025))


def _repository_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dvc.yaml").exists()
    )


def _out_paths(stage: dict[str, object]) -> set[str]:
    outputs = cast("list[object]", stage.get("outs", [])) + cast(
        "list[object]", stage.get("metrics", [])
    )
    return {
        str(next(iter(output))) if isinstance(output, dict) else str(output)
        for output in outputs
    }


def _stages() -> dict[str, dict[str, object]]:
    document = yaml.safe_load(
        (_repository_root() / "dvc.yaml").read_text(encoding="utf-8")
    )
    return cast("dict[str, dict[str, object]]", document["stages"])


def test_distributional_comparator_emits_only_governed_pairs() -> None:
    """The pair producer does not run unused summary inference."""
    stage = _stages()["p1_wasserstein_comparator"]
    assert _out_paths(stage) == {
        "${p1}/wasserstein_comparator/pairs.parquet",
    }
    command = str(stage["cmd"])
    assert "--typed-distance-root ${shared}/typed_distances" in command
    assert "linear_mmd" not in command


def test_representation_stage_owns_encoder_dimension_and_distance_grid() -> None:
    """Assign the W2 encoder, width, and distance grid to one stage."""
    stage = _stages()["p1_representation_sensitivity"]
    deps = set(cast("list[str]", stage["deps"]))
    assert "${p1}/wasserstein_comparator/pairs.parquet" in deps
    assert any("-64-unit/wasserstein_w2" in dep for dep in deps)
    assert any("-256-unit/wasserstein_w2" in dep for dep in deps)
    assert {
        "${shared}/typed_distances/"
        f"chronobert-{year}/chronobert-{year}-unit/wasserstein_w2"
        for year in _ANNUAL_YEARS
    } <= deps
    command = str(stage["cmd"])
    assert command.count("--vintage-dir") == len(_ANNUAL_YEARS)
    assert "--vintage-equivalence-bound" in command
    assert "encoder_vintage_ablation_specs" in cast("list[str]", stage["params"])
    assert "encoder_vintage_analysis" in cast("list[str]", stage["params"])
    assert _out_paths(stage) == {
        "${p1}/representation_sensitivity/results.json",
        "${p1}/representation_sensitivity/bootstrap_draws.parquet",
    }


def test_encoder_vintage_w2_is_direct_and_does_not_require_geometry_components() -> (
    None
):
    """Give each vintage one direct W2 leaf without widening typed rosters."""
    stage = _stages()["typed_distances"]
    assert stage["foreach"] == "${typed_distance_specs}"
    body = cast("dict[str, object]", stage["do"])
    command = str(body["cmd"])
    dependencies = set(cast("list[str]", body["deps"]))
    params = yaml.safe_load(
        (_repository_root() / "params.yaml").read_text(encoding="utf-8")
    )
    roster = params["typed_distance_specs"]
    for vintage in _ANNUAL_YEARS:
        cell = roster[f"chronobert_{vintage}__unit__wasserstein_w2"]
        assert cell["distance_id"] == "wasserstein_w2"
        assert cell["representation_id"] == f"chronobert-{vintage}-unit"
    assert "--distance-id ${item.distance_id}" in command
    assert "--representation-id ${item.representation_id}" in command
    assert not any("typed_geometry_components" in path for path in dependencies)


def test_table_stage_registers_only_surviving_paper1_fragments() -> None:
    """Publish only the theory-method paper's surviving compact tables."""
    stages = _stages()
    representation = stages["p1_representation_sensitivity_render"]
    assert "${p1}/representation_sensitivity/results.json" in set(
        cast("list[str]", representation["deps"])
    )
    outputs = _out_paths(representation) | _out_paths(
        stages["p1_dyadic_confound_render"]
    )
    assert any(path.endswith("w2_representation_sensitivity.tex") for path in outputs)
    assert any(path.endswith("w2_sampling_diagnostics.tex") for path in outputs)
    assert any(path.endswith("dyadic_ladder.tex") for path in outputs)
    assert not any("wasserstein_comparator.tex" in path for path in outputs)
    assert "p1_tables" not in stages


def test_number_writer_emits_panel_metadata_without_retired_comparator_keys(
    tmp_path: Path,
) -> None:
    """Emit governed panel dates without retired comparator bindings."""
    numbers = tmp_path / "numbers.tex"
    write_paper1_numbers(
        numbers,
        Paper1NumberInputs(
            return_panel={
                "return_panel_rows": "1207",
                "return_panel_columns": str(_PUBLICATION_FIRMS),
                "return_panel_start_date": "2018-03-19",
                "return_panel_end_date": "2022-12-30",
            },
            oos_return_panel={
                "return_panel_rows": "814",
                "return_panel_columns": str(_PUBLICATION_FIRMS),
                "return_panel_start_date": "2023-01-04",
                "return_panel_end_date": "2026-04-06",
            },
        ),
    )
    emitted = numbers.read_text(encoding="utf-8")
    assert r"\ppDeclareValue{return-panel-n-observations}{1207}" in emitted
    assert r"\ppDeclareValue{oos-return-panel-n-observations}{814}" in emitted
    assert r"\ppDeclareValue{oos-return-panel-date-start}{2023-01-04}" in emitted
    assert r"\ppDeclareValue{oos-return-panel-date-end}{2026-04-06}" in emitted
    assert "mantel-" not in emitted
    assert "baseline-" not in emitted
    assert "energy-beyond-mean" not in emitted

"""Regression guards for typed-distance DVC fan-out."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import cast

import pytest
import yaml
from pipeline.io.typed_analysis import load_typed_analysis

REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER3_W2_ROOT = (
    "${shared}/typed_distances/${scientific_selectors.typed_distance_provider}/"
    "${scientific_selectors.typed_distance_representation}/"
    "${paper3.primary_distance_id}"
)
PAPER5_TYPED_DISTANCE_ROOT = (
    "${shared}/typed_distances/${paper5.provider_id}/"
    "${paper5.representation_id}/${paper5.distance_id}"
)


def _stages() -> dict[str, dict[str, object]]:
    """Load authored DVC stages without resolving matrix families."""
    document = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    return cast("dict[str, dict[str, object]]", document["stages"])


def _mapping(value: object) -> dict[str, object]:
    """Narrow one parsed YAML mapping for Pyrefly and readable assertions."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _sequence(value: object) -> list[object]:
    """Narrow one parsed YAML sequence for Pyrefly and readable assertions."""
    assert isinstance(value, list)
    return cast("list[object]", value)


def _command(stage: dict[str, object]) -> list[str]:
    """Normalize quoting and option assignment without expanding DVC variables."""
    return [
        part
        for token in shlex.split(str(stage["cmd"]))
        for part in (token.split("=", 1) if token.startswith("--") else [token])
    ]


def _option(command: list[str], name: str) -> str:
    """Read one unambiguous option value from an authored command."""
    assert command.count(name) == 1, f"expected exactly one {name}: {command}"
    return command[command.index(name) + 1]


@pytest.mark.parametrize(
    "command",
    [
        "pipeline typed-distance --provider-id ${item.provider_id} --pooled-only",
        'pipeline typed-distance --provider-id="${item.provider_id}" --pooled-only',
        "pipeline typed-distance --pooled-only\n --provider-id '${item.provider_id}'",
    ],
)
def test_command_checks_ignore_equivalent_shell_spelling(command: str) -> None:
    """Quotes, whitespace, assignments and option order preserve scientific identity."""
    tokens = _command({"cmd": command})
    assert _option(tokens, "--provider-id") == "${item.provider_id}"
    assert "--pooled-only" in tokens


@pytest.mark.parametrize(
    "command",
    [
        "pipeline typed-distance --representation-id ${item.provider_id}",
        "pipeline typed-distance --provider-id other --provider-id ${item.provider_id}",
    ],
)
def test_command_checks_reject_missing_or_ambiguous_identity(command: str) -> None:
    """A different option or duplicate assignment cannot satisfy the identity guard."""
    with pytest.raises(AssertionError, match="--provider-id"):
        _option(_command({"cmd": command}), "--provider-id")


def test_typed_distance_outputs_feed_matching_paper_consumers() -> None:
    """Paper 3 and Paper 5 must consume the shared typed matrix directly."""
    stages = _stages()
    typed = stages["typed_distances"]
    typed_do = _mapping(typed.get("do", typed))
    typed_outputs = typed_do["outs"]
    artifact = (
        "${shared}/typed_distances/${item.provider_id}/"
        "${item.representation_id}/${item.distance_id}"
    )
    assert typed_outputs == [artifact]
    command = _command(typed_do)
    assert _option(command, "--provider-id") == "${item.provider_id}"
    assert _option(command, "--representation-id") == "${item.representation_id}"
    assert _option(command, "--distance-id") == "${item.distance_id}"
    assert _option(command, "--output-dir") == artifact
    assert "--roster-key" not in command

    paper3 = stages["p3_certificate"]
    assert PAPER3_W2_ROOT in _sequence(paper3["deps"])
    assert PAPER3_W2_ROOT in _command(paper3)
    paper5 = stages["p5_economic_value"]
    assert PAPER5_TYPED_DISTANCE_ROOT in _sequence(paper5["deps"])
    assert PAPER5_TYPED_DISTANCE_ROOT in _command(paper5)


def test_typed_distance_consumer_does_not_retain_legacy_baseline_path() -> None:
    """The migrated consumers cannot silently fall back to energy_tests."""
    stages = _stages()
    legacy_path = (
        "${shared}/energy_tests/${scientific_selectors.baseline_model}/"
        "energy_tests.parquet"
    )
    for consumer in ("p3_certificate", "p5_economic_value"):
        stage = stages[consumer]
        assert legacy_path not in _sequence(stage["deps"])
        assert all(legacy_path not in token for token in _command(stage))


def test_legacy_energy_producers_are_removed() -> None:
    """Only the typed family may produce canonical shared distances."""
    stages = _stages()
    assert "energy_tests" not in stages
    assert "energy_tests_ablation_models" not in stages


def test_component_derived_energy_does_not_depend_on_raw_embeddings() -> None:
    """A cached raw component is sufficient to rebuild the final energy matrix."""
    stage = _stages()["typed_distances_energy"]
    command = _command(stage)
    deps = _sequence(stage["deps"])
    assert _option(command, "--geometry-component-dir") in deps
    assert "--embeddings-dir" not in command
    assert not any("/embeddings/" in str(path) for path in deps)


def test_substrate_consumers_have_zero_legacy_energy_edges() -> None:
    """Cut-over substrate stages only depend on typed distance directories."""
    stages = _stages()
    for name in ("p1_pcoa",):
        serialized = yaml.safe_dump(stages[name])
        assert "/energy_tests" not in serialized
        assert "/typed_distances" in serialized


def test_unscheduled_measure_barycentres_keep_their_scientific_registry_api() -> None:
    """Retiring unconsumed artifacts preserves the three source-measure arms."""
    stages = _stages()
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    registry = load_typed_analysis(REPO_ROOT / "params.yaml")
    assert "typed_measure_barycentres" not in stages
    scheduled = {
        _mapping(cell)["arm_id"]
        for cell in _mapping(params["barycentre_specs"]).values()
    }
    for arm_id, geometry_id, solver in (
        ("wasserstein_w1_measure", "wasserstein_w1", "wasserstein_ot"),
        ("wasserstein_w2_measure", "wasserstein_w2", "wasserstein_ot"),
        (
            "wasserstein_w2_free_support",
            "wasserstein_w2",
            "wasserstein_free_support",
        ),
    ):
        arm, geometry = registry.barycentre_for(arm_id)
        assert arm.problem == "source_barycentre"
        assert geometry.distance_id == geometry_id
        assert arm.solver == solver
        assert arm_id not in scheduled


def test_typed_wasserstein_projection_roster_feeds_paper5() -> None:
    """Paper 5 consumes canonical W1 and roster-driven W2 projection leaves."""
    stages = _stages()
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    family = _mapping(stages["barycentres"])
    assert family["foreach"] == "${barycentre_specs}"
    body = _mapping(family["do"])
    assert _option(_command(body), "--provider-id") == "${item.provider_id}"
    assert _option(_command(body), "--representation-id") == "${item.representation_id}"
    assert _option(_command(body), "--arm-id") == "${item.arm_id}"
    assert "--geometry-component-dir" not in _command(body)
    roster = [_mapping(cell) for cell in _mapping(params["barycentre_specs"]).values()]
    ablations = [
        _mapping(cell)
        for key in ("ablation_cells", "encoder_vintage_ablation_specs")
        for cell in _mapping(params[key]).values()
    ]
    for arm_id in ("wasserstein_w1_loo", "wasserstein_w2_loo"):
        actual = {
            (cell["provider_id"], cell["representation_id"])
            for cell in roster
            if cell["arm_id"] == arm_id
        }
        expected = {
            (cell["provider_id"], cell["representation_id"])
            for cell in ablations
            if arm_id == "wasserstein_w2_loo"
            or cell["provider_id"] == "qwen3-embedding-8b"
        }
        assert actual == expected
    paper5 = _mapping(stages["p5_economic_value"])
    paper5_cmd = _command(paper5)
    paper5_deps = _sequence(paper5["deps"])
    for arm_param in ("wasserstein_w1_arm_id", "barycentre_arm_id"):
        path = (
            "${shared}/barycentres/${paper5.provider_id}/"
            "${paper5.representation_id}/"
            f"${{paper5.{arm_param}}}"
        )
        assert path in paper5_cmd
        assert path in paper5_deps


def test_representation_ablation_roster_is_focused_and_typed() -> None:
    """Publication ablations have seven ordered cells and one canonical identity."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    roster = _mapping(params["representation_ablation_specs"])
    assert list(roster) == [
        "qwen8b_full",
        "qwen4b_full",
        "qwen8b_1024",
        "qwen4b_1024",
        "bge_large_full",
        "qwen8b_256",
        "qwen8b_64",
    ]
    assert sum(bool(_mapping(cell)["canonical"]) for cell in roster.values()) == 1
    typed = _mapping(_mapping(params["typed_analysis"])["representations"])
    for order, raw_cell in enumerate(roster.values(), start=1):
        cell = _mapping(raw_cell)
        representation = _mapping(typed[str(cell["representation_key"])])
        assert representation["representation_id"] == cell["representation_id"]
        assert representation["provider_id"] == cell["provider_id"]
        assert cell["display_order"] == order


def test_representation_ablation_consumers_share_one_roster() -> None:
    """Paper 3 and Paper 5 share the merged base/vintage execution roster."""
    stages = _stages()
    paper1 = _mapping(stages["p1_representation_sensitivity"])
    assert "representation_ablation_specs" in _sequence(paper1["params"])
    assert _option(_command(paper1), "--params-file") == "params.yaml"

    for name in (
        "p3_representation_ablation_cells",
        "p5_representation_ablation_cells",
    ):
        family = _mapping(stages[name])
        assert family["foreach"] == "${ablation_cells}"
        body = _mapping(family["do"])
        assert "ablation_cells.${key}" in _sequence(body["params"])
        output_dir = _option(_command(body), "--output-dir")
        assert all(
            str(path).startswith(output_dir + "/") for path in _sequence(body["outs"])
        )

    paper5_command = _command(
        _mapping(stages["p5_representation_ablation_cells"]["do"])
    )
    assert "--pooled-only" in paper5_command
    assert "--focal-matrices-only" in paper5_command


@pytest.mark.parametrize("paper", [3, 5])
def test_ablation_execution_covers_the_aggregate_representation_rosters(
    paper: int,
) -> None:
    """Every aggregate-required cell has a producer with matching input identity."""
    stages = _stages()
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    base = _mapping(params["representation_ablation_specs"])
    vintages = _mapping(params["encoder_vintage_ablation_specs"])
    required = base | vintages
    scheduled = _mapping(params["ablation_cells"])
    assert scheduled.keys() == required.keys()
    for cell_id, cell in required.items():
        assert scheduled[cell_id] == cell

    family = _mapping(stages[f"p{paper}_representation_ablation_cells"])
    assert family["foreach"] == "${ablation_cells}"
    producer = _mapping(family["do"])
    aggregate = _mapping(stages[f"p{paper}_representation_ablation"])
    assert {
        "representation_ablation_specs",
        "encoder_vintage_ablation_specs",
        "representation_contrast_specs",
    } <= set(_sequence(aggregate["params"]))
    cells_dir = _option(_command(aggregate), "--cells-dir")
    assert cells_dir in _sequence(aggregate["deps"])
    assert _option(_command(producer), "--output-dir") == cells_dir + "/${key}"


def test_shared_co_mentions_graph_feeds_paper5() -> None:
    """Paper 5 consumes the governed persistent co-mention adjacency."""
    paper5 = _mapping(_stages()["p5_economic_value"])
    path = "${shared}/co_mentions/adjacency.parquet"
    assert _option(_command(paper5), "--co-mentions-adjacency") == path
    assert path in _sequence(paper5["deps"])


def test_irregular_rosters_use_semantic_mapping_keys() -> None:
    """Reordering authored cells cannot change expanded DVC stage identities."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))

    for roster_name in (
        "typed_geometry_component_specs",
        "typed_distance_specs",
        "typed_distance_windowed_specs",
        "barycentre_specs",
        "ablation_cells",
    ):
        roster = params[roster_name]
        assert isinstance(roster, dict)
        assert roster
        assert all(not str(key).isdigit() for key in roster)


def test_scientific_selectors_are_params_not_untracked_dvc_vars() -> None:
    """Canonical scientific choices remain visible to DVC experiment diffs."""
    document = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    serialized_vars = yaml.safe_dump(document["vars"])

    assert "scientific_selectors" in params
    assert "baseline_model:" not in serialized_vars
    assert "typed_distance_provider:" not in serialized_vars
    assert "typed_barycentre_provider" not in params
    assert "typed_barycentre_representation" not in params


def test_typed_cells_declare_dynamic_cell_scoped_parameter_paths() -> None:
    """Each typed family tracks its own roster and referenced registry leaves."""
    stages = _stages()
    geometry_do = _mapping(stages["typed_geometry_components"]["do"])
    windowed_do = _mapping(stages["typed_distances_windowed"]["do"])
    distance_do = _mapping(stages["typed_distances"]["do"])
    barycentre_do = _mapping(stages["barycentres"]["do"])

    assert "typed_geometry_component_specs.${key}" in _sequence(geometry_do["params"])
    assert "typed_analysis" not in _sequence(geometry_do["params"])
    assert "typed_analysis.windows.${item.window_key}" in _sequence(
        windowed_do["params"]
    )
    assert "typed_distance_specs.${key}" in _sequence(distance_do["params"])
    assert "typed_analysis.providers.${item.provider_key}" in _sequence(
        distance_do["params"]
    )
    assert "barycentre_specs.${key}" in _sequence(barycentre_do["params"])
    assert "typed_analysis.barycentres.${item.arm_id}" in _sequence(
        barycentre_do["params"]
    )
    for body in (geometry_do, windowed_do, distance_do, barycentre_do):
        embeddings_dir = _option(_command(body), "--embeddings-dir")
        assert embeddings_dir == "${shared}/embeddings/${item.provider_id}"
        assert embeddings_dir in _sequence(body["deps"])


def test_params_readers_declare_scoped_parameters_not_whole_file_dependencies() -> None:
    """Runtime params readers must expose their exact DVC invalidation closure."""
    stages = _stages()
    expected = {
        "p1_neighbour_outcome_split": {
            "typed_analysis.distances.euclidean_chord",
            "typed_analysis.distances.wasserstein_w2_pit_2020",
            "typed_analysis.windows.pit-2020-12-31",
        },
        "p1_numbers": {
            "annual_raw_article_floor",
            "paper1_text_controls.max_features",
            "symbols",
            "matryoshka_range",
            "embedding.models.${scientific_selectors.baseline_model_key}.dims",
            "start_date",
            "end_date",
            "n_samples",
        },
    }
    for stage_name, exact_params in expected.items():
        stage = stages[stage_name]
        assert "params.yaml" not in _sequence(stage["deps"])
        assert set(_sequence(stage["params"])) == exact_params

    pit = stages["p3_pit_w2"]
    assert "params.yaml" not in _sequence(pit["deps"])
    assert (
        _option(_command(pit), "--distance-root")
        == "${shared}/typed_distances_windowed"
    )
    assert all(
        "${shared}/typed_distances_windowed/" in str(dependency)
        for dependency in _sequence(pit["deps"])
        if "wasserstein_w2_pit_" in str(dependency)
    )
    assert set(_sequence(pit["params"])) == {
        "paper3.pit_vintage_grid",
        "typed_analysis.providers.qwen3-embedding-8b",
        "typed_analysis.representations.qwen3-embedding-8b-unit",
        "typed_analysis.distances.euclidean_chord",
        *(
            f"typed_analysis.distances.wasserstein_w2_pit_{year}"
            for year in range(2018, 2023)
        ),
        *(f"typed_analysis.windows.pit-{year}-12-31" for year in range(2018, 2023)),
    }

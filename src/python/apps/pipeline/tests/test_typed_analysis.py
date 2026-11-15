"""Runtime validation tests for the typed-analysis registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pipeline._kernels.typed_barycentre import solve_energy_target_projections
from pipeline._kernels.typed_distances import compute_statistical_distance
from pipeline.io.barycentre_artifacts import (
    BarycentreArtifactMetadata,
    read_target_projection_artifact,
    write_target_projection_artifact,
)
from pipeline.io.typed_analysis import (
    BarycentreArm,
    DvcScientificConfig,
    EncoderVintageProvider,
    LocalParquetProvider,
    PointDistanceSpec,
    SimplexFeasibleSet,
    StatisticalDistanceSpec,
    TypedAnalysisConfig,
    canonical_typed_analysis,
    iteration_budget,
    load_dvc_scientific_config,
    load_typed_analysis,
    typed_analysis_parameter_identity,
)
from pipeline.io.typed_providers import (
    ProviderInputError,
    load_dated_provider_observations,
    load_provider_clouds,
)
from pipeline.stages.papers.paper3.pit_distance import (
    _parameter_identity as pit_parameter_identity,
)
from pipeline.stages.papers.paper3.pit_distance import _vintage_roster
from pydantic import ValidationError

EXPECTED_QWEN_DIMENSIONS = 2560
EXPECTED_ENERGY_MAXITER = 10_000
EXPECTED_CHRONOBERT_VINTAGES = {str(year) for year in range(2017, 2025)}
ANNUAL_PROVIDER_IDS = tuple(f"chronobert-{year}" for year in range(2017, 2025))
EXPECTED_CHRONOBERT_DIMENSIONS = 768
EXPECTED_MARKET_COUNT = 100
EXPECTED_ARRAY_RANK = 2
MATRYOSHKA_DIMENSION = 2
FREE_SUPPORT_MAXITER = 128
EXPECTED_SMOKE_CLOUD_COUNT = 3
# The scheduled roster contains only artifacts consumed by the manuscripts.
EXPECTED_FULL_COMPONENT_CELLS = 1
EXPECTED_WINDOWED_DISTANCE_CELLS = 5
# tests/ -> pipeline -> apps -> python -> src -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[5]


def _registry() -> dict[str, object]:
    return {
        "providers": {
            "local": {
                "provider_id": "local",
                "kind": "local_parquet",
                "model_id": "local-model",
                "dimensions": 2,
                "normalization": "raw",
            }
        },
        "representations": {
            "raw": {
                "representation_id": "raw",
                "provider_id": "local",
                "transform": "identity",
                "normalization": "raw",
            }
        },
        "distances": {
            "point": {
                "distance_id": "point",
                "kind": "point",
                "metric": "euclidean",
                "domain": "euclidean",
            },
            "energy": {
                "distance_id": "energy",
                "kind": "statistical",
                "family": "energy",
                "ground_distance": "point",
                "exponent": 1.0,
                "estimator": "v_statistic",
                "normalization": "unhalved",
                "sample_design": "full_sample",
            },
        },
        "feasible_sets": {
            "simplex": {
                "feasible_set_id": "simplex",
                "kind": "simplex_nonnegative",
            }
        },
        "barycentres": {
            "energy_arm": {
                "arm_id": "energy_arm",
                "problem": "target_projection",
                "geometry": "energy",
                "feasible_set": "simplex",
                "solver": "energy_qp",
                "target_policy": "leave_one_out",
                "tol": 1e-8,
                "maxiter": 10_000,
            }
        },
    }


def test_runtime_registry_resolves_discriminated_variants() -> None:
    """Resolve provider, distance, and feasible-set discriminator variants."""
    config = TypedAnalysisConfig.model_validate(_registry())

    assert isinstance(config.providers["local"], LocalParquetProvider)
    assert isinstance(config.distances["point"], PointDistanceSpec)
    assert isinstance(config.distances["energy"], StatisticalDistanceSpec)
    assert isinstance(config.feasible_sets["simplex"], SimplexFeasibleSet)


def _wasserstein_registry(solver: str, weights: list[float]) -> dict[str, object]:
    """Registry with one Wasserstein source-barycentre arm at a given arity."""
    raw = _registry()
    distances = raw["distances"]
    assert isinstance(distances, dict)
    distances["w2"] = {
        "distance_id": "w2",
        "kind": "statistical",
        "family": "wasserstein",
        "ground_distance": "point",
        "exponent": 1.0,
        "estimator": "balanced_wasserstein_2",
        "normalization": "rooted",
        "sample_design": "full_sample",
    }
    feasible_sets = raw["feasible_sets"]
    assert isinstance(feasible_sets, dict)
    feasible_sets["prescribed"] = {
        "feasible_set_id": "prescribed",
        "kind": "prescribed_source_weights",
        "source_weights": weights,
    }
    barycentres = raw["barycentres"]
    assert isinstance(barycentres, dict)
    arm: dict[str, object] = {
        "arm_id": "w2_arm",
        "problem": "source_barycentre",
        "geometry": "w2",
        "feasible_set": "prescribed",
        "solver": solver,
        "target_policy": "explicit",
    }
    if solver != "wasserstein_ot":
        arm["tol"] = 1e-12
        arm["maxiter"] = 128
    barycentres["w2_arm"] = arm
    return raw


def test_exact_wasserstein_arm_rejects_a_third_prescribed_source() -> None:
    """The closed form is exact only at two sources, so arity is gated early.

    Without this the third weight would reach the kernel and be rejected at
    stage runtime, after the DVC graph had already scheduled the cell.
    """
    raw = _wasserstein_registry("wasserstein_ot", [1 / 3, 1 / 3, 1 / 3])

    with pytest.raises(ValidationError, match="two-source"):
        TypedAnalysisConfig.model_validate(raw)


def test_free_support_wasserstein_arm_accepts_three_sources() -> None:
    """The iterative arm is the declared route past two sources."""
    raw = _wasserstein_registry("wasserstein_free_support", [1 / 3, 1 / 3, 1 / 3])

    config = TypedAnalysisConfig.model_validate(raw)

    arm = config.barycentres["w2_arm"]
    assert arm.solver == "wasserstein_free_support"
    assert arm.maxiter == FREE_SUPPORT_MAXITER


def test_free_support_wasserstein_arm_requires_a_solver_budget() -> None:
    """An iterative arm without tol/maxiter is a silently unbounded solve."""
    raw = _wasserstein_registry("wasserstein_free_support", [0.5, 0.5])
    barycentres = raw["barycentres"]
    assert isinstance(barycentres, dict)
    del barycentres["w2_arm"]["tol"]  # type: ignore[index]
    del barycentres["w2_arm"]["maxiter"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="explicit tol and maxiter"):
        TypedAnalysisConfig.model_validate(raw)


def test_runtime_registry_rejects_unknown_provider_kind() -> None:
    """Reject provider kinds outside the declared discriminator union."""
    raw = _registry()
    providers = raw["providers"]
    assert isinstance(providers, dict)
    providers["local"]["kind"] = "unknown"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


def test_provider_registry_rejects_runtime_artifact_paths() -> None:
    """Runtime locations belong to DVC commands rather than scientific identity."""
    raw = _registry()
    providers = raw["providers"]
    assert isinstance(providers, dict)
    providers["local"]["artifact_root"] = "data/local"  # type: ignore[index]

    with pytest.raises(ValidationError, match="artifact_root"):
        TypedAnalysisConfig.model_validate(raw)


def test_runtime_registry_validates_matryoshka_dimension() -> None:
    """A representation dimension is positive and bounded by native width."""
    raw = _registry()
    representations = raw["representations"]
    assert isinstance(representations, dict)
    representations["raw"].update(  # type: ignore[union-attr]
        transform="l2_normalize_rows",
        normalization="unit_rows",
        dimension=MATRYOSHKA_DIMENSION,
    )

    config = TypedAnalysisConfig.model_validate(raw)

    assert config.representations["raw"].dimension == MATRYOSHKA_DIMENSION
    representations["raw"]["dimension"] = 3  # type: ignore[index]
    with pytest.raises(ValidationError, match="native width"):
        TypedAnalysisConfig.model_validate(raw)


def test_provider_loading_truncates_then_renormalizes(tmp_path: Path) -> None:
    """Provider loading validates native rows before realizing a Matryoshka arm."""
    provider = LocalParquetProvider(
        provider_id="local",
        model_id="local-model",
        dimensions=3,
        normalization="unit_rows",
    )
    representation = {
        "representation_id": "configured-arm",
        "provider_id": "local",
        "transform": "l2_normalize_rows",
        "normalization": "unit_rows",
        "dimension": MATRYOSHKA_DIMENSION,
    }
    pd.DataFrame(
        {
            "url_hash": ["b", "a"],
            "embedding": [[0.0, 0.6, 0.8], [0.8, 0.0, 0.6]],
        }
    ).to_parquet(tmp_path / "item.parquet", index=False)
    config = TypedAnalysisConfig.model_validate(
        {
            **_registry(),
            "providers": {"local": provider.model_dump()},
            "representations": {"configured-arm": representation},
        }
    )

    (cloud,) = load_provider_clouds(
        provider, config.representations["configured-arm"], embeddings_dir=tmp_path
    )

    assert cloud.representation_id == "configured-arm"
    assert np.allclose(cloud.values, [[1.0, 0.0], [0.0, 1.0]])
    assert np.linalg.norm(cloud.values, axis=1) == pytest.approx([1.0, 1.0])
    assert (cloud.source_dimension, cloud.native_dimension) == (3, 3)
    assert cloud.realized_dimension == MATRYOSHKA_DIMENSION


def test_provider_loading_rejects_zero_prefix_after_native_validation(
    tmp_path: Path,
) -> None:
    """Truncation cannot turn a valid native unit row into an undefined unit row."""
    provider = LocalParquetProvider(
        provider_id="local",
        model_id="local-model",
        dimensions=3,
        normalization="unit_rows",
    )
    pd.DataFrame({"embedding": [[0.0, 0.0, 1.0]]}).to_parquet(
        tmp_path / "item.parquet", index=False
    )
    raw = _registry()
    representations = raw["representations"]
    assert isinstance(representations, dict)
    representations["raw"].update(  # type: ignore[union-attr]
        transform="l2_normalize_rows",
        normalization="unit_rows",
        dimension=MATRYOSHKA_DIMENSION,
    )
    representation = TypedAnalysisConfig.model_validate(raw).representations["raw"]

    with pytest.raises(ProviderInputError, match="finite non-zero rows"):
        load_provider_clouds(provider, representation, embeddings_dir=tmp_path)


def test_encoder_vintage_provider_uses_shared_distance_and_barycentre_path(
    tmp_path: Path,
) -> None:
    """An encoder-vintage entry reaches the generic distance and solver APIs."""
    provider = EncoderVintageProvider(
        provider_id="vintage-2024",
        model_id="encoder-model",
        vintage="2024",
        dimensions=2,
        normalization="raw",
        status="exploratory",
        caveat="diagnostic only",
    )
    representations = _registry()["representations"]
    assert isinstance(representations, dict)
    representation = representations["raw"]
    assert isinstance(representation, dict)
    typed_representation = {
        **representation,
        "provider_id": provider.provider_id,
        "representation_id": "vintage-2024-unit",
    }
    config = TypedAnalysisConfig.model_validate(
        {
            **_registry(),
            "providers": {provider.provider_id: provider.model_dump()},
            "representations": {
                "vintage-2024-unit": typed_representation,
            },
        }
    )
    for ticker, values in {
        "A": [[1.0, 0.0], [0.0, 1.0]],
        "B": [[1.0, 1.0], [0.0, 1.0]],
        "C": [[1.0, 0.5], [0.5, 1.0]],
    }.items():
        pd.DataFrame({"url_hash": ["b", "a"], "embedding": values}).to_parquet(
            tmp_path / f"{ticker}.parquet", index=False
        )

    clouds = load_provider_clouds(
        provider,
        config.representations["vintage-2024-unit"],
        embeddings_dir=tmp_path,
        item_ids=["A", "B", "C"],
    )
    geometry = config.distances["energy"]
    assert isinstance(geometry, StatisticalDistanceSpec)
    ground = config.ground_distance_for(geometry)
    distance = compute_statistical_distance(clouds, geometry, ground)
    arm = BarycentreArm(
        arm_id="energy_arm",
        problem="target_projection",
        geometry="energy",
        feasible_set="simplex_nonnegative",
        solver="energy_qp",
        target_policy="leave_one_out",
        tol=1e-6,
        maxiter=100,
    )
    barycentre = solve_energy_target_projections(clouds, geometry, arm, ground)

    assert distance.values.shape == (EXPECTED_SMOKE_CLOUD_COUNT,) * 2  # noqa: PD011
    assert len(barycentre.weights) == EXPECTED_SMOKE_CLOUD_COUNT
    assert all(np.all(np.isfinite(weights)) for weights in barycentre.weights)


def test_dated_provider_reads_the_explicit_embeddings_directory(tmp_path: Path) -> None:
    """Inference rows retain canonical dates without a registry-owned location."""
    config = TypedAnalysisConfig.model_validate(_registry())
    embeddings_dir = tmp_path / "selected"
    embeddings_dir.mkdir()
    pd.DataFrame(
        {
            "url_hash": ["b", "a"],
            "created_date": ["2020-01-02", "2020-01-01"],
            "embedding": [[0.0, 1.0], [1.0, 0.0]],
        }
    ).to_parquet(embeddings_dir / "A.parquet", index=False)

    (observations,) = load_dated_provider_observations(
        config.providers["local"],
        config.representations["raw"],
        embeddings_dir=embeddings_dir,
        item_ids=["A"],
    )

    assert observations.url_hashes == ("a", "b")
    np.testing.assert_array_equal(observations.values, [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(
        observations.dates,
        np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
    )


def test_real_encoder_vintages_pass_shared_artifact_smoke_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real annual outputs use coverage, distance, solver, and artifact APIs."""
    monkeypatch.chdir(REPO_ROOT)
    config = load_typed_analysis(REPO_ROOT / "params.yaml")
    provider_ids = ANNUAL_PROVIDER_IDS
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    shared_root = next(entry["shared"] for entry in dvc["vars"] if "shared" in entry)
    embeddings_root = REPO_ROOT / shared_root / "embeddings"
    roots = [embeddings_root / provider_id for provider_id in provider_ids]
    if not all(root.is_dir() for root in roots):
        pytest.skip("DVC ChronoBERT embedding outputs are not available")

    for provider_id, root in zip(provider_ids, roots, strict=True):
        provider = config.providers[provider_id]
        representation = config.representations[f"{provider_id}-unit"]
        paths = sorted(root.glob("*.parquet"))
        if len(paths) != EXPECTED_MARKET_COUNT:
            pytest.skip(
                f"{provider_id} does not have the complete "
                f"{EXPECTED_MARKET_COUNT}-file output"
            )
        clouds = load_provider_clouds(
            provider,
            representation,
            embeddings_dir=root,
            item_ids=[path.stem for path in paths],
        )
        assert len(clouds) == EXPECTED_MARKET_COUNT
        assert all(
            np.asarray(cloud.values).ndim == EXPECTED_ARRAY_RANK
            and np.asarray(cloud.values).shape[0] > 0
            and np.asarray(cloud.values).shape[1] == provider.dimensions
            for cloud in clouds
        )
        assert all(cloud.provider_vintage == provider.vintage for cloud in clouds)

        smoke_clouds = clouds[:EXPECTED_SMOKE_CLOUD_COUNT]
        geometry = config.distances["energy_v"]
        assert isinstance(geometry, StatisticalDistanceSpec)
        ground = config.ground_distance_for(geometry)
        arm = config.barycentres["energy_free_signed"]
        distance = compute_statistical_distance(smoke_clouds, geometry, ground)
        result = solve_energy_target_projections(
            smoke_clouds,
            geometry,
            arm,
            ground,
        )
        output_dir = tmp_path / provider_id
        summary = write_target_projection_artifact(
            output_dir,
            result,
            BarycentreArtifactMetadata(
                arm_id=arm.arm_id,
                provider_id=provider_id,
                representation_id=representation.representation_id,
                geometry_id="energy_v",
                feasible_set=arm.feasible_set,
                problem="target_projection",
                solver="energy_qp",
                estimator=geometry.estimator,
                input_hashes=tuple(
                    cloud.source_hash for cloud in smoke_clouds if cloud.source_hash
                ),
                parameter_identity=typed_analysis_parameter_identity(
                    REPO_ROOT / "params.yaml",
                    provider_ids=(provider_id,),
                    representation_ids=(representation.representation_id,),
                    arm_ids=(arm.arm_id,),
                ),
                code_identity="c" * 64,
                solver_tol=arm.tol,
                solver_maxiter=arm.maxiter,
                provider_model_id=provider.model_id,
                provider_vintage=provider.vintage,
                provider_status=provider.status,
                provider_caveat=provider.caveat,
                input_dtype=str(np.asarray(smoke_clouds[0].values).dtype),
                compute_dtype=str(np.asarray(smoke_clouds[0].values).dtype),
                output_dtype=str(np.asarray(smoke_clouds[0].values).dtype),
            ),
        )
        assert distance.values.shape == (EXPECTED_SMOKE_CLOUD_COUNT,) * 2  # noqa: PD011
        assert summary["provider_caveat"] == provider.caveat
        read_target_projection_artifact(output_dir)


def test_runtime_loader_reads_canonical_params() -> None:
    """Parse the repository's canonical typed-analysis registry."""
    config = load_typed_analysis(REPO_ROOT / "params.yaml")

    assert config.providers["qwen3-embedding-4b"].dimensions == EXPECTED_QWEN_DIMENSIONS
    energy_distance = config.distances["energy_v"]
    assert isinstance(energy_distance, StatisticalDistanceSpec)
    assert energy_distance.sample_design == "full_sample"
    ground = config.ground_distance_for(energy_distance)
    assert isinstance(ground, PointDistanceSpec)
    assert ground.metric == "euclidean"
    arm = config.barycentres["energy_simplex"]
    assert arm.maxiter == EXPECTED_ENERGY_MAXITER
    vintages = {
        config.providers[provider_id].vintage for provider_id in ANNUAL_PROVIDER_IDS
    }
    assert vintages == EXPECTED_CHRONOBERT_VINTAGES
    assert all(
        config.providers[provider_id].status == "exploratory"
        for provider_id in ANNUAL_PROVIDER_IDS
    )
    assert all(
        config.providers[provider_id].dimensions == EXPECTED_CHRONOBERT_DIMENSIONS
        and config.providers[provider_id].normalization == "raw"
        and config.providers[provider_id].model_id
        == f"manelalab/chrono-bert-v1-{year}1231"
        and config.representations[f"{provider_id}-unit"].normalization == "unit_rows"
        and config.representations[f"{provider_id}-unit"].dimension is None
        for year, provider_id in zip(
            range(2017, 2025), ANNUAL_PROVIDER_IDS, strict=True
        )
    )


def _canonical_scientific_config() -> dict[str, object]:
    """Load the complete authored scientific configuration for mutation tests."""
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_canonical_dvc_scientific_config_satisfies_cross_registry_contract() -> None:
    """Validate all authored typed rosters and scientific selectors together."""
    config = load_dvc_scientific_config(REPO_ROOT / "params.yaml")

    assert len(config.typed_geometry_component_specs) == EXPECTED_FULL_COMPONENT_CELLS
    assert len(config.typed_distance_windowed_specs) == EXPECTED_WINDOWED_DISTANCE_CELLS
    assert {cell.distance_id for cell in config.typed_distance_specs.values()} == {
        "wasserstein_w1",
        "wasserstein_w2",
    }
    assert {cell.arm_id for cell in config.barycentre_specs.values()} == {
        "wasserstein_w1_loo",
        "wasserstein_w2_loo",
    }


def _first_cell(raw: dict[str, object], roster: str) -> dict[str, object]:
    """Return the first authored cell of one roster for in-place mutation."""
    cells = raw[roster]
    assert isinstance(cells, dict)
    first = next(iter(cells.values()))
    assert isinstance(first, dict)
    return first


def _mutate_scientific_config(  # noqa: C901, PLR0912, PLR0915
    raw: dict[str, object], mutation: str
) -> None:
    """Apply one intentionally invalid cross-registry mutation."""
    if mutation == "market_roster":
        # A copy: params.yaml aliases market_symbols to symbols.
        market_symbols = raw["market_symbols"]
        assert isinstance(market_symbols, list)
        raw["market_symbols"] = market_symbols[:-1]
    elif mutation == "component_duplicate":
        cells = raw["typed_geometry_component_specs"]
        assert isinstance(cells, dict)
        cells["duplicate_cell"] = dict(
            _first_cell(raw, "typed_geometry_component_specs")
        )
    elif mutation == "unresolved_geometry":
        _first_cell(raw, "typed_geometry_component_specs")["geometry_id"] = "missing"
    elif mutation == "wasserstein_component":
        _first_cell(raw, "typed_geometry_component_specs")["geometry_id"] = (
            "wasserstein_w2"
        )
    elif mutation == "distance_provider_mismatch":
        _first_cell(raw, "typed_distance_specs")["provider_key"] = "bge-m3"
    elif mutation == "distance_without_component":
        cell = dict(_first_cell(raw, "typed_distance_specs"))
        cell["representation_key"] = "qwen3-embedding-8b-64-unit"
        cell["representation_id"] = "qwen3-embedding-8b-64-unit"
        cell["distance_id"] = "energy_v"
        cells = raw["typed_distance_specs"]
        assert isinstance(cells, dict)
        cells["orphan_energy"] = cell
    elif mutation == "windowed_full_sample_cell":
        windowed = _first_cell(raw, "typed_distance_windowed_specs")
        cell = {key: value for key, value in windowed.items() if key != "window_key"}
        cells = raw["typed_distance_specs"]
        assert isinstance(cells, dict)
        cells["windowed_in_full_roster"] = cell
    elif mutation == "window_mismatch":
        _first_cell(raw, "typed_distance_windowed_specs")["window_key"] = (
            "pit-2022-12-31"
        )
    elif mutation == "pit_date_mismatch":
        typed = raw["typed_analysis"]
        assert isinstance(typed, dict)
        windows = typed["windows"]
        assert isinstance(windows, dict)
        window = windows["pit-2018-12-31"]
        assert isinstance(window, dict)
        window["end"] = "2019-12-31"
    elif mutation == "barycentre_geometry_mismatch":
        _first_cell(raw, "barycentre_specs")["geometry_id"] = "energy_v"
    elif mutation == "barycentre_unknown_arm":
        _first_cell(raw, "barycentre_specs")["arm_id"] = "missing_arm"
    elif mutation == "selector_key_mismatch":
        selectors = raw["scientific_selectors"]
        assert isinstance(selectors, dict)
        selectors["typed_distance_provider_key"] = "bge-m3"
    else:
        selectors = raw["scientific_selectors"]
        assert isinstance(selectors, dict)
        selectors["typed_distance_id"] = "linear_mmd"


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("market_roster", "market_symbols"),
        ("component_duplicate", "duplicate identities"),
        ("unresolved_geometry", "invalid geometry"),
        ("wasserstein_component", "no reusable Wasserstein component"),
        ("distance_provider_mismatch", "invalid provider"),
        ("distance_without_component", "no geometry component cell"),
        ("windowed_full_sample_cell", "windowed cells belong to"),
        ("window_mismatch", "windowed distance cell"),
        ("pit_date_mismatch", "distance year"),
        ("barycentre_geometry_mismatch", "disagrees with registry arm"),
        ("barycentre_unknown_arm", "names no registry arm"),
        ("selector_key_mismatch", "provider key/ID"),
        ("selector_cell_missing", "not in typed_distance_specs"),
    ],
)
def test_dvc_scientific_config_rejects_cross_registry_drift(
    mutation: str, diagnostic: str
) -> None:
    """Reject roster, window, universe, and selector drift."""
    raw = _canonical_scientific_config()
    _mutate_scientific_config(raw, mutation)

    with pytest.raises(ValidationError) as exc_info:
        DvcScientificConfig.model_validate(raw)
    assert diagnostic in str(exc_info.value)


def test_typed_registry_identity_is_independent_of_mapping_order(
    tmp_path: Path,
) -> None:
    """Typed parameter identity follows resolved content, not YAML ordering."""
    first = {"typed_analysis": _registry()}
    second = {
        "typed_analysis": {
            key: first["typed_analysis"][key]
            for key in reversed(tuple(first["typed_analysis"]))
        }
    }
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    # Exercise both the public canonicalizer and the path-level identity helper.
    first_config = TypedAnalysisConfig.model_validate(first["typed_analysis"])
    second_config = TypedAnalysisConfig.model_validate(second["typed_analysis"])
    assert canonical_typed_analysis(first_config) == canonical_typed_analysis(
        second_config
    )
    first_path.write_text(yaml.safe_dump(first), encoding="utf-8")
    second_path.write_text(yaml.safe_dump(second, sort_keys=False), encoding="utf-8")
    assert typed_analysis_parameter_identity(first_path) == (
        typed_analysis_parameter_identity(second_path)
    )


def test_scoped_identity_ignores_exploratory_provider_changes(
    tmp_path: Path,
) -> None:
    """Exploratory registry edits do not change canonical artifact identity."""
    changed = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    changed["typed_analysis"]["providers"]["chronobert-2024"]["caveat"] = "changed"
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")

    scope = {
        "provider_ids": ("qwen3-embedding-4b",),
        "representation_ids": ("qwen3-embedding-4b-unit",),
        "distance_ids": ("energy_v",),
    }
    assert typed_analysis_parameter_identity(
        REPO_ROOT / "params.yaml", **scope
    ) == typed_analysis_parameter_identity(changed_path, **scope)
    assert typed_analysis_parameter_identity(REPO_ROOT / "params.yaml") != (
        typed_analysis_parameter_identity(changed_path)
    )


def test_barycentre_parameter_identity_carries_its_ground_metric(
    tmp_path: Path,
) -> None:
    """The arm's statistical geometry must include its point-distance closure."""
    changed = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    changed["typed_analysis"]["distances"]["euclidean_chord"]["metric"] = "angular"
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    scope = {
        "provider_ids": ("qwen3-embedding-8b",),
        "representation_ids": ("qwen3-embedding-8b-unit",),
        "arm_ids": ("wasserstein_w2_loo",),
    }

    assert typed_analysis_parameter_identity(
        REPO_ROOT / "params.yaml", **scope
    ) != typed_analysis_parameter_identity(changed_path, **scope)


@pytest.mark.parametrize(
    ("solver", "budget"),
    [("energy_qp", {"tol": None, "maxiter": None}), ("wasserstein_ot", {"tol": 1e-8})],
)
def test_runtime_registry_gates_the_declared_solver_budget(
    solver: str, budget: dict[str, object]
) -> None:
    """Iterative arms must declare a budget; the OT arm must not carry one."""
    raw = _registry()
    arms = raw["barycentres"]
    assert isinstance(arms, dict)
    arms["energy_arm"].update(budget)  # type: ignore[index]
    if solver == "wasserstein_ot":
        distances = raw["distances"]
        assert isinstance(distances, dict)
        distances["energy"]["family"] = "wasserstein"  # type: ignore[index]
        distances["energy"]["estimator"] = "balanced_wasserstein_1"  # type: ignore[index]
        distances["energy"]["normalization"] = "rooted"  # type: ignore[index]
        arms["energy_arm"]["solver"] = solver  # type: ignore[index]

    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [("solver", "wasserstein_ot"), ("problem", "source_barycentre")],
)
def test_runtime_registry_rejects_incompatible_energy_arm(
    field: str, value: str
) -> None:
    """Reject barycentre solver/problem combinations that do not match geometry."""
    raw = _registry()
    arms = raw["barycentres"]
    assert isinstance(arms, dict)
    arms["energy_arm"][field] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


def test_runtime_registry_accepts_mmd_simplex_arm() -> None:
    """A V-statistic MMD geometry may own a leave-one-out simplex QP arm."""
    raw = _registry()
    distances = raw["distances"]
    arms = raw["barycentres"]
    assert isinstance(distances, dict)
    assert isinstance(arms, dict)
    distances["energy"].update(  # type: ignore[union-attr]
        family="mmd", kernel="linear"
    )
    arms["energy_arm"]["solver"] = "mmd_qp"  # type: ignore[index]

    config = TypedAnalysisConfig.model_validate(raw)

    assert config.barycentres["energy_arm"].solver == "mmd_qp"


@pytest.mark.parametrize(
    ("field", "value"),
    [("solver", "energy_qp"), ("target_policy", "explicit")],
)
def test_runtime_registry_rejects_invalid_mmd_arm(field: str, value: str) -> None:
    """MMD arms cannot substitute an energy solver or non-LOO target policy."""
    raw = _registry()
    distances = raw["distances"]
    arms = raw["barycentres"]
    assert isinstance(distances, dict)
    assert isinstance(arms, dict)
    distances["energy"].update(  # type: ignore[union-attr]
        family="mmd", kernel="linear"
    )
    arms["energy_arm"]["solver"] = "mmd_qp"  # type: ignore[index]
    arms["energy_arm"][field] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


def _windowed_registry(**window_overrides: object) -> dict[str, object]:
    """Registry whose energy geometry is restricted to a declared window."""
    raw = _registry()
    distances = raw["distances"]
    assert isinstance(distances, dict)
    distances["energy"].update(  # type: ignore[union-attr]
        sample_design="point_in_time", sample_window="w1"
    )
    raw["windows"] = {
        "w1": {
            "window_id": "w1",
            "start": "2018-01-01",
            "end": "2020-12-31",
            **window_overrides,
        }
    }
    return raw


def test_registry_resolves_a_declared_sample_window() -> None:
    """A point-in-time geometry resolves to its registered window."""
    config = TypedAnalysisConfig.model_validate(_windowed_registry())
    geometry = config.distances["energy"]
    assert isinstance(geometry, StatisticalDistanceSpec)
    window = config.window_for(geometry)
    assert window is not None
    assert window.window_id == "w1"
    assert window.end is not None
    assert window.end.isoformat() == "2020-12-31"


def test_registry_reports_no_window_for_a_full_sample_geometry() -> None:
    """A full-sample geometry resolves to no window rather than to a default."""
    config = TypedAnalysisConfig.model_validate(_registry())
    geometry = config.distances["energy"]
    assert isinstance(geometry, StatisticalDistanceSpec)
    assert config.window_for(geometry) is None


def test_registry_rejects_a_full_sample_geometry_naming_a_window() -> None:
    """A geometry cannot claim the full sample and a restriction at once."""
    raw = _windowed_registry()
    distances = raw["distances"]
    assert isinstance(distances, dict)
    distances["energy"]["sample_design"] = "full_sample"  # type: ignore[index]

    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


def test_registry_rejects_a_restricted_geometry_without_a_window() -> None:
    """A restricted sample design must name the restriction it applied."""
    raw = _windowed_registry()
    distances = raw["distances"]
    assert isinstance(distances, dict)
    del distances["energy"]["sample_window"]  # type: ignore[index]

    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


def test_registry_rejects_an_unknown_sample_window() -> None:
    """A geometry cannot reference a window the registry does not declare."""
    raw = _windowed_registry()
    empty_windows: dict[str, object] = {}
    raw["windows"] = empty_windows

    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(raw)


@pytest.mark.parametrize(
    "overrides",
    [{"start": None, "end": None}, {"start": "2021-01-01", "end": "2020-12-31"}],
)
def test_registry_rejects_an_unbounded_or_inverted_window(
    overrides: dict[str, object],
) -> None:
    """A window must bound at least one side and must not end before it starts."""
    with pytest.raises(ValidationError):
        TypedAnalysisConfig.model_validate(_windowed_registry(**overrides))


def test_resolved_registry_separates_two_windows_of_one_functional() -> None:
    """Moving a window must move the resolved registry.

    The window is the only thing distinguishing ``energy_v_pit_2020`` from
    ``energy_v_pit_2019``: if it fell out of the serialized closure, two
    components over different samples would share a parameter identity and one
    would be accepted as a cache hit for the other.
    """
    base = canonical_typed_analysis(
        TypedAnalysisConfig.model_validate(_windowed_registry())
    )
    moved = canonical_typed_analysis(
        TypedAnalysisConfig.model_validate(_windowed_registry(end="2019-12-31"))
    )
    assert base != moved


def test_scoped_parameter_identity_carries_the_geometry_window() -> None:
    """A scoped identity must include the window its geometry declares."""
    identity = typed_analysis_parameter_identity(
        REPO_ROOT / "params.yaml",
        provider_ids=("qwen3-embedding-4b",),
        representation_ids=("qwen3-embedding-4b-unit",),
        distance_ids=("energy_v_pit_2020",),
    )
    other = typed_analysis_parameter_identity(
        REPO_ROOT / "params.yaml",
        provider_ids=("qwen3-embedding-4b",),
        representation_ids=("qwen3-embedding-4b-unit",),
        distance_ids=("energy_v_pit_2019",),
    )
    assert identity != other


def test_pit_parameter_identity_ignores_unrelated_params(tmp_path: Path) -> None:
    """PIT provenance must hash its resolved grid/registry closure, not the file."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    params["n_samples"] += 1
    second.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    first_roster = _vintage_roster(yaml.safe_load(first.read_text(encoding="utf-8")))
    second_roster = _vintage_roster(yaml.safe_load(second.read_text(encoding="utf-8")))
    assert pit_parameter_identity(first, first_roster) == pit_parameter_identity(
        second, second_roster
    )


def test_pit_parameter_identity_changes_with_owning_window(tmp_path: Path) -> None:
    """A PIT window edit must change the owning provenance identity."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    params["typed_analysis"]["windows"]["pit-2018-12-31"]["start"] = "2017-01-02"
    second.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    roster = _vintage_roster(params)
    assert pit_parameter_identity(first, roster) != pit_parameter_identity(
        second, roster
    )


def test_iterative_arm_exposes_its_declared_budget() -> None:
    """Consumers receive concrete settings without casting optional schema fields."""
    arm = BarycentreArm(
        arm_id="budget-test",
        problem="target_projection",
        geometry="wasserstein_w2",
        feasible_set="simplex_nonnegative",
        solver="wasserstein_w2_target",
        target_policy="leave_one_out",
        tol=3e-6,
        maxiter=7,
    )
    assert iteration_budget(arm) == (3e-6, 7)


def test_closed_form_arm_has_no_iteration_budget() -> None:
    """An exact noniterative solve must not acquire invented convergence settings."""
    arm = BarycentreArm(
        arm_id="closed-form-test",
        problem="source_barycentre",
        geometry="wasserstein_w2",
        feasible_set="simplex_nonnegative",
        solver="wasserstein_ot",
        target_policy="explicit",
    )
    with pytest.raises(ValueError, match="no iterative solver budget"):
        iteration_budget(arm)

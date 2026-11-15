"""Tests for the Paper 5 GMV validation (exploratory portfolio comparison)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from pipeline._kernels.paper3.validation import solve_certificate
from pipeline.cli.commands.p5 import gmv_validation as adapter
from pipeline.io.contracts import write_manifest
from pipeline.stages.papers.paper3.certificate import _ReferenceDesign
from pipeline.stages.papers.paper5 import gmv_validation as comparison
from pipeline.stages.papers.paper5.gmv_validation import (
    _ALL_RESIDUAL_KEYS,
    _ALL_TOTAL_KEYS,
    _CANDIDATES,
    _SPATIAL_CANDIDATE_KEYS,
    _TOTAL_COVARIANCE_CANDIDATES,
    Paper5GMVValidationConfig,
    Paper5GMVValidationPaths,
    _candidate_weights,
    _capital_from_budget,
    _capped_gmv,
    _certificate_gap,
    _date_digest,
    _gmv_diagnostics,
    _parameter_identity,
    _PitCertificateIdentity,
    _reference_rows,
    _validate_covariance,
    _validate_pit_certificate_contract,
    _validate_spatial_artifact_protocol,
    _validate_spatial_date_identity,
    _variance_rows,
)
from typer.testing import CliRunner

_ROOT = Path(__file__).resolve().parents[7]
_TEST_CAP = 0.15
_TEST_GAP_TOLERANCE = 1e-8
_STAGE_FORMATION_CALLS = 2
_RETURN_POISON_MIN_VARIANCE_DELTA = 1e-8


def _pit_identity() -> _PitCertificateIdentity:
    """Return the selected typed PIT identity used by contract fixtures."""
    return _PitCertificateIdentity(
        formation_vintage="2022-12-31",
        provider_id="qwen3-embedding-8b",
        representation_id="qwen3-embedding-8b-unit",
        distance_id="wasserstein_w2_pit_2022",
        family="wasserstein",
        ground_distance="euclidean_chord",
        estimator="balanced_wasserstein_2",
        normalization="rooted",
        sample_design="point_in_time",
        sample_size=128,
        sample_window_id="pit-2022-12-31",
        window_start="2018-01-01",
        window_end="2022-12-31",
        source_scale="rooted_wasserstein_2",
        archive_scale="squared_wasserstein_2_transport_cost",
    )


@pytest.mark.parametrize(
    "section",
    [
        "invocation",
        "factor_data",
        "pit_grid",
        "reference",
        "distance",
        "geometry",
        "window",
    ],
)
def test_parameter_identity_tracks_only_consumed_settings(
    tmp_path: Path, section: str
) -> None:
    """Unused settings are irrelevant; each consumed parameter family is recorded."""
    params = yaml.safe_load((_ROOT / "params.yaml").read_text(encoding="utf-8"))
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
    config = Paper5GMVValidationConfig()
    baseline = _parameter_identity(params_file, config)
    params["unused_parameter"] = {"anything": 99}
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
    assert _parameter_identity(params_file, config) == baseline

    if section == "invocation":
        config = Paper5GMVValidationConfig(reference_seed=config.reference_seed + 1)
    elif section == "factor_data":
        params["paper5"]["factor_data"]["sources"][0]["source_vintage"] = "changed"
    elif section == "pit_grid":
        params["paper3"]["pit_vintage_grid"] = list(
            reversed(params["paper3"]["pit_vintage_grid"])
        )
    elif section == "reference":
        params["paper3"]["certificate_feasibility"]["news_reference"]["seed"] += 1
    elif section == "distance":
        params["typed_analysis"]["distances"]["wasserstein_w2_pit_2022"][
            "sample_size"
        ] += 1
    elif section == "geometry":
        for key in ("wasserstein_w2", "wasserstein_w2_pit_2022"):
            params["typed_analysis"]["distances"][key]["exponent"] = 1.5
    else:
        params["typed_analysis"]["windows"]["pit-2022-12-31"]["start"] = "2018-01-02"
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")

    assert _parameter_identity(params_file, config) != baseline


def _pit_manifest(identity: _PitCertificateIdentity) -> dict[str, object]:
    """Build a complete v2 PIT manifest fixture."""
    values = asdict(identity)
    values.pop("formation_vintage")
    return {
        "stage_key": "p3_pit_w2",
        "protocol_id": "paper3.pit-w2.v2",
        "metadata": {
            "source_scale": identity.source_scale,
            "archive_scale": identity.archive_scale,
            "item_ids": ["A", "B", "C"],
            "vintage_windows": {identity.formation_vintage: identity.sample_window_id},
            "vintage_identity_schema": "pit_vintage_identity.v1",
            "vintage_identities": {identity.formation_vintage: values},
        },
        "upstream_artifacts": [
            {
                "kind": "path",
                "value": (
                    "data/shared/typed_distances_windowed/"
                    f"{identity.provider_id}/{identity.representation_id}/"
                    f"{identity.distance_id}"
                ),
                "vintage": identity.formation_vintage,
                "distance_id": identity.distance_id,
                "provider_id": identity.provider_id,
                "representation_id": identity.representation_id,
            }
        ],
    }


def _write_spatial_fixture(
    directory: Path,
    tickers: list[str],
    calibration_dates: np.ndarray,
    evaluation_dates: np.ndarray,
) -> None:
    """Write a valid v2 covariance artifact with distinct evaluation-only cells."""
    directory.mkdir(parents=True, exist_ok=True)
    n_assets = len(tickers)
    calibration_matrix = np.eye(n_assets, dtype=np.float64)
    evaluation_matrix = np.diag(np.linspace(0.01, 10.0, n_assets))
    arrays: dict[str, np.ndarray] = {"tickers": np.asarray(tickers, dtype="U")}
    for key in (*_ALL_TOTAL_KEYS.values(), *_ALL_RESIDUAL_KEYS.values()):
        arrays[key] = calibration_matrix.copy()
        arrays[key.replace("calibration_", "evaluation_", 1)] = evaluation_matrix.copy()
    covariance_path = directory / "covariances.npz"
    np.savez_compressed(
        covariance_path,
        **arrays,  # pyrefly: ignore[bad-argument-type]  # NumPy kwargs are array payloads
    )
    config = Paper5GMVValidationConfig()
    summary = {
        "schema_version": "paper5_spatial_covariance.v2",
        "analysis_id": "paper5-spatial-covariance-v2",
        "sample": {
            "n_assets": n_assets,
            "tickers": tickers,
            "calibration_start": str(pd.Timestamp(calibration_dates.min()).date()),
            "calibration_end": str(pd.Timestamp(calibration_dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(evaluation_dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(evaluation_dates.max()).date()),
            "calibration_dates": len(calibration_dates),
            "evaluation_dates": len(evaluation_dates),
            "calibration_date_sha256": _date_digest(calibration_dates),
            "evaluation_date_sha256": _date_digest(evaluation_dates),
        },
        "parameters": {
            "calibration_start": config.calibration_start,
            "calibration_end": config.calibration_end,
            "evaluation_start": config.evaluation_start,
            "evaluation_end": config.evaluation_end,
            "factor_model": config.factor_model,
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "w1_barycentre_arm_id": config.w1_barycentre_arm_id,
            "w2_barycentre_arm_id": config.w2_barycentre_arm_id,
            "w1_geometry_id": config.w1_geometry_id,
            "w2_geometry_id": config.w2_geometry_id,
        },
        "estimator": {
            "risk_unit": "excess_return",
            "residual_shrinkage_target": config.residual_shrinkage_target,
            "implementation": "statadic.estimation.covariance.LedoitWolf",
            "input": "factor-residualized spatial innovations",
            "fit_mode": (
                "two-step plug-in: fit rho/QMLE, then shrink fitted innovations"
            ),
            "reduced_form_loadings": "B=A^-1 C_f",
        },
    }
    summary_path = directory / "results.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    upstream_artifacts: list[dict[str, object]] = []
    environment: dict[str, object] = {}
    manifest = {
        "schema_version": "1.0",
        "stage_key": "p5_spatial_covariance",
        "lane": "paper5",
        "phase": "exploratory",
        "protocol_id": "paper5.spatial_covariance.ledoit_wolf.v2",
        "code_identity": {"kind": "fixture", "value": "fixture"},
        "parameter_identity": {"kind": "fixture", "value": "fixture"},
        "seed_policy": "fixture",
        "upstream_artifacts": upstream_artifacts,
        "environment": environment,
        "outputs": [
            {
                "path": covariance_path.name,
                "kind": "npz",
                "sha256": digest(covariance_path),
            },
            {
                "path": summary_path.name,
                "kind": "json",
                "sha256": digest(summary_path),
            },
        ],
        "risk_class": "moderate",
    }
    write_manifest(directory / "provenance.manifest.json", manifest)


def _assert_panel_certificate_contract(summary: dict[str, object]) -> None:
    """Check the panel-specific certificate metadata in the stage summary."""
    contracts = cast(
        "dict[str, dict[str, dict[str, str]]]", summary["candidate_contract"]
    )
    assert contracts["oos_primary"]["paper3_w2_certificate"]["formation_rule"] == (
        "normalized_w2_certificate_capital_map"
    )
    assert (
        contracts["paper3_in_sample"]["paper3_w2_certificate"]["formation_rule"]
        == "normalized_w2_certificate"
    )


def _rewrite_evaluation_covariances(directory: Path) -> None:
    """Change evaluation-only covariance cells and refresh their manifest hash."""
    covariance_path = directory / "covariances.npz"
    with np.load(covariance_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    for key in tuple(arrays):
        if key.startswith("evaluation_"):
            arrays[key] = np.diag(np.linspace(100.0, 1_000.0, len(arrays[key])))
    np.savez_compressed(
        covariance_path,
        **arrays,  # pyrefly: ignore[bad-argument-type]  # NumPy kwargs are array payloads
    )
    manifest_path = directory / "provenance.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for output in manifest["outputs"]:
        if output["path"] == covariance_path.name:
            output["sha256"] = hashlib.sha256(covariance_path.read_bytes()).hexdigest()
    write_manifest(manifest_path, manifest)


@pytest.fixture
def _fixed_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the semflow code identity so the stage test is hermetic."""
    monkeypatch.setattr(comparison, "_stage_source_digest", lambda: "0" * 64)


@pytest.mark.usefixtures("_fixed_code_identity")
def test_stage_holdout_poison_preserves_formation_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full stage keeps formation inputs independent of evaluation inputs."""
    monkeypatch.chdir(_ROOT)
    rng = np.random.default_rng(9)
    tickers = [f"T{index:03d}" for index in range(100)]
    calibration_dates = np.arange(
        np.datetime64("2018-01-01"), np.datetime64("2018-03-02")
    ).astype("datetime64[ns]")
    evaluation_dates = np.arange(
        np.datetime64("2023-01-01"), np.datetime64("2023-03-02")
    ).astype("datetime64[ns]")
    calibration_returns = rng.normal(size=(len(calibration_dates), len(tickers)))
    evaluation_panel = rng.normal(size=(len(evaluation_dates), len(tickers)))
    baseline_evaluation_panel = evaluation_panel.copy()
    return_poison_spatial_dir, covariance_poison_spatial_dir = (
        tmp_path / "return-poison-spatial",
        tmp_path / "covariance-poison-spatial",
    )
    for spatial_dir in (return_poison_spatial_dir, covariance_poison_spatial_dir):
        _write_spatial_fixture(
            spatial_dir, tickers, calibration_dates, evaluation_dates
        )
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        (_ROOT / "params.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "returns").mkdir()
    (tmp_path / "oos_returns").mkdir()
    (tmp_path / "pit.npz").write_bytes(b"fixture")
    (tmp_path / "pit.json").write_bytes(b"fixture")
    config = Paper5GMVValidationConfig(
        reference_draws=1_000,
        maximum_weight=_TEST_CAP,
    )
    design = _ReferenceDesign("uniform_cap_15", 1.0, _TEST_CAP)

    def fake_panel(
        _params_file: Path,
        _returns_dir: Path,
        _oos_returns_dir: Path,
        _tickers: list[str],
        start: str,
        _end: str,
        _factor_model: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if start == config.calibration_start:
            return calibration_dates.copy(), calibration_returns.copy()
        return evaluation_dates.copy(), evaluation_panel.copy()

    monkeypatch.setattr(comparison, "_load_excess_return_panel", fake_panel)
    candidate_weights_spy = Mock(wraps=_candidate_weights)
    monkeypatch.setattr(comparison, "_candidate_weights", candidate_weights_spy)
    monkeypatch.setattr(
        comparison,
        "_reference_contract",
        lambda _params_file, _config: (
            (design,),
            42,
            {"requested_draws": 1_000, "seed": 42},
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_load_certificate_budget",
        lambda _path, _manifest_path, _params_file, _tickers, _config: (
            np.full(100, 0.01, dtype=np.float64),
            {"formation_vintage": "2022-12-31"},
        ),
    )

    def run(
        output_dir: Path, spatial_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        comparison.run_paper5_gmv_validation(
            Paper5GMVValidationPaths(
                params_file=params_file,
                returns_dir=tmp_path / "returns",
                oos_returns_dir=tmp_path / "oos_returns",
                certificate_pit_w2_path=tmp_path / "pit.npz",
                certificate_pit_manifest_path=tmp_path / "pit.json",
                spatial_covariance_dir=spatial_dir,
                output_dir=output_dir,
            ),
            config,
        )
        return (
            pd.read_parquet(output_dir / "weights.parquet"),
            pd.read_parquet(output_dir / "results.parquet"),
            cast(
                "dict[str, object]",
                json.loads((output_dir / "results.json").read_text(encoding="utf-8")),
            ),
        )

    first_weights, first_results, first_summary = run(
        tmp_path / "first", return_poison_spatial_dir
    )
    np.testing.assert_array_equal(
        candidate_weights_spy.call_args_list[0].args[0], calibration_returns
    )
    _assert_panel_certificate_contract(first_summary)

    evaluation_panel[:] = (
        evaluation_panel
        * np.linspace(0.5, 3.0, len(tickers), dtype=np.float64)[None, :]
    )
    second_weights, second_results, _ = run(
        tmp_path / "return-poison", return_poison_spatial_dir
    )
    assert len(candidate_weights_spy.call_args_list) == 2 * _STAGE_FORMATION_CALLS
    np.testing.assert_array_equal(
        candidate_weights_spy.call_args_list[_STAGE_FORMATION_CALLS].args[0],
        calibration_returns,
    )
    first_primary = first_weights.loc[
        first_weights["panel"] == "oos_primary"
    ].sort_values(["candidate", "ticker"])
    second_primary = second_weights.loc[
        second_weights["panel"] == "oos_primary"
    ].sort_values(["candidate", "ticker"])
    np.testing.assert_allclose(
        first_primary["weight"].to_numpy(),
        second_primary["weight"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    first_variances = (
        first_results.loc[first_results["panel"] == "oos_primary"]
        .sort_values("candidate")["variance"]
        .to_numpy()
    )
    second_variances = (
        second_results.loc[second_results["panel"] == "oos_primary"]
        .sort_values("candidate")["variance"]
        .to_numpy()
    )
    assert np.any(
        np.abs(first_variances - second_variances) > _RETURN_POISON_MIN_VARIANCE_DELTA
    )

    evaluation_panel[:] = baseline_evaluation_panel
    _rewrite_evaluation_covariances(covariance_poison_spatial_dir)
    third_weights, third_results, _ = run(
        tmp_path / "covariance-poison", covariance_poison_spatial_dir
    )
    assert len(candidate_weights_spy.call_args_list) == 3 * _STAGE_FORMATION_CALLS
    np.testing.assert_array_equal(
        candidate_weights_spy.call_args_list[2 * _STAGE_FORMATION_CALLS].args[0],
        calibration_returns,
    )
    third_primary = third_weights.loc[
        third_weights["panel"] == "oos_primary"
    ].sort_values(["candidate", "ticker"])
    np.testing.assert_allclose(
        first_primary["weight"].to_numpy(),
        third_primary["weight"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    third_variances = (
        third_results.loc[third_results["panel"] == "oos_primary"]
        .sort_values("candidate")["variance"]
        .to_numpy()
    )
    np.testing.assert_allclose(first_variances, third_variances, rtol=0.0, atol=0.0)


def test_capped_gmv_respects_simplex_and_common_cap() -> None:
    """All covariance-based candidates use the same long-only cap."""
    covariance = np.diag(np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64))
    weights = _capped_gmv(covariance, 0.40, "synthetic")
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.0
    assert weights.max() <= 0.40 + 1e-8


def test_capped_gmv_matches_diagonal_inverse_variance_and_is_scale_invariant() -> None:
    """The solver reaches the analytical diagonal optimum in either scale."""
    variances = np.linspace(1e-6, 1e-5, 100)
    covariance = np.diag(variances)
    expected = (1.0 / variances) / np.sum(1.0 / variances)
    first = _capped_gmv(covariance, 0.15, "diagonal")
    second = _capped_gmv(covariance * 1e6, 0.15, "scaled diagonal")
    np.testing.assert_allclose(first, expected, atol=2e-7, rtol=0.0)
    np.testing.assert_allclose(second, first, atol=2e-7, rtol=0.0)
    assert (
        _gmv_diagnostics(covariance, first, _TEST_CAP)["convex_first_order_gap"]
        <= _TEST_GAP_TOLERANCE
    )
    assert (
        _gmv_diagnostics(covariance * 1e6, second, _TEST_CAP)["convex_first_order_gap"]
        <= _TEST_GAP_TOLERANCE
    )


def test_capped_gmv_handles_zero_covariance_and_infeasible_cap() -> None:
    """Zero-risk ties are deterministic and an impossible cap fails closed."""
    np.testing.assert_allclose(
        _capped_gmv(np.zeros((4, 4)), 0.25, "zero"), np.full(4, 0.25)
    )
    with pytest.raises(ValueError, match="infeasible"):
        _capped_gmv(np.eye(4), 0.24, "infeasible")


def test_capital_map_matches_paper3_conversion() -> None:
    """The primary certificate weights are q_i/sigma_i, renormalized."""
    budget = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    volatility = np.array([0.01, 0.02, 0.04, 0.05], dtype=np.float64)
    expected = budget / volatility
    expected /= expected.sum()
    np.testing.assert_allclose(_capital_from_budget(budget, volatility), expected)


def test_capital_map_retains_cap_violation_without_silent_clipping() -> None:
    """The normalized certificate rule remains visible when capital is infeasible."""
    budget = np.array([0.9, 0.1], dtype=np.float64)
    mapped = _capital_from_budget(budget, np.array([0.001, 0.1]))
    assert mapped.max() > _TEST_CAP
    np.testing.assert_allclose(mapped.sum(), 1.0)


def test_holdout_mutation_cannot_change_formation_weights() -> None:
    """Evaluation-only mutations affect outcomes, never calibration weights."""
    generator = np.random.default_rng(7)
    calibration = generator.normal(size=(80, 4))
    evaluation = generator.normal(size=(40, 4))
    mutated_evaluation = evaluation * 1_000.0
    model_covariances = {
        name: np.eye(4, dtype=np.float64) * (index + 1.0)
        for index, name in enumerate(_TOTAL_COVARIANCE_CANDIDATES)
    }
    certificate = np.full(4, 0.25, dtype=np.float64)

    first = _candidate_weights(
        calibration, certificate, model_covariances, 0.40, capital=True
    )
    second = _candidate_weights(
        calibration, certificate, model_covariances, 0.40, capital=True
    )
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_allclose(first[name], second[name])

    references = cast(
        "list[dict[str, object]]", [{"candidate": name} for name in first]
    )
    first_rows = _variance_rows(
        "oos_primary",
        evaluation,
        first,
        references,
        maximum_weight=0.40,
    )
    mutated_rows = _variance_rows(
        "oos_primary",
        mutated_evaluation,
        second,
        references,
        maximum_weight=0.40,
    )
    assert [row["candidate"] for row in first_rows] == [
        row["candidate"] for row in mutated_rows
    ]
    assert any(
        row["variance"] != mutated["variance"]
        for row, mutated in zip(first_rows, mutated_rows, strict=True)
    )


def test_result_metadata_separates_formation_from_evaluation_risk() -> None:
    """Formation metadata identifies both portfolio panels and their coordinates."""
    generator = np.random.default_rng(8)
    returns = generator.normal(size=(60, 4))
    equal = np.full(4, 0.25, dtype=np.float64)
    candidates = dict.fromkeys(_CANDIDATES, equal)
    references = cast(
        "list[dict[str, object]]", [{"candidate": name} for name in _CANDIDATES]
    )
    rows = _variance_rows(
        "oos_primary",
        returns,
        candidates,
        references,
        maximum_weight=0.40,
    )
    by_candidate = {str(row["candidate"]): row for row in rows}
    assert by_candidate["paper3_w2_certificate"]["formation_rule"] == (
        "normalized_w2_certificate_capital_map"
    )
    assert by_candidate["paper3_w2_certificate"]["formation_objective_kind"] == (
        "w2_certificate"
    )
    assert by_candidate["equal_weight"]["formation_objective_kind"] == "rule_based"
    assert by_candidate["inverse_volatility"]["formation_objective_kind"] == (
        "rule_based"
    )
    assert by_candidate["sample_gmv"]["formation_objective_kind"] == (
        "total_excess_return_covariance"
    )
    assert (
        by_candidate["spatial_spherical_w1_qmle_residual"]["formation_objective_kind"]
        == "residual_covariance"
    )
    assert {row["evaluation_risk"] for row in rows} == {"total_excess_return_variance"}
    assert {row["weight_coordinates"] for row in rows} == {"capital"}
    assert {row["formation_input_scaling"] for row in rows} == {
        "unstandardized_excess_returns"
    }

    secondary_rows = _variance_rows(
        "paper3_in_sample",
        returns,
        candidates,
        references,
        maximum_weight=0.40,
    )
    secondary_by_candidate = {str(row["candidate"]): row for row in secondary_rows}
    assert secondary_by_candidate["paper3_w2_certificate"]["formation_rule"] == (
        "normalized_w2_certificate"
    )
    assert (
        secondary_by_candidate["paper3_w2_certificate"]["weight_coordinates"]
        == "normalized_standardized_asset"
    )
    assert {row["formation_input_scaling"] for row in secondary_rows} == {
        "calibration_volatility_standardized_excess_returns"
    }
    assert {row["evaluation_input_scaling"] for row in secondary_rows} == {
        "calibration_volatility_standardized_excess_returns"
    }
    assert secondary_by_candidate["sample_gmv"]["formation_objective_kind"] == (
        "standardized_excess_return_covariance"
    )


def test_certificate_solver_uses_already_squared_archive_scale() -> None:
    """The PIT contract accepts squared W2 and rejects rooted archive labels."""
    identity = _pit_identity()
    tickers = ["A", "B", "C"]
    manifest = _pit_manifest(identity)
    _validate_pit_certificate_contract(
        cast("dict[str, object]", manifest),
        {"2022-12-31"},
        tickers,
        identity,
    )
    bad_manifest = {
        **manifest,
        "metadata": {
            **cast("dict[str, object]", manifest["metadata"]),
            "archive_scale": "rooted_wasserstein_2",
        },
    }
    with pytest.raises(ValueError, match="already contain squared"):
        _validate_pit_certificate_contract(
            cast("dict[str, object]", bad_manifest),
            {"2022-12-31"},
            tickers,
            identity,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representation_id", "qwen3-embedding-8b-64-unit"),
        ("distance_id", "wasserstein_w2_pit_2021"),
        ("sample_window_id", "pit-2021-12-31"),
        ("window_end", "2025-12-31"),
    ],
)
def test_pit_contract_rejects_near_miss_vintage_identity(
    field: str, value: str
) -> None:
    """A valid archive hash cannot excuse a scientifically wrong PIT identity."""
    identity = _pit_identity()
    manifest = _pit_manifest(identity)
    metadata = cast("dict[str, object]", manifest["metadata"])
    identities = cast("dict[str, object]", metadata["vintage_identities"])
    selected = cast("dict[str, object]", identities[identity.formation_vintage])
    poisoned_selected = {**selected, field: value}
    poisoned_metadata = {
        **metadata,
        "vintage_identities": {
            identity.formation_vintage: poisoned_selected,
        },
    }
    with pytest.raises(ValueError, match="identity"):
        _validate_pit_certificate_contract(
            {**manifest, "metadata": poisoned_metadata},
            {identity.formation_vintage},
            ["A", "B", "C"],
            identity,
        )


def test_certificate_gap_is_small_for_a_valid_squared_distance_solution() -> None:
    """The selected PIT certificate carries a simplex optimality certificate."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    matrix = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    solution = solve_certificate(matrix, cap=1.0)
    assert _certificate_gap(matrix, solution.weights) <= _TEST_GAP_TOLERANCE


def test_reference_percentiles_are_deterministic_and_complete() -> None:
    """The Paper 3 capped-Dirichlet rank is reproducible for every candidate."""
    covariance = np.diag(np.array([1.0, 1.5, 2.0], dtype=np.float64))
    equal = np.full(3, 1.0 / 3.0, dtype=np.float64)
    candidates = dict.fromkeys(_CANDIDATES, equal)
    design = _ReferenceDesign("uniform", 1.0, 0.60)
    first = _reference_rows(
        covariance,
        candidates,
        (design,),
        2_000,
        42,
        panel="paper3_in_sample",
    )
    second = _reference_rows(
        covariance,
        candidates,
        (design,),
        2_000,
        42,
        panel="paper3_in_sample",
    )
    assert first == second
    assert len(first) == len(_CANDIDATES)
    assert all(0.0 <= cast("float", row["percentile"]) <= 1.0 for row in first)


def test_calibration_spatial_keys_exclude_evaluation_inputs() -> None:
    """The portfolio stage's spatial optimization inputs are calibration-only."""
    assert all(
        key.startswith("calibration_") for key in _SPATIAL_CANDIDATE_KEYS.values()
    )


def test_spatial_artifact_protocol_rejects_stale_or_poisoned_dates() -> None:
    """A calibration name cannot override a stale summary formation window."""
    config = Paper5GMVValidationConfig()
    tickers = ["A", "B"]
    sample = {
        "tickers": tickers,
        "calibration_start": "2018-01-02",
        "calibration_end": "2022-12-30",
        "evaluation_start": "2023-01-03",
        "evaluation_end": "2026-04-06",
        "calibration_date_sha256": "calibration-date-hash",
        "evaluation_date_sha256": "evaluation-date-hash",
    }
    parameters = {
        "calibration_start": config.calibration_start,
        "calibration_end": config.calibration_end,
        "evaluation_start": config.evaluation_start,
        "evaluation_end": config.evaluation_end,
        "factor_model": config.factor_model,
        "provider_id": config.provider_id,
        "representation_id": config.representation_id,
        "w1_barycentre_arm_id": config.w1_barycentre_arm_id,
        "w2_barycentre_arm_id": config.w2_barycentre_arm_id,
        "w1_geometry_id": config.w1_geometry_id,
        "w2_geometry_id": config.w2_geometry_id,
        "residual_shrinkage_target": config.residual_shrinkage_target,
    }
    summary = {
        "schema_version": "paper5_spatial_covariance.v2",
        "analysis_id": "paper5-spatial-covariance-v2",
        "sample": sample,
        "parameters": parameters,
        "estimator": {
            "risk_unit": "excess_return",
            "residual_shrinkage_target": config.residual_shrinkage_target,
            "implementation": "statadic.estimation.covariance.LedoitWolf",
            "input": "factor-residualized spatial innovations",
            "fit_mode": (
                "two-step plug-in: fit rho/QMLE, then shrink fitted innovations"
            ),
            "reduced_form_loadings": "B=A^-1 C_f",
        },
    }
    manifest = {
        "stage_key": "p5_spatial_covariance",
        "protocol_id": "paper5.spatial_covariance.ledoit_wolf.v2",
    }
    _validate_spatial_artifact_protocol(
        cast("dict[str, object]", summary),
        cast("dict[str, object]", manifest),
        tickers,
        config,
    )
    poisoned = {**summary, "sample": {**sample, "calibration_end": "2025-12-31"}}
    with pytest.raises(ValueError, match="dates"):
        _validate_spatial_artifact_protocol(
            cast("dict[str, object]", poisoned),
            cast("dict[str, object]", manifest),
            tickers,
            config,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("parameters", "w1_geometry_id", "wasserstein_w2"),
        ("parameters", "w1_barycentre_arm_id", "wrong_w1_arm"),
        ("parameters", "w2_barycentre_arm_id", "wrong_w2_arm"),
        ("estimator", "residual_shrinkage_target", "scaled_identity"),
    ],
)
def test_spatial_artifact_rejects_mismatched_model_identity(
    section: str, field: str, value: str
) -> None:
    """Every declared spatial arm and shrinkage target is enforced downstream."""
    config = Paper5GMVValidationConfig()
    tickers = ["A", "B"]
    sample = {
        "tickers": tickers,
        "calibration_start": "2018-01-02",
        "calibration_end": "2022-12-30",
        "evaluation_start": "2023-01-03",
        "evaluation_end": "2026-04-06",
        "calibration_date_sha256": "calibration-date-hash",
        "evaluation_date_sha256": "evaluation-date-hash",
    }
    parameters = {
        "calibration_start": config.calibration_start,
        "calibration_end": config.calibration_end,
        "evaluation_start": config.evaluation_start,
        "evaluation_end": config.evaluation_end,
        "factor_model": config.factor_model,
        "provider_id": config.provider_id,
        "representation_id": config.representation_id,
        "w1_barycentre_arm_id": config.w1_barycentre_arm_id,
        "w2_barycentre_arm_id": config.w2_barycentre_arm_id,
        "w1_geometry_id": config.w1_geometry_id,
        "w2_geometry_id": config.w2_geometry_id,
    }
    estimator = {
        "risk_unit": "excess_return",
        "residual_shrinkage_target": config.residual_shrinkage_target,
        "implementation": "statadic.estimation.covariance.LedoitWolf",
        "input": "factor-residualized spatial innovations",
        "fit_mode": "two-step plug-in: fit rho/QMLE, then shrink fitted innovations",
        "reduced_form_loadings": "B=A^-1 C_f",
    }
    poisoned_section = {
        **(parameters if section == "parameters" else estimator),
        field: value,
    }
    summary = {
        "schema_version": "paper5_spatial_covariance.v2",
        "analysis_id": "paper5-spatial-covariance-v2",
        "sample": sample,
        "parameters": poisoned_section if section == "parameters" else parameters,
        "estimator": poisoned_section if section == "estimator" else estimator,
    }
    manifest = {
        "stage_key": "p5_spatial_covariance",
        "protocol_id": "paper5.spatial_covariance.ledoit_wolf.v2",
    }
    with pytest.raises(ValueError, match=r"(parameter|target|shrinkage)"):
        _validate_spatial_artifact_protocol(
            cast("dict[str, object]", summary),
            cast("dict[str, object]", manifest),
            tickers,
            config,
        )


def test_spatial_artifact_requires_identical_factor_complete_dates() -> None:
    """Paper 3 rejects a covariance artifact formed on a different date panel."""
    calibration = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"
    )
    evaluation = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03"], dtype="datetime64[D]"
    )
    sample = {
        "calibration_date_sha256": _date_digest(calibration),
        "evaluation_date_sha256": _date_digest(evaluation),
    }
    _validate_spatial_date_identity(
        cast("dict[str, object]", sample), calibration, evaluation
    )
    poisoned = {**sample, "calibration_date_sha256": "poisoned"}
    with pytest.raises(ValueError, match="date_sha256"):
        _validate_spatial_date_identity(
            cast("dict[str, object]", poisoned), calibration, evaluation
        )


def test_non_psd_covariance_fails_closed() -> None:
    """Numerical optimization rejects a materially indefinite covariance."""
    covariance = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        _validate_covariance(covariance, 2, "synthetic")


def test_dvc_stage_is_exploratory_and_has_no_manuscript_outputs() -> None:
    """The comparison has dedicated data outputs and stays outside LaTeX."""
    root = Path(__file__).parents[7]
    document = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))
    stage = document["stages"]["p5_gmv_validation"]
    assert "paper5-gmv-validation" in stage["cmd"]
    assert any(
        "spatial_covariance/covariances.npz" in str(dep) for dep in stage["deps"]
    )
    outputs = " ".join(str(output) for output in stage["outs"])
    assert "gmv_validation" in outputs
    assert "src/latex" not in outputs


def test_cli_adapter_passes_every_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter has no defaults and forwards every option to the stage."""
    runner = CliRunner()
    assert runner.invoke(adapter.commands, []).exit_code == 2  # noqa: PLR2004
    seen: list[tuple[object, object]] = []
    monkeypatch.setattr(
        comparison,
        "run_paper5_gmv_validation",
        lambda paths, config: seen.append((paths, config)),
    )
    for name in ("returns", "oos", "spatial"):
        (tmp_path / name).mkdir()
    for name in ("params.yaml", "pit.npz", "pit.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    config = Paper5GMVValidationConfig()
    args = [
        "paper5-gmv-validation",
        "--params-file", str(tmp_path / "params.yaml"),
        "--returns-dir", str(tmp_path / "returns"),
        "--oos-returns-dir", str(tmp_path / "oos"),
        "--certificate-pit-w2-path", str(tmp_path / "pit.npz"),
        "--certificate-pit-manifest-path", str(tmp_path / "pit.json"),
        "--spatial-covariance-dir", str(tmp_path / "spatial"),
        "--output-dir", str(tmp_path / "out"),
    ]  # fmt: skip
    for field, value in asdict(config).items():
        args += [f"--{field.replace('_', '-')}", str(value)]
    result = runner.invoke(adapter.commands, args)
    assert result.exit_code == 0, result.output
    ((paths, forwarded),) = seen
    assert forwarded == config
    assert paths == Paper5GMVValidationPaths(
        params_file=tmp_path / "params.yaml",
        returns_dir=tmp_path / "returns",
        oos_returns_dir=tmp_path / "oos",
        certificate_pit_w2_path=tmp_path / "pit.npz",
        certificate_pit_manifest_path=tmp_path / "pit.json",
        spatial_covariance_dir=tmp_path / "spatial",
        output_dir=tmp_path / "out",
    )

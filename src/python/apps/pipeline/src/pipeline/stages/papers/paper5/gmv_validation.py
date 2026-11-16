"""Paper 5 GMV validation: retrospective chronological portfolio comparison.

The stage keeps the portfolio comparison outside both manuscripts. It freezes
all portfolio weights at the 2022-12-31 formation vintage using the realized
2018--2022 calibration panel and evaluates constant excess-return mixtures on
the 2023--2026 panel. The comparison is conditional on the frozen
representation and selected universe; it is not a claim of historical
point-in-time implementability.

The spatial covariance artifact is consumed only for calibration covariance
matrices. Evaluation-period covariances are never used to form the primary
weights; they are used only to calculate realized holdout variances.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Never, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from statadic.numerics.optimize.simplex import simplex_qp_result

from pipeline._kernels.paper3.validation import solve_certificate
from pipeline.io.contracts import read_manifest, write_manifest
from pipeline.io.fama_french import load_factor_data_config, load_factor_model
from pipeline.io.params import load_params
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.io.typed_analysis import (
    StatisticalDistanceSpec,
    load_typed_analysis,
)
from pipeline.stages.papers.paper3.certificate import (
    _reference_designs,
    _reference_draws,
)
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pipeline.io.typed_analysis import SampleWindowSpec
    from pipeline.stages.papers.paper3.certificate import _ReferenceDesign

type FloatVector = NDArray[np.float64]
type FloatMatrix = NDArray[np.float64]
type PanelName = Literal["oos_primary", "paper3_in_sample"]

logger = logging.getLogger(__name__)

_BASE_CANDIDATES = (
    "paper3_w2_certificate",
    "equal_weight",
    "inverse_volatility",
    "sample_gmv",
    "ledoit_wolf_gmv",
)
_FACTOR_CANDIDATES = (
    "factor_spherical_qmle",
    "factor_spherical_ledoit_wolf",
    "factor_diagonal_qmle",
    "factor_diagonal_ledoit_wolf",
)
_SPATIAL_CANDIDATES = (
    "spatial_spherical_w1_qmle",
    "spatial_spherical_w1_ledoit_wolf",
    "spatial_spherical_w2_qmle",
    "spatial_spherical_w2_ledoit_wolf",
    "spatial_diagonal_w1_qmle",
    "spatial_diagonal_w1_ledoit_wolf",
    "spatial_diagonal_w2_qmle",
    "spatial_diagonal_w2_ledoit_wolf",
)
_RESIDUAL_CANDIDATES = tuple(f"{name}_residual" for name in _SPATIAL_CANDIDATES)
_CANDIDATES = (
    _BASE_CANDIDATES + _FACTOR_CANDIDATES + _SPATIAL_CANDIDATES + _RESIDUAL_CANDIDATES
)
_PRIMARY_CANDIDATES = _BASE_CANDIDATES + _FACTOR_CANDIDATES + _SPATIAL_CANDIDATES
_TOTAL_COVARIANCE_CANDIDATES = _FACTOR_CANDIDATES + _SPATIAL_CANDIDATES
_SPATIAL_CANDIDATE_KEYS = {
    name: f"calibration_{variance}_{order}_{estimator}_total"
    for variance, order, estimator, name in (
        ("spherical", "w1", "qmle", "spatial_spherical_w1_qmle"),
        ("spherical", "w1", "ledoit_wolf", "spatial_spherical_w1_ledoit_wolf"),
        ("spherical", "w2", "qmle", "spatial_spherical_w2_qmle"),
        ("spherical", "w2", "ledoit_wolf", "spatial_spherical_w2_ledoit_wolf"),
        ("diagonal", "w1", "qmle", "spatial_diagonal_w1_qmle"),
        ("diagonal", "w1", "ledoit_wolf", "spatial_diagonal_w1_ledoit_wolf"),
        ("diagonal", "w2", "qmle", "spatial_diagonal_w2_qmle"),
        ("diagonal", "w2", "ledoit_wolf", "spatial_diagonal_w2_ledoit_wolf"),
    )
}
_SPATIAL_RESIDUAL_KEYS = {
    name: key.replace("_total", "_residual")
    for name, key in _SPATIAL_CANDIDATE_KEYS.items()
}
_FACTOR_CANDIDATE_KEYS = {
    "factor_spherical_qmle": "calibration_spherical_none_qmle_total",
    "factor_spherical_ledoit_wolf": "calibration_spherical_none_ledoit_wolf_total",
    "factor_diagonal_qmle": "calibration_diagonal_none_qmle_total",
    "factor_diagonal_ledoit_wolf": "calibration_diagonal_none_ledoit_wolf_total",
}
_FACTOR_RESIDUAL_KEYS = {
    name: key.replace("_total", "_residual")
    for name, key in _FACTOR_CANDIDATE_KEYS.items()
}
_ALL_TOTAL_KEYS = {**_FACTOR_CANDIDATE_KEYS, **_SPATIAL_CANDIDATE_KEYS}
_ALL_RESIDUAL_KEYS = {**_FACTOR_RESIDUAL_KEYS, **_SPATIAL_RESIDUAL_KEYS}
_FORMATION_RULES = {
    "paper3_w2_certificate": "normalized_w2_certificate_capital_map",
    "equal_weight": "equal_weight",
    "inverse_volatility": "inverse_volatility",
    "sample_gmv": "total_excess_return_sample_covariance_gmv",
    "ledoit_wolf_gmv": "total_excess_return_scaled_identity_ledoit_wolf_gmv",
    **dict.fromkeys(_TOTAL_COVARIANCE_CANDIDATES, "total_excess_return_covariance_gmv"),
    **dict.fromkeys(_RESIDUAL_CANDIDATES, "residual_covariance_gmv"),
}
_FORMATION_OBJECTIVE_KINDS = {
    "paper3_w2_certificate": "w2_certificate",
    "equal_weight": "rule_based",
    "inverse_volatility": "rule_based",
    "sample_gmv": "total_excess_return_covariance",
    "ledoit_wolf_gmv": "total_excess_return_covariance",
    **dict.fromkeys(_TOTAL_COVARIANCE_CANDIDATES, "total_excess_return_covariance"),
    **dict.fromkeys(_RESIDUAL_CANDIDATES, "residual_covariance"),
}
_EVALUATION_RISK = "total_excess_return_variance"
_PRIMARY_WEIGHT_COORDINATES = "capital"
_SECONDARY_WEIGHT_COORDINATES = "normalized_standardized_asset"
_PRIMARY_RETURN_SCALING = "unstandardized_excess_returns"
_SECONDARY_RETURN_SCALING = "calibration_volatility_standardized_excess_returns"
_WEIGHT_SUM_TOLERANCE = 1e-8
_PSD_RELATIVE_TOLERANCE = 1e-10
_GMV_GAP_TOLERANCE = 1e-8
_MIN_FACTOR_COMPLETE_DATES = 30
_MIN_REFERENCE_DRAWS = 1_000
_TRADING_DAYS = 252.0
_MATRIX_DIMENSIONS = 2
_EXPECTED_ROSTER_SIZE = 100


@dataclass(frozen=True, slots=True)
class Paper5GMVValidationPaths:
    """Input and output boundaries for the exploratory comparison."""

    params_file: Path
    returns_dir: Path
    oos_returns_dir: Path
    certificate_pit_w2_path: Path
    certificate_pit_manifest_path: Path
    spatial_covariance_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5GMVValidationConfig:
    """Frozen timing, cap, and reference-population contract."""

    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    maximum_weight: float = 0.15
    reference_draws: int = 20_000
    reference_seed: int = 42
    reference_design_label: str = "uniform_cap_15"
    formation_vintage: str = "2022-12-31"
    risk_unit: Literal["excess_return"] = "excess_return"
    factor_model: Literal["ff5_mom"] = "ff5_mom"
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    w1_barycentre_arm_id: str = "wasserstein_w1_loo"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w1_geometry_id: str = "wasserstein_w1"
    w2_geometry_id: str = "wasserstein_w2"
    residual_shrinkage_target: Literal["constant_correlation"] = "constant_correlation"

    def validate(self) -> None:
        """Reject malformed timing and reference settings before loading data."""
        dates = tuple(
            pd.Timestamp(value)
            for value in (
                self.calibration_start,
                self.calibration_end,
                self.evaluation_start,
                self.evaluation_end,
            )
        )
        if not dates[0] <= dates[1] < dates[2] <= dates[3]:
            _invalid("calibration and evaluation windows must be ordered")
        if not 0.0 < self.maximum_weight <= 1.0:
            _invalid("maximum_weight must lie in (0, 1]")
        if self.reference_draws < _MIN_REFERENCE_DRAWS:
            _invalid(f"reference_draws must be at least {_MIN_REFERENCE_DRAWS}")
        if self.reference_seed < 0:
            _invalid("reference_seed must be nonnegative")
        if pd.Timestamp(self.formation_vintage) != pd.Timestamp(self.calibration_end):
            _invalid("formation_vintage must equal the calibration end date")
        if self.risk_unit != "excess_return":
            _invalid("the comparison uses excess returns as its common risk unit")


@dataclass(frozen=True, slots=True)
class _PitCertificateIdentity:
    """Resolved typed identity for the selected PIT W2 certificate vintage."""

    formation_vintage: str
    provider_id: str
    representation_id: str
    distance_id: str
    family: str
    ground_distance: str
    estimator: str
    normalization: str
    sample_design: str
    sample_size: int | None
    sample_window_id: str
    window_start: str | None
    window_end: str
    source_scale: str
    archive_scale: str


def _invalid(message: str) -> Never:
    raise ValueError(message)


def _symmetrize(matrix: FloatMatrix) -> FloatMatrix:
    """Return a float64 symmetric matrix."""
    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim != _MATRIX_DIMENSIONS or result.shape[0] != result.shape[1]:
        _invalid("covariance must be a square matrix")
    return np.asarray(0.5 * (result + result.T), dtype=np.float64)


def _validate_covariance(matrix: FloatMatrix, n_assets: int, label: str) -> FloatMatrix:
    """Validate a covariance matrix before it enters an optimizer."""
    result = _symmetrize(matrix)
    if result.shape != (n_assets, n_assets) or not np.isfinite(result).all():
        _invalid(f"{label} covariance has the wrong shape or nonfinite entries")
    eigenvalues = np.linalg.eigvalsh(result)
    scale = max(float(np.max(np.abs(result))), np.finfo(np.float64).tiny)
    if eigenvalues[0] < -_PSD_RELATIVE_TOLERANCE * scale:
        _invalid(f"{label} covariance is not positive semidefinite")
    return result


def _validate_weights(weights: FloatVector, n_assets: int, label: str) -> FloatVector:
    """Validate a finite long-only simplex allocation."""
    result = np.asarray(weights, dtype=np.float64)
    if (
        result.ndim != 1
        or result.shape[0] != n_assets
        or not np.isfinite(result).all()
        or np.any(result < -_WEIGHT_SUM_TOLERANCE)
        or abs(float(result.sum()) - 1.0) > _WEIGHT_SUM_TOLERANCE
    ):
        _invalid(f"{label} weights must be finite, nonnegative, and sum to one")
    result = np.maximum(result, 0.0)
    result /= result.sum()
    return result


def _capped_gmv(
    covariance: FloatMatrix, maximum_weight: float, label: str
) -> FloatVector:
    """Solve and certify a long-only capped minimum-variance problem."""
    n_assets = covariance.shape[0]
    covariance64 = _validate_covariance(covariance, n_assets, label)
    if maximum_weight * n_assets < 1.0 - _WEIGHT_SUM_TOLERANCE:
        _invalid(f"{label} cap is infeasible for the simplex")
    initial = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    if maximum_weight <= 1.0 / n_assets + _WEIGHT_SUM_TOLERANCE:
        return initial
    if not np.any(covariance64):
        return initial
    objective_scale = float(np.linalg.norm(covariance64, ord=np.inf))
    objective_scale = max(objective_scale, np.finfo(np.float64).tiny)
    quadratic = 2.0 * jnp.asarray(covariance64) / objective_scale
    result = simplex_qp_result(
        quadratic,
        jnp.zeros(n_assets, dtype=quadratic.dtype),
        maximum_weight=maximum_weight,
        tolerance=1e-10,
        max_steps=5_000,
    )
    raw_weights = np.asarray(result.solver.value)
    if not bool(result.solver.converged) or not np.isfinite(raw_weights).all():
        _invalid(
            f"{label} capped GMV solver failed: status {int(result.solver.status)}"
        )
    weights = _validate_weights(raw_weights, n_assets, label)
    if weights.max() > maximum_weight + _WEIGHT_SUM_TOLERANCE:
        _invalid(f"{label} solver returned a weight above the cap")
    gradient = 2.0 * (covariance64 @ weights) / objective_scale
    oracle = _capped_simplex_linear_oracle(gradient, maximum_weight)
    gap = float(gradient @ (weights - oracle))
    gap_tolerance = _GMV_GAP_TOLERANCE * max(
        1.0, float(np.linalg.norm(gradient, ord=np.inf))
    )
    if gap < -gap_tolerance or gap > gap_tolerance:
        _invalid(f"{label} capped GMV failed convex optimality gap: {gap:.3e}")
    return weights


def _capped_simplex_linear_oracle(
    gradient: FloatVector, maximum_weight: float
) -> FloatVector:
    """Minimize a linear objective over the capped simplex."""
    values = np.asarray(gradient, dtype=np.float64)
    weights = np.zeros(values.shape[0], dtype=np.float64)
    remaining = 1.0
    for index in np.argsort(values, kind="stable"):
        allocation = min(maximum_weight, remaining)
        weights[index] = allocation
        remaining -= allocation
        if remaining <= _WEIGHT_SUM_TOLERANCE:
            break
    if remaining > _WEIGHT_SUM_TOLERANCE:
        _invalid("capped simplex linear oracle could not fill unit mass")
    return weights


def _gmv_diagnostics(
    covariance: FloatMatrix, weights: FloatVector, maximum_weight: float
) -> dict[str, float]:
    """Persist scale and independent convex-gap diagnostics for a GMV solve."""
    covariance64 = _validate_covariance(covariance, len(weights), "GMV diagnostic")
    scale = max(
        float(np.linalg.norm(covariance64, ord=np.inf)), np.finfo(np.float64).tiny
    )
    gradient = 2.0 * (covariance64 @ weights) / scale
    oracle = _capped_simplex_linear_oracle(gradient, maximum_weight)
    return {
        "objective_scale": scale,
        "normalized_objective": float(weights @ covariance64 @ weights / scale),
        "convex_first_order_gap": float(gradient @ (weights - oracle)),
        "simplex_residual": abs(float(weights.sum()) - 1.0),
        "cap_residual": max(0.0, float(weights.max()) - maximum_weight),
    }


def _standardize(returns: FloatMatrix) -> tuple[FloatMatrix, FloatVector]:
    """Standardize a panel by its calibration-period sample volatility."""
    panel = np.asarray(returns, dtype=np.float64)
    volatility = np.std(panel, axis=0, ddof=1)
    if not np.isfinite(volatility).all() or np.any(volatility <= 0.0):
        _invalid("every asset must have finite positive calibration volatility")
    standardized = panel / volatility
    if not np.isfinite(standardized).all():
        _invalid("standardized returns are nonfinite")
    return standardized, np.asarray(volatility, dtype=np.float64)


def _capital_from_budget(budget: FloatVector, volatility: FloatVector) -> FloatVector:
    """Map Paper 3 normalized risk weights to capital weights."""
    result = np.asarray(budget / volatility, dtype=np.float64)
    result /= result.sum()
    return result


def _certificate_gap(matrix: FloatMatrix, weights: FloatVector) -> float:
    """Return the capped-simplex linear gap for certificate maximization."""
    matrix64 = np.asarray(matrix, dtype=np.float64)
    scale = max(float(np.linalg.norm(matrix64, ord=np.inf)), np.finfo(np.float64).tiny)
    gradient = -(matrix64 @ weights) / scale
    oracle = _capped_simplex_linear_oracle(gradient, 1.0)
    return float(gradient @ (weights - oracle))


def _validate_manifest_output_hash(
    manifest: dict[str, object], path: Path, label: str
) -> None:
    """Require the supplied artifact bytes to match its producer manifest."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        _invalid(f"{label} manifest has no output records")
    matches = [
        record
        for record in outputs
        if isinstance(record, dict)
        and Path(str(record.get("path", ""))).name == path.name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != _sha256_path(path):
        _invalid(f"{label} bytes do not match the producer manifest")


def _resolve_pit_certificate_identity(
    params_file: Path,
    config: Paper5GMVValidationConfig,
) -> _PitCertificateIdentity:
    """Resolve the selected PIT vintage from the typed-analysis registry."""
    distance, window = _resolve_pit_distance_and_window(params_file, config)
    window_end = window.end
    if window_end is None:
        _invalid("selected PIT window has no end date")
    return _PitCertificateIdentity(
        formation_vintage=config.formation_vintage,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        distance_id=distance.distance_id,
        family=distance.family,
        ground_distance=distance.ground_distance,
        estimator=distance.estimator,
        normalization=distance.normalization,
        sample_design=distance.sample_design,
        sample_size=distance.sample_size,
        sample_window_id=window.window_id,
        window_start=(window.start.isoformat() if window.start is not None else None),
        window_end=window_end.isoformat(),
        source_scale="rooted_wasserstein_2",
        archive_scale="squared_wasserstein_2_transport_cost",
    )


def _pit_distance_id(params_file: Path, formation_vintage: str) -> str:
    """Resolve the one configured PIT distance for a formation vintage."""
    params = load_params(params_file)
    paper3 = params.get("paper3")
    if not isinstance(paper3, dict):
        _invalid("params.yaml paper3 section must be a mapping")
    grid = paper3.get("pit_vintage_grid")
    if not isinstance(grid, list):
        _invalid("params.yaml paper3 PIT vintage grid must be a list")
    matches = [
        entry
        for entry in grid
        if isinstance(entry, dict) and str(entry.get("vintage")) == formation_vintage
    ]
    if len(matches) != 1:
        _invalid(
            "params.yaml PIT vintage grid must contain exactly one requested "
            f"formation vintage {formation_vintage}"
        )
    distance_id = matches[0].get("distance_id")
    if not isinstance(distance_id, str) or not distance_id:
        _invalid("params.yaml PIT vintage entry has no distance identity")
    return distance_id


def _resolve_pit_distance_and_window(
    params_file: Path,
    config: Paper5GMVValidationConfig,
) -> tuple[StatisticalDistanceSpec, SampleWindowSpec]:
    """Resolve and validate the selected PIT distance and its sample window."""
    distance_id = _pit_distance_id(params_file, config.formation_vintage)
    typed = load_typed_analysis(params_file)
    distance = typed.distances.get(distance_id)
    if not isinstance(distance, StatisticalDistanceSpec):
        _invalid(f"PIT vintage {distance_id!r} is not a statistical distance")
    window = typed.window_for(distance)
    if window is None or distance.sample_window is None:
        _invalid(f"PIT vintage {distance_id!r} has no closed sample window")
    window_end = window.end
    if window_end is None:
        _invalid(f"PIT vintage {distance_id!r} has no closed sample window")
    base_geometry = typed.distances.get(config.w2_geometry_id)
    if not isinstance(base_geometry, StatisticalDistanceSpec):
        _invalid(f"configured W2 geometry {config.w2_geometry_id!r} is not registered")
    for field in (
        "family",
        "ground_distance",
        "exponent",
        "estimator",
        "normalization",
    ):
        if getattr(distance, field) != getattr(base_geometry, field):
            _invalid(
                f"PIT vintage {distance_id!r} disagrees with configured W2 "
                f"geometry on {field}"
            )
    if distance.sample_design != "point_in_time":
        _invalid(f"PIT vintage {distance_id!r} is not point-in-time data")
    if window.window_id != distance.sample_window:
        _invalid(f"PIT vintage {distance_id!r} disagrees with its sample window")
    if pd.Timestamp(window_end) != pd.Timestamp(config.formation_vintage):
        _invalid("selected PIT window end does not equal the formation vintage")
    return distance, window


def _pit_expected_fields(expected: _PitCertificateIdentity) -> dict[str, object]:
    """Return the exact typed fields required for one PIT vintage."""
    return {
        "provider_id": expected.provider_id,
        "representation_id": expected.representation_id,
        "distance_id": expected.distance_id,
        "family": expected.family,
        "ground_distance": expected.ground_distance,
        "estimator": expected.estimator,
        "normalization": expected.normalization,
        "sample_design": expected.sample_design,
        "sample_size": expected.sample_size,
        "sample_window_id": expected.sample_window_id,
        "window_start": expected.window_start,
        "window_end": expected.window_end,
        "source_scale": expected.source_scale,
        "archive_scale": expected.archive_scale,
    }


def _validate_pit_manifest_metadata(
    metadata: dict[str, object],
    archive_keys: set[str],
    tickers: list[str],
    expected: _PitCertificateIdentity,
) -> dict[str, object]:
    """Validate PIT scales, roster, vintage keys, and selected formation key."""
    if metadata.get("source_scale") != "rooted_wasserstein_2":
        _invalid("PIT W2 source scale must be rooted Wasserstein-2")
    if metadata.get("archive_scale") != "squared_wasserstein_2_transport_cost":
        _invalid("PIT W2 archive must already contain squared W2 transport costs")
    item_ids = metadata.get("item_ids")
    if not isinstance(item_ids, list) or [str(value) for value in item_ids] != tickers:
        _invalid("PIT W2 roster must exactly match the return roster")
    windows = metadata.get("vintage_windows")
    if not isinstance(windows, dict) or not windows:
        _invalid("PIT W2 manifest declares no vintage windows")
    if {str(key) for key in windows} != archive_keys:
        _invalid("PIT W2 archive keys do not match its manifest windows")
    if expected.formation_vintage not in windows:
        _invalid(
            "PIT W2 archive has no requested formation vintage "
            f"{expected.formation_vintage}"
        )
    return windows


def _validate_pit_vintage_metadata(
    metadata: dict[str, object],
    windows: dict[str, object],
    archive_keys: set[str],
    expected: _PitCertificateIdentity,
) -> None:
    """Validate the exact typed identity of the selected PIT vintage."""
    if metadata.get("vintage_identity_schema") != "pit_vintage_identity.v1":
        _invalid("PIT W2 manifest has no typed vintage identity schema")
    identities = metadata.get("vintage_identities")
    if not isinstance(identities, dict):
        _invalid("PIT W2 manifest has no typed vintage identities")
    if {str(key) for key in identities} != archive_keys:
        _invalid("PIT W2 manifest vintage identities do not match archive keys")
    selected = identities.get(expected.formation_vintage)
    if not isinstance(selected, dict):
        _invalid("PIT W2 manifest has no selected vintage identity")
    for field, value in _pit_expected_fields(expected).items():
        if selected.get(field) != value:
            _invalid(f"PIT W2 selected identity {field} does not match the registry")
    if windows.get(expected.formation_vintage) != expected.sample_window_id:
        _invalid("PIT W2 selected window ID does not match the typed registry")


def _validate_pit_upstream_identity(
    manifest: dict[str, object], expected: _PitCertificateIdentity
) -> None:
    """Validate the selected PIT source record and its typed path identity."""
    upstream = manifest.get("upstream_artifacts")
    if not isinstance(upstream, list):
        _invalid("PIT W2 manifest has no upstream artifact records")
    selected = [
        record
        for record in upstream
        if isinstance(record, dict)
        and record.get("vintage") == expected.formation_vintage
    ]
    if len(selected) != 1:
        _invalid("PIT W2 manifest has no unique selected upstream record")
    record = selected[0]
    expected_identity = (
        expected.distance_id,
        expected.provider_id,
        expected.representation_id,
    )
    actual_identity = (
        record.get("distance_id"),
        record.get("provider_id"),
        record.get("representation_id"),
    )
    if actual_identity != expected_identity:
        _invalid("PIT W2 selected upstream identity does not match the registry")
    if Path(str(record.get("value", ""))).parts[-3:] != (
        expected.provider_id,
        expected.representation_id,
        expected.distance_id,
    ):
        _invalid("PIT W2 selected upstream path does not match the typed identity")


def _validate_pit_certificate_contract(
    manifest: dict[str, object],
    archive_keys: set[str],
    tickers: list[str],
    expected: _PitCertificateIdentity,
) -> None:
    """Validate PIT roster, typed identity, scale, and requested vintage."""
    if manifest.get("stage_key") != "p3_pit_w2":
        _invalid("certificate input must be produced by the Paper 3 PIT W2 stage")
    if manifest.get("protocol_id") != "paper3.pit-w2.v2":
        _invalid("certificate input must use PIT W2 protocol v2")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        _invalid("PIT W2 manifest is missing metadata")
    windows = _validate_pit_manifest_metadata(
        cast("dict[str, object]", metadata), archive_keys, tickers, expected
    )
    _validate_pit_vintage_metadata(
        cast("dict[str, object]", metadata), windows, archive_keys, expected
    )
    _validate_pit_upstream_identity(manifest, expected)


def _load_certificate_budget(
    path: Path,
    manifest_path: Path,
    params_file: Path,
    tickers: list[str],
    config: Paper5GMVValidationConfig,
) -> tuple[FloatVector, dict[str, object]]:
    """Solve the verified, already-squared PIT W2 certificate at formation."""
    manifest = read_manifest(manifest_path)
    _validate_manifest_output_hash(manifest, path, "PIT W2 archive")
    with np.load(path, allow_pickle=False) as archive:
        archive_keys = {str(key) for key in archive.files}
        archive_keys.discard("tickers")
        expected = _resolve_pit_certificate_identity(params_file, config)
        _validate_pit_certificate_contract(manifest, archive_keys, tickers, expected)
        matrix = np.asarray(archive[config.formation_vintage], dtype=np.float64)
    if matrix.shape != (len(tickers), len(tickers)) or not np.isfinite(matrix).all():
        _invalid("selected PIT W2 matrix has the wrong shape or nonfinite entries")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        _invalid("selected PIT W2 matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
        _invalid("selected PIT W2 matrix must have zero diagonal")
    n_assets = len(tickers)
    centering = np.eye(n_assets) - np.full((n_assets, n_assets), 1.0 / n_assets)
    centered = -0.5 * centering @ matrix @ centering
    centered_eigenvalues = np.linalg.eigvalsh(centered)
    scale = max(float(np.max(np.abs(centered))), np.finfo(np.float64).tiny)
    if centered_eigenvalues[0] < -_PSD_RELATIVE_TOLERANCE * scale:
        _invalid(
            "selected PIT W2 matrix fails the centered certificate convexity check"
        )
    try:
        solution = solve_certificate(matrix, cap=1.0)
    except (TypeError, ValueError) as error:
        _invalid(f"selected PIT W2 certificate solve failed: {error}")
    weights = _validate_weights(
        np.asarray(solution.weights, dtype=np.float64),
        n_assets,
        "paper3 PIT W2 certificate",
    )
    gap = _certificate_gap(matrix, weights)
    if gap > _GMV_GAP_TOLERANCE:
        _invalid(
            f"selected PIT W2 certificate failed its simplex optimality gap: {gap:.3e}"
        )
    return weights, {
        "formation_vintage": config.formation_vintage,
        "archive_scale": "squared_wasserstein_2_transport_cost",
        "objective": "0.5 q' M q",
        "centered_kernel_minimum_eigenvalue": float(centered_eigenvalues[0]),
        "simplex_first_order_gap": gap,
        "solver_kkt_residual": float(solution.kkt_residual),
        "cap": 1.0,
    }


def _validate_spatial_parameters(
    parameters: dict[str, object], config: Paper5GMVValidationConfig
) -> None:
    """Require Paper 5 formation parameters to match the portfolio protocol."""
    expected = {
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
    for field, value in expected.items():
        if parameters.get(field) != value:
            _invalid(
                f"spatial covariance parameter {field} does not match "
                "the portfolio protocol"
            )


def _spatial_artifact_sections(
    summary: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Extract the required typed sections from a Paper 5 summary."""
    sample = summary.get("sample")
    parameters = summary.get("parameters")
    estimator = summary.get("estimator")
    if (
        not isinstance(sample, dict)
        or not isinstance(parameters, dict)
        or not isinstance(estimator, dict)
    ):
        _invalid(
            "spatial covariance summary is missing sample, parameter, or "
            "estimator metadata"
        )
    return (
        cast("dict[str, object]", sample),
        cast("dict[str, object]", parameters),
        cast("dict[str, object]", estimator),
    )


def _validate_spatial_estimator(
    estimator: dict[str, object],
    config: Paper5GMVValidationConfig,
) -> None:
    """Require the declared residual estimator to match the protocol."""
    expected = {
        "risk_unit": config.risk_unit,
        "residual_shrinkage_target": config.residual_shrinkage_target,
        "implementation": "statadic.estimation.covariance.LedoitWolf",
        "input": "factor-residualized spatial innovations",
        "fit_mode": "two-step plug-in: fit rho/QMLE, then shrink fitted innovations",
        "reduced_form_loadings": "B=A^-1 C_f",
    }
    for field, value in expected.items():
        if estimator.get(field) != value:
            _invalid(f"spatial covariance estimator field {field} is not declared")


def _validate_spatial_artifact_header(
    summary: dict[str, object],
    manifest: dict[str, object],
    tickers: list[str],
    config: Paper5GMVValidationConfig,
) -> dict[str, object]:
    """Validate Paper 5 covariance identity and return its sample metadata."""
    if summary.get("schema_version") != "paper5_spatial_covariance.v2":
        _invalid("spatial covariance artifact must use schema v2")
    if summary.get("analysis_id") != "paper5-spatial-covariance-v2":
        _invalid("spatial covariance artifact has an unexpected analysis identity")
    if manifest.get("stage_key") != "p5_spatial_covariance":
        _invalid("spatial covariance manifest has an unexpected stage identity")
    if manifest.get("protocol_id") != "paper5.spatial_covariance.ledoit_wolf.v2":
        _invalid("spatial covariance manifest has an unexpected protocol identity")
    sample, parameters, estimator = _spatial_artifact_sections(summary)
    raw_tickers = sample.get("tickers")
    if (
        not isinstance(raw_tickers, list)
        or [str(value) for value in raw_tickers] != tickers
    ):
        _invalid(
            "spatial covariance summary and return panels use different ticker order"
        )
    for field in ("calibration_date_sha256", "evaluation_date_sha256"):
        value = sample.get(field)
        if not isinstance(value, str) or not value:
            _invalid(f"spatial covariance summary is missing {field}")
    _validate_spatial_estimator(estimator, config)
    _validate_spatial_parameters(parameters, config)
    return sample


def _validate_spatial_artifact_dates(
    sample: dict[str, object], config: Paper5GMVValidationConfig
) -> None:
    """Require producer dates to lie within the requested disjoint windows."""
    actual_dates = tuple(
        pd.Timestamp(str(sample[field]))
        for field in (
            "calibration_start",
            "calibration_end",
            "evaluation_start",
            "evaluation_end",
        )
    )
    requested_dates = tuple(
        pd.Timestamp(value)
        for value in (
            config.calibration_start,
            config.calibration_end,
            config.evaluation_start,
            config.evaluation_end,
        )
    )
    if not (
        requested_dates[0]
        <= actual_dates[0]
        <= actual_dates[1]
        <= requested_dates[1]
        < requested_dates[2]
        <= actual_dates[2]
        <= actual_dates[3]
        <= requested_dates[3]
    ):
        _invalid(
            "spatial covariance artifact dates do not fit the requested "
            "disjoint windows"
        )


def _validate_spatial_artifact_protocol(
    summary: dict[str, object],
    manifest: dict[str, object],
    tickers: list[str],
    config: Paper5GMVValidationConfig,
) -> None:
    """Reject stale, mismatched, or undated Paper 5 covariance artifacts."""
    sample = _validate_spatial_artifact_header(summary, manifest, tickers, config)
    _validate_spatial_artifact_dates(sample, config)


def _validate_spatial_date_identity(
    sample: dict[str, object],
    calibration_dates: np.ndarray,
    evaluation_dates: np.ndarray,
) -> None:
    """Require Paper 5 and Paper 3 to use identical realized date panels."""
    expected = {
        "calibration_date_sha256": _date_digest(calibration_dates),
        "evaluation_date_sha256": _date_digest(evaluation_dates),
    }
    for field, digest in expected.items():
        if sample.get(field) != digest:
            _invalid(
                f"spatial covariance {field} does not match the portfolio "
                "factor-complete return dates"
            )


def _load_spatial_covariances(
    path: Path, tickers: list[str], config: Paper5GMVValidationConfig
) -> tuple[dict[str, FloatMatrix], dict[str, FloatMatrix], dict[str, object]]:
    """Read and validate dated Paper 5 total and residual covariance cells."""
    summary_path = path / "results.json"
    manifest_path = path / "provenance.manifest.json"
    summary = cast(
        "dict[str, object]", json.loads(summary_path.read_text(encoding="utf-8"))
    )
    manifest = read_manifest(manifest_path)
    _validate_manifest_output_hash(
        manifest, path / "covariances.npz", "spatial covariance"
    )
    _validate_manifest_output_hash(manifest, summary_path, "spatial covariance summary")
    _validate_spatial_artifact_protocol(summary, manifest, tickers, config)
    covariance_path = path / "covariances.npz"
    with np.load(covariance_path, allow_pickle=False) as artifact:
        artifact_tickers = [str(value) for value in artifact["tickers"]]
        if artifact_tickers != tickers:
            _invalid("spatial covariance and return panels use different ticker order")
        total = {
            name: _validate_covariance(
                np.asarray(artifact[key], dtype=np.float64), len(tickers), name
            )
            for name, key in _ALL_TOTAL_KEYS.items()
        }
        residual = {
            name: _validate_covariance(
                np.asarray(artifact[key], dtype=np.float64), len(tickers), name
            )
            for name, key in _ALL_RESIDUAL_KEYS.items()
        }
    return total, residual, summary


def _candidate_weights(
    returns: FloatMatrix,
    certificate_budget: FloatVector,
    model_covariances: dict[str, FloatMatrix],
    maximum_weight: float,
    *,
    capital: bool,
) -> dict[str, FloatVector]:
    """Build one complete benchmark set using only the supplied panel."""
    n_assets = returns.shape[1]
    volatility = np.std(returns, axis=0, ddof=1)
    if not np.isfinite(volatility).all() or np.any(volatility <= 0.0):
        _invalid("every candidate-formation asset must have positive volatility")
    equal = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    inverse = np.reciprocal(volatility)
    inverse /= inverse.sum()
    sample = np.cov(returns, rowvar=False, ddof=1)
    ledoit_wolf = LedoitWolf().fit(returns).covariance_
    certificate = (
        _capital_from_budget(certificate_budget, volatility)
        if capital
        else certificate_budget.copy()
    )
    weights: dict[str, FloatVector] = {
        "paper3_w2_certificate": _validate_weights(
            certificate, n_assets, "paper3 W2 certificate"
        ),
        "equal_weight": equal,
        "inverse_volatility": _validate_weights(
            inverse, n_assets, "inverse-volatility"
        ),
        "sample_gmv": _capped_gmv(
            np.asarray(sample, dtype=np.float64), maximum_weight, "sample GMV"
        ),
        "ledoit_wolf_gmv": _capped_gmv(
            np.asarray(ledoit_wolf, dtype=np.float64),
            maximum_weight,
            "Ledoit--Wolf GMV",
        ),
    }
    for name in _TOTAL_COVARIANCE_CANDIDATES:
        weights[name] = _capped_gmv(model_covariances[name], maximum_weight, name)
    if tuple(weights) != _PRIMARY_CANDIDATES:
        _invalid("candidate benchmark order drifted")
    return weights


def _reference_rows(
    covariance: FloatMatrix,
    candidates: dict[str, FloatVector],
    designs: tuple[_ReferenceDesign, ...],
    draws: int,
    seed: int,
    *,
    panel: PanelName,
    seed_offset: int = 0,
) -> list[dict[str, object]]:
    """Rank candidates using Paper 3's exact capped Dirichlet convention."""
    rows: list[dict[str, object]] = []
    for index, design in enumerate(designs):
        sample = _reference_draws(
            design, covariance.shape[0], draws, seed + seed_offset + index
        )
        reference_variances = np.einsum("ij,jk,ik->i", sample, covariance, sample)
        if not np.isfinite(reference_variances).all():
            _invalid(f"{design.label} reference variances are nonfinite")
        for name in _CANDIDATES:
            weights = candidates[name]
            feasible = weights.max() <= design.maximum_weight + _WEIGHT_SUM_TOLERANCE
            candidate_variance = float(weights @ covariance @ weights)
            rows.append(
                {
                    "panel": panel,
                    "candidate": name,
                    "reference_design": design.label,
                    "reference_concentration": design.concentration,
                    "reference_maximum_weight": design.maximum_weight,
                    "reference_draws": len(sample),
                    "candidate_status": ("computed" if feasible else "infeasible_cap"),
                    "percentile": (
                        float(np.mean(reference_variances <= candidate_variance))
                        if feasible
                        else None
                    ),
                }
            )
    return rows


def _candidate_contract(candidate: str, panel: PanelName) -> dict[str, str]:
    """Describe candidate formation and evaluation for one panel."""
    try:
        formation_rule = _FORMATION_RULES[candidate]
        formation_objective_kind = _FORMATION_OBJECTIVE_KINDS[candidate]
    except KeyError:
        _invalid(f"candidate {candidate!r} has no formation contract")
    if panel == "paper3_in_sample":
        if candidate == "paper3_w2_certificate":
            formation_rule = "normalized_w2_certificate"
        elif candidate in {"sample_gmv", "ledoit_wolf_gmv"}:
            formation_rule = formation_rule.replace(
                "total_excess_return", "standardized_excess_return"
            )
        elif candidate in _TOTAL_COVARIANCE_CANDIDATES:
            formation_rule = "standardized_excess_return_covariance_gmv"
        elif candidate in _RESIDUAL_CANDIDATES:
            formation_rule = "standardized_residual_covariance_gmv"
        if formation_objective_kind == "total_excess_return_covariance":
            formation_objective_kind = "standardized_excess_return_covariance"
        elif formation_objective_kind == "residual_covariance":
            formation_objective_kind = "standardized_residual_covariance"
    is_primary = panel == "oos_primary"
    return {
        "formation_rule": formation_rule,
        "formation_objective_kind": formation_objective_kind,
        "evaluation_risk": _EVALUATION_RISK,
        "weight_coordinates": (
            _PRIMARY_WEIGHT_COORDINATES if is_primary else _SECONDARY_WEIGHT_COORDINATES
        ),
        "formation_input_scaling": (
            _PRIMARY_RETURN_SCALING if is_primary else _SECONDARY_RETURN_SCALING
        ),
        "evaluation_input_scaling": (
            _PRIMARY_RETURN_SCALING if is_primary else _SECONDARY_RETURN_SCALING
        ),
    }


def _variance_rows(
    panel: PanelName,
    returns: FloatMatrix,
    candidates: dict[str, FloatVector],
    reference_rows: list[dict[str, object]],
    *,
    formation_covariance: FloatMatrix | None = None,
    objective_covariances: dict[str, FloatMatrix] | None = None,
    objective_diagnostics: dict[str, dict[str, float]] | None = None,
    maximum_weight: float,
) -> list[dict[str, object]]:
    """Attach realized excess-risk and optimizer diagnostics.

    The realized panel uses constant weights, equivalent to a costless
    rebalanced mixture in a trading interpretation.  It is not a holdings
    drift or transaction-cost backtest.
    """
    covariance = _validate_covariance(
        np.cov(returns, rowvar=False, ddof=1), returns.shape[1], f"{panel} sample"
    )
    equal_variance = float(
        candidates["equal_weight"] @ covariance @ candidates["equal_weight"]
    )
    if equal_variance <= 0.0:
        _invalid(f"{panel} equal-weight variance must be positive")
    formation_equal_variance = (
        float(
            candidates["equal_weight"]
            @ formation_covariance
            @ candidates["equal_weight"]
        )
        if formation_covariance is not None
        else None
    )
    if formation_equal_variance is not None and formation_equal_variance <= 0.0:
        _invalid(f"{panel} formation equal-weight variance must be positive")
    rows: list[dict[str, object]] = []
    for reference in reference_rows:
        candidate = str(reference["candidate"])
        weights = candidates[candidate]
        variance = float(weights @ covariance @ weights)
        formation_variance = (
            float(weights @ formation_covariance @ weights)
            if formation_covariance is not None
            else None
        )
        objective_covariance = (
            objective_covariances.get(candidate)
            if objective_covariances is not None
            else None
        )
        objective_variance = (
            float(weights @ objective_covariance @ weights)
            if objective_covariance is not None
            else None
        )
        role = (
            "residual_gmv_diagnostic"
            if candidate in _RESIDUAL_CANDIDATES
            else "total_risk_primary"
        )
        candidate_contract = _candidate_contract(candidate, panel)
        rows.append(
            {
                **reference,
                "portfolio_role": role,
                "risk_unit": "excess_return",
                **candidate_contract,
                # Retain the old column as an accurate formation-objective
                # alias for downstream readers of the v2 artifact.
                "objective_kind": candidate_contract["formation_objective_kind"],
                "implementation_convention": "constant_weight_mixture",
                "cap_policy": (
                    "excluded_from_common_cap_ranking"
                    if weights.max() > maximum_weight + _WEIGHT_SUM_TOLERANCE
                    else "included_in_common_cap_ranking"
                ),
                "variance": variance,
                "formation_variance": formation_variance,
                "formation_objective_variance": objective_variance,
                "formation_objective_diagnostics": (
                    objective_diagnostics.get(candidate)
                    if objective_diagnostics is not None
                    else None
                ),
                "formation_relative_to_equal_pct": (
                    100.0 * formation_variance / formation_equal_variance
                    if formation_variance is not None
                    and formation_equal_variance is not None
                    else None
                ),
                "relative_to_equal_pct": 100.0 * variance / equal_variance,
                "variance_reduction_vs_equal_pct": (
                    100.0 * (1.0 - variance / equal_variance)
                ),
                "maximum_weight": float(weights.max()),
                "effective_number_assets": float(1.0 / (weights @ weights)),
                "annualized_volatility": (
                    float(np.sqrt(variance * _TRADING_DAYS))
                    if panel == "oos_primary"
                    else None
                ),
            }
        )
    return rows


def _output_record(path: Path, root: Path, kind: str) -> dict[str, object]:
    """Describe one output for the contract manifest."""
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }


def _sha256_path(path: Path) -> str:
    """Hash a file or directory with deterministic path framing."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_path(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _date_digest(dates: np.ndarray) -> str:
    """Hash the ordered factor-complete dates used in the comparison."""
    values = np.asarray(dates, dtype="datetime64[ns]").astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _stage_source_digest() -> str:
    return entrypoint_code_identity("paper5-gmv-validation")


def _parameter_identity(params_file: Path, config: Paper5GMVValidationConfig) -> str:
    """Hash the invoked and registry values consumed by this comparison."""
    params = load_params(params_file)
    selected_distance, window = _resolve_pit_distance_and_window(params_file, config)
    typed = load_typed_analysis(params_file)
    distance = selected_distance.model_dump(mode="json")
    geometry = typed.distances[config.w2_geometry_id].model_dump(mode="json")
    geometry_fields = (
        "family",
        "ground_distance",
        "exponent",
        "estimator",
        "normalization",
    )
    payload = {
        "invocation": asdict(config),
        "factor_data": params["paper5"]["factor_data"],
        "pit_vintage_grid": params["paper3"]["pit_vintage_grid"],
        "news_reference": params["paper3"]["certificate_feasibility"]["news_reference"],
        "distance": {
            field: distance[field]
            for field in (
                *geometry_fields,
                "sample_design",
                "sample_size",
                "sample_window",
            )
        },
        "geometry": {field: geometry[field] for field in geometry_fields},
        "window": window.model_dump(mode="json"),
    }
    parameter_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(parameter_bytes).hexdigest()


def _reference_contract(
    params_file: Path, config: Paper5GMVValidationConfig
) -> tuple[tuple[_ReferenceDesign, ...], int, dict[str, object]]:
    """Read Paper 3's declared reference population and select uniform-cap-15."""
    params = load_params(params_file)
    paper3 = params.get("paper3")
    if not isinstance(paper3, dict):
        _invalid("params.yaml paper3 section must be a mapping")
    feasibility = paper3.get("certificate_feasibility")
    if not isinstance(feasibility, dict):
        _invalid("paper3 certificate-feasibility section must be a mapping")
    reference = feasibility.get("news_reference")
    if not isinstance(reference, dict):
        _invalid("paper3 news-reference section must be a mapping")
    designs = _reference_designs(reference.get("designs"))
    seed = reference.get("seed")
    draws = reference.get("draws")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _invalid("Paper 3 reference seed must be an integer")
    if (
        not isinstance(draws, int)
        or isinstance(draws, bool)
        or draws < _MIN_REFERENCE_DRAWS
    ):
        _invalid("Paper 3 reference draws must be a sufficiently large integer")
    selected = tuple(
        design for design in designs if design.label == config.reference_design_label
    )
    if len(selected) != 1:
        _invalid(f"reference design {config.reference_design_label!r} is not unique")
    selected_design = selected[0]
    if selected_design.maximum_weight != config.maximum_weight:
        _invalid("primary cap must match the selected Paper 3 reference design")
    if draws != config.reference_draws or seed != config.reference_seed:
        _invalid("stage reference settings must match Paper 3's declared settings")
    design_index = next(
        index
        for index, design in enumerate(designs)
        if design.label == selected_design.label
    )
    contract: dict[str, object] = {
        "requested_draws": draws,
        "seed": seed,
        "seed_for_selected_design": seed + design_index,
        "designs": [asdict(design) for design in designs],
    }
    return designs, seed, contract


def _load_excess_return_panel(
    params_file: Path,
    returns_dir: Path,
    oos_returns_dir: Path,
    tickers: list[str],
    start: str,
    end: str,
    factor_model: Literal["ff5_mom"],
) -> tuple[np.ndarray, FloatMatrix]:
    """Load one factor-complete panel and subtract its daily risk-free rate."""
    factor_config = load_factor_data_config(params_file, "paper5")
    if factor_config.primary_model != factor_model:
        _invalid(
            "portfolio factor model does not match the frozen Paper 5 factor contract"
        )
    factor_panel = load_factor_model(factor_config, factor_model)
    dates, raw_returns = load_aligned_return_panel(
        returns_dir,
        tickers,
        start,
        end,
        oos_returns_dir=oos_returns_dir,
    )
    aligned = factor_panel.frame.set_index("Date").reindex(
        pd.DatetimeIndex(pd.to_datetime(dates))
    )
    columns = (*factor_panel.factor_columns, "RF")
    complete = aligned.loc[:, list(columns)].notna().all(axis=1).to_numpy()
    if int(np.sum(complete)) < _MIN_FACTOR_COMPLETE_DATES:
        _invalid("fewer than 30 factor-complete return dates remain")
    values = aligned.loc[complete, list(columns)].to_numpy(dtype=np.float64)
    excess = np.asarray(raw_returns[complete] - values[:, -1, None], dtype=np.float64)
    if not np.isfinite(excess).all():
        _invalid("factor-complete excess returns are nonfinite")
    return np.asarray(dates[complete], dtype="datetime64[ns]"), excess


def run_paper5_gmv_validation(
    paths: Paper5GMVValidationPaths,
    config: Paper5GMVValidationConfig,
) -> dict[str, object]:
    """Write frozen OOS and Paper 3-style in-sample comparison artifacts."""
    config.validate()
    designs, reference_seed, reference_contract = _reference_contract(
        paths.params_file, config
    )
    with np.load(
        paths.spatial_covariance_dir / "covariances.npz", allow_pickle=False
    ) as artifact:
        tickers = [str(value) for value in artifact["tickers"]]
    if len(tickers) != _EXPECTED_ROSTER_SIZE or len(set(tickers)) != len(tickers):
        _invalid("the comparison requires Paper 3's 100-firm unique roster")
    calibration_dates, calibration_returns = _load_excess_return_panel(
        paths.params_file,
        paths.returns_dir,
        paths.oos_returns_dir,
        tickers,
        config.calibration_start,
        config.calibration_end,
        config.factor_model,
    )
    evaluation_dates, evaluation_returns = _load_excess_return_panel(
        paths.params_file,
        paths.returns_dir,
        paths.oos_returns_dir,
        tickers,
        config.evaluation_start,
        config.evaluation_end,
        config.factor_model,
    )
    if calibration_dates.max() >= evaluation_dates.min():
        _invalid("calibration and evaluation panels overlap")
    certificate_budget, certificate_contract = _load_certificate_budget(
        paths.certificate_pit_w2_path,
        paths.certificate_pit_manifest_path,
        paths.params_file,
        tickers,
        config,
    )
    total_covariances, residual_covariances, spatial_summary = (
        _load_spatial_covariances(paths.spatial_covariance_dir, tickers, config)
    )
    _validate_spatial_date_identity(
        cast("dict[str, object]", spatial_summary["sample"]),
        calibration_dates,
        evaluation_dates,
    )
    calibration_std, calibration_volatility = _standardize(calibration_returns)
    primary_weights = _candidate_weights(
        calibration_returns,
        certificate_budget,
        total_covariances,
        config.maximum_weight,
        capital=True,
    )
    primary_residual_weights = {
        f"{name}_residual": _capped_gmv(
            residual_covariances[name], config.maximum_weight, f"{name} residual GMV"
        )
        for name in _SPATIAL_CANDIDATES
    }
    primary_weights.update(primary_residual_weights)
    secondary_total = {
        name: _validate_covariance(
            total_covariances[name]
            / calibration_volatility[:, None]
            / calibration_volatility[None, :],
            len(tickers),
            f"standardized {name}",
        )
        for name in _TOTAL_COVARIANCE_CANDIDATES
    }
    secondary_residual = {
        name: _validate_covariance(
            residual_covariances[name.removesuffix("_residual")]
            / calibration_volatility[:, None]
            / calibration_volatility[None, :],
            len(tickers),
            f"standardized {name}",
        )
        for name in _RESIDUAL_CANDIDATES
    }
    secondary_weights = _candidate_weights(
        calibration_std,
        certificate_budget,
        secondary_total,
        config.maximum_weight,
        capital=False,
    )
    secondary_weights.update(
        {
            name: _capped_gmv(covariance, config.maximum_weight, f"{name} residual GMV")
            for name, covariance in secondary_residual.items()
        }
    )
    calibration_covariance = _validate_covariance(
        np.cov(calibration_returns, rowvar=False, ddof=1),
        len(tickers),
        "calibration",
    )
    standardized_calibration_covariance = _validate_covariance(
        np.cov(calibration_std, rowvar=False, ddof=1),
        len(tickers),
        "standardized calibration",
    )
    primary_direct_covariances = {
        "sample_gmv": calibration_covariance,
        "ledoit_wolf_gmv": _validate_covariance(
            LedoitWolf().fit(calibration_returns).covariance_,
            len(tickers),
            "calibration Ledoit--Wolf",
        ),
        **total_covariances,
        **{
            f"{name}_residual": residual_covariances[name]
            for name in _SPATIAL_CANDIDATES
        },
    }
    secondary_direct_covariances = {
        "sample_gmv": standardized_calibration_covariance,
        "ledoit_wolf_gmv": _validate_covariance(
            LedoitWolf().fit(calibration_std).covariance_,
            len(tickers),
            "standardized calibration Ledoit--Wolf",
        ),
        **secondary_total,
        **secondary_residual,
    }
    primary_objective_diagnostics = {
        name: _gmv_diagnostics(covariance, primary_weights[name], config.maximum_weight)
        for name, covariance in primary_direct_covariances.items()
    }
    secondary_objective_diagnostics = {
        name: _gmv_diagnostics(
            covariance, secondary_weights[name], config.maximum_weight
        )
        for name, covariance in secondary_direct_covariances.items()
    }
    primary_design = tuple(
        design for design in designs if design.label == config.reference_design_label
    )
    primary_reference_rows = _reference_rows(
        _validate_covariance(
            np.cov(evaluation_returns, rowvar=False, ddof=1),
            len(tickers),
            "evaluation",
        ),
        primary_weights,
        primary_design,
        config.reference_draws,
        reference_seed,
        panel="oos_primary",
        seed_offset=1,
    )
    secondary_reference_rows = _reference_rows(
        _validate_covariance(
            np.cov(calibration_std, rowvar=False, ddof=1),
            len(tickers),
            "standardized calibration",
        ),
        secondary_weights,
        designs,
        config.reference_draws,
        reference_seed,
        panel="paper3_in_sample",
    )
    primary_rows = _variance_rows(
        "oos_primary",
        evaluation_returns,
        primary_weights,
        primary_reference_rows,
        formation_covariance=calibration_covariance,
        objective_covariances=primary_direct_covariances,
        objective_diagnostics=primary_objective_diagnostics,
        maximum_weight=config.maximum_weight,
    )
    secondary_rows = _variance_rows(
        "paper3_in_sample",
        calibration_std,
        secondary_weights,
        secondary_reference_rows,
        formation_covariance=standardized_calibration_covariance,
        objective_covariances=secondary_direct_covariances,
        objective_diagnostics=secondary_objective_diagnostics,
        maximum_weight=config.maximum_weight,
    )
    result_rows = primary_rows + secondary_rows
    weights_frame = pd.DataFrame(
        [
            {
                "panel": panel,
                "candidate": candidate,
                "ticker": ticker,
                "weight": float(weight),
                "capital_weights": panel == "oos_primary",
                "risk_unit": "excess_return",
                **_candidate_contract(candidate, panel),
                "cap_policy": (
                    "excluded_from_common_cap_ranking"
                    if weights.max() > config.maximum_weight + _WEIGHT_SUM_TOLERANCE
                    else "included_in_common_cap_ranking"
                ),
            }
            for panel, candidates in (
                ("oos_primary", primary_weights),
                ("paper3_in_sample", secondary_weights),
            )
            for candidate, weights in candidates.items()
            for ticker, weight in zip(tickers, weights, strict=True)
        ]
    )
    results_frame = pd.DataFrame(result_rows)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = paths.output_dir / "weights.parquet"
    results_path = paths.output_dir / "results.parquet"
    summary_path = paths.output_dir / "results.json"
    weights_frame.to_parquet(weights_path, index=False)
    results_frame.to_parquet(results_path, index=False)
    summary: dict[str, object] = {
        "schema_version": "paper3_spatial_portfolio_comparison.v4",
        "analysis_id": "paper3-spatial-portfolio-comparison-v4",
        "evidence_status": "exploratory_retrospective_chronological_holdout",
        "manuscript_integration": "not wired into Paper 3 or Paper 5",
        "sample": {
            "n_assets": len(tickers),
            "tickers": tickers,
            "calibration_start": str(pd.Timestamp(calibration_dates.min()).date()),
            "calibration_end": str(pd.Timestamp(calibration_dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(evaluation_dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(evaluation_dates.max()).date()),
            "calibration_dates": len(calibration_dates),
            "evaluation_dates": len(evaluation_dates),
            "calibration_date_sha256": _date_digest(calibration_dates),
            "evaluation_date_sha256": _date_digest(evaluation_dates),
            "risk_unit": "excess_return",
            "factor_model": config.factor_model,
        },
        "protocol": {
            "calibration": "factor-complete 2018--2022 excess returns only",
            "frozen_vintage": config.formation_vintage,
            "evaluation": (
                "constant-weight excess-return mixtures evaluated on 2023--2026; "
                "not buy-and-hold holdings"
            ),
            "claim_scope": (
                "retrospective chronological return-holdout comparison conditional "
                "on the frozen representation and selected universe"
            ),
            "risk_unit": "excess_return",
            "factor_model": config.factor_model,
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "w1_barycentre_arm_id": config.w1_barycentre_arm_id,
            "w2_barycentre_arm_id": config.w2_barycentre_arm_id,
            "w1_geometry_id": config.w1_geometry_id,
            "w2_geometry_id": config.w2_geometry_id,
            "residual_shrinkage_target": config.residual_shrinkage_target,
            "primary_objective": "total excess-return covariance",
            "residual_diagnostics": (
                "spatial residual-covariance GMV, explicitly secondary"
            ),
            "matched_control": (
                "factor-only rho=0 with the same constant-correlation "
                "residual shrinkage"
            ),
            "direct_benchmark_target": "scaled_identity Ledoit--Wolf on excess returns",
            "maximum_weight": config.maximum_weight,
            "primary_percentile": (
                "evaluation realized variance ranked against the "
                "uniform-cap-15 Paper 3 capped-Dirichlet reference population"
            ),
            "secondary_percentile": (
                "Paper 3 standardized in-sample variance ranked against every "
                "declared Paper 3 reference population"
            ),
            "reference_population": reference_contract,
            "no_lookahead": (
                "primary weights use calibration factor-complete excess returns, "
                "dated PIT W2 certificate, and calibration covariance artifacts only; "
                "evaluation returns/covariances are not optimization inputs"
            ),
            "capital_cap_policy": (
                "retain normalized certificate and inverse-volatility maps unchanged; "
                "mark any mapped allocation above the common cap as excluded "
                "rather than clip"
            ),
            "weight_convention": (
                "constant-weight mixture with costless rebalancing interpretation"
            ),
        },
        "certificate": certificate_contract,
        "spatial_covariance_artifact": {
            "analysis_id": spatial_summary["analysis_id"],
            "schema_version": spatial_summary["schema_version"],
            "risk_unit": "excess_return",
            "formation_window": spatial_summary["sample"],
        },
        "candidates": {
            "paper3_w2_certificate": (
                "Paper 3 zero-slack W2 certificate; converted to capital weights "
                "as q_i/sigma_i in the primary panel"
            ),
            "equal_weight": "equal capital weights",
            "inverse_volatility": "calibration inverse-volatility capital weights",
            "sample_gmv": (
                "calibration sample covariance of excess returns, capped long-only GMV"
            ),
            "ledoit_wolf_gmv": (
                "calibration excess-return Ledoit--Wolf capped GMV; "
                "scaled-identity target"
            ),
            "factor_spherical_qmle": (
                "matched factor-only spherical QMLE residual plus systematic "
                "total covariance capped GMV"
            ),
            "factor_spherical_ledoit_wolf": (
                "matched factor-only spherical constant-correlation residual "
                "shrinkage plus systematic total covariance capped GMV"
            ),
            "factor_diagonal_qmle": (
                "matched factor-only diagonal QMLE residual plus systematic "
                "total covariance capped GMV"
            ),
            "factor_diagonal_ledoit_wolf": (
                "matched factor-only diagonal constant-correlation residual "
                "shrinkage plus systematic total covariance capped GMV"
            ),
            "spatial_spherical_w1_qmle": (
                "Paper 5 spherical W1 total excess-return covariance capped GMV"
            ),
            "spatial_spherical_w1_ledoit_wolf": (
                "Paper 5 spherical W1 total covariance with "
                "constant-correlation residual shrinkage capped GMV"
            ),
            "spatial_spherical_w2_qmle": (
                "Paper 5 spherical W2 total excess-return covariance capped GMV"
            ),
            "spatial_spherical_w2_ledoit_wolf": (
                "Paper 5 spherical W2 total covariance with "
                "constant-correlation residual shrinkage capped GMV"
            ),
            "spatial_diagonal_w1_qmle": (
                "Paper 5 diagonal W1 total excess-return covariance capped GMV"
            ),
            "spatial_diagonal_w1_ledoit_wolf": (
                "Paper 5 diagonal W1 total covariance with "
                "constant-correlation residual shrinkage capped GMV"
            ),
            "spatial_diagonal_w2_qmle": (
                "Paper 5 diagonal W2 total excess-return covariance capped GMV"
            ),
            "spatial_diagonal_w2_ledoit_wolf": (
                "Paper 5 diagonal W2 total covariance with "
                "constant-correlation residual shrinkage capped GMV"
            ),
            **{
                f"{name}_residual": (
                    f"Diagnostic capped GMV using the {name} Paper 5 residual "
                    "covariance only"
                )
                for name in _SPATIAL_CANDIDATES
            },
        },
        "candidate_contract": {
            panel: {name: _candidate_contract(name, panel) for name in _CANDIDATES}
            for panel in ("oos_primary", "paper3_in_sample")
        },
        "rows": result_rows,
        "calibration_volatility": {
            "minimum": float(calibration_volatility.min()),
            "maximum": float(calibration_volatility.max()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upstream_paths = (
        paths.returns_dir,
        paths.oos_returns_dir,
        paths.certificate_pit_w2_path,
        paths.certificate_pit_manifest_path,
        paths.spatial_covariance_dir,
    )
    manifest = {
        "schema_version": "1.0",
        "stage_key": "p5_gmv_validation",
        "lane": "paper5",
        "phase": "exploratory",
        "protocol_id": "paper3.spatial_portfolio_comparison.excess_total_risk.v4",
        "code_identity": {
            "kind": "sha256",
            "value": _stage_source_digest(),
            "label": "Paper 5 GMV validation stage",
        },
        "parameter_identity": {
            "kind": "sha256",
            "value": _parameter_identity(paths.params_file, config),
            "label": "Paper 5 GMV validation consumed parameters",
        },
        "seed_policy": (
            "Paper 3 reference draws use the declared seed plus the design's "
            "position; no random draws are persisted"
        ),
        "upstream_artifacts": [
            {
                "kind": "path-sha256",
                "value": f"{path}:{_sha256_path(path)}",
                "label": path.name,
            }
            for path in upstream_paths
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": sys.platform,
            "compute_dtype": "float64",
            "covariance_solver": (
                "Statadic QPAX implicit QP with normalized objective/gradient "
                "and capped-simplex conditional-gradient gap"
            ),
        },
        "outputs": [
            _output_record(weights_path, paths.output_dir, "parquet"),
            _output_record(results_path, paths.output_dir, "parquet"),
            _output_record(summary_path, paths.output_dir, "json"),
        ],
        "risk_class": "moderate",
        "metadata": {
            "ticker_order": "stored in results.json and weights.parquet",
            "primary_panel": "oos_primary",
            "secondary_panel": "paper3_in_sample",
            "risk_unit": "excess_return",
            "capital_cap_policy": (
                "retain and mark mapped violations; never silently clip"
            ),
            "formation_artifacts": (
                "dated PIT W2 v2 certificate plus Paper 5 v2 calibration covariances"
            ),
        },
    }
    write_manifest(paths.output_dir / "provenance.manifest.json", manifest)
    logger.info("Wrote Paper 5 GMV validation to %s", paths.output_dir)
    return summary

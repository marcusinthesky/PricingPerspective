"""Exploratory Ledoit--Wolf covariance estimates for Paper 5 spatial fits.

This stage is deliberately separate from the manuscript-facing comparison.  It
applies the constant-correlation Ledoit--Wolf estimator to the
factor-residualized spatial innovations and maps the result back through the
fitted spatial filter.  The existing QMLE covariance is emitted alongside it
as the reference estimate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import numpy as np
import pandas as pd
import statadic
from statadic.estimand.core import Covariance, Estimand
from statadic.estimation.covariance import LedoitWolf

from pipeline.io.contracts import write_manifest
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.stages.papers.paper5.spatial_value import (
    Paper5SpatialValueConfig,
    Paper5SpatialValuePaths,
    _fit_models,
    _load_analysis_inputs,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline.io.fama_french import FactorModelId
    from pipeline.stages.papers.paper5.estimate import FactorSarFit
    from pipeline.stages.papers.paper5.spatial_value import _AnalysisInputs
    from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel

logger = logging.getLogger(__name__)

type FloatMatrix = NDArray[np.float64]
type VarianceModel = Literal["spherical", "diagonal"]

_VARIANCE_MODELS: tuple[VarianceModel, ...] = ("spherical", "diagonal")
_SPATIAL_MODELS = ("none", "w1", "w2")
_MATRIX_DIMENSIONS = 2
_PSD_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class Paper5SpatialCovariancePaths:
    """Input, guard, and output boundaries for the exploratory stage."""

    returns_dir: Path
    oos_returns_dir: Path
    w1_barycentre_dir: Path
    w2_barycentre_dir: Path
    transport_results_json: Path
    spatial_value_results_json: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5SpatialCovarianceConfig:
    """Frozen factor and time-window contract for covariance diagnostics."""

    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    factor_model: FactorModelId = "ff5_mom"
    provider_id: str = "qwen3-embedding-8b"
    representation_id: str = "qwen3-embedding-8b-unit"
    w1_barycentre_arm_id: str = "wasserstein_w1_loo"
    w2_barycentre_arm_id: str = "wasserstein_w2_loo"
    w1_geometry_id: str = "wasserstein_w1"
    w2_geometry_id: str = "wasserstein_w2"

    def validate(self) -> None:
        """Reject malformed or overlapping windows before reading inputs."""
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
            message = "spatial covariance windows must be ordered and disjoint"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SpatialCovarianceEstimate:
    """QMLE and Ledoit--Wolf covariance estimates for one fitted operator."""

    # ``qmle`` and ``ledoit_wolf`` are retained as residual-risk aliases for
    # callers of the original v1 API.  New artifacts use the explicit names
    # below so a residual covariance cannot be mistaken for total risk.
    qmle: FloatMatrix
    ledoit_wolf: FloatMatrix
    systematic: FloatMatrix
    total_qmle: FloatMatrix
    total_ledoit_wolf: FloatMatrix
    innovation_qmle: FloatMatrix
    innovation_ledoit_wolf: FloatMatrix
    shrinkage: float


def spatial_ledoit_wolf_covariance(
    fit: FactorSarFit,
    panel: _AlignedFactorPanel,
    weights: FloatMatrix,
) -> SpatialCovarianceEstimate:
    """Estimate innovation covariance and spatialize both covariance choices.

    The Ledoit--Wolf input is the exact innovation panel used by the spatial
    likelihood: ``(I-rho W) r_t - X_t C``.  The returned matrices use the
    stable solve representation ``A^-1 D A^-T`` rather than forming an
    explicit inverse, where ``A = I-rho W``.
    """
    if not fit.converged:
        message = "cannot build covariance from a nonconverged spatial fit"
        raise RuntimeError(message)
    returns = np.asarray(panel.excess_returns, dtype=np.float64)
    factors = np.asarray(panel.factors, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    if returns.ndim != _MATRIX_DIMENSIONS or factors.ndim != _MATRIX_DIMENSIONS:
        message = "spatial covariance inputs must be two-dimensional"
        raise ValueError(message)
    n_dates, n_assets = returns.shape
    if factors.shape != (n_dates, fit.n_factors):
        message = "factor panel does not match the fitted factor dimension"
        raise ValueError(message)
    if weights64.shape != (n_assets, n_assets) or fit.n_assets != n_assets:
        message = "operator does not match the fitted asset dimension"
        raise ValueError(message)
    if fit.n_obs != n_dates or not np.isfinite(returns).all():
        message = "returns do not match the fitted observation count or are nonfinite"
        raise ValueError(message)
    if not np.isfinite(weights64).all():
        message = "spatial operator contains nonfinite values"
        raise ValueError(message)

    design = np.column_stack((np.ones(n_dates, dtype=np.float64), factors))
    innovations = returns - fit.rho_hat * (returns @ weights64.T)
    innovations -= design @ np.asarray(fit.coefficients, dtype=np.float64)
    if not np.isfinite(innovations).all():
        message = "factor-residualized spatial innovations are nonfinite"
        raise ValueError(message)

    with jax.enable_x64(new_val=True):
        ledoit_wolf = Estimand(Covariance(), estimator=LedoitWolf()).estimate(
            innovations
        )
    innovation_lw = _symmetrize(np.asarray(ledoit_wolf.sigma, dtype=np.float64))
    innovation_qmle = np.diag(np.asarray(fit.innovation_variances, dtype=np.float64))
    system = np.eye(n_assets, dtype=np.float64) - fit.rho_hat * weights64
    qmle = _spatialize_covariance(system, innovation_qmle)
    lw = _spatialize_covariance(system, innovation_lw)
    factor_covariance = _symmetrize(
        np.asarray(np.cov(factors, rowvar=False, ddof=1), dtype=np.float64)
    )
    if factor_covariance.shape != (fit.n_factors, fit.n_factors):
        message = "factor covariance has the wrong shape"
        raise ValueError(message)
    reduced_loadings = np.linalg.solve(
        system, np.asarray(fit.coefficients, dtype=np.float64)[1:].T
    )
    systematic = _symmetrize(
        cast(
            "FloatMatrix",
            reduced_loadings @ factor_covariance @ reduced_loadings.T,
        )
    )
    total_qmle = _symmetrize(qmle + systematic)
    total_lw = _symmetrize(lw + systematic)
    return SpatialCovarianceEstimate(
        qmle=qmle,
        ledoit_wolf=lw,
        systematic=systematic,
        total_qmle=total_qmle,
        total_ledoit_wolf=total_lw,
        innovation_qmle=_symmetrize(innovation_qmle),
        innovation_ledoit_wolf=innovation_lw,
        shrinkage=float(cast("float", ledoit_wolf.shrinkage)),
    )


def _symmetrize(matrix: FloatMatrix) -> FloatMatrix:
    """Return a float64 symmetric matrix without changing its diagonal."""
    result = np.asarray(matrix, dtype=np.float64)
    return cast("FloatMatrix", 0.5 * (result + result.T))


def _spatialize_covariance(system: FloatMatrix, innovation: FloatMatrix) -> FloatMatrix:
    """Map an innovation covariance through the spatial filter by linear solves."""
    left = np.linalg.solve(system, innovation)
    result = np.linalg.solve(system, left.T).T
    if not np.isfinite(result).all():
        message = "spatialized covariance is nonfinite"
        raise np.linalg.LinAlgError(message)
    return _symmetrize(cast("FloatMatrix", result))


def _matrix_diagnostics(matrix: FloatMatrix) -> dict[str, float]:
    """Summarize a covariance while preserving float64 eigenvalue checks."""
    symmetric = _symmetrize(matrix)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if not np.isfinite(eigenvalues).all():
        message = "covariance eigenvalues are nonfinite"
        raise ValueError(message)
    scale = max(float(np.max(np.abs(symmetric))), np.finfo(np.float64).tiny)
    if eigenvalues[0] < -_PSD_RELATIVE_TOLERANCE * scale:
        message = "covariance is not positive semidefinite"
        raise ValueError(message)
    return {
        "trace": float(np.trace(symmetric)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(np.linalg.cond(symmetric)),
    }


def _sha256_path(path: Path) -> str:
    """Hash a file or directory with deterministic relative-path framing."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _date_digest(dates: NDArray[np.datetime64]) -> str:
    """Hash the ordered factor-complete dates used to form each covariance."""
    values = np.asarray(dates, dtype="datetime64[ns]").astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _stage_source_digest() -> str:
    return entrypoint_code_identity("paper5-spatial-covariance")


def _output_record(path: Path, root: Path, kind: str) -> dict[str, object]:
    """Return a manifest output record relative to the stage directory."""
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }


def _spatial_value_guard(path: Path) -> dict[str, object]:
    """Require the existing comparison stage before emitting exploratory output."""
    with path.open(encoding="utf-8") as handle:
        payload = cast("dict[str, object]", json.load(handle))
    if payload.get("analysis_id") != "paper5-spatial-value-comparison-v1":
        message = "spatial covariance requires the governed spatial-value comparison"
        raise ValueError(message)
    return payload


def _value_config(config: Paper5SpatialCovarianceConfig) -> Paper5SpatialValueConfig:
    """Adapt the shared timing/operator contract for the existing loader."""
    return Paper5SpatialValueConfig(
        calibration_start=config.calibration_start,
        calibration_end=config.calibration_end,
        evaluation_start=config.evaluation_start,
        evaluation_end=config.evaluation_end,
        factor_model=config.factor_model,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        w1_barycentre_arm_id=config.w1_barycentre_arm_id,
        w2_barycentre_arm_id=config.w2_barycentre_arm_id,
        w1_geometry_id=config.w1_geometry_id,
        w2_geometry_id=config.w2_geometry_id,
    )


def _fit_cell(
    window: str,
    panel: _AlignedFactorPanel,
    variance_model: VarianceModel,
    order: str,
    inputs: _AnalysisInputs,
) -> tuple[FactorSarFit, SpatialCovarianceEstimate]:
    """Fit and convert one window/order/variance cell."""
    fits = _fit_models(
        panel,
        inputs.matrices,
        inputs.eigenvalues,
        variance_model,
    )
    fit = fits[order]
    operator = (
        np.eye(fit.n_assets, dtype=np.float64)
        if order == "none"
        else inputs.matrices[order]
    )
    estimate = spatial_ledoit_wolf_covariance(fit, panel, operator)
    logger.debug("Built %s/%s/%s covariance cell", window, variance_model, order)
    return fit, estimate


def run_paper5_spatial_covariance(
    paths: Paper5SpatialCovariancePaths,
    config: Paper5SpatialCovarianceConfig,
) -> dict[str, object]:
    """Write residual and total excess-return covariance artifacts."""
    config.validate()
    guard = _spatial_value_guard(paths.spatial_value_results_json)
    value_config = _value_config(config)
    inputs = _load_analysis_inputs(
        Paper5SpatialValuePaths(
            returns_dir=paths.returns_dir,
            oos_returns_dir=paths.oos_returns_dir,
            w1_barycentre_dir=paths.w1_barycentre_dir,
            w2_barycentre_dir=paths.w2_barycentre_dir,
            transport_results_json=paths.transport_results_json,
            params_file=paths.params_file,
            output_dir=paths.output_dir,
        ),
        value_config,
    )

    registered = cast("dict[str, Any]", inputs.transport["post_corpus_factor_qmle"])
    registered_results = cast("dict[str, dict[str, Any]]", registered["results"])
    arrays: dict[str, FloatMatrix] = {}
    cells: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    panels = {"calibration": inputs.calibration, "evaluation": inputs.evaluation}
    for window, panel in panels.items():
        cells[window] = {}
        systematic_reference: FloatMatrix | None = None
        for variance_model in _VARIANCE_MODELS:
            cells[window][variance_model] = {}
            for order in _SPATIAL_MODELS:
                fit, estimate = _fit_cell(window, panel, variance_model, order, inputs)
                if (
                    order != "none"
                    and window == "evaluation"
                    and variance_model == "spherical"
                ):
                    registered_rho = float(registered_results[order]["rho_hat"])
                    if not np.isclose(
                        fit.rho_hat, registered_rho, rtol=0.0, atol=1e-10
                    ):
                        message = (
                            f"spherical {order} covariance fit does not reproduce "
                            "the registered transport-order estimate"
                        )
                        raise RuntimeError(message)
                prefix = f"{window}_{variance_model}_{order}"
                if systematic_reference is None:
                    systematic_reference = estimate.systematic
                elif not np.allclose(
                    estimate.systematic,
                    systematic_reference,
                    rtol=1e-8,
                    atol=1e-12,
                ):
                    message = (
                        f"{window}/{variance_model}/{order} changed the reduced-form "
                        "systematic covariance across matched unrestricted fits"
                    )
                    raise RuntimeError(message)
                arrays[f"{prefix}_qmle_residual"] = estimate.qmle
                arrays[f"{prefix}_ledoit_wolf_residual"] = estimate.ledoit_wolf
                arrays[f"{prefix}_systematic"] = estimate.systematic
                arrays[f"{prefix}_qmle_total"] = estimate.total_qmle
                arrays[f"{prefix}_ledoit_wolf_total"] = estimate.total_ledoit_wolf
                qml_diag = _matrix_diagnostics(estimate.qmle)
                lw_diag = _matrix_diagnostics(estimate.ledoit_wolf)
                systematic_diag = _matrix_diagnostics(estimate.systematic)
                total_qml_diag = _matrix_diagnostics(estimate.total_qmle)
                total_lw_diag = _matrix_diagnostics(estimate.total_ledoit_wolf)
                cells[window][variance_model][order] = {
                    "rho_hat": float(fit.rho_hat),
                    "converged": bool(fit.converged),
                    "n_dates": int(fit.n_obs),
                    "n_assets": int(fit.n_assets),
                    "n_factors": int(fit.n_factors),
                    "shrinkage": estimate.shrinkage,
                    "residual_qmle": qml_diag,
                    "residual_ledoit_wolf": lw_diag,
                    "systematic": systematic_diag,
                    "total_qmle": total_qml_diag,
                    "total_ledoit_wolf": total_lw_diag,
                    "residual_risk_unit": "excess_return",
                    "total_risk_unit": "excess_return",
                    "innovation_qmle_trace": float(np.trace(estimate.innovation_qmle)),
                    "innovation_ledoit_wolf_trace": float(
                        np.trace(estimate.innovation_ledoit_wolf)
                    ),
                }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    covariance_path = paths.output_dir / "covariances.npz"
    np.savez_compressed(
        covariance_path,
        tickers=np.asarray(inputs.tickers, dtype="U"),
        **cast("dict[str, Any]", arrays),
    )
    parameters = asdict(config)
    summary: dict[str, object] = {
        "schema_version": "paper5_spatial_covariance.v2",
        "analysis_id": "paper5-spatial-covariance-v2",
        "estimator": {
            "name": "ledoit_wolf",
            "implementation": "statadic.estimation.covariance.LedoitWolf",
            "residual_shrinkage_target": "constant_correlation",
            "input": "factor-residualized spatial innovations",
            "mapping": "A^-1 D A^-T, A=I-rho W",
            "qmle_reference": "fitted innovation variances mapped through A",
            "risk_unit": "excess_return",
            "systematic_component": "B Cov(f) B' with centered factors and ddof=1",
            "reduced_form_loadings": "B=A^-1 C_f",
            "total_covariance": "systematic + residual",
            "covariance_denominator": (
                "factor covariance uses T-1; estimator-defined residual covariance"
            ),
            "fit_mode": (
                "two-step plug-in: fit rho/QMLE, then shrink fitted innovations"
            ),
            "matched_factor_control": (
                "operator none with rho=0 and the same constant-correlation "
                "residual shrinkage"
            ),
        },
        "sample": {
            "n_assets": len(inputs.tickers),
            "tickers": inputs.tickers,
            "calibration_start": str(
                pd.Timestamp(inputs.calibration.dates.min()).date()
            ),
            "calibration_end": str(pd.Timestamp(inputs.calibration.dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(inputs.evaluation.dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(inputs.evaluation.dates.max()).date()),
            "calibration_dates": len(inputs.calibration.dates),
            "evaluation_dates": len(inputs.evaluation.dates),
            "calibration_date_sha256": _date_digest(inputs.calibration.dates),
            "evaluation_date_sha256": _date_digest(inputs.evaluation.dates),
        },
        "windows": ["calibration", "evaluation"],
        "variance_models": list(_VARIANCE_MODELS),
        "operators": list(_SPATIAL_MODELS),
        "spatial_operators": ["w1", "w2"],
        "matched_factor_operator": "none",
        "cells": cells,
        "timing": {
            "calibration": "uses calibration returns and factors only",
            "evaluation": (
                "diagnostic covariance uses evaluation returns; not an ex-ante "
                "forecast input"
            ),
            "look_ahead_status": (
                "calibration matrices use only the realized calibration "
                "return/factor window; representation and universe availability "
                "are not historical point-in-time claims"
            ),
        },
        "upstream_spatial_value": {
            "analysis_id": guard["analysis_id"],
            "results_path": str(paths.spatial_value_results_json),
        },
        "parameters": parameters,
    }
    results_path = paths.output_dir / "results.json"
    results_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    upstream_paths = (
        paths.returns_dir,
        paths.oos_returns_dir,
        paths.w1_barycentre_dir,
        paths.w2_barycentre_dir,
        paths.transport_results_json,
        paths.spatial_value_results_json,
    )
    parameter_bytes = json.dumps(
        parameters, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "stage_key": "p5_spatial_covariance",
        "lane": "paper5",
        "phase": "exploratory",
        "protocol_id": "paper5.spatial_covariance.ledoit_wolf.v2",
        "code_identity": {
            "kind": "sha256",
            "value": _stage_source_digest(),
            "label": "Paper 5 covariance stage and QMLE source",
        },
        "parameter_identity": {
            "kind": "sha256",
            "value": hashlib.sha256(parameter_bytes).hexdigest(),
            "label": "Paper 5 covariance invocation parameters",
        },
        "seed_policy": "deterministic-no-randomness",
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
            "statadic": statadic.__version__,
            "platform": sys.platform,
            "compute_dtype": "float64",
            "estimator_target": "constant_correlation",
            "risk_unit": "excess_return",
            "covariance_schema": "residual, systematic, and total matrices per cell",
        },
        "outputs": [
            _output_record(covariance_path, paths.output_dir, "npz"),
            _output_record(results_path, paths.output_dir, "json"),
        ],
        "risk_class": "moderate",
        "metadata": {
            "covariance_order": "ticker order stored in covariances.npz/tickers",
            "manuscript_integration": "not wired into Paper 3 or Paper 5",
        },
    }
    write_manifest(paths.output_dir / "provenance.manifest.json", manifest)
    logger.info("Wrote Paper 5 spatial covariance diagnostics to %s", paths.output_dir)
    return summary

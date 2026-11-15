"""Operator and classification loaders shared by the Paper 5 stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd

from pipeline.io.barycentre_artifacts import read_target_projection_artifact
from pipeline.stages.papers.paper5.baskets import (
    Paper5OperatorError,
    raise_operator_error,
    validate_weight_matrix,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class _OperatorIdentity(Protocol):
    """Structural identity required to load the frozen W-flat artifact.

    Declared read-only: every stage config that supplies it is a frozen
    dataclass, which cannot satisfy a protocol demanding settable attributes.
    """

    @property
    def provider_id(self) -> str:
        """Return the embedding provider the artifact must have been built with."""

    @property
    def representation_id(self) -> str:
        """Return the representation the artifact must have been built with."""

    @property
    def barycentre_arm_id(self) -> str:
        """Return the barycentre arm the artifact must have been built with."""

    @property
    def geometry_id(self) -> str:
        """Return the ground geometry the artifact must have been built with."""


@dataclass(frozen=True)
class _FlatDiagnostic:
    """Minimal diagnostics adapter required by the perturbation bound."""

    tangent_curvature_min: float
    converged: bool
    assignment_gap_min: float
    assignment_support_size: int


def load_shared_w_flat(
    artifact_dir: Path,
    tickers: list[str],
    *,
    provider_id: str,
    representation_id: str,
    arm_id: str,
    geometry_id: str,
) -> tuple[np.ndarray, list[_FlatDiagnostic]]:
    """Load row-stochastic Paper 5 weights from the shared artifact."""
    weights, diagnostics, _ = read_target_projection_artifact(
        artifact_dir,
        expected_identity={
            "arm_id": arm_id,
            "provider_id": provider_id,
            "representation_id": representation_id,
            "geometry": geometry_id,
            "feasible_set": "simplex_nonnegative",
        },
    )
    w_flat = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    flat_diagnostics: list[_FlatDiagnostic] = []
    for target_index, ticker in enumerate(tickers):
        target_rows = weights.loc[weights["target"] == ticker].sort_values(
            "candidate_index", kind="mergesort"
        )
        expected_candidates = [
            candidate for candidate in tickers if candidate != ticker
        ]
        if target_rows["candidate"].tolist() != expected_candidates:
            message = f"shared barycentre candidate ordering disagrees for {ticker}"
            raise Paper5OperatorError(message)
        candidate_indices = [
            candidate_index
            for candidate_index in range(len(tickers))
            if candidate_index != target_index
        ]
        if int(target_rows["target_index"].iloc[0]) != target_index:
            message = f"shared barycentre target ordering disagrees for {ticker}"
            raise Paper5OperatorError(message)
        w_flat[target_index, candidate_indices] = target_rows["weight"].to_numpy(
            dtype=np.float64
        )
        target_diagnostic = diagnostics.loc[diagnostics["target_id"] == ticker]
        if len(target_diagnostic) != 1:
            message = f"shared barycentre diagnostics missing target {ticker}"
            raise Paper5OperatorError(message)
        curvature = target_diagnostic.iloc[0].get("tangent_curvature_min")
        if curvature is None or not np.isfinite(float(curvature)):
            message = f"shared barycentre curvature missing target {ticker}"
            raise Paper5OperatorError(message)
        flat_diagnostics.append(
            _FlatDiagnostic(
                tangent_curvature_min=float(curvature),
                converged=bool(target_diagnostic.iloc[0]["converged"]),
                assignment_gap_min=float(
                    target_diagnostic.iloc[0].get("assignment_gap_min", 0.0)
                ),
                assignment_support_size=int(
                    target_diagnostic.iloc[0].get("assignment_support_size", 0)
                ),
            )
        )
    return w_flat, flat_diagnostics


def load_operator(
    artifact_dir: Path,
    config: _OperatorIdentity,
) -> tuple[list[str], NDArray[np.float64], dict[str, object]]:
    """Load W-flat by ticker identity; candidate indices are row-local."""
    weights, _diagnostics, summary = read_target_projection_artifact(
        artifact_dir,
        expected_identity={
            "arm_id": config.barycentre_arm_id,
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "geometry": config.geometry_id,
            "feasible_set": "simplex_nonnegative",
        },
    )
    targets = weights[["target_index", "target"]].drop_duplicates()
    targets = targets.sort_values("target_index", kind="mergesort")
    tickers = cast("list[str]", targets["target"].tolist())
    if len(tickers) != int(weights["target_index"].nunique()):
        raise_operator_error("target indices do not define a unique ticker order")
    matrix, _ = load_shared_w_flat(
        artifact_dir,
        tickers,
        provider_id=config.provider_id,
        representation_id=config.representation_id,
        arm_id=config.barycentre_arm_id,
        geometry_id=config.geometry_id,
    )
    validate_weight_matrix(matrix)
    return tickers, matrix, summary


def load_classifications(
    universe_csv: Path,
    tickers: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Load industry and sector labels for exactly the priced universe."""
    universe = pd.read_csv(universe_csv)
    required = {"Symbol", "Industry", "Sector"}
    if not required.issubset(universe.columns):
        raise_operator_error(f"universe is missing columns {sorted(required)}")
    indexed = universe.set_index("Symbol")
    missing = sorted(set(tickers) - set(indexed.index))
    if missing:
        raise_operator_error(f"universe classifications missing tickers {missing}")
    industry = {ticker: str(indexed.loc[ticker, "Industry"]) for ticker in tickers}
    sector = {ticker: str(indexed.loc[ticker, "Sector"]) for ticker in tickers}
    return industry, sector


__all__ = ["load_classifications", "load_operator", "load_shared_w_flat"]

"""Matryoshka dimension mechanism diagnostic for Paper 5's W1/W2 fields."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist as scipy_cdist

from pipeline.io.fama_french import load_factor_data_config, load_factor_model
from pipeline.io.params import load_params
from pipeline.io.typed_analysis import (
    StatisticalDistanceSpec,
    load_typed_analysis,
)
from pipeline.io.typed_providers import load_provider_clouds
from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5.operators import load_operator
from pipeline.stages.papers.paper5.transport_order import _align_factor_panel
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pipeline._kernels.typed_geometry import EmbeddingCloud
    from pipeline.io.fama_french import FactorModelId
    from pipeline.io.typed_analysis import ProviderConfig, RepresentationSpec
    from pipeline.stages.papers.paper5.transport_order import _AlignedFactorPanel

logger = logging.getLogger(__name__)

type FloatMatrix = NDArray[np.float64]
type ComplexVector = NDArray[np.complex128]

_EXPECTED_DIMENSIONS = (4096, 1024, 256, 64)
_UNIT_NORM_TOLERANCE = 1.0e-10
_ASSIGNMENT_OPTIMALITY_TOLERANCE = 1.0e-12
_MINIMUM_CLOUDS = 2
# Tail reference for the switched-assignment diagnostic. The threshold is a
# quantile of the pair's complete cost matrix, so it is fixed by the geometry
# and cannot move with either assignment.
_TAIL_QUANTILE = 0.95


class Paper5TransportDimensionError(ValueError):
    """Raised when the dimension ladder violates its registered contract."""


@dataclass(frozen=True, slots=True)
class Paper5TransportDimensionPaths:
    """Input and output boundaries for the dimension-ladder diagnostic."""

    embeddings_dir: Path
    returns_dir: Path
    oos_returns_dir: Path
    barycentres_root: Path
    params_file: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Paper5TransportDimensionConfig:
    """Frozen timing and factor-model contract for the diagnostic."""

    calibration_start: str = "2018-01-01"
    calibration_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2026-04-06"
    factor_model: FactorModelId = "ff5_mom"

    def validate(self) -> None:
        """Reject overlapping or inverted calibration/evaluation windows."""
        calibration_start = pd.Timestamp(self.calibration_start)
        calibration_end = pd.Timestamp(self.calibration_end)
        evaluation_start = pd.Timestamp(self.evaluation_start)
        evaluation_end = pd.Timestamp(self.evaluation_end)
        if calibration_start > calibration_end or evaluation_start > evaluation_end:
            message = "transport-dimension sample window is inverted"
            raise Paper5TransportDimensionError(message)
        if calibration_end >= evaluation_start:
            message = "transport-dimension calibration and evaluation overlap"
            raise Paper5TransportDimensionError(message)


@dataclass(frozen=True, slots=True)
class _DimensionCell:
    cell_id: str
    provider_key: str
    provider_id: str
    representation_key: str
    representation_id: str
    effective_dimension: int
    native_dimension: int


@dataclass(frozen=True, slots=True)
class _OperatorIdentity:
    provider_id: str
    representation_id: str
    barycentre_arm_id: str
    geometry_id: str


@dataclass(slots=True)
class _GeometryAccumulator:
    """Stable online moments and assignment totals across cost matrices."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    distance_min: float = float("inf")
    distance_max: float = float("-inf")
    assignment_matches: int = 0
    assignment_atoms: int = 0

    def update(self, cost: np.ndarray, matches: int, atoms: int) -> None:
        """Merge one pair's complete cost matrix and assignment overlap."""
        chunk_count = int(cost.size)
        chunk_mean = float(np.mean(cost))
        chunk_m2 = float(np.sum(np.square(cost - chunk_mean), dtype=np.float64))
        if self.count == 0:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
        else:
            combined_count = self.count + chunk_count
            delta = chunk_mean - self.mean
            self.m2 += (
                chunk_m2 + delta * delta * self.count * chunk_count / combined_count
            )
            self.mean += delta * chunk_count / combined_count
            self.count = combined_count
        self.distance_min = min(self.distance_min, float(np.min(cost)))
        self.distance_max = max(self.distance_max, float(np.max(cost)))
        self.assignment_matches += matches
        self.assignment_atoms += atoms


def _ladder_documents(
    params_file: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load the authored representation roster and ladder subtree."""
    params = load_params(params_file)
    paper5 = params.get("paper5")
    roster = params.get("representation_ablation_specs")
    if not isinstance(paper5, dict) or not isinstance(roster, dict):
        message = "params.yaml is missing Paper 5 or representation roster"
        raise Paper5TransportDimensionError(message)
    ladder = paper5.get("transport_dimension_ladder")
    if not isinstance(ladder, dict):
        message = "params.yaml is missing paper5.transport_dimension_ladder"
        raise Paper5TransportDimensionError(message)
    return cast("dict[str, object]", roster), cast("dict[str, object]", ladder)


def _ladder_cell(cell_id: str, raw: object) -> _DimensionCell:
    """Parse one representation-roster entry into a dimension cell."""
    if not isinstance(raw, dict):
        message = f"transport-dimension cell {cell_id!r} is absent from the roster"
        raise Paper5TransportDimensionError(message)
    try:
        return _DimensionCell(
            cell_id=cell_id,
            provider_key=str(raw["provider_key"]),
            provider_id=str(raw["provider_id"]),
            representation_key=str(raw["representation_key"]),
            representation_id=str(raw["representation_id"]),
            effective_dimension=int(raw["effective_dimension"]),
            native_dimension=int(raw["native_dimension"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = f"transport-dimension cell {cell_id!r} is malformed"
        raise Paper5TransportDimensionError(message) from error


def _registered_cell_ids(
    ladder: dict[str, object], config: Paper5TransportDimensionConfig
) -> list[str]:
    """Validate the non-rescaled ladder header and return its cell IDs."""
    if ladder.get("ground_metric") != "euclidean_chord":
        message = "transport-dimension ladder must retain Euclidean chord distance"
        raise Paper5TransportDimensionError(message)
    if ladder.get("distance_rescaling") != "none":
        message = "transport-dimension ladder does not admit distance rescaling"
        raise Paper5TransportDimensionError(message)
    if ladder.get("factor_model") != config.factor_model:
        message = "transport-dimension factor model disagrees with the CLI contract"
        raise Paper5TransportDimensionError(message)
    cell_ids = ladder.get("cell_ids")
    if not isinstance(cell_ids, list) or not all(
        isinstance(cell_id, str) for cell_id in cell_ids
    ):
        message = "transport-dimension cell_ids must be a list of strings"
        raise Paper5TransportDimensionError(message)
    expected_dimensions = ladder.get("expected_dimensions")
    if not isinstance(expected_dimensions, list) or not all(
        isinstance(dimension, int) for dimension in expected_dimensions
    ):
        message = "transport-dimension expected_dimensions must contain integers"
        raise Paper5TransportDimensionError(message)
    if tuple(expected_dimensions) != _EXPECTED_DIMENSIONS:
        message = "transport-dimension expected dimensions are not 4096/1024/256/64"
        raise Paper5TransportDimensionError(message)
    return cast("list[str]", cell_ids)


def _ladder_cells(
    params_file: Path,
    config: Paper5TransportDimensionConfig,
) -> tuple[_DimensionCell, ...]:
    """Resolve the four registered Qwen3-8B prefix cells without rescaling."""
    roster, ladder = _ladder_documents(params_file)
    cell_ids = _registered_cell_ids(ladder, config)
    cells = [_ladder_cell(cell_id, roster.get(cell_id)) for cell_id in cell_ids]
    dimensions = tuple(cell.effective_dimension for cell in cells)
    if dimensions != _EXPECTED_DIMENSIONS or len(set(cell_ids)) != len(cell_ids):
        message = "transport-dimension cells must uniquely follow 4096/1024/256/64"
        raise Paper5TransportDimensionError(message)
    if len({cell.provider_id for cell in cells}) != 1:
        message = "transport-dimension cells must use one embedding provider"
        raise Paper5TransportDimensionError(message)
    return tuple(cells)


def _validate_geometry_contract(
    params_file: Path,
    cells: tuple[_DimensionCell, ...],
) -> tuple[ProviderConfig, dict[str, RepresentationSpec], int]:
    """Validate truncate-then-normalize and the shared balanced sample size."""
    registry = load_typed_analysis(params_file)
    provider = registry.providers.get(cells[0].provider_key)
    if provider is None or provider.provider_id != cells[0].provider_id:
        message = "transport-dimension provider does not resolve in typed_analysis"
        raise Paper5TransportDimensionError(message)
    if provider.dimensions != cells[0].native_dimension:
        message = "transport-dimension native width disagrees with the provider"
        raise Paper5TransportDimensionError(message)

    representations: dict[str, RepresentationSpec] = {}
    for cell in cells:
        representation = registry.representations.get(cell.representation_key)
        if representation is None:
            message = f"unknown representation {cell.representation_key!r}"
            raise Paper5TransportDimensionError(message)
        realized_dimension = representation.dimension or provider.dimensions
        if (
            representation.provider_id != provider.provider_id
            or representation.representation_id != cell.representation_id
            or representation.transform != "l2_normalize_rows"
            or realized_dimension != cell.effective_dimension
        ):
            message = f"cell {cell.cell_id!r} violates truncate-then-normalize identity"
            raise Paper5TransportDimensionError(message)
        representations[cell.cell_id] = representation

    geometries = [
        registry.distances.get(name) for name in ("wasserstein_w1", "wasserstein_w2")
    ]
    if not all(isinstance(item, StatisticalDistanceSpec) for item in geometries):
        message = "transport-dimension W1/W2 geometries are not registered"
        raise Paper5TransportDimensionError(message)
    w1, w2 = cast(
        "tuple[StatisticalDistanceSpec, StatisticalDistanceSpec]", tuple(geometries)
    )
    if (
        w1.ground_distance != "euclidean_chord"
        or w2.ground_distance != "euclidean_chord"
        or w1.sample_size is None
        or w1.sample_size != w2.sample_size
    ):
        message = "transport-dimension W1/W2 ground geometry or sample sizes differ"
        raise Paper5TransportDimensionError(message)
    return provider, representations, w1.sample_size


def _cloud_values(cloud: EmbeddingCloud) -> np.ndarray:
    """Return one embedding array; ``values`` is not a pandas accessor."""
    return np.asarray(cloud.values, dtype=np.float64)


def _validated_cloud_shape(
    clouds: tuple[EmbeddingCloud, ...], *, cell_id: str, dimension: int
) -> tuple[int, int]:
    """Require equal post-truncation shapes and unit-normalized rows."""
    if len(clouds) < _MINIMUM_CLOUDS:
        message = "assignment diagnostic requires at least two embedding clouds"
        raise Paper5TransportDimensionError(message)
    shape = _cloud_values(clouds[0]).shape
    if any(_cloud_values(cloud).shape != shape for cloud in clouds):
        message = "assignment diagnostic requires equal cloud shapes"
        raise Paper5TransportDimensionError(message)
    if shape[1] != dimension:
        message = "assignment diagnostic cloud width disagrees with its dimension"
        raise Paper5TransportDimensionError(message)
    for cloud in clouds:
        norms = np.linalg.norm(_cloud_values(cloud), axis=1)
        if not np.allclose(norms, 1.0, atol=_UNIT_NORM_TOLERANCE, rtol=0.0):
            message = f"{cell_id}: {cloud.item_id} is not normalized after truncation"
            raise Paper5TransportDimensionError(message)
    return shape


def _switch_tail_fields(
    cost: np.ndarray,
    rows: np.ndarray,
    permutation_w1: np.ndarray,
    permutation_w2: np.ndarray,
    *,
    matches: int,
    atom_count: int,
) -> dict[str, object]:
    """Describe what the W1-versus-W2 reassignment does to matched distances.

    Both assignments are scored on the same chord distance, never on the cost
    each of them minimizes, so the comparison is on one common scale. The tail
    threshold is a quantile of the pair's complete cost matrix and therefore
    does not move with either assignment.

    Switched atoms are reported as sums rather than means. Pairs whose two
    assignments coincide have no switched atoms and contribute zero to both the
    numerator and the denominator of the pooled switched-atom mean.
    """
    matched_w1 = cost[rows, permutation_w1]
    matched_w2 = cost[rows, permutation_w2]
    switched = permutation_w1 != permutation_w2
    threshold = float(np.quantile(cost, _TAIL_QUANTILE))
    return {
        "switched_atom_count": atom_count - matches,
        "matched_tail_threshold": threshold,
        "w1_matched_chord_max": float(np.max(matched_w1)),
        "w2_matched_chord_max": float(np.max(matched_w2)),
        "w1_matched_chord_p95": float(np.quantile(matched_w1, _TAIL_QUANTILE)),
        "w2_matched_chord_p95": float(np.quantile(matched_w2, _TAIL_QUANTILE)),
        "w1_matched_chord_population_sd": float(np.std(matched_w1, ddof=0)),
        "w2_matched_chord_population_sd": float(np.std(matched_w2, ddof=0)),
        "w1_matched_tail_count": int(np.count_nonzero(matched_w1 > threshold)),
        "w2_matched_tail_count": int(np.count_nonzero(matched_w2 > threshold)),
        "w1_switched_chord_sum": float(np.sum(matched_w1[switched])),
        "w2_switched_chord_sum": float(np.sum(matched_w2[switched])),
    }


def _assignment_pair_record(
    left: EmbeddingCloud,
    right: EmbeddingCloud,
    *,
    cell_id: str,
    dimension: int,
    atom_count: int,
) -> tuple[dict[str, object], np.ndarray, int]:
    """Solve both orders on one shared chord-cost matrix and record overlap."""
    cost = scipy_cdist(_cloud_values(left), _cloud_values(right), metric="euclidean")
    row_w1, permutation_w1 = linear_sum_assignment(cost)
    squared_cost = np.square(cost)
    row_w2, permutation_w2 = linear_sum_assignment(squared_cost)
    expected_rows = np.arange(atom_count, dtype=np.int64)
    if not np.array_equal(row_w1, expected_rows) or not np.array_equal(
        row_w2, expected_rows
    ):
        message = "balanced assignment did not cover every source atom"
        raise Paper5TransportDimensionError(message)

    w1_cost = float(np.mean(cost[row_w1, permutation_w1]))
    w1_cross_cost = float(np.mean(cost[row_w2, permutation_w2]))
    w2_cost = float(np.mean(squared_cost[row_w2, permutation_w2]))
    w2_cross_cost = float(np.mean(squared_cost[row_w1, permutation_w1]))
    if (
        w1_cost > w1_cross_cost + _ASSIGNMENT_OPTIMALITY_TOLERANCE
        or w2_cost > w2_cross_cost + _ASSIGNMENT_OPTIMALITY_TOLERANCE
    ):
        message = "W1/W2 assignment optimality cross-check failed"
        raise Paper5TransportDimensionError(message)

    matches = int(np.count_nonzero(permutation_w1 == permutation_w2))
    pair_mean = float(np.mean(cost))
    pair_sd = float(np.std(cost, ddof=0))
    switch_fields = _switch_tail_fields(
        cost,
        row_w1,
        permutation_w1,
        permutation_w2,
        matches=matches,
        atom_count=atom_count,
    )
    record: dict[str, object] = {
        "cell_id": cell_id,
        "effective_dimension": dimension,
        "left_id": left.item_id,
        "right_id": right.item_id,
        "distance_count": int(cost.size),
        "chord_distance_mean": pair_mean,
        "chord_distance_population_sd": pair_sd,
        "chord_distance_cv": pair_sd / pair_mean,
        "chord_distance_min": float(np.min(cost)),
        "chord_distance_max": float(np.max(cost)),
        "assignment_atom_count": atom_count,
        "assignment_match_count": matches,
        "assignment_overlap": matches / atom_count,
        "identical_assignment": matches == atom_count,
        "w1_mean_matched_chord_cost": w1_cost,
        "w1_cost_under_w2_assignment": w1_cross_cost,
        "w2_mean_matched_squared_cost": w2_cost,
        "w2_cost_under_w1_assignment": w2_cross_cost,
        **switch_fields,
    }
    return record, cost, matches


def _conditional_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean over a subset, or zero when the subset is empty."""
    if not bool(np.any(mask)):
        return 0.0
    return float(np.mean(values[mask]))


def _switch_tail_summary(
    pair_records: list[dict[str, object]],
    pair_count: int,
) -> dict[str, float | int | bool]:
    """Pool the per-pair switched-assignment tail statistics.

    The signed differences are all W2 minus W1 on the common chord scale. The
    registered pattern is that quadratic transport buys a shorter matched tail
    by accepting longer matches on the atoms it reassigns; the flag records
    whether the pooled evidence has that sign, not whether it is significant.
    """

    def column(key: str) -> np.ndarray:
        return np.asarray([float(cast("float", r[key])) for r in pair_records])

    switched = column("switched_atom_count")
    switched_total = float(np.sum(switched))
    if switched_total <= 0.0:
        message = "no firm pair assigns any target atom differently under d and d^2"
        raise Paper5TransportDimensionError(message)
    max_difference = column("w2_matched_chord_max") - column("w1_matched_chord_max")
    p95_difference = column("w2_matched_chord_p95") - column("w1_matched_chord_p95")
    sd_difference = column("w2_matched_chord_population_sd") - column(
        "w1_matched_chord_population_sd"
    )
    tail_difference = column("w2_matched_tail_count") - column("w1_matched_tail_count")
    switched_difference = (
        float(np.sum(column("w2_switched_chord_sum")))
        - float(np.sum(column("w1_switched_chord_sum")))
    ) / switched_total
    mean_max = float(np.mean(max_difference))
    mean_p95 = float(np.mean(p95_difference))
    return {
        "switched_atom_total": int(switched_total),
        "matched_chord_max_w2_minus_w1": mean_max,
        "matched_chord_p95_w2_minus_w1": mean_p95,
        "matched_chord_sd_w2_minus_w1": float(np.mean(sd_difference)),
        "matched_tail_count_w2_minus_w1": float(np.mean(tail_difference)),
        "switched_matched_chord_w2_minus_w1": switched_difference,
        "max_reduction_pair_share": float(np.mean(max_difference < 0.0)),
        "max_increase_pair_share": float(np.mean(max_difference > 0.0)),
        "max_reduction_given_lower": _conditional_mean(
            max_difference, max_difference < 0.0
        ),
        "max_increase_given_higher": _conditional_mean(
            max_difference, max_difference > 0.0
        ),
        "p95_reduction_pair_share": float(np.mean(p95_difference < 0.0)),
        "tail_reduction_pair_share": float(np.mean(tail_difference < 0.0)),
        "tail_trade_pattern": bool(
            mean_max < 0.0 and mean_p95 <= 0.0 and switched_difference > 0.0
        ),
        "switch_diagnostic_pair_count": pair_count,
    }


def _geometry_summary(
    accumulator: _GeometryAccumulator,
    pair_records: list[dict[str, object]],
    pair_count: int,
) -> dict[str, float | int | bool]:
    """Finalize census moments and pooled assignment overlap."""
    if accumulator.mean <= 0.0 or accumulator.assignment_atoms == 0:
        message = "assignment diagnostic produced an empty or zero-distance census"
        raise Paper5TransportDimensionError(message)
    population_sd = float(np.sqrt(accumulator.m2 / accumulator.count))
    identical_pairs = sum(
        bool(record["identical_assignment"]) for record in pair_records
    )
    return {
        "firm_pair_count": pair_count,
        "article_distance_count": accumulator.count,
        "chord_distance_mean": accumulator.mean,
        "chord_distance_population_sd": population_sd,
        "chord_distance_cv": population_sd / accumulator.mean,
        "chord_distance_min": accumulator.distance_min,
        "chord_distance_max": accumulator.distance_max,
        "assignment_atom_count": accumulator.assignment_atoms,
        "assignment_match_count": accumulator.assignment_matches,
        "assignment_overlap": (
            accumulator.assignment_matches / accumulator.assignment_atoms
        ),
        "identical_assignment_pair_share": identical_pairs / pair_count,
        **_switch_tail_summary(pair_records, pair_count),
    }


def _assignment_geometry(
    clouds: tuple[EmbeddingCloud, ...],
    *,
    cell_id: str,
    dimension: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Measure every cross-firm chord cost and both optimal assignments."""
    shape = _validated_cloud_shape(clouds, cell_id=cell_id, dimension=dimension)
    accumulator = _GeometryAccumulator()
    pair_records: list[dict[str, object]] = []
    pair_count = len(clouds) * (len(clouds) - 1) // 2
    completed = 0
    for left_index, left in enumerate(clouds):
        for right in clouds[left_index + 1 :]:
            record, cost, matches = _assignment_pair_record(
                left,
                right,
                cell_id=cell_id,
                dimension=dimension,
                atom_count=shape[0],
            )
            pair_records.append(record)
            accumulator.update(cost, matches, shape[0])
            completed += 1
            if completed % 250 == 0 or completed == pair_count:
                logger.info(
                    "%s assignment diagnostic %d/%d firm pairs",
                    cell_id,
                    completed,
                    pair_count,
                )

    summary = _geometry_summary(accumulator, pair_records, pair_count)
    return pd.DataFrame.from_records(pair_records), summary


def _operator_certificate(diagnostics: pd.DataFrame) -> dict[str, object]:
    """Summarize the persisted simplex certificate for one operator."""
    required = {
        "converged",
        "objective",
        "optimality_gap",
        "projected_gradient_norm",
        "constraint_violation",
    }
    if not required.issubset(diagnostics.columns) or diagnostics.empty:
        message = "target-operator diagnostics have an incomplete certificate"
        raise Paper5TransportDimensionError(message)
    ratio = diagnostics["optimality_gap"] / (1.0 + diagnostics["objective"].abs())
    return {
        "all_converged": bool(diagnostics["converged"].all()),
        "max_optimality_gap": float(diagnostics["optimality_gap"].max()),
        "max_certificate_ratio": float(ratio.max()),
        "max_projected_gradient_norm": float(
            diagnostics["projected_gradient_norm"].max()
        ),
        "max_constraint_violation": float(diagnostics["constraint_violation"].max()),
    }


def _load_operators(
    paths: Paper5TransportDimensionPaths,
    cell: _DimensionCell,
) -> tuple[list[str], dict[str, FloatMatrix], dict[str, dict[str, object]]]:
    """Load and validate one dimension's target-anchored W1/W2 operators."""
    matrices: dict[str, FloatMatrix] = {}
    certificates: dict[str, dict[str, object]] = {}
    tickers: list[str] | None = None
    for order in ("w1", "w2"):
        artifact_dir = (
            paths.barycentres_root
            / cell.provider_id
            / cell.representation_id
            / f"wasserstein_{order}_loo"
        )
        order_tickers, matrix, _summary = load_operator(
            artifact_dir,
            _OperatorIdentity(
                provider_id=cell.provider_id,
                representation_id=cell.representation_id,
                barycentre_arm_id=f"wasserstein_{order}_loo",
                geometry_id=f"wasserstein_{order}",
            ),
        )
        if tickers is None:
            tickers = order_tickers
        elif tickers != order_tickers:
            message = f"{cell.cell_id}: W1 and W2 ticker orders differ"
            raise Paper5TransportDimensionError(message)
        matrices[order] = np.asarray(matrix, dtype=np.float64)
        diagnostics = pd.read_parquet(artifact_dir / "diagnostics.parquet")
        certificates[order] = _operator_certificate(diagnostics)
        if certificates[order]["all_converged"] is not True:
            message = f"{cell.cell_id}: {order.upper()} target solver did not converge"
            raise Paper5TransportDimensionError(message)
    if tickers is None:
        message = f"{cell.cell_id}: no target operators were loaded"
        raise Paper5TransportDimensionError(message)
    return tickers, matrices, certificates


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite pooled Pearson correlation."""
    value = float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])
    if not np.isfinite(value):
        message = "transport-dimension comparison produced a degenerate correlation"
        raise Paper5TransportDimensionError(message)
    return value


def _operator_return_diagnostics(
    matrices: dict[str, FloatMatrix],
    calibration: _AlignedFactorPanel,
    evaluation: _AlignedFactorPanel,
) -> dict[str, float | int | bool]:
    """Trace operator differences through signals and held-out FF5+MOM scores."""
    w1 = matrices["w1"]
    w2 = matrices["w2"]
    off_diagonal = ~np.eye(w1.shape[0], dtype=bool)
    signal_w1 = evaluation.excess_returns @ w1.T
    signal_w2 = evaluation.excess_returns @ w2.T
    eigenvalues = {
        order: cast("ComplexVector", np.linalg.eigvals(matrix))
        for order, matrix in matrices.items()
    }
    calibration_fits = {
        order: est.sar_factor_qmle(
            calibration.excess_returns,
            matrix,
            calibration.factors,
            eigvals_w=eigenvalues[order],
        )
        for order, matrix in matrices.items()
    }
    evaluation_fits = {
        order: est.sar_factor_qmle(
            evaluation.excess_returns,
            matrix,
            evaluation.factors,
            eigvals_w=eigenvalues[order],
        )
        for order, matrix in matrices.items()
    }
    if not all(
        fit.converged for fit in (*calibration_fits.values(), *evaluation_fits.values())
    ):
        message = "transport-dimension factor QMLE did not converge"
        raise Paper5TransportDimensionError(message)
    daily_scores = {
        order: est.sar_factor_daily_logscore(
            calibration_fits[order],
            evaluation.excess_returns,
            matrices[order],
            evaluation.factors,
            eigvals_w=eigenvalues[order],
        )
        / w1.shape[0]
        for order in ("w1", "w2")
    }
    score_difference = daily_scores["w2"] - daily_scores["w1"]
    signal_difference = signal_w2 - signal_w1
    signal_scale = float(np.std(signal_w2, ddof=1))
    if signal_scale <= 0.0:
        message = "transport-dimension W2 spatial lag has zero dispersion"
        raise Paper5TransportDimensionError(message)
    return {
        "off_diagonal_weight_correlation": _correlation(
            w1[off_diagonal], w2[off_diagonal]
        ),
        "off_diagonal_mean_absolute_difference": float(
            np.mean(np.abs(w2[off_diagonal] - w1[off_diagonal]))
        ),
        "off_diagonal_rms_difference": float(
            np.sqrt(np.mean(np.square(w2[off_diagonal] - w1[off_diagonal])))
        ),
        "spatial_lag_correlation": _correlation(signal_w1, signal_w2),
        "spatial_lag_difference_sd": float(np.std(signal_difference, ddof=1)),
        "spatial_lag_relative_difference_sd": float(
            np.std(signal_difference, ddof=1) / signal_scale
        ),
        "calibration_rho_w1": calibration_fits["w1"].rho_hat,
        "calibration_rho_w2": calibration_fits["w2"].rho_hat,
        "evaluation_rho_w1": evaluation_fits["w1"].rho_hat,
        "evaluation_rho_w2": evaluation_fits["w2"].rho_hat,
        "held_out_score_w1": float(np.mean(daily_scores["w1"])),
        "held_out_score_w2": float(np.mean(daily_scores["w2"])),
        "held_out_score_w2_minus_w1": float(np.mean(score_difference)),
        "held_out_daily_score_difference_sd": float(np.std(score_difference, ddof=1)),
        "held_out_w2_score_win_share": float(np.mean(score_difference > 0.0)),
        "calibration_trading_days": len(calibration.dates),
        "evaluation_trading_days": len(evaluation.dates),
        "all_factor_qmle_converged": True,
    }


def _plain(value: object) -> object:
    """Convert NumPy values for deterministic JSON serialization."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _market_item_ids(params_file: Path) -> list[str]:
    """Load the exact registered market-item roster."""
    item_ids = load_params(params_file).get("market_symbols")
    if not isinstance(item_ids, list) or not all(
        isinstance(item, str) for item in item_ids
    ):
        message = "params.yaml market_symbols must be a list of ticker IDs"
        raise Paper5TransportDimensionError(message)
    return cast("list[str]", item_ids)


def _load_ladder_operators(
    paths: Paper5TransportDimensionPaths,
    cells: tuple[_DimensionCell, ...],
) -> tuple[
    list[str],
    dict[str, dict[str, FloatMatrix]],
    dict[str, dict[str, dict[str, object]]],
]:
    """Load every dimension and enforce one shared ticker ordering."""
    operators: dict[str, dict[str, FloatMatrix]] = {}
    certificates: dict[str, dict[str, dict[str, object]]] = {}
    tickers: list[str] | None = None
    for cell in cells:
        order_tickers, matrices, cell_certificates = _load_operators(paths, cell)
        if tickers is None:
            tickers = order_tickers
        elif tickers != order_tickers:
            message = f"{cell.cell_id}: ticker order differs across dimensions"
            raise Paper5TransportDimensionError(message)
        operators[cell.cell_id] = matrices
        certificates[cell.cell_id] = cell_certificates
    if tickers is None:
        message = "transport-dimension ladder loaded no operators"
        raise Paper5TransportDimensionError(message)
    return tickers, operators, certificates


def _factor_samples(
    paths: Paper5TransportDimensionPaths,
    config: Paper5TransportDimensionConfig,
    tickers: list[str],
) -> tuple[_AlignedFactorPanel, _AlignedFactorPanel]:
    """Load the common FF5+MOM calibration and evaluation panels once."""
    factor_config = load_factor_data_config(paths.params_file, "paper5")
    if factor_config.primary_model != config.factor_model:
        message = "transport-dimension model is not Paper 5's primary factor model"
        raise Paper5TransportDimensionError(message)
    factor_panel = load_factor_model(factor_config, config.factor_model)
    calibration_dates, calibration_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        config.calibration_start,
        config.calibration_end,
        oos_returns_dir=paths.oos_returns_dir,
    )
    evaluation_dates, evaluation_returns = load_aligned_return_panel(
        paths.returns_dir,
        tickers,
        config.evaluation_start,
        config.evaluation_end,
        oos_returns_dir=paths.oos_returns_dir,
    )
    calibration = _align_factor_panel(
        cast("NDArray[np.datetime64]", calibration_dates),
        cast("FloatMatrix", calibration_returns),
        factor_panel,
    )
    evaluation = _align_factor_panel(
        cast("NDArray[np.datetime64]", evaluation_dates),
        cast("FloatMatrix", evaluation_returns),
        factor_panel,
    )
    return calibration, evaluation


def _endpoint_comparison(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize the controlled 4096-to-64 endpoint movement."""

    def numeric(record: dict[str, object], key: str) -> float:
        value = record.get(key)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            message = f"transport-dimension result {key!r} is not numeric"
            raise Paper5TransportDimensionError(message)
        return float(value)

    by_dimension = {
        int(numeric(record, "effective_dimension")): record for record in records
    }
    full = by_dimension[4096]
    low = by_dimension[64]
    return {
        "chord_distance_cv_64_minus_4096": (
            numeric(low, "chord_distance_cv") - numeric(full, "chord_distance_cv")
        ),
        "assignment_overlap_64_minus_4096": (
            numeric(low, "assignment_overlap") - numeric(full, "assignment_overlap")
        ),
        "operator_correlation_64_minus_4096": (
            numeric(low, "off_diagonal_weight_correlation")
            - numeric(full, "off_diagonal_weight_correlation")
        ),
        "spatial_lag_correlation_64_minus_4096": (
            numeric(low, "spatial_lag_correlation")
            - numeric(full, "spatial_lag_correlation")
        ),
        "held_out_score_difference_64_minus_4096": (
            numeric(low, "held_out_score_w2_minus_w1")
            - numeric(full, "held_out_score_w2_minus_w1")
        ),
        "directional_concentration_pattern": bool(
            numeric(low, "chord_distance_cv") > numeric(full, "chord_distance_cv")
            and numeric(low, "assignment_overlap") < numeric(full, "assignment_overlap")
        ),
    }


def _result_payload(
    records: list[dict[str, object]],
    tickers: list[str],
    sample_size: int,
    calibration: _AlignedFactorPanel,
    evaluation: _AlignedFactorPanel,
    config: Paper5TransportDimensionConfig,
) -> dict[str, object]:
    """Assemble the self-describing ladder result."""
    return {
        "schema_version": 1,
        "analysis_id": "paper5-matryoshka-w1-w2-mechanism-v1",
        "design": {
            "dimensions": list(_EXPECTED_DIMENSIONS),
            "representation_transform": (
                "leading-coordinate truncation then row-wise L2 normalization"
            ),
            "ground_metric": "Euclidean chord distance on unit-normalized prefixes",
            "distance_rescaling": "none",
            "distance_population": (
                "every article-to-article chord cost in each unordered cross-firm "
                "balanced-assignment problem"
            ),
            "distance_sd": (
                "population standard deviation of the enumerated cost census"
            ),
            "assignment_overlap": (
                "share of target atoms assigned to the same candidate atom by "
                "minimum-sum d and minimum-sum d-squared assignments"
            ),
            "switched_atoms": (
                "target atoms the two assignments send to different candidate "
                "atoms; pooled switched-atom means weight each firm pair by its "
                "own switched count, so pairs with identical assignments "
                "contribute nothing"
            ),
            "matched_tail_threshold": (
                f"the {_TAIL_QUANTILE:.2f} quantile of each firm pair's complete "
                "chord-cost matrix, fixed by the geometry and invariant to the "
                "assignment being scored"
            ),
            "matched_distance_scale": (
                "both assignments are scored on chord distance, never on the "
                "cost each one minimizes"
            ),
            "factor_model": config.factor_model,
            "held_out_design": (
                "fit 2018-2022 factor QMLE parameters and score 2023-2026 returns"
            ),
        },
        "sample": {
            "n_assets": len(tickers),
            "articles_per_firm": sample_size,
            "tickers": tickers,
            "calibration_start": str(pd.Timestamp(calibration.dates.min()).date()),
            "calibration_end": str(pd.Timestamp(calibration.dates.max()).date()),
            "evaluation_start": str(pd.Timestamp(evaluation.dates.min()).date()),
            "evaluation_end": str(pd.Timestamp(evaluation.dates.max()).date()),
        },
        "cells": records,
        "endpoint_comparison": _endpoint_comparison(records),
        "claim_boundary": {
            "supported": (
                "descriptive representation/transport mechanism evidence and its "
                "propagation into the return operator"
            ),
            "not_supported": (
                "structural linear-versus-convex investment adjustment costs or a "
                "causal investment-convexity test"
            ),
        },
    }


def _write_results(
    paths: Paper5TransportDimensionPaths,
    payload: dict[str, object],
    records: list[dict[str, object]],
    pair_frames: list[pd.DataFrame],
) -> None:
    """Persist the aggregate and pair-level diagnostic surfaces."""
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "results.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame.from_records(records).to_parquet(
        paths.output_dir / "results.parquet", index=False
    )
    pd.concat(pair_frames, ignore_index=True).to_parquet(
        paths.output_dir / "assignment_pairs.parquet", index=False
    )


def run_paper5_transport_dimension_ladder(
    paths: Paper5TransportDimensionPaths,
    config: Paper5TransportDimensionConfig,
) -> dict[str, object]:
    """Run the registered dimension ladder from embedding costs to return scores."""
    config.validate()
    cells = _ladder_cells(paths.params_file, config)
    provider, representations, sample_size = _validate_geometry_contract(
        paths.params_file, cells
    )
    item_ids = _market_item_ids(paths.params_file)
    tickers, loaded_operators, loaded_certificates = _load_ladder_operators(
        paths, cells
    )
    calibration, evaluation = _factor_samples(paths, config, tickers)

    records: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    for cell in cells:
        logger.info(
            "Computing Paper 5 W1/W2 mechanism diagnostic at D=%d",
            cell.effective_dimension,
        )
        clouds = load_provider_clouds(
            provider,
            representations[cell.cell_id],
            embeddings_dir=paths.embeddings_dir,
            item_ids=item_ids,
            sample_size=sample_size,
        )
        if [cloud.item_id for cloud in clouds] != tickers:
            message = f"{cell.cell_id}: cloud and operator ticker orders differ"
            raise Paper5TransportDimensionError(message)
        pair_frame, geometry = _assignment_geometry(
            clouds,
            cell_id=cell.cell_id,
            dimension=cell.effective_dimension,
        )
        pair_frames.append(pair_frame)
        downstream = _operator_return_diagnostics(
            loaded_operators[cell.cell_id], calibration, evaluation
        )
        certificates = loaded_certificates[cell.cell_id]
        records.append(
            {
                "cell_id": cell.cell_id,
                "provider_id": cell.provider_id,
                "representation_id": cell.representation_id,
                "effective_dimension": cell.effective_dimension,
                "native_dimension": cell.native_dimension,
                "representation_transform": "truncate_then_l2_normalize_rows",
                "ground_metric": "euclidean_chord",
                "distance_rescaling": "none",
                **geometry,
                **downstream,
                **{
                    f"{order}_{key}": value
                    for order, certificate in certificates.items()
                    for key, value in certificate.items()
                },
            }
        )

    payload = _result_payload(
        records, tickers, sample_size, calibration, evaluation, config
    )
    _write_results(paths, payload, records, pair_frames)
    return payload


__all__ = [
    "Paper5TransportDimensionConfig",
    "Paper5TransportDimensionError",
    "Paper5TransportDimensionPaths",
    "run_paper5_transport_dimension_ladder",
]

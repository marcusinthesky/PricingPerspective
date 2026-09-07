"""Preregistered Paper 1 original-space neighbour outcome split.

Selection uses the shared 2018--2020 Qwen3-Embedding-8B rooted W2 distance
artifact from the frozen point-in-time sample; outcomes are 2021--2022 simple
returns.

**This is a temporal split, not a holdout.** Only the *selection* input is
date-restricted. The 2021-01-04--2022-12-30 outcome panel is a strict subset of
the in-sample return panel that also feeds ``compute_covariance@window0``,
the in-window covariance stage and ``p1_dyadic_confound``, so no other stage in the
graph is denied these dates. The genuinely out-of-sample lane is the 2023--2026
panel (``compute_returns_oos``, the later covariance cell, ``p1_dyadic_oos``).
The stage was named ``p1_neighbour_holdout`` until t80; "holdout" claimed a
data separation the DAG does not implement.

**The W2 distance is read, not recomputed.** The selection matrix is the shared
``wasserstein_w2_pit_2020`` typed distance artifact. This keeps the selection
metric identical to the Paper 1 primary geometry and avoids a second local
implementation of the transport calculation.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from pipeline._kernels.neighbour_stability import neighbour_order, ticker_ids
from pipeline.io.typed_analysis import (
    StatisticalDistanceSpec,
    load_typed_analysis,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from collections.abc import Buffer
    from pathlib import Path

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2020-12-31")
EVAL_START = pd.Timestamp("2021-01-04")
EVAL_END = pd.Timestamp("2022-12-30")
EXPECTED_EVAL_DATES = 503
# Roster size of the expanded universe. This is a preregistered eligibility
# count, not a mechanical roster echo: the protocol also requires
# MIN_PREPERIOD_EMBEDDINGS pre-2021 embeddings per firm, so the realised
# eligible set must be rederived from the expanded run before this is trusted.
EXPECTED_ELIGIBLE_FIRMS = 100
MIN_PREPERIOD_EMBEDDINGS = 32
MODEL_ID = "qwen/qwen3-embedding-8b"
MIN_SECTOR_PAIR_MEMBERS = 2

#: Identity of the shared component this protocol selects on. The window ID is
#: the load-bearing half: it fixes the preregistered selection vintage.
SELECTION_PROVIDER_ID = "qwen3-embedding-8b"
SELECTION_REPRESENTATION_ID = "qwen3-embedding-8b-unit"
SELECTION_GEOMETRY_ID = "wasserstein_w2_pit_2020"
SELECTION_WINDOW_ID = "pit-2020-12-31"


class _Hasher(Protocol):
    """Describe the digest update operation used by canonical hashing."""

    def update(self, value: Buffer, /) -> None:
        """Add bytes to the digest state."""
        ...


class NeighbourOutcomeSplitError(ValueError):
    """Report a violation of the preregistered outcome-split contract."""


class UnsupportedHashValueError(TypeError):
    """Report a value that has no canonical logical-hash encoding."""


@dataclass(frozen=True)
class _SelectionGeometry:
    """Validated shared rooted-W2 distance backing the pair selection."""

    item_ids: tuple[str, ...]
    metric: np.ndarray
    article_counts: dict[str, int]
    component_summary: dict[str, object]


@dataclass(frozen=True)
class _EligibleUniverse:
    """Resolved classifications and return files for the preregistered universe."""

    symbols: list[str]
    sectors: dict[str, str]
    return_paths: dict[str, Path]
    exclusions: list[dict[str, str]]


@dataclass(frozen=True)
class _PairSelection:
    """Deterministic within-sector nearest-pair selection."""

    sector_members: dict[str, list[str]]
    selected_pairs: list[dict[str, object]]
    singleton_sectors: list[dict[str, object]]


@dataclass(frozen=True)
class _SplitArtifacts:
    """Tables the split analysis computes, persisted or hashed-only."""

    pair_metrics: pd.DataFrame
    returns: pd.DataFrame
    pair_paths: pd.DataFrame


@dataclass(frozen=True)
class _SourcePaths:
    """Input and output locations for one outcome-split run."""

    distance_artifact_dir: Path
    returns_dir: Path
    universe_csv: Path
    params_file: Path
    output_dir: Path


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _hash_text(hasher: _Hasher, value: str) -> None:
    raw = value.encode("utf-8")
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def _hash_float64(hasher: _Hasher, value: float) -> None:
    if not np.isfinite(value):
        message = "cannot hash a non-finite numeric value"
        raise NeighbourOutcomeSplitError(message)
    hasher.update(struct.pack("<d", value))


def _hash_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash a logical table independently of its parquet encoding."""
    hasher = hashlib.sha256()
    for column in columns:
        _hash_text(hasher, column)
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        for value in row:
            if isinstance(value, pd.Timestamp):
                _hash_text(hasher, value.date().isoformat())
            elif isinstance(value, str):
                _hash_text(hasher, value)
            elif isinstance(value, (bool, np.bool_)):
                hasher.update(b"\x01" if bool(value) else b"\x00")
            elif isinstance(value, (int, np.integer)):
                hasher.update(struct.pack("<q", int(value)))
            elif isinstance(value, (float, np.floating)):
                _hash_float64(hasher, float(value))
            elif value is None:
                hasher.update(b"\xff")
            else:
                message = f"unsupported hash value {value!r}"
                raise UnsupportedHashValueError(message)
    return hasher.hexdigest()


def _resolve_geometry_spec(params_file: Path) -> StatisticalDistanceSpec:
    """Resolve the declared selection geometry and assert its preregistered window."""
    config = load_typed_analysis(params_file)
    geometry = config.distances.get(SELECTION_GEOMETRY_ID)
    if not isinstance(geometry, StatisticalDistanceSpec):
        message = f"{SELECTION_GEOMETRY_ID!r} is not a registered statistical geometry"
        raise NeighbourOutcomeSplitError(message)
    if geometry.sample_window != SELECTION_WINDOW_ID:
        message = (
            f"{SELECTION_GEOMETRY_ID!r} declares window "
            f"{geometry.sample_window!r}, expected {SELECTION_WINDOW_ID!r}"
        )
        raise NeighbourOutcomeSplitError(message)
    window = config.window_for(geometry)
    if window is None:
        message = f"{SELECTION_GEOMETRY_ID!r} resolved to no window"
        raise NeighbourOutcomeSplitError(message)
    # The protocol is preregistered on dates, the artifact is keyed on a window
    # ID; this is the one place the two are tied together. Editing the registry
    # window must fail here rather than silently reselect the pairs.
    if (
        window.start is None
        or window.end is None
        or pd.Timestamp(window.start) != TRAIN_START
        or pd.Timestamp(window.end) != TRAIN_END
    ):
        message = (
            f"window {SELECTION_WINDOW_ID!r} is {window.start}..{window.end}, "
            f"but the protocol preregisters "
            f"{TRAIN_START.date()}..{TRAIN_END.date()}"
        )
        raise NeighbourOutcomeSplitError(message)
    return geometry


def _load_selection_geometry(paths: _SourcePaths) -> _SelectionGeometry:
    """Read and validate the shared windowed rooted W2 distance artifact."""
    geometry = _resolve_geometry_spec(paths.params_file)
    if (
        geometry.family != "wasserstein"
        or geometry.estimator != "balanced_wasserstein_2"
        or geometry.normalization != "rooted"
        or geometry.value_semantics != "statistical_distance"
    ):
        message = (
            f"{SELECTION_GEOMETRY_ID!r} must be rooted balanced W2, got "
            f"{geometry.family}/{geometry.estimator}/{geometry.normalization}"
        )
        raise NeighbourOutcomeSplitError(message)
    frame, artifact_summary = read_typed_distance_artifact(
        paths.distance_artifact_dir,
        expected_identity={
            "provider_id": SELECTION_PROVIDER_ID,
            "representation_id": SELECTION_REPRESENTATION_ID,
            "distance_id": SELECTION_GEOMETRY_ID,
            "value_semantics": "statistical_distance",
        },
    )
    item_ids_value = artifact_summary.get("item_ids")
    metadata_value = artifact_summary.get("metadata")
    if not isinstance(item_ids_value, list) or not isinstance(metadata_value, dict):
        message = "distance artifact has malformed item or metadata declarations"
        raise NeighbourOutcomeSplitError(message)
    item_ids = tuple(str(item) for item in item_ids_value)
    metadata = metadata_value
    if metadata.get("sample_window") != SELECTION_WINDOW_ID:
        message = (
            f"distance artifact carries window {metadata.get('sample_window')!r}, "
            f"expected {SELECTION_WINDOW_ID!r}"
        )
        raise NeighbourOutcomeSplitError(message)
    if metadata.get("provider_model_id") != MODEL_ID:
        message = (
            f"distance artifact carries model {metadata.get('provider_model_id')!r}, "
            f"expected {MODEL_ID!r}"
        )
        raise NeighbourOutcomeSplitError(message)
    if metadata.get("representation_transform") != "l2_normalize_rows":
        message = (
            "the protocol selects on unit-normalized rows, but the distance "
            f"artifact carries transform {metadata.get('representation_transform')!r}"
        )
        raise NeighbourOutcomeSplitError(message)
    sample_size = metadata.get("sample_size")
    if not isinstance(sample_size, int) or sample_size < MIN_PREPERIOD_EMBEDDINGS:
        message = (
            "the point-in-time W2 artifact must use a sample size of at least "
            f"{MIN_PREPERIOD_EMBEDDINGS}"
        )
        raise NeighbourOutcomeSplitError(message)
    counts = dict.fromkeys(item_ids, sample_size)
    if len(frame) != len(item_ids) ** 2:
        message = "distance artifact coverage does not match its item roster"
        raise NeighbourOutcomeSplitError(message)
    metric = (
        frame["value"].to_numpy(dtype=np.float64).reshape(len(item_ids), len(item_ids))
    )
    return _SelectionGeometry(
        item_ids=item_ids,
        metric=metric,
        article_counts=counts,
        component_summary={
            "distance_id": metadata["distance_id"],
            "family": metadata["family"],
            "provider_id": metadata["provider_id"],
            "representation_id": metadata["representation_id"],
            "sample_design": metadata["sample_design"],
            "sample_window": metadata["sample_window"],
            "ground_distance": metadata["ground_distance"],
            "exponent": geometry.exponent,
            "estimator": metadata["estimator"],
            "normalization": metadata["normalization"],
            "sample_size": sample_size,
            "input_hashes": list(metadata["input_hashes"]),
        },
    )


def _load_return_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"Date", "ticker", "return"}
    missing = required - set(frame.columns)
    if missing:
        message = f"{path}: missing return columns {sorted(missing)}"
        raise NeighbourOutcomeSplitError(message)
    frame = frame.loc[:, ["Date", "ticker", "return"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    frame = frame[frame["Date"].between(EVAL_START, EVAL_END, inclusive="both")]
    frame = frame.sort_values("Date", kind="mergesort").reset_index(drop=True)
    if (
        len(frame) != EXPECTED_EVAL_DATES
        or frame["Date"].nunique() != EXPECTED_EVAL_DATES
    ):
        message = (
            f"{path}: expected {EXPECTED_EVAL_DATES} unique evaluation dates, "
            f"found {len(frame)} rows/{frame['Date'].nunique()} unique"
        )
        raise NeighbourOutcomeSplitError(message)
    if frame["Date"].iloc[0] != EVAL_START or frame["Date"].iloc[-1] != EVAL_END:
        message = f"{path}: evaluation endpoints do not match the protocol"
        raise NeighbourOutcomeSplitError(message)
    tickers = frame["ticker"].astype(str).unique().tolist()
    if tickers != [path.stem]:
        message = f"{path}: ticker column does not identify {path.stem}"
        raise NeighbourOutcomeSplitError(message)
    returns = frame["return"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(returns)) or np.any(returns <= -1.0):
        message = f"{path}: returns are non-finite or outside log1p support"
        raise NeighbourOutcomeSplitError(message)
    frame["return"] = returns
    frame["log_return"] = np.log1p(returns)
    frame["cumulative_log_return"] = frame["log_return"].cumsum()
    return frame.rename(columns={"Date": "date", "ticker": "symbol"})


def _outcome_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    gap = left - right
    return float(np.sqrt(np.mean(np.square(gap)))), float(abs(gap[-1]))


def _comparative_percentile(
    pair_rows: pd.DataFrame, symbol1: str, symbol2: str
) -> tuple[int, int, float | None]:
    ordered = pair_rows.sort_values(
        ["rms_cumulative_log_gap", "symbol1", "symbol2"], kind="mergesort"
    ).reset_index(drop=True)
    hit = ordered.index[
        (ordered["symbol1"] == symbol1) & (ordered["symbol2"] == symbol2)
    ].tolist()
    if len(hit) != 1:
        message = f"selected pair {symbol1}-{symbol2} missing from comparator"
        raise NeighbourOutcomeSplitError(message)
    rank = int(hit[0]) + 1
    n_pairs = len(ordered)
    percentile = None if n_pairs == 1 else 100.0 * (rank - 1) / (n_pairs - 1)
    return rank, n_pairs, percentile


def _resolve_eligible_universe(
    paths: _SourcePaths, geometry: _SelectionGeometry
) -> _EligibleUniverse:
    """Validate the universe against the priced component and return files.

    The component's item roster *is* the priced universe (``market_symbols``),
    so eligibility is read from it rather than from the raw embedding tree.
    Missing geometry or return inputs are data faults, not preregistered
    carve-outs.
    """
    universe = pd.read_csv(paths.universe_csv, dtype=str)
    required_universe = {"Symbol", "Sector"}
    if required_universe - set(universe.columns):
        message = "universe CSV must contain Symbol and Sector"
        raise NeighbourOutcomeSplitError(message)
    universe = universe.loc[:, ["Symbol", "Sector"]].copy()
    universe["Symbol"] = universe["Symbol"].astype(str)
    if universe["Symbol"].duplicated().any():
        message = "universe contains duplicate symbols"
        raise NeighbourOutcomeSplitError(message)

    priced = set(geometry.item_ids)
    return_paths = {path.stem: path for path in paths.returns_dir.glob("*.parquet")}
    exclusions: list[dict[str, str]] = []
    eligible: list[str] = []
    sectors: dict[str, str] = {}
    for row in universe.sort_values("Symbol").itertuples(index=False):
        symbol = str(row.Symbol)
        sector = str(row.Sector).strip()
        reasons: list[str] = []
        if not sector or sector.lower() == "nan":
            reasons.append("missing_sector")
        if symbol not in priced:
            reasons.append("missing_w2_distance")
        if symbol not in return_paths:
            reasons.append("missing_return_file")
        if reasons:
            exclusions.append({"symbol": symbol, "reason": ",".join(reasons)})
        else:
            eligible.append(symbol)
            sectors[symbol] = sector

    if len(eligible) != EXPECTED_ELIGIBLE_FIRMS:
        message = (
            f"preregistered eligible count is {EXPECTED_ELIGIBLE_FIRMS}, "
            f"found {len(eligible)}"
        )
        raise NeighbourOutcomeSplitError(message)
    # Mirrors params.yaml:market_symbol_exclusions. An exclusion here means a
    # firm lost its W2 distance or return file and must be surfaced as a data
    # fault rather than accepted as a preregistered carve-out.
    expected_exclusions: list[dict[str, str]] = []
    if exclusions != expected_exclusions:
        message = f"preregistered exclusion changed: {exclusions}"
        raise NeighbourOutcomeSplitError(message)
    return _EligibleUniverse(
        symbols=eligible,
        sectors=sectors,
        return_paths=return_paths,
        exclusions=exclusions,
    )


def _load_returns_panel(universe: _EligibleUniverse) -> dict[str, pd.DataFrame]:
    """Load outcome returns and enforce a common evaluation-date panel."""
    returns_by_symbol: dict[str, pd.DataFrame] = {}
    reference_dates: pd.Series | None = None
    for symbol in universe.symbols:
        return_frame = _load_return_frame(universe.return_paths[symbol])
        if reference_dates is None:
            reference_dates = return_frame["date"]
        elif (
            not return_frame["date"]
            .reset_index(drop=True)
            .equals(reference_dates.reset_index(drop=True))
        ):
            message = f"{symbol}: evaluation dates differ from the common panel"
            raise NeighbourOutcomeSplitError(message)
        returns_by_symbol[symbol] = return_frame
    return returns_by_symbol


def _compute_pair_metrics(
    universe: _EligibleUniverse,
    geometry: _SelectionGeometry,
    returns_by_symbol: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join the shared W2 metric to the outcome distances for every pair."""
    position = {item: index for index, item in enumerate(geometry.item_ids)}
    pair_records: list[dict[str, object]] = []
    for symbol1, symbol2 in combinations(universe.symbols, 2):
        cumulative1 = returns_by_symbol[symbol1]["cumulative_log_return"].to_numpy()
        cumulative2 = returns_by_symbol[symbol2]["cumulative_log_return"].to_numpy()
        rms_gap, terminal_gap = _outcome_metrics(cumulative1, cumulative2)
        sector1 = universe.sectors[symbol1]
        sector2 = universe.sectors[symbol2]
        pair_records.append(
            {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "sector1": sector1,
                "sector2": sector2,
                "same_sector": sector1 == sector2,
                "w2_distance": float(
                    geometry.metric[position[symbol1], position[symbol2]]
                ),
                "rms_cumulative_log_gap": rms_gap,
                "terminal_absolute_log_gap": terminal_gap,
                "n_preperiod1": geometry.article_counts[symbol1],
                "n_preperiod2": geometry.article_counts[symbol2],
                "n_evaluation_dates": EXPECTED_EVAL_DATES,
            }
        )
    pair_metrics = pd.DataFrame(pair_records).sort_values(
        ["symbol1", "symbol2"], kind="mergesort"
    )
    expected_pairs = EXPECTED_ELIGIBLE_FIRMS * (EXPECTED_ELIGIBLE_FIRMS - 1) // 2
    if len(pair_metrics) != expected_pairs:
        message = f"expected {expected_pairs} pair metrics, found {len(pair_metrics)}"
        raise NeighbourOutcomeSplitError(message)
    return pair_metrics


def _partition_sector_members(universe: _EligibleUniverse) -> dict[str, list[str]]:
    """Partition the governed universe using its own sector taxonomy.

    Sector labels and sparse-sector membership belong to ``data/universe.csv``.
    Deriving the partition here avoids a second roster embedded in stage code;
    the repository test pins the current nine-sector taxonomy independently.
    """
    sector_members: dict[str, list[str]] = {}
    for symbol in universe.symbols:
        sector_members.setdefault(universe.sectors[symbol], []).append(symbol)
    return {
        sector: sorted(members) for sector, members in sorted(sector_members.items())
    }


def _nearest_same_sector_pairs(
    universe: _EligibleUniverse, geometry: _SelectionGeometry
) -> dict[str, tuple[str, str]]:
    """Rank neighbours once, then read each sector's nearest pair off the ranking.

    Delegates the ordering to :func:`jcor.geometry.neighbour.neighbour_order`
    -- the same kernel behind ``p1_encoder_neighbour_stability`` and the
    ``dimensionality_analysis`` neighbour table -- rather than re-sorting a
    pair frame. Its tie-break is ascending ``(distance, object id)`` on ids
    that are ranks in the sorted ticker list, so walking anchors in id order
    reproduces the preregistered ``(w2_distance, symbol1, symbol2)`` sort
    exactly: the minimal-distance pair is reported from both endpoints, and the
    smaller endpoint wins the anchor tie.
    """
    symbols = list(geometry.item_ids)
    order = np.asarray(neighbour_order(ticker_ids(symbols), geometry.metric))
    sector_of = universe.sectors
    nearest: dict[str, tuple[str, str]] = {}
    best: dict[str, tuple[float, int, int]] = {}
    for anchor_index, anchor in enumerate(symbols):
        sector = sector_of.get(anchor)
        if sector is None:
            continue
        for neighbour_index in order[anchor_index]:
            neighbour = symbols[int(neighbour_index)]
            if sector_of.get(neighbour) != sector:
                continue
            key = (
                float(geometry.metric[anchor_index, int(neighbour_index)]),
                anchor_index,
                int(neighbour_index),
            )
            if sector not in best or key < best[sector]:
                best[sector] = key
                nearest[sector] = (
                    min(anchor, neighbour),
                    max(anchor, neighbour),
                )
            break
    return nearest


def _select_sector_pairs(
    universe: _EligibleUniverse,
    geometry: _SelectionGeometry,
    pair_metrics: pd.DataFrame,
) -> _PairSelection:
    """Select the nearest embedding pair within each nonsingleton sector."""
    sector_members = _partition_sector_members(universe)
    pairable_sectors = {
        sector
        for sector, members in sector_members.items()
        if len(members) >= MIN_SECTOR_PAIR_MEMBERS
    }
    nearest = _nearest_same_sector_pairs(universe, geometry)
    nearest_sectors = set(nearest)
    if nearest_sectors != pairable_sectors:
        message = (
            "same-sector nearest-pair coverage does not match the governed "
            f"partition: missing={sorted(pairable_sectors - nearest_sectors)}, "
            f"unexpected={sorted(nearest_sectors - pairable_sectors)}"
        )
        raise NeighbourOutcomeSplitError(message)

    selected_pairs: list[dict[str, object]] = []
    singleton_sectors: list[dict[str, object]] = []
    for sector, members in sector_members.items():
        if len(members) < MIN_SECTOR_PAIR_MEMBERS:
            singleton_sectors.append({"sector": sector, "members": members})
            continue
        symbol1, symbol2 = nearest[sector]
        within = pair_metrics[
            (pair_metrics["sector1"] == sector) & (pair_metrics["sector2"] == sector)
        ].copy()
        row = within[
            (within["symbol1"] == symbol1) & (within["symbol2"] == symbol2)
        ].iloc[0]
        rank, comparator_count, percentile = _comparative_percentile(
            within, symbol1, symbol2
        )
        selected_pairs.append(
            {
                "sector": sector,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "w2_distance": float(row["w2_distance"]),
                "rms_cumulative_log_gap": float(row["rms_cumulative_log_gap"]),
                "terminal_absolute_log_gap": float(row["terminal_absolute_log_gap"]),
                "within_sector_rms_gap_rank": rank,
                "within_sector_pair_count": comparator_count,
                "within_sector_rms_gap_percentile": percentile,
                "n_preperiod1": int(row["n_preperiod1"]),
                "n_preperiod2": int(row["n_preperiod2"]),
            }
        )

    selected_sectors = {str(row["sector"]) for row in selected_pairs}
    if selected_sectors != pairable_sectors:
        message = (
            "selected sectors do not match the governed pairable partition: "
            f"missing={sorted(pairable_sectors - selected_sectors)}, "
            f"unexpected={sorted(selected_sectors - pairable_sectors)}"
        )
        raise NeighbourOutcomeSplitError(message)
    return _PairSelection(sector_members, selected_pairs, singleton_sectors)


def _build_artifacts(
    universe: _EligibleUniverse,
    returns_by_symbol: dict[str, pd.DataFrame],
    pair_metrics: pd.DataFrame,
    selection: _PairSelection,
) -> _SplitArtifacts:
    """Assemble deterministically ordered tables for persistence and hashing."""
    returns = pd.concat(
        [returns_by_symbol[symbol] for symbol in universe.symbols],
        ignore_index=True,
    ).sort_values(["symbol", "date"], kind="mergesort")
    path_records: list[pd.DataFrame] = []
    for selected in selection.selected_pairs:
        symbol1 = str(selected["symbol1"])
        symbol2 = str(selected["symbol2"])
        left = returns_by_symbol[symbol1]
        right = returns_by_symbol[symbol2]
        cumulative1 = left["cumulative_log_return"].to_numpy()
        cumulative2 = right["cumulative_log_return"].to_numpy()
        path_records.append(
            pd.DataFrame(
                {
                    "sector": str(selected["sector"]),
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "date": left["date"].to_numpy(),
                    "cumulative_log_return1": cumulative1,
                    "cumulative_log_return2": cumulative2,
                    "cumulative_log_gap": cumulative1 - cumulative2,
                }
            )
        )
    pair_paths = pd.concat(path_records, ignore_index=True).sort_values(
        ["sector", "date"], kind="mergesort"
    )
    return _SplitArtifacts(pair_metrics, returns, pair_paths)


def _persist_artifacts(output_dir: Path, artifacts: _SplitArtifacts) -> None:
    """Persist the one tabular artifact the graph consumes.

    ``pair_metrics`` and the concatenated ``returns`` panel are deliberately
    *not* written. No stage declared them as a dependency, and the returns
    table was a re-slice of ``${shared}/returns`` restricted to dates that
    every other Paper 1 stage already reads. Their logical SHA-256s are still
    recorded in ``provenance.json``, so dropping the files loses the bytes and
    keeps the evidence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.pair_paths.to_parquet(output_dir / "pair_paths.parquet", index=False)


def _build_documents(
    paths: _SourcePaths,
    universe: _EligibleUniverse,
    geometry: _SelectionGeometry,
    selection: _PairSelection,
    artifacts: _SplitArtifacts,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build selection, summary, and provenance documents with logical hashes."""
    input_hashes = {
        "selection_distance_artifact": _hash_frame(
            pd.DataFrame(
                {
                    "item": list(geometry.item_ids),
                    "count": [
                        geometry.article_counts[item] for item in geometry.item_ids
                    ],
                }
            ),
            ["item", "count"],
        ),
        "eligible_evaluation_returns": _hash_frame(
            artifacts.returns, list(artifacts.returns.columns)
        ),
        "eligible_sector_membership": hashlib.sha256(
            _canonical_json_bytes(
                {
                    symbol: universe.sectors[symbol]
                    for symbol in sorted(universe.symbols)
                }
            )
        ).hexdigest(),
    }
    output_hashes = {
        "pair_metrics": _hash_frame(
            artifacts.pair_metrics, list(artifacts.pair_metrics.columns)
        ),
        "returns": _hash_frame(artifacts.returns, list(artifacts.returns.columns)),
        "selected_pair_paths": _hash_frame(
            artifacts.pair_paths, list(artifacts.pair_paths.columns)
        ),
    }
    selection_document: dict[str, object] = {
        "eligible_universe": universe.symbols,
        "exclusions": universe.exclusions,
        "sector_members": {
            key: sorted(value) for key, value in selection.sector_members.items()
        },
        "singleton_sectors": selection.singleton_sectors,
        "selected_pairs": selection.selected_pairs,
        "selection_sort": ["w2_distance", "symbol1", "symbol2"],
    }
    summary: dict[str, object] = {
        "protocol_id": "paper1-neighbour-outcome-split-v2",
        "model": MODEL_ID,
        "selection_distance": geometry.component_summary,
        "training_window": {
            "start": TRAIN_START.date().isoformat(),
            "end": TRAIN_END.date().isoformat(),
        },
        "evaluation_window": {
            "start": EVAL_START.date().isoformat(),
            "end": EVAL_END.date().isoformat(),
        },
        "n_eligible_firms": len(universe.symbols),
        "n_pair_metrics": len(artifacts.pair_metrics),
        "n_selected_pairs": len(selection.selected_pairs),
        "n_evaluation_dates": EXPECTED_EVAL_DATES,
        "minimum_preperiod_embeddings": min(
            geometry.article_counts[symbol] for symbol in universe.symbols
        ),
        "selected_pairs": selection.selected_pairs,
        "singleton_sectors": selection.singleton_sectors,
        "interpretation_scope": "descriptive outcome split only",
        "not_strict_vintage": True,
        "evaluation_window_is_in_sample": True,
    }
    provenance: dict[str, object] = {
        "source_paths": {
            "distance_artifact_dir": str(paths.distance_artifact_dir),
            "returns_dir": str(paths.returns_dir),
            "universe_csv": str(paths.universe_csv),
        },
        "model_definition": (
            "Qwen3-Embedding-8B rows from the frozen point-in-time 2018-2020 "
            "sample; the selection artifact uses a deterministic sample of 128 "
            "rows per firm"
        ),
        "normalization_definition": "each full-width embedding row is L2-normalized",
        "metric_definition": (
            "rooted balanced quadratic Wasserstein distance with Euclidean chord "
            "ground distance, read from typed distance "
            f"{SELECTION_GEOMETRY_ID!r} over window {SELECTION_WINDOW_ID!r}; "
            "not recomputed by this stage"
        ),
        "return_definition": (
            "simple daily returns transformed with log1p and cumulatively summed on "
            "the common 503-date 2021-2022 panel"
        ),
        "evaluation_window_definition": (
            "a strict subset of the in-sample 2018-2022 return panel also consumed "
            "by compute_covariance@window0 and "
            "p1_dyadic_confound; only the selection input is date-restricted"
        ),
        "comparator_definition": (
            "rank selected-pair RMS cumulative-log-return gap among all same-sector "
            "pairs; percentile is 100*(rank-1)/(N-1), so lower is closer; undefined "
            "when N=1"
        ),
        "input_array_sha256s": input_hashes,
        "output_array_sha256s": output_hashes,
        "logical_hash_encoding": (
            "SHA256 over length-prefixed UTF-8 labels and little-endian numeric values "
            "in the persisted deterministic row/column order"
        ),
    }
    return selection_document, summary, provenance


def run_neighbour_outcome_split(
    distance_artifact_dir: Path,
    returns_dir: Path,
    universe_csv: Path,
    output_dir: Path,
    params_file: Path,
) -> dict[str, object]:
    """Materialize the preregistered original-space neighbour outcome split."""
    paths = _SourcePaths(
        distance_artifact_dir, returns_dir, universe_csv, params_file, output_dir
    )
    geometry = _load_selection_geometry(paths)
    universe = _resolve_eligible_universe(paths, geometry)
    returns_by_symbol = _load_returns_panel(universe)
    pair_metrics = _compute_pair_metrics(universe, geometry, returns_by_symbol)

    selection = _select_sector_pairs(universe, geometry, pair_metrics)
    artifacts = _build_artifacts(universe, returns_by_symbol, pair_metrics, selection)
    _persist_artifacts(output_dir, artifacts)
    selection_document, summary, provenance = _build_documents(
        paths, universe, geometry, selection, artifacts
    )
    _write_canonical_json(output_dir / "selection.json", selection_document)
    _write_canonical_json(output_dir / "summary.json", summary)
    _write_canonical_json(output_dir / "provenance.json", provenance)
    return summary

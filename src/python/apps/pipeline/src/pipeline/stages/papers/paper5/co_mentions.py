"""Persistent news co-mention comparator operator for Paper 5."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper5.baskets import Paper5OperatorError

if TYPE_CHECKING:
    from pathlib import Path


class _CoMentionDiagnostics(TypedDict):
    """Validated metadata emitted with the co-mention comparator matrix."""

    construction: str
    retained_undirected_pairs: int
    retained_pair_weight: float
    row_stochastic_max_dev: float
    zero_diagonal_max_abs: float
    certified: bool


def load_co_mentions_w(
    adjacency_path: Path, tickers: list[str]
) -> tuple[np.ndarray, _CoMentionDiagnostics]:
    """Load persistent co-mention counts as a row-stochastic comparator.

    The shared artifact stores each undirected pair once. Paper 5 intersects
    those edges with its priced universe, mirrors the weights, and normalizes
    each row. Edges to the one shared-substrate ticker outside the priced
    universe are dropped rather than folded into row mass.
    """
    edges = pd.read_parquet(adjacency_path)
    required = {"firm_i", "firm_j", "weight"}
    missing = required.difference(edges.columns)
    if missing:
        message = f"co-mention adjacency missing columns {sorted(missing)}"
        raise Paper5OperatorError(message)

    selected = edges.loc[:, ["firm_i", "firm_j", "weight"]].copy()
    if selected[["firm_i", "firm_j"]].isna().any().any():
        message = "co-mention adjacency contains missing firm identifiers"
        raise Paper5OperatorError(message)
    selected["firm_i"] = selected["firm_i"].astype(str).str.upper()
    selected["firm_j"] = selected["firm_j"].astype(str).str.upper()
    edge_weights = selected["weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(edge_weights)) or np.any(edge_weights <= 0.0):
        message = "co-mention adjacency weights must be finite and strictly positive"
        raise Paper5OperatorError(message)
    if bool((selected["firm_i"] == selected["firm_j"]).any()):
        message = "co-mention adjacency must not contain self-edges"
        raise Paper5OperatorError(message)

    pair_keys = selected.apply(
        lambda row: tuple(sorted((row["firm_i"], row["firm_j"]))), axis=1
    )
    if bool(pair_keys.duplicated().any()):
        message = "co-mention adjacency contains duplicate undirected pairs"
        raise Paper5OperatorError(message)

    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    matrix = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    retained_pairs = 0
    retained_weight = 0.0
    for left_name, right_name, edge_weight in zip(
        selected["firm_i"].astype(str),
        selected["firm_j"].astype(str),
        edge_weights,
        strict=True,
    ):
        left = ticker_index.get(left_name)
        right = ticker_index.get(right_name)
        if left is None or right is None:
            continue
        weight = float(edge_weight)
        matrix[left, right] = weight
        matrix[right, left] = weight
        retained_pairs += 1
        retained_weight += weight

    row_sums = matrix.sum(axis=1)
    isolated = [
        ticker for ticker, total in zip(tickers, row_sums, strict=True) if total <= 0
    ]
    if isolated:
        message = f"co-mention adjacency isolates priced tickers {isolated}"
        raise Paper5OperatorError(message)
    matrix /= row_sums[:, None]
    diagnostics: _CoMentionDiagnostics = {
        "construction": "persistent_weighted_news_co_mentions",
        "retained_undirected_pairs": retained_pairs,
        "retained_pair_weight": retained_weight,
        "row_stochastic_max_dev": float(np.max(np.abs(matrix.sum(axis=1) - 1.0))),
        "zero_diagonal_max_abs": float(np.max(np.abs(np.diag(matrix)))),
        "certified": False,
    }
    return matrix, diagnostics


def compute_co_mentions(
    events_file: Path,
    universe_csv: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    min_persistence_years: int,
) -> None:
    """Aggregate the frozen exact-two-firm events with the original year rule."""
    events = pd.read_parquet(events_file)
    required = {"article_id", "created_date", "publication_year", "firm_i", "firm_j"}
    if set(events.columns) != required or min_persistence_years < 1:
        message = "co-mention inputs require the text-free exact-two schema"
        raise ValueError(message)
    universe = pd.read_csv(universe_csv)["Symbol"].str.upper()
    firms = set(universe)
    if not set(events["firm_i"]).union(events["firm_j"]).issubset(firms):
        message = "co-mention event refers to a firm outside the universe"
        raise ValueError(message)
    dates = pd.to_datetime(events["created_date"], errors="raise", utc=True)
    if dates.isna().any() or not (dates.dt.year == events["publication_year"]).all():
        message = "co-mention publication years differ from their dates"
        raise ValueError(message)
    events = events.loc[
        (dates >= pd.Timestamp(start_date, tz="UTC"))
        & (dates <= pd.Timestamp(end_date, tz="UTC"))
    ]
    events = events.sort_values(["created_date", "article_id", "firm_i", "firm_j"])
    pair_year = (
        events.groupby(["firm_i", "firm_j", "publication_year"], sort=True)
        .size()
        .reset_index(name="co_mention_count")
    )
    adjacency = (
        pair_year.groupby(["firm_i", "firm_j"], sort=True)
        .agg(
            weight=("co_mention_count", "sum"),
            n_years=("publication_year", "size"),
            first_year=("publication_year", "min"),
            last_year=("publication_year", "max"),
        )
        .reset_index()
    )
    adjacency = adjacency.loc[adjacency["n_years"] >= min_persistence_years]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("events", events),
        ("pair_year", pair_year),
        ("adjacency", adjacency),
    ):
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    summary = {
        "artifact_type": "news_co_mentions",
        "start_date": start_date,
        "end_date": end_date,
        "min_persistence_years": min_persistence_years,
        "n_nodes": len(universe),
        "exact_two_articles": len(events),
        "observed_pairs": len(events[["firm_i", "firm_j"]].drop_duplicates()),
        "pair_mentions": int(pair_year["co_mention_count"].sum()),
        "persistent_pairs": len(adjacency),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


__all__ = ["compute_co_mentions", "load_co_mentions_w"]

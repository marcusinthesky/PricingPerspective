"""Shared priced-universe and aligned-return-panel construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


#: Parquet metadata key written by ``scripts/strip_close_price``.
SEAM_RETURN_KEY = b"seam_return"


def read_seam_return(path: Path) -> float | None:
    """Return the OOS seam return against the last in-sample close, if recorded.

    The return panels carry no close price -- it is the withheld vendor series --
    so the seam return cannot be recomputed from the published files. It is
    recorded in the OOS panel's Parquet metadata instead of its leading row,
    which stays null so that single-window readers keep dropping it from the
    complete-case panel.
    """
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(SEAM_RETURN_KEY)
    return None if raw is None else float(raw)


def load_aligned_return_panel(
    returns_dir: Path,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    oos_returns_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dense daily-return panel in the requested ticker order.

    Per-ticker files are inner-joined on ``Date``. When an out-of-sample
    directory is supplied, prices are joined before returns are recomputed so
    the first evaluation date uses the final training price.
    """
    con = duckdb.connect(":memory:")
    frames: list[pd.DataFrame] = []
    try:
        for ticker in tickers:
            path = returns_dir / f"{ticker}.parquet"
            if not path.exists():
                message = (
                    f"No return panel for ticker {ticker!r} at {path}; "
                    "pre-filter to the priced intersection before calling "
                    "load_aligned_return_panel."
                )
                raise FileNotFoundError(message)
            if oos_returns_dir is None:
                frame = con.execute(
                    "SELECT Date, return FROM read_parquet(?) ORDER BY Date",
                    [str(path)],
                ).df()
            else:
                oos_path = oos_returns_dir / f"{ticker}.parquet"
                if not oos_path.exists():
                    message = (
                        f"No OOS return panel for ticker {ticker!r} at {oos_path}; "
                        "every in-sample ticker must also have an OOS parquet."
                    )
                    raise FileNotFoundError(message)
                # The panels carry no close price -- it is the withheld vendor
                # series (see `scripts/strip_close_price`). The OOS panel's
                # leading row instead stores the seam return measured against
                # the last in-sample close, so concatenating the stored returns
                # reproduces the unioned series exactly.
                train = con.execute(
                    "SELECT Date, return FROM read_parquet(?) ORDER BY Date",
                    [str(path)],
                ).df()
                evaluation = con.execute(
                    "SELECT Date, return FROM read_parquet(?) ORDER BY Date",
                    [str(oos_path)],
                ).df()
                seam = read_seam_return(oos_path)
                if seam is not None and not evaluation.empty:
                    evaluation.loc[evaluation.index[0], "return"] = seam
                frame = (
                    pd.concat([train, evaluation], ignore_index=True)
                    .drop_duplicates(subset="Date", keep="first")
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

            if start:
                frame = frame[frame["Date"] >= pd.Timestamp(start)]
            if end:
                frame = frame[frame["Date"] <= pd.Timestamp(end)]
            frames.append(frame.rename(columns={"return": ticker}).set_index("Date"))
    finally:
        con.close()

    if not frames:
        message = "tickers must contain at least one identifier"
        raise ValueError(message)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.join(frame, how="inner")
    panel = panel.dropna(how="any")[tickers]
    return panel.index.to_numpy(), panel.to_numpy(dtype=np.float64)


def load_priced_distance_universe(
    distance_artifact_dir: Path,
    returns_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Restrict a governed typed-distance artifact to priced identifiers."""
    frame, summary = read_typed_distance_artifact(
        distance_artifact_dir,
        expected_identity={},
    )
    item_ids = summary.get("item_ids")
    metadata = summary.get("metadata")
    if not isinstance(item_ids, list) or not all(
        isinstance(item_id, str) for item_id in item_ids
    ):
        message = "typed distance artifact has malformed item identifiers"
        raise ValueError(message)
    if not isinstance(metadata, dict):
        message = "typed distance artifact has malformed metadata"
        raise TypeError(message)
    all_items = cast("list[str]", item_ids)
    priced = [
        item_id
        for item_id in all_items
        if (returns_dir / f"{item_id}.parquet").exists()
    ]
    excluded = sorted(set(all_items) - set(priced))
    if excluded:
        logger.info("excluding unpriced typed-distance items %s", excluded)

    raw_full = (
        frame["value"]
        .to_numpy(dtype=np.float64)
        .reshape(len(all_items), len(all_items))
    )
    indices = [all_items.index(item_id) for item_id in priced]
    raw = np.array(raw_full, dtype=np.float64, copy=True)[np.ix_(indices, indices)]
    value_semantics = metadata.get("value_semantics")
    if value_semantics == "squared_statistical_distance":
        squared = raw
        distance = np.sqrt(np.maximum(squared, 0.0))
    elif value_semantics == "statistical_distance":
        distance = raw
        squared = np.square(distance)
    else:
        message = f"unsupported distance value semantics: {value_semantics}"
        raise ValueError(message)
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(squared, 0.0)
    return priced, distance, squared

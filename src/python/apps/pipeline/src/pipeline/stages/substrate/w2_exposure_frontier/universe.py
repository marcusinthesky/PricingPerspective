"""Priced-universe construction for the H1 identification frontier."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from pipeline.stages.substrate.energy_shared import _load_typed_distances

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(".".join((*__name__.split(".")[:-2], "pilot")))


def load_universe(
    distance_artifact_dir: Path,
    returns_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the text-distance matrix and restrict to the 100 priced tickers.

    ``D`` is the governed text distance from ``distance_artifact_dir``. For the
    Paper-2 primary W2 artifact, ``D2`` is its elementwise square; the legacy
    energy-comparator artifact instead stores squared Hilbertian distances and
    is square-rooted. No embeddings are read here. Tickers without a
    ``<ticker>.parquet`` return panel are dropped defensively. Under the current
    governed roster, the text and priced universes coincide at 100 firms, so
    this projection is the identity and the frontier uses the same firms as the
    other price-dependent stages (plan Requirement 3).

    Args:
        distance_artifact_dir: Validated typed baseline distance artifact directory.
        returns_dir: Directory of per-ticker return parquets; a ticker is
            kept iff ``<returns_dir>/<ticker>.parquet`` exists.

    Returns:
        Tuple ``(tickers, d, d2)``: the priced ticker list (sorted), the
        ``(n, n)`` text-distance matrix ``d`` and its squared matrix ``d2``.

    """
    all_tickers, raw_distance, summary = _load_typed_distances(distance_artifact_dir)

    priced = [t for t in all_tickers if (returns_dir / f"{t}.parquet").exists()]
    excluded = sorted(set(all_tickers) - set(priced))
    if excluded:
        logger.info("load_universe: excluding text-only tickers %s", excluded)

    idx = [all_tickers.index(t) for t in priced]
    raw = np.array(raw_distance, dtype=np.float64, copy=True)[np.ix_(idx, idx)]
    metadata = cast("dict[str, object]", summary["metadata"])
    value_semantics = metadata["value_semantics"]
    if value_semantics == "squared_statistical_distance":
        d2 = raw
        d = np.sqrt(np.maximum(d2, 0.0))
    elif value_semantics == "statistical_distance":
        d = raw
        d2 = np.square(d)
    else:
        message = f"unsupported H1 distance value semantics: {value_semantics}"
        raise ValueError(message)
    np.fill_diagonal(d2, 0.0)
    np.fill_diagonal(d, 0.0)

    return priced, d, d2

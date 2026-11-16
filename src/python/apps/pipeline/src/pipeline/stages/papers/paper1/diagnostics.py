"""Paper 1 Matryoshka distance-preservation diagnostic.

The manuscript truncates embeddings to k=64 with almost no change in the
return association. The claim under test is about the *distance matrix*, so
this stage aligns each truncated W2 matrix with the full-width one over the
same dyads and records their agreement. An earlier version plotted the
variance-explained spectrum of the pooled embedding matrix instead; that was
the wrong diagnostic (the leading 64 Matryoshka coordinates carry ~3% of raw
variance, which says nothing about whether the induced geometry survives
truncation). Agreement is partial (Spearman ~0.75 at k=64), so the honest
reading is that the return-relevant structure is coarse enough to survive a
lossy compression, not that truncation is near-lossless.

Writes the per-width agreement statistics to ``output_file`` (JSON) and the
aligned per-dyad distances to ``output_preservation_file`` (Parquet, columns
``width``, ``truncated``, ``full``); ``render-paper1-diagnostics`` draws the
figures from those saved results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Truncation widths asserted in the manuscript text (Matryoshka ablation).
TRUNCATION_WIDTHS: tuple[int, ...] = (64, 256)


@dataclass(frozen=True, slots=True)
class Paper1DiagnosticsConfig:
    """Typed distance inputs and result outputs for the diagnostic stage."""

    full_distance_artifact_dir: Path
    distance_artifact_64_dir: Path
    distance_artifact_256_dir: Path
    provider_id: str
    representation_id: str
    representation_64_id: str
    representation_256_id: str
    distance_id: str
    output_file: Path
    output_preservation_file: Path


def _load_pair_series(
    artifact_dir: Path,
    *,
    provider_id: str,
    representation_id: str,
    distance_id: str,
) -> pd.DataFrame:
    """Off-diagonal typed distances in a canonical dyad order."""
    frame, summary = read_typed_distance_artifact(
        artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
        },
    )
    item_ids = summary.get("item_ids")
    if not isinstance(item_ids, list):
        message = "typed distance summary has no item roster"
        raise TypeError(message)
    values = frame["value"].to_numpy(dtype=float).reshape(len(item_ids), len(item_ids))
    rows = [
        {"a": str(item_ids[i]), "b": str(item_ids[j]), "distance": float(values[i, j])}
        for i in range(len(item_ids))
        for j in range(i + 1, len(item_ids))
    ]
    return pd.DataFrame(rows).sort_values(["a", "b"]).reset_index(drop=True)


def distance_preservation(
    *,
    full_artifact_dir: Path,
    truncated_artifact_dirs: dict[int, Path],
    provider_id: str,
    full_representation_id: str,
    truncated_representation_ids: dict[int, str],
    distance_id: str,
    widths: tuple[int, ...],
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, float]]]:
    """Truncated-vs-full typed distances and their agreement statistics.

    If truncating to the leading ``k`` coordinates left the pairwise geometry
    intact, all original-space pair rankings would also be intact, so the
    diagnostic measures how closely each truncated matrix tracks the
    full-width one.

    Returns the aligned per-width dyad frames and, per width, the Pearson and
    Spearman correlations against full width.
    """
    full = _load_pair_series(
        full_artifact_dir,
        provider_id=provider_id,
        representation_id=full_representation_id,
        distance_id=distance_id,
    )
    frames: dict[int, pd.DataFrame] = {}
    stats: dict[int, dict[str, float]] = {}
    for k in widths:
        trunc = _load_pair_series(
            truncated_artifact_dirs[k],
            provider_id=provider_id,
            representation_id=truncated_representation_ids[k],
            distance_id=distance_id,
        )
        if not (
            trunc["a"].equals(full["a"]) and trunc["b"].equals(full["b"])
        ):  # pragma: no cover - guards a silent misalignment
            message = (
                f"dyad ordering differs between dim_{k} and dim_full distance tests"
            )
            raise ValueError(message)
        merged = pd.DataFrame(
            {
                "truncated": trunc["distance"].to_numpy(),
                "full": full["distance"].to_numpy(),
            }
        )
        frames[k] = merged
        stats[k] = {
            "pearson": float(np.corrcoef(merged["truncated"], merged["full"])[0, 1]),
            "spearman": float(spearmanr(merged["truncated"], merged["full"]).statistic),
            "n_pairs": len(merged),
        }
        logger.info(
            "distance preservation k=%d: pearson=%.4f spearman=%.4f over %d dyads",
            k,
            stats[k]["pearson"],
            stats[k]["spearman"],
            stats[k]["n_pairs"],
        )
    return frames, stats


def run_paper1_diagnostics(config: Paper1DiagnosticsConfig) -> dict[str, object]:
    """Compute and persist the truncated-vs-full W2 agreement diagnostic."""
    frames, preservation = distance_preservation(
        full_artifact_dir=config.full_distance_artifact_dir,
        truncated_artifact_dirs={
            64: config.distance_artifact_64_dir,
            256: config.distance_artifact_256_dir,
        },
        provider_id=config.provider_id,
        full_representation_id=config.representation_id,
        truncated_representation_ids={
            64: config.representation_64_id,
            256: config.representation_256_id,
        },
        distance_id=config.distance_id,
        widths=TRUNCATION_WIDTHS,
    )
    results: dict[str, object] = {
        "distance_identity": {
            "provider_id": config.provider_id,
            "representation_id": config.representation_id,
            "distance_id": config.distance_id,
        },
        "truncation_widths": list(TRUNCATION_WIDTHS),
        "distance_preservation": {str(k): v for k, v in preservation.items()},
        "note": (
            "distance_preservation compares the truncate-then-renormalize "
            "W2-distance matrix at each width against the full-width "
            "matrix over the same dyads; it measures whether the geometry "
            "survives truncation, which is what the manuscript's robustness "
            "claim is about"
        ),
    }
    pairs = pd.concat(
        [frame.assign(width=width) for width, frame in sorted(frames.items())],
        ignore_index=True,
    )[["width", "truncated", "full"]]
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.output_preservation_file.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(config.output_preservation_file, index=False)
    config.output_file.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results

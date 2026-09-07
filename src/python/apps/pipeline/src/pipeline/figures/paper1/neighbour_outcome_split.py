"""Dedicated renderer and semantic contract for Paper 1's neighbour outcome split.

The stage was renamed from "holdout" in t80 because its evaluation window is a
subset of the in-sample return panel; see
``stages/papers/paper1/neighbour_outcome_split.py``. The rename stops at the
publication boundary on purpose: ``CONTRACT_ID`` moves, but the
``neighbour-holdout-*`` binding keys, ``MANUSCRIPT_ANCHOR``, and the
``neighbour_holdout_paths`` figure stem below do **not**, because each is a
name the manuscript source already spells and renaming them would churn ~40
sites in ``appendix_robustness.tex`` for no change in meaning. What the exhibit
*claims* is corrected here instead, in the title and axis label.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.figures._common import _C, _save, _style, figsize, fit_to_width
from pipeline.io.values import Comment, PublicationValue, format_float

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.io.values import Record

CONTRACT_ID = "paper1-neighbour-outcome-split-v2"
MANUSCRIPT_ANCHOR = "sec:exploratory-geometry-outcomes"
MANUSCRIPT_FRACTION = 0.78
# Sector -> LaTeX macro slug. Deliberately a registry rather than a derived
# lowercasing: each entry mints publication macro names, so a sector entering
# the partition should be a conscious act, not a silent side effect of a roster
# change. `Energy` and `Financial Services` are singletons in the current
# 100-firm universe and so are never *selected* (a singleton has no
# within-sector pair), but they are registered here so a composition shift that
# gives either a second member does not fail at render time.
SECTOR_SLUGS = {
    "Communication Services": "communication-services",
    "Consumer Cyclical": "consumer-cyclical",
    "Consumer Defensive": "consumer-defensive",
    "Energy": "energy",
    "Financial Services": "financial-services",
    "Healthcare": "healthcare",
    "Industrials": "industrials",
    "Technology": "technology",
    "Utilities": "utilities",
}
DECLARE_VALUE_RE = re.compile(r"\\ppDeclareValue\{([^{}]+)\}\{([^{}]*)\}")


@dataclass(frozen=True, slots=True)
class NeighbourOutcomeSplitRenderPaths:
    """Artifacts consumed and produced by the neighbour outcome-split renderer."""

    summary_file: Path
    provenance_file: Path
    selection_file: Path
    pair_paths_file: Path
    numbers_path: Path
    figure_path: Path
    contract_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"{path}: expected a JSON object"
        raise TypeError(message)
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sector_slug(sector: str) -> str:
    try:
        return SECTOR_SLUGS[sector]
    except KeyError as exc:
        message = f"unregistered selected sector {sector!r}"
        raise ValueError(message) from exc


def neighbour_outcome_split_value_records(summary: dict[str, Any]) -> list[Record]:
    """Return the generated numeric bindings registered by the live contract."""
    records: list[Record] = [
        Comment(
            "--- Preregistered original-space neighbour outcome split (t30, t80). "
            "Keys keep their neighbour-holdout- prefix; the evaluation window is "
            "in-sample. ---"
        ),
        PublicationValue(
            "neighbour-holdout-n-firms", int(summary["n_eligible_firms"]), precision=0
        ),
        PublicationValue(
            "neighbour-holdout-n-pairs", int(summary["n_pair_metrics"]), precision=0
        ),
        PublicationValue(
            "neighbour-holdout-n-selected",
            int(summary["n_selected_pairs"]),
            precision=0,
        ),
        PublicationValue(
            "neighbour-holdout-n-dates", int(summary["n_evaluation_dates"]), precision=0
        ),
        PublicationValue(
            "neighbour-holdout-min-prearticles",
            int(summary["minimum_preperiod_embeddings"]),
            precision=0,
        ),
        PublicationValue("neighbour-holdout-training-year-start", 2018, precision=0),
        PublicationValue("neighbour-holdout-training-year-end", 2020, precision=0),
        PublicationValue("neighbour-holdout-pool-year-end", 2022, precision=0),
        PublicationValue("neighbour-holdout-evaluation-year-start", 2021, precision=0),
        PublicationValue("neighbour-holdout-evaluation-year-end", 2022, precision=0),
    ]
    for pair in summary["selected_pairs"]:
        slug = _sector_slug(str(pair["sector"]))
        records.extend(
            [
                PublicationValue(
                    f"neighbour-holdout-{slug}-w2",
                    float(pair["w2_distance"]),
                    precision=3,
                ),
                PublicationValue(
                    f"neighbour-holdout-{slug}-rms-gap",
                    float(pair["rms_cumulative_log_gap"]),
                    precision=3,
                ),
                PublicationValue(
                    f"neighbour-holdout-{slug}-terminal-gap",
                    float(pair["terminal_absolute_log_gap"]),
                    precision=3,
                ),
            ]
        )
        percentile = pair["within_sector_rms_gap_percentile"]
        if percentile is not None:
            records.append(
                PublicationValue(
                    f"neighbour-holdout-{slug}-percentile",
                    float(percentile),
                    precision=1,
                )
            )
    return records


def _displayed_values(summary: dict[str, Any]) -> list[dict[str, Any]]:
    displayed: list[dict[str, Any]] = [
        {
            "binding_key": "neighbour-holdout-n-firms",
            "raw_value": int(summary["n_eligible_firms"]),
            "formatted_value": str(int(summary["n_eligible_firms"])),
        },
        {
            "binding_key": "neighbour-holdout-n-pairs",
            "raw_value": int(summary["n_pair_metrics"]),
            "formatted_value": str(int(summary["n_pair_metrics"])),
        },
        {
            "binding_key": "neighbour-holdout-n-selected",
            "raw_value": int(summary["n_selected_pairs"]),
            "formatted_value": str(int(summary["n_selected_pairs"])),
        },
        {
            "binding_key": "neighbour-holdout-n-dates",
            "raw_value": int(summary["n_evaluation_dates"]),
            "formatted_value": str(int(summary["n_evaluation_dates"])),
        },
        {
            "binding_key": "neighbour-holdout-min-prearticles",
            "raw_value": int(summary["minimum_preperiod_embeddings"]),
            "formatted_value": str(int(summary["minimum_preperiod_embeddings"])),
        },
        {
            "binding_key": "neighbour-holdout-training-year-start",
            "raw_value": 2018,
            "formatted_value": "2018",
        },
        {
            "binding_key": "neighbour-holdout-training-year-end",
            "raw_value": 2020,
            "formatted_value": "2020",
        },
        {
            "binding_key": "neighbour-holdout-pool-year-end",
            "raw_value": 2022,
            "formatted_value": "2022",
        },
        {
            "binding_key": "neighbour-holdout-evaluation-year-start",
            "raw_value": 2021,
            "formatted_value": "2021",
        },
        {
            "binding_key": "neighbour-holdout-evaluation-year-end",
            "raw_value": 2022,
            "formatted_value": "2022",
        },
    ]
    for pair in summary["selected_pairs"]:
        slug = _sector_slug(str(pair["sector"]))
        for suffix, field, precision in (
            ("w2", "w2_distance", 3),
            ("rms-gap", "rms_cumulative_log_gap", 3),
            ("terminal-gap", "terminal_absolute_log_gap", 3),
        ):
            raw_value = float(pair[field])
            displayed.append(
                {
                    "binding_key": f"neighbour-holdout-{slug}-{suffix}",
                    "raw_value": raw_value,
                    "formatted_value": format_float(raw_value, precision),
                }
            )
        percentile = pair["within_sector_rms_gap_percentile"]
        if percentile is not None:
            raw_value = float(percentile)
            displayed.append(
                {
                    "binding_key": f"neighbour-holdout-{slug}-percentile",
                    "raw_value": raw_value,
                    "formatted_value": format_float(raw_value, 1),
                }
            )
    return displayed


def _assert_number_bindings(
    numbers_path: Path, displayed: list[dict[str, Any]]
) -> None:
    bindings = dict(DECLARE_VALUE_RE.findall(numbers_path.read_text(encoding="utf-8")))
    for record in displayed:
        key = str(record["binding_key"])
        formatted = str(record["formatted_value"])
        if bindings.get(key) != formatted:
            actual = bindings.get(key)
            binding_state = f"numbers binding {key!r} is {actual!r}"
            message = f"{binding_state}, expected {formatted!r}"
            raise ValueError(message)


def _render_figure(
    pair_paths: pd.DataFrame, summary: dict[str, Any], path: Path
) -> None:
    _style()
    pairs = summary["selected_pairs"]
    expected_pairs = int(summary["n_selected_pairs"])
    sectors = [str(pair["sector"]) for pair in pairs]
    if not pairs or len(pairs) != expected_pairs or len(sectors) != len(set(sectors)):
        message = (
            "selected-sector figure metadata is inconsistent: "
            f"declared={expected_pairs}, rows={len(pairs)}, sectors={sectors}"
        )
        raise ValueError(message)
    # Two columns, as many rows as the partition needs. The grid was fixed at
    # 3x2 for the retired six-sector partition, so a seventh sector overran it.
    # Height scales with the row count to hold the per-panel aspect ratio.
    columns = 2
    rows = math.ceil(len(pairs) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize(MANUSCRIPT_FRACTION, (7.0 / 7.1) * rows / 3.0),
        sharex=True,
        squeeze=False,
    )
    # An odd sector count leaves a trailing empty cell; blank it rather than
    # ship an axis frame with no series in it.
    for spare in axes.flat[len(pairs) :]:
        spare.set_axis_off()
    for ax, pair in zip(axes.flat, pairs, strict=False):
        sector = str(pair["sector"])
        symbol1 = str(pair["symbol1"])
        symbol2 = str(pair["symbol2"])
        panel = pair_paths[pair_paths["sector"] == sector].sort_values("date")
        if len(panel) != int(summary["n_evaluation_dates"]):
            message = f"{sector}: incomplete selected-pair path"
            raise ValueError(message)
        ax.plot(
            panel["date"],
            panel["cumulative_log_return1"],
            color=_C["black"],
            lw=0.95,
            label=symbol1,
        )
        ax.plot(
            panel["date"],
            panel["cumulative_log_return2"],
            color=_C["gray"],
            lw=0.95,
            ls="--",
            label=symbol2,
        )
        percentile = pair["within_sector_rms_gap_percentile"]
        comparator = (
            "rank n/a (one sector pair)"
            if percentile is None
            else f"RMS percentile={format_float(float(percentile), 1)}"
        )
        # Three short lines at 7pt, not two long ones at 8pt: at the printed panel
        # width (2.95in, down from 3.55in) the old second line ran past the axes
        # into the neighbouring column's tick labels.
        ax.text(
            0.02,
            0.04,
            (
                f"W2={format_float(float(pair['w2_distance']), 3)}; "
                f"RMS gap={format_float(float(pair['rms_cumulative_log_gap']), 3)}\n"
                "terminal gap="
                f"{format_float(float(pair['terminal_absolute_log_gap']), 3)}\n"
                f"{comparator}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0,
            linespacing=1.15,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.2},
        )
        ax.set_title(
            f"{sector}\n{symbol1} (solid), {symbol2} (dashed)",
            fontsize=8.5,
        )
        ax.axhline(0.0, color=_C["gray"], lw=0.45, alpha=0.65)
        ax.grid(visible=True, linestyle=":", linewidth=0.38, alpha=0.4)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=8.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative log return", fontsize=8.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("Evaluation date", fontsize=8.5)
    fig.suptitle(
        "Later-window outcomes for text-nearest within-sector pairs",
        fontsize=10.0,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=1.2, w_pad=1.0)
    fit_to_width(fig, MANUSCRIPT_FRACTION)
    _save(fig, path)


def render_paper1_neighbour_outcome_split(
    paths: NeighbourOutcomeSplitRenderPaths,
) -> dict[str, Any]:
    """Render the sector-panel exhibit and registered semantic contract."""
    summary_file = paths.summary_file
    provenance_file = paths.provenance_file
    selection_file = paths.selection_file
    pair_paths_file = paths.pair_paths_file
    numbers_path = paths.numbers_path
    figure_path = paths.figure_path
    contract_path = paths.contract_path

    summary = _load_json(summary_file)
    provenance = _load_json(provenance_file)
    selection = _load_json(selection_file)
    pair_paths = pd.read_parquet(pair_paths_file)
    _render_figure(pair_paths, summary, figure_path)

    displayed = _displayed_values(summary)
    _assert_number_bindings(numbers_path, displayed)
    selected_pairs = []
    for rank, pair in enumerate(summary["selected_pairs"], start=1):
        selected_pairs.append(
            {
                **pair,
                "pair_id": f"{pair['symbol1']}-{pair['symbol2']}",
                "selection_rank_within_sector": 1,
                "display_order": rank,
            }
        )
    contract: dict[str, Any] = {
        "schema_version": 2,
        "contract_id": CONTRACT_ID,
        "protocol_id": str(summary["protocol_id"]),
        "selection_rule": {
            "geometry": "original full-width embedding space; never MDS",
            "group_rule": (
                "one nearest pair for every sector with at least two eligible firms"
            ),
            "sort": ["w2_distance", "symbol1", "symbol2"],
            "fallback": "none",
            "randomness": "none",
        },
        "training_window": summary["training_window"],
        "evaluation_window": summary["evaluation_window"],
        "model_definition": str(provenance["model_definition"]),
        "normalization_definition": str(provenance["normalization_definition"]),
        "metric_definition": str(provenance["metric_definition"]),
        # Registered in the contract, not only in the stage provenance: a reader
        # checking what this exhibit is entitled to claim must be able to see
        # that the evaluation window is reused in-sample data.
        "evaluation_window_definition": str(provenance["evaluation_window_definition"]),
        "manuscript_anchor": MANUSCRIPT_ANCHOR,
        "eligible_universe": selection["eligible_universe"],
        "exclusions": selection["exclusions"] + selection["singleton_sectors"],
        "tie_break_rule": "ascending (w2_distance, symbol1, symbol2)",
        "comparator_rule": str(provenance["comparator_definition"]),
        "input_array_sha256s": provenance["input_array_sha256s"],
        "selected_pairs": selected_pairs,
        "displayed_values": displayed,
        "output_array_sha256s": provenance["output_array_sha256s"],
    }
    semantic_payload = dict(contract)
    contract["semantic_sha256"] = hashlib.sha256(
        _canonical_json_bytes(semantic_payload)
    ).hexdigest()
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_bytes(_canonical_json_bytes(contract) + b"\n")
    return contract

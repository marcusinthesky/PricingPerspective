"""Generated-regressor, DISCO, sample, and projection-fidelity bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipeline.figures.paper1._numbers.common import _opt_float
from pipeline.figures.paper1.neighbour_outcome_split import (
    neighbour_outcome_split_value_records,
)
from pipeline.io.values import Blank, Comment, PublicationValue

if TYPE_CHECKING:
    from pipeline.io.values import Record


def _append_disco_records(records: list[Record], disco: dict[str, Any] | None) -> None:
    if disco:
        records += [
            Blank(),
            Comment(
                "--- DISCO (energy-distance ANOVA, Rizzo-Szekely 2010; t19) global "
                "sector-separation statistic, in place of the silhouette table; "
                "t13 dyadic confound-aware regression above remains primary ---"
            ),
            Comment("Source: data/papers/paper1/disco/summary.yaml"),
            PublicationValue("disco-r2", float(disco.get("r2_e", 0.0))),
            PublicationValue("disco-f", float(disco.get("f_statistic", 0.0))),
            PublicationValue("disco-p", float(disco.get("disco_perm_p", 0.0))),
            PublicationValue(
                "disco-dispersion-p", float(disco.get("dispersion_perm_p", 0.0))
            ),
        ]


def _append_sample_records(
    records: list[Record], sample_descriptives: dict[str, Any] | None
) -> None:
    if sample_descriptives:
        overall = sample_descriptives.get("overall", {}) or {}
        records += [
            Blank(),
            Comment(
                "--- Sample descriptives, pooled/overall "
                "(data/papers/paper1/sample_descriptives/summary.yaml) ---"
            ),
            PublicationValue(
                "sample-desc-total-articles", int(overall.get("total_articles", 0))
            ),
        ]


def _append_diagnostic_records(
    records: list[Record], diagnostics: dict[str, Any] | None
) -> None:
    if diagnostics:
        preservation = diagnostics.get("distance_preservation", {}) or {}
        if preservation:
            records += [
                Blank(),
                Comment(
                    "--- Truncated-vs-full W2 agreement per "
                    "Matryoshka width (data/papers/paper1/diagnostics/summary.yaml) ---"
                ),
            ]
            for width in sorted(preservation, key=int):
                stats = preservation[width] or {}
                records += [
                    PublicationValue(
                        f"matryoshka-preserve-{width}-spearman",
                        _opt_float(stats.get("spearman")),
                    ),
                ]


def _append_dimensionality_records(
    records: list[Record], dimensionality: dict[str, Any] | None
) -> None:
    if dimensionality:
        records += [
            Blank(),
            Comment(
                "--- Descriptive two-dimensional MDS projection fidelity (t17) ---"
            ),
            PublicationValue("mds-stress-1", float(dimensionality["mds_stress_1"])),
            PublicationValue(
                "mds-shepard-pearson", float(dimensionality["mds_shepard_pearson"])
            ),
            PublicationValue(
                "mds-shepard-spearman", float(dimensionality["mds_shepard_spearman"])
            ),
            PublicationValue(
                "mds-top5-neighbour-recall",
                float(dimensionality["mds_top5_neighbour_recall_mean"]),
            ),
            PublicationValue(
                "mds-exact-nearest-preserved",
                int(dimensionality["mds_exact_nearest_neighbour_preserved"]),
            ),
        ]


def _append_neighbour_outcome_split_records(
    records: list[Record], neighbour_outcome_split: dict[str, Any] | None
) -> None:
    if neighbour_outcome_split:
        records += [
            Blank(),
            *neighbour_outcome_split_value_records(neighbour_outcome_split),
        ]

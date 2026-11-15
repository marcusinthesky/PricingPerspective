"""Paper 1 generated numbers: the numbers.tex record set.

This module owns the loader and the record ORDER. Each experiment's records
are built by that experiment's exhibit module (``dyadic_confound``, ``w2_permanova``,
``sample_descriptives``, ``diagnostics``, ``neighbour_outcome_split``,
``dyadic_oos``, ``text_controls``); the configuration, corpus, return-panel,
and projection records are built here. :func:`write_paper1_numbers` states
the emission sequence -- reordering its calls reorders the generated LaTeX file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import pandas as pd
import pyarrow.parquet as pq

from pipeline.figures._common import _load_yaml
from pipeline.figures.paper1._format import opt_float
from pipeline.figures.paper1.diagnostics import append_diagnostics_records
from pipeline.figures.paper1.dyadic_confound import append_dyadic_confound_records
from pipeline.figures.paper1.dyadic_oos import append_dyadic_oos_records
from pipeline.figures.paper1.neighbour_outcome_split import (
    neighbour_outcome_split_value_records,
)
from pipeline.figures.paper1.sample_descriptives import (
    append_sample_descriptives_records,
)
from pipeline.figures.paper1.text_controls import append_text_controls_records
from pipeline.figures.paper1.w2_permanova import append_w2_permanova_records
from pipeline.io.values import Blank, Comment, PublicationValue, write_generated_numbers
from pipeline.io.w2_permanova_artifacts import read_w2_permanova_results

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.io.values import Record

logger = logging.getLogger(__name__)
_MIN_IDENTIFYING_INDUSTRY_SIZE = 4


@dataclass(frozen=True, slots=True)
class Paper1NumberPaths:
    """Artifact paths consumed by the Paper 1 number-binding leaf.

    Every results file is optional at read time: a missing file contributes no
    records, so a partial graph still renders the bindings it can.
    """

    numbers_path: Path
    params_path: Path
    corpus_summary: Path
    dyadic_confound_results: Path
    w2_permanova_results: Path
    diagnostics_results: Path
    sample_descriptives_results: Path
    pcoa_summary: Path
    neighbour_outcome_split_results: Path
    covariance_matrix: Path
    oos_covariance_matrix: Path | None
    oos_dyadic_results: Path | None
    universe_csv: Path | None = None
    source_articles_dir: Path | None = None
    text_controls_results: Path | None = None


@dataclass(frozen=True, slots=True)
class Paper1NumberInputs:
    """Loaded artifacts consumed by the Paper 1 number writer."""

    provenance_counts: dict[str, int] | None = None
    model: str = "qwen3-embedding-8b"
    params: dict[str, Any] | None = None
    corpus_summary: dict[str, Any] | None = None
    dyadic_confound: dict[str, Any] | None = None
    w2_permanova: dict[str, Any] | None = None
    sample_descriptives: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    pcoa: dict[str, Any] | None = None
    neighbour_outcome_split: dict[str, Any] | None = None
    return_panel: dict[str, Any] | None = None
    oos_return_panel: dict[str, Any] | None = None
    oos_dyadic: dict[str, Any] | None = None
    text_controls: dict[str, Any] | None = None


def write_paper1_numbers(
    path: Path,
    inputs: Paper1NumberInputs,
) -> None:
    """Emit Paper 1 W2, projection, sector-test, and sample LaTeX bindings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _initial_paper1_records(inputs.return_panel, inputs.oos_return_panel)
    _append_configuration_records(records, inputs.params, inputs.model)
    _append_corpus_records(records, inputs.corpus_summary, inputs.params)
    append_dyadic_confound_records(records, inputs.dyadic_confound)
    append_w2_permanova_records(records, inputs.w2_permanova)
    append_sample_descriptives_records(records, inputs.sample_descriptives)
    append_diagnostics_records(records, inputs.diagnostics)
    _append_pcoa_records(records, inputs.pcoa)
    _append_neighbour_outcome_split_records(records, inputs.neighbour_outcome_split)
    append_dyadic_oos_records(records, inputs.oos_dyadic)
    append_text_controls_records(records, inputs.text_controls)
    if inputs.provenance_counts:
        records.extend(
            PublicationValue(key, value)
            for key, value in inputs.provenance_counts.items()
        )
    write_generated_numbers(path, records)


def _read_return_panel_metadata(covariance_matrix: Path | None) -> dict[str, Any]:
    """Read the governed complete-case return-panel metadata off a covariance artifact.

    Shared by the in-sample and out-of-sample (t16) covariance parquets so panel
    composition reaches the manuscript from artifact metadata rather than prose.
    """
    if covariance_matrix is None or not covariance_matrix.exists():
        return {}
    raw_metadata = pq.read_schema(covariance_matrix).metadata or {}
    metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in raw_metadata.items()
    }
    required = {
        "return_panel_rows",
        "return_panel_columns",
        "return_panel_start_date",
        "return_panel_end_date",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        message = (
            "covariance artifact lacks governed return-panel metadata: "
            + ", ".join(missing)
        )
        raise ValueError(message)
    return {key: metadata[key] for key in required}


def provenance_counts(
    universe_csv: Path,
    source_articles_dir: Path,
    symbols: list[str],
    dyadic: dict[str, Any],
) -> dict[str, int]:
    """Count source articles and industries with residual pair variation."""
    rows = dyadic["classification_robustness"]["rows"]
    within = next(row for row in rows if row["key"] == "within-industry")
    roster = pd.read_csv(universe_csv).set_index("Symbol", verify_integrity=True)
    industries = roster.loc[within["bootstrap_nodes"], "Industry"]
    if industries.isna().any():
        message = "Within-industry firms must have industry labels"
        raise ValueError(message)
    sizes = industries.value_counts()
    identifying = sizes[sizes >= _MIN_IDENTIFYING_INDUSTRY_SIZE]
    return {
        "source-pool-total-articles": sum(
            pq.read_metadata(source_articles_dir / f"{symbol}.parquet").num_rows
            for symbol in symbols
        ),
        "classification-identifying-industries": len(identifying),
        "classification-identifying-firms": int(identifying.sum()),
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    """Load a JSON results object, or ``{}`` when the file is absent."""
    if path is None or not path.exists():
        return {}
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def render_paper1_numbers(paths: Paper1NumberPaths) -> None:
    """Write Paper 1's number bindings from saved results only."""
    params = _load_yaml(paths.params_path) if paths.params_path.exists() else {}
    corpus_sum = (
        _load_yaml(paths.corpus_summary) if paths.corpus_summary.exists() else {}
    )
    dyadic_confound = _load_json(paths.dyadic_confound_results)
    write_paper1_numbers(
        paths.numbers_path,
        Paper1NumberInputs(
            provenance_counts=(
                provenance_counts(
                    paths.universe_csv,
                    paths.source_articles_dir,
                    params["symbols"],
                    dyadic_confound,
                )
                if paths.universe_csv is not None
                and paths.source_articles_dir is not None
                else None
            ),
            params=params,
            corpus_summary=corpus_sum,
            dyadic_confound=dyadic_confound,
            w2_permanova=(
                read_w2_permanova_results(paths.w2_permanova_results)
                if paths.w2_permanova_results.exists()
                else {}
            ),
            sample_descriptives=_load_json(paths.sample_descriptives_results),
            diagnostics=_load_json(paths.diagnostics_results),
            pcoa=_load_json(paths.pcoa_summary),
            neighbour_outcome_split=_load_json(paths.neighbour_outcome_split_results),
            return_panel=_read_return_panel_metadata(paths.covariance_matrix),
            oos_return_panel=_read_return_panel_metadata(paths.oos_covariance_matrix),
            oos_dyadic=_load_json(paths.oos_dyadic_results),
            text_controls=_load_json(paths.text_controls_results),
        ),
    )


def _initial_paper1_records(
    return_panel: dict[str, Any] | None,
    oos_return_panel: dict[str, Any] | None,
) -> list[Record]:
    records: list[Record] = [
        Comment(
            "AUTO-GENERATED by pipeline render-paper1-numbers — do not edit by hand."
        ),
    ]
    if return_panel:
        records += [
            Blank(),
            Comment(
                "--- Exact complete-case return panel "
                "(data/shared/covariance_matrix.parquet metadata) ---"
            ),
            PublicationValue(
                "return-panel-n-observations",
                int(return_panel["return_panel_rows"]),
            ),
            PublicationValue(
                "return-panel-n-tickers", int(return_panel["return_panel_columns"])
            ),
            PublicationValue(
                "return-panel-date-start", str(return_panel["return_panel_start_date"])
            ),
            PublicationValue(
                "return-panel-date-end", str(return_panel["return_panel_end_date"])
            ),
        ]
    if oos_return_panel:
        records += [
            Blank(),
            Comment(
                "--- Timing-separated complete-case return panel "
                "(data/shared/oos/covariance_matrix.parquet metadata) ---"
            ),
            PublicationValue(
                "oos-return-panel-n-observations",
                int(oos_return_panel["return_panel_rows"]),
            ),
            PublicationValue(
                "oos-return-panel-n-tickers",
                int(oos_return_panel["return_panel_columns"]),
            ),
            PublicationValue(
                "oos-return-panel-date-start",
                str(oos_return_panel["return_panel_start_date"]),
            ),
            PublicationValue(
                "oos-return-panel-date-end",
                str(oos_return_panel["return_panel_end_date"]),
            ),
        ]
    return records


def _append_configuration_records(
    records: list[Record],
    params: dict[str, Any] | None,
    model: str,
) -> None:
    if params:
        symbols = params.get("symbols", []) or []
        matryoshka_range = params.get("matryoshka_range", []) or []
        embed_cfg = params.get("embedding", {}).get("models", {}).get(model, {}) or {}
        start_year = str(params.get("start_date", ""))[:4]
        end_year = str(params.get("end_date", ""))[:4]
        records += [
            Blank(),
            Comment("--- Pipeline configuration (params.yaml) ---"),
            PublicationValue("n-samples-cap", int(params.get("n_samples", 0))),
            PublicationValue(
                "annual-raw-article-floor",
                int(params.get("annual_raw_article_floor", 0)),
            ),
            PublicationValue("n-firms-news", len(symbols)),
            PublicationValue(
                "matryoshka-dim-low",
                int(matryoshka_range[0])
                if len(matryoshka_range) > 0 and matryoshka_range[0] is not None
                else 0,
            ),
            PublicationValue(
                "matryoshka-dim-mid",
                int(matryoshka_range[1])
                if len(matryoshka_range) > 1 and matryoshka_range[1] is not None
                else 0,
            ),
            PublicationValue("embed-dim", int(embed_cfg.get("dims", 0))),
            PublicationValue("sample-year-start", int(start_year)),
            PublicationValue("sample-year-end", int(end_year)),
        ]
        if "paper1_text_controls" in params:
            records.append(
                PublicationValue(
                    "text-controls-max-features",
                    int(params["paper1_text_controls"]["max_features"]),
                )
            )


def _append_corpus_records(
    records: list[Record],
    corpus_summary: dict[str, Any] | None,
    params: dict[str, Any] | None,
) -> None:
    if corpus_summary:
        rows_in = int(corpus_summary.get("rows_in", 0))
        rows_out = int(corpus_summary.get("rows_out", 0))
        duplicates_dropped = int(corpus_summary.get("duplicates_dropped", 0))
        date_min = str(corpus_summary.get("date_min", ""))[:4]
        date_max = str(corpus_summary.get("date_max", ""))[:4]
        hist = corpus_summary.get("per_year_histogram", {}) or {}
        start_y = int(str(params.get("start_date", ""))[:4]) if params else 2018
        end_y = int(str(params.get("end_date", ""))[:4]) if params else 2022
        window_total = sum(
            int(v) for k, v in hist.items() if start_y <= int(k) <= end_y
        )
        records += [
            Blank(),
            Comment("--- News corpus summary (data/shared/corpus/summary.yaml) ---"),
            PublicationValue("corpus-raw-records", rows_in),
            PublicationValue("corpus-articles-total", rows_out),
            PublicationValue("corpus-duplicates-dropped", duplicates_dropped),
            PublicationValue("corpus-year-min", int(date_min)),
            PublicationValue("corpus-year-max", int(date_max)),
            PublicationValue("corpus-articles-window", window_total),
        ]


def _append_pcoa_records(records: list[Record], pcoa: dict[str, Any] | None) -> None:
    if pcoa:
        canonical = [
            cell
            for cell in pcoa["cells"]
            if cell["cell_id"] == pcoa["canonical_cell_id"]
        ]
        if len(canonical) != 1:
            message = "PCoA canonical cell must identify exactly one saved result"
            raise ValueError(message)
        result = canonical[0]
        records += [
            Blank(),
            Comment("--- Descriptive two-axis PCoA: fidelity and inertia shares ---"),
            PublicationValue("pcoa-stress-1", opt_float(result["stress_1"])),
            PublicationValue(
                "pcoa-shepard-pearson", opt_float(result["shepard_pearson"])
            ),
            PublicationValue(
                "pcoa-shepard-spearman", opt_float(result["shepard_spearman"])
            ),
            PublicationValue(
                "pcoa-top5-neighbour-recall",
                opt_float(result["top5_neighbour_recall_mean"]),
            ),
            PublicationValue(
                "pcoa-exact-nearest-preserved",
                int(result["exact_nearest_neighbour_preserved"]),
            ),
            PublicationValue("pcoa-two-axis-share", float(result["two_axis_share"])),
            PublicationValue(
                "pcoa-negative-inertia-share",
                float(result["negative_inertia_fraction"]),
            ),
        ]
        metric_mds = pcoa["metric_mds"]
        records += [
            Blank(),
            Comment(
                "--- Descriptive two-axis metric MDS: original-distance fidelity ---"
            ),
            *(
                PublicationValue("metric-mds-" + name, opt_float(metric_mds[key]))
                for name, key in (
                    ("stress-1", "stress_1"),
                    ("shepard-pearson", "shepard_pearson"),
                    ("shepard-spearman", "shepard_spearman"),
                    ("top5-neighbour-recall", "top5_neighbour_recall_mean"),
                )
            ),
            PublicationValue(
                "metric-mds-exact-nearest-preserved",
                int(metric_mds["exact_nearest_neighbour_preserved"]),
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

"""Paper 1 generated numbers: the numbers.tex record set.

This module owns the role's public contracts, the artifact loader, and the
record ORDER; the record bodies live in the private
:mod:`pipeline.figures.paper1._numbers` package. ``_write_paper1_numbers`` is
the single place the emission sequence is stated -- reordering its calls
reorders the generated LaTeX file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import pandas as pd
import pyarrow.parquet as pq

from pipeline.figures._common import _load_yaml
from pipeline.figures.paper1._numbers.diagnostics import (
    _append_diagnostic_records,
    _append_dimensionality_records,
    _append_disco_records,
    _append_neighbour_outcome_split_records,
    _append_sample_records,
)
from pipeline.figures.paper1._numbers.dyadic import _append_dyadic_records
from pipeline.figures.paper1._numbers.headline import (
    _append_configuration_records,
    _append_corpus_records,
    _initial_paper1_records,
)
from pipeline.figures.paper1._numbers.oos import (
    _append_oos_dyadic_records,
)
from pipeline.io.values import write_generated_numbers

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Paper1NumberPaths:
    """Artifact paths consumed by the Paper 1 number-binding leaf."""

    numbers_path: Path
    params_path: Path
    corpus_summary: Path
    dyadic_confound_summary: Path
    disco_summary: Path
    diagnostics_summary: Path
    sample_descriptives_summary: Path
    dimensionality_csv: Path
    neighbour_outcome_split_summary: Path
    covariance_matrix: Path
    oos_covariance_matrix: Path | None
    oos_dyadic_summary: Path | None


@dataclass(frozen=True, slots=True)
class Paper1NumberInputs:
    """Loaded artifacts consumed by the Paper 1 number writer."""

    model: str = "qwen3-embedding-8b"
    params: dict[str, Any] | None = None
    corpus_summary: dict[str, Any] | None = None
    dyadic_confound: dict[str, Any] | None = None
    disco: dict[str, Any] | None = None
    sample_descriptives: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    dimensionality: dict[str, Any] | None = None
    neighbour_outcome_split: dict[str, Any] | None = None
    return_panel: dict[str, Any] | None = None
    oos_return_panel: dict[str, Any] | None = None
    oos_dyadic: dict[str, Any] | None = None


def _write_paper1_numbers(
    path: Path,
    inputs: Paper1NumberInputs,
) -> None:
    """Emit Paper 1 W2, MDS, DISCO, and sample LaTeX bindings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _initial_paper1_records(inputs.return_panel, inputs.oos_return_panel)
    _append_configuration_records(records, inputs.params, inputs.model)
    _append_corpus_records(records, inputs.corpus_summary, inputs.params)
    _append_dyadic_records(records, inputs.dyadic_confound)
    _append_disco_records(records, inputs.disco)
    _append_sample_records(records, inputs.sample_descriptives)
    _append_diagnostic_records(records, inputs.diagnostics)
    _append_dimensionality_records(records, inputs.dimensionality)
    _append_neighbour_outcome_split_records(records, inputs.neighbour_outcome_split)
    _append_oos_dyadic_records(records, inputs.oos_dyadic)
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


def render_paper1_numbers(paths: Paper1NumberPaths) -> None:
    """Write Paper 1's number bindings from summary data only.

    Split from ``render_paper1_figures`` (t15 R3) so a numbers-only edit or an
    unrelated summary refresh does not invalidate the PDF figure render.
    """
    params = _load_yaml(paths.params_path) if paths.params_path.exists() else {}
    corpus_sum = (
        _load_yaml(paths.corpus_summary) if paths.corpus_summary.exists() else {}
    )
    dyadic_confound = (
        _load_yaml(paths.dyadic_confound_summary)
        if paths.dyadic_confound_summary.exists()
        else {}
    )
    disco = _load_yaml(paths.disco_summary) if paths.disco_summary.exists() else {}
    diagnostics = (
        _load_yaml(paths.diagnostics_summary)
        if paths.diagnostics_summary.exists()
        else {}
    )
    sample_descriptives = (
        _load_yaml(paths.sample_descriptives_summary)
        if paths.sample_descriptives_summary.exists()
        else {}
    )
    dimensionality = (
        pd.read_csv(paths.dimensionality_csv).iloc[0].to_dict()
        if paths.dimensionality_csv.exists()
        else {}
    )
    neighbour_outcome_split = (
        json.loads(paths.neighbour_outcome_split_summary.read_text(encoding="utf-8"))
        if paths.neighbour_outcome_split_summary.exists()
        else {}
    )
    return_panel = _read_return_panel_metadata(paths.covariance_matrix)
    oos_return_panel = _read_return_panel_metadata(paths.oos_covariance_matrix)
    oos_dyadic = (
        _load_yaml(paths.oos_dyadic_summary)
        if paths.oos_dyadic_summary is not None and paths.oos_dyadic_summary.exists()
        else {}
    )
    _write_paper1_numbers(
        paths.numbers_path,
        Paper1NumberInputs(
            params=params,
            corpus_summary=corpus_sum,
            dyadic_confound=dyadic_confound,
            disco=disco,
            sample_descriptives=sample_descriptives,
            diagnostics=diagnostics,
            dimensionality=dimensionality,
            neighbour_outcome_split=neighbour_outcome_split,
            return_panel=return_panel,
            oos_return_panel=oos_return_panel,
            oos_dyadic=oos_dyadic,
        ),
    )

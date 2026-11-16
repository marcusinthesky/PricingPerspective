"""Shared generated table renderers used by multiple manuscripts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from pipeline.figures._latex import escape_latex
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_METADATA_COLUMNS = ("Symbol", "Name", "Sector")
_CAPTION = (
    r"Canonical empirical roster: ticker symbol, firm name, and sector label "
    r"from Nasdaq summary metadata for the firms in the canonical priced "
    r"$W_2$ geometry, sorted by sector then symbol."
)
_SECTOR_LIST_CAPTION = (
    r"Canonical empirical roster: ticker symbols of the firms in the canonical "
    r"priced $W_2$ geometry, grouped by the sector label in Nasdaq summary "
    r"metadata. Firm names accompany the symbols in the replication archive's "
    r"universe metadata."
)

RosterLayout = Literal["longtable", "sector-list"]


def _distance_item_ids(
    distance_artifact_dir: Path,
    returns_dir: Path,
    expected_identity: dict[str, object],
) -> list[str]:
    _frame, summary = read_typed_distance_artifact(
        distance_artifact_dir,
        expected_identity=expected_identity,
    )
    raw_item_ids = summary.get("item_ids")
    if (
        not isinstance(raw_item_ids, list)
        or not raw_item_ids
        or not all(isinstance(item_id, str) and item_id for item_id in raw_item_ids)
    ):
        message = "typed distance artifact has malformed item identifiers"
        raise ValueError(message)
    item_ids = cast("list[str]", raw_item_ids)
    if len(set(item_ids)) != len(item_ids):
        message = "typed distance artifact contains duplicate item identifiers"
        raise ValueError(message)

    priced = [
        item_id
        for item_id in item_ids
        if (returns_dir / f"{item_id}.parquet").is_file()
    ]
    excluded = sorted(set(item_ids) - set(priced))
    if excluded:
        logger.info("excluding unpriced typed-distance items %s", excluded)
    if not priced:
        message = "typed distance artifact has no priced item identifiers"
        raise ValueError(message)
    return priced


def _metadata_rows(
    universe_csv: Path, item_ids: Sequence[str]
) -> list[tuple[str, str, str]]:
    frame = pd.read_csv(universe_csv, dtype=str, keep_default_na=False)
    missing_columns = [
        column for column in _REQUIRED_METADATA_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "universe metadata is missing required columns: "
            + ", ".join(missing_columns)
        )
    metadata = frame.loc[:, list(_REQUIRED_METADATA_COLUMNS)].copy()
    if metadata["Symbol"].duplicated().any():
        message = "universe metadata contains duplicate symbols"
        raise ValueError(message)
    if (metadata == "").any().any():
        message = "universe metadata contains blank symbol, name, or sector"
        raise ValueError(message)

    selected = metadata[metadata["Symbol"].isin(item_ids)].copy()
    if len(selected) != len(item_ids):
        missing = sorted(set(item_ids) - set(selected["Symbol"]))
        raise ValueError(
            "universe metadata has no row for typed-distance items: "
            + ", ".join(missing)
        )
    selected = selected.sort_values(["Sector", "Symbol"], kind="stable")
    return [
        (str(row.Symbol), str(row.Name), str(row.Sector))
        for row in selected.itertuples(index=False)
    ]


def _render_longtable(
    rows: Sequence[tuple[str, str, str]] | Sequence[str],
    *,
    columns: str = "lll",
    header: str = r"Symbol & Name & Sector \\",
    caption: str = _CAPTION,
    label: str = "tab:sample-universe",
    font_size: str | None = None,
    tabcolsep: str | None = None,
) -> str:
    lines = []
    if font_size is not None or tabcolsep is not None:
        lines.append(r"\begingroup")
        if font_size is not None:
            lines.append(rf"\{font_size}")
        if tabcolsep is not None:
            lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines.extend(
        [
            rf"\begin{{longtable}}{{{columns}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}} \\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endfirsthead",
            (
                rf"\multicolumn{{{header.count('&') + 1}}}{{c}}"
                r"{\tablename\ \thetable\ (continued)} \\"
            ),
            r"\toprule",
            header,
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    rendered_rows = [
        row
        if isinstance(row, str)
        else " & ".join(escape_latex(cell) for cell in row) + r" \\"
        for row in rows
    ]
    lines.append("\n".join(rendered_rows))
    lines.append(r"\end{longtable}")
    if font_size is not None or tabcolsep is not None:
        lines.append(r"\endgroup")
    return "\n".join(lines)


def _render_sector_list(rows: Sequence[tuple[str, str, str]]) -> str:
    """Render one row per sector listing its ticker symbols."""
    by_sector: dict[str, list[str]] = {}
    for symbol, _name, sector in rows:
        by_sector.setdefault(sector, []).append(symbol)
    rendered_rows = [
        f"{escape_latex(sector)} & {len(symbols)} & "
        + ", ".join(escape_latex(symbol) for symbol in symbols)
        + r" \\"
        for sector, symbols in by_sector.items()
    ]
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabularx}{\linewidth}{@{}lr>{\raggedright\arraybackslash}X@{}}",
            r"\toprule",
            r"Sector & Firms & Ticker symbols \\",
            r"\midrule",
            *rendered_rows,
            r"\bottomrule",
            r"\end{tabularx}",
            rf"\caption{{{_SECTOR_LIST_CAPTION}}}",
            r"\label{tab:sample-universe}",
            r"\end{table}",
        ]
    )


def render_sample_universe_table(
    *,
    universe_csv: Path,
    distance_artifact_dir: Path,
    returns_dir: Path,
    output_file: Path,
    provider_id: str,
    representation_id: str,
    distance_id: str = "wasserstein_w2",
    layout: RosterLayout = "longtable",
) -> str:
    """Render the canonical priced W2 roster as an appendix table.

    ``longtable`` lists symbol, name, and sector per firm; ``sector-list`` is a
    compact one-row-per-sector list of ticker symbols.
    """
    item_ids = _distance_item_ids(
        distance_artifact_dir,
        returns_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
        },
    )
    rows = _metadata_rows(universe_csv, item_ids)
    fragment = (
        "% AUTO-GENERATED by pipeline render-sample-universe-table "
        "-- do not edit by hand.\n\n"
        + (
            _render_sector_list(rows)
            if layout == "sector-list"
            else _render_longtable(rows)
        )
        + "\n"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(fragment, encoding="utf-8")
    return fragment


__all__ = ["RosterLayout", "render_sample_universe_table"]

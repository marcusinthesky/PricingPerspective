"""Paper 1 publication-value contracts for the Nasdaq article corpus."""

from typing import TYPE_CHECKING

import pandas as pd
from pipeline.figures.paper1.numbers import (
    Paper1NumberInputs,
    provenance_counts,
    write_paper1_numbers,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_corpus_flow_counts_are_emitted_as_publication_values(
    tmp_path: "Path",
) -> None:
    """Expose raw, deduplicated, and sample-window corpus sizes to LaTeX."""
    numbers = tmp_path / "numbers.tex"
    write_paper1_numbers(
        numbers,
        Paper1NumberInputs(
            params={"start_date": "2018-01-01", "end_date": "2022-12-31"},
            corpus_summary={
                "rows_in": 397_041,
                "rows_out": 157_791,
                "duplicates_dropped": 239_250,
                "date_min": "2009-10-14",
                "date_max": "2023-01-02",
                "per_year_histogram": {
                    2017: 21_357,
                    2018: 21_005,
                    2019: 15_885,
                    2020: 18_600,
                    2021: 19_015,
                    2022: 22_987,
                    2023: 16,
                },
            },
        ),
    )

    emitted = numbers.read_text(encoding="utf-8")
    assert r"\ppDeclareValue{corpus-raw-records}{397041}" in emitted
    assert r"\ppDeclareValue{corpus-articles-total}{157791}" in emitted
    assert r"\ppDeclareValue{corpus-duplicates-dropped}{239250}" in emitted
    assert r"\ppDeclareValue{corpus-articles-window}{97492}" in emitted


def test_provenance_counts_respect_roster_and_identifying_group_size(
    tmp_path: "Path",
) -> None:
    """Exclude unselected files and groups exhausted by symmetric firm effects."""
    symbols: list[str] = list("ABCDEFGHIJ")
    roster = pd.DataFrame(
        {
            "Symbol": symbols,
            "Industry": ["pair"] * 2 + ["triple"] * 3 + ["large"] * 5,
        }
    )
    roster_path = tmp_path / "universe.csv"
    roster.to_csv(roster_path, index=False)
    for symbol in [*symbols, "OUTSIDE"]:
        pd.DataFrame({"article": [1, 2, 3]}).to_parquet(tmp_path / f"{symbol}.parquet")
    summary = {
        "classification_robustness": {
            "rows": [
                {
                    "key": "within-industry",
                    "bootstrap_nodes": symbols[:-1],
                }
            ]
        }
    }
    assert provenance_counts(roster_path, tmp_path, symbols, summary) == {
        "source-pool-total-articles": 30,
        "classification-identifying-industries": 1,
        "classification-identifying-firms": 4,
    }

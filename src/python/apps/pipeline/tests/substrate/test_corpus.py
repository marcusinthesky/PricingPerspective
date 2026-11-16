"""Independent SQL witnesses for the published numeric co-mention event adapter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
import pytest
from pipeline.stages.papers.paper5.co_mentions import compute_co_mentions

if TYPE_CHECKING:
    from pathlib import Path


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["before", "first", "repeat", "next", "other", "after"],
            "created_date": pd.to_datetime(
                [
                    "2017-12-31",
                    "2018-01-01",
                    "2018-07-01",
                    "2019-12-31",
                    "2019-06-01",
                    "2020-01-01",
                ]
            ),
            "publication_year": [2017, 2018, 2018, 2019, 2019, 2020],
            "firm_i": ["A"] * 6,
            "firm_j": ["B", "B", "B", "B", "C", "C"],
        }
    )


def test_numeric_events_preserve_windowed_sql_counts_and_persistence(
    tmp_path: Path,
) -> None:
    """Inclusive dates, repeated events and multi-year persistence match SQL."""
    events = _events()
    events_file = tmp_path / "events.parquet"
    universe_file = tmp_path / "universe.csv"
    events.to_parquet(events_file, index=False)
    pd.DataFrame({"Symbol": ["A", "B", "C"]}).to_csv(universe_file, index=False)
    output = tmp_path / "output"
    compute_co_mentions(
        events_file, universe_file, output, "2018-01-01", "2019-12-31", 2
    )

    with duckdb.connect() as connection:
        connection.register("input_events", events)
        connection.execute(
            "CREATE TEMP TABLE selected AS SELECT * FROM input_events "
            "WHERE created_date BETWEEN DATE '2018-01-01' AND DATE '2019-12-31'"
        )
        connection.execute(
            "CREATE TEMP TABLE pair_year AS SELECT firm_i, firm_j, publication_year, "
            "COUNT(*)::BIGINT AS co_mention_count FROM selected "
            "GROUP BY firm_i, firm_j, publication_year"
        )
        expected_pairs = connection.execute(
            "SELECT * FROM pair_year ORDER BY firm_i, firm_j, publication_year"
        ).df()
        expected_adjacency = connection.execute(
            "SELECT firm_i, firm_j, SUM(co_mention_count)::BIGINT AS weight, "
            "COUNT(*)::INTEGER AS n_years, "
            "MIN(publication_year)::INTEGER AS first_year, "
            "MAX(publication_year)::INTEGER AS last_year FROM pair_year "
            "GROUP BY firm_i, firm_j HAVING COUNT(*) >= 2 ORDER BY firm_i, firm_j"
        ).df()
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "pair_year.parquet"), expected_pairs, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "adjacency.parquet").reset_index(drop=True),
        expected_adjacency,
        check_dtype=False,
    )
    assert json.loads((output / "summary.json").read_text()) == {
        "artifact_type": "news_co_mentions",
        "start_date": "2018-01-01",
        "end_date": "2019-12-31",
        "min_persistence_years": 2,
        "n_nodes": 3,
        "exact_two_articles": 4,
        "observed_pairs": 2,
        "pair_mentions": 4,
        "persistent_pairs": 1,
    }


def test_numeric_events_reject_inconsistent_publication_year(tmp_path: Path) -> None:
    """Date-based windows and year-based persistence cannot use different years."""
    events = _events()
    events.loc[1, "publication_year"] = 2019
    events_file = tmp_path / "events.parquet"
    universe_file = tmp_path / "universe.csv"
    events.to_parquet(events_file, index=False)
    pd.DataFrame({"Symbol": ["A", "B", "C"]}).to_csv(universe_file, index=False)
    with pytest.raises(ValueError, match="year"):
        compute_co_mentions(
            events_file,
            universe_file,
            tmp_path / "output",
            "2018-01-01",
            "2019-12-31",
            2,
        )

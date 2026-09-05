"""Corpus normalization and subsampling.

Provides:
- ``normalize_corpus`` — deduplicate, date-parse, and filter the raw NASDAQ corpus
  to the 100-ticker universe; output articles.parquet + summary.yaml.
- ``subsample`` — filter and subsample the normalized corpus for a single ticker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

logger = logging.getLogger(__name__)


class _MissingNormalizationSummaryError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("DuckDB returned no row for the corpus normalization summary")


class _MissingSampleCountError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("DuckDB returned no row for the corpus sample count")


@dataclass(frozen=True)
class SubsampleOptions:
    """Filters and deterministic sample size for one ticker corpus."""

    start_date: str
    end_date: str
    min_body_chars: int
    n_samples: int


def load_symbols(params_path: Path) -> list[str]:
    """Load the 100-ticker universe from params.yaml."""
    with Path(params_path).open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    symbols = params["symbols"]
    logger.info("Loaded %d symbols from params.yaml", len(symbols))
    return [s.upper() for s in symbols]


def normalize_corpus(
    input_path: Path,
    output_parquet: Path,
    output_summary: Path,
    params_path: Path,
) -> None:
    """Normalize raw NASDAQ corpus: dedupe, parse dates, filter to universe.

    Reads the 1.8 GB ``nasdaq.jsonlines`` file via DuckDB streaming (never loads
    into pandas), deduplicates by ``id`` (keeping first occurrence), parses
    ``created_date`` from ``ago`` with fallback to ``timestamp_published``,
    computes ``url_hash = md5(url)``, filters to the 100-ticker universe, and
    writes ``data/shared/corpus/articles.parquet`` and the governed summary
    YAML artifact.
    """
    symbols = load_symbols(params_path)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s via DuckDB streaming …", input_path)

    with duckdb.connect() as con:
        rows_in_result = con.execute(
            "SELECT COUNT(*) FROM read_json(?, format='newline_delimited')",
            [str(input_path)],
        ).fetchone()
        rows_in = rows_in_result[0] if rows_in_result else 0
        logger.info("rows_in (raw): %s", f"{rows_in:,}")

        con.execute(
            """
        CREATE OR REPLACE TEMP TABLE normalized AS
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY id
                    ORDER BY date_accessed DESC NULLS LAST, symbol, url, title, body,
                             publisher, author, primarysymbol, primarytopic
                ) AS rn
            FROM read_json(?, format='newline_delimited')
        ),
        deduped AS (
            SELECT * FROM ranked WHERE rn = 1
        ),
        date_parsed AS (
            SELECT
                *,
                TRY_STRPTIME(ago, '%b %d, %Y') AS date_from_ago,
                TRY_STRPTIME(
                    REGEXP_REPLACE(timestamp_published, ' —.*$', ''),
                    '%B %d, %Y'
                ) AS date_from_ts
            FROM deduped
        ),
        universe AS (
            SELECT
                CAST(id AS VARCHAR)                              AS id,
                UPPER(symbol)                                    AS symbol,
                UPPER(primarysymbol)                             AS primarysymbol,
                title,
                body,
                LENGTH(body)                                     AS body_len,
                publisher,
                author,
                COALESCE(date_from_ago, date_from_ts)::DATE      AS created_date,
                url,
                md5(url)                                         AS url_hash,
                primarytopic,
                related_symbols,
                date_accessed,
                (date_from_ago IS NULL AND date_from_ts IS NULL) AS _missing_date
            FROM date_parsed
            WHERE UPPER(symbol) IN (SELECT UNNEST(?))
        )
        SELECT * FROM universe
        """,
            [str(input_path), symbols],
        )

        logger.info("Writing deduplicated, filtered parquet …")
        con.table("normalized").write_parquet(str(output_parquet))

        logger.info("Computing summary statistics …")

        stats = con.execute("""
            SELECT
                COUNT(*)                        AS rows_out,
                COUNT(DISTINCT id)              AS distinct_ids,
                COUNT(DISTINCT symbol)          AS distinct_symbols,
                SUM(_missing_date::INTEGER)     AS rows_missing_date,
                MIN(created_date)               AS date_min,
                MAX(created_date)               AS date_max
            FROM normalized
        """).fetchone()
        if stats is None:
            raise _MissingNormalizationSummaryError

        (
            rows_out,
            _distinct_ids,
            distinct_symbols,
            rows_missing_date,
            date_min,
            date_max,
        ) = stats

        duplicates_dropped = rows_in - rows_out
        logger.info(
            "rows_out=%s, dupes_dropped=%s, missing_date=%s",
            f"{rows_out:,}",
            f"{duplicates_dropped:,}",
            f"{rows_missing_date:,}",
        )

        ticker_counts_rows = con.execute("""
            SELECT symbol, COUNT(*) AS n
            FROM normalized
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()
        per_ticker = {row[0]: int(row[1]) for row in ticker_counts_rows}

        year_hist_rows = con.execute("""
            SELECT
                YEAR(created_date) AS yr,
                COUNT(*)           AS n
            FROM normalized
            WHERE created_date IS NOT NULL
            GROUP BY yr
            ORDER BY yr
        """).fetchall()
        per_year = {int(row[0]): int(row[1]) for row in year_hist_rows}

    ticker_counts_sorted = sorted(per_ticker.values())
    n = len(ticker_counts_sorted)
    median_count = (
        ticker_counts_sorted[n // 2]
        if n % 2 == 1
        else (ticker_counts_sorted[n // 2 - 1] + ticker_counts_sorted[n // 2]) // 2
    )

    summary = {
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "duplicates_dropped": int(duplicates_dropped),
        "rows_missing_date": int(rows_missing_date),
        "distinct_symbols": int(distinct_symbols),
        "date_min": str(date_min) if date_min else None,
        "date_max": str(date_max) if date_max else None,
        "per_ticker_counts": per_ticker,
        "per_ticker_min": int(min(per_ticker.values())) if per_ticker else 0,
        "per_ticker_median": int(median_count),
        "per_ticker_max": int(max(per_ticker.values())) if per_ticker else 0,
        "per_year_histogram": per_year,
    }

    with Path(output_summary).open("w", encoding="utf-8") as f:
        yaml.dump(summary, f, sort_keys=False, default_flow_style=False)

    logger.info("Summary written to %s", output_summary)

    logger.info("=== Per-year date histogram (full deduped in-universe corpus) ===")
    for yr, cnt in sorted(per_year.items()):
        bar = "#" * min(cnt // 1000, 60)
        logger.info("  %s: %7s  %s", yr, f"{cnt:,}", bar)
    logger.info("=== End histogram ===")
    logger.info(
        "Done. rows_out=%s, distinct_symbols=%s, date range=%s to %s",
        f"{rows_out:,}",
        distinct_symbols,
        date_min,
        date_max,
    )


def subsample(
    symbol: str,
    input_path: Path,
    output: Path,
    options: SubsampleOptions,
) -> None:
    """Filter and subsample corpus for one ticker.

    Reads the normalized corpus parquet, filters to a given ticker symbol, date
    window, and minimum body length, then takes up to ``options.n_samples`` rows
    ordered by ``url_hash`` (deterministic md5-ordered sampling).
    """
    sym = symbol.upper()
    output.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE sampled AS
            SELECT *
            FROM read_parquet(?)
            WHERE symbol = ?
              AND created_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND body_len >= ?
            ORDER BY url_hash
            LIMIT ?
            """,
            [
                str(input_path),
                sym,
                options.start_date,
                options.end_date,
                options.min_body_chars,
                options.n_samples,
            ],
        )
        con.table("sampled").write_parquet(str(output))

        count_row = con.execute("SELECT COUNT(*) FROM sampled").fetchone()
        if count_row is None:
            raise _MissingSampleCountError
        count = count_row[0]
    logger.info("%s: %s rows written to %s", sym, count, output)

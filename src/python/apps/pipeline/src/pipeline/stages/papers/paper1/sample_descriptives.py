"""Full-universe sample-descriptives tabulation for Paper 1.

The manuscript previously deferred the full ticker list and per-sector
descriptive statistics to "the replication package". This module computes
them directly from the same inputs already used by the primary W2 /
energy-distance pipeline, so the paper can print the complete sample
composition rather than pointing a referee elsewhere.

Inputs
------
- ``data/universe.csv`` (``Symbol``, ``Name``, ``Sector``, ``Industry``):
  the sector partition and firm names.
- ``data/shared/embeddings/qwen3-embedding-8b/<TICKER>.parquet``: one row per
  article; ``created_date`` (``VARCHAR``, ``YYYY-MM-DD``) gives the article's
  publication date, used both for the raw per-firm article count and for the
  per-firm-per-year count.
- ``data/shared/returns/<TICKER>.parquet``: pre-computed per-firm daily
  ``return`` series (see
  :mod:`pipeline.stages.substrate.w2_exposure_frontier.returns_loadings`, which consumes
  the same files). This module does not recompute returns; it takes the
  ``return`` column's sample standard deviation as the firm's **daily**
  return-volatility figure, exactly as upstream defines it. No annualisation
  is applied anywhere in this module (all volatility figures reported here
  are daily-scale; see ``units`` keys in the emitted summary).
- ``data/shared/typed_distances/qwen3-embedding-8b/qwen3-embedding-8b-unit/``
  ``wasserstein_w2``: pairwise rooted W2 distances.

Sparse-sector honesty
----------------------
The 100-firm universe partitions into 9 sectors, two of them singletons
(Energy/FANG and Financial Services/PYPL) and one with ``n=3`` (Utilities);
the largest, Technology, holds 38. The expansion made the partition more
unbalanced, not less: it added two sparse sectors while concentrating a third
of the roster in one. Every per-sector
row carries its firm count (``n_firms``) so this imbalance is visible in the
table itself. A statistic that is mathematically undefined for a given
sector size — e.g. a *within-sector* pairwise energy distance for a
singleton sector, which has zero within-sector pairs — is emitted as
``None`` (YAML ``null``) with the pair count that produced it (``0``),
never silently replaced with ``0.0`` and never dropped from the per-sector
mapping. No sector is merged, renamed, or excluded to make the table appear
balanced.

Output
------
Writes a YAML summary to ``output_file`` (default
``data/papers/paper1/sample_descriptives/summary.yaml``) and returns the
same nested ``dict``. See ``run_sample_descriptives`` for the exact key
layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

logger = logging.getLogger(__name__)

_DEFAULT_UNIVERSE_CSV = Path("data/universe.csv")
_DEFAULT_EMBEDDINGS_DIR = Path("data/shared/embeddings/qwen3-embedding-8b")
_DEFAULT_RETURNS_DIR = Path("data/shared/returns")
_DEFAULT_DISTANCE_ARTIFACT_DIR = Path(
    "data/shared/typed_distances/qwen3-embedding-8b/"
    "qwen3-embedding-8b-unit/wasserstein_w2"
)
_DEFAULT_OUTPUT_FILE = Path("data/papers/paper1/sample_descriptives/summary.yaml")
_DEFAULT_PROVIDER_ID = "qwen3-embedding-8b"
_DEFAULT_REPRESENTATION_ID = "qwen3-embedding-8b-unit"
_DEFAULT_DISTANCE_ID = "wasserstein_w2"
MIN_VOLATILITY_OBSERVATIONS = 2


def _load_universe(universe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_csv)
    return df[["Symbol", "Name", "Sector"]].copy()


def _article_counts(embeddings_dir: Path, tickers: list[str]) -> pd.DataFrame:
    """Per-firm total article count and per-firm-per-year article count.

    Returns a DataFrame indexed by ticker with columns ``n_articles_total``
    and a ``dict[int, int]`` column ``n_articles_by_year``. A missing
    per-ticker embedding file yields ``float("nan")`` rather than ``0``: a
    firm that legitimately published no articles and a misconfigured
    ``embeddings_dir`` are different conditions, and collapsing them onto
    ``0`` would silently depress every article-count aggregate. This matches
    the undefined-value convention of :func:`_return_volatility`.
    """
    con = duckdb.connect(":memory:")
    rows = []
    for ticker in tickers:
        path = embeddings_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(
                "no embeddings parquet for %s at %s; article counts undefined",
                ticker,
                path,
            )
            empty_by_year: dict[int, int] = {}
            rows.append(
                {
                    "Symbol": ticker,
                    "n_articles_total": float("nan"),
                    "n_articles_by_year": empty_by_year,
                }
            )
            continue
        count_row = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(path)]
        ).fetchone()
        total = 0 if count_row is None else count_row[0]
        by_year_df = con.execute(
            """
                SELECT extract(year FROM CAST(created_date AS DATE)) AS yr,
                       count(*) AS n
                FROM read_parquet(?)
                GROUP BY yr
                ORDER BY yr
            """,
            [str(path)],
        ).df()
        # Cast through the frame's dtypes rather than per-row attributes:
        # duckdb's extract(year ...) may surface as int64 or float64 depending
        # on null handling, and .tolist() yields plain Python ints either way.
        by_year = dict(
            zip(
                by_year_df["yr"].astype("int64").tolist(),
                by_year_df["n"].astype("int64").tolist(),
                strict=True,
            )
        )
        rows.append(
            {
                "Symbol": ticker,
                "n_articles_total": int(total),
                "n_articles_by_year": by_year,
            }
        )
    con.close()
    return pd.DataFrame(rows).set_index("Symbol")


def _return_volatility(returns_dir: Path, tickers: list[str]) -> dict[str, float]:
    """Per-firm sample standard deviation of the daily ``return`` column.

    NaN rows (the first observation of each per-ticker file has no prior
    close) are dropped before computing the standard deviation. A firm
    with fewer than 2 valid observations returns ``float("nan")`` (never
    ``0.0``), since a standard deviation is undefined on <2 points.
    """
    con = duckdb.connect(":memory:")
    vol: dict[str, float] = {}
    for ticker in tickers:
        path = returns_dir / f"{ticker}.parquet"
        if not path.exists():
            vol[ticker] = float("nan")
            continue
        row = con.execute(
            """
                SELECT stddev_samp("return") AS vol, count("return") AS n
                FROM read_parquet(?)
                WHERE "return" IS NOT NULL
            """,
            [str(path)],
        ).fetchone()
        if (
            row is None
            or row[1] is None
            or row[1] < MIN_VOLATILITY_OBSERVATIONS
            or row[0] is None
        ):
            vol[ticker] = float("nan")
        else:
            vol[ticker] = float(row[0])
    con.close()
    return vol


def _load_w2_pairs(
    distance_artifact_dir: Path,
    provider_id: str = _DEFAULT_PROVIDER_ID,
    representation_id: str = _DEFAULT_REPRESENTATION_ID,
    distance_id: str = _DEFAULT_DISTANCE_ID,
) -> pd.DataFrame:
    # The identity is a parameter, not a constant, because Papers 1 and 5 run
    # this tabulation over different encoders. It stays an *asserted* identity
    # so a caller still cannot silently read one representation's distances
    # while claiming another's.
    frame, summary = read_typed_distance_artifact(
        distance_artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": distance_id,
            "value_semantics": "statistical_distance",
        },
    )
    item_ids = summary.get("item_ids")
    if not isinstance(item_ids, list):
        message = "typed W2 summary has no item roster"
        raise TypeError(message)
    values = frame["value"].to_numpy(dtype=float).reshape(len(item_ids), len(item_ids))
    return pd.DataFrame(
        [
            {
                "symbol1": str(item_ids[i]),
                "symbol2": str(item_ids[j]),
                "w2_distance": float(values[i, j]),
            }
            for i in range(len(item_ids))
            for j in range(i + 1, len(item_ids))
        ]
    )


def _sector_w2_distances(
    pairs: pd.DataFrame, sectors: dict[str, str]
) -> dict[str, dict[str, float | int | None]]:
    """Compute within- and cross-sector mean pairwise rooted W2 distances.

    ``within`` is ``None`` when the sector has fewer than 2 same-sector
    pairs (e.g. any singleton sector) — an undefined statistic, not a zero.
    ``cross`` requires at least 1 sector-vs-other pair to be defined; across
    the sector partition this is not expected to be empty, but the same
    ``None``-on-undefined convention is applied for consistency.
    """
    # A firm with no Sector arrives as NaN and reaches `sorted()` alongside the
    # string labels, where it fails as "'<' not supported between float and
    # str" -- an error that names neither the column nor the firm. Check first
    # so a gap in universe.csv is reported as the data problem it is.
    unlabelled = sorted(
        symbol
        for symbol, sector in sectors.items()
        if not isinstance(sector, str) or not sector.strip()
    )
    if unlabelled:
        message = (
            "universe.csv has no Sector for: "
            + ", ".join(unlabelled)
            + ". Sector drives the within/cross partition, so a blank cannot be "
            "defaulted; supply the classification."
        )
        raise ValueError(message)

    sec1 = pairs["symbol1"].map(sectors)
    sec2 = pairs["symbol2"].map(sectors)
    same = sec1 == sec2
    out: dict[str, dict[str, float | int | None]] = {}
    for sector in sorted(set(sectors.values())):
        within_mask = same & (sec1 == sector)
        cross_mask = (~same) & ((sec1 == sector) | (sec2 == sector))
        within_vals = pairs.loc[within_mask, "w2_distance"]
        cross_vals = pairs.loc[cross_mask, "w2_distance"]
        out[sector] = {
            "mean_within_sector_w2_distance": (
                float(within_vals.mean()) if len(within_vals) >= 1 else None
            ),
            "n_within_sector_pairs": len(within_vals),
            "mean_cross_sector_w2_distance": (
                float(cross_vals.mean()) if len(cross_vals) >= 1 else None
            ),
            "n_cross_sector_pairs": len(cross_vals),
        }
    return out


def run_sample_descriptives(
    universe_csv: Path = _DEFAULT_UNIVERSE_CSV,
    embeddings_dir: Path = _DEFAULT_EMBEDDINGS_DIR,
    returns_dir: Path = _DEFAULT_RETURNS_DIR,
    distance_artifact_dir: Path = _DEFAULT_DISTANCE_ARTIFACT_DIR,
    output_file: Path = _DEFAULT_OUTPUT_FILE,
    provider_id: str = _DEFAULT_PROVIDER_ID,
    representation_id: str = _DEFAULT_REPRESENTATION_ID,
    distance_id: str = _DEFAULT_DISTANCE_ID,
) -> dict[str, object]:
    """Compute and persist the full-universe sample-descriptives table.

    Top-level keys of the returned/written mapping:

    - ``units``: documents the scale of each reported statistic (all
      volatility figures are daily, undivided by any annualisation factor;
      article counts are raw row counts).
    - ``overall``: pooled statistics across the full universe (see
      docstring of the module for exact fields).
    - ``sectors``: mapping ``Sector -> {...}``, one row per sector actually
      present in ``universe_csv`` (unbalanced sizes preserved, nothing
      merged or dropped). Fields absent/undefined for a given sector are
      ``None``, never ``0.0``.
    - ``universe``: the complete ``Symbol -> {name, sector}`` listing,
      sorted by (Sector, Symbol), so the manuscript can print the full
      ticker roster.
    """
    universe = _load_universe(universe_csv)
    tickers = sorted(universe["Symbol"].tolist())
    sectors = dict(zip(universe["Symbol"], universe["Sector"], strict=False))
    names = dict(zip(universe["Symbol"], universe["Name"], strict=False))

    article_df = _article_counts(embeddings_dir, tickers)
    vol_by_ticker = _return_volatility(returns_dir, tickers)
    pairs = _load_w2_pairs(
        distance_artifact_dir, provider_id, representation_id, distance_id
    )
    sector_w2 = _sector_w2_distances(pairs, sectors)

    sector_sizes = universe["Sector"].value_counts()

    sectors_out: dict[str, dict[str, object]] = {}
    for sector in sorted(sector_sizes.index):
        members = sorted(universe.loc[universe["Sector"] == sector, "Symbol"])
        n_firms = len(members)
        article_totals = article_df.loc[members, "n_articles_total"]

        # Per-firm-per-year article counts, pooled across the sector's
        # firms: mean/median firm-year article count (a firm contributes
        # one observation per year it has >=1 article; years with zero
        # articles for a firm are not fabricated as explicit zero-rows).
        firm_year_counts: list[int] = []
        for m in members:
            by_year = article_df.loc[m, "n_articles_by_year"]
            firm_year_counts.extend(by_year.values())

        firm_vols = [
            vol_by_ticker[m] for m in members if not np.isnan(vol_by_ticker[m])
        ]

        sectors_out[sector] = {
            "n_firms": int(n_firms),
            "mean_articles_per_firm": float(article_totals.mean()),
            "median_articles_per_firm": float(article_totals.median()),
            "mean_articles_per_firm_year": (
                float(np.mean(firm_year_counts)) if firm_year_counts else None
            ),
            "median_articles_per_firm_year": (
                float(np.median(firm_year_counts)) if firm_year_counts else None
            ),
            "mean_daily_return_volatility": (
                float(np.mean(firm_vols)) if firm_vols else None
            ),
            "n_firms_with_volatility": len(firm_vols),
            **sector_w2[sector],
        }

    all_article_totals = article_df.loc[tickers, "n_articles_total"]
    all_firm_year_counts: list[int] = []
    for m in tickers:
        all_firm_year_counts.extend(article_df.loc[m, "n_articles_by_year"].values())
    all_vols = [vol_by_ticker[m] for m in tickers if not np.isnan(vol_by_ticker[m])]

    all_within_vals = pairs.loc[
        pairs["symbol1"].map(sectors) == pairs["symbol2"].map(sectors),
        "w2_distance",
    ]
    all_cross_vals = pairs.loc[
        pairs["symbol1"].map(sectors) != pairs["symbol2"].map(sectors),
        "w2_distance",
    ]

    overall = {
        "n_firms": len(tickers),
        "n_sectors": int(sector_sizes.shape[0]),
        # Reported over firms with an observed article count; a firm whose
        # embeddings parquet is missing contributes NaN (never 0), so the
        # companion count below states how many firms the total covers. With
        # no observed counts at all the total is undefined, not zero --
        # pandas' .sum() would otherwise report a fabricated 0.
        "total_articles": (
            int(all_article_totals.sum()) if all_article_totals.notna().any() else None
        ),
        "n_firms_with_article_counts": int(all_article_totals.notna().sum()),
        "mean_articles_per_firm": float(all_article_totals.mean()),
        "median_articles_per_firm": float(all_article_totals.median()),
        "mean_articles_per_firm_year": (
            float(np.mean(all_firm_year_counts)) if all_firm_year_counts else None
        ),
        "median_articles_per_firm_year": (
            float(np.median(all_firm_year_counts)) if all_firm_year_counts else None
        ),
        "mean_daily_return_volatility": (
            float(np.mean(all_vols)) if all_vols else None
        ),
        "n_firms_with_volatility": len(all_vols),
        "mean_within_sector_w2_distance": (
            float(all_within_vals.mean()) if len(all_within_vals) >= 1 else None
        ),
        "n_within_sector_pairs": len(all_within_vals),
        "mean_cross_sector_w2_distance": (
            float(all_cross_vals.mean()) if len(all_cross_vals) >= 1 else None
        ),
        "n_cross_sector_pairs": len(all_cross_vals),
    }

    universe_listing = [
        {"symbol": t, "name": str(names[t]), "sector": str(sectors[t])}
        for t in sorted(tickers, key=lambda t: (sectors[t], t))
    ]

    result: dict[str, object] = {
        "distance_identity": {
            "provider_id": "qwen3-embedding-8b",
            "representation_id": "qwen3-embedding-8b-unit",
            "distance_id": "wasserstein_w2",
        },
        "sector_taxonomy": {
            "source": "Nasdaq summary metadata materialized in data/universe.csv",
            "snapshot_date": "not recorded in the source artifact",
            "uses": ["descriptives", "dyadic regression", "sector alignment", "DISCO"],
        },
        "units": {
            "articles_per_firm": (
                "raw article count (rows in per-ticker embeddings parquet)"
            ),
            "articles_per_firm_year": (
                "raw article count, one observation per (firm, calendar year) "
                "with >=1 article"
            ),
            "return_volatility": (
                "daily, sample std of the pre-computed `return` column in "
                "data/shared/returns/<TICKER>.parquet; NOT annualised"
            ),
            "w2_distance": (
                "rooted balanced quadratic Wasserstein distance with Euclidean "
                "chord ground distance"
            ),
        },
        "overall": overall,
        "sectors": sectors_out,
        "universe": universe_listing,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return result

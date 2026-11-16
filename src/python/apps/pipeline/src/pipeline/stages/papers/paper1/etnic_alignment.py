"""External validation of Paper 1 news geometry against Hoberg-Phillips ETNIC.

ETNIC (Hoberg and Phillips 2025) embeds each firm's 10-K Item 1 with doc2vec.
ETNIC-All carries every pair's similarity; ETNIC-3 is the same score shifted by
a constant and truncated at the three-digit-SIC granularity threshold. The
extract keeps both: the continuous score for Mantel tests and the exact
ETNIC-3 membership for link-classification diagnostics.

The two sources differ in text (news vs 10-K) and encoder (Qwen3 vs doc2vec),
so agreement provides convergent evidence across text sources. The matrix
correlations and pair similarities feed Paper 1's generated values and
ETNIC-controlled firm-effects regressions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from statadic.hypothesis.core import Hypothesis
from statadic.hypothesis.homogeneity import PermutationPlan
from statadic.hypothesis.independence import Mantel, MantelInput, MatrixExchangeability

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.association.mantel import MantelMethod
    from statadic.core.typing import Array

_TOP_K = 5


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EtnicAlignmentConfig:
    """Inputs and frozen settings for the ETNIC alignment diagnostics."""

    extract: Path
    crosswalk: Path
    universe_csv: Path
    distance_root: Path
    geometries: tuple[str, ...]
    primary_geometry: str
    permutations: int
    bootstrap_draws: int
    seed: int
    output_summary: Path
    output_pairs: Path


def _complete_roster(
    frame: pd.DataFrame, firms: list[int]
) -> tuple[list[int], list[int]]:
    """Drop the fewest-year firm until every retained pair is co-observed."""
    kept: list[int] = list(firms)
    dropped: list[int] = []
    years = frame.groupby("gvkey1").year.apply(set).to_dict()
    while True:
        missing = [
            (a, b)
            for i, a in enumerate(kept)
            for b in kept[i + 1 :]
            if not years.get(a, set()) & years.get(b, set())
        ]
        if not missing:
            return kept, dropped
        involved = {g for pair in missing for g in pair}
        worst = min(involved, key=lambda g: (len(years.get(g, set())), g))
        kept.remove(worst)
        dropped.append(worst)


def _similarity(frame: pd.DataFrame, index: dict[int, int]) -> np.ndarray:
    """Mean score over co-observed years, both directions averaged."""
    n = len(index)
    total, count = np.zeros((n, n)), np.zeros((n, n))
    f = frame[frame.gvkey1.isin(index) & frame.gvkey2.isin(index)]
    i, j = f.gvkey1.map(index).to_numpy(), f.gvkey2.map(index).to_numpy()
    np.add.at(total, (i, j), f.score.to_numpy())
    np.add.at(count, (i, j), 1.0)
    total, count = total + total.T, count + count.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    np.fill_diagonal(out, np.nan)
    return out


def _distance(root: Path, geometry: str, tickers: list[str]) -> np.ndarray:
    frame = pd.read_parquet(root / geometry / "distances.parquet")
    index = {t: k for k, t in enumerate(tickers)}
    frame = frame[frame.item_i.isin(index) & frame.item_j.isin(index)]
    out = np.full((len(tickers), len(tickers)), np.nan)
    out[frame.item_i.map(index), frame.item_j.map(index)] = frame.value
    out = np.fmax(out, out.T)
    np.fill_diagonal(out, 0.0)
    if np.isnan(out).any():
        message = f"{geometry} distances do not cover the ETNIC roster"
        raise ValueError(message)
    return out


def _mantel(
    x: np.ndarray, y: np.ndarray, method: MantelMethod, cfg: EtnicAlignmentConfig
) -> dict[str, float]:
    plan = PermutationPlan(n=cfg.permutations, seed=cfg.seed)
    fitted = Mantel(method).estimate(MantelInput(x, y))
    outcome = fitted.infer(resampling=plan).test(Hypothesis(MatrixExchangeability()))
    if outcome.pvalue is None:
        message = "the Mantel correlation or its permutation null is undefined"
        raise ValueError(message)
    return {"r": float(fitted.statistic), "p": outcome.pvalue}


def _sector_partial(
    x: np.ndarray, doc: np.ndarray, sector: np.ndarray
) -> dict[str, float | str]:
    """Correlate the two matrices after removing sector matching from both.

    Descriptive only. Residualizing once and then permuting firms without
    restriction treats firms as exchangeable across sectors, which removing
    sector means does not establish, so no permutation p-value is reported;
    the firm-bootstrap dyadic regression carries the conditional evidence.
    """
    partial = Mantel().estimate(MantelInput(x, doc, control=sector)).partial
    return {
        "r": float(cast("Array", partial)),
        "inference": "descriptive; cross-sector exchangeability is not established",
    }


def _mantel_block(
    x: np.ndarray, doc: np.ndarray, sector: np.ndarray, cfg: EtnicAlignmentConfig
) -> dict[str, dict[str, float] | dict[str, float | str]]:
    """Permutation-tested Pearson and Spearman, and descriptive sector-partial."""
    return {
        "pearson": _mantel(x, doc, "pearson", cfg),
        "spearman": _mantel(x, doc, "spearman", cfg),
        "partial_sector": _sector_partial(x, doc, sector),
    }


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    positives = int(label.sum())
    negatives = len(label) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(score)
    return float(
        (ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def _link_stats(
    ids: np.ndarray,
    dist: np.ndarray,
    years_linked: np.ndarray,
    etnic3: np.ndarray,
    same: np.ndarray,
) -> dict[str, float]:
    """Link-classification statistics on a (possibly resampled) firm roster."""
    a, b = np.triu_indices(len(ids), 1)
    i, j = ids[a], ids[b]
    keep = i != j
    i, j = i[keep], j[keep]
    d, yr, s3, sm = dist[i, j], years_linked[i, j], etnic3[i, j], same[i, j]
    link = yr > 0
    sub = dist[np.ix_(ids, ids)].copy()
    sub[ids[:, None] == ids[None, :]] = np.inf
    linked = years_linked[np.ix_(ids, ids)] > 0
    hits = [
        linked[k, np.argsort(sub[k])[:_TOP_K]].mean()
        for k in range(len(ids))
        if linked[k].any()
    ]
    out = {
        "auc": _auc(-d, link),
        "auc_sector": _auc(sm.astype(float), link),
        "auc_within_sector": _auc(-d[sm], link[sm]),
        "auc_cross_sector": _auc(-d[~sm], link[~sm]),
        "spearman_years_linked": float(spearmanr(d, yr).statistic),
        "spearman_etnic3_score": float(spearmanr(d[link], s3[link]).statistic),
        "top5_linked_share": float(np.mean(hits)),
        "link_base_rate": float(link.mean()),
    }
    out["auc_minus_sector"] = out["auc"] - out["auc_sector"]
    return out


def run_etnic_alignment(cfg: EtnicAlignmentConfig) -> dict[str, object]:
    """Compute Mantel and link-classification diagnostics; write summary and pairs."""
    extract = pd.read_parquet(cfg.extract)
    extract = extract[extract.gvkey1 != extract.gvkey2]
    link = pd.read_csv(cfg.crosswalk, dtype={"cik": str})
    universe = pd.read_csv(cfg.universe_csv)
    roster = link.merge(
        universe[["Symbol", "Sector"]], left_on="ticker", right_on="Symbol"
    )
    observed = set(extract.gvkey1)
    firms, dropped = _complete_roster(
        extract, sorted(g for g in roster.gvkey.astype(int) if g in observed)
    )
    roster = roster.set_index("gvkey").loc[firms].reset_index()
    tickers = roster.ticker.tolist()
    index = {g: k for k, g in enumerate(firms)}
    sector_labels = roster.Sector.to_numpy(dtype=object)
    same = sector_labels[:, None] == sector_labels[None, :]
    sector = same.astype(float)

    doc = -_similarity(extract, index)
    np.fill_diagonal(doc, 0.0)
    if np.isnan(doc).any():
        message = "ETNIC-All matrix incomplete after roster completion"
        raise ValueError(message)

    geometry = {
        name: _mantel_block(
            _distance(cfg.distance_root, name, tickers), doc, sector, cfg
        )
        for name in cfg.geometries
    }
    primary = _distance(cfg.distance_root, cfg.primary_geometry, tickers)

    per_year = {}
    for year, frame in extract.groupby("year"):
        present = sorted(set(frame.gvkey1) & set(firms))
        sub = [index[g] for g in present]
        year_doc = -_similarity(frame, {g: k for k, g in enumerate(present)})
        np.fill_diagonal(year_doc, 0.0)
        per_year[int(cast("int", year))] = {
            "firms": len(present),
            **_mantel(primary[np.ix_(sub, sub)], year_doc, "pearson", cfg),
        }

    etnic3 = extract[extract.etnic3_link]
    years_linked = np.zeros((len(firms), len(firms)))
    score3 = np.full((len(firms), len(firms)), np.nan)
    grouped = (
        etnic3.groupby(["gvkey1", "gvkey2"])
        .agg(years=("year", "nunique"), score=("etnic3_score", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped.gvkey1.isin(index) & grouped.gvkey2.isin(index)]
    gi, gj = grouped.gvkey1.map(index).to_numpy(), grouped.gvkey2.map(index).to_numpy()
    years_linked[gi, gj] = grouped.years.to_numpy()
    score3[gi, gj] = grouped.score.to_numpy()

    point = _link_stats(np.arange(len(firms)), primary, years_linked, score3, same)
    rng = np.random.default_rng(cfg.seed)
    draws = pd.DataFrame(
        [
            _link_stats(
                rng.integers(0, len(firms), len(firms)),
                primary,
                years_linked,
                score3,
                same,
            )
            for _ in range(cfg.bootstrap_draws)
        ]
    )
    link_block = {
        key: {
            "estimate": value,
            "ci95": [float(v) for v in np.nanpercentile(draws[key], [2.5, 97.5])],
            "se": float(draws[key].std()),
        }
        for key, value in point.items()
    }
    link_block["auc_minus_sector"]["share_draws_le_zero"] = float(
        (draws.auc_minus_sector <= 0).mean()
    )

    a, b = np.triu_indices(len(firms), 1)
    pairs = pd.DataFrame(
        {
            "ticker_i": [tickers[k] for k in a],
            "ticker_j": [tickers[k] for k in b],
            "gvkey_i": [firms[k] for k in a],
            "gvkey_j": [firms[k] for k in b],
            "same_sector": same[a, b],
            cfg.primary_geometry: primary[a, b],
            "etnic_all_similarity": -doc[a, b],
            "etnic3_years_linked": years_linked[a, b].astype(int),
            "etnic3_mean_score": score3[a, b],
        }
    )
    cfg.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(cfg.output_pairs, index=False)

    summary: dict[str, object] = {
        "schema": "paper1-etnic-alignment-v1",
        "interpretation": (
            "convergent evidence: news article-cloud geometry vs 10-K doc2vec; "
            "correlations and pair similarities feed Paper 1 classification controls"
        ),
        "roster": {
            "crosswalk_firms": len(link),
            "etnic_firms": len(firms),
            "dropped_for_incomplete_pairs": [roster_ticker(link, g) for g in dropped],
            "pairs": len(pairs),
            "years": sorted(int(y) for y in extract.year.unique()),
        },
        "doc2vec_distance": "negative mean ETNIC-All score over co-observed years",
        "permutations": cfg.permutations,
        "bootstrap_draws": cfg.bootstrap_draws,
        "seed": cfg.seed,
        "mantel": geometry,
        "mantel_sector_reference": _mantel(1.0 - sector, doc, "pearson", cfg),
        "mantel_per_year_primary": per_year,
        "etnic3_link_classification": link_block,
    }
    cfg.output_summary.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def roster_ticker(link: pd.DataFrame, gvkey: int) -> str:
    """Resolve a gvkey back to the crosswalk ticker for reporting."""
    match = link.loc[link.gvkey.astype(int) == gvkey, "ticker"]
    return str(match.iloc[0]) if len(match) else str(gvkey)

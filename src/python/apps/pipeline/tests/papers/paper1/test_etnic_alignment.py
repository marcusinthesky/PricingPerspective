"""ETNIC alignment fixtures on synthetic saved numeric networks."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml
from pipeline.stages.papers.paper1.etnic_alignment import (
    EtnicAlignmentConfig,
    _complete_roster,
    _mantel,
    _sector_partial,
    run_etnic_alignment,
)
from statadic.hypothesis.core import Hypothesis
from statadic.hypothesis.homogeneity import PermutationPlan
from statadic.hypothesis.independence import Mantel, MantelInput, MatrixExchangeability

if TYPE_CHECKING:
    from pathlib import Path

_OFFSET = 0.2367
_FIRMS = 12
_SECTORS = 3


def _network(years: list[int], gvkeys: list[int], sim: np.ndarray) -> pd.DataFrame:
    rows = [
        (y, a, b, np.nan if a == b else float(sim[i, j]))
        for y in years
        for (i, a), (j, b) in product(enumerate(gvkeys), repeat=2)
    ]
    return pd.DataFrame(rows, columns=["year", "gvkey1", "gvkey2", "score"])


def _world(tmp_path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], list[int]]:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(_FIRMS, 4)) + np.repeat(
        np.eye(_SECTORS, 4) * 3, _FIRMS // _SECTORS, axis=0
    )
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    sim = 0.8 - 0.1 * dist  # doc2vec similarity decreasing in news distance
    tickers = [f"T{k:02d}" for k in range(_FIRMS)]
    gvkeys = [1000 + k for k in range(_FIRMS)]
    pd.DataFrame(
        {
            "ticker": tickers,
            "cik": [f"{k:010d}" for k in range(_FIRMS)],
            "gvkey": gvkeys,
        }
    ).to_csv(tmp_path / "crosswalk.csv", index=False)
    pd.DataFrame(
        {
            "Symbol": tickers,
            "Sector": [f"S{k // (_FIRMS // _SECTORS)}" for k in range(_FIRMS)],
        }
    ).to_csv(tmp_path / "universe.csv", index=False)
    geometry = tmp_path / "distances" / "wasserstein_w2"
    geometry.mkdir(parents=True)
    pd.DataFrame(
        [
            (a, b, float(dist[i, j]))
            for (i, a), (j, b) in product(enumerate(tickers), repeat=2)
        ],
        columns=["item_i", "item_j", "value"],
    ).to_parquet(geometry / "distances.parquet")
    frame = _network([2017, 2018, 2019], [*gvkeys, 9999], np.pad(sim, (0, 1)))
    return frame, dist, tickers, gvkeys


def test_roster_completion_drops_the_never_co_observed_firm() -> None:
    """A firm seen only in a year its partner is absent is removed, not imputed."""
    # Firm 1 spans both years; 2 is seen only in 2018 and 3 only in 2020, so the
    # pair (2, 3) is never co-observed. Ties on year count break on the gvkey.
    frame = pd.DataFrame(
        {"year": [2018, 2019, 2020, 2018, 2020], "gvkey1": [1, 1, 1, 2, 3]}
    )
    kept, dropped = _complete_roster(frame, [1, 2, 3])
    assert dropped == [2]
    assert kept == [1, 3]


def test_alignment_recovers_planted_geometry(tmp_path: Path) -> None:
    """A saved numeric comparator preserves planted alignment and the roster."""
    frame, _dist, _tickers, gvkeys = _world(tmp_path)
    extract = frame.loc[
        frame["year"].between(2018, 2019)
        & frame["gvkey1"].isin(gvkeys)
        & frame["gvkey2"].isin(gvkeys)
        & (frame["gvkey1"] != frame["gvkey2"])
    ].copy()
    extract["etnic3_link"] = extract["score"] > _OFFSET
    extract["etnic3_score"] = (
        (extract["score"] - _OFFSET).round(4).where(extract["etnic3_link"])
    )
    extract.to_parquet(tmp_path / "extract.parquet", index=False)
    cfg = EtnicAlignmentConfig(
        extract=tmp_path / "extract.parquet",
        crosswalk=tmp_path / "crosswalk.csv",
        universe_csv=tmp_path / "universe.csv",
        distance_root=tmp_path / "distances",
        geometries=("wasserstein_w2",),
        primary_geometry="wasserstein_w2",
        permutations=199,
        bootstrap_draws=49,
        seed=1,
        output_summary=tmp_path / "out/summary.yaml",
        output_pairs=tmp_path / "out/pairs.parquet",
    )
    run_etnic_alignment(cfg)
    summary = yaml.safe_load(cfg.output_summary.read_text())
    mantel = summary["mantel"]["wasserstein_w2"]
    assert mantel["spearman"]["r"] == pytest.approx(1.0)
    assert mantel["pearson"]["p"] <= 0.01  # noqa: PLR2004
    assert "p" not in mantel["partial_sector"]
    assert summary["roster"]["etnic_firms"] == _FIRMS
    link = summary["etnic3_link_classification"]
    assert link["auc"]["estimate"] == pytest.approx(1.0)
    pairs = pd.read_parquet(cfg.output_pairs)
    assert len(pairs) == _FIRMS * (_FIRMS - 1) // 2


def test_stage_mantel_blocks_equal_native_owners(tmp_path: Path) -> None:
    """The migrated stage helpers return the native Mantel test and partial."""
    rng = np.random.default_rng(2)
    points = rng.normal(size=(_FIRMS, 3))
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    noise = rng.normal(size=(_FIRMS, _FIRMS)) * 0.2
    doc = dist + noise + noise.T
    np.fill_diagonal(doc, 0.0)
    labels = np.repeat(np.arange(_SECTORS), _FIRMS // _SECTORS)
    sector = (labels[:, None] == labels[None, :]).astype(float)
    cfg = EtnicAlignmentConfig(
        extract=tmp_path,
        crosswalk=tmp_path,
        universe_csv=tmp_path,
        distance_root=tmp_path,
        geometries=(),
        primary_geometry="",
        permutations=99,
        bootstrap_draws=0,
        seed=7,
        output_summary=tmp_path,
        output_pairs=tmp_path,
    )
    fit = Mantel("spearman").estimate(MantelInput(dist, doc))
    native = fit.infer(resampling=PermutationPlan(n=99, seed=7)).test(
        Hypothesis(MatrixExchangeability())
    )
    assert native.pvalue is not None
    assert _mantel(dist, doc, "spearman", cfg) == {
        "r": float(fit.statistic),
        "p": float(native.pvalue),
    }
    a, b = np.triu_indices(_FIRMS, 1)
    design = np.column_stack([np.ones(len(a)), sector[a, b]])
    left, right = (
        v[a, b] - design @ np.linalg.lstsq(design, v[a, b], rcond=None)[0]
        for v in (dist, doc)
    )
    assert _sector_partial(dist, doc, sector)["r"] == pytest.approx(
        np.corrcoef(left, right)[0, 1], rel=1e-6
    )

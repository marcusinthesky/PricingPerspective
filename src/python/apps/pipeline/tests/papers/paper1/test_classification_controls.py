"""Classification controls must isolate W2's incremental conditional fit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import pandas as pd
import pytest
from pipeline.stages.papers.paper1._dyadic.classification import (
    classification_indicators,
    etnic_control,
    fit_classification_controls,
)
from statadic.inference.dyadic import DyadicInference
from statadic.law.dyadic import DyadicInputError, DyadicModel, DyadicSample
from statadic.numerics.linalg.dyadic import symmetric_dyadic_effects

if TYPE_CHECKING:
    from pathlib import Path

MIN_VALID_DRAWS = 100


def test_controls_and_within_industry_match_independent_least_squares(
    tmp_path: Path,
) -> None:
    """Correlated matching controls cannot be silently omitted from the baseline."""
    rng = np.random.default_rng(91)
    firms = np.array([f"F{k:02d}" for k in range(24)])
    i, j = np.triu_indices(len(firms), 1)
    sector = (i // 12 == j // 12).astype(float)
    industry = (i // 6 == j // 6).astype(float)
    w2 = rng.uniform(0.1, 0.5, len(i)) + 0.3 * sector + 0.2 * industry
    margins = rng.normal(0.0, 0.1, len(firms))
    y = 0.8 * w2 - 0.3 * sector - 0.2 * industry + margins[i] + margins[j]
    y += rng.normal(0.0, 0.04, len(i))
    sample = DyadicSample[str](y, firms[i].tolist(), firms[j].tolist())
    roster = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "Symbol": firms,
            "Sector": [f"S{k // 12}" for k in range(len(firms))],
            "Industry": [f"I{k // 6}" for k in range(len(firms))],
        }
    ).to_csv(roster, index=False)
    # The stage owns float64 through its CLI command; the test must do the same.
    with jax.enable_x64(new_val=True):
        summary, draws = fit_classification_controls(
            roster,
            DyadicModel(["w2_distance"], {"w2_distance": w2}, sample),
            DyadicInference(bootstrap_iters=120, bootstrap_seed=19),
        )
    for row in cast("list[dict[str, Any]]", summary["rows"]):
        keep = (
            industry.astype(bool)
            if row["key"] == "within-industry"
            else np.ones(len(y), bool)
        )
        effects, _ = symmetric_dyadic_effects(
            firms[i][keep].tolist(), firms[j][keep].tolist()
        )
        controls = {"same_sector": sector, "same_industry": industry}
        nuisance = np.column_stack(
            [
                np.ones(keep.sum()),
                np.asarray(effects),
                *[controls[name][keep] for name in row["controls"]],
            ]
        )
        full = np.column_stack([nuisance, w2[keep]])
        beta = np.linalg.lstsq(full, y[keep], rcond=None)[0]
        residual = y[keep] - full @ beta
        restricted = (
            y[keep] - nuisance @ np.linalg.lstsq(nuisance, y[keep], rcond=None)[0]
        )
        assert row["coefficient"] == pytest.approx(beta[-1], abs=1e-10)
        assert row["partial_r2"] == pytest.approx(
            1 - residual @ residual / (restricted @ restricted), abs=1e-10
        )
        assert row["n_dyads"] == keep.sum()
        assert row["bootstrap_valid_draws"] >= MIN_VALID_DRAWS
        chord = np.median(y[keep])
        delta = row["effect_one_sd"]
        assert row["delta_correlation"] == pytest.approx(-chord * delta - delta**2 / 2)
    assert (
        cast("dict[str, object]", summary["classification"])["point_in_time"] is False
    )
    assert {int(frame.spec_index.iloc[0]) for frame in draws} == {10, 11, 12}


@pytest.mark.parametrize("bad_label", [None, " "])
def test_missing_classification_is_not_a_matching_pair(
    tmp_path: Path, bad_label: str | None
) -> None:
    """Two absent labels must never silently become an industry match."""
    roster = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "Sector": ["S", "S"],
            "Industry": [bad_label, bad_label],
        }
    ).to_csv(roster, index=False)
    sample = DyadicSample[str](
        np.array([1.0]), np.array(["A"]).tolist(), np.array(["B"]).tolist()
    )
    with pytest.raises(DyadicInputError, match="cover every sample firm"):
        classification_indicators(roster, sample)


def _etnic_alignment(tmp_path: Path, roster: np.ndarray, score: np.ndarray) -> Path:
    """Write a minimal ETNIC alignment output over ``roster``'s pairs."""
    out = tmp_path / "etnic_alignment"
    out.mkdir()
    a, b = np.triu_indices(len(roster), 1)
    pd.DataFrame(
        {
            "ticker_i": roster[a],
            "ticker_j": roster[b],
            "etnic_all_similarity": score[a, b],
        }
    ).to_parquet(out / "pairs.parquet", index=False)
    stat = {"r": 0.5, "p": 1e-4}
    (out / "results.json").write_text(
        json.dumps(
            {
                "roster": {"years": [2018, 2019]},
                "permutations": 99,
                "mantel": {
                    "wasserstein_w2": {
                        "pearson": stat,
                        "spearman": stat,
                        "partial_sector": {"r": 0.3, "inference": "descriptive"},
                    }
                },
            }
        )
    )
    return out


def test_etnic_rows_share_a_roster_and_match_least_squares(tmp_path: Path) -> None:
    """Adding ETNIC changes only the control, never the sample W2 is fitted on."""
    rng = np.random.default_rng(7)
    firms = np.array([f"F{k:02d}" for k in range(24)])
    i, j = np.triu_indices(len(firms), 1)
    sector = (i // 12 == j // 12).astype(float)
    score = rng.uniform(0.0, 0.3, (len(firms), len(firms)))
    score = (score + score.T) / 2
    etnic = score[i, j]
    w2 = rng.uniform(0.1, 0.5, len(i)) - 0.4 * etnic + 0.3 * sector
    margins = rng.normal(0.0, 0.1, len(firms))
    y = 0.6 * w2 - 0.5 * etnic - 0.2 * sector + margins[i] + margins[j]
    y += rng.normal(0.0, 0.04, len(i))
    roster = firms[:18]
    alignment = _etnic_alignment(tmp_path, roster, score[:18, :18])
    universe = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "Symbol": firms,
            "Sector": [f"S{k // 12}" for k in range(len(firms))],
            "Industry": [f"I{k // 6}" for k in range(len(firms))],
        }
    ).to_csv(universe, index=False)
    with jax.enable_x64(new_val=True):
        summary, _draws = fit_classification_controls(
            universe,
            DyadicModel(
                ["w2_distance"],
                {"w2_distance": w2},
                DyadicSample[str](y, firms[i].tolist(), firms[j].tolist()),
            ),
            DyadicInference(bootstrap_iters=120, bootstrap_seed=19),
            alignment,
        )
    rows = {row["key"]: row for row in cast("list[dict[str, Any]]", summary["rows"])}
    keep = (i < len(roster)) & (j < len(roster))
    assert rows["etnic-roster"]["n_dyads"] == rows["etnic"]["n_dyads"] == keep.sum()
    assert rows["etnic-roster"]["bootstrap_nodes"] == rows["etnic"]["bootstrap_nodes"]
    effects, _ = symmetric_dyadic_effects(
        firms[i][keep].tolist(), firms[j][keep].tolist()
    )
    design = np.column_stack(
        [np.ones(keep.sum()), np.asarray(effects), sector[keep], etnic[keep], w2[keep]]
    )
    beta = np.linalg.lstsq(design, y[keep], rcond=None)[0]
    assert rows["etnic"]["coefficient"] == pytest.approx(beta[-1], abs=1e-10)
    control = rows["etnic"]["etnic_similarity"]
    assert control["coefficient"] == pytest.approx(beta[-2], abs=1e-10)
    assert control["ci_low"] <= control["coefficient"] <= control["ci_high"]
    assert cast("dict[str, Any]", summary["etnic"])["n_firms"] == len(roster)


def test_partially_covered_etnic_roster_is_rejected(tmp_path: Path) -> None:
    """A roster firm with a missing pair must fail, not shrink the sample."""
    firms = np.array(["A", "B", "C"])
    score = np.full((3, 3), 0.1)
    alignment = _etnic_alignment(tmp_path, firms, score)
    pairs = pd.read_parquet(alignment / "pairs.parquet")
    pairs.iloc[1:].to_parquet(alignment / "pairs.parquet", index=False)
    sample = DyadicSample[str](
        np.zeros(3),
        np.array(["A", "A", "B"]).tolist(),
        np.array(["B", "C", "C"]).tolist(),
    )
    with pytest.raises(DyadicInputError, match="every pair among its roster"):
        etnic_control(alignment, sample)

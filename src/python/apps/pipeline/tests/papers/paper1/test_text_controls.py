"""Independent numerical witnesses for the supplementary text controls."""

from __future__ import annotations

import json
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline.figures.paper1.numbers import Paper1NumberInputs, write_paper1_numbers
from pipeline.figures.paper1.text_controls import render_paper1_text_controls
from pipeline.stages.papers.paper1.text_controls import (
    run_text_controls,
    tfidf_pair_distances,
)
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

if TYPE_CHECKING:
    from pathlib import Path


def test_tfidf_uses_separate_capped_fits_and_exact_squared_cost_assignment() -> None:
    """Sparse lexical transport agrees with exhaustive permutation search."""
    texts = [
        ["apple apple banana", "orange market", "x"],
        ["banana orange", "stock stock market", "apple"],
        ["market stock", "orange orange", "banana stock"],
    ]
    tickers = ["A", "B", "C"]
    cap = 4
    firms = TfidfVectorizer(max_features=cap).fit_transform(
        ["\n\n".join(rows) for rows in texts]
    )
    documents = csr_matrix(
        TfidfVectorizer(max_features=cap).fit_transform(
            [text for rows in texts for text in rows]
        )
    ).toarray()
    pairs, metadata = tfidf_pair_distances(
        csr_matrix(firms),
        csr_matrix(documents),
        np.array([0, 3, 6, 9]),
        tickers,
        max_features=cap,
    )
    for left, right in combinations(range(len(tickers)), 2):
        observed = pairs.loc[
            (pairs["ticker_i"] == tickers[left]) & (pairs["ticker_j"] == tickers[right])
        ].set_index("arm")["distance"]
        a, b = (
            documents[left * 3 : (left + 1) * 3],
            documents[right * 3 : (right + 1) * 3],
        )
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        oracle = np.sqrt(
            min(
                sum(squared[index, match] for index, match in enumerate(order)) / len(a)
                for order in permutations(range(len(b)))
            )
        )
        assert observed["tfidf_document_w2"] == pytest.approx(oracle, abs=1e-12)
        assert observed["tfidf_firm_cosine"] == pytest.approx(
            cosine_distances(firms)[left, right], abs=1e-12
        )
    assert metadata["max_features"] == cap
    assert metadata["zero_article_rows"] == 1


@pytest.mark.parametrize("violation", ["firm_count", "boundaries", "nonfinite"])
def test_numeric_tfidf_rejects_a_changed_cohort_or_nonfinite_vectors(
    violation: str,
) -> None:
    """A numeric handoff cannot silently alter alignment or introduce NaNs."""
    firms = csr_matrix(np.eye(2))
    documents = csr_matrix(np.tile(np.eye(2), (2, 1)))
    boundaries = np.array([0, 2, 4])
    if violation == "firm_count":
        firms = csr_matrix(firms[:1])
    elif violation == "boundaries":
        boundaries = np.array([0, 2, 3])
    else:
        firms.data[0] = np.nan
    with pytest.raises(ValueError, match="cohort"):
        tfidf_pair_distances(firms, documents, boundaries, ["A", "B"], max_features=2)


def test_review_fits_common_panel_with_saved_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OLS matches an independent design and rescaling preserves paired draws."""
    monkeypatch.setattr(
        "pipeline.stages.papers.paper1.text_controls.entrypoint_code_identity",
        lambda _command: "0" * 64,
    )
    rng = np.random.default_rng(42)
    tickers = [f"F{i}" for i in range(8)]
    pairs = list(combinations(tickers, 2))
    x = rng.uniform(0.1, 0.8, len(pairs))
    fe = rng.normal(0, 0.03, len(tickers))
    lookup = dict(zip(tickers, fe, strict=True))
    y = (
        1
        + 0.3 * x
        + np.array([lookup[a] + lookup[b] for a, b in pairs])
        + rng.normal(0, 0.01, len(pairs))
    )
    frame = pd.DataFrame(pairs, columns=["ticker_i", "ticker_j"])
    frame = frame.assign(
        model="qwen3-embedding-8b", m=128, return_chord=y, wasserstein_w2=x
    )
    pair_file = tmp_path / "canonical.parquet"
    frame.to_parquet(pair_file, index=False)
    counts = rng.multinomial(
        len(tickers), np.full(len(tickers), 1 / len(tickers)), size=127
    )
    counts_file = tmp_path / "counts.parquet"
    pd.DataFrame(
        [
            {
                "draw_id": draw,
                "ticker": ticker,
                "multiplicity": int(counts[draw, index]),
            }
            for draw in range(len(counts))
            for index, ticker in enumerate(tickers)
        ]
    ).to_parquet(counts_file, index=False)
    control = tmp_path / "control"
    control.mkdir()
    frame[["ticker_i", "ticker_j"]].assign(
        arm="test_control", distance=2 * x
    ).to_parquet(control / "pairs.parquet", index=False)
    (control / "provenance.manifest.json").write_text("{}\n")
    result = run_text_controls(
        pairs_file=pair_file,
        node_counts_file=counts_file,
        control_dirs=[control],
        output_dir=tmp_path / "review",
    )
    # Independently constructed symmetric firm-effect design and OLS slope.
    design = np.column_stack(
        [
            np.ones(len(pairs)),
            x,
            *[
                np.array([int(a == ticker) + int(b == ticker) for a, b in pairs])
                for ticker in tickers[1:]
            ],
        ]
    )
    oracle = np.linalg.lstsq(design, y, rcond=None)[0][1]
    canonical, candidate = result["cells"]
    assert canonical["coefficient"] == pytest.approx(oracle, abs=1e-10)
    assert candidate["coefficient"] == pytest.approx(oracle / 2, abs=1e-10)
    assert canonical["effect_one_sd"] == pytest.approx(
        candidate["effect_one_sd"], abs=1e-10
    )
    draws = pd.read_parquet(tmp_path / "review/bootstrap_draws.parquet")
    wide = draws.pivot_table(
        index="draw_id", columns="arm", values="standardized_effect"
    )
    np.testing.assert_allclose(
        wide.iloc[:, 0], wide.iloc[:, 1], atol=1e-10, equal_nan=True
    )
    assert result["bootstrap_requested_draws"] == len(counts)
    assert (
        json.loads((tmp_path / "review/results.json").read_text())["publication_status"]
        == "supplementary"
    )


def test_publication_reports_standardized_effect_and_saved_fit(tmp_path: Path) -> None:
    """Published units and intervals come from the saved standardized estimates."""
    summary = {
        "return_window": "2018-2022",
        "n_firms": 100,
        "n_dyads": 4950,
        "bootstrap_requested_draws": 1999,
        "cells": [
            {
                "arm": "tfidf_document_w2",
                "coefficient": 1.3512,
                "effect_one_sd": 0.0540,
                "effect_one_sd_ci_low": 0.0399,
                "effect_one_sd_ci_high": 0.0704,
                "partial_r2": 0.1328,
                "geometry_spearman_vs_canonical_w2": 0.8018,
            }
        ],
    }
    results = tmp_path / "results.json"
    results.write_text(json.dumps(summary))
    render_paper1_text_controls(results, tmp_path)
    table = (tmp_path / "text_controls.tex").read_text()
    assert "0.054 [0.040, 0.070] & 0.133 & 0.802" in table
    numbers = tmp_path / "numbers.tex"
    write_paper1_numbers(numbers, Paper1NumberInputs(text_controls=summary))
    assert (
        r"\ppDeclareValue{text-controls-tfidf-document-partial-r2}{0.132800}"
        in numbers.read_text()
    )

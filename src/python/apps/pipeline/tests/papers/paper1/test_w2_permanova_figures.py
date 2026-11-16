"""Saved-result publication for the original-W2 sector diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pipeline.figures.paper1.numbers import (
    Paper1NumberInputs,
    Paper1NumberPaths,
    render_paper1_numbers,
    write_paper1_numbers,
)
from pipeline.figures.paper1.w2_permanova import (
    Paper1W2PermanovaRenderPaths,
    render_paper1_w2_permanova,
    render_w2_permanova_table,
)
from pipeline.io.w2_permanova_artifacts import write_w2_permanova_results

if TYPE_CHECKING:
    from pathlib import Path


def _summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "n_firms": 5,
        "n_groups": 3,
        "sector_sizes": {"A&B": 2, "B": 2, "Singleton": 1},
        "permanova": {
            "status": "computed",
            "ss_total": 20.0,
            "ss_within": 3.0,
            "ss_between": 17.0,
            "r2": 0.85,
            "pseudo_f": 5.666667,
            "pseudo_f_status": "finite",
            "pvalue": 0.03,
            "n_permutations": 99,
            "random_state": 42,
            "null_r2_expectation": 0.5,
            "null_r2_mean": 0.51,
            "excess_r2": 0.34,
        },
        "dispersion": {
            "status": "computed",
            "reason": None,
            "n_firms": 4,
            "n_groups": 2,
            "excluded_singleton_sectors": ["Singleton"],
            "statistic": 1.25,
            "statistic_status": "finite",
            "pvalue": 0.4,
            "n_permutations": 99,
        },
    }


def test_saved_sector_result_renders_both_analysis_scopes(tmp_path: Path) -> None:
    """Publish supplied values and disclose singleton exclusion only for spread."""
    result = tmp_path / "results.json"
    write_w2_permanova_results(tmp_path, _summary(), {"schedule": np.arange(4)})
    output = tmp_path / "generated"
    render_paper1_w2_permanova(Paper1W2PermanovaRenderPaths(result, output))
    table = (output / "w2_permanova.tex").read_text(encoding="utf-8")
    assert "3.000 & 17.000 & 20.000 & 0.850 & 5.667 & 0.0300" in table
    assert "0.500 & 0.510 & 0.340" in table
    assert "4 & 2 & 1.250 & 0.4000" in table
    assert "5 firms in 3 sectors" in table
    assert "excluded singleton sectors are Singleton" in table
    assert r"A\&B=2" in table
    assert "residual permutations" in table
    assert "requires exchangeability" in table


def test_undefined_companion_publishes_missing_value_and_reason(tmp_path: Path) -> None:
    """An unavailable companion must not turn into a zero or successful test."""
    summary = _summary()
    summary["dispersion"].update(
        status="not_computed",
        reason="material_negative_inertia",
        statistic=None,
        statistic_status="undefined",
        pvalue=None,
    )
    table = render_w2_permanova_table(summary)
    assert r"4 & 2 & \textemdash & \textemdash" in table
    assert "not computed: material negative inertia" in table
    path = tmp_path / "numbers.tex"
    write_paper1_numbers(path, Paper1NumberInputs(w2_permanova=summary))
    numbers = path.read_text(encoding="utf-8")
    assert r"\ppDeclareValue{w2-permanova-r2}{0.850000}" in numbers
    assert r"\ppDeclareValue{w2-permanova-null-r2}{0.500000}" in numbers
    assert r"\ppDeclareValue{w2-permanova-excess-r2}{0.340000}" in numbers
    assert r"\ppDeclareValue{w2-permanova-dispersion-p}{\ppmissing}" in numbers


def test_infinite_statistic_remains_distinct_from_uncomputed() -> None:
    """A zero denominator is shown explicitly rather than as a finite statistic."""
    summary = _summary()
    summary["permanova"].update(pseudo_f=None, pseudo_f_status="positive_infinity")
    rendered = render_w2_permanova_table(summary)
    assert r"0.850 & $\infty$ & 0.0300" in rendered


def _number_paths(tmp_path: Path) -> Paper1NumberPaths:
    absent = tmp_path / "absent"
    return Paper1NumberPaths(
        numbers_path=tmp_path / "numbers.tex",
        params_path=absent,
        corpus_summary=absent,
        dyadic_confound_results=absent,
        w2_permanova_results=tmp_path / "results.json",
        diagnostics_results=absent,
        sample_descriptives_results=absent,
        pcoa_summary=absent,
        neighbour_outcome_split_results=absent,
        covariance_matrix=absent,
        oos_covariance_matrix=None,
        oos_dyadic_results=None,
    )


@pytest.mark.parametrize("audit_state", ["missing", "tampered"])
def test_publication_rejects_unverifiable_permutations(
    tmp_path: Path, audit_state: str
) -> None:
    """Neither table nor number publication may consume an unverifiable audit."""
    write_w2_permanova_results(tmp_path, _summary(), {"schedule": np.arange(4)})
    audit_path = tmp_path / "permutation_audit.npz"
    if audit_state == "missing":
        audit_path.unlink()
    else:
        audit_path.write_bytes(b"another permutation archive")
    with pytest.raises(ValueError, match="audit"):
        render_paper1_w2_permanova(
            Paper1W2PermanovaRenderPaths(tmp_path / "results.json", tmp_path)
        )
    with pytest.raises(ValueError, match="audit"):
        render_paper1_numbers(_number_paths(tmp_path))
    assert not (tmp_path / "w2_permanova.tex").exists()
    assert not (tmp_path / "numbers.tex").exists()


def test_number_loader_accepts_verified_sector_results(tmp_path: Path) -> None:
    """The saved-result number path publishes after verifying the paired audit."""
    write_w2_permanova_results(tmp_path, _summary(), {"schedule": np.arange(4)})
    render_paper1_numbers(_number_paths(tmp_path))
    numbers = (tmp_path / "numbers.tex").read_text(encoding="utf-8")
    assert r"\ppDeclareValue{w2-permanova-p}{0.030000}" in numbers

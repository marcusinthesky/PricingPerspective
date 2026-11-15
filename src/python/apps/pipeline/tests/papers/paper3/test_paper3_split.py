"""Focused tests for the independently cacheable Paper 3 publication leaves."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pipeline.figures.paper3 import exhibits
from pipeline.figures.paper3.exhibits import (
    CERTIFICATE_EXHIBIT,
    ILLUSTRATION_EXHIBIT,
    render_paper3_certificate,
    render_paper3_illustrations,
)
from pipeline.figures.paper3.numbers import Paper3NumberPaths, render_paper3_numbers

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _result_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    folders = []
    for name in ("certificate", "certificate_validation", "portfolio_anatomy"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "results.json").write_text("{}\n", encoding="utf-8")
        folders.append(folder)
    return folders[0], folders[1], folders[2]


def test_numbers_leaf_writes_only_numbers(tmp_path: Path) -> None:
    """The numbers leaf writes numbers.tex and no provenance sidecar."""
    certificate, validation, anatomy = _result_dirs(tmp_path)
    output = tmp_path / "generated"
    output.mkdir()
    render_paper3_numbers(
        Paper3NumberPaths(
            numbers_path=output / "numbers.tex",
            certificate_dir=certificate,
            validation_dir=validation,
            anatomy_dir=anatomy,
        )
    )
    assert sorted(path.name for path in output.iterdir()) == ["numbers.tex"]
    assert "n-tickers" in (output / "numbers.tex").read_text(encoding="utf-8")


def test_certificate_render_reads_results_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The certificate render passes the parsed results.json to its figure."""
    certificate, _validation, _anatomy = _result_dirs(tmp_path)
    (certificate / "results.json").write_text(
        json.dumps({"news_only_benchmark": {"status": "x"}}), encoding="utf-8"
    )
    seen: list[object] = []

    def fake_figure(path: Path, summary: object) -> None:
        seen.append(summary)
        path.write_text("%% PGF\n", encoding="utf-8")

    monkeypatch.setattr(exhibits, "_fig_news_variance_benchmark", fake_figure)
    output = tmp_path / "images"
    render_paper3_certificate(certificate, output)
    assert seen == [{"news_only_benchmark": {"status": "x"}}]
    assert sorted(path.name for path in output.iterdir()) == [CERTIFICATE_EXHIBIT]


def test_illustrations_render_is_data_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The illustration leaf writes only the conceptual pipeline figure."""

    def fake_figure(path: Path) -> None:
        path.write_text("%% PGF\n", encoding="utf-8")

    monkeypatch.setattr(exhibits, "_fig_conceptual_pipeline", fake_figure)
    output = tmp_path / "images"
    render_paper3_illustrations(output)
    assert sorted(path.name for path in output.iterdir()) == [ILLUSTRATION_EXHIBIT]

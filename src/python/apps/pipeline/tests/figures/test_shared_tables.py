"""Tests for shared generated appendix tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pipeline.figures import shared_tables

if TYPE_CHECKING:
    from pathlib import Path

_HEADER_COUNT = 2


def test_sample_universe_table_uses_priced_artifact_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render sorted metadata for artifact items with return histories only."""
    calls: list[dict[str, object]] = []

    def fake_reader(
        _artifact_dir: Path, *, expected_identity: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append(expected_identity)
        return pd.DataFrame(), {"item_ids": ["WBA", "AAA", "BBB"]}

    monkeypatch.setattr(shared_tables, "read_typed_distance_artifact", fake_reader)
    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    (returns_dir / "AAA.parquet").write_bytes(b"")
    (returns_dir / "BBB.parquet").write_bytes(b"")
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text(
        "Symbol,Name,Sector,Industry\n"
        "AAA,A & B,Technology,Software\n"
        "BBB,B Corp,Energy,Utilities\n"
        "WBA,Walgreens,Health Care,Pharmacy\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "sample_universe.tex"
    rendered = shared_tables.render_sample_universe_table(
        universe_csv=universe_csv,
        distance_artifact_dir=tmp_path / "artifact",
        returns_dir=returns_dir,
        output_file=output_file,
        provider_id="provider",
        representation_id="representation",
    )

    assert calls == [
        {
            "provider_id": "provider",
            "representation_id": "representation",
            "distance_id": "wasserstein_w2",
        }
    ]
    assert output_file.read_text(encoding="utf-8") == rendered
    assert r"\begin{longtable}{lll}" in rendered
    assert rendered.count(r"Symbol & Name & Sector \\") == _HEADER_COUNT
    assert rendered.index("BBB & B Corp & Energy") < rendered.index(r"AAA & A \& B")
    assert "WBA" not in rendered
    assert "Industry" not in rendered
    assert r"\label{tab:sample-universe}" in rendered


def test_sample_universe_table_rejects_missing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not silently publish a roster row without metadata."""

    def fake_reader(
        _artifact_dir: Path, *, expected_identity: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        del expected_identity
        return pd.DataFrame(), {"item_ids": ["AAA", "MISSING"]}

    monkeypatch.setattr(shared_tables, "read_typed_distance_artifact", fake_reader)
    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    for symbol in ("AAA", "MISSING"):
        (returns_dir / f"{symbol}.parquet").write_bytes(b"")
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text(
        "Symbol,Name,Sector\nAAA,A,Technology\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no row for typed-distance items"):
        shared_tables.render_sample_universe_table(
            universe_csv=universe_csv,
            distance_artifact_dir=tmp_path / "artifact",
            returns_dir=returns_dir,
            output_file=tmp_path / "sample_universe.tex",
            provider_id="provider",
            representation_id="representation",
        )


def test_sample_universe_sector_list_groups_symbols_by_sector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compact layout lists every priced symbol once under its sector."""

    def fake_reader(
        _artifact_dir: Path, *, expected_identity: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        del expected_identity
        return pd.DataFrame(), {"item_ids": ["AAA", "BBB", "CCC"]}

    monkeypatch.setattr(shared_tables, "read_typed_distance_artifact", fake_reader)
    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    for symbol in ("AAA", "BBB", "CCC"):
        (returns_dir / f"{symbol}.parquet").write_bytes(b"")
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text(
        "Symbol,Name,Sector\n"
        "AAA,A & B,Technology\n"
        "BBB,B Corp,Energy\n"
        "CCC,C Corp,Technology\n",
        encoding="utf-8",
    )

    rendered = shared_tables.render_sample_universe_table(
        universe_csv=universe_csv,
        distance_artifact_dir=tmp_path / "artifact",
        returns_dir=returns_dir,
        output_file=tmp_path / "sample_universe.tex",
        provider_id="provider",
        representation_id="representation",
        layout="sector-list",
    )

    assert "longtable" not in rendered
    assert r"Energy & 1 & BBB \\" in rendered
    assert r"Technology & 2 & AAA, CCC \\" in rendered
    assert rendered.index("Energy") < rendered.index("Technology & 2")
    assert "A & B" not in rendered
    assert r"\label{tab:sample-universe}" in rendered

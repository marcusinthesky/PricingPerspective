"""Tests for narration parsing and deterministic media-path behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from manimize.narration import Narration, parse_narration

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_narration_ignores_intro_and_normalizes_paragraphs(
    tmp_path: Path,
) -> None:
    script = tmp_path / "NARRATION.md"
    script.write_text(
        "# Intro\n\nRead this.\n\n## 01 — First idea\n\n"
        "One sentence.\n\nTwo sentences.\n",
        encoding="utf-8",
    )

    assert parse_narration(script) == [
        Narration("01", "First idea", "One sentence. Two sentences.")
    ]


def test_parse_narration_rejects_duplicate_or_unordered_sections(
    tmp_path: Path,
) -> None:
    script = tmp_path / "NARRATION.md"
    script.write_text("## 02 — B\n\nText.\n## 01 — A\n\nText.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique and ordered"):
        parse_narration(script)


def test_narration_stem_is_stable() -> None:
    assert (
        Narration("01", "A firm as a probability law", "Text").stem
        == "01_firm_as_distribution"
    )

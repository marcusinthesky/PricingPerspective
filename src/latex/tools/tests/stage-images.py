#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Regression fixtures for reachable-image staging."""

from __future__ import annotations

import runpy
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
namespace = runpy.run_path(str(TOOLS_DIR / "stage-images"))
StageError = namespace["StageError"]
stage_reachable_images = namespace["stage_reachable_images"]

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    source = root / "source"
    stage = root / "stage"
    for base in (source, stage):
        (base / "chapters").mkdir(parents=True)
    (source / "images").mkdir()
    (source / "images" / "referenced.pdf").write_bytes(b"referenced")
    (source / "images" / "unreferenced.pdf").write_bytes(b"unreferenced")
    (source / "images" / "dead-branch.pdf").write_bytes(b"dead branch")
    (stage / "main.tex").write_text(r"\input{chapters/live}", encoding="utf-8")
    (stage / "chapters" / "live.tex").write_text(
        r"\includegraphics{images/referenced.pdf}", encoding="utf-8"
    )
    (stage / "chapters" / "dead.tex").write_text(
        r"\includegraphics{images/dead-branch.pdf}", encoding="utf-8"
    )

    staged = stage_reachable_images(source, stage, "main.tex")
    assert staged == [Path("images/referenced.pdf")]
    assert (stage / "images" / "referenced.pdf").read_bytes() == b"referenced"
    assert not (stage / "images" / "unreferenced.pdf").exists()
    assert not (stage / "images" / "dead-branch.pdf").exists()

    (stage / "chapters" / "live.tex").write_text(
        r"\includegraphics{images/missing.pdf}", encoding="utf-8"
    )
    try:
        stage_reachable_images(source, stage, "main.tex")
    except StageError as exc:
        assert "missing reachable LaTeX dependency" in str(exc)
        assert "images/missing.pdf" in str(exc)
    else:
        raise AssertionError("a missing reachable image must fail staging")

    pgf_source = root / "pgf-source"
    base_stage = root / "pgf-base-stage"
    venue_stage = root / "pgf-venue-stage"
    (pgf_source / "images").mkdir(parents=True)
    for target_stage in (base_stage, venue_stage):
        (target_stage / "chapters").mkdir(parents=True)
        (target_stage / "chapters" / "live.tex").write_text(
            r"\ppfigure{1}{referenced}", encoding="utf-8"
        )
    (base_stage / "main.tex").write_text(r"\input{chapters/live}", encoding="utf-8")
    (venue_stage / "main.tex").write_text(
        r"\renewcommand{\ppfiguresuffix}{-venue}\input{chapters/live}",
        encoding="utf-8",
    )
    (pgf_source / "images" / "referenced.pgf").write_text(
        r"\includegraphics{referenced-img0.png}", encoding="utf-8"
    )
    (pgf_source / "images" / "referenced-img0.png").write_bytes(b"base raster")
    (pgf_source / "images" / "referenced-venue.pgf").write_text(
        r"\includegraphics{referenced-venue-img0.png}", encoding="utf-8"
    )
    (pgf_source / "images" / "referenced-venue-img0.png").write_bytes(b"venue raster")

    base_staged = stage_reachable_images(pgf_source, base_stage, "main.tex")
    assert base_staged == [
        Path("images/referenced-img0.png"),
        Path("images/referenced.pgf"),
    ]
    assert not (base_stage / "images" / "referenced-venue.pgf").exists()

    venue_staged = stage_reachable_images(pgf_source, venue_stage, "main.tex")
    assert venue_staged == [
        Path("images/referenced-venue-img0.png"),
        Path("images/referenced-venue.pgf"),
    ]
    assert not (venue_stage / "images" / "referenced.pgf").exists()
    assert not (venue_stage / "images" / "referenced-img0.png").exists()

print("reachable-image staging fixtures passed")

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
    (pgf_source / "images").mkdir(parents=True)
    (base_stage / "chapters").mkdir(parents=True)
    (base_stage / "chapters/live.tex").write_text(
        r"\ppfigure{1}{referenced}", encoding="utf-8"
    )
    (base_stage / "main.tex").write_text(r"\input{chapters/live}", encoding="utf-8")
    (pgf_source / "images" / "referenced.pgf").write_text(
        r"\includegraphics{referenced-img0.png}", encoding="utf-8"
    )
    (pgf_source / "images" / "referenced-img0.png").write_bytes(b"base raster")
    base_staged = stage_reachable_images(pgf_source, base_stage, "main.tex")
    assert base_staged == [
        Path("images/referenced-img0.png"),
        Path("images/referenced.pgf"),
    ]
    (pgf_source / "images/referenced-img0.png").unlink()
    try:
        stage_reachable_images(pgf_source, base_stage, "main.tex")
    except StageError as error:
        assert "missing reachable LaTeX dependency" in str(error)
        assert "referenced-img0.png" in str(error)
    else:
        raise AssertionError("a missing PGF raster sidecar must fail staging")

print("reachable-image staging fixtures passed")

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Regression checks for compiler bytes and repeatable preview/release ownership."""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parent.parent
namespace = runpy.run_path(str(TOOLS_DIR / "bundle-audit"))
AuditError = namespace["AuditError"]
run = namespace["run"]


success = run(
    [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'success \\xf7\\n')",
    ],
    TOOLS_DIR,
)
assert success.returncode == 0
assert success.stdout == b"success \xf7\n"

try:
    run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'failure \\xf7\\n'); "
                "sys.stderr.buffer.write(b'compiler error \\xf7\\n'); "
                "raise SystemExit(7)"
            ),
        ],
        TOOLS_DIR,
    )
except AuditError as exc:
    message = str(exc)
    assert "command failed (7)" in message
    assert "failure \\xf7" in message
    assert "compiler error \\xf7" in message
else:
    raise AssertionError("nonzero subprocess must remain an audit failure")


# Metadata-bearing documents expose both wall-clock CreationDate drift and
# jobname-dependent PDF IDs. Exercise the actual shared compiler in distinct
# clean directories rather than comparing mocked command-line arguments.
bundle = str(TOOLS_DIR.parent / "bundle/tlextras-2022.0r0-pricing-perspective.zip")
document = (
    r"\documentclass[11pt]{article}\usepackage{hyperref}"
    r"\hypersetup{pdftitle={Repeatable publication}}"
    r"\begin{document}Original publication text.\end{document}"
)
main = namespace["main"]
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    stages = [root / name for name in ("release", "preview", "second-release")]
    for index, stage in enumerate(stages):
        stage.mkdir()
        source = stage / "main.tex"
        source.write_text(document, encoding="utf-8")
        os.utime(source, (100 + index, 100 + index))
    output = root / "published"
    archive = output / "arxiv.tar.gz"
    pdf = output / "arxiv.pdf"

    def arguments(stage: Path, *, preview: bool = False) -> list[str]:
        """Use the public release and preview interfaces with identical outputs."""
        result = [
            "bundle-audit",
            str(stage),
            "arxiv",
            str(archive),
            bundle,
            "--pdf",
            str(pdf),
        ]
        return result + (["--preview"] if preview else [])

    with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "1234567890"}):
        assert main(arguments(stages[0])) == 0
    certified_pdf, certified_archive = pdf.read_bytes(), archive.read_bytes()
    certified_manifest = (stages[0] / "MANIFEST.json").read_bytes()
    time.sleep(1.05)
    with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "987654321"}):
        assert main(arguments(stages[1], preview=True)) == 0
        assert pdf.read_bytes() == certified_pdf
        assert (output / "arxiv.log").is_file()
        assert archive.read_bytes() == certified_archive
        assert (stages[0] / "MANIFEST.json").read_bytes() == certified_manifest
        assert not (stages[1] / "MANIFEST.json").exists()
        assert main(arguments(stages[2])) == 0
    assert pdf.read_bytes() == certified_pdf
    assert archive.read_bytes() == certified_archive
    assert (stages[2] / "MANIFEST.json").read_bytes() == certified_manifest

    # A preview must actually compile edits, while preserving the last release.
    (stages[1] / "main.tex").write_text(
        document.replace("Original publication text.", "Changed publication text."),
        encoding="utf-8",
    )
    assert main(arguments(stages[1], preview=True)) == 0
    assert pdf.read_bytes() != certified_pdf
    assert archive.read_bytes() == certified_archive
    assert (stages[2] / "MANIFEST.json").read_bytes() == certified_manifest
    changed_pdf = pdf.read_bytes()
    changed_log = (output / "arxiv.log").read_bytes()
    (stages[1] / "main.tex").write_text(
        document.replace("Original publication text.", r"\undefinedPublicationFixture"),
        encoding="utf-8",
    )
    assert main(arguments(stages[1], preview=True)) == 1
    assert pdf.read_bytes() == changed_pdf
    assert (output / "arxiv.log").read_bytes() == changed_log
    assert archive.read_bytes() == certified_archive
    assert (stages[2] / "MANIFEST.json").read_bytes() == certified_manifest

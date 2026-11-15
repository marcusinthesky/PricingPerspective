#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Keep Paper 6 uploads free of unused generated research material."""

from __future__ import annotations

import json
import runpy
import tarfile
import tempfile
import time
from pathlib import Path

import tomllib

TOOLS = Path(__file__).resolve().parent.parent
audit = runpy.run_path(str(TOOLS / "bundle-audit"))
AuditError = audit["AuditError"]
prune = audit["prune_paper6_generated"]
management = runpy.run_path(str(TOOLS / "management-science"))
stage_sources = management["stage_sources"]
# Source staging is checked without invoking an image renderer or compiler.
compiler_run = stage_sources.__globals__["AUDIT"]["run"]
stage_sources.__globals__["AUDIT"]["run"] = lambda *args, **kwargs: None
USED_VALUES = (
    b"% Generated fixture; retained declaration bytes must not change.\n"
    b"\\ppDeclareValue{used}{1.234500} % preserve this exact line\n"
    b"\\ppDeclareValue{dependent}{\\ppvalue{source}}\n"
    b"\\ppDeclareValue{source}{9.0000}\n"
)
ALL_VALUES = (
    USED_VALUES
    + b"\\ppDeclareValue{historical}{999}\n"
    + b"\\ppDeclareValue{unused-parent}{\\ppvalue{unused-child}}\n"
    + b"\\ppDeclareValue{unused-child}{100}\n"
)
TABLE = b"\\ppnume[2]{dependent} \\ppvalue{used}\n"


def fixture(project: Path) -> None:
    """Create a paired document with retained and obsolete generated files."""
    source = project / "src"
    for directory in ("chapters", "generated", "targets/management-science"):
        (source / directory).mkdir(parents=True)
    (source / "metadata.tex").write_text("", encoding="utf-8")
    (source / "pp-manuscript.sty").write_text("", encoding="utf-8")
    (source / "refs.bib").write_text("", encoding="utf-8")
    (source / "body.tex").write_text(r"\input{chapters/main}", encoding="utf-8")
    (source / "appendix.tex").write_text(
        r"\input{generated/retained_table}", encoding="utf-8"
    )
    (source / "chapters/main.tex").write_text(
        r"\input{generated/numbers}\ppnum[4]{used}"
        r"% \input{generated/lean_status} \ppnum[2]{historical}",
        encoding="utf-8",
    )
    (source / "generated/numbers.tex").write_bytes(ALL_VALUES)
    (source / "generated/retained_table.tex").write_bytes(TABLE)
    for name in ("lean_status", "historical_table"):
        (source / f"generated/{name}.tex").write_text(name, encoding="utf-8")
    for name in ("management-science", "management-science-companion"):
        (source / f"targets/{name}.tex").write_text(
            r"\input{body}\input{appendix}", encoding="utf-8"
        )
    (source / "targets/management-science/combined.tex").write_text(
        r"\input{body}\input{appendix}", encoding="utf-8"
    )


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    paper6 = root / "06_information_motion"
    other = root / "01_other"
    for project in (paper6, other):
        fixture(project)
        stage = root / f"{project.name}-stage"
        stage_sources(project, stage)
        expected = USED_VALUES if project == paper6 else ALL_VALUES
        assert (stage / "generated/numbers.tex").read_bytes() == expected
        assert (stage / "generated/retained_table.tex").read_bytes() == TABLE
        assert (project / "src/generated/numbers.tex").read_bytes() == ALL_VALUES
        for name in ("lean_status", "historical_table"):
            assert (stage / f"generated/{name}.tex").exists() == (project == other)
            assert (project / f"src/generated/{name}.tex").is_file()
        notice = (stage / "REPRODUCTION-NOTICE.txt").read_text()
        assert ("Lean" in notice) == (project == other)

    # The same helper serves ordinary arXiv/QF source staging.
    stage = root / "arxiv-stage"
    fixture(stage)
    source = stage / "src"
    (source / "main.tex").write_text(r"\input{body}\input{appendix}", encoding="utf-8")
    assert prune(source, "main.tex") == [
        "generated/historical_table.tex",
        "generated/lean_status.tex",
    ]
    assert (source / "generated/numbers.tex").read_bytes() == USED_VALUES
    assert prune(source, "main.tex") == []
    (source / "main.tex").write_text(r"\input{generated/missing}", encoding="utf-8")
    try:
        prune(source, "main.tex")
    except AuditError as error:
        assert "missing reachable LaTeX input" in str(error)
    else:
        raise AssertionError("missing published inputs must fail staging")

    # Unsupported declarations fail before any generated file is changed.
    bad = source / "generated/bad.tex"
    unsupported = b"\\ppDeclareValue{broken}\n{1}\n"
    bad.write_bytes(unsupported)
    untouched = source / "generated/unread.tex"
    untouched.write_bytes(b"unread")
    (source / "main.tex").write_text(r"\input{generated/bad}", encoding="utf-8")
    try:
        prune(source, "main.tex")
    except AuditError as error:
        assert "unsupported value declaration" in str(error)
    else:
        raise AssertionError("unsupported declarations must not be silently pruned")
    assert bad.read_bytes() == unsupported
    assert untouched.read_bytes() == b"unread"

    # Governed multiline disclosures are preserved, including their scalar dependencies.
    status = source / "generated/lean_status.tex"
    governed = (
        b"% Generated formal disclosure; preserve every byte.\n"
        b"\\ppDeclareValue{lean-disclosure-prop:example}{Checked under\n"
        b"  the stated assumptions; benchmark \\ppvalue{historical}.}\n"
    )
    status.write_bytes(governed)
    (source / "generated/numbers.tex").write_bytes(ALL_VALUES)
    (source / "main.tex").write_text(
        r"\input{generated/numbers}\input{generated/lean_status}\ppvalue{used}",
        encoding="utf-8",
    )
    prune(source, "main.tex")
    assert status.read_bytes() == governed
    assert (source / "generated/numbers.tex").read_bytes() == (
        USED_VALUES.splitlines(keepends=True)[0]
        + USED_VALUES.splitlines(keepends=True)[1]
        + b"\\ppDeclareValue{historical}{999}\n"
    )
    assert prune(source, "main.tex") == []

# The paired builder uses independently staged temporary roots. Neither their
# names nor the compile time may change the tracked reviewer-facing PDF bytes.
management["compile_document"].__globals__["AUDIT"]["run"] = compiler_run
project = TOOLS.parent / "projects/06_information_motion"
config = tomllib.loads((project / "Tectonic.toml").read_text())
bundle = str((project / config["doc"]["bundle"]).resolve())
with tempfile.TemporaryDirectory() as temporary:
    pdfs = []
    for name in ("first-clean-room", "independent-clean-room"):
        stage = Path(temporary) / name
        (stage / "targets/management-science").mkdir(parents=True)
        document = (
            "\\documentclass[12pt]{article}\n"
            "\\begin{document}Independent source-closure compilation.\\end{document}\n"
        )
        for path in (
            "main.tex",
            "companion.tex",
            "catalog.tex",
            "targets/management-science.tex",
            "targets/management-science-companion.tex",
            "targets/management-science/combined.tex",
        ):
            (stage / path).write_text(document, encoding="utf-8")
        management["compile_document"](stage, "main.tex", bundle)
        pdfs.append((stage / "main.pdf").read_bytes())
        if len(pdfs) == 1:
            time.sleep(1.1)  # Cross the PDF CreationDate's one-second resolution.
    assert pdfs[0] == pdfs[1], "clean-room paths or compile dates changed PDF bytes"

# Interactive validation still unpacks its temporary archive, but only a release
# may replace the certified source archive alongside the reviewer-facing PDFs.
with tempfile.TemporaryDirectory() as temporary:
    project = Path(temporary) / "06_information_motion"
    fixture(project)
    source = project / "src"
    (source / "body.tex").write_text(
        r"\refstepcounter{section}Main\label{sec:main}", encoding="utf-8"
    )
    (source / "appendix.tex").write_text(
        r"\refstepcounter{section}Companion\label{sec:companion}", encoding="utf-8"
    )
    preamble = (
        "\\documentclass[12pt]{article}\n"
        "\\newcommand{\\ppMSBoundary}{}\n\\begin{document}\n"
    )
    main = preamble + "\\input{body}\n\\end{document}\n"
    companion = (
        preamble
        + "\\input{generated/management_science_counters}\n"
        + "\\input{appendix}\n\\end{document}\n"
    )
    combined = (
        preamble
        + "\\input{body}\\clearpage\n"
        + "\\makeatletter\\immediate\\write\\@auxout{\\string\\ppMSBoundary}"
        + "\\makeatother\n"
        + "\\newwrite\\handoff\\immediate\\openout\\handoff=ms-counter-state.tex\n"
        + "\\immediate\\write\\handoff{\\string\\setcounter{section}{0}}\n"
        + "\\immediate\\closeout\\handoff\n"
        + "\\setcounter{section}{0}\\input{appendix}\n\\end{document}\n"
    )
    for name, content in (
        ("targets/management-science.tex", main),
        ("targets/management-science-companion.tex", companion),
        ("targets/management-science/combined.tex", combined),
    ):
        (source / name).write_text(content, encoding="utf-8")
    management["build"](project, bundle, release=True)
    archive = project / "build/management-science/management-science.tar.gz"
    certified = archive.read_bytes()
    with tarfile.open(archive) as uploaded:
        manifest = json.load(uploaded.extractfile("MANIFEST.json"))
    assert "interactive_only" not in manifest
    management["build"](project, bundle, release=False)
    assert archive.read_bytes() == certified, (
        "interactive build replaced certified archive"
    )
    (source / "body.tex").write_text(
        r"\refstepcounter{section}Updated main\label{sec:main}", encoding="utf-8"
    )
    management["build"](project, bundle, release=True)
    assert archive.read_bytes() != certified, "bundle did not publish changed sources"

print("PASS: Paper 6 prunes unread generated files and scalars; retained bytes agree")
print("PASS: paired compilation produces byte-identical PDFs in independent roots")
print("PASS: builds preserve certified archives; bundles publish changed sources")

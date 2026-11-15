"""``_save`` accepts PGF and nothing else.

The repository ships figures as PGF so Tectonic typesets their text in the
document's font (t38). ``.pdf`` used to be allowed alongside it for diagnostics
"read by opening them", but that stopped working the moment t38 made PGF the
*global* backend in ``7b87d1ad``: under PGF, ``savefig(format="pdf")`` does not
use matplotlib's pure-Python PDF writer. It emits a standalone wrapper and shells
out to ``pdflatex``, and ``backend_pgf.py:848-850`` hard-codes ``hyperref`` and
``geometry`` into that wrapper — neither of which the dev shell's ``pgf-metrics``
closure carries.

The result was a stage that died several hundred log lines into a ``dvc repro``
with ``! LaTeX Error: File `hyperref.sty' not found.``. These tests pin the
format restriction that turns that into one sentence at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
from pipeline.figures._common import _FORMATS, _save, _style

if TYPE_CHECKING:
    from pathlib import Path


def test_pgf_is_the_only_supported_format() -> None:
    """The format table is PGF-only.

    Asserted on the table rather than through a save so the intent is legible:
    adding a format here is a deliberate act, and anything added must actually be
    producible by the *global* backend, which is the property ``.pdf`` silently
    lacked.
    """
    assert set(_FORMATS) == {".pgf"}


@pytest.mark.parametrize("suffix", [".pdf", ".png", ".svg", ".eps", ""])
def test_save_refuses_any_other_suffix(tmp_path: Path, suffix: str) -> None:
    """A non-PGF suffix fails at the call site, not inside pdflatex.

    ``.pdf`` is the case that matters — it is the one that used to be accepted
    and then failed deep in a LaTeX run — but the guard is on the whole table.

    Args:
        tmp_path: Pytest temporary directory.
        suffix: Candidate figure suffix.

    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        with pytest.raises(ValueError, match="unsupported figure format"):
            _save(fig, tmp_path / f"figure{suffix}")
    finally:
        plt.close(fig)


def test_the_rejection_says_how_to_view_a_figure(tmp_path: Path) -> None:
    """The error names the preview recipe.

    Dropping ``.pdf`` removed the only way to eyeball a figure without a LaTeX
    run, so the message has to carry its replacement or the restriction just
    looks like an obstruction.

    Args:
        tmp_path: Pytest temporary directory.

    """
    fig, _ = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"just latex::preview") as excinfo:
            _save(fig, tmp_path / "figure.pdf")
    finally:
        plt.close(fig)
    assert ".pgf" in str(excinfo.value)


def test_pgf_still_writes(tmp_path: Path) -> None:
    """The supported format round-trips, so the guard is not vacuous.

    Without this, a table narrowed to ``{}`` would pass every assertion above.

    ``_style()`` first because that is what sets ``pgf.texsystem`` to ``pdflatex``;
    matplotlib's default is ``xelatex``, which ``pgf-metrics`` does not ship. Every
    production figure reaches ``_save`` through ``figure()``, which calls
    ``_style()`` — so calling it here mirrors real usage rather than working
    around a defect.

    Args:
        tmp_path: Pytest temporary directory.

    """
    _style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "figure.pgf"
    _save(fig, out)
    assert out.is_file()
    assert out.read_text().lstrip().startswith("%% Creator: Matplotlib")

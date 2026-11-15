r"""Print-size guard: a figure must be authored at the width it is printed at (t39).

`\includegraphics[width=0.8\linewidth]` scales the figure file to 0.8 of the text
block. If the file is wider than that, every label in it is scaled *down* by the same
factor, so an 8pt tick label can print at 3.7pt against 11pt body copy — which is
what this repo shipped before t39, across a 4.3-12.6pt range.

Nothing else catches it. The figure renders, every numerical test passes, and the
defect only exists on the printed page, so it is invisible to anything that does not
compare the two sides.

The measured quantity is the **saved PGF width**, not the authored `figsize`. Those
differ: `_common._save` writes with ``bbox_inches="tight"``, so the file on disk is
the trimmed content box, which for a schematic with dead canvas was as little as 63%
of the figure it was authored at. `_common.fit_to_width` exists to close that gap and
this test is what proves it stayed closed.

Reading the width straight out of the PGF picture rectangle keeps the test
dependency-free.
"""

import re
from pathlib import Path

import pytest
from pipeline.figures._common import TEXTWIDTH_IN

_PROJECTS = Path(__file__).resolve().parents[5] / "latex" / "projects"

# Every figure is included as `\ppfigure{<frac>}{<name>}`. Validate every invocation
# against this interface, then check the saved asset's actual printed width.
#
# `\ppfigure`'s first argument sets nothing at compile time -- PGF is placed at
# natural size. It exists so this test has *two independent* operands: the intended
# width from the manuscript and the actual width from the renderer. Take the
# fraction from Python instead and the guard would only compare the renderer against
# itself, and could no longer catch a figure authored for a width the manuscript
# does not use.
_INCLUDE = re.compile(r"\\ppfigure\{([0-9.]*)\}\{([A-Za-z0-9_\-]+)\}")

# A `.pgf` states its own natural size in the first line of the picture, in inches:
#     \pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{4.663000in}{2.303000in}}
# which is the same quantity `/MediaBox` carried for the PDFs this replaced.
_PGF_SIZE = re.compile(
    r"\\pgfpathrectangle\{\\pgfpointorigin\}\{\\pgfqpoint\{([0-9.]+)in\}"
)

# `bbox_inches="tight"` adds `pad_inches=0.02` per side and rounds, so an exactly
# fitted figure lands a few tenths of a point off. The band is far tighter than the
# 0.46-1.31 spread t39 started from, and still catches a figure whose widest artist
# is point-valued text that `fit_to_width` cannot move.
_MIN_RATIO = 0.97
_MAX_RATIO = 1.03


def _included() -> list[tuple[str, float, float, Path]]:
    """Return (name, fraction, measure, pgf_path) for every shared figure."""
    found: dict[str, tuple[str, float, Path]] = {}
    for tex in sorted(_PROJECTS.glob("*/src/**/*.tex")):
        text = re.sub(r"(?<!\\)%[^\n]*", "", tex.read_text(encoding="utf-8"))
        for invocation in re.finditer(r"\\ppfigure\b", text):
            include = _INCLUDE.match(text, invocation.start())
            assert include is not None, f"unsupported figure inclusion in {tex}"
            frac_text, name = include.groups()
            images = tex.parent
            while images != _PROJECTS and not (images / "images").is_dir():
                images = images.parent
            found[name] = (
                name,
                float(frac_text) if frac_text else 1.0,
                images / "images" / f"{name}.pgf",
            )
    return [
        (name, frac, TEXTWIDTH_IN, path) for name, frac, path in sorted(found.values())
    ]


def _pgf_width_in(path: Path) -> float:
    match = _PGF_SIZE.search(path.read_text(encoding="utf-8"))
    if match is None:
        message = f"no pgfpicture rectangle in {path}"
        raise AssertionError(message)
    return float(match.group(1))


_FIGURES = _included()


def _assert_authored_width(
    name: str,
    frac: float,
    measure: float,
    pgf: Path,
) -> None:
    """Compare one saved PGF with the measure recorded by its manuscript call."""
    assert pgf.exists(), f"{name} is included but {pgf} is missing"
    target = frac * measure
    width = _pgf_width_in(pgf)
    ratio = width / target
    assert _MIN_RATIO <= ratio <= _MAX_RATIO, (
        f"{name} is {width:.2f}in but prints at "
        f"{frac:g}*textwidth = {target:.2f}in (ratio {ratio:.3f}). PGF cannot be "
        f"rescaled, so it will overrun or under-fill the text block by that factor. "
        f"Author it with figsize({frac:g}, aspect) and, if it is a schematic or has "
        f"artists outside the axes, call fit_to_width(fig, {frac:g}) before saving."
    )


@pytest.mark.parametrize(
    ("name", "frac", "measure", "pgf"),
    _FIGURES,
    ids=[figure[0] for figure in _FIGURES],
)
def test_figure_is_authored_at_its_printed_width(
    name: str,
    frac: float,
    measure: float,
    pgf: Path,
) -> None:
    r"""A shared `.pgf` is authored at the width recorded by its manuscript."""
    _assert_authored_width(name, frac, measure, pgf)

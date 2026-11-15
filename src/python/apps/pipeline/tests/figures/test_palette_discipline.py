"""Keep figure data colours on the shared publication palettes."""

import ast
import re
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from pipeline.figures._common import _C, _TINT
from pipeline.figures.paper3.portfolio_anatomy import _sector_colours

_FIGURES = Path(__file__).resolve().parents[2] / "src" / "pipeline" / "figures"
_HEX = re.compile(r"#[0-9A-Fa-f]{3,8}")
_NAMED = re.compile(
    r"red|blue|green|orange|purple|cyan|magenta|yellow|lime|pink|brown|navy|teal|"
    r"olive|maroon|salmon|violet|gold|silver|light[a-z]+|dark[a-z]+|C\d+",
    re.IGNORECASE,
)
_COLOUR_KEYWORDS = {"color", "colors", "facecolor", "edgecolor", "ec", "fc", "c"}


def _inline_colours(source: str) -> list[str]:
    """Find colour literals independently of quoting, comments and annotations."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _HEX.fullmatch(node.value)
        ):
            found.append(f"{node.lineno}: {node.value}")
        if isinstance(node, ast.keyword) and node.arg in _COLOUR_KEYWORDS:
            value = node.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and _NAMED.fullmatch(value.value)
            ):
                found.append(f"{node.lineno}: {value.value}")
    return found


def _figure_modules() -> list[Path]:
    return sorted(p for p in _FIGURES.rglob("*.py") if p != _FIGURES / "_common.py")


def test_figure_modules_were_found() -> None:
    """A broken discovery path must not silently disable the palette guard."""
    assert _figure_modules()


@pytest.mark.parametrize(
    "module", _figure_modules(), ids=lambda p: str(p.relative_to(_FIGURES))
)
def test_no_inline_data_colours(module: Path) -> None:
    """Data colours use shared names; backgrounds and colormaps remain valid."""
    assert not _inline_colours(module.read_text(encoding="utf-8")), (
        f"{module}: use _C/_TINT from pipeline.figures._common"
    )


@pytest.mark.parametrize(
    ("source", "violates"),
    [
        ("colour = '#FF0000'", True),
        ('colour: str = "#FF0000"', True),
        ("ax.plot(x, color='red')", True),
        ('ax.plot(x, color="C0")', True),
        ('# colour = "#FF0000"', False),
        ('"""Example: color="red"."""', False),
        ("ax.plot(x, label='red', color=_C['blue'])", False),
        ("ax.plot(x, color='0.35')", False),
        ("ax.bar(x, facecolor='white', edgecolor='none')", False),
        ("ax.imshow(x, cmap='viridis')", False),
    ],
)
def test_literal_guard_accepts_formatting_and_rejects_data_colours(
    source: str, *, violates: bool
) -> None:
    """Formatting changes cannot hide a forbidden colour or create a false positive."""
    assert bool(_inline_colours(source)) is violates


def test_palettes_retain_semantic_colours() -> None:
    """Validate palette values rather than assignment spelling or literal counts."""
    assert {"blue", "orange", "black", "gray"} <= _C.keys()
    assert {"blue", "orange", "neutral"} <= _TINT.keys()
    assert to_rgba(_C["blue"]) == to_rgba("#0072B2")
    assert to_rgba(_C["orange"]) == to_rgba("#E69F00")
    assert to_rgba(_TINT["blue"]) != to_rgba(_TINT["orange"])
    for value in [*_C.values(), *_TINT.values()]:
        assert to_rgba(value)[-1] == 1


def test_sector_colours_reach_bar_artists() -> None:
    """A production colour selector supplies the expected colours to plotted bars."""
    sectors = pd.DataFrame({"sector": ["Technology", "Healthcare"]})
    colours = _sector_colours(sectors)
    fig = Figure()
    bars = fig.subplots().bar([0, 1], [1, 2], color=list(colours.values()))
    assert [bar.get_facecolor() for bar in bars] == [
        to_rgba(_C["blue"]),
        to_rgba(_C["orange"]),
    ]

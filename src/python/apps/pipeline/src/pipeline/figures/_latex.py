"""Shared LaTeX text-formatting primitives."""

from __future__ import annotations

_TEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def escape_latex(text: str) -> str:
    """Escape characters with special meaning in LaTeX text cells."""
    return "".join(_TEX_SPECIALS.get(character, character) for character in text)


__all__ = ["escape_latex"]

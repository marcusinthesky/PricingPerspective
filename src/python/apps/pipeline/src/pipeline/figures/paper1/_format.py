"""LaTeX escaping and fixed-precision formatting shared by Paper 1 exhibits."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pipeline.io.values import MISSING

if TYPE_CHECKING:
    from pipeline.io.values import PublicationScalar

EM_DASH = r"\textemdash"

__all__ = ["EM_DASH", "fmt", "fmt_int", "opt_float"]


def fmt(value: object, precision: int) -> str:
    """Format a possibly-``None`` numeric value to fixed ``precision``.

    decimals, or an em dash when the upstream statistic is undefined
    (``None``) rather than rounding a fabricated zero.
    """
    if value is None:
        return EM_DASH
    if not isinstance(value, str | bytes | int | float):
        message = f"expected a numeric scalar, received {type(value).__name__}"
        raise TypeError(message)
    number = float(value)
    if abs(number) < 0.5 * 10**-precision:
        number = 0.0
    return f"{number:.{precision}f}"


def fmt_int(value: object) -> str:
    """Format a possibly-``None`` integer, or an em dash when undefined."""
    if value is None:
        return EM_DASH
    if not isinstance(value, str | bytes | int | float):
        message = f"expected an integer scalar, received {type(value).__name__}"
        raise TypeError(message)
    return str(int(value))


def opt_float(value: object) -> PublicationScalar:
    r"""Coerce an optional numeric summary field to a publication scalar.

    ``None`` (e.g. a JSON ``null`` for a singleton sector with no within-sector
    pairs) degrades to the :data:`MISSING` sentinel, rendered ``\\ppmissing``
    in the manuscript, rather than a fabricated ``0.0``.
    """
    if value is None:
        return MISSING
    return float(cast("str | bytes | int | float", value))

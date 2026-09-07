"""LaTeX escaping and fixed-precision number formatting primitives."""

from __future__ import annotations

from pipeline.figures._latex import escape_latex as _escape

_EM_DASH = r"\textemdash"


def _fmt(value: object, precision: int) -> str:
    """Format a possibly-``None`` numeric value to fixed ``precision``.

    decimals, or an em dash when the upstream statistic is undefined
    (``None``) rather than rounding a fabricated zero.
    """
    if value is None:
        return _EM_DASH
    if not isinstance(value, str | bytes | int | float):
        message = f"expected a numeric scalar, received {type(value).__name__}"
        raise TypeError(message)
    number = float(value)
    if abs(number) < 0.5 * 10**-precision:
        number = 0.0
    return f"{number:.{precision}f}"


def _fmt_int(value: object) -> str:
    if value is None:
        return _EM_DASH
    if not isinstance(value, str | bytes | int | float):
        message = f"expected an integer scalar, received {type(value).__name__}"
        raise TypeError(message)
    return str(int(value))


def _sector_row(sector: str, row: dict[str, object]) -> str:
    cells = [
        _escape(sector),
        _fmt_int(row["n_firms"]),
        _fmt(row["mean_articles_per_firm"], 1),
        _fmt(row.get("mean_daily_return_volatility"), 4),
        _fmt(row.get("mean_within_sector_w2_distance"), 3),
        _fmt(row.get("mean_cross_sector_w2_distance"), 3),
    ]
    return " & ".join(cells) + r" \\"

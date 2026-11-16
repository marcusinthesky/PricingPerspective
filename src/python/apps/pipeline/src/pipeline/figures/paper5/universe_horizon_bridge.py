"""Render Paper 5's sequential horizon-and-universe bridge table."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

_REFERENCE_LONG_CELL = "reference_52_long"
_REFERENCE_SHORT_CELL = "reference_52_short"
_CURRENT_SHORT_CELL = "current_100_short"
_HORIZON_CONTRAST = "horizon_long_minus_short_at_52"
_UNIVERSE_CONTRAST = "universe_100_minus_52_at_short"


def _mapping(value: object, context: str) -> dict[str, object]:
    """Narrow a decoded JSON value to an object with string keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{context} must be a JSON object with string keys"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _number(mapping: dict[str, object], key: str, context: str) -> float:
    """Read one finite numeric field used in the rendered table."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"{context} requires numeric field {key!r}"
        raise TypeError(message)
    return float(value)


def _iso_date(cell: dict[str, object], key: str, cell_id: str) -> date:
    """Read one ISO calendar date used in the rendered table."""
    value = cell.get(key)
    if not isinstance(value, str):
        message = f"{cell_id} requires ISO date field {key!r}"
        raise TypeError(message)
    return date.fromisoformat(value)


def _short_date(value: date) -> str:
    """Format a date compactly enough to sit on one table line."""
    return f"{value.day} {value:%b} {value.year}"


def _cell_row(
    label: str,
    cell: dict[str, object],
) -> str:
    """Render one observed sequential specification."""
    cell_id = str(cell.get("cell_id", label))
    if cell.get("status") != "observed":
        message = f"{cell_id} must be an observed bridge specification"
        raise ValueError(message)
    universe_size = cell.get("universe_size")
    trading_days = cell.get("trading_days")
    if isinstance(universe_size, bool) or not isinstance(universe_size, int):
        message = f"{cell_id} universe_size must be an integer"
        raise TypeError(message)
    if isinstance(trading_days, bool) or not isinstance(trading_days, int):
        message = f"{cell_id} trading_days must be an integer"
        raise TypeError(message)
    end = _iso_date(cell, "window_end", cell_id)
    rho = _number(cell, "rho_hat", cell_id)
    rho_lower = _number(cell, "rho_ci_lower", cell_id)
    rho_upper = _number(cell, "rho_ci_upper", cell_id)
    return (
        " & ".join(
            [
                label,
                str(universe_size),
                _short_date(end),
                str(trading_days),
                f"{rho:.3f} [{rho_lower:.3f}, {rho_upper:.3f}]",
            ]
        )
        + r" \\"
    )


def _require_descriptive(contrast: dict[str, object], label: str) -> None:
    """Reject step contrasts that claim more than an arithmetic difference."""
    contrast_id = str(contrast.get("contrast_id", label))
    if contrast.get("status") != "descriptive_only":
        message = f"{contrast_id} must be marked descriptive_only"
        raise ValueError(message)
    if contrast.get("inference") != "none":
        message = f"{contrast_id} must not claim contrast inference"
        raise ValueError(message)


def _common_start(cells: list[dict[str, object]]) -> date:
    """Return the start date shared by every specification, or fail."""
    starts = {_iso_date(cell, "window_start", "bridge cell") for cell in cells}
    if len(starts) != 1:
        message = "bridge specifications must share one window_start"
        raise ValueError(message)
    return starts.pop()


def render_paper5_universe_horizon_bridge_table(
    summary_path: Path,
    output_path: Path,
) -> None:
    """Render one booktabs table for the three-step descriptive bridge."""
    raw: object = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = _mapping(raw, "Paper 5 universe-and-horizon bridge summary")
    if payload.get("schema_version") != 1:
        message = "Paper 5 universe-and-horizon bridge requires schema_version 1"
        raise ValueError(message)
    design = _mapping(payload.get("design"), "bridge design")
    expected_order = [
        _REFERENCE_LONG_CELL,
        _REFERENCE_SHORT_CELL,
        _CURRENT_SHORT_CELL,
    ]
    if design.get("specification_order") != expected_order:
        message = "bridge design has an unexpected sequential specification order"
        raise ValueError(message)
    scope = _mapping(payload.get("scope"), "bridge scope")
    if scope.get("interpretation") != "descriptive_noncausal":
        message = "bridge scope must be descriptive_noncausal"
        raise ValueError(message)
    cells = _mapping(payload.get("cells"), "bridge cells")
    contrasts = _mapping(payload.get("contrasts"), "bridge contrasts")
    reference_long = _mapping(
        cells.get(_REFERENCE_LONG_CELL),
        _REFERENCE_LONG_CELL,
    )
    reference_short = _mapping(
        cells.get(_REFERENCE_SHORT_CELL),
        _REFERENCE_SHORT_CELL,
    )
    current_short = _mapping(
        cells.get(_CURRENT_SHORT_CELL),
        _CURRENT_SHORT_CELL,
    )
    horizon_contrast = _mapping(
        contrasts.get(_HORIZON_CONTRAST),
        _HORIZON_CONTRAST,
    )
    universe_contrast = _mapping(
        contrasts.get(_UNIVERSE_CONTRAST),
        _UNIVERSE_CONTRAST,
    )
    _require_descriptive(horizon_contrast, _HORIZON_CONTRAST)
    _require_descriptive(universe_contrast, _UNIVERSE_CONTRAST)
    start = _common_start([reference_long, reference_short, current_short])
    rows = [
        _cell_row("Historical reference", reference_long),
        _cell_row("Horizon-matched", reference_short),
        _cell_row("Current universe", current_short),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrrl}",
                r"\toprule",
                (
                    r"Specification & Firms & End date & Trading days & "
                    r"$\hat\rho$ [95\% CI] \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Sequential universe-and-horizon bridge for the "
                    r"barycentric $W^\flat$ single-field SAR QMLE in the "
                    r"raw-return, common-intercept specification "
                    r"(\ppAppendixRef{sec:app-unadjusted}); it is not the primary "
                    r"factor-conditioned estimate. Every row starts on "
                    rf"{start.day} {start:%B} {start.year}. "
                    r"The historical row uses the recovered 52-firm operator and "
                    r"the long return window. The horizon-matched row holds that "
                    r"operator and universe fixed while shortening the return "
                    r"endpoint; its interval is newly estimated with the "
                    r"joint-date stationary bootstrap. The current-universe row "
                    r"expands to the 100-firm operator at the same endpoint, so "
                    r"it changes both the roster and the operator; the first and "
                    r"third intervals come from the reported source results. "
                    r"Complete-case trading-day counts can differ across "
                    r"universes.}"
                ),
                r"\label{tab:p5-universe-horizon-bridge}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = ["render_paper5_universe_horizon_bridge_table"]

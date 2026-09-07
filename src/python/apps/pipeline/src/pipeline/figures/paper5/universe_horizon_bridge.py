"""Render Paper 5's sequential horizon-and-universe bridge table."""

from __future__ import annotations

import json
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
    horizon = str(cell.get("horizon", ""))
    window = f"{cell.get('window_start')}--{cell.get('window_end')}"
    rho = _number(cell, "rho_hat", cell_id)
    rho_lower = _number(cell, "rho_ci_lower", cell_id)
    rho_upper = _number(cell, "rho_ci_upper", cell_id)
    intensity = _number(cell, "lambda_hat", cell_id)
    intensity_lower = _number(cell, "lambda_ci_lower", cell_id)
    intensity_upper = _number(cell, "lambda_ci_upper", cell_id)
    return (
        " & ".join(
            [
                label,
                str(universe_size),
                f"{horizon.title()} ({window})",
                str(trading_days),
                f"{rho:.3f} [{rho_lower:.3f}, {rho_upper:.3f}]",
                f"{intensity:.2f} [{intensity_lower:.2f}, {intensity_upper:.2f}]",
            ]
        )
        + r" \\"
    )


def _contrast_row(
    label: str,
    held_fixed: str,
    comparison: str,
    contrast: dict[str, object],
) -> str:
    """Render one explicitly descriptive step contrast."""
    contrast_id = str(contrast.get("contrast_id", label))
    if contrast.get("status") != "descriptive_only":
        message = f"{contrast_id} must be marked descriptive_only"
        raise ValueError(message)
    if contrast.get("inference") != "none":
        message = f"{contrast_id} must not claim contrast inference"
        raise ValueError(message)
    return (
        " & ".join(
            [
                label,
                held_fixed,
                comparison,
                r"\textemdash",
                f"{_number(contrast, 'rho_difference', contrast_id):.3f}",
                f"{_number(contrast, 'lambda_difference', contrast_id):.3f}",
            ]
        )
        + r" \\"
    )


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
    rows = [
        r"\multicolumn{6}{l}{\textit{Panel A: sequential specifications}} \\",
        r"\addlinespace[1.5pt]",
        _cell_row("Historical reference", reference_long),
        _cell_row("Horizon-matched", reference_short),
        _cell_row("Current governed", current_short),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel B: descriptive step contrasts}} \\",
        r"\addlinespace[1.5pt]",
        _contrast_row(
            "Horizon",
            "52 firms",
            r"Long $-$ short",
            horizon_contrast,
        ),
        _contrast_row(
            "Universe",
            "Short horizon",
            r"100 $-$ 52 firms",
            universe_contrast,
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{3.2pt}",
                r"\begin{tabularx}{\linewidth}{llXrll}",
                r"\toprule",
                (
                    r"Specification / step & Universe / fixed axis & Horizon / "
                    r"comparison & $T$ & $\hat\rho$ [95\% CI] & "
                    r"$\hat\lambda$ [95\% CI] \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabularx}",
                (
                    r"\caption{Sequential universe-and-horizon bridge for the "
                    r"barycentric $W^\flat$ SAR QMLE. The historical first row uses "
                    r"the recovered 52-firm operator and long return window. The "
                    r"second "
                    r"row holds that operator and universe fixed while shortening the "
                    r"return endpoint to 6 April 2026; its interval is newly estimated "
                    r"with the joint-date stationary bootstrap. The third row expands "
                    r"to the governed current 100-firm operator at the same endpoint; "
                    r"the first and third intervals come from their verified governed "
                    r"results. Complete-case date counts can differ across universes. "
                    r"Panel B reports arithmetic differences only: they are "
                    r"descriptive, "
                    r"noncausal, and carry no contrast-level inference. "
                    r"$\hat\lambda=\hat\rho/(1-\hat\rho)$.}"
                ),
                r"\label{tab:p5-universe-horizon-bridge}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = ["render_paper5_universe_horizon_bridge_table"]

"""Shared display conventions for independently saved Paper 6 evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.io.values import MISSING, Comment, PublicationValue, write_latex

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pipeline.io.values import Record

PRIMARY = "d2560_expanded"
CELLS = (
    (PRIMARY, "Native, expanded"),
    ("d2560_matched", "Native, matched"),
    ("d64_expanded", "Leading 64, expanded"),
    ("d64_matched", "Leading 64, matched"),
)


def number(key: str, precision: int = 6) -> str:
    """Refer to a saved scalar without rounding its stored value."""
    return rf"\ppnum[{precision}]{{ir-{key.replace('_', '-')}}}"


def table(caption: str, label: str, columns: str, header: str, rows: list[str]) -> str:
    """Build the common booktabs wrapper; callers own scientific captions."""
    return "\n".join(
        [
            r"% AUTO-GENERATED from saved Paper 6 results; do not edit.",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def write_numbers(path: Path, values: Mapping[str, float | int | str | None]) -> None:
    """Write one producer's exclusive scalar bindings."""
    records: list[Record] = [Comment("AUTO-GENERATED from saved results; do not edit.")]
    records.extend(
        PublicationValue(
            f"ir-{key.replace('_', '-')}",
            MISSING if value is None else value,
            precision=17,
            kind="g",
        )
        for key, value in values.items()
    )
    write_latex(path, records)

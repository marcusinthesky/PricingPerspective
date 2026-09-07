r"""Renderer-neutral publication-value model with a LaTeX emitter.

One ordered sequence of records is the single upstream representation the LaTeX
``\\ppDeclareValue{kebab-name}{value}`` catalog (see
``src/latex/templates/pp-manuscript.sty``) is built from. Each paper's number
writer builds a ``list[Record]`` once and emits ``src/latex/projects/<slug>/
src/generated/numbers.tex``.

Record kinds (a discriminated union the emitter iterates):

- :class:`PublicationValue` — one public binding. Its ``value`` is one of
  ``int``, ``float``, ``bool``, ``str``, a homogeneous/heterogeneous ``tuple``
  of those, or the :data:`MISSING` sentinel. Floats are never re-rounded here:
  ``precision``/``kind`` (``"f"`` fixed, ``"e"`` scientific, ``"g"`` general)
  are recorded per-binding and applied only at emit time via
  :func:`format_float`, matching the ``f"{v:.6f}"`` shape of the legacy
  hand-rolled writers exactly.
- :class:`Comment` — one comment line, rendered ``% text``.
- :class:`Blank` — one blank separator line.

Ordering: the emitter iterates the given sequence in **input order** and never
re-sorts. Duplicate public keys are rejected before writing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_TEX_SPECIALS = {
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
}


def format_float(value: float, precision: int = 6, kind: str = "f") -> str:
    r"""Format a finite float for a binding.

    ``kind`` selects the Python format spec type: ``"f"`` (fixed, the
    default), ``"e"`` (scientific, used for small p-values/eigenvalues), or
    ``"g"`` (general). Callers guard NaN/MISSING upstream (rendered
    ``\\ppmissing``) and ``PublicationValue`` rejects non-finite non-NaN floats
    at construction, so only finite floats reach here.
    """
    return f"{value:.{precision}{kind}}"


class _Missing(Enum):
    """Sentinel type for a genuinely uncomputed value (upstream ``float.nan``)."""

    MISSING = "MISSING"

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "MISSING"


MISSING = _Missing.MISSING
"""Sentinel for a value that was not computed this pass.

The LaTeX emitter renders it as ``\\ppmissing`` (an em dash).
"""

PublicationScalar = bool | int | float | str | _Missing
PublicationValueType = PublicationScalar | tuple[PublicationScalar, ...]


class PublicationValueError(ValueError):
    """Raised when a publication value cannot be represented losslessly."""


class _EmptyKeyError(PublicationValueError):
    def __init__(self) -> None:
        super().__init__("publication value key must not be empty")


class _ReservedKeyError(PublicationValueError):
    def __init__(self, key: str) -> None:
        super().__init__(f"key {key!r} starts with '_', reserved for private aliases")


class _InvalidKeyCharacterError(PublicationValueError):
    def __init__(self, key: str, character: str) -> None:
        super().__init__(
            f"key {key!r} must be ASCII kebab-case (alphanumerics and '-' only); "
            f"illegal character {character!r}"
        )


class _InvalidStringValueError(PublicationValueError):
    def __init__(self, key: str, detail: str) -> None:
        super().__init__(f"value for {key!r} {detail}")


class _InvalidScalarError(PublicationValueError):
    def __init__(self, key: str, detail: str) -> None:
        super().__init__(f"key {key!r}: {detail}")


class _InvalidBindingError(PublicationValueError):
    def __init__(self, key: str, detail: str) -> None:
        super().__init__(f"key {key!r}: {detail}")


class _SingleLineCommentError(PublicationValueError):
    def __init__(self) -> None:
        super().__init__("comment text must be a single line")


class _DuplicatePublicKeysError(PublicationValueError):
    def __init__(self, duplicates: list[str]) -> None:
        super().__init__(
            "duplicate public keys are not allowed (each key must be emitted "
            f"exactly once): {duplicates}"
        )


def _validate_key(key: str) -> None:
    if not key:
        raise _EmptyKeyError
    if key.startswith("_"):
        raise _ReservedKeyError(key)
    # Accept the existing public grammar verbatim: ASCII alphanumerics and '-'.
    # Mixed case is intentional — stable keys such as 'h1-L-hat' and
    # 'sample-T60-vol' must NOT be silently lowercased.
    for ch in key:
        if not (ch.isascii() and (ch.isalnum() or ch == "-")):
            raise _InvalidKeyCharacterError(key, ch)


def _validate_string(key: str, value: str) -> None:
    if '"' in value:
        raise _InvalidStringValueError(
            key,
            "contains an embedded double quote, which the publication-string "
            "convention does not support: reject rather than under-escape",
        )
    if "\n" in value or "\r" in value:
        raise _InvalidStringValueError(
            key, "contains a newline, which cannot appear inside a single-line binding"
        )


def _validate_scalar(key: str, value: PublicationScalar) -> None:
    # Total over PublicationScalar: every accepted runtime type returns
    # explicitly and any other type is rejected at construction time, rather
    # than silently passing here and only failing (or worse, mis-rendering) at
    # emit time.
    if value is MISSING:
        return
    if isinstance(value, str):
        _validate_string(key, value)
        return
    # bool is a subclass of int; both are representable as-is.
    if isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value) and not math.isnan(value):
            raise _InvalidScalarError(
                key,
                f"non-finite non-NaN float {value!r} is not representable "
                "(use MISSING for uncomputed values, not +/-inf)",
            )
        return
    raise _InvalidScalarError(
        key,
        f"unsupported value type {type(value).__name__!r}; publication scalars "
        "must be bool, int, float, str, or MISSING",
    )


@dataclass(frozen=True)
class Comment:
    """A comment line, rendered ``% text`` in the LaTeX catalog."""

    text: str

    def __post_init__(self) -> None:
        """Reject comments that would break the one-record-per-line format."""
        if "\n" in self.text or "\r" in self.text:
            raise _SingleLineCommentError


@dataclass(frozen=True)
class Blank:
    """A blank separator line."""


@dataclass(frozen=True)
class PublicationValue:
    """One renderer-neutral empirical value bound to a public ``key``.

    ``precision``/``kind`` apply only to ``float`` values (and float tuple
    elements). ``trailing_comment`` is a provenance note retained for callers
    and not rendered in the LaTeX catalog; ``source`` is optional free-text
    provenance retained for callers and not rendered on the line.
    """

    key: str
    value: PublicationValueType
    precision: int = 6
    kind: str = "f"
    trailing_comment: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        """Validate the complete publication binding at construction time."""
        _validate_key(self.key)
        if self.kind not in ("f", "e", "g"):
            raise _InvalidBindingError(
                self.key, f"kind must be 'f', 'e', or 'g', got {self.kind!r}"
            )
        if "\n" in self.trailing_comment or "\r" in self.trailing_comment:
            raise _InvalidBindingError(
                self.key, "trailing_comment must be a single line"
            )
        if isinstance(self.value, tuple):
            for element in self.value:
                if isinstance(element, tuple):
                    raise _InvalidBindingError(
                        self.key, "nested tuples are not supported"
                    )
                if element is MISSING:
                    raise _InvalidBindingError(
                        self.key, "MISSING is not allowed inside a tuple"
                    )
                _validate_scalar(self.key, element)
        elif self.value is not MISSING:
            _validate_scalar(self.key, self.value)


Record = PublicationValue | Comment | Blank


def _escape_tex(s: str) -> str:
    """Escape TeX specials in a literal string value."""
    out: list[str] = []
    for ch in s:
        if ch in _TEX_SPECIALS:
            out.append(_TEX_SPECIALS[ch])
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def _render_scalar_latex(
    key: str, v: PublicationScalar, precision: int, kind: str
) -> str:
    if v is MISSING or (isinstance(v, float) and math.isnan(v)):
        return r"\ppmissing"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return format_float(v, precision=precision, kind=kind)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return _escape_tex(v)
    raise _InvalidScalarError(key, f"unsupported value type {type(v)!r}")


def _render_value_latex(pv: PublicationValue) -> str:
    v = pv.value
    if isinstance(v, tuple):
        return ", ".join(
            _render_scalar_latex(pv.key, e, pv.precision, pv.kind) for e in v
        )
    return _render_scalar_latex(pv.key, v, pv.precision, pv.kind)


def public_values(records: list[Record]) -> list[PublicationValue]:
    """Return the :class:`PublicationValue` records in input order."""
    return [r for r in records if isinstance(r, PublicationValue)]


def _reject_duplicate_keys(records: list[Record]) -> None:
    seen: dict[str, int] = {}
    for pv in public_values(records):
        seen[pv.key] = seen.get(pv.key, 0) + 1
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    if duplicates:
        raise _DuplicatePublicKeysError(duplicates)


def emit_latex(records: list[Record]) -> str:
    r"""Render ``records`` as a LaTeX ``\\ppDeclareValue`` catalog.

    Per-binding ``trailing_comment`` notes are provenance-only and dropped
    here. Missing/NaN render ``\\ppmissing``; string values have TeX specials
    escaped; keys pass through verbatim (hyphens are valid inside the
    ``\\csname``-backed interface). Iterates in input order and rejects
    duplicate public keys.
    """
    _reject_duplicate_keys(records)
    lines: list[str] = []
    for r in records:
        if isinstance(r, Comment):
            lines.append(f"% {r.text}")
        elif isinstance(r, Blank):
            lines.append("")
        else:
            lines.append(f"\\ppDeclareValue{{{r.key}}}{{{_render_value_latex(r)}}}")
    return "\n".join(lines) + "\n"


def write_latex(path: Path, records: list[Record]) -> None:
    """Write publication records as a UTF-8 LaTeX value catalog."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(emit_latex(records), encoding="utf-8")


def write_generated_numbers(path: Path, records: list[Record]) -> None:
    """Emit the LaTeX ``numbers.tex`` catalog from one record list.

    This is the single production entry point that replaces per-writer
    ``path.write_text`` string assembly. ``path`` is the paper's
    ``src/latex/projects/<slug>/src/generated/numbers.tex``.
    """
    write_latex(path, records)


__all__ = [
    "MISSING",
    "Blank",
    "Comment",
    "PublicationScalar",
    "PublicationValue",
    "PublicationValueError",
    "PublicationValueType",
    "Record",
    "emit_latex",
    "format_float",
    "public_values",
    "write_generated_numbers",
    "write_latex",
]

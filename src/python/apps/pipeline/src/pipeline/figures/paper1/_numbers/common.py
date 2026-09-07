"""Shared scalar coercion for the Paper 1 record builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pipeline.io.values import MISSING

if TYPE_CHECKING:
    from pipeline.io.values import PublicationScalar


def _opt_float(value: object) -> PublicationScalar:
    r"""Coerce an optional numeric summary field to a publication scalar.

    ``None`` (e.g. a YAML ``null`` for a singleton sector with no within-sector
    pairs) degrades to the :data:`MISSING` sentinel, rendered ``\\ppmissing``
    in the manuscript, rather than a fabricated ``0.0``.
    """
    if value is None:
        return MISSING
    return float(cast("str | bytes | int | float", value))

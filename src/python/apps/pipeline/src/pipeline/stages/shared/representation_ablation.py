"""Validated cross-paper embedding-representation ablation roster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from pathlib import Path

_PUBLICATION_ROSTER_SIZE = 7
_EXTENDED_ROSTER_SIZE = 10
_CANONICAL_CONTRAST_ROLE = "canonical"
# Sole role of the separate encoder-vintage registry. Kept out of
# `_GOVERNED_CONTRAST_ROLES` so a vintage cell cannot be authored into the
# locked publication roster by mistake.
_VINTAGE_CONTRAST_ROLE = "encoder_vintage"
# Closed vocabulary of publication contrast axes. The broad `ablation.grid` is
# the exploratory roster; a promoted cell must name which axis it isolates, so
# an added cell cannot silently widen Papers 1, 3, and 5 without a policy edit.
_GOVERNED_CONTRAST_ROLES = frozenset(
    {
        _CANONICAL_CONTRAST_ROLE,
        "within_family_capacity_and_native_width",
        "matryoshka_truncation",
        "fixed_width_capacity",
        "external_architecture_training_provider",
    }
)


def _raise_type(message: str) -> None:
    raise TypeError(message)


def _raise_value(message: str) -> None:
    raise ValueError(message)


@dataclass(frozen=True, slots=True)
class RepresentationAblationSpec:
    """One governed encoder/dimension cell shared by publication ablations."""

    cell_id: str
    provider_key: str
    provider_id: str
    representation_key: str
    representation_id: str
    label: str
    effective_dimension: int
    native_dimension: int
    display_order: int
    contrast_role: str
    canonical: bool


@dataclass(frozen=True, slots=True)
class RepresentationContrastSpec:
    """One predeclared candidate-minus-reference publication contrast."""

    contrast_id: str
    candidate_cell_id: str
    reference_cell_id: str
    family: str
    display_order: int


def _required(mapping: dict[str, object], key: str, cell_id: str) -> object:
    if key not in mapping:
        message = f"representation ablation cell {cell_id!r} is missing {key!r}"
        raise ValueError(message)
    return mapping[key]


def _parse_cell(cell_id: str, raw_cell: object) -> RepresentationAblationSpec:
    if not isinstance(raw_cell, dict):
        message = "representation ablation cells require mapping values"
        raise TypeError(message)
    cell = cast("dict[str, object]", raw_cell)
    string_fields = {
        key: _required(cell, key, cell_id)
        for key in (
            "provider_key",
            "provider_id",
            "representation_key",
            "representation_id",
            "label",
            "contrast_role",
        )
    }
    if any(not isinstance(value, str) or not value for value in string_fields.values()):
        message = (
            f"representation ablation cell {cell_id!r} has an invalid string field"
        )
        raise TypeError(message)
    effective_dimension = _required(cell, "effective_dimension", cell_id)
    native_dimension = _required(cell, "native_dimension", cell_id)
    display_order = _required(cell, "display_order", cell_id)
    canonical = _required(cell, "canonical", cell_id)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (effective_dimension, native_dimension, display_order)
    ):
        message = (
            f"representation ablation cell {cell_id!r} has invalid dimensions/order"
        )
        raise TypeError(message)
    if not isinstance(canonical, bool):
        message = f"representation ablation cell {cell_id!r} canonical must be boolean"
        raise TypeError(message)
    return RepresentationAblationSpec(
        cell_id=cell_id,
        provider_key=cast("str", string_fields["provider_key"]),
        provider_id=cast("str", string_fields["provider_id"]),
        representation_key=cast("str", string_fields["representation_key"]),
        representation_id=cast("str", string_fields["representation_id"]),
        label=cast("str", string_fields["label"]),
        effective_dimension=cast("int", effective_dimension),
        native_dimension=cast("int", native_dimension),
        display_order=cast("int", display_order),
        contrast_role=cast("str", string_fields["contrast_role"]),
        canonical=canonical,
    )


def _validate_contrast_roles(ordered: tuple[RepresentationAblationSpec, ...]) -> None:
    roles = {spec.contrast_role for spec in ordered}
    ungoverned = roles - _GOVERNED_CONTRAST_ROLES
    if ungoverned:
        message = (
            "representation ablation roster declares ungoverned contrast roles: "
            f"{sorted(ungoverned)}"
        )
        raise ValueError(message)
    missing = _GOVERNED_CONTRAST_ROLES - roles
    if missing:
        message = (
            "representation ablation roster must cover every governed contrast "
            f"axis; missing: {sorted(missing)}"
        )
        raise ValueError(message)
    if any(
        (spec.contrast_role == _CANONICAL_CONTRAST_ROLE) != spec.canonical
        for spec in ordered
    ):
        message = (
            "representation ablation canonical flag must agree with the canonical "
            "contrast role"
        )
        raise ValueError(message)


def load_representation_ablation_specs(
    params_file: Path,
) -> tuple[RepresentationAblationSpec, ...]:
    """Load, validate, and order the focused publication representation roster."""
    document = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "params file must contain a top-level mapping"
        raise TypeError(message)
    raw_roster = document.get("representation_ablation_specs")
    if not isinstance(raw_roster, dict) or not raw_roster:
        message = "representation_ablation_specs must be a non-empty mapping"
        raise TypeError(message)

    specs: list[RepresentationAblationSpec] = []
    for cell_id, raw_cell in raw_roster.items():
        if not isinstance(cell_id, str):
            message = "representation ablation cells require string keys"
            raise TypeError(message)
        specs.append(_parse_cell(cell_id, raw_cell))

    ordered = tuple(sorted(specs, key=lambda spec: spec.display_order))
    if len(ordered) != _PUBLICATION_ROSTER_SIZE:
        message = "publication representation roster must contain exactly seven cells"
        raise ValueError(message)
    if [spec.display_order for spec in ordered] != list(range(1, len(ordered) + 1)):
        message = "representation ablation display_order must be consecutive from one"
        raise ValueError(message)
    if len({spec.cell_id for spec in ordered}) != len(ordered):
        message = "representation ablation cell IDs must be unique"
        raise ValueError(message)
    if len({spec.representation_id for spec in ordered}) != len(ordered):
        message = "representation ablation representation IDs must be unique"
        raise ValueError(message)
    if sum(spec.canonical for spec in ordered) != 1:
        message = (
            "representation ablation roster must declare exactly one canonical cell"
        )
        raise ValueError(message)
    _validate_contrast_roles(ordered)
    return ordered


def load_encoder_vintage_specs(
    params_file: Path,
) -> tuple[RepresentationAblationSpec, ...]:
    """Load the encoder-vintage roster, which is separate by design.

    `representation_ablation_specs` is the locked seven-cell base contract.
    Vintage arms live in their own registry because Paper 1 treats them as a
    paired negative control, while Papers 3 and 5 append them as ordinary
    representation-sensitivity rows.

    An empty or absent registry is valid and yields no cells.
    """
    document = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "params file must contain a top-level mapping"
        raise TypeError(message)
    raw_roster = document.get("encoder_vintage_ablation_specs")
    if raw_roster is None:
        return ()
    if not isinstance(raw_roster, dict):
        message = "encoder_vintage_ablation_specs must be a mapping"
        raise TypeError(message)

    specs = _parse_encoder_vintage_cells(raw_roster)
    ordered = tuple(sorted(specs, key=lambda spec: spec.display_order))
    if [spec.display_order for spec in ordered] != list(range(1, len(ordered) + 1)):
        message = "encoder vintage display_order must be consecutive from one"
        raise ValueError(message)
    if len({spec.cell_id for spec in ordered}) != len(ordered):
        message = "encoder vintage cell IDs must be unique"
        raise ValueError(message)
    if len({spec.representation_id for spec in ordered}) != len(ordered):
        message = "encoder vintage representation IDs must be unique"
        raise ValueError(message)
    if any(spec.canonical for spec in ordered):
        message = "encoder vintage roster cannot declare a canonical cell"
        raise ValueError(message)
    if any(spec.contrast_role != _VINTAGE_CONTRAST_ROLE for spec in ordered):
        message = (
            f"encoder vintage cells must declare contrast_role "
            f"{_VINTAGE_CONTRAST_ROLE!r}"
        )
        raise ValueError(message)
    return ordered


def load_extended_representation_ablation_specs(
    params_file: Path,
) -> tuple[RepresentationAblationSpec, ...]:
    """Return the seven base cells followed by the three encoder vintages."""
    base = load_representation_ablation_specs(params_file)
    vintages = load_encoder_vintage_specs(params_file)
    extended = (*base, *vintages)
    if len({spec.cell_id for spec in extended}) != len(extended):
        message = "extended representation ladder cell IDs must be unique"
        raise ValueError(message)
    if len({spec.representation_id for spec in extended}) != len(extended):
        message = "extended representation ladder representation IDs must be unique"
        raise ValueError(message)
    if len(extended) != _EXTENDED_ROSTER_SIZE:
        message = "extended representation ladder must contain exactly ten cells"
        raise ValueError(message)
    return extended


def _parse_contrast(
    contrast_id: str, raw_contrast: object
) -> RepresentationContrastSpec:
    if not isinstance(raw_contrast, dict):
        _raise_type("representation contrast cells require mapping values")
    contrast = cast("dict[str, object]", raw_contrast)
    required = ("candidate_cell_id", "reference_cell_id", "family", "display_order")
    values = {key: _required(contrast, key, contrast_id) for key in required}
    if any(
        not isinstance(values[key], str) or not values[key]
        for key in ("candidate_cell_id", "reference_cell_id", "family")
    ):
        message = f"representation contrast {contrast_id!r} has invalid text"
        _raise_type(message)
    order = values["display_order"]
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        message = f"representation contrast {contrast_id!r} has invalid order"
        _raise_type(message)
    return RepresentationContrastSpec(
        contrast_id=contrast_id,
        candidate_cell_id=cast("str", values["candidate_cell_id"]),
        reference_cell_id=cast("str", values["reference_cell_id"]),
        family=cast("str", values["family"]),
        display_order=cast("int", order),
    )


_EXPECTED_CONTRASTS = (
    ("qwen4b_full_minus_qwen8b_full", "qwen4b_full", "qwen8b_full", "base"),
    (
        "qwen4b_1024_minus_qwen8b_1024",
        "qwen4b_1024",
        "qwen8b_1024",
        "base",
    ),
    (
        "qwen8b_1024_minus_qwen8b_full",
        "qwen8b_1024",
        "qwen8b_full",
        "base",
    ),
    ("qwen8b_256_minus_qwen8b_1024", "qwen8b_256", "qwen8b_1024", "base"),
    ("qwen8b_64_minus_qwen8b_256", "qwen8b_64", "qwen8b_256", "base"),
    ("bge_large_minus_qwen8b_1024", "bge_large_full", "qwen8b_1024", "base"),
    ("bge_large_minus_qwen4b_1024", "bge_large_full", "qwen4b_1024", "base"),
    ("ettax_v1_minus_ettax_v0", "ettax_v1", "ettax_v0", "vintage"),
    ("ettax_v3_minus_ettax_v0", "ettax_v3", "ettax_v0", "vintage"),
    ("ettax_v3_minus_ettax_v1", "ettax_v3", "ettax_v1", "vintage"),
)


def load_representation_contrast_specs(  # noqa: C901
    params_file: Path,
) -> tuple[RepresentationContrastSpec, ...]:
    """Load the explicit, ordered candidate-minus-reference contrast registry."""
    document = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        _raise_type("params file must contain a top-level mapping")
    raw_registry = document.get("representation_contrast_specs")
    if not isinstance(raw_registry, dict) or not raw_registry:
        _raise_type("representation_contrast_specs must be a non-empty mapping")
    parsed = tuple(
        sorted(
            (_parse_contrast(str(key), value) for key, value in raw_registry.items()),
            key=lambda item: item.display_order,
        )
    )
    if len(parsed) != len(_EXPECTED_CONTRASTS):
        _raise_value("representation contrast registry must contain ten cells")
    if [item.display_order for item in parsed] != list(range(1, 11)):
        _raise_value("representation contrast display_order must be consecutive")
    observed = tuple(
        (item.contrast_id, item.candidate_cell_id, item.reference_cell_id, item.family)
        for item in parsed
    )
    if observed != _EXPECTED_CONTRASTS:
        _raise_value(
            "representation contrast registry does not match the governed pair order"
        )
    if len({item.contrast_id for item in parsed}) != len(parsed):
        _raise_value("representation contrast IDs must be unique")
    roster = {
        spec.cell_id: spec
        for spec in load_extended_representation_ablation_specs(params_file)
    }
    if any(
        item.candidate_cell_id not in roster
        or item.reference_cell_id not in roster
        or item.candidate_cell_id == item.reference_cell_id
        for item in parsed
    ):
        _raise_value("representation contrasts must reference distinct roster cells")
    base_ids = {
        spec.cell_id for spec in load_representation_ablation_specs(params_file)
    }
    vintage_ids = {spec.cell_id for spec in load_encoder_vintage_specs(params_file)}
    for item in parsed:
        expected = base_ids if item.family == "base" else vintage_ids
        if item.family not in {"base", "vintage"}:
            message = f"unknown representation contrast family {item.family!r}"
            _raise_value(message)
        if {item.candidate_cell_id, item.reference_cell_id} - expected:
            message = f"{item.family} contrast crosses roster families"
            _raise_value(message)
    return parsed


def _parse_encoder_vintage_cells(
    raw_roster: dict[object, object],
) -> list[RepresentationAblationSpec]:
    """Parse vintage mappings before applying roster-level invariants."""
    specs: list[RepresentationAblationSpec] = []
    for cell_id, raw_cell in raw_roster.items():
        if not isinstance(cell_id, str):
            message = "encoder vintage cells require string keys"
            raise TypeError(message)
        specs.append(_parse_cell(cell_id, raw_cell))
    return specs


__all__ = [
    "RepresentationAblationSpec",
    "RepresentationContrastSpec",
    "load_encoder_vintage_specs",
    "load_extended_representation_ablation_specs",
    "load_representation_ablation_specs",
    "load_representation_contrast_specs",
]

"""Contracts for the shared publication representation roster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pipeline.stages.papers.paper1.representation_sensitivity import (
    _candidate_rows,
    _paired_contrasts,
)
from pipeline.stages.shared.representation_ablation import (
    load_encoder_vintage_specs,
    load_extended_representation_ablation_specs,
    load_representation_ablation_specs,
    load_representation_contrast_specs,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
_PAIRED_DRAW_COUNT = 6
_EXTENDED_CONTRAST_COUNT = 35
_ANNUAL_YEARS = tuple(range(2017, 2025))
_NATIVE_WIDTH = 768
_NONFINITE_ARM_INDEX = 2
_EQUIVALENCE_BOUND = 0.012


def _params_with(tmp_path: Path, *substitutions: tuple[str, str]) -> Path:
    """Copy the governed params file with exact roster substitutions applied."""
    text = (REPO_ROOT / "params.yaml").read_text(encoding="utf-8")
    for old, new in substitutions:
        assert old in text
        text = text.replace(old, new)
    params = tmp_path / "params.yaml"
    params.write_text(text, encoding="utf-8")
    return params


def test_shared_roster_has_the_governed_seven_contrasts() -> None:
    """Load the ordered scale, Matryoshka, and external-provider contrasts."""
    specs = load_representation_ablation_specs(REPO_ROOT / "params.yaml")
    assert [spec.cell_id for spec in specs] == [
        "qwen8b_full",
        "qwen4b_full",
        "qwen8b_1024",
        "qwen4b_1024",
        "bge_large_full",
        "qwen8b_256",
        "qwen8b_64",
    ]
    assert [spec.effective_dimension for spec in specs] == [
        4096,
        2560,
        1024,
        1024,
        1024,
        256,
        64,
    ]
    assert sum(spec.canonical for spec in specs) == 1


def test_shared_roster_covers_every_governed_contrast_axis() -> None:
    """Each promoted cell names the axis it isolates, and only one is canonical."""
    specs = load_representation_ablation_specs(REPO_ROOT / "params.yaml")
    assert {spec.contrast_role for spec in specs} == {
        "canonical",
        "within_family_capacity_and_native_width",
        "matryoshka_truncation",
        "fixed_width_capacity",
        "external_architecture_training_provider",
    }
    assert [spec.cell_id for spec in specs if spec.canonical] == ["qwen8b_full"]
    assert [spec.cell_id for spec in specs if spec.contrast_role == "canonical"] == [
        "qwen8b_full"
    ]


def test_shared_roster_rejects_an_ungoverned_contrast_role(tmp_path: Path) -> None:
    """A cell cannot enter the publication roster under an unlisted axis."""
    params = _params_with(
        tmp_path,
        ("contrast_role: matryoshka_truncation", "contrast_role: nested_truncation"),
    )
    with pytest.raises(ValueError, match="ungoverned contrast roles"):
        load_representation_ablation_specs(params)


def test_shared_roster_requires_every_governed_contrast_axis(tmp_path: Path) -> None:
    """Dropping the external-provider contrast fails the roster contract."""
    params = _params_with(
        tmp_path,
        (
            "contrast_role: external_architecture_training_provider",
            "contrast_role: matryoshka_truncation",
        ),
    )
    with pytest.raises(ValueError, match="every governed contrast"):
        load_representation_ablation_specs(params)


def test_shared_roster_requires_canonical_flag_and_role_to_agree(
    tmp_path: Path,
) -> None:
    """Moving the canonical flag off the canonical-role cell is rejected."""
    external = "contrast_role: external_architecture_training_provider"
    params = _params_with(
        tmp_path,
        (
            "contrast_role: canonical\n    canonical: true",
            "contrast_role: canonical\n    canonical: false",
        ),
        (f"{external}\n    canonical: false", f"{external}\n    canonical: true"),
    )
    with pytest.raises(ValueError, match="canonical flag must agree"):
        load_representation_ablation_specs(params)


def test_paper1_encoder_panel_resolves_all_shared_cells() -> None:
    """Paper 1's broad diagnostic keeps exactly the shared encoder subpanel."""
    specs = load_representation_ablation_specs(REPO_ROOT / "params.yaml")
    pairs = pd.DataFrame(
        {
            "model": [spec.provider_id for spec in specs],
            "m": [128] * len(specs),
            "embedding_dimension": [spec.effective_dimension for spec in specs],
        }
    )
    rows = _candidate_rows(pairs, specs)
    encoder = [row for row in rows if row[1] == "encoder_dimension"]
    assert [row[0] for row in encoder] == [spec.cell_id for spec in specs]


def test_vintage_roster_is_separate_and_ordered() -> None:
    """Keep the eight annual sensitivity arms out of the publication roster."""
    publication = load_representation_ablation_specs(REPO_ROOT / "params.yaml")
    vintages = load_encoder_vintage_specs(REPO_ROOT / "params.yaml")

    assert [spec.cell_id for spec in vintages] == [
        f"chronobert_{year}" for year in _ANNUAL_YEARS
    ]
    assert [spec.provider_id for spec in vintages] == [
        f"chronobert-{year}" for year in _ANNUAL_YEARS
    ]
    assert [spec.representation_id for spec in vintages] == [
        f"chronobert-{year}-unit" for year in _ANNUAL_YEARS
    ]
    assert all(
        spec.native_dimension == spec.effective_dimension == _NATIVE_WIDTH
        for spec in vintages
    )
    assert all(spec.contrast_role == "encoder_vintage" for spec in vintages)
    assert not {spec.cell_id for spec in publication} & {
        spec.cell_id for spec in vintages
    }


def test_extended_ladder_appends_vintages_after_the_locked_base() -> None:
    """Expose fifteen Paper 3/5 rows without widening Paper 1's base."""
    extended = load_extended_representation_ablation_specs(REPO_ROOT / "params.yaml")

    assert [spec.cell_id for spec in extended] == [
        "qwen8b_full",
        "qwen4b_full",
        "qwen8b_1024",
        "qwen4b_1024",
        "bge_large_full",
        "qwen8b_256",
        "qwen8b_64",
        *(f"chronobert_{year}" for year in _ANNUAL_YEARS),
    ]
    assert sum(spec.canonical for spec in extended) == 1


def test_contrast_registry_has_explicit_candidate_minus_reference_direction() -> None:
    """The shared registry fixes all seven base and 28 annual vintage directions."""
    contrasts = load_representation_contrast_specs(REPO_ROOT / "params.yaml")
    assert len(contrasts) == _EXTENDED_CONTRAST_COUNT
    assert [
        (item.candidate_cell_id, item.reference_cell_id) for item in contrasts[:7]
    ] == [
        ("qwen4b_full", "qwen8b_full"),
        ("qwen4b_1024", "qwen8b_1024"),
        ("qwen8b_1024", "qwen8b_full"),
        ("qwen8b_256", "qwen8b_1024"),
        ("qwen8b_64", "qwen8b_256"),
        ("bge_large_full", "qwen8b_1024"),
        ("bge_large_full", "qwen4b_1024"),
    ]
    assert [
        (item.candidate_cell_id, item.reference_cell_id) for item in contrasts[7:]
    ] == [
        (f"chronobert_{candidate}", f"chronobert_{reference}")
        for candidate in range(2018, 2025)
        for reference in range(2017, candidate)
    ]
    assert [item.display_order for item in contrasts] == list(range(1, 36))
    assert all(item.family == "base" for item in contrasts[:7])
    assert all(item.family == "vintage" for item in contrasts[7:])


def test_paper1_vintage_panel_resolves_without_widening_encoder_panel() -> None:
    """Append annual vintage cells under their own diagnostic group."""
    publication = load_representation_ablation_specs(REPO_ROOT / "params.yaml")
    vintages = load_encoder_vintage_specs(REPO_ROOT / "params.yaml")
    all_specs = (*publication, *vintages)
    pairs = pd.DataFrame(
        {
            "model": [spec.provider_id for spec in all_specs],
            "m": [128] * len(all_specs),
            "embedding_dimension": [spec.effective_dimension for spec in all_specs],
        }
    )

    rows = _candidate_rows(pairs, publication, vintages)
    encoder = [row for row in rows if row[1] == "encoder_dimension"]
    vintage = [row for row in rows if row[1] == "encoder_vintage"]

    assert [row[0] for row in encoder] == [spec.cell_id for spec in publication]
    assert [row[0] for row in vintage] == [spec.cell_id for spec in vintages]


def test_paired_vintage_contrasts_align_draw_ids_and_apply_equivalence_bound() -> None:
    """Pair successful common draw IDs across all years, including the endpoint."""
    vintages = load_encoder_vintage_specs(REPO_ROOT / "params.yaml")
    contrast_specs = load_representation_contrast_specs(REPO_ROOT / "params.yaml")
    offsets = (0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.018, 0.020)
    cells: list[dict[str, object]] = [
        {
            "cell_id": index,
            "representation_cell": spec.cell_id,
            "group": "encoder_vintage",
            "status": "computed",
            "effect_one_sd": 0.050 + offset,
        }
        for index, (spec, offset) in enumerate(
            zip(vintages, offsets, strict=True), start=11
        )
    ]
    draw_frames = []
    for index, (cell, offset) in enumerate(zip(cells, offsets, strict=True)):
        draw_ids = np.arange(_PAIRED_DRAW_COUNT, dtype=np.int32)
        effects = 0.04 + draw_ids * 0.01 + offset + index * draw_ids * 0.0001
        # Failed fits carry an extreme value to detect accidental admission.
        effects[index % 5] = 100.0 + index
        if index == _NONFINITE_ARM_INDEX:
            effects[5] = np.nan
        frame = pd.DataFrame(
            {
                "cell_id": cell["cell_id"],
                "draw_id": draw_ids,
                "standardized_effect": effects,
                "solve_ok": draw_ids != index % 5,
            }
        )
        draw_frames.append(frame.sample(frac=1.0, random_state=100 + index))
    # One computable base pair keeps its Holm family separate from all 28 vintages.
    for cell_id, name, offset in (
        (1, "qwen8b_full", 0.0),
        (2, "qwen4b_full", 0.03),
    ):
        cells.append(
            {
                "cell_id": cell_id,
                "representation_cell": name,
                "status": "computed",
                "effect_one_sd": 0.05 + offset,
            }
        )
        draw_frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell_id,
                    "draw_id": np.arange(_PAIRED_DRAW_COUNT),
                    "standardized_effect": np.linspace(0.04, 0.09, _PAIRED_DRAW_COUNT)
                    + offset,
                    "solve_ok": True,
                }
            ).sample(frac=1.0, random_state=cell_id)
        )
    draws = pd.concat(draw_frames, ignore_index=True)
    contrasts = _paired_contrasts(
        cells,
        draws,
        contrast_specs,
        equivalence_bound=0.012,
        confidence_level=0.95,
    )
    vintage_results = contrasts[7:]
    assert [row["contrast_id"] for row in vintage_results] == [
        spec.contrast_id for spec in contrast_specs[7:]
    ]
    by_id = {row["contrast_id"]: row for row in vintage_results}
    for candidate in range(1, len(vintages)):
        for reference in range(candidate):
            row = by_id[
                f"{vintages[candidate].cell_id}_minus_{vintages[reference].cell_id}"
            ]
            valid_ids = set(range(_PAIRED_DRAW_COUNT)) - {
                candidate % 5,
                reference % 5,
            }
            if _NONFINITE_ARM_INDEX in (candidate, reference):
                valid_ids.discard(5)
            candidate_valid_ids = set(
                draws.loc[
                    (draws["cell_id"] == candidate + 11)
                    & draws["solve_ok"]
                    & np.isfinite(draws["standardized_effect"]),
                    "draw_id",
                ]
            )
            reference_valid_ids = set(
                draws.loc[
                    (draws["cell_id"] == reference + 11)
                    & draws["solve_ok"]
                    & np.isfinite(draws["standardized_effect"]),
                    "draw_id",
                ]
            )
            assert candidate_valid_ids & reference_valid_ids == valid_ids
            differences = [
                offsets[candidate]
                - offsets[reference]
                + (candidate - reference) * draw_id * 0.0001
                for draw_id in sorted(valid_ids)
            ]
            expected_interval = np.quantile(differences, [0.025, 0.975])
            assert row["estimate"] == pytest.approx(
                offsets[candidate] - offsets[reference]
            )
            assert [row["ci_low"], row["ci_high"]] == pytest.approx(expected_interval)
            assert row["bootstrap_valid_draws"] == len(valid_ids)
            assert row["equivalent"] is bool(
                expected_interval[0] > -_EQUIVALENCE_BOUND
                and expected_interval[1] < _EQUIVALENCE_BOUND
            )
            assert row["holm_pvalue"] == pytest.approx(1.0)
    assert by_id["chronobert_2024_minus_chronobert_2023"]["equivalent"] is True
    assert by_id["chronobert_2024_minus_chronobert_2017"]["equivalent"] is False
    assert contrasts[0]["holm_pvalue"] == contrasts[0]["pvalue"]
    assert "equivalent" not in contrasts[0]
    assert all(row["status"] == "unavailable" for row in contrasts[1:7])


@pytest.mark.parametrize("mutation", ["missing", "reordered", "renamed", "duplicate"])
def test_extended_ladder_rejects_unapproved_annual_rosters(
    tmp_path: Path, mutation: str
) -> None:
    """The extended contract admits only the exact eight chronological cells."""
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    roster = raw["encoder_vintage_ablation_specs"]
    if mutation == "missing":
        del roster["chronobert_2024"]
        error = "exactly fifteen"
    elif mutation == "reordered":
        roster["chronobert_2017"]["display_order"] = 8
        roster["chronobert_2024"]["display_order"] = 1
        error = "ordered ChronoBERT"
    elif mutation == "renamed":
        roster["chronobert_2016"] = roster.pop("chronobert_2017")
        error = "ordered ChronoBERT"
    else:
        roster["chronobert_2024"]["representation_id"] = roster["chronobert_2017"][
            "representation_id"
        ]
        error = "representation IDs must be unique"
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_extended_representation_ablation_specs(params)


@pytest.mark.parametrize("mutation", ["missing", "order", "direction", "family"])
def test_contrast_registry_rejects_malformed_annual_pairs(
    tmp_path: Path, mutation: str
) -> None:
    """Neither direction, family, completeness nor ordering can drift."""
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    registry = raw["representation_contrast_specs"]
    endpoint = registry["chronobert_2024_minus_chronobert_2017"]
    if mutation == "missing":
        del registry["chronobert_2024_minus_chronobert_2017"]
        error = "thirty-five"
    elif mutation == "order":
        endpoint["display_order"] = 36
        error = "consecutive"
    elif mutation == "direction":
        endpoint["candidate_cell_id"], endpoint["reference_cell_id"] = (
            endpoint["reference_cell_id"],
            endpoint["candidate_cell_id"],
        )
        error = "governed pair order"
    else:
        endpoint["family"] = "base"
        error = "governed pair order"
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_representation_contrast_specs(params)


@pytest.mark.parametrize(
    ("bound", "confidence", "error"),
    [
        (0.0, 0.95, "bound must be positive"),
        (0.012, 0.0, "confidence level"),
        (0.012, 1.0, "confidence level"),
    ],
)
def test_paired_contrasts_reject_invalid_inference_settings(
    bound: float, confidence: float, error: str
) -> None:
    """Do not admit undefined margins or percentile interval levels."""
    with pytest.raises(ValueError, match=error):
        _paired_contrasts(
            [],
            pd.DataFrame(),
            (),
            equivalence_bound=bound,
            confidence_level=confidence,
        )


def test_paired_contrasts_require_strict_containment_and_successful_unique_draws() -> (
    None
):
    """Touching the margin fails equivalence; failed or duplicated draws fail."""
    spec = load_representation_contrast_specs(REPO_ROOT / "params.yaml")[7]
    cells: list[dict[str, object]] = [
        {
            "cell_id": index,
            "representation_cell": cell_id,
            "status": "computed",
            "effect_one_sd": effect,
        }
        for index, cell_id, effect in (
            (1, spec.reference_cell_id, 0.0),
            (2, spec.candidate_cell_id, 0.012),
        )
    ]
    draws = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "draw_id": [0, 1, 1, 0],
            "standardized_effect": [0.0, 0.0, 0.012, 0.012],
            "solve_ok": [True] * 4,
        }
    )
    result = _paired_contrasts(
        cells, draws, (spec,), confidence_level=0.95, equivalence_bound=0.012
    )[0]
    assert result["ci_high"] == pytest.approx(_EQUIVALENCE_BOUND)
    assert result["equivalent"] is False
    failed = draws.assign(solve_ok=False)
    unavailable = _paired_contrasts(
        cells, failed, (spec,), confidence_level=0.95, equivalence_bound=0.012
    )[0]
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "no common finite successful bootstrap draws"
    duplicate = pd.concat([draws, draws.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        _paired_contrasts(
            cells, duplicate, (spec,), confidence_level=0.95, equivalence_bound=0.012
        )

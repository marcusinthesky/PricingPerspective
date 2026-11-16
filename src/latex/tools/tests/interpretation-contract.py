#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Positive and hostile fixtures for tools/interpretation-contract."""

from __future__ import annotations

import copy
import json
import math
import runpy
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent.parent
namespace = runpy.run_path(str(TOOLS_DIR / "interpretation-contract"))
canonical_json_bytes = namespace["canonical_json_bytes"]
semantic_digest = namespace["semantic_digest"]
check_project = namespace["check_project"]

HEX_A = "a" * 64
HEX_B = "b" * 64
CONTRACT_ID = "paper1-neighbour-holdout-v1"


def base_payload() -> dict[str, Any]:
    return {
        "comparator_rule": "compare every selected pair with the prespecified sector control",
        "contract_id": CONTRACT_ID,
        "displayed_values": [
            {
                "binding_key": "neighbour-holdout-spread",
                "formatted_value": "1.235",
                "raw_value": 1.234567,
            }
        ],
        "eligible_universe": ["AAA", "BBB"],
        "evaluation_window": {"end": "2022-12-31", "start": "2021-01-01"},
        "exclusions": [{"asset": "CCC", "reason": "missing held-out returns"}],
        "input_array_sha256s": {"embeddings_2018_2020": HEX_A},
        "manuscript_anchor": "sec:fixture-interpretation",
        "metric_definition": "Euclidean chord distance on unit-normalized embeddings",
        "model_definition": "pinned embedding model and revision",
        "normalization_definition": "row L2 normalization before distance calculation",
        "output_array_sha256s": {"held_out_pair_returns": HEX_B},
        "protocol_id": "paper1-neighbour-outcome-holdout-v1",
        "schema_version": 2,
        "selected_pairs": [
            {"left": "AAA", "pair_id": "AAA-BBB", "rank": 1, "right": "BBB"}
        ],
        "selection_rule": "nearest eligible original-space pair after date filtering",
        "tie_break_rule": "lexicographic ticker order",
        "training_window": {"end": "2020-12-31", "start": "2018-01-01"},
    }


def finalize(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["semantic_sha256"] = semantic_digest(result)
    return result


def write_project(
    root: Path,
    *,
    payload: dict[str, Any] | None = None,
    declaration: str | None = None,
    numbers_value: str = "1.235",
    raw_contract: bytes | None = None,
    second_declaration: str | None = None,
    intervening_section: bool = False,
    blank_before_declaration: bool = False,
    write_numbers: bool = True,
    numbers_key: str = "neighbour-holdout-spread",
) -> Path:
    src = root / "project" / "src"
    (src / "generated").mkdir(parents=True)
    (src / "chapters").mkdir()
    (src / "targets").mkdir()
    if write_numbers:
        (src / "generated" / "numbers.tex").write_text(
            "% AUTO-GENERATED fixture\n"
            f"\\ppDeclareValue{{{numbers_key}}}{{{numbers_value}}}\n",
            encoding="utf-8",
        )
    if payload is not None or raw_contract is not None:
        body = raw_contract
        if body is None:
            assert payload is not None
            body = canonical_json_bytes(payload) + b"\n"
        (src / "generated" / "neighbour_holdout.interpretation.json").write_bytes(body)
    lines = [
        r"\subsection{Interpretation}\label{sec:fixture-interpretation}",
        "Future interpretation.",
    ]
    if intervening_section:
        lines.extend(
            [
                r"\subsection{Unrelated material}\label{sec:unrelated}",
                "Unrelated prose.",
            ]
        )
    if blank_before_declaration:
        lines.append("")
    if declaration is not None:
        lines.append(declaration)
    if second_declaration is not None:
        lines.append(second_declaration)
    (src / "chapters" / "results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (src / "targets" / "arxiv.tex").write_text("% fixture\n", encoding="utf-8")
    return src


def valid_declaration(payload: dict[str, Any]) -> str:
    return f"% pp-interpretation-hash{{{payload['contract_id']}}}{{{payload['semantic_sha256']}}}"


def run_fixture(
    name: str,
    configure: Callable[[Path], Path],
    expected_error: str | None,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        src = configure(Path(td))
        _, errors = check_project(src)
        if expected_error is None:
            assert not errors, f"{name}: unexpected errors: {errors}"
        else:
            assert any(expected_error in error for error in errors), (
                f"{name}: expected {expected_error!r}, got {errors}"
            )
    print(f"PASS {name}")


payload = finalize(base_payload())

run_fixture(
    "zero registrations are valid",
    lambda root: write_project(root, write_numbers=False),
    None,
)
run_fixture(
    "canonical contract and unique matching declaration",
    lambda root: write_project(
        root, payload=payload, declaration=valid_declaration(payload)
    ),
    None,
)
run_fixture(
    "declaration outside registered manuscript anchor rejected",
    lambda root: write_project(
        root,
        payload=payload,
        declaration=valid_declaration(payload),
        intervening_section=True,
    ),
    "crosses a sectioning boundary",
)
run_fixture(
    "declaration separated from interpretation prose rejected",
    lambda root: write_project(
        root,
        payload=payload,
        declaration=valid_declaration(payload),
        blank_before_declaration=True,
    ),
    "must immediately follow interpretation prose",
)
run_fixture(
    "missing declaration rejected",
    lambda root: write_project(root, payload=payload),
    "has no TeX declaration",
)
run_fixture(
    "duplicate declaration rejected",
    lambda root: write_project(
        root,
        payload=payload,
        declaration=valid_declaration(payload),
        second_declaration=valid_declaration(payload),
    ),
    "duplicate TeX declarations",
)
run_fixture(
    "orphan declaration rejected",
    lambda root: write_project(root, declaration=valid_declaration(payload)),
    "orphan interpretation declaration",
)
run_fixture(
    "malformed declaration rejected",
    lambda root: write_project(
        root,
        payload=payload,
        declaration=f"% pp-interpretation-hash{{{CONTRACT_ID}}}{{NOT-A-DIGEST}}",
    ),
    "malformed interpretation declaration",
)
run_fixture(
    "declaration digest mismatch rejected",
    lambda root: write_project(
        root,
        payload=payload,
        declaration=f"% pp-interpretation-hash{{{CONTRACT_ID}}}{{{'0' * 64}}}",
    ),
    "declaration digest mismatch",
)


def semantic_mismatch(root: Path) -> Path:
    broken = copy.deepcopy(payload)
    broken["semantic_sha256"] = "0" * 64
    return write_project(root, payload=broken, declaration=valid_declaration(broken))


run_fixture(
    "semantic digest mismatch rejected", semantic_mismatch, "semantic_sha256 mismatch"
)


def noncanonical(root: Path) -> Path:
    pretty = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return write_project(
        root,
        raw_contract=pretty,
        declaration=valid_declaration(payload),
    )


run_fixture(
    "noncanonical JSON rejected", noncanonical, "contract JSON is not canonical"
)


def malformed_payload(root: Path) -> Path:
    broken = base_payload()
    broken.pop("protocol_id")
    broken = finalize(broken)
    return write_project(root, payload=broken, declaration=valid_declaration(broken))


run_fixture(
    "missing payload field rejected",
    malformed_payload,
    "missing required contract fields",
)


def nan_payload(root: Path) -> Path:
    broken = copy.deepcopy(payload)
    broken["displayed_values"][0]["raw_value"] = float("nan")
    raw = (json.dumps(broken, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return write_project(root, raw_contract=raw, declaration=valid_declaration(payload))


run_fixture(
    "nonfinite JSON rejected", nan_payload, "nonfinite JSON number is forbidden"
)


def missing_binding(root: Path) -> Path:
    broken = base_payload()
    broken["displayed_values"][0]["binding_key"] = "missing-key"
    broken = finalize(broken)
    return write_project(root, payload=broken, declaration=valid_declaration(broken))


run_fixture(
    "missing numeric binding rejected", missing_binding, "is missing from numbers.tex"
)
run_fixture(
    "formatted numeric binding mismatch rejected",
    lambda root: write_project(
        root,
        payload=payload,
        declaration=valid_declaration(payload),
        numbers_value="9.999",
    ),
    "does not match numbers.tex binding",
)


def reviewed_fixture(
    root: Path,
    change: Callable[[dict[str, Any]], None],
    *,
    contract_id: str = "paper1-neighbour-outcome-split-v2",
    corrupt_baseline: bool = False,
) -> Path:
    baseline = base_payload()
    baseline["contract_id"] = contract_id
    baseline["selected_pairs"][0]["w2_distance"] = 1.234567
    baseline["selected_pairs"][0]["p_value"] = 0.05
    baseline["displayed_values"][0]["binding_key"] = "neighbour-holdout-w2"
    baseline["output_array_sha256s"]["pair_metrics"] = HEX_A
    baseline = finalize(baseline)
    current = copy.deepcopy(baseline)
    change(current)
    current = finalize(current)
    src = write_project(
        root,
        payload=current,
        declaration=valid_declaration(baseline),
        numbers_key="neighbour-holdout-w2",
        numbers_value=current["displayed_values"][0]["formatted_value"],
    )
    reviews = src / "interpretation_reviews"
    reviews.mkdir()
    if corrupt_baseline:
        baseline["semantic_sha256"] = "0" * 64
    (reviews / f"{contract_id}.json").write_bytes(
        canonical_json_bytes(baseline) + b"\n"
    )
    return src


def advance_w2(value: dict[str, Any], steps: int = 1) -> None:
    number = value["selected_pairs"][0]["w2_distance"]
    for _ in range(steps):
        number = math.nextafter(number, math.inf)
    value["selected_pairs"][0]["w2_distance"] = number
    value["displayed_values"][0]["raw_value"] = number
    value["output_array_sha256s"]["pair_metrics"] = HEX_B


run_fixture(
    "reviewed one-ULP W2 change preserves acknowledgment",
    lambda root: reviewed_fixture(root, advance_w2),
    None,
)
run_fixture(
    "documented W2 roundoff boundary accepted",
    lambda root: reviewed_fixture(root, lambda value: advance_w2(value, 8)),
    None,
)
run_fixture(
    "beyond W2 roundoff boundary rejected despite same displayed value",
    lambda root: reviewed_fixture(root, lambda value: advance_w2(value, 9)),
    "selected_pairs[0].w2_distance",
)
run_fixture(
    "other protocols retain exact review",
    lambda root: reviewed_fixture(root, advance_w2, contract_id=CONTRACT_ID),
    "interpretation review required",
)
run_fixture(
    "tampered reviewed snapshot rejected",
    lambda root: reviewed_fixture(root, advance_w2, corrupt_baseline=True),
    "semantic_sha256 mismatch",
)


def fixed_baseline(root: Path) -> Path:
    src = reviewed_fixture(root, advance_w2)
    paths = list((src / "interpretation_reviews").glob("*.json"))
    paths += list((src / "generated").glob("*.json"))
    paths += list((src / "chapters").glob("*.tex"))
    before = {path: path.read_bytes() for path in paths}
    _, errors = check_project(src)
    assert not errors
    assert before == {path: path.read_bytes() for path in paths}
    return src


run_fixture(
    "accepted roundoff never rewrites baseline, generated contract or acknowledgment",
    fixed_baseline,
    None,
)

for field, replacement in (
    ("pair_id", "AAA-CCC"),
    ("rank", 2),
    ("p_value", 0.049999999999999996),
):
    run_fixture(
        f"selected-pair decision field {field} remains strict",
        lambda root, field=field, replacement=replacement: reviewed_fixture(
            root,
            lambda value: value["selected_pairs"][0].__setitem__(field, replacement),
        ),
        f"selected_pairs[0].{field}",
    )

for field, replacement in (
    ("eligible_universe", ["AAA", "CCC"]),
    ("metric_definition", "squared Euclidean distance"),
    ("evaluation_window", {"start": "2020-01-01", "end": "2022-12-31"}),
):
    run_fixture(
        f"scientific specification {field} remains strict",
        lambda root, field=field, replacement=replacement: reviewed_fixture(
            root, lambda value: value.__setitem__(field, replacement)
        ),
        field,
    )

run_fixture(
    "displayed number change requires review with matching new number binding",
    lambda root: reviewed_fixture(
        root,
        lambda value: value["displayed_values"][0].__setitem__(
            "formatted_value", "1.236"
        ),
    ),
    "displayed_values[0].formatted_value",
)
for hash_field, array in (
    ("input_array_sha256s", "embeddings_2018_2020"),
    ("output_array_sha256s", "held_out_pair_returns"),
):
    run_fixture(
        f"{hash_field} {array} remains strict",
        lambda root, hash_field=hash_field, array=array: reviewed_fixture(
            root, lambda value: value[hash_field].__setitem__(array, "c" * 64)
        ),
        f"{hash_field}.{array}",
    )

print("all interpretation-contract fixtures passed")

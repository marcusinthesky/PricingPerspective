#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Positive + negative tests for the Lean claim manifest extractor.

Uses the pre-built `src/lean` Lake workspace directly (`lake env lean` on a
single declaration is fast; this never runs a full `just lean::build`).

    uv run src/lean/tools/claim-manifest/tests/test_manifest.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import extract  # noqa: E402
import render  # noqa: E402


def _write_mapping(tmp_dir: Path, toml_body: str) -> Path:
    path = tmp_dir / "mapping.toml"
    path.write_text(toml_body)
    return path


def test_module_of() -> None:
    got = extract.module_of("packages/EnergyStatistics/EnergyStatistics/VStatistic.lean")
    assert got == "EnergyStatistics.VStatistic", got


def test_empty_mapping_is_valid() -> None:
    # Tests the empty-INPUT behavior via a temp file; the production mapping.toml
    # is no longer necessarily empty (claims may be populated after author review).
    with tempfile.TemporaryDirectory() as td:
        mapping = _write_mapping(Path(td), "")
        manifest, ok = extract.build_manifest(mapping, verbose=False)
    assert ok is True
    assert manifest["claims"] == []
    assert manifest["schema_version"] == "1.0"
    assert len(manifest["content_hash"]) == 64
    print("[PASS] empty mapping -> valid empty manifest, hash", manifest["content_hash"])


def test_positive_real_declaration() -> None:
    with tempfile.TemporaryDirectory() as td:
        mapping = _write_mapping(
            Path(td),
            """
[[claims]]
paper = "05_spatial_pricing"
claim_id = "prop:v-statistic-nonneg"
statement_summary = "The two-sample energy V-statistic is nonnegative on spaces of negative type."
declaration = "EnergyStatistics.v_statistic_nonneg"
package = "EnergyStatistics"
file = "packages/EnergyStatistics/EnergyStatistics/VStatistic.lean"
assumptions = ["DistNegativeType alpha"]
disclosure = "Machine-checked in the pinned Lean 4 / mathlib v4.31.0 workspace."
""",
        )
        manifest, ok = extract.build_manifest(mapping, verbose=True)
        assert ok is True, manifest
        claim = manifest["claims"][0]
        assert claim["build_result"] == "checked"
        assert set(claim["axioms"]) <= extract.ALLOWED_AXIOMS
        print("[PASS] positive real declaration, axioms =", claim["axioms"])


def test_negative_missing_declaration() -> None:
    with tempfile.TemporaryDirectory() as td:
        mapping = _write_mapping(
            Path(td),
            """
[[claims]]
paper = "05_spatial_pricing"
claim_id = "prop:does-not-exist"
statement_summary = "n/a"
declaration = "EnergyStatistics.v_statistic_nonneg_typo_does_not_exist"
package = "EnergyStatistics"
file = "packages/EnergyStatistics/EnergyStatistics/VStatistic.lean"
disclosure = "n/a"
""",
        )
        manifest, ok = extract.build_manifest(mapping, verbose=True)
        assert ok is False
        assert manifest["claims"][0]["build_result"] == "failed"
        print("[PASS] missing declaration -> generation fails as designed")


def test_negative_duplicate_claim_ids() -> None:
    with tempfile.TemporaryDirectory() as td:
        mapping = _write_mapping(
            Path(td),
            """
[[claims]]
paper = "01_continuous_bounds"
claim_id = "thm:duplicate"
statement_summary = "first"
declaration = "EnergyStatistics.v_statistic_nonneg"
package = "EnergyStatistics"
file = "packages/EnergyStatistics/EnergyStatistics/VStatistic.lean"
disclosure = "first"

[[claims]]
paper = "01_continuous_bounds"
claim_id = "thm:duplicate"
statement_summary = "second"
declaration = "EnergyStatistics.v_statistic_nonneg"
package = "EnergyStatistics"
file = "packages/EnergyStatistics/EnergyStatistics/VStatistic.lean"
disclosure = "second"
""",
        )
        try:
            extract.build_manifest(mapping)
        except ValueError as exc:
            assert "duplicate claim_id" in str(exc)
        else:
            raise AssertionError("duplicate claim IDs were accepted")
        print("[PASS] duplicate claim IDs -> generation fails before Lean checks")


def test_negative_sorry() -> None:
    result = extract.check_declaration_from_snippet_fixture(
        "theorem sorry_thm : True := by sorry\n", "sorry_thm", label="test_sorry"
    )
    assert result.build_result == "failed"
    assert "sorryAx" in result.axioms
    print("[PASS] sorry declaration -> fails on sorryAx, axioms =", result.axioms)


def test_negative_unexpected_axiom() -> None:
    snippet = (
        "axiom myFakeAxiom : True\n"
        "theorem uses_fake_axiom : True := myFakeAxiom\n"
    )
    result = extract.check_declaration_from_snippet_fixture(
        snippet, "uses_fake_axiom", label="test_unexpected_axiom"
    )
    assert result.build_result == "failed"
    assert "myFakeAxiom" in result.axioms
    print("[PASS] unexpected axiom -> fails, axioms =", result.axioms)


def test_parse_axioms_handles_wrapped_output() -> None:
    # Regression: `#print axioms <decl>` wraps its `[...]` list across lines when
    # the fully-qualified name is long; the parser must still recover the axiom
    # set (it previously returned None -> spurious "could not parse" failure,
    # false-failing every long-named declaration).
    wrapped = (
        "'PricingPerspective.ContinuousAPT.systematic_covariance_lower_bound'"
        " depends on axioms: [propext,\n Classical.choice,\n Quot.sound]\n"
    )
    assert extract.parse_axioms(wrapped) == ["propext", "Classical.choice", "Quot.sound"]
    # single-line form and the no-axioms sentinel must still work
    assert extract.parse_axioms("foo depends on axioms: [propext]") == ["propext"]
    assert extract.parse_axioms("bar does not depend on any axioms") == []
    assert extract.parse_axioms("no axiom line here") is None
    print("[PASS] parse_axioms handles wrapped, single-line, and no-axioms output")


def test_parse_statement_type() -> None:
    # `#check @decl` wraps long types across lines at a width that depends on the
    # declaration name, and the `#print axioms` verdict follows immediately after
    # with no blank line. The parser must slice between the two and normalize the
    # wrapping away, so the stored type moves only when the statement moves.
    stdout = (
        "@Foo.bar : forall {E : Type u_1} (w : Fin n -> Real),\n"
        "  (forall (i : Fin n), 0 <= w i) ->\n"
        "    0 <= 1\n"
        "'Foo.bar' depends on axioms: [propext,\n Classical.choice,\n Quot.sound]\n"
    )
    got = extract.parse_statement_type(stdout, "Foo.bar")
    assert got == (
        "forall {E : Type u_1} (w : Fin n -> Real), "
        "(forall (i : Fin n), 0 <= w i) -> 0 <= 1"
    ), got
    assert "depends on axioms" not in got
    # A different declaration name, or output with no `#check` echo at all, must
    # yield "" rather than a wrong type — classify() demotes that to "failed".
    assert extract.parse_statement_type(stdout, "Foo.other") == ""
    assert extract.parse_statement_type("no check output here", "Foo.bar") == ""
    # Regression: `#check @f` prints a BARE `f : ...` when f has no implicit or
    # instance-implicit binders, because the `@` is then a no-op. Matching only
    # the `@` form silently demoted such a claim to "failed" with an empty type.
    bare = "drift_demo : forall (n : Nat), 0 <= n\n'drift_demo' does not depend on any axioms\n"
    assert extract.parse_statement_type(bare, "drift_demo") == "forall (n : Nat), 0 <= n"
    print("[PASS] parse_statement_type slices and normalizes the #check type")


def test_statement_drift_moves_the_content_hash() -> None:
    # The guard this whole field exists for: a mapped theorem can be weakened in
    # place without touching mapping.toml, and every other recorded field
    # (declaration name, axioms, disclosure prose) stays identical. Only the
    # statement type distinguishes the two, so content_hash must be sensitive to
    # it — otherwise the manifest re-derives clean and the manuscript keeps
    # asserting "machine-checked" for a claim it no longer proves.
    strong = [{"claim_id": "thm:x", "statement_type": "forall n, 0 <= f n"}]
    weakened = [{"claim_id": "thm:x", "statement_type": "forall n, 0 <= 1"}]
    strong_hash = hashlib.sha256(extract.canonical_claims_json(strong)).hexdigest()
    weak_hash = hashlib.sha256(extract.canonical_claims_json(weakened)).hexdigest()
    assert strong_hash != weak_hash
    print("[PASS] a statement edit alone moves content_hash")


def test_real_declaration_records_its_statement() -> None:
    result = extract.check_declaration(
        "EnergyStatistics.v_statistic_nonneg",
        "EnergyStatistics.VStatistic",
        label="test_statement_type",
    )
    assert result.build_result == "checked", result.reason
    assert result.statement_type, "statement_type must be populated for a checked claim"
    assert "depends on axioms" not in result.statement_type
    assert "\n" not in result.statement_type
    print("[PASS] real declaration records its statement:", result.statement_type[:60], "…")


def test_manifest_claim_keys_match_schema() -> None:
    # schema.json is not enforced by any hook, so assert the produced shape
    # against it here; otherwise the contract and the extractor drift apart.
    import json

    schema = json.loads((TOOL_DIR / "schema.json").read_text())
    claim_schema = schema["$defs"]["claim"]
    required = set(claim_schema["required"])
    declared = set(claim_schema["properties"])
    assert claim_schema["additionalProperties"] is False
    assert required == declared, required ^ declared
    manifest = json.loads((TOOL_DIR / "generated" / "manifest.json").read_text())
    for claim in manifest["claims"]:
        assert set(claim) == required, set(claim) ^ required
    print("[PASS] every manifest claim matches the schema's required key set")


def test_render_bindings_share_hash() -> None:
    import json
    import subprocess

    manifest_path = TOOL_DIR / "generated" / "manifest.json"
    subprocess.run(
        [sys.executable, str(TOOL_DIR / "extract.py"), "--out", str(manifest_path)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(TOOL_DIR / "render.py"), "--manifest", str(manifest_path)],
        check=True,
    )
    manifest = json.loads(manifest_path.read_text())
    latex_text = (TOOL_DIR / "generated" / "claim_manifest.tex").read_text()
    blueprint_text = (
        TOOL_DIR.parents[1] / "blueprint" / "src" / "generated" / "claim_bindings.tex"
    ).read_text()
    assert manifest["content_hash"] in latex_text
    assert manifest["content_hash"] in blueprint_text
    print("[PASS] latex + Blueprint bindings share manifest hash", manifest["content_hash"])


def test_blueprint_verified_status_is_generated_only_for_checked_claims() -> None:
    manifest = {
        "content_hash": "0" * 64,
        "claims": [
            {
                "claim_id": "thm:checked",
                "declaration": "Example.checked",
                "build_result": "checked",
            },
            {
                "claim_id": "thm:failed",
                "declaration": "Example.failed",
                "build_result": "failed",
            },
        ],
    }
    text = render.render_blueprint(manifest)
    assert "\\lean{Example.checked}\\leanok" in text
    assert "Example.failed" not in text
    assert "ppClaimBinding:thm:failed\\endcsname{\\notready}" in text
    assert "ppClaimProofBinding:thm:failed\\endcsname{\\notready}" in text
    # A checked claim fills both LeanBlueprint slots (statement and proof); a
    # failed one fills neither.
    assert "ppClaimProofBinding:thm:checked\\endcsname{\\leanok}" in text
    assert text.count("\\leanok") == 2
    print("[PASS] only checked claims emit Blueprint declaration/verified bindings")


if __name__ == "__main__":
    tests = [
        test_module_of,
        test_empty_mapping_is_valid,
        test_positive_real_declaration,
        test_negative_missing_declaration,
        test_negative_duplicate_claim_ids,
        test_negative_sorry,
        test_negative_unexpected_axiom,
        test_parse_axioms_handles_wrapped_output,
        test_parse_statement_type,
        test_statement_drift_moves_the_content_hash,
        test_real_declaration_records_its_statement,
        # Must follow test_render_bindings_share_hash: that test regenerates
        # generated/manifest.json, which this one validates against schema.json.
        test_render_bindings_share_hash,
        test_manifest_claim_keys_match_schema,
        test_blueprint_verified_status_is_generated_only_for_checked_claims,
    ]
    for t in tests:
        t()
    print("\nALL TESTS PASSED")

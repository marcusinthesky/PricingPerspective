#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Publication adapter contract tests, plus compiled malicious-contract fixtures."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))
import extract
import render


def expect_error(call, text):
    try:
        call()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected rejection: {text}")


def test_native_join():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "mapping.toml"
        path.write_text(
            '[[claims]]\nclaim_id="claim"\npaper="paper"\ndisclosure="checked"\n'
        )
        extract.validate_native_join([path], {"claim": {}})
        expect_error(
            lambda: extract.validate_native_join([path], {}), "missing=['claim']"
        )
        expect_error(
            lambda: extract.validate_native_join([path], {"claim": {}, "stale": {}}),
            "unclaimed=['stale']",
        )
        expect_error(
            lambda: extract.validate_native_join([path, path], {"claim": {}}),
            "duplicate claim_id",
        )
        path.write_text(path.read_text() + 'declaration="Duplicated.authority"\n')
        expect_error(lambda: extract.read_entries(path), "duplicate native metadata")


def test_native_loader():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = root / ".lake/build/blueprint"
        (base / "library").mkdir(parents=True)
        (base / "module").mkdir()
        (base / "library/Publication.json").write_text(
            '{"modules":["module/Claims.json"]}'
        )
        for library in ("PricingPerspective", "WassersteinGeometry"):
            (base / "library" / f"{library}.json").write_text('{"modules":[]}')
        node = {
            "name": "Example.claim",
            "latexLabel": "claim",
            "statement": {"text": "Summary"},
            "notReady": False,
            "hasLean": True,
            "location": {"module": "Claims"},
        }
        module = base / "module/Claims.json"
        module.write_text(json.dumps([{"type": "node", "data": node}]))
        with (
            patch.object(extract, "LEAN_DIR", root),
            patch.object(extract.subprocess, "run") as run,
        ):
            assert extract.load_native_claims() == {"claim": node}
            run.assert_called_once_with(
                [
                    "lake",
                    "build",
                    *(
                        f"{library}:blueprintJson"
                        for library in extract.NATIVE_LIBRARIES
                    ),
                ],
                cwd=root,
                check=True,
            )
            for field, bad_values in (
                ("notReady", [True, None, "false", 0]),
                ("hasLean", [False, None, "true", 1]),
            ):
                for value in bad_values:
                    invalid = {**node, field: value}
                    module.write_text(json.dumps([{"type": "node", "data": invalid}]))
                    expect_error(
                        lambda: extract.load_native_claims(rebuild=False),
                        "not explicitly ready",
                    )
                invalid = {key: value for key, value in node.items() if key != field}
                module.write_text(json.dumps([{"type": "node", "data": invalid}]))
                expect_error(
                    lambda: extract.load_native_claims(rebuild=False),
                    "not explicitly ready",
                )
            misplaced = {**node, "location": {"module": "PricingPerspective.Original"}}
            module.write_text(json.dumps([{"type": "node", "data": misplaced}]))
            expect_error(
                lambda: extract.load_native_claims(rebuild=False), "must be inline"
            )
            # The assembly may annotate an imported mathlib declaration, but no project declaration.
            (base / "module/Publication").mkdir()
            assembly = base / "module/Publication/Claims.json"
            (base / "library/Publication.json").write_text(
                '{"modules":["module/Publication/Claims.json"]}'
            )
            external = {**node, "location": {"module": "Mathlib.Analysis.Example"}}
            assembly.write_text(json.dumps([{"type": "node", "data": external}]))
            assert extract.load_native_claims(rebuild=False) == {"claim": external}
            assembly.write_text(json.dumps([{"type": "node", "data": misplaced}]))
            expect_error(
                lambda: extract.load_native_claims(rebuild=False), "must be inline"
            )
            (base / "library/Publication.json").write_text(
                '{"modules":["module/Claims.json"]}'
            )
            module.write_text(json.dumps([{"type": "node", "data": external}]))
            expect_error(
                lambda: extract.load_native_claims(rebuild=False), "must be inline"
            )
            module.write_text(json.dumps([{"type": "node", "data": node}] * 2))
            expect_error(
                lambda: extract.load_native_claims(rebuild=False),
                "duplicate native claim",
            )


def test_native_drift_and_failure():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "mapping.toml"
        path.write_text(
            '[[claims]]\npaper="paper"\nclaim_id="claim"\ndisclosure="Checked"\n'
        )
        node = {
            "name": "Example.claim",
            "statement": {"text": "Summary"},
            "location": {"module": "Example"},
        }
        with (
            patch.object(
                extract, "native_source", return_value=("Example", "source.lean")
            ),
            patch.object(
                extract,
                "check_declaration",
                return_value=extract.CheckResult("checked", [], "Nat = Nat"),
            ),
        ):
            first, ok = extract.build_manifest(path, native={"claim": node})
            assert ok
            assert first["claims"][0]["statement_summary"] == "Summary"
            node["statement"]["text"] = "Changed summary"
            second, _ = extract.build_manifest(path, native={"claim": node})
            assert first["content_hash"] != second["content_hash"]
            node["name"] = "Example.changed"
            third, _ = extract.build_manifest(path, native={"claim": node})
            assert second["content_hash"] != third["content_hash"]
        with (
            patch.object(
                extract, "native_source", return_value=("Example", "source.lean")
            ),
            patch.object(
                extract,
                "check_declaration",
                return_value=extract.CheckResult("failed", reason="unsafe"),
            ),
        ):
            manifest, ok = extract.build_manifest(path, native={"claim": node})
            assert not ok
            assert manifest["claims"][0]["disclosure"] == "NOT verified — unsafe"


def test_projection_and_schema():
    for mapping in render.discover_mappings():
        manifest = json.loads(render.manifest_path(mapping).read_text())
        schema = json.loads((TOOL_DIR / "schema.json").read_text())["$defs"]["claim"]
        for claim in manifest["claims"]:
            assert set(claim) == set(schema["required"])
        assert (
            manifest["content_hash"]
            == hashlib.sha256(
                extract.canonical_claims_json(manifest["claims"])
            ).hexdigest()
        )
        for paper in {c["paper"] for c in manifest["claims"]}:
            text = render.render_paper_status(manifest, paper)
            assert manifest["content_hash"] in text
            for claim in manifest["claims"]:
                assert claim["declaration"] not in text
        assert "lean-verified" not in render.render_paper_status(manifest, "withdrawn")


def test_compiled_contracts():
    cases = [
        ("theorem safe : 1 = 1 := rfl", "safe", True),
        ("theorem missingProof : False := by sorry", "missingProof", False),
        ("axiom fake : False\ntheorem custom : False := fake", "custom", False),
        ("theorem vacuous : True := True.intro", "vacuous", False),
        (
            "def Alias : Prop := True\ntheorem aliasTrue : Alias := True.intro",
            "aliasTrue",
            False,
        ),
        ("theorem truePremise (_h : True) : 1 = 1 := rfl", "truePremise", False),
        ("def trueDefinition : Prop := True", "trueDefinition", False),
        ("theorem trueIff : True ↔ (1 = 1) := by simp", "trueIff", False),
    ]
    for source, name, ok in cases:
        result = extract.check_declaration_from_snippet_fixture(
            source, name, label=name
        )
        assert (result.build_result == "checked") == ok, (
            name,
            result.reason,
            result.raw_stdout,
            result.raw_stderr,
        )
        if not ok:
            assert result.reason


if __name__ == "__main__":
    for test in (
        test_native_join,
        test_native_loader,
        test_native_drift_and_failure,
        test_projection_and_schema,
        test_compiled_contracts,
    ):
        test()
        print(f"PASS {test.__name__}")

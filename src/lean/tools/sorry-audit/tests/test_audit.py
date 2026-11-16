"""Semantic Lean fixtures and fail-closed inventory/preflight regression tests."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sorry_audit", TOOL / "audit.py")
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

FIXTURE = """
namespace AuditFixture
def Alias : Prop := True
def AliasForall : Prop := ∀ n : Nat, True
theorem literal : True := True.intro
private theorem privateAlias : Alias := True.intro
def proofDefinition : Alias := True.intro
theorem aliasedForall : AliasForall := by intro; trivial
def predicate (_ : Nat) : Prop := Alias
theorem iffPlaceholder : (1 : Nat) = 1 ↔ Alias := by simp [Alias]
theorem premise (h : Alias) : (1 : Nat) = 1 := rfl
def dataPremise (h : Alias) : Nat := 1
namespace Repeat
theorem Repeat : True := True.intro
end Repeat
/-- `sorry`, `admit`, `axiom`, and `True` in prose are harmless. -/
theorem harmless : (1 : Nat) = 1 := by
  have localWitness : True := True.intro
  exact rfl
public axiom custom : False
theorem usesCustom : False := custom
theorem directAdmission : False := sorryAx False false
macro "hidden_admission" : tactic => `(tactic| sorry)
theorem macroAdmission : False := by hidden_admission
elab "sneaky" : command => do
  let id := mkIdentFrom (← getRef) `sneaky.fake
  elabCommand (← `(theorem $id : True := True.intro))
sneaky
end AuditFixture
"""


class CompiledFixtures(unittest.TestCase):
    """Compile one shared fixture: results test elaboration rather than token patterns."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="sorry-audit-fixture-")
        cls.source = Path(cls.temporary.name) / "Fixture.lean"
        cls.source.write_text(
            (TOOL / "Audit.lean").read_text()
            + FIXTURE
            + "\nset_option maxHeartbeats 0 in\n"
            + "run_cmd liftTermElabM do SorryAudit.run #[(← getEnv).header.mainModule] true\n"
        )
        result = subprocess.run(
            ["lake", "env", "lean", str(cls.source)],
            cwd=audit.LEAN,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.records = [
            r
            for r in audit.parse_records(result.stdout)
            if r["user_name"].startswith("AuditFixture.")
        ]
        cls.modules = {r["module"]: cls.source for r in cls.records}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def selected(self, suffix: str) -> list[dict]:
        return [r for r in self.records if r["user_name"] == "AuditFixture." + suffix]

    def test_true_shapes_and_aliases(self) -> None:
        for name, kind in (
            ("literal", "true-result"),
            ("privateAlias", "true-result"),
            ("proofDefinition", "true-result"),
            ("aliasedForall", "true-result"),
            ("Alias", "true-definition"),
            ("predicate", "true-definition"),
            ("Repeat.Repeat", "true-result"),
            ("iffPlaceholder", "true-iff"),
        ):
            with self.subTest(name=name):
                records = self.selected(name)
                self.assertEqual(len(records), 1)
                rows = audit.inventory_for(records, self.modules)
                self.assertIn(kind + " " + records[0]["declaration"], rows)

    def test_true_alias_premise(self) -> None:
        for name in ("premise", "dataPremise"):
            with self.subTest(name=name):
                self.assertEqual(
                    audit.inventory_for(self.selected(name), self.modules),
                    [f"true-premise AuditFixture.{name}:h"],
                )

    def test_command_elaborator_generated_true_is_named(self) -> None:
        records = self.selected("sneaky.fake")
        self.assertEqual(len(records), 1)
        self.assertEqual(
            audit.inventory_for(records, self.modules),
            ["true-result AuditFixture.sneaky.fake"],
        )

    def test_missing_requested_module_fails_closed(self) -> None:
        snippet = Path(self.temporary.name) / "Missing.lean"
        snippet.write_text(
            (TOOL / "Audit.lean").read_text()
            + "\nrun_cmd liftTermElabM do SorryAudit.run #[`NeverImported]\n"
        )
        result = subprocess.run(
            ["lake", "env", "lean", str(snippet)],
            cwd=audit.LEAN,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requested source module was not imported", result.stdout)

    def test_local_true_and_prose_are_accepted(self) -> None:
        records = [
            r
            for r in self.records
            if r["user_name"].startswith("AuditFixture.harmless")
        ]
        self.assertTrue(records)
        self.assertEqual(audit.inventory_for(records, self.modules), [])

    def test_custom_public_axiom_and_transitive_dependency_fail(self) -> None:
        for name in ("custom", "usesCustom"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "unallowlistable"),
            ):
                audit.inventory_for(self.selected(name), self.modules)

    def test_direct_and_macro_admission_are_named(self) -> None:
        for name in ("directAdmission", "macroAdmission"):
            with self.subTest(name=name):
                self.assertEqual(
                    audit.inventory_for(self.selected(name), self.modules),
                    ["sorry AuditFixture." + name],
                )


class GateContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sorry-audit-contract-")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def source(self, text: str) -> Path:
        path = self.directory / "Fixture.lean"
        path.write_text(text)
        return path

    def test_multiline_and_quoted_debug_options_rejected(self) -> None:
        for code in (
            "set_option\n debug.skipKernelTC true",
            "set_option /- nested /- comment -/ -/ debug.byAsSorry true",
            "set_option «debug».skipKernelTC true",
            "set_option «debug.skipKernelTC» true",
        ):
            with (
                self.subTest(code=code),
                self.assertRaisesRegex(ValueError, "forbidden"),
            ):
                audit.check_debug_options([self.source(code)])

    def test_debug_words_in_comments_and_strings_accepted(self) -> None:
        audit.check_debug_options(
            [
                self.source(
                    "-- set_option debug.skipKernelTC true\n"
                    "/- /- nested -/ set_option debug.byAsSorry true -/\n"
                    'def message := "set_option debug.skipKernelTC true"\n'
                    'def rawMessage := r##"set_option debug.byAsSorry true"##\n'
                    """def quoteCharacter : Char := '"'"""
                    "\n"
                    "set_option maxRecDepth 4096\n"
                )
            ]
        )

    def test_duplicate_malformed_and_missing_allowlist_rejected(self) -> None:
        for text in (
            "sorry A\nsorry A\n",
            "packages/X.lean:1\n",
            "axiom A\n",
            "true-premise A\n",
            "sorry A:b\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.read_allowlist(self.source(text))
        with self.assertRaises(FileNotFoundError):
            audit.read_allowlist(self.directory / "missing.txt")

    def test_new_and_stale_entries_fail(self) -> None:
        for actual, expected in ((["sorry A"], []), ([], ["sorry A"])):
            with (
                self.subTest(actual=actual),
                self.assertRaisesRegex(ValueError, "inventory drift"),
            ):
                audit.compare_inventory(actual, expected)
        audit.compare_inventory(["true-result A"], ["true-result A"])

    def test_incomplete_census_rejected(self) -> None:
        for stdout in ("", "SORRY_AUDIT_COUNT 1\n", "SORRY_AUDIT_JSON {}\n"):
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                audit.parse_records(stdout)

    def test_discovery_covers_unimported_modules(self) -> None:
        package = self.directory / "packages" / "Example"
        (package / "Example").mkdir(parents=True)
        (package / "Example.lean").write_text("")
        (package / "Example" / "Unimported.lean").write_text("")
        (package / "lakefile.lean").write_text("")
        modules, sources = audit.discover(self.directory)
        self.assertEqual(set(modules), {"Example", "Example.Unimported"})
        self.assertEqual(len(sources), 3)

    def test_build_failure_prevents_using_stale_oleans(self) -> None:
        modules = {"Example": self.source("theorem changed : False := by rfl")}
        with (
            patch.object(
                audit.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "lake build"),
            ) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            audit.collect(modules, self.directory)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["lake", "build", "Example"])

    def test_publication_assembly_is_in_the_integrity_preflight(self) -> None:
        package = self.directory / "packages" / "Example"
        package.mkdir(parents=True)
        (package / "Example.lean").write_text("")
        (self.directory / "Publication.lean").write_text("import Publication.Claims")
        assembly = self.directory / "Publication"
        assembly.mkdir()
        dangerous = assembly / "Claims.lean"
        dangerous.write_text("set_option debug.skipKernelTC true")
        modules, sources = audit.discover(self.directory)
        self.assertIn("Publication", modules)
        self.assertIn("Publication.Claims", modules)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.check_debug_options(sources)


if __name__ == "__main__":
    unittest.main()

"""Dependency pin regressions; no Lean build is required."""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
TOOL = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dependency_audit", TOOL / "audit.py")
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

def manifests(lean: Path) -> None:
    """Write the exact coordinated package and toolchain fixture to a temporary root."""
    for relative, required in audit.MANIFESTS:
        path = lean / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        packages = [
            {"name": "mathlib", "rev": audit.MATHLIB, "inputRev": f"v{audit.VERSION}"}
        ]
        for name in sorted(required):
            rev = audit.EXTERNAL[name]
            entry = {"name": name, "rev": rev, "inputRev": rev}
            if name == "proofs":
                entry.update(
                    url=audit.STATADIC_URL, subDir="src/lean"
                )
            packages.append(entry)
        path.write_text(json.dumps({"packages": packages}))
        path.with_name("lean-toolchain").write_text(
            f"leanprover/lean4:v{audit.VERSION}\n"
        )
    source = lean / "packages/EnergyStatistics/lakefile.lean"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        'require proofs from git "' + audit.STATADIC_URL + '" '
        '@ "' + audit.STATADIC_LEAN + '" / "src/lean"\n'
    )

class DependencyPinTests(unittest.TestCase):
    def test_valid_pins(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                manifests(lean)
                self.assertEqual(len(audit.check_pins(lean)), 4)

    def test_pin_mismatch_is_rejected(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                manifests(lean)
                path = lean / audit.MANIFESTS[0][0]
                data = json.loads(path.read_text())
                data["packages"][0]["rev"] = "0" * 40
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "inconsistent mathlib pin"):
                    audit.check_pins(lean)

    def test_statadic_source_is_checked(self) -> None:
            for field, value in (("subDir", "src/python"), ("url", "https://example.com")):
                with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                    lean = Path(directory)
                    manifests(lean)
                    path = lean / "lake-manifest.json"
                    data = json.loads(path.read_text())
                    next(p for p in data["packages"] if p["name"] == "proofs")[field] = (
                        value
                    )
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, "inconsistent proofs source"):
                        audit.check_pins(lean)

    def test_unmanifested_direct_source_pin_is_checked(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                source = lean / "packages/EnergyStatistics/lakefile.lean"
                source.parent.mkdir(parents=True)
                source.write_text(
                    'require proofs from git "' + audit.STATADIC_URL + '" '
                    '@ "' + audit.STATADIC_LEAN + '" / "src/lean"\n'
                )
                audit.check_source_pins(lean)
                source.write_text(source.read_text().replace(audit.STATADIC_LEAN, "0" * 40))
                with self.assertRaisesRegex(ValueError, "inconsistent proofs source pin"):
                    audit.check_source_pins(lean)

    def test_unmanifested_direct_requirement_cannot_be_deleted(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                manifests(lean)
                (lean / "packages/EnergyStatistics/lakefile.lean").write_text(
                    "import Lake\nopen Lake DSL\npackage EnergyStatistics\n"
                )
                with self.assertRaisesRegex(
                    ValueError, "missing direct dependency.*proofs"
                ):
                    audit.check_pins(lean)

    def test_missing_statadic_pin_is_rejected(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                manifests(lean)
                path = lean / "lake-manifest.json"
                data = json.loads(path.read_text())
                data["packages"] = [p for p in data["packages"] if p["name"] != "proofs"]
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "missing proofs dependency"):
                    audit.check_pins(lean)

    def test_statadic_pin_mismatch_is_rejected(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                manifests(lean)
                path = lean / "lake-manifest.json"
                data = json.loads(path.read_text())
                next(p for p in data["packages"] if p["name"] == "proofs")["rev"] = "0" * 40
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "inconsistent proofs pin"):
                    audit.check_pins(lean)

    def test_toolchain_mismatch_is_rejected(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                lean = Path(directory)
                manifests(lean)
                (lean / "lean-toolchain").write_text("leanprover/lean4:v4.31.0\n")
                with self.assertRaisesRegex(ValueError, "inconsistent toolchain pin"):
                    audit.check_pins(lean)

if __name__ == "__main__":
    unittest.main()

"""Check the coordinated public geometry, mathlib and Lean dependency pins."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
LEAN = Path(__file__).resolve().parents[2]
STATADIC_LEAN = "213a32399705af3463a3c1d8835409f284025ca1"
STATADIC_URL = "https://github.com/marcusinthesky/statadic.git"
MATHLIB = "56a005855ea17e0b0958f73ec801ae140b0a1679"
VERSION = "4.35.0"
EXTERNAL = {"proofs": STATADIC_LEAN}
MANIFESTS = (
    ("lake-manifest.json", frozenset({"proofs"})),
    ("packages/PricingPerspective/lake-manifest.json", frozenset({"proofs"})),
    ("packages/WassersteinGeometry/lake-manifest.json", frozenset()),
    ("blueprint/checkdecls/lake-manifest.json", frozenset({"proofs"})),
)
UNMANIFESTED_REQUIREMENTS = {
    "packages/EnergyStatistics/lakefile.lean": frozenset({"proofs"}),
}

def check_source_pins(lean: Path) -> None:
    """Check declared Git revisions even for packages without tracked manifests."""
    declared: dict[str, set[str]] = {}
    requirement = re.compile(
        r"^\s*require\s+(?:«(?P<quoted>[^»]+)»|(?P<name>\w+))\s+from\s+git\s+"
        r'"(?P<url>[^"]+)"\s*@\s*"(?P<revision>[^"]+)"'
        r'(?:\s*/\s*"(?P<subdirectory>[^"]+)")?',
        re.MULTILINE,
    )
    for path in lean.rglob("lakefile.lean"):
        relative = path.relative_to(lean)
        if ".lake" in relative.parts:
            continue
        names = declared.setdefault(relative.as_posix(), set())
        for entry in requirement.finditer(path.read_text()):
            name = entry["quoted"] or entry["name"]
            names.add(name)
            if name not in EXTERNAL:
                continue
            if entry["revision"] != EXTERNAL[name]:
                raise ValueError(f"inconsistent {name} source pin: {path}")
            if name == "proofs" and (
                entry["url"] != STATADIC_URL
                or entry["subdirectory"] != "src/lean"
            ):
                raise ValueError(f"inconsistent proofs source: {path}")
    for relative, required in UNMANIFESTED_REQUIREMENTS.items():
        missing = required - declared.get(relative, set())
        if missing:
            raise ValueError(
                f"missing direct dependency {sorted(missing)}: {lean / relative}"
            )

def check_pins(lean: Path = LEAN) -> dict[str, Any]:
    """Check all tracked manifests, their external pins, and coordinated toolchain files."""
    check_source_pins(lean)
    result = {}
    for relative, required in MANIFESTS:
        path = lean / relative
        packages = json.loads(path.read_text())["packages"]
        mathlib = [p for p in packages if p["name"] == "mathlib"]
        if (
            len(mathlib) != 1
            or mathlib[0].get("rev") != MATHLIB
            or mathlib[0].get("inputRev") != f"v{VERSION}"
        ):
            raise ValueError(f"inconsistent mathlib pin: {path}")
        resolved: dict[str, str | None] = {}
        for name, rev in EXTERNAL.items():
            entries = [p for p in packages if p["name"] == name]
            if not entries:
                if name in required:
                    raise ValueError(f"missing {name} dependency: {path}")
                resolved[name] = None
                continue
            if (
                len(entries) != 1
                or entries[0].get("rev") != rev
                or entries[0].get("inputRev") != rev
            ):
                raise ValueError(f"inconsistent {name} pin: {path}")
            if name == "proofs" and (
                entries[0].get("url") != STATADIC_URL
                or entries[0].get("subDir") != "src/lean"
            ):
                raise ValueError(f"inconsistent proofs source: {path}")
            resolved[name] = rev
        toolchain = path.with_name("lean-toolchain").read_text().strip()
        if toolchain != f"leanprover/lean4:v{VERSION}":
            raise ValueError(
                f"inconsistent toolchain pin: {path.with_name('lean-toolchain')}"
            )
        result[relative] = {"mathlib": MATHLIB, **resolved, "toolchain": toolchain}
    return result

if __name__ == "__main__":
    check_pins()
    print("Coordinated public dependency and toolchain pins pass.")

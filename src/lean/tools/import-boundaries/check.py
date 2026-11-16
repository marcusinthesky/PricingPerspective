"""Check durable dependency direction using Lake's native module import graph."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

LEAN = Path(__file__).resolve().parents[2]
FOUNDATIONS = ("EnergyStatistics", "WassersteinGeometry")
CORES = ("Exposure", "Spatial", "Transmission")


def within(module: str, prefix: str) -> bool:
    """Treat namespace prefixes as components, never textual name prefixes."""
    return module == prefix or module.startswith(prefix + ".")


def violations(edges: list[tuple[str, str]]) -> list[str]:
    """Return forbidden direct edges; module cycles are checked by Lean itself."""
    failures = []
    for source, target in edges:
        if any(within(source, root) for root in FOUNDATIONS) and within(
            target, "PricingPerspective"
        ):
            failures.append(
                f"{source} -> {target}: foundation depends on an application"
            )
        if within(source, "PricingPerspective.Operating") and any(
            within(target, "PricingPerspective." + root)
            for root in (*CORES, "Connections")
        ):
            failures.append(
                f"{source} -> {target}: operating model depends on financial composition"
            )
        if any(within(source, "PricingPerspective." + root) for root in CORES) and any(
            within(target, "PricingPerspective." + root)
            for root in ("Operating", "Connections")
        ):
            failures.append(
                f"{source} -> {target}: core depends on downstream composition"
            )
    return sorted(set(failures))


def main() -> None:
    """Inspect every current first-party module, including non-root imports."""
    modules = sorted(
        ".".join(path.relative_to(LEAN / "packages").with_suffix("").parts[1:])
        for path in (LEAN / "packages").rglob("*.lean")
        if ".lake" not in path.parts and path.name != "lakefile.lean"
    )
    if not modules:
        raise SystemExit("import boundaries: no first-party modules discovered")
    result = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(Path(__file__).with_name("Imports.lean")),
            *modules,
        ],
        cwd=LEAN,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SystemExit(result.stderr or result.stdout)
    records = json.loads(result.stdout)
    if set(records) != set(modules):
        raise SystemExit("import boundaries: incomplete compiled module coverage")
    edges = [
        (source, target) for source, targets in records.items() for target in targets
    ]
    if not edges:
        raise SystemExit("import boundaries: native graph returned no edges")
    failures = violations(edges)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"import boundaries: checked native graph for {len(modules)} source modules")


if __name__ == "__main__":
    main()

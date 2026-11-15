"""Exercise freshness and fail-fast contracts without compiling mathematical libraries."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "quality.sh"


class QualityOrchestrationTests(unittest.TestCase):
    """A checker must see fresh artifacts and must not run after a failed build."""

    def run_gate(self, action: str, failure: str = "") -> tuple[int, list[str]]:
        """Use fake tool processes that enforce a fresh-build marker at audit time."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copyfile(SCRIPT, root / "scripts" / "quality.sh")
            binaries = root / "bin"
            binaries.mkdir()
            lake = binaries / "lake"
            lake.write_text(
                '#!/usr/bin/env bash\nset -eu\nprintf "lake %s\\n" "$*" >> "$CALLS"\n'
                'if [[ "$*" == "build Main Publication" ]]; then\n'
                '  [[ "$FAILURE" != build ]] || exit 17\n'
                '  touch "$FRESH"\nfi\n'
                'if [[ "$*" == "exe batteries/runLinter --no-build $FAILURE" ]]; then\n'
                '  exit 29\nfi\n'
            )
            uv = binaries / "uv"
            uv.write_text(
                '#!/usr/bin/env bash\nset -eu\nprintf "audit %s\\n" "$*" >> "$CALLS"\n'
                '[[ -f "$FRESH" ]] || exit 41\n'
                '[[ "$FAILURE" != boundary ]] || exit 23\n'
            )
            for executable in (lake, uv):
                executable.chmod(0o755)
            for package in (
                "EnergyStatistics",
                "WassersteinGeometry",
                "PricingPerspective",
            ):
                (root / "packages" / package).mkdir(parents=True)
            mk_all = root / ".lake/packages/mathlib/.lake/build/bin/mk_all"
            mk_all.parent.mkdir(parents=True)
            mk_all.write_text("#!/usr/bin/env bash\nexit 0\n")
            mk_all.chmod(0o755)
            calls = root / "calls"
            environment = {
                **os.environ,
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "CALLS": str(calls),
                "FRESH": str(root / "fresh"),
                "FAILURE": failure,
            }
            result = subprocess.run(
                ["bash", str(root / "scripts/quality.sh"), action],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode, calls.read_text().splitlines()

    def test_both_entry_points_build_before_auditing(self) -> None:
        """Lint and the standalone boundary command must rebuild on a cold checkout."""
        for action in ("lint", "boundaries"):
            with self.subTest(action=action):
                status, calls = self.run_gate(action)
                self.assertEqual(status, 0, calls)
                self.assertTrue(
                    any("tools/import-boundaries/check.py" in c for c in calls)
                )

    def test_failed_build_stops_before_audit(self) -> None:
        """A failed rebuild must never turn a stale artifact into a passing audit."""
        for action in ("lint", "boundaries"):
            with self.subTest(action=action):
                status, calls = self.run_gate(action, "build")
                self.assertEqual(status, 17, calls)
                self.assertFalse(any(c.startswith("audit ") for c in calls))
                self.assertFalse(any("batteries/runLinter" in c for c in calls))

    def test_failed_boundary_check_stops_lint(self) -> None:
        """An architecture violation must fail the gate before later linters run."""
        status, calls = self.run_gate("lint", "boundary")
        self.assertEqual(status, 23, calls)
        self.assertFalse(any("batteries/runLinter" in c for c in calls))

    def test_lint_checks_every_root_in_separate_processes(self) -> None:
        """Every package and publication declaration remains covered."""
        status, calls = self.run_gate("lint")
        self.assertEqual(status, 0, calls)
        self.assertEqual(
            [c for c in calls if "batteries/runLinter" in c],
            [
                f"lake exe batteries/runLinter --no-build {root}"
                for root in (
                    "EnergyStatistics", "WassersteinGeometry",
                    "PricingPerspective", "Publication",
                )
            ],
        )

    def test_failed_declaration_lint_stops_before_later_roots_and_style(self) -> None:
        """A package lint failure must fail the complete quality gate."""
        status, calls = self.run_gate("lint", "WassersteinGeometry")
        self.assertEqual(status, 29, calls)
        self.assertFalse(any(c.endswith("PricingPerspective") for c in calls))
        self.assertFalse(any("text-style" in c for c in calls))


if __name__ == "__main__":
    unittest.main()

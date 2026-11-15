#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Regression checks for binary-safe compiler output handling."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent.parent
namespace = runpy.run_path(str(TOOLS_DIR / "bundle-audit"))
AuditError = namespace["AuditError"]
run = namespace["run"]


success = run(
    [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'success \\xf7\\n')",
    ],
    TOOLS_DIR,
)
assert success.returncode == 0
assert success.stdout == b"success \xf7\n"

try:
    run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'failure \\xf7\\n'); "
                "sys.stderr.buffer.write(b'compiler error \\xf7\\n'); "
                "raise SystemExit(7)"
            ),
        ],
        TOOLS_DIR,
    )
except AuditError as exc:
    message = str(exc)
    assert "command failed (7)" in message
    assert "failure \\xf7" in message
    assert "compiler error \\xf7" in message
else:
    raise AssertionError("nonzero subprocess must remain an audit failure")
